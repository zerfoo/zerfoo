package cli

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/zerfoo/zerfoo/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// ForecastCommand implements the "forecast" CLI command for time-series
// forecasting using a foundation model.
type ForecastCommand struct {
	out    io.Writer
	loadFn func(path string, engine compute.Engine[float32]) (*timeseries.FoundationForecaster, error)
}

// NewForecastCommand creates a new ForecastCommand using the given output writer.
func NewForecastCommand(out io.Writer) *ForecastCommand {
	return &ForecastCommand{
		out:    out,
		loadFn: timeseries.LoadFoundationModel,
	}
}

// Name implements Command.Name.
func (c *ForecastCommand) Name() string { return "forecast" }

// Description implements Command.Description.
func (c *ForecastCommand) Description() string {
	return "Produce time-series forecasts using a foundation model"
}

// Run implements Command.Run.
func (c *ForecastCommand) Run(ctx context.Context, args []string) error {
	var modelPath, inputPath, format string
	var horizon int

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--model":
			s, err := nextVal("--model")
			if err != nil {
				return err
			}
			modelPath = s
		case "--input":
			s, err := nextVal("--input")
			if err != nil {
				return err
			}
			inputPath = s
		case "--horizon":
			s, err := nextVal("--horizon")
			if err != nil {
				return err
			}
			v, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("--horizon: %w", err)
			}
			horizon = v
		case "--format":
			s, err := nextVal("--format")
			if err != nil {
				return err
			}
			format = s
		default:
			return fmt.Errorf("unexpected argument: %s", args[i])
		}
	}

	if modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if inputPath == "" {
		return fmt.Errorf("--input is required")
	}
	if horizon <= 0 {
		return fmt.Errorf("--horizon must be a positive integer")
	}
	if format == "" {
		format = "csv"
	}
	format = strings.ToLower(format)
	if format != "csv" && format != "json" {
		return fmt.Errorf("--format must be csv or json, got %q", format)
	}

	// Read CSV input: columns = variates, rows = time steps.
	input, headers, err := readTimeSeriesCSV(inputPath)
	if err != nil {
		return fmt.Errorf("read input: %w", err)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	li := startLoading(os.Stderr)
	forecaster, err := c.loadFn(modelPath, engine)
	li.stop()
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	result, err := forecaster.Forecast(ctx, input, horizon)
	if err != nil {
		return fmt.Errorf("forecast: %w", err)
	}

	switch format {
	case "json":
		return writeJSONForecast(c.out, result, headers)
	default:
		return writeCSVForecast(c.out, result, headers)
	}
}

// Usage implements Command.Usage.
func (c *ForecastCommand) Usage() string {
	return `forecast [OPTIONS]

Produce time-series forecasts using a foundation model.

OPTIONS:
  --model <path>       Path to GGUF model file (required)
  --input <path>       Path to CSV input file (required)
  --horizon <int>      Number of future time steps to forecast (required)
  --format <format>    Output format: csv, json (default: csv)`
}

// Examples implements Command.Examples.
func (c *ForecastCommand) Examples() []string {
	return []string{
		"forecast --model tirex.gguf --input data.csv --horizon 24",
		"forecast --model tirex.gguf --input data.csv --horizon 12 --format json",
	}
}

// Static interface assertion.
var _ Command = (*ForecastCommand)(nil)

// readTimeSeriesCSV reads a CSV file where columns are variates and rows are
// time steps. Returns the data as [][]float64 and the column headers.
func readTimeSeriesCSV(path string) ([][]float64, []string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = f.Close() }()

	reader := csv.NewReader(f)
	header, err := reader.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("read CSV header: %w", err)
	}

	if len(header) == 0 {
		return nil, nil, fmt.Errorf("CSV has no columns")
	}

	var data [][]float64
	for {
		record, readErr := reader.Read()
		if readErr != nil {
			break
		}
		row := make([]float64, len(header))
		for i, cell := range record {
			if i >= len(header) {
				break
			}
			v, parseErr := strconv.ParseFloat(strings.TrimSpace(cell), 64)
			if parseErr != nil {
				return nil, nil, fmt.Errorf("row %d, column %q: %w", len(data)+1, header[i], parseErr)
			}
			row[i] = v
		}
		data = append(data, row)
	}

	if len(data) == 0 {
		return nil, nil, fmt.Errorf("CSV has no data rows")
	}

	return data, header, nil
}

// writeCSVForecast writes forecast results as CSV to the given writer.
func writeCSVForecast(w io.Writer, result [][]float64, headers []string) error {
	writer := csv.NewWriter(w)
	defer writer.Flush()

	if err := writer.Write(headers); err != nil {
		return err
	}
	for _, row := range result {
		record := make([]string, len(row))
		for i, v := range row {
			record[i] = strconv.FormatFloat(v, 'f', 6, 64)
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	return nil
}

// writeJSONForecast writes forecast results as JSON to the given writer.
func writeJSONForecast(w io.Writer, result [][]float64, headers []string) error {
	type forecastRow struct {
		Step   int                `json:"step"`
		Values map[string]float64 `json:"values"`
	}

	rows := make([]forecastRow, len(result))
	for i, row := range result {
		values := make(map[string]float64, len(headers))
		for j, h := range headers {
			if j < len(row) {
				values[h] = row[j]
			}
		}
		rows[i] = forecastRow{Step: i + 1, Values: values}
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(rows)
}
