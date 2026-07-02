package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/timeseries"
	"github.com/zerfoo/ztensor/compute"
)

func TestForecastCommand_Name(t *testing.T) {
	cmd := NewForecastCommand(nil)
	if cmd.Name() != "forecast" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "forecast")
	}
}

func TestForecastCommand_Description(t *testing.T) {
	cmd := NewForecastCommand(nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestForecastCommand_Usage(t *testing.T) {
	cmd := NewForecastCommand(nil)
	if !strings.Contains(cmd.Usage(), "forecast") {
		t.Error("Usage() should contain 'forecast'")
	}
}

func TestForecastCommand_Examples(t *testing.T) {
	cmd := NewForecastCommand(nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestForecastCommand_Interface(t *testing.T) {
	var _ Command = (*ForecastCommand)(nil)
}

func TestForecastCommand_MissingModel(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{"--input", "data.csv", "--horizon", "24"})
	if err == nil {
		t.Fatal("expected error for missing --model")
	}
	if !strings.Contains(err.Error(), "--model is required") {
		t.Errorf("error = %q, want to contain '--model is required'", err.Error())
	}
}

func TestForecastCommand_MissingInput(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{"--model", "tirex.gguf", "--horizon", "24"})
	if err == nil {
		t.Fatal("expected error for missing --input")
	}
	if !strings.Contains(err.Error(), "--input is required") {
		t.Errorf("error = %q, want to contain '--input is required'", err.Error())
	}
}

func TestForecastCommand_MissingHorizon(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{"--model", "tirex.gguf", "--input", "data.csv"})
	if err == nil {
		t.Fatal("expected error for missing --horizon")
	}
	if !strings.Contains(err.Error(), "--horizon must be a positive integer") {
		t.Errorf("error = %q, want to contain '--horizon must be a positive integer'", err.Error())
	}
}

func TestForecastCommand_InvalidHorizon(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{"--model", "m.gguf", "--input", "d.csv", "--horizon", "abc"})
	if err == nil {
		t.Fatal("expected error for invalid --horizon")
	}
	if !strings.Contains(err.Error(), "--horizon") {
		t.Errorf("error = %q, want to contain '--horizon'", err.Error())
	}
}

func TestForecastCommand_InvalidFormat(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{
		"--model", "m.gguf", "--input", "d.csv", "--horizon", "1", "--format", "xml",
	})
	if err == nil {
		t.Fatal("expected error for invalid --format")
	}
	if !strings.Contains(err.Error(), "--format must be csv or json") {
		t.Errorf("error = %q, want to contain '--format must be csv or json'", err.Error())
	}
}

func TestForecastCommand_UnexpectedArg(t *testing.T) {
	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	err := cmd.Run(context.Background(), []string{"--model", "m.gguf", "--input", "d.csv", "--horizon", "1", "extra"})
	if err == nil {
		t.Fatal("expected error for unexpected argument")
	}
	if !strings.Contains(err.Error(), "unexpected argument") {
		t.Errorf("error = %q, want to contain 'unexpected argument'", err.Error())
	}
}

func TestForecastCommand_FlagMissingValue(t *testing.T) {
	tests := []struct {
		name string
		args []string
		err  string
	}{
		{"model missing value", []string{"--model"}, "--model requires a value"},
		{"input missing value", []string{"--input"}, "--input requires a value"},
		{"horizon missing value", []string{"--horizon"}, "--horizon requires a value"},
		{"format missing value", []string{"--format"}, "--format requires a value"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewForecastCommand(&out)
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Error("expected error")
			}
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.err)
			}
		})
	}
}

// writeTestCSV creates a temporary CSV file and returns its path.
func writeTestCSV(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestForecastCommand_CSVOutput(t *testing.T) {
	csvPath := writeTestCSV(t, "temp,humidity\n20.0,50.0\n21.0,51.0\n22.0,52.0\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return timeseries.NewTestForecaster(2, 4)
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "tirex.gguf",
		"--input", csvPath,
		"--horizon", "2",
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	output := out.String()
	// Should contain the CSV header.
	if !strings.Contains(output, "temp") || !strings.Contains(output, "humidity") {
		t.Errorf("CSV output should contain column headers, got %q", output)
	}
	// Should have data rows (header + 2 forecast rows).
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 lines (header + 2 data), got %d: %v", len(lines), lines)
	}
}

func TestForecastCommand_JSONOutput(t *testing.T) {
	csvPath := writeTestCSV(t, "temp,humidity\n20.0,50.0\n21.0,51.0\n22.0,52.0\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return timeseries.NewTestForecaster(2, 4)
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "tirex.gguf",
		"--input", csvPath,
		"--horizon", "2",
		"--format", "json",
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}

	var result []map[string]interface{}
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatalf("JSON unmarshal error: %v, output: %q", err, out.String())
	}
	if len(result) != 2 {
		t.Errorf("expected 2 forecast steps, got %d", len(result))
	}
	// Each entry should have step and values.
	for _, row := range result {
		if _, ok := row["step"]; !ok {
			t.Error("JSON row missing 'step' field")
		}
		if _, ok := row["values"]; !ok {
			t.Error("JSON row missing 'values' field")
		}
	}
}

func TestForecastCommand_EqualsSyntax(t *testing.T) {
	csvPath := writeTestCSV(t, "val\n1.0\n2.0\n3.0\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return timeseries.NewTestForecaster(1, 4)
	}

	err := cmd.Run(context.Background(), []string{
		"--model=tirex.gguf",
		"--input=" + csvPath,
		"--horizon=2",
		"--format=csv",
	})
	if err != nil {
		t.Fatalf("Run with --flag=value syntax failed: %v", err)
	}
}

func TestForecastCommand_LoadError(t *testing.T) {
	csvPath := writeTestCSV(t, "val\n1.0\n2.0\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return nil, errors.New("model not found")
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "missing.gguf",
		"--input", csvPath,
		"--horizon", "1",
	})
	if err == nil {
		t.Fatal("expected error from load")
	}
	if !strings.Contains(err.Error(), "load model") {
		t.Errorf("error = %q, want to contain 'load model'", err.Error())
	}
}

func TestForecastCommand_BadCSV(t *testing.T) {
	csvPath := writeTestCSV(t, "val\nabc\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return timeseries.NewTestForecaster(1, 4)
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "tirex.gguf",
		"--input", csvPath,
		"--horizon", "1",
	})
	if err == nil {
		t.Fatal("expected error for non-numeric CSV data")
	}
	if !strings.Contains(err.Error(), "read input") {
		t.Errorf("error = %q, want to contain 'read input'", err.Error())
	}
}

func TestForecastCommand_EmptyCSV(t *testing.T) {
	csvPath := writeTestCSV(t, "val\n")

	var out bytes.Buffer
	cmd := NewForecastCommand(&out)
	cmd.loadFn = func(_ string, _ compute.Engine[float32]) (*timeseries.FoundationForecaster, error) {
		return timeseries.NewTestForecaster(1, 4)
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "tirex.gguf",
		"--input", csvPath,
		"--horizon", "1",
	})
	if err == nil {
		t.Fatal("expected error for empty CSV")
	}
	if !strings.Contains(err.Error(), "no data rows") {
		t.Errorf("error = %q, want to contain 'no data rows'", err.Error())
	}
}
