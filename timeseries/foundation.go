package timeseries

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// FoundationForecaster provides zero-shot time-series forecasting using a
// pre-trained foundation model (TiRex). It handles input normalization
// (instance norm), graph execution, and output denormalization.
type FoundationForecaster struct {
	graph   *graph.Graph[float32]
	engine  compute.Engine[float32]
	cfg     *its.TiRexConfig
	numVars int
	horizon int
}

// LoadFoundationModel loads a TiRex foundation model from a GGUF file and
// returns a forecaster ready for zero-shot inference.
func LoadFoundationModel(path string, engine compute.Engine[float32]) (*FoundationForecaster, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := loadTiRexConfigFromMeta(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("load TiRex config: %w", err)
	}

	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load GGUF tensors: %w", err)
	}

	g, err := its.BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TiRex graph: %w", err)
	}

	return &FoundationForecaster{
		graph:   g,
		engine:  engine,
		cfg:     cfg,
		numVars: cfg.NumVars,
		horizon: cfg.Horizon,
	}, nil
}

// newFoundationForecasterFromConfig creates a FoundationForecaster directly
// from a config and engine, bypassing GGUF loading. Used for testing.
func newFoundationForecasterFromConfig(cfg *its.TiRexConfig, engine compute.Engine[float32]) (*FoundationForecaster, error) {
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	g, err := its.BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TiRex graph: %w", err)
	}
	return &FoundationForecaster{
		graph:   g,
		engine:  engine,
		cfg:     cfg,
		numVars: cfg.NumVars,
		horizon: cfg.Horizon,
	}, nil
}

// Forecast produces a zero-shot forecast for a single time series.
//
// Input shape: [seq_len][num_vars] as [][]float64.
// Output shape: [horizon][num_vars] as [][]float64.
//
// The pipeline:
//  1. Per-channel instance normalization: x_norm = (x - mean) / (std + eps)
//  2. Forward pass through TiRex graph
//  3. Denormalize predictions: y = y_norm * (std + eps) + mean
func (f *FoundationForecaster) Forecast(ctx context.Context, input [][]float64, horizon int) ([][]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input must not be empty")
	}
	numVars := len(input[0])
	if numVars != f.numVars {
		return nil, fmt.Errorf("input variables must be %d, got %d", f.numVars, numVars)
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", horizon)
	}

	seqLen := len(input)

	// Per-channel instance normalization.
	mean, std := instanceNorm(input, seqLen, numVars)

	// Build input tensor: [1, seq_len, num_vars].
	data := make([]float32, seqLen*numVars)
	for t := range seqLen {
		for c := range numVars {
			data[t*numVars+c] = float32((input[t][c] - mean[c]) / (std[c] + 1e-8))
		}
	}

	inputTensor, err := tensor.New[float32]([]int{1, seqLen, numVars}, data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputTensor, err := f.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("forward pass: %w", err)
	}

	// The graph outputs [1, model_horizon, num_vars]. Extract and denormalize.
	outData := outputTensor.Data()
	modelHorizon := f.horizon
	effectiveHorizon := min(horizon, modelHorizon)

	result := make([][]float64, effectiveHorizon)
	for t := range effectiveHorizon {
		result[t] = make([]float64, numVars)
		for c := range numVars {
			val := float64(outData[t*numVars+c])
			result[t][c] = val*(std[c]+1e-8) + mean[c]
		}
	}

	return result, nil
}

// BatchForecast produces zero-shot forecasts for multiple time series.
//
// Input shape: [batch][seq_len][num_vars] as [][][]float64.
// Output shape: [batch][horizon][num_vars] as [][][]float64.
func (f *FoundationForecaster) BatchForecast(ctx context.Context, inputs [][][]float64, horizon int) ([][][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("batch input must not be empty")
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", horizon)
	}

	batchSize := len(inputs)
	seqLen := len(inputs[0])
	numVars := f.numVars

	// Validate all samples.
	for i, input := range inputs {
		if len(input) == 0 {
			return nil, fmt.Errorf("batch[%d]: input must not be empty", i)
		}
		if len(input) != seqLen {
			return nil, fmt.Errorf("batch[%d]: sequence length must be %d, got %d", i, seqLen, len(input))
		}
		if len(input[0]) != numVars {
			return nil, fmt.Errorf("batch[%d]: input variables must be %d, got %d", i, numVars, len(input[0]))
		}
	}

	// Per-sample normalization stats.
	means := make([][]float64, batchSize)
	stds := make([][]float64, batchSize)

	// Build batched tensor: [batch, seq_len, num_vars].
	data := make([]float32, batchSize*seqLen*numVars)
	for b := range batchSize {
		mean, std := instanceNorm(inputs[b], seqLen, numVars)
		means[b] = mean
		stds[b] = std

		for t := range seqLen {
			for c := range numVars {
				data[b*seqLen*numVars+t*numVars+c] = float32((inputs[b][t][c] - mean[c]) / (std[c] + 1e-8))
			}
		}
	}

	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen, numVars}, data)
	if err != nil {
		return nil, fmt.Errorf("create batch input tensor: %w", err)
	}

	outputTensor, err := f.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("batch forward pass: %w", err)
	}

	// Convert output [batch, model_horizon, num_vars] and denormalize.
	outData := outputTensor.Data()
	modelHorizon := f.horizon
	effectiveHorizon := min(horizon, modelHorizon)

	results := make([][][]float64, batchSize)
	for b := range batchSize {
		results[b] = make([][]float64, effectiveHorizon)
		for t := range effectiveHorizon {
			results[b][t] = make([]float64, numVars)
			for c := range numVars {
				idx := b*modelHorizon*numVars + t*numVars + c
				val := float64(outData[idx])
				results[b][t][c] = val*(stds[b][c]+1e-8) + means[b][c]
			}
		}
	}

	return results, nil
}

// instanceNorm computes per-channel mean and standard deviation.
func instanceNorm(input [][]float64, seqLen, numVars int) ([]float64, []float64) {
	mean := make([]float64, numVars)
	std := make([]float64, numVars)

	for c := range numVars {
		var sum float64
		for t := range seqLen {
			sum += input[t][c]
		}
		mean[c] = sum / float64(seqLen)

		var sumSq float64
		for t := range seqLen {
			diff := input[t][c] - mean[c]
			sumSq += diff * diff
		}
		std[c] = math.Sqrt(sumSq / float64(seqLen))
	}

	return mean, std
}

// loadTiRexConfigFromMeta extracts a TiRexConfig from GGUF metadata.
func loadTiRexConfigFromMeta(meta map[string]interface{}) (*its.TiRexConfig, error) {
	getInt := func(key string) (int, error) {
		v, ok := meta[key]
		if !ok {
			return 0, fmt.Errorf("missing metadata key %q", key)
		}
		switch n := v.(type) {
		case uint32:
			return int(n), nil
		case int:
			return n, nil
		case int64:
			return int(n), nil
		case uint64:
			return int(n), nil
		default:
			return 0, fmt.Errorf("metadata key %q has unexpected type %T", key, v)
		}
	}

	numLayers, err := getInt("tirex.block_count")
	if err != nil {
		return nil, err
	}
	hiddenDim, err := getInt("tirex.hidden_dim")
	if err != nil {
		return nil, err
	}

	// InputDim defaults to 1 for univariate foundation models.
	inputDim := 1
	if v, ok := meta["tirex.input_dim"]; ok {
		switch n := v.(type) {
		case uint32:
			inputDim = int(n)
		case int:
			inputDim = n
		}
	}

	// Horizon defaults to 1.
	horizon := 1
	if v, ok := meta["tirex.horizon"]; ok {
		switch n := v.(type) {
		case uint32:
			horizon = int(n)
		case int:
			horizon = n
		}
	}

	// NumVars defaults to InputDim.
	numVars := inputDim
	if v, ok := meta["tirex.num_vars"]; ok {
		switch n := v.(type) {
		case uint32:
			numVars = int(n)
		case int:
			numVars = n
		}
	}

	// Block types from string array.
	var blockTypes []string
	if v, ok := meta["tirex.block_types"]; ok {
		switch arr := v.(type) {
		case []string:
			blockTypes = arr
		case []any:
			blockTypes = make([]string, len(arr))
			for i, s := range arr {
				str, ok := s.(string)
				if !ok {
					return nil, fmt.Errorf("tirex.block_types[%d] is not a string", i)
				}
				blockTypes[i] = str
			}
		}
	}

	return &its.TiRexConfig{
		NumLayers:  numLayers,
		InputDim:   inputDim,
		HiddenDim:  hiddenDim,
		Horizon:    horizon,
		NumVars:    numVars,
		BlockTypes: blockTypes,
	}, nil
}
