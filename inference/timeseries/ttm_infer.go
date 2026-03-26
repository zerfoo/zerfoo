package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TTMInference provides high-level TTM inference with per-channel standard
// scaling normalization and batch support. It wraps a compiled TTMModel and
// handles input normalization, forward pass, and output denormalization.
type TTMInference struct {
	model  *TTMModel
	config *TTMConfig
}

// NewTTMInference creates a TTM inference instance from a GGUF model path.
// The model is loaded, compiled, and ready for inference upon return.
func NewTTMInference(modelPath string, opts ...Option) (*TTMInference, error) {
	m, err := LoadTTM(modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("load TTM model: %w", err)
	}
	return &TTMInference{
		model:  m,
		config: m.cfg,
	}, nil
}

// newTTMInferenceFromConfig creates a TTMInference directly from a config and
// engine, bypassing GGUF loading. This is used for testing without model files.
func newTTMInferenceFromConfig(cfg *TTMConfig) (*TTMInference, error) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, node, err := buildTTMWithNode[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TTM graph: %w", err)
	}

	m := &TTMModel{
		graph:  g,
		cfg:    cfg,
		engine: engine,
		node:   node,
	}
	return &TTMInference{
		model:  m,
		config: cfg,
	}, nil
}

// Forecast produces normalized forecasts from raw time series input.
// Input shape: [context_len][channels] as [][]float64.
// Output shape: [forecast_len][channels] as [][]float64.
//
// The pipeline:
//  1. Per-channel standard scaling: x_norm = (x - mean) / (std + 1e-8)
//  2. Forward pass through TTM graph
//  3. Denormalize forecast: y = y_norm * (std + 1e-8) + mean
func (t *TTMInference) Forecast(input [][]float64) ([][]float64, error) {
	if err := t.validateInput(input); err != nil {
		return nil, err
	}

	// The TTMModel.Forecast already handles normalization inside the graph
	// node (channelStats + normalizeInput + denormalizeOutput). We add an
	// extra layer of standard scaling here at the float64 level for the
	// high-level API to ensure output is in the original input scale.
	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen

	// Compute per-channel mean and std.
	mean, std := channelMeanStd(input, contextLen, numChannels)

	// Normalize input.
	normalized := normalizeChannels(input, mean, std, contextLen, numChannels)

	// Run model forward (the graph also has internal normalization, but we
	// pass pre-normalized data and denormalize the output ourselves).
	result, err := t.model.Forecast(normalized)
	if err != nil {
		return nil, fmt.Errorf("model forecast: %w", err)
	}

	// Denormalize output back to original scale.
	denormalized := denormalizeChannels(result, mean, std, t.config.ForecastLen, numChannels)

	return denormalized, nil
}

// ForecastWithExogenous produces forecasts using future known exogenous variables.
// Input shape: [context_len][channels] as [][]float64.
// Exogenous shape: [forecast_len][num_exog] as [][]float64.
// Output shape: [forecast_len][channels] as [][]float64.
func (t *TTMInference) ForecastWithExogenous(input [][]float64, exogenous [][]float64) ([][]float64, error) {
	if err := t.validateInput(input); err != nil {
		return nil, err
	}
	if err := t.validateExogenous(exogenous); err != nil {
		return nil, err
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen

	// Compute per-channel mean and std.
	mean, std := channelMeanStd(input, contextLen, numChannels)

	// Normalize input.
	normalized := normalizeChannels(input, mean, std, contextLen, numChannels)

	// Run model forward with exogenous data.
	result, err := t.model.ForecastWithExogenous(normalized, exogenous)
	if err != nil {
		return nil, fmt.Errorf("model forecast with exogenous: %w", err)
	}

	// Denormalize output back to original scale.
	denormalized := denormalizeChannels(result, mean, std, t.config.ForecastLen, numChannels)

	return denormalized, nil
}

// ForecastBatch produces forecasts for multiple time series in a batch.
// Input shape: [batch][context_len][channels] as [][][]float64.
// Output shape: [batch][forecast_len][channels] as [][][]float64.
func (t *TTMInference) ForecastBatch(inputs [][][]float64) ([][][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("batch input must not be empty")
	}

	// Validate all inputs in the batch.
	for i, input := range inputs {
		if err := t.validateInput(input); err != nil {
			return nil, fmt.Errorf("batch[%d]: %w", i, err)
		}
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen
	forecastLen := t.config.ForecastLen
	batchSize := len(inputs)

	// Build a batched tensor: [batch, context_len, channels].
	data := make([]float32, batchSize*contextLen*numChannels)

	// Per-sample normalization stats.
	means := make([][]float64, batchSize)
	stds := make([][]float64, batchSize)

	for b := range batchSize {
		mean, std := channelMeanStd(inputs[b], contextLen, numChannels)
		means[b] = mean
		stds[b] = std

		normalized := normalizeChannels(inputs[b], mean, std, contextLen, numChannels)
		for ti := range contextLen {
			for c := range numChannels {
				data[b*contextLen*numChannels+ti*numChannels+c] = float32(normalized[ti][c])
			}
		}
	}

	inputTensor, err := tensor.New[float32]([]int{batchSize, contextLen, numChannels}, data)
	if err != nil {
		return nil, fmt.Errorf("create batch input tensor: %w", err)
	}

	ctx := context.Background()
	outputTensor, err := t.model.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("batch forward pass: %w", err)
	}

	// Convert output [batch, forecast_len, channels] to [][][]float64 and denormalize.
	outData := outputTensor.Data()
	results := make([][][]float64, batchSize)
	for b := range batchSize {
		results[b] = make([][]float64, forecastLen)
		for ti := range forecastLen {
			results[b][ti] = make([]float64, numChannels)
			for c := range numChannels {
				idx := b*forecastLen*numChannels + ti*numChannels + c
				// Denormalize: y = y_norm * (std + eps) + mean
				results[b][ti][c] = float64(outData[idx])*(stds[b][c]+1e-8) + means[b][c]
			}
		}
	}

	return results, nil
}

// Config returns the TTM configuration.
func (t *TTMInference) Config() *TTMConfig {
	return t.config
}

// validateInput checks that the input has the correct shape for the model.
func (t *TTMInference) validateInput(input [][]float64) error {
	if len(input) != t.config.ContextLen {
		return fmt.Errorf("input context_len must be %d, got %d", t.config.ContextLen, len(input))
	}
	if len(input) == 0 {
		return fmt.Errorf("input must not be empty")
	}
	if len(input[0]) != t.config.NumChannels {
		return fmt.Errorf("input channels must be %d, got %d", t.config.NumChannels, len(input[0]))
	}
	return nil
}

// validateExogenous checks that the exogenous input has the correct shape.
func (t *TTMInference) validateExogenous(exogenous [][]float64) error {
	if t.config.NumExogenous <= 0 {
		return fmt.Errorf("model not configured for exogenous variables (NumExogenous=%d)", t.config.NumExogenous)
	}
	if len(exogenous) != t.config.ForecastLen {
		return fmt.Errorf("exogenous forecast_len must be %d, got %d", t.config.ForecastLen, len(exogenous))
	}
	if len(exogenous) == 0 {
		return fmt.Errorf("exogenous must not be empty")
	}
	if len(exogenous[0]) != t.config.NumExogenous {
		return fmt.Errorf("exogenous channels must be %d, got %d", t.config.NumExogenous, len(exogenous[0]))
	}
	return nil
}

// channelMeanStd computes per-channel mean and standard deviation over the
// context window. Returns (mean, std) each of length numChannels.
func channelMeanStd(input [][]float64, contextLen, numChannels int) ([]float64, []float64) {
	mean := make([]float64, numChannels)
	std := make([]float64, numChannels)

	for c := range numChannels {
		var sum float64
		for t := range contextLen {
			sum += input[t][c]
		}
		mean[c] = sum / float64(contextLen)

		var sumSq float64
		for t := range contextLen {
			diff := input[t][c] - mean[c]
			sumSq += diff * diff
		}
		variance := sumSq / float64(contextLen)
		std[c] = float64Sqrt(variance)
	}

	return mean, std
}

// normalizeChannels applies per-channel standard scaling: (x - mean) / (std + eps).
func normalizeChannels(input [][]float64, mean, std []float64, contextLen, numChannels int) [][]float64 {
	const eps = 1e-8
	out := make([][]float64, contextLen)
	for t := range contextLen {
		out[t] = make([]float64, numChannels)
		for c := range numChannels {
			out[t][c] = (input[t][c] - mean[c]) / (std[c] + eps)
		}
	}
	return out
}

// denormalizeChannels reverses per-channel standard scaling: y = y_norm * (std + eps) + mean.
func denormalizeChannels(output [][]float64, mean, std []float64, forecastLen, numChannels int) [][]float64 {
	const eps = 1e-8
	out := make([][]float64, forecastLen)
	for t := range forecastLen {
		out[t] = make([]float64, numChannels)
		for c := range numChannels {
			out[t][c] = output[t][c]*(std[c]+eps) + mean[c]
		}
	}
	return out
}

// TTMVariant describes a known TTM model variant with its context and forecast
// lengths.
type TTMVariant struct {
	Name        string // e.g., "ttm-512-96"
	ContextLen  int
	ForecastLen int
}

// knownTTMVariants lists all known TTM model variants.
var knownTTMVariants = []TTMVariant{
	{Name: "ttm-512-96", ContextLen: 512, ForecastLen: 96},
	{Name: "ttm-1024-96", ContextLen: 1024, ForecastLen: 96},
	{Name: "ttm-1536-96", ContextLen: 1536, ForecastLen: 96},
	{Name: "ttm-512-192", ContextLen: 512, ForecastLen: 192},
	{Name: "ttm-512-336", ContextLen: 512, ForecastLen: 336},
	{Name: "ttm-512-720", ContextLen: 512, ForecastLen: 720},
}

// SelectTTMVariant returns the recommended model variant name for the given
// context and forecast lengths. TTM variants: 512-96, 1024-96, 1536-96,
// 512-192, 512-336, 512-720.
//
// Selection logic:
//  1. If an exact match exists, return it.
//  2. Otherwise, select the variant with the smallest context_len >= requested
//     and smallest forecast_len >= requested.
//  3. If no variant can cover the requested lengths, return the largest variant.
func SelectTTMVariant(contextLen, forecastLen int) string {
	// Check for exact match first.
	for _, v := range knownTTMVariants {
		if v.ContextLen == contextLen && v.ForecastLen == forecastLen {
			return v.Name
		}
	}

	// Find smallest variant that covers both requested dimensions.
	var best *TTMVariant
	for i := range knownTTMVariants {
		v := &knownTTMVariants[i]
		if v.ContextLen >= contextLen && v.ForecastLen >= forecastLen {
			if best == nil || v.ContextLen < best.ContextLen ||
				(v.ContextLen == best.ContextLen && v.ForecastLen < best.ForecastLen) {
				best = v
			}
		}
	}
	if best != nil {
		return best.Name
	}

	// Fallback: return the variant with the largest context, then largest forecast.
	best = &knownTTMVariants[0]
	for i := range knownTTMVariants {
		v := &knownTTMVariants[i]
		if v.ContextLen > best.ContextLen ||
			(v.ContextLen == best.ContextLen && v.ForecastLen > best.ForecastLen) {
			best = v
		}
	}
	return best.Name
}
