package timeseries

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// TSPulseInference provides high-level TSPulse inference with preprocessing.
// It wraps a TSPulseModel and handles input normalization, context length
// adjustment, and output denormalization automatically.
type TSPulseInference struct {
	model  *TSPulseModel
	config *TSPulseConfig
}

// NewTSPulseInference creates a TSPulse inference instance from a GGUF model
// path. The model is loaded and ready for multi-task inference upon return.
func NewTSPulseInference(modelPath string, opts ...Option) (*TSPulseInference, error) {
	m, err := LoadTSPulse(modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("load TSPulse model: %w", err)
	}
	return &TSPulseInference{
		model:  m,
		config: m.config,
	}, nil
}

// newTSPulseInferenceFromConfig creates a TSPulseInference directly from a
// config, bypassing GGUF loading. This is used for testing without model files.
func newTSPulseInferenceFromConfig(cfg *TSPulseConfig) (*TSPulseInference, error) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	model, err := newTSPulseModel(cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TSPulse model: %w", err)
	}
	return &TSPulseInference{
		model:  model,
		config: cfg,
	}, nil
}

// Config returns the TSPulse configuration.
func (t *TSPulseInference) Config() *TSPulseConfig {
	return t.config
}

// AnomalyDetect returns anomaly scores per timestep. Input is preprocessed
// with per-channel normalization. For robust scoring, a sliding window with
// stride = context_len/2 is used and reconstruction errors are averaged across
// overlapping windows.
//
// Input: [][]float64 with shape [timesteps][channels].
// Output: []float64 with shape [timesteps] (anomaly score per timestep).
func (t *TSPulseInference) AnomalyDetect(input [][]float64) ([]float64, error) {
	if err := t.validateInputChannels(input); err != nil {
		return nil, err
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen
	inputLen := len(input)

	// Per-channel normalization.
	mean, std := channelMeanStd(input, inputLen, numChannels)
	normalized := normalizeChannels(input, mean, std, inputLen, numChannels)

	// If input fits exactly, single-window evaluation.
	if inputLen == contextLen {
		return t.model.AnomalyDetect(normalized)
	}

	// If input is shorter than context, pad with zeros to context_len.
	if inputLen < contextLen {
		padded := padInput(normalized, contextLen, numChannels)
		scores, err := t.model.AnomalyDetect(padded)
		if err != nil {
			return nil, err
		}
		// Return only the scores for the original input length.
		return scores[:inputLen], nil
	}

	// Multi-window evaluation: slide a window with stride = context_len/2.
	stride := contextLen / 2
	if stride < 1 {
		stride = 1
	}

	// Accumulate scores and counts for averaging.
	scores := make([]float64, inputLen)
	counts := make([]float64, inputLen)

	for start := 0; start <= inputLen-contextLen; start += stride {
		window := normalized[start : start+contextLen]
		windowScores, err := t.model.AnomalyDetect(window)
		if err != nil {
			return nil, fmt.Errorf("anomaly detect window at %d: %w", start, err)
		}
		for i, s := range windowScores {
			scores[start+i] += s
			counts[start+i]++
		}
	}

	// Handle trailing portion that wasn't covered.
	lastStart := inputLen - contextLen
	if lastStart > 0 {
		// Ensure the last window is included if stride didn't land on it.
		alreadyCovered := false
		for s := 0; s <= inputLen-contextLen; s += stride {
			if s == lastStart {
				alreadyCovered = true
				break
			}
		}
		if !alreadyCovered {
			window := normalized[lastStart : lastStart+contextLen]
			windowScores, err := t.model.AnomalyDetect(window)
			if err != nil {
				return nil, fmt.Errorf("anomaly detect trailing window: %w", err)
			}
			for i, s := range windowScores {
				scores[lastStart+i] += s
				counts[lastStart+i]++
			}
		}
	}

	// Average overlapping scores.
	for i := range scores {
		if counts[i] > 0 {
			scores[i] /= counts[i]
		}
	}

	return scores, nil
}

// Classify returns class probabilities. Input is preprocessed with per-channel
// normalization and optionally resampled to 512 steps via linear interpolation
// if the input length differs from context_len.
//
// Input: [][]float64 with shape [timesteps][channels].
// Output: []float64 with shape [num_classes].
func (t *TSPulseInference) Classify(input [][]float64) ([]float64, error) {
	if err := t.validateInputChannels(input); err != nil {
		return nil, err
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen
	inputLen := len(input)

	// Per-channel normalization.
	mean, std := channelMeanStd(input, inputLen, numChannels)
	normalized := normalizeChannels(input, mean, std, inputLen, numChannels)

	// Resample to context_len if needed.
	if inputLen != contextLen {
		normalized = resampleLinear(normalized, contextLen, numChannels)
	}

	return t.model.Classify(normalized)
}

// Impute fills missing values in the time series. Input is preprocessed with
// per-channel normalization, and the output is denormalized back to the
// original scale.
//
// Input: [][]float64 with shape [context_len][channels].
// mask: []bool with shape [context_len] where true indicates missing values.
// Output: [][]float64 with shape [context_len][channels].
func (t *TSPulseInference) Impute(input [][]float64, mask []bool) ([][]float64, error) {
	if err := t.validateInputChannels(input); err != nil {
		return nil, err
	}
	if len(input) != t.config.ContextLen {
		return nil, fmt.Errorf("input length must be %d for imputation, got %d",
			t.config.ContextLen, len(input))
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen

	// Per-channel normalization (computed only from unmasked positions).
	mean, std := maskedChannelMeanStd(input, mask, contextLen, numChannels)
	normalized := normalizeChannels(input, mean, std, contextLen, numChannels)

	// Run imputation on normalized data.
	result, err := t.model.Impute(normalized, mask)
	if err != nil {
		return nil, err
	}

	// Denormalize output back to original scale.
	denormalized := denormalizeChannels(result, mean, std, contextLen, numChannels)

	return denormalized, nil
}

// Embed returns the semantic embedding vector for similarity search. Input is
// preprocessed with per-channel normalization.
//
// Input: [][]float64 with shape [timesteps][channels].
// Output: []float64 with shape [d_model].
func (t *TSPulseInference) Embed(input [][]float64) ([]float64, error) {
	if err := t.validateInputChannels(input); err != nil {
		return nil, err
	}

	numChannels := t.config.NumChannels
	contextLen := t.config.ContextLen
	inputLen := len(input)

	// Per-channel normalization.
	mean, std := channelMeanStd(input, inputLen, numChannels)
	normalized := normalizeChannels(input, mean, std, inputLen, numChannels)

	// Adjust to context_len if needed.
	if inputLen < contextLen {
		normalized = padInput(normalized, contextLen, numChannels)
	} else if inputLen > contextLen {
		// Truncate to the last context_len steps (most recent data).
		normalized = normalized[inputLen-contextLen:]
	}

	return t.model.Embed(normalized)
}

// validateInputChannels checks that input is non-empty and has the correct
// number of channels.
func (t *TSPulseInference) validateInputChannels(input [][]float64) error {
	if len(input) == 0 {
		return fmt.Errorf("input must not be empty")
	}
	if len(input[0]) != t.config.NumChannels {
		return fmt.Errorf("input channels must be %d, got %d",
			t.config.NumChannels, len(input[0]))
	}
	return nil
}

// maskedChannelMeanStd computes per-channel mean and standard deviation using
// only unmasked positions (mask[t] == false). If all positions are masked,
// falls back to mean=0, std=1.
func maskedChannelMeanStd(input [][]float64, mask []bool, contextLen, numChannels int) ([]float64, []float64) {
	mean := make([]float64, numChannels)
	std := make([]float64, numChannels)

	for c := range numChannels {
		var sum float64
		var count int
		for t := range contextLen {
			if !mask[t] {
				sum += input[t][c]
				count++
			}
		}
		if count == 0 {
			// All masked: use identity transform.
			mean[c] = 0
			std[c] = 1
			continue
		}
		mean[c] = sum / float64(count)

		var sumSq float64
		for t := range contextLen {
			if !mask[t] {
				diff := input[t][c] - mean[c]
				sumSq += diff * diff
			}
		}
		variance := sumSq / float64(count)
		std[c] = math.Sqrt(variance)
	}

	return mean, std
}

// padInput pads input with zeros to the target length.
func padInput(input [][]float64, targetLen, numChannels int) [][]float64 {
	result := make([][]float64, targetLen)
	for t := range targetLen {
		result[t] = make([]float64, numChannels)
		if t < len(input) {
			copy(result[t], input[t])
		}
	}
	return result
}

// resampleLinear resamples a time series to the target length using linear
// interpolation per channel.
func resampleLinear(input [][]float64, targetLen, numChannels int) [][]float64 {
	inputLen := len(input)
	if inputLen == targetLen {
		return input
	}

	result := make([][]float64, targetLen)
	for t := range targetLen {
		result[t] = make([]float64, numChannels)

		// Map target index to source position.
		srcPos := float64(t) * float64(inputLen-1) / float64(targetLen-1)
		if targetLen == 1 {
			srcPos = 0
		}

		srcIdx := int(math.Floor(srcPos))
		frac := srcPos - float64(srcIdx)

		if srcIdx >= inputLen-1 {
			copy(result[t], input[inputLen-1])
		} else {
			for c := range numChannels {
				result[t][c] = input[srcIdx][c]*(1-frac) + input[srcIdx+1][c]*frac
			}
		}
	}

	return result
}
