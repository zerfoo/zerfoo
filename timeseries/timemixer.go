package timeseries

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// TimeMixerConfig holds configuration for a TimeMixer model.
type TimeMixerConfig struct {
	InputLen    int     // lookback window
	OutputLen   int     // forecast horizon
	NumFeatures int     // number of variates
	NumScales   int     // number of decomposition scales (default 4)
	HiddenSize  int     // hidden dimension (default 256)
	NumLayers   int     // number of mixer layers (default 3)
	Dropout     float64 // dropout rate (unused in CPU path, reserved for GPU)
}

// scaleDecomposition holds the trend and seasonal components at a single scale.
type scaleDecomposition struct {
	trend    [][]float64 // [numFeatures][inputLen]
	seasonal [][]float64 // [numFeatures][inputLen]
}

// TimeMixer implements the TimeMixer time-series forecasting model (ICLR 2024).
// It decomposes input into trend and seasonal components at multiple scales
// using learnable moving average weights.
type TimeMixer struct {
	config TimeMixerConfig

	// maWeights holds learnable moving average kernel weights per scale.
	// maWeights[s] has length 2^(s+1) for scale s (0-indexed).
	maWeights [][]float64
}

// NewTimeMixer creates a new TimeMixer model with the given configuration.
func NewTimeMixer(cfg TimeMixerConfig) *TimeMixer {
	if cfg.NumScales <= 0 {
		cfg.NumScales = 4
	}
	if cfg.HiddenSize <= 0 {
		cfg.HiddenSize = 256
	}
	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 3
	}

	m := &TimeMixer{
		config:    cfg,
		maWeights: make([][]float64, cfg.NumScales),
	}

	// Initialize learnable MA weights per scale with uniform initialization
	// then softmax-normalize so they sum to 1.
	for s := 0; s < cfg.NumScales; s++ {
		kernelSize := 1 << (s + 1) // 2, 4, 8, 16, ...
		m.maWeights[s] = make([]float64, kernelSize)
		// Initialize with small random perturbations around uniform.
		for i := range m.maWeights[s] {
			m.maWeights[s][i] = 1.0/float64(kernelSize) + rand.NormFloat64()*0.01
		}
		normalizeWeights(m.maWeights[s])
	}

	return m
}

// normalizeWeights applies softmax normalization so weights sum to 1 and are non-negative.
func normalizeWeights(w []float64) {
	// Softmax for non-negativity and sum-to-one.
	maxVal := w[0]
	for _, v := range w[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i, v := range w {
		w[i] = math.Exp(v - maxVal)
		sum += w[i]
	}
	for i := range w {
		w[i] /= sum
	}
}

// weightedMovingAverage computes a causal weighted moving average using the
// learnable kernel weights. The kernel is applied as a left-aligned causal
// convolution with edge padding (repeating the boundary value).
// Input: [length], kernel: [kernelSize], output: [length].
func weightedMovingAverage(x, kernel []float64) []float64 {
	n := len(x)
	k := len(kernel)
	out := make([]float64, n)

	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < k; j++ {
			idx := i - j
			if idx < 0 {
				idx = 0 // edge padding
			}
			sum += kernel[j] * x[idx]
		}
		out[i] = sum
	}
	return out
}

// decompose splits input into trend and seasonal components at each scale.
// Input: [numFeatures][inputLen].
// Returns a slice of scaleDecomposition, one per scale.
func (m *TimeMixer) decompose(input [][]float64) []scaleDecomposition {
	scales := make([]scaleDecomposition, m.config.NumScales)

	for s := 0; s < m.config.NumScales; s++ {
		nf := len(input)
		scales[s] = scaleDecomposition{
			trend:    make([][]float64, nf),
			seasonal: make([][]float64, nf),
		}

		kernel := m.maWeights[s]
		for f := 0; f < nf; f++ {
			scales[s].trend[f] = weightedMovingAverage(input[f], kernel)
			scales[s].seasonal[f] = make([]float64, len(input[f]))
			for i := range input[f] {
				scales[s].seasonal[f][i] = input[f][i] - scales[s].trend[f][i]
			}
		}
	}

	return scales
}

// MultiScaleOutput holds the decomposed multi-scale representation from Forward.
type MultiScaleOutput struct {
	// Scales contains the trend and seasonal decomposition at each scale.
	// Scales[s].trend and Scales[s].seasonal are [numFeatures][inputLen].
	Scales []scaleDecomposition
}

// Forward takes input [numFeatures][inputLen] and produces the decomposed
// multi-scale representation.
func (m *TimeMixer) Forward(input [][]float64) (*MultiScaleOutput, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("timemixer: empty input")
	}
	if len(input) != m.config.NumFeatures {
		return nil, fmt.Errorf("timemixer: expected %d features, got %d", m.config.NumFeatures, len(input))
	}
	for f, ch := range input {
		if len(ch) != m.config.InputLen {
			return nil, fmt.Errorf("timemixer: feature %d has length %d, expected %d", f, len(ch), m.config.InputLen)
		}
	}

	scales := m.decompose(input)
	return &MultiScaleOutput{Scales: scales}, nil
}

// MAWeights returns the learnable moving average weights for the given scale.
// This is exported for testing and inspection.
func (m *TimeMixer) MAWeights(scale int) []float64 {
	if scale < 0 || scale >= len(m.maWeights) {
		return nil
	}
	out := make([]float64, len(m.maWeights[scale]))
	copy(out, m.maWeights[scale])
	return out
}
