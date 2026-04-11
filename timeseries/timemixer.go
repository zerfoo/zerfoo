package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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

// mixingMLP holds the weight and bias parameters for a two-layer MLP that
// mixes components across scales. Input dimension is numScales, hidden
// dimension is hiddenSize, output dimension is numScales.
type mixingMLP struct {
	w1 [][]float64 // [hiddenSize][numScales]
	b1 []float64   // [hiddenSize]
	w2 [][]float64 // [numScales][hiddenSize]
	b2 []float64   // [numScales]
}

// forward runs the two-layer MLP with ReLU activation.
// Input: [numScales], output: [numScales].
func (mlp *mixingMLP) forward(x []float64) []float64 {
	hidden := make([]float64, len(mlp.b1))
	for i := range hidden {
		sum := mlp.b1[i]
		for j, v := range x {
			sum += mlp.w1[i][j] * v
		}
		// ReLU
		if sum < 0 {
			sum = 0
		}
		hidden[i] = sum
	}
	out := make([]float64, len(mlp.b2))
	for i := range out {
		sum := mlp.b2[i]
		for j, v := range hidden {
			sum += mlp.w2[i][j] * v
		}
		out[i] = sum
	}
	return out
}

// randFloat64 returns a uniform [0,1) sample from r if non-nil, else from the
// global math/rand/v2 generator.
func randFloat64(r *rand.Rand) float64 {
	if r != nil {
		return r.Float64()
	}
	return rand.Float64()
}

// randNormFloat64 returns a standard-normal sample from r if non-nil, else
// from the global math/rand/v2 generator.
func randNormFloat64(r *rand.Rand) float64 {
	if r != nil {
		return r.NormFloat64()
	}
	return rand.NormFloat64()
}

// newMixingMLP creates a mixing MLP with Kaiming uniform initialization.
// If rng is non-nil it is used for weight sampling; otherwise the global
// math/rand/v2 generator is used.
func newMixingMLP(numScales, hiddenSize int, rng *rand.Rand) *mixingMLP {
	// Kaiming uniform: U(-bound, bound) where bound = sqrt(6 / fan_in)
	bound1 := math.Sqrt(6.0 / float64(numScales))
	w1 := make([][]float64, hiddenSize)
	for i := range w1 {
		w1[i] = make([]float64, numScales)
		for j := range w1[i] {
			w1[i][j] = (randFloat64(rng)*2 - 1) * bound1
		}
	}
	b1 := make([]float64, hiddenSize)

	bound2 := math.Sqrt(6.0 / float64(hiddenSize))
	w2 := make([][]float64, numScales)
	for i := range w2 {
		w2[i] = make([]float64, hiddenSize)
		for j := range w2[i] {
			w2[i][j] = (randFloat64(rng)*2 - 1) * bound2
		}
	}
	b2 := make([]float64, numScales)

	return &mixingMLP{w1: w1, b1: b1, w2: w2, b2: b2}
}

// TimeMixer implements the TimeMixer time-series forecasting model (ICLR 2024).
// It decomposes input into trend and seasonal components at multiple scales
// using learnable moving average weights, then produces forecasts via
// scale-specific linear heads combined with learned mixing weights.
type TimeMixer struct {
	config TimeMixerConfig

	// maWeights holds learnable moving average kernel weights per scale.
	// maWeights[s] has length 2^(s+1) for scale s (0-indexed).
	maWeights [][]float64

	// seasonalMLPs holds one mixing MLP per layer for seasonal components.
	seasonalMLPs []*mixingMLP
	// trendMLPs holds one mixing MLP per layer for trend components.
	trendMLPs []*mixingMLP

	// trendHeads[s] is a linear projection [inputLen][outputLen] for scale s trend.
	trendHeads [][][]float64

	// seasonalHeads[s] is a linear projection [inputLen][outputLen] for scale s seasonal.
	seasonalHeads [][][]float64

	// mixWeights holds the raw (pre-softmax) mixing weights per scale.
	// Length: NumScales. Softmax is applied at inference time.
	mixWeights []float64

	engine compute.Engine[float32]     // optional; enables GPU-accelerated forward
	ops    numeric.Arithmetic[float32] // arithmetic ops for engine path

	// Training state for TrainableBackend.
	grads []float64 // gradient accumulator

	// initRNG is an optional caller-provided RNG used only during NewTimeMixer
	// weight initialization. Cleared at the end of NewTimeMixer so it cannot
	// affect later behavior.
	initRNG *rand.Rand
}

// NewTimeMixer creates a new TimeMixer model with the given configuration.
func NewTimeMixer(cfg TimeMixerConfig, opts ...TimeMixerOption) *TimeMixer {
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
		config:        cfg,
		maWeights:     make([][]float64, cfg.NumScales),
		seasonalMLPs:  make([]*mixingMLP, cfg.NumLayers),
		trendMLPs:     make([]*mixingMLP, cfg.NumLayers),
		trendHeads:    make([][][]float64, cfg.NumScales),
		seasonalHeads: make([][][]float64, cfg.NumScales),
		mixWeights:    make([]float64, cfg.NumScales),
	}

	// Apply options FIRST so caller-provided RNG is available for weight init.
	for _, opt := range opts {
		opt(m)
	}
	rng := m.initRNG

	// Initialize learnable MA weights per scale with uniform initialization
	// then softmax-normalize so they sum to 1.
	for s := 0; s < cfg.NumScales; s++ {
		kernelSize := 1 << (s + 1) // 2, 4, 8, 16, ...
		m.maWeights[s] = make([]float64, kernelSize)
		// Initialize with small random perturbations around uniform.
		for i := range m.maWeights[s] {
			m.maWeights[s][i] = 1.0/float64(kernelSize) + randNormFloat64(rng)*0.01
		}
		normalizeWeights(m.maWeights[s])
	}

	// Initialize mixing MLPs for each layer.
	for l := 0; l < cfg.NumLayers; l++ {
		m.seasonalMLPs[l] = newMixingMLP(cfg.NumScales, cfg.HiddenSize, rng)
		m.trendMLPs[l] = newMixingMLP(cfg.NumScales, cfg.HiddenSize, rng)
	}

	// Initialize scale-specific linear heads with Xavier uniform initialization.
	xavierBound := math.Sqrt(6.0 / float64(cfg.InputLen+cfg.OutputLen))
	for s := 0; s < cfg.NumScales; s++ {
		m.trendHeads[s] = make([][]float64, cfg.InputLen)
		m.seasonalHeads[s] = make([][]float64, cfg.InputLen)
		for i := 0; i < cfg.InputLen; i++ {
			m.trendHeads[s][i] = make([]float64, cfg.OutputLen)
			m.seasonalHeads[s][i] = make([]float64, cfg.OutputLen)
			for j := 0; j < cfg.OutputLen; j++ {
				m.trendHeads[s][i][j] = (randFloat64(rng)*2 - 1) * xavierBound
				m.seasonalHeads[s][i][j] = (randFloat64(rng)*2 - 1) * xavierBound
			}
		}
	}

	// Initialize mixing weights to uniform (equal contribution from each scale).
	for s := range m.mixWeights {
		m.mixWeights[s] = 0.0
	}

	// Drop RNG reference so it does not outlive init.
	m.initRNG = nil

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

// pastDecomposableMixing applies bottom-up mixing across scales for both
// seasonal and trend components. For each layer, an MLP learns mixing weights
// across scales, and coarse-scale information flows to finer scales via
// additive residual connections.
//
// The mixing iterates from coarsest to finest scale (bottom-up). At each
// position (feature, time), the MLP takes the vector of values across all
// scales and produces new mixed values. The result at scale s is then added
// as a residual to scale s-1 (the next finer scale).
func (m *TimeMixer) pastDecomposableMixing(scales []scaleDecomposition) []scaleDecomposition {
	numScales := len(scales)
	nf := len(scales[0].trend)
	seqLen := len(scales[0].trend[0])

	for l := 0; l < m.config.NumLayers; l++ {
		seasonalMLP := m.seasonalMLPs[l]
		trendMLP := m.trendMLPs[l]

		// Collect cross-scale vectors, apply MLP, then write back.
		// For seasonal:
		newSeasonal := make([]scaleDecomposition, numScales)
		newTrend := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			newSeasonal[s] = scaleDecomposition{
				trend:    scales[s].trend,
				seasonal: make([][]float64, nf),
			}
			newTrend[s] = scaleDecomposition{
				trend:    make([][]float64, nf),
				seasonal: scales[s].seasonal,
			}
			for f := 0; f < nf; f++ {
				newSeasonal[s].seasonal[f] = make([]float64, seqLen)
				newTrend[s].trend[f] = make([]float64, seqLen)
			}
		}

		scaleVec := make([]float64, numScales)

		// mlpFwd dispatches to engine MLP when available, else CPU.
		mlpFwd := func(mlp *mixingMLP, x []float64) []float64 {
			if m.engine != nil {
				if out := m.engineMLPForward(mlp, x); out != nil {
					return out
				}
			}
			return mlp.forward(x)
		}

		// Mix seasonal across scales.
		for f := 0; f < nf; f++ {
			for t := 0; t < seqLen; t++ {
				for s := 0; s < numScales; s++ {
					scaleVec[s] = scales[s].seasonal[f][t]
				}
				mixed := mlpFwd(seasonalMLP, scaleVec)
				for s := 0; s < numScales; s++ {
					newSeasonal[s].seasonal[f][t] = mixed[s]
				}
			}
		}

		// Mix trend across scales.
		for f := 0; f < nf; f++ {
			for t := 0; t < seqLen; t++ {
				for s := 0; s < numScales; s++ {
					scaleVec[s] = scales[s].trend[f][t]
				}
				mixed := mlpFwd(trendMLP, scaleVec)
				for s := 0; s < numScales; s++ {
					newTrend[s].trend[f][t] = mixed[s]
				}
			}
		}

		// Bottom-up residual: coarse scale (higher index) adds to next finer scale.
		// When engine is available, use engine.Add for the residual connections.
		if m.engine != nil {
			m.engineBottomUpResidual(newSeasonal, newTrend, numScales, nf, seqLen)
		} else {
			for s := numScales - 2; s >= 0; s-- {
				for f := 0; f < nf; f++ {
					for t := 0; t < seqLen; t++ {
						newSeasonal[s].seasonal[f][t] += newSeasonal[s+1].seasonal[f][t]
						newTrend[s].trend[f][t] += newTrend[s+1].trend[f][t]
					}
				}
			}
		}

		// Assemble updated scales for next layer.
		for s := 0; s < numScales; s++ {
			scales[s] = scaleDecomposition{
				trend:    newTrend[s].trend,
				seasonal: newSeasonal[s].seasonal,
			}
		}
	}

	return scales
}

// MultiScaleOutput holds the decomposed and mixed multi-scale representation.
type MultiScaleOutput struct {
	// Scales contains the mixed trend and seasonal components at each scale.
	// Scales[s].trend and Scales[s].seasonal are [numFeatures][inputLen].
	Scales []scaleDecomposition
}

// TimeMixerOutput holds the forecast output and decomposed multi-scale representation.
type TimeMixerOutput struct {
	// Forecast is the final combined forecast: [numFeatures][outputLen].
	Forecast [][]float64
	// Scales contains the mixed decomposition (for inspection/debugging).
	MultiScaleOutput
}

// linearProject applies a linear projection head [inputLen][outputLen] to an
// input vector [inputLen], producing output [outputLen].
func linearProject(input []float64, weight [][]float64, outputLen int) []float64 {
	out := make([]float64, outputLen)
	for i, x := range input {
		for j := 0; j < outputLen; j++ {
			out[j] += x * weight[i][j]
		}
	}
	return out
}

// projectVector applies a linear projection using Engine[T] when available,
// falling back to the CPU path otherwise.
// input: [inputLen], weight: [inputLen][outputLen], output: [outputLen].
func (m *TimeMixer) projectVector(input []float64, weight [][]float64, outputLen int) []float64 {
	if m.engine != nil {
		if out := m.engineProjectVector(input, weight, outputLen); out != nil {
			return out
		}
	}
	return linearProject(input, weight, outputLen)
}

// Forward takes input [numFeatures][inputLen] and produces a forecast
// [numFeatures][outputLen] by decomposing at multiple scales, projecting
// each scale's trend and seasonal components via learned linear heads,
// and combining with softmax-gated mixing weights.
func (m *TimeMixer) Forward(input [][]float64) (*TimeMixerOutput, error) {
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
	mixed := m.pastDecomposableMixing(scales)

	// Compute softmax mixing weights.
	smWeights := make([]float64, len(m.mixWeights))
	copy(smWeights, m.mixWeights)
	normalizeWeights(smWeights)

	// Project each scale's trend and seasonal, combine with mixing weights.
	nf := m.config.NumFeatures
	outLen := m.config.OutputLen
	forecast := make([][]float64, nf)
	for f := 0; f < nf; f++ {
		forecast[f] = make([]float64, outLen)
		for s := 0; s < len(mixed); s++ {
			trendProj := m.projectVector(mixed[s].trend[f], m.trendHeads[s], outLen)
			seasonProj := m.projectVector(mixed[s].seasonal[f], m.seasonalHeads[s], outLen)
			for j := 0; j < outLen; j++ {
				forecast[f][j] += smWeights[s] * (trendProj[j] + seasonProj[j])
			}
		}
	}

	return &TimeMixerOutput{
		Forecast:         forecast,
		MultiScaleOutput: MultiScaleOutput{Scales: mixed},
	}, nil
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

// TrainWindowed trains the TimeMixer on windowed time-series data using AdamW
// with gradient clipping and linear LR warmup.
func (m *TimeMixer) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	if len(windows) == 0 {
		return nil, fmt.Errorf("timemixer: empty training data")
	}
	if len(windows) != len(labels) {
		return nil, fmt.Errorf("timemixer: windows (%d) and labels (%d) length mismatch", len(windows), len(labels))
	}

	nSamples := len(windows)
	outDim := m.config.NumFeatures * m.config.OutputLen

	// Expand scalar labels to match the full output dimension.
	// The original interface passes one scalar per sample; replicate it
	// across all feature/output positions so TrainLoop's MSE is equivalent.
	expandedLabels := make([]float64, nSamples*outDim)
	for s := 0; s < nSamples; s++ {
		for j := 0; j < outDim; j++ {
			expandedLabels[s*outDim+j] = labels[s]
		}
	}

	// Apply defaults for optional fields.
	if config.LR == 0 {
		config.LR = 1e-3
	}
	if config.WeightDecay == 0 {
		config.WeightDecay = 0.01
	}
	if config.GradClip == 0 {
		config.GradClip = 1.0
	}
	if config.WarmupEpochs == 0 {
		config.WarmupEpochs = min(5, config.Epochs/5+1)
	}
	return TrainLoop(m, windows, expandedLabels, config)
}

// ForwardSample runs the TimeMixer forward pass on a single sample and returns
// a flat output with cached activations for BackwardSample.
func (m *TimeMixer) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	msOut, cache := m.forwardWithCache(input)
	pred := m.predict(msOut)

	// Flatten [numFeatures][outputLen] to [numFeatures*outputLen].
	nf := m.config.NumFeatures
	outLen := m.config.OutputLen
	if outLen <= 0 {
		outLen = m.config.InputLen
	}
	flat := make([]float64, nf*outLen)
	for f := 0; f < nf; f++ {
		copy(flat[f*outLen:], pred[f])
	}

	return flat, &timeMixerTrainCache{msOut: msOut, cache: cache}, nil
}

// timeMixerTrainCache holds the forward pass state for BackwardSample.
type timeMixerTrainCache struct {
	msOut *MultiScaleOutput
	cache *timeMixerCache
}

// BackwardSample accumulates parameter gradients for a single sample.
func (m *TimeMixer) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	tc, ok := cacheIface.(*timeMixerTrainCache)
	if !ok {
		return fmt.Errorf("timemixer: invalid cache type")
	}

	nf := m.config.NumFeatures
	outLen := m.config.OutputLen
	if outLen <= 0 {
		outLen = m.config.InputLen
	}

	// Convert flat dOutput [nf*outLen] to dScales for backward.
	dScales := make([]scaleDecomposition, len(tc.msOut.Scales))
	for si := range dScales {
		dScales[si].trend = make([][]float64, nf)
		dScales[si].seasonal = make([][]float64, nf)
		for f := 0; f < nf; f++ {
			dScales[si].trend[f] = make([]float64, m.config.InputLen)
			dScales[si].seasonal[f] = make([]float64, m.config.InputLen)
		}
	}

	// Propagate through predict: average of trend across scales.
	nScales := float64(len(tc.msOut.Scales))
	for f := 0; f < nf; f++ {
		for j := 0; j < outLen; j++ {
			dPred := dOutput[f*outLen+j]
			srcIdx := m.config.InputLen - outLen + j
			if srcIdx < 0 {
				srcIdx = 0
			}
			for si := range dScales {
				dScales[si].trend[f][srcIdx] += dPred / nScales
			}
		}
	}

	// Backward into structured grads, then accumulate into flat grads.
	structGrads := newTimeMixerGrads(m)
	m.backward(dScales, tc.cache, &structGrads)
	gradVec := structGrads.collectGrads(m)

	flatGrads := m.FlatGrads()
	for i := range flatGrads {
		flatGrads[i] += gradVec[i]
	}
	return nil
}

// engineMLPForward runs a two-layer MLP forward pass using Engine[T] ops.
// Input: [numScales], output: [numScales]. Returns nil on engine error.
func (m *TimeMixer) engineMLPForward(mlp *mixingMLP, x []float64) []float64 {
	ctx := context.Background()
	numScales := len(x)
	hiddenSize := len(mlp.b1)

	// Convert input to float32 tensor [1, numScales].
	xF32 := make([]float32, numScales)
	for i, v := range x {
		xF32[i] = float32(v)
	}
	xT, _ := tensor.New[float32]([]int{1, numScales}, xF32)

	// W1^T [numScales, hiddenSize] for x @ W1^T. W1 is [hiddenSize, numScales].
	w1T := make([]float32, numScales*hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < numScales; j++ {
			w1T[j*hiddenSize+i] = float32(mlp.w1[i][j])
		}
	}
	w1Tensor, _ := tensor.New[float32]([]int{numScales, hiddenSize}, w1T)

	// b1 [1, hiddenSize].
	b1F32 := make([]float32, hiddenSize)
	for i, v := range mlp.b1 {
		b1F32[i] = float32(v)
	}
	b1Tensor, _ := tensor.New[float32]([]int{1, hiddenSize}, b1F32)

	// hidden = ReLU(x @ W1^T + b1)
	h1, err := m.engine.MatMul(ctx, xT, w1Tensor)
	if err != nil {
		return nil
	}
	h1b, err := m.engine.Add(ctx, h1, b1Tensor, nil)
	if err != nil {
		return nil
	}
	hidden, err := m.engine.UnaryOp(ctx, h1b, m.ops.ReLU, nil)
	if err != nil {
		return nil
	}

	// W2^T [hiddenSize, numScales] for hidden @ W2^T. W2 is [numScales, hiddenSize].
	w2T := make([]float32, hiddenSize*numScales)
	for i := 0; i < numScales; i++ {
		for j := 0; j < hiddenSize; j++ {
			w2T[j*numScales+i] = float32(mlp.w2[i][j])
		}
	}
	w2Tensor, _ := tensor.New[float32]([]int{hiddenSize, numScales}, w2T)

	// b2 [1, numScales].
	b2F32 := make([]float32, numScales)
	for i, v := range mlp.b2 {
		b2F32[i] = float32(v)
	}
	b2Tensor, _ := tensor.New[float32]([]int{1, numScales}, b2F32)

	// out = hidden @ W2^T + b2
	h2, err := m.engine.MatMul(ctx, hidden, w2Tensor)
	if err != nil {
		return nil
	}
	outT, err := m.engine.Add(ctx, h2, b2Tensor, nil)
	if err != nil {
		return nil
	}

	// Convert back to float64.
	outData := outT.Data()
	out := make([]float64, numScales)
	for i := range out {
		out[i] = float64(outData[i])
	}
	return out
}

// engineBottomUpResidual applies bottom-up residual connections using Engine[T].
func (m *TimeMixer) engineBottomUpResidual(newSeasonal, newTrend []scaleDecomposition, numScales, nf, seqLen int) {
	ctx := context.Background()
	for s := numScales - 2; s >= 0; s-- {
		for f := 0; f < nf; f++ {
			// Seasonal residual via engine.Add.
			sF := make([]float32, seqLen)
			sCoarse := make([]float32, seqLen)
			for t := 0; t < seqLen; t++ {
				sF[t] = float32(newSeasonal[s].seasonal[f][t])
				sCoarse[t] = float32(newSeasonal[s+1].seasonal[f][t])
			}
			sFT, _ := tensor.New[float32]([]int{seqLen}, sF)
			sCoarseT, _ := tensor.New[float32]([]int{seqLen}, sCoarse)
			if sumT, err := m.engine.Add(ctx, sFT, sCoarseT, nil); err == nil {
				for t, v := range sumT.Data() {
					newSeasonal[s].seasonal[f][t] = float64(v)
				}
			} else {
				for t := 0; t < seqLen; t++ {
					newSeasonal[s].seasonal[f][t] += newSeasonal[s+1].seasonal[f][t]
				}
			}

			// Trend residual via engine.Add.
			tF := make([]float32, seqLen)
			tCoarse := make([]float32, seqLen)
			for t := 0; t < seqLen; t++ {
				tF[t] = float32(newTrend[s].trend[f][t])
				tCoarse[t] = float32(newTrend[s+1].trend[f][t])
			}
			tFT, _ := tensor.New[float32]([]int{seqLen}, tF)
			tCoarseT, _ := tensor.New[float32]([]int{seqLen}, tCoarse)
			if sumT, err := m.engine.Add(ctx, tFT, tCoarseT, nil); err == nil {
				for t, v := range sumT.Data() {
					newTrend[s].trend[f][t] = float64(v)
				}
			} else {
				for t := 0; t < seqLen; t++ {
					newTrend[s].trend[f][t] += newTrend[s+1].trend[f][t]
				}
			}
		}
	}
}

// engineProjectVector applies a linear projection via engine.MatMul.
// input: [inputLen], weight: [inputLen][outputLen]. Returns nil on error.
func (m *TimeMixer) engineProjectVector(input []float64, weight [][]float64, outputLen int) []float64 {
	ctx := context.Background()
	inputLen := len(input)

	// Convert input to [1, inputLen] float32 tensor.
	xF32 := make([]float32, inputLen)
	for i, v := range input {
		xF32[i] = float32(v)
	}
	xT, _ := tensor.New[float32]([]int{1, inputLen}, xF32)

	// Weight [inputLen][outputLen] -> flat [inputLen, outputLen] float32.
	wF32 := make([]float32, inputLen*outputLen)
	for i := 0; i < inputLen; i++ {
		for j := 0; j < outputLen; j++ {
			wF32[i*outputLen+j] = float32(weight[i][j])
		}
	}
	wT, _ := tensor.New[float32]([]int{inputLen, outputLen}, wF32)

	// out = x @ W : [1, inputLen] @ [inputLen, outputLen] = [1, outputLen]
	outT, err := m.engine.MatMul(ctx, xT, wT)
	if err != nil {
		return nil
	}

	outData := outT.Data()
	out := make([]float64, outputLen)
	for i := range out {
		out[i] = float64(outData[i])
	}
	return out
}

// FlatGrads returns the internal gradient accumulator.
func (m *TimeMixer) FlatGrads() []float64 {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
	}
	return m.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (m *TimeMixer) ZeroGrads() {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
		return
	}
	for i := range m.grads {
		m.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
// Order: maWeights (per scale), then for each layer: seasonalMLP (w1, b1, w2, b2),
// trendMLP (w1, b1, w2, b2).
func (m *TimeMixer) FlatParams() []*float64 {
	var params []*float64

	// MA weights.
	for s := range m.maWeights {
		for i := range m.maWeights[s] {
			params = append(params, &m.maWeights[s][i])
		}
	}

	// MLP parameters per layer.
	for l := 0; l < m.config.NumLayers; l++ {
		params = append(params, flatMLP(m.seasonalMLPs[l])...)
		params = append(params, flatMLP(m.trendMLPs[l])...)
	}

	return params
}

// flatMLP returns pointers to all parameters of a mixing MLP.
func flatMLP(mlp *mixingMLP) []*float64 {
	var params []*float64
	for i := range mlp.w1 {
		for j := range mlp.w1[i] {
			params = append(params, &mlp.w1[i][j])
		}
	}
	for i := range mlp.b1 {
		params = append(params, &mlp.b1[i])
	}
	for i := range mlp.w2 {
		for j := range mlp.w2[i] {
			params = append(params, &mlp.w2[i][j])
		}
	}
	for i := range mlp.b2 {
		params = append(params, &mlp.b2[i])
	}
	return params
}

// Parameters returns all trainable parameters as graph.Parameter[float32] tensors.
func (m *TimeMixer) Parameters() []*graph.Parameter[float32] {
	var params []*graph.Parameter[float32]

	// MA weights per scale.
	for s := range m.maWeights {
		data := make([]float32, len(m.maWeights[s]))
		for i, v := range m.maWeights[s] {
			data[i] = float32(v)
		}
		t, _ := tensor.New[float32]([]int{len(data)}, data)
		p, _ := graph.NewParameter(fmt.Sprintf("ma_weights.%d", s), t, tensor.New[float32])
		params = append(params, p)
	}

	// MLP parameters per layer.
	for l := 0; l < m.config.NumLayers; l++ {
		params = append(params, mlpParameters(fmt.Sprintf("layer%d.seasonal", l), m.seasonalMLPs[l])...)
		params = append(params, mlpParameters(fmt.Sprintf("layer%d.trend", l), m.trendMLPs[l])...)
	}

	return params
}

// mlpParameters converts a mixingMLP's weights to graph.Parameter[float32] tensors.
func mlpParameters(prefix string, mlp *mixingMLP) []*graph.Parameter[float32] {
	hiddenSize := len(mlp.b1)
	numScales := len(mlp.w1[0])
	var params []*graph.Parameter[float32]

	// W1 [hiddenSize, numScales]
	w1Data := make([]float32, hiddenSize*numScales)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < numScales; j++ {
			w1Data[i*numScales+j] = float32(mlp.w1[i][j])
		}
	}
	w1T, _ := tensor.New[float32]([]int{hiddenSize, numScales}, w1Data)
	w1P, _ := graph.NewParameter(prefix+".w1", w1T, tensor.New[float32])
	params = append(params, w1P)

	// B1 [hiddenSize]
	b1Data := make([]float32, hiddenSize)
	for i, v := range mlp.b1 {
		b1Data[i] = float32(v)
	}
	b1T, _ := tensor.New[float32]([]int{hiddenSize}, b1Data)
	b1P, _ := graph.NewParameter(prefix+".b1", b1T, tensor.New[float32])
	params = append(params, b1P)

	// W2 [numScales, hiddenSize]
	w2Data := make([]float32, numScales*hiddenSize)
	for i := 0; i < numScales; i++ {
		for j := 0; j < hiddenSize; j++ {
			w2Data[i*hiddenSize+j] = float32(mlp.w2[i][j])
		}
	}
	w2T, _ := tensor.New[float32]([]int{numScales, hiddenSize}, w2Data)
	w2P, _ := graph.NewParameter(prefix+".w2", w2T, tensor.New[float32])
	params = append(params, w2P)

	// B2 [numScales]
	b2Data := make([]float32, numScales)
	for i, v := range mlp.b2 {
		b2Data[i] = float32(v)
	}
	b2T, _ := tensor.New[float32]([]int{numScales}, b2Data)
	b2P, _ := graph.NewParameter(prefix+".b2", b2T, tensor.New[float32])
	params = append(params, b2P)

	return params
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (m *TimeMixer) ParamCount() int {
	return len(m.FlatParams())
}

// Compile-time check that TimeMixer implements TrainableBackend.
var _ TrainableBackend = (*TimeMixer)(nil)
