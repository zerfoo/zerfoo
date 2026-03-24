package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// CfCConfig holds the configuration for a CfC model.
type CfCConfig struct {
	InputSize  int // number of input features per time step
	HiddenSize int // hidden state dimension
	OutputSize int // output dimension per time step
	NumLayers  int // number of stacked CfC layers
	OutputLen  int // forecast horizon (number of output time steps)
}

// cfcLayer holds weights for a single CfC recurrent layer.
type cfcLayer struct {
	Wh   [][]float64 // hidden-to-hidden weights [hiddenSize][hiddenSize]
	Wx   [][]float64 // input-to-hidden weights [inputSize][hiddenSize] (first layer) or [hiddenSize][hiddenSize]
	Bh   []float64   // hidden bias [hiddenSize]
	Wtau [][]float64 // time constant weights [inputSize+hiddenSize][hiddenSize]
	Btau []float64   // time constant bias [hiddenSize]
}

// CfC implements Closed-form Continuous-time neural networks
// (Hasani et al., Nature Machine Intelligence 2022).
//
// CfC uses liquid time-constant neurons with analytical ODE solutions,
// avoiding numerical solvers entirely. The closed-form update is:
//
//	tau = sigmoid(W_tau * [x, h] + b_tau)
//	f = exp(-dt / tau)
//	h_new = f * h_old + (1 - f) * tanh(W_x * x + W_h * h_old + b_h)
//
// For uniformly sampled time series, dt = 1.0.
type CfC struct {
	config CfCConfig
	layers []cfcLayer
	outW   [][]float64 // output projection [hiddenSize][outputSize * outputLen]
	outB   []float64   // output bias [outputSize * outputLen]

	// Optional GPU engine for accelerated training. When non-nil,
	// TrainWindowed uses float32 tensor operations instead of the
	// pure-Go float64 CPU path.
	engine    compute.Engine[float32]
	ops       numeric.Arithmetic[float32]
	normMeans [][]float64 // per-channel normalization means from training
	normStds  [][]float64 // per-channel normalization stds from training
}

// CfCOption configures a CfC model.
type CfCOption func(*CfC)

// WithCfCEngine sets the compute engine and arithmetic ops for GPU-accelerated
// training. When set, TrainWindowed dispatches to the engine-based path.
func WithCfCEngine(engine compute.Engine[float32], ops numeric.Arithmetic[float32]) CfCOption {
	return func(c *CfC) {
		c.engine = engine
		c.ops = ops
	}
}

// NewCfC creates a new CfC model with the given configuration.
func NewCfC(config CfCConfig, opts ...CfCOption) (*CfC, error) {
	if config.InputSize <= 0 {
		return nil, fmt.Errorf("cfc: InputSize must be positive, got %d", config.InputSize)
	}
	if config.HiddenSize <= 0 {
		return nil, fmt.Errorf("cfc: HiddenSize must be positive, got %d", config.HiddenSize)
	}
	if config.OutputSize <= 0 {
		return nil, fmt.Errorf("cfc: OutputSize must be positive, got %d", config.OutputSize)
	}
	if config.NumLayers <= 0 {
		return nil, fmt.Errorf("cfc: NumLayers must be positive, got %d", config.NumLayers)
	}
	if config.OutputLen <= 0 {
		return nil, fmt.Errorf("cfc: OutputLen must be positive, got %d", config.OutputLen)
	}

	c := &CfC{config: config}
	c.layers = make([]cfcLayer, config.NumLayers)

	for l := 0; l < config.NumLayers; l++ {
		inSize := config.InputSize
		if l > 0 {
			inSize = config.HiddenSize
		}
		c.layers[l] = newCfCLayer(inSize, config.HiddenSize)
	}

	// Output projection: hidden -> outputSize * outputLen.
	outDim := config.OutputSize * config.OutputLen
	scale := math.Sqrt(2.0 / float64(config.HiddenSize+outDim))
	c.outW = make([][]float64, config.HiddenSize)
	for i := range c.outW {
		c.outW[i] = make([]float64, outDim)
		for j := range c.outW[i] {
			c.outW[i][j] = rand.NormFloat64() * scale
		}
	}
	c.outB = make([]float64, outDim)

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// newCfCLayer initializes a single CfC layer with Xavier weights.
func newCfCLayer(inSize, hiddenSize int) cfcLayer {
	l := cfcLayer{
		Wh:   make([][]float64, hiddenSize),
		Wx:   make([][]float64, inSize),
		Bh:   make([]float64, hiddenSize),
		Wtau: make([][]float64, inSize+hiddenSize),
		Btau: make([]float64, hiddenSize),
	}

	whScale := math.Sqrt(2.0 / float64(hiddenSize+hiddenSize))
	for i := range l.Wh {
		l.Wh[i] = make([]float64, hiddenSize)
		for j := range l.Wh[i] {
			l.Wh[i][j] = rand.NormFloat64() * whScale
		}
	}

	wxScale := math.Sqrt(2.0 / float64(inSize+hiddenSize))
	for i := range l.Wx {
		l.Wx[i] = make([]float64, hiddenSize)
		for j := range l.Wx[i] {
			l.Wx[i][j] = rand.NormFloat64() * wxScale
		}
	}

	tauScale := math.Sqrt(2.0 / float64(inSize+hiddenSize+hiddenSize))
	for i := range l.Wtau {
		l.Wtau[i] = make([]float64, hiddenSize)
		for j := range l.Wtau[i] {
			l.Wtau[i][j] = rand.NormFloat64() * tauScale
		}
	}

	return l
}

// forward runs the CfC forward pass on a single windowed sample.
// input: [seqLen][inputSize], returns: [outputSize * outputLen].
func (c *CfC) forward(input [][]float64) []float64 {
	seqLen := len(input)
	h := make([]float64, c.config.HiddenSize)

	for t := 0; t < seqLen; t++ {
		x := input[t]
		for l := 0; l < c.config.NumLayers; l++ {
			h = c.cfcStep(c.layers[l], x, h)
			x = h // feed hidden state to next layer
		}
	}

	// Output projection.
	outDim := c.config.OutputSize * c.config.OutputLen
	out := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		out[j] = c.outB[j]
		for i := 0; i < c.config.HiddenSize; i++ {
			out[j] += h[i] * c.outW[i][j]
		}
	}
	return out
}

// cfcStep computes a single CfC time step for one layer.
// h(t) = f * h(t-1) + (1-f) * tanh(Wx*x + Wh*h + bh)
// where f = exp(-dt/tau), tau = sigmoid(Wtau*[x,h] + btau), dt = 1.0.
func (c *CfC) cfcStep(layer cfcLayer, x, h []float64) []float64 {
	hiddenSize := c.config.HiddenSize
	inSize := len(layer.Wx)

	// Compute tau = sigmoid(Wtau * [x, h] + btau).
	tau := make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		val := layer.Btau[j]
		for i := 0; i < inSize; i++ {
			val += x[i] * layer.Wtau[i][j]
		}
		for i := 0; i < hiddenSize; i++ {
			val += h[i] * layer.Wtau[inSize+i][j]
		}
		tau[j] = sigmoid(val)
	}

	// Compute pre-activation: tanh(Wx*x + Wh*h + bh).
	preact := make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		val := layer.Bh[j]
		for i := 0; i < inSize; i++ {
			val += x[i] * layer.Wx[i][j]
		}
		for i := 0; i < hiddenSize; i++ {
			val += h[i] * layer.Wh[i][j]
		}
		preact[j] = math.Tanh(val)
	}

	// Closed-form ODE update: h_new = f * h_old + (1-f) * preact
	// where f = exp(-1.0 / tau). Clamp tau to avoid division by zero.
	hNew := make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		tauClamped := math.Max(tau[j], 1e-6)
		f := math.Exp(-1.0 / tauClamped)
		hNew[j] = f*h[j] + (1-f)*preact[j]
	}

	return hNew
}

// sigmoid computes 1 / (1 + exp(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// TrainWindowed trains the CfC model on windowed data using AdamW with BPTT.
// windows: [nSamples][channels][inputLen] — input windows.
// labels: flat slice of length nSamples * outputSize * outputLen.
func (c *CfC) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("cfc: empty training set")
	}

	outDim := c.config.OutputSize * c.config.OutputLen
	expectedLabels := nSamples * outDim
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("cfc: expected %d labels, got %d", expectedLabels, len(labels))
	}

	if c.engine != nil {
		return c.trainWindowedEngine(windows, labels, config)
	}

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, c.normMeans, c.normStds = normalizeWindows(windows)

	nParams := c.paramCount()
	m := make([]float64, nParams)
	v := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			batchWindows := windows[start:end]
			batchLabels := labels[start*outDim : end*outDim]
			bs := end - start

			grads := make([]float64, nParams)
			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				// Convert [channels][inputLen] to [inputLen][channels] for recurrent processing.
				seqInput := transposeWindow(batchWindows[s])
				sampleGrads, pred := c.backwardSample(seqInput)
				sampleLabels := batchLabels[s*outDim : (s+1)*outDim]

				for j := 0; j < outDim; j++ {
					diff := pred[j] - sampleLabels[j]
					batchLoss += diff * diff
					dOut := 2.0 * diff / float64(bs*outDim)
					// Accumulate gradients scaled by output gradient.
					for p := range grads {
						grads[p] += dOut * sampleGrads[j][p]
					}
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			// AdamW update with LR warmup.
			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			params := c.flatParams()
			for i := range params {
				m[i] = config.Beta1*m[i] + (1-config.Beta1)*grads[i]
				v[i] = config.Beta2*v[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := m[i] / (1 - math.Pow(config.Beta1, t))
				vHat := v[i] / (1 - math.Pow(config.Beta2, t))
				*params[i] = *params[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*params[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		// Early halt on NaN/Inf loss.
		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("cfc: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// backwardSample computes per-output-element gradients for a single sample using BPTT.
// Returns gradients[outDim][nParams] and predictions[outDim].
func (c *CfC) backwardSample(input [][]float64) ([][]float64, []float64) {
	seqLen := len(input)
	hiddenSize := c.config.HiddenSize
	numLayers := c.config.NumLayers
	outDim := c.config.OutputSize * c.config.OutputLen
	nParams := c.paramCount()

	// Forward pass storing all intermediate states for BPTT.
	// hStates[t][l] = hidden state at time t, layer l (after CfC step).
	// hStates[t][l] has shape [hiddenSize].
	type stepState struct {
		h      []float64   // hidden state after step
		tau    []float64   // time constants
		preact []float64   // tanh pre-activation (before tanh)
		x      []float64   // input to this layer
		hPrev  []float64   // hidden state before step
	}

	states := make([][]stepState, seqLen)
	hPrev := make([][]float64, numLayers)
	for l := 0; l < numLayers; l++ {
		hPrev[l] = make([]float64, hiddenSize)
	}

	for t := 0; t < seqLen; t++ {
		states[t] = make([]stepState, numLayers)
		x := input[t]
		for l := 0; l < numLayers; l++ {
			layer := c.layers[l]
			inSize := len(layer.Wx)
			ss := stepState{
				x:     make([]float64, len(x)),
				hPrev: make([]float64, hiddenSize),
			}
			copy(ss.x, x)
			copy(ss.hPrev, hPrev[l])

			// Compute tau.
			ss.tau = make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				val := layer.Btau[j]
				for i := 0; i < inSize; i++ {
					val += x[i] * layer.Wtau[i][j]
				}
				for i := 0; i < hiddenSize; i++ {
					val += hPrev[l][i] * layer.Wtau[inSize+i][j]
				}
				ss.tau[j] = sigmoid(val)
			}

			// Compute pre-activation (value before tanh).
			ss.preact = make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				val := layer.Bh[j]
				for i := 0; i < inSize; i++ {
					val += x[i] * layer.Wx[i][j]
				}
				for i := 0; i < hiddenSize; i++ {
					val += hPrev[l][i] * layer.Wh[i][j]
				}
				ss.preact[j] = val
			}

			// h_new = f * h_old + (1-f) * tanh(preact)
			ss.h = make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				tauClamped := math.Max(ss.tau[j], 1e-6)
				f := math.Exp(-1.0 / tauClamped)
				ss.h[j] = f*hPrev[l][j] + (1-f)*math.Tanh(ss.preact[j])
			}

			hPrev[l] = ss.h
			x = ss.h
			states[t][l] = ss
		}
	}

	// Final hidden state.
	finalH := hPrev[numLayers-1]

	// Forward through output projection to get predictions.
	pred := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		pred[j] = c.outB[j]
		for i := 0; i < hiddenSize; i++ {
			pred[j] += finalH[i] * c.outW[i][j]
		}
	}

	// Compute gradients per output element.
	allGrads := make([][]float64, outDim)
	for oi := 0; oi < outDim; oi++ {
		grads := make([]float64, nParams)

		// Gradient of output w.r.t. output projection weights.
		outWOff := c.outWParamOffset()
		for i := 0; i < hiddenSize; i++ {
			grads[outWOff+i*outDim+oi] = finalH[i]
		}
		grads[c.outBParamOffset()+oi] = 1.0

		// dL/dh for the final hidden state from the output projection.
		dh := make([]float64, hiddenSize)
		for i := 0; i < hiddenSize; i++ {
			dh[i] = c.outW[i][oi]
		}

		// BPTT: backpropagate through time and layers.
		for t := seqLen - 1; t >= 0; t-- {
			// Back through layers (reverse order).
			for l := numLayers - 1; l >= 0; l-- {
				ss := states[t][l]
				layer := c.layers[l]
				inSize := len(layer.Wx)
				layerOff := c.layerParamOffset(l)

				// h_new = f * h_old + (1-f) * tanh(preact)
				// f = exp(-1/tau), tau = sigmoid(z_tau)

				// For each hidden unit j:
				tanhPreact := make([]float64, hiddenSize)
				fVals := make([]float64, hiddenSize)
				for j := 0; j < hiddenSize; j++ {
					tauClamped := math.Max(ss.tau[j], 1e-6)
					fVals[j] = math.Exp(-1.0 / tauClamped)
					tanhPreact[j] = math.Tanh(ss.preact[j])
				}

				// Gradient w.r.t. preact: dh[j] * (1-f[j]) * (1 - tanh^2(preact[j]))
				dPreact := make([]float64, hiddenSize)
				for j := 0; j < hiddenSize; j++ {
					dPreact[j] = dh[j] * (1 - fVals[j]) * (1 - tanhPreact[j]*tanhPreact[j])
				}

				// Gradient w.r.t. tau:
				// df/dtau = exp(-1/tau) * (1/tau^2) = f / tau^2
				// dh/df = h_old - tanh(preact)
				// dh/dtau = dh/df * df/dtau
				// dtau/dz = tau * (1-tau) (sigmoid derivative)
				dZtau := make([]float64, hiddenSize)
				for j := 0; j < hiddenSize; j++ {
					tauClamped := math.Max(ss.tau[j], 1e-6)
					dfDtau := fVals[j] / (tauClamped * tauClamped)
					dhDf := ss.hPrev[j] - tanhPreact[j]
					dtauDz := ss.tau[j] * (1 - ss.tau[j])
					dZtau[j] = dh[j] * dhDf * dfDtau * dtauDz
				}

				// Accumulate parameter gradients.
				// Wx gradients.
				wxOff := layerOff + hiddenSize*hiddenSize // after Wh
				for i := 0; i < inSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wxOff+i*hiddenSize+j] += dPreact[j] * ss.x[i]
					}
				}

				// Wh gradients.
				whOff := layerOff
				for i := 0; i < hiddenSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[whOff+i*hiddenSize+j] += dPreact[j] * ss.hPrev[i]
					}
				}

				// Bh gradients.
				bhOff := layerOff + hiddenSize*hiddenSize + inSize*hiddenSize
				for j := 0; j < hiddenSize; j++ {
					grads[bhOff+j] += dPreact[j]
				}

				// Wtau gradients.
				wtauOff := bhOff + hiddenSize
				for i := 0; i < inSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wtauOff+i*hiddenSize+j] += dZtau[j] * ss.x[i]
					}
				}
				for i := 0; i < hiddenSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wtauOff+(inSize+i)*hiddenSize+j] += dZtau[j] * ss.hPrev[i]
					}
				}

				// Btau gradients.
				btauOff := wtauOff + (inSize+hiddenSize)*hiddenSize
				for j := 0; j < hiddenSize; j++ {
					grads[btauOff+j] += dZtau[j]
				}

				// Propagate dh backward.
				// dh/dh_prev = f[j] (from the ODE update)
				//            + (1-f[j]) * (1-tanh^2) * dPreact/dh_prev (through Wh)
				//            + dhDf * dfDtau * dtauDz * dZtau/dh_prev (through Wtau's h part)
				dhPrev := make([]float64, hiddenSize)
				for i := 0; i < hiddenSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						dhPrev[i] += dPreact[j] * layer.Wh[i][j]
						dhPrev[i] += dZtau[j] * layer.Wtau[inSize+i][j]
					}
					dhPrev[i] += dh[i] * fVals[i]
				}

				// If not the first layer, propagate to the layer below via dx.
				if l > 0 {
					dx := make([]float64, inSize)
					for i := 0; i < inSize; i++ {
						for j := 0; j < hiddenSize; j++ {
							dx[i] += dPreact[j] * layer.Wx[i][j]
							dx[i] += dZtau[j] * layer.Wtau[i][j]
						}
					}
					dh = dx
				} else {
					dh = dhPrev
				}

				// For the current layer, update dh for the next time step backward.
				if l == 0 {
					dh = dhPrev
				}
			}
		}

		allGrads[oi] = grads
	}

	return allGrads, pred
}

// transposeWindow converts [channels][seqLen] to [seqLen][channels].
func transposeWindow(window [][]float64) [][]float64 {
	if len(window) == 0 {
		return nil
	}
	channels := len(window)
	seqLen := len(window[0])
	out := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		out[t] = make([]float64, channels)
		for c := 0; c < channels; c++ {
			out[t][c] = window[c][t]
		}
	}
	return out
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * outputSize * outputLen.
func (c *CfC) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := c.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("cfc: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("cfc: empty input")
	}

	// Apply normalization from training if available.
	if c.normMeans != nil {
		windows = applyNormalization(windows, c.normMeans, c.normStds)
	}

	outDim := c.config.OutputSize * c.config.OutputLen
	out := make([]float64, 0, nSamples*outDim)
	for _, w := range windows {
		seqInput := transposeWindow(w)
		pred := c.forward(seqInput)
		out = append(out, pred...)
	}
	return out, nil
}

// paramCount returns the total number of trainable parameters.
func (c *CfC) paramCount() int {
	total := 0
	for l := 0; l < c.config.NumLayers; l++ {
		inSize := c.config.InputSize
		if l > 0 {
			inSize = c.config.HiddenSize
		}
		hiddenSize := c.config.HiddenSize
		// Wh: hiddenSize * hiddenSize
		// Wx: inSize * hiddenSize
		// Bh: hiddenSize
		// Wtau: (inSize + hiddenSize) * hiddenSize
		// Btau: hiddenSize
		total += hiddenSize*hiddenSize + inSize*hiddenSize + hiddenSize +
			(inSize+hiddenSize)*hiddenSize + hiddenSize
	}
	// Output projection: hiddenSize * outDim + outDim
	outDim := c.config.OutputSize * c.config.OutputLen
	total += c.config.HiddenSize*outDim + outDim
	return total
}

// layerParamOffset returns the flat parameter offset for CfC layer l.
// Layout per layer: Wh, Wx, Bh, Wtau, Btau.
func (c *CfC) layerParamOffset(l int) int {
	offset := 0
	for i := 0; i < l; i++ {
		inSize := c.config.InputSize
		if i > 0 {
			inSize = c.config.HiddenSize
		}
		hiddenSize := c.config.HiddenSize
		offset += hiddenSize*hiddenSize + inSize*hiddenSize + hiddenSize +
			(inSize+hiddenSize)*hiddenSize + hiddenSize
	}
	return offset
}

// outWParamOffset returns the flat offset where output projection weights start.
func (c *CfC) outWParamOffset() int {
	return c.layerParamOffset(c.config.NumLayers)
}

// outBParamOffset returns the flat offset where output projection bias starts.
func (c *CfC) outBParamOffset() int {
	outDim := c.config.OutputSize * c.config.OutputLen
	return c.outWParamOffset() + c.config.HiddenSize*outDim
}

// flatParams returns pointers to all trainable parameters in a flat slice.
// Order per layer: Wh (row-major), Wx (row-major), Bh, Wtau (row-major), Btau.
// Then: outW (row-major), outB.
func (c *CfC) flatParams() []*float64 {
	n := c.paramCount()
	params := make([]*float64, 0, n)
	for l := 0; l < c.config.NumLayers; l++ {
		layer := &c.layers[l]
		for i := range layer.Wh {
			for j := range layer.Wh[i] {
				params = append(params, &layer.Wh[i][j])
			}
		}
		for i := range layer.Wx {
			for j := range layer.Wx[i] {
				params = append(params, &layer.Wx[i][j])
			}
		}
		for j := range layer.Bh {
			params = append(params, &layer.Bh[j])
		}
		for i := range layer.Wtau {
			for j := range layer.Wtau[i] {
				params = append(params, &layer.Wtau[i][j])
			}
		}
		for j := range layer.Btau {
			params = append(params, &layer.Btau[j])
		}
	}
	for i := range c.outW {
		for j := range c.outW[i] {
			params = append(params, &c.outW[i][j])
		}
	}
	for j := range c.outB {
		params = append(params, &c.outB[j])
	}
	return params
}

// cfcWeights is the JSON-serializable form of CfC parameters.
type cfcWeights struct {
	Config    CfCConfig       `json:"config"`
	Layers    []cfcLayerFile  `json:"layers"`
	OutW      [][]float64     `json:"out_w"`
	OutB      []float64       `json:"out_b"`
	NormMeans [][]float64     `json:"norm_means,omitempty"`
	NormStds  [][]float64     `json:"norm_stds,omitempty"`
}

type cfcLayerFile struct {
	Wh   [][]float64 `json:"wh"`
	Wx   [][]float64 `json:"wx"`
	Bh   []float64   `json:"bh"`
	Wtau [][]float64 `json:"wtau"`
	Btau []float64   `json:"btau"`
}

// SaveWeights writes the model weights to a JSON file.
func (c *CfC) SaveWeights(path string) error {
	w := cfcWeights{
		Config:    c.config,
		OutW:      c.outW,
		OutB:      c.outB,
		NormMeans: c.normMeans,
		NormStds:  c.normStds,
	}
	for _, l := range c.layers {
		w.Layers = append(w.Layers, cfcLayerFile{
			Wh:   l.Wh,
			Wx:   l.Wx,
			Bh:   l.Bh,
			Wtau: l.Wtau,
			Btau: l.Btau,
		})
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("cfc: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (c *CfC) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w cfcWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != c.config {
		return fmt.Errorf("cfc: config mismatch: file has %+v, model has %+v", w.Config, c.config)
	}
	if len(w.Layers) != len(c.layers) {
		return fmt.Errorf("cfc: layer count mismatch: file=%d, model=%d", len(w.Layers), len(c.layers))
	}
	for i, lf := range w.Layers {
		c.layers[i].Wh = lf.Wh
		c.layers[i].Wx = lf.Wx
		c.layers[i].Bh = lf.Bh
		c.layers[i].Wtau = lf.Wtau
		c.layers[i].Btau = lf.Btau
	}
	c.outW = w.OutW
	c.outB = w.OutB
	c.normMeans = w.NormMeans
	c.normStds = w.NormStds
	return nil
}
