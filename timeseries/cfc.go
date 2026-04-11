package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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
	grads     []float64   // gradient accumulator for TrainableBackend
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

	// Output projection via cpuEngine64.MatMul: h[1, hiddenSize] @ outW[hiddenSize, outDim] + outB.
	ctx := context.Background()
	outDim := c.config.OutputSize * c.config.OutputLen
	proj := cfcMatVecF64(ctx, h, c.outW, c.config.HiddenSize, outDim)
	out := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		out[j] = proj[j] + c.outB[j]
	}
	return out
}

// ForwardBatch runs the CfC forward pass on a batch of windowed samples.
// windows: [batch][channels][inputLen], returns: [batch][outputSize * outputLen].
// Each sample maintains an independent hidden state. The ODE integration is
// batched across samples at each time step to reduce allocation overhead.
func (c *CfC) ForwardBatch(windows [][][]float64) [][]float64 {
	batch := len(windows)
	if batch == 0 {
		return nil
	}

	hiddenSize := c.config.HiddenSize
	numLayers := c.config.NumLayers
	outDim := c.config.OutputSize * c.config.OutputLen

	// Transpose all windows: [batch][channels][inputLen] -> [batch][seqLen][channels].
	inputs := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		inputs[b] = transposeWindow(windows[b])
	}

	seqLen := len(inputs[0])

	// Allocate per-sample hidden states: [batch][hiddenSize].
	h := make([][]float64, batch)
	for b := 0; b < batch; b++ {
		h[b] = make([]float64, hiddenSize)
	}

	// Process all time steps, batching samples at each step.
	for t := 0; t < seqLen; t++ {
		// x[b] is the input for sample b at time t.
		x := make([][]float64, batch)
		for b := 0; b < batch; b++ {
			x[b] = inputs[b][t]
		}

		for l := 0; l < numLayers; l++ {
			layer := c.layers[l]

			for b := 0; b < batch; b++ {
				hNew := c.cfcStep(layer, x[b], h[b])
				h[b] = hNew
				x[b] = hNew // feed hidden state to next layer
			}
		}
	}

	// Output projection for each sample via cpuEngine64.MatMul.
	ctx := context.Background()
	out := make([][]float64, batch)
	for b := 0; b < batch; b++ {
		proj := cfcMatVecF64(ctx, h[b], c.outW, hiddenSize, outDim)
		out[b] = make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			out[b][j] = proj[j] + c.outB[j]
		}
	}
	return out
}

// cfcMatVecF64 computes vec[1, rows] @ mat[rows, cols] using cpuEngine64.MatMul.
// Returns a slice of length cols. Falls back to scalar multiply on error.
func cfcMatVecF64(ctx context.Context, vec []float64, mat [][]float64, rows, cols int) []float64 {
	vT, err := tensor.New[float64]([]int{1, rows}, vec)
	if err != nil {
		return cfcScalarMatVecF64(vec, mat, rows, cols)
	}
	mFlat := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		copy(mFlat[i*cols:], mat[i])
	}
	mT, err := tensor.New[float64]([]int{rows, cols}, mFlat)
	if err != nil {
		return cfcScalarMatVecF64(vec, mat, rows, cols)
	}
	out, err := cpuEngine64.MatMul(ctx, vT, mT)
	if err != nil {
		return cfcScalarMatVecF64(vec, mat, rows, cols)
	}
	return out.Data()
}

// cfcScalarMatVecF64 computes vec @ mat on the CPU as a fallback.
func cfcScalarMatVecF64(vec []float64, mat [][]float64, rows, cols int) []float64 {
	out := make([]float64, cols)
	for j := 0; j < cols; j++ {
		for i := 0; i < rows; i++ {
			out[j] += vec[i] * mat[i][j]
		}
	}
	return out
}

// cfcMatVecF64TransposeW computes vec[1, cols] @ W^T[cols, rows] using cpuEngine64.MatMul.
// W is stored as [rows][cols], so W^T is [cols][rows].
// Returns a slice of length rows.
func cfcMatVecF64TransposeW(ctx context.Context, vec []float64, w [][]float64, rows, cols int) []float64 {
	// Transpose W to [cols][rows].
	wt := make([]float64, cols*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			wt[j*rows+i] = w[i][j]
		}
	}
	vT, err := tensor.New[float64]([]int{1, cols}, vec)
	if err != nil {
		return cfcScalarMatVecTransposeF64(vec, w, rows, cols)
	}
	wtT, err := tensor.New[float64]([]int{cols, rows}, wt)
	if err != nil {
		return cfcScalarMatVecTransposeF64(vec, w, rows, cols)
	}
	out, err := cpuEngine64.MatMul(ctx, vT, wtT)
	if err != nil {
		return cfcScalarMatVecTransposeF64(vec, w, rows, cols)
	}
	return out.Data()
}

// cfcScalarMatVecTransposeF64 computes vec @ W^T on CPU as fallback.
func cfcScalarMatVecTransposeF64(vec []float64, w [][]float64, rows, cols int) []float64 {
	out := make([]float64, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i] += vec[j] * w[i][j]
		}
	}
	return out
}

// cfcStep computes a single CfC time step for one layer.
// h(t) = f * h(t-1) + (1-f) * tanh(Wx*x + Wh*h + bh)
// where f = exp(-dt/tau), tau = sigmoid(Wtau*[x,h] + btau), dt = 1.0.
func (c *CfC) cfcStep(layer cfcLayer, x, h []float64) []float64 {
	ctx := context.Background()
	hiddenSize := c.config.HiddenSize
	inSize := len(layer.Wx)

	// Compute tau = sigmoid(Wtau * [x, h] + btau).
	// Split Wtau into x-part [inSize, hiddenSize] and h-part [hiddenSize, hiddenSize].
	tauX := cfcMatVecF64(ctx, x, layer.Wtau[:inSize], inSize, hiddenSize)
	tauH := cfcMatVecF64(ctx, h, layer.Wtau[inSize:], hiddenSize, hiddenSize)
	tau := make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		tau[j] = sigmoid(layer.Btau[j] + tauX[j] + tauH[j])
	}

	// Compute pre-activation: tanh(Wx*x + Wh*h + bh).
	preX := cfcMatVecF64(ctx, x, layer.Wx, inSize, hiddenSize)
	preH := cfcMatVecF64(ctx, h, layer.Wh, hiddenSize, hiddenSize)
	preact := make([]float64, hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		preact[j] = math.Tanh(layer.Bh[j] + preX[j] + preH[j])
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

// cfcCache holds activations from a forward pass needed for backpropagation.
type cfcCache struct {
	seqInput [][]float64 // transposed input [seqLen][channels]
	output   []float64   // model output [outputSize * outputLen]
}

// ForwardSample runs the CfC forward pass on a single sample and returns
// a flat output with cached activations for BackwardSample.
func (c *CfC) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	seqInput := transposeWindow(input)
	pred := c.forward(seqInput)

	cache := &cfcCache{
		seqInput: seqInput,
		output:   pred,
	}
	return pred, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (c *CfC) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*cfcCache)
	if !ok {
		return fmt.Errorf("cfc: invalid cache type")
	}

	if c.grads == nil {
		c.grads = make([]float64, c.paramCount())
	}

	sampleGrads := c.backwardSample(cache.seqInput, dOutput)
	for i := range c.grads {
		c.grads[i] += sampleGrads[i]
	}
	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (c *CfC) FlatGrads() []float64 {
	if c.grads == nil {
		c.grads = make([]float64, c.paramCount())
	}
	return c.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (c *CfC) ZeroGrads() {
	if c.grads == nil {
		c.grads = make([]float64, c.paramCount())
		return
	}
	for i := range c.grads {
		c.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
// Order per layer: Wh (row-major), Wx (row-major), Bh, Wtau (row-major), Btau.
// Then: outW (row-major), outB.
func (c *CfC) FlatParams() []*float64 {
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

// Parameters returns all trainable parameters as float32 graph parameters.
func (c *CfC) Parameters() []*graph.Parameter[float32] {
	var params []*graph.Parameter[float32]
	idx := 0
	addParam := func(name string, data []float64, shape []int) {
		f32 := make([]float32, len(data))
		for i, v := range data {
			f32[i] = float32(v)
		}
		t, _ := tensor.New[float32](shape, f32)
		p, _ := graph.NewParameter(fmt.Sprintf("%s_%d", name, idx), t, tensor.New[float32])
		params = append(params, p)
		idx++
	}
	for l := 0; l < c.config.NumLayers; l++ {
		layer := c.layers[l]
		inSize := len(layer.Wx)
		addParam("wh", cfcFlatten2D(layer.Wh), []int{c.config.HiddenSize, c.config.HiddenSize})
		addParam("wx", cfcFlatten2D(layer.Wx), []int{inSize, c.config.HiddenSize})
		addParam("bh", layer.Bh, []int{c.config.HiddenSize})
		addParam("wtau", cfcFlatten2D(layer.Wtau), []int{inSize + c.config.HiddenSize, c.config.HiddenSize})
		addParam("btau", layer.Btau, []int{c.config.HiddenSize})
	}
	outDim := c.config.OutputSize * c.config.OutputLen
	addParam("outw", cfcFlatten2D(c.outW), []int{c.config.HiddenSize, outDim})
	addParam("outb", c.outB, []int{outDim})
	return params
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (c *CfC) ParamCount() int {
	return c.paramCount()
}

// Compile-time check that CfC implements TrainableBackend.
var _ TrainableBackend = (*CfC)(nil)

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

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, c.normMeans, c.normStds = normalizeWindows(windows)

	return TrainLoop(c, windows, labels, config)
}

// backwardSample computes gradients for a single sample using BPTT with a
// vector-Jacobian product. dLoss is the upstream gradient (length outDim).
// Returns a single gradient vector (length nParams).
func (c *CfC) backwardSample(input [][]float64, dLoss []float64) []float64 {
	seqLen := len(input)
	hiddenSize := c.config.HiddenSize
	numLayers := c.config.NumLayers
	outDim := c.config.OutputSize * c.config.OutputLen
	nParams := c.paramCount()

	// Forward pass storing all intermediate states for BPTT.
	type stepState struct {
		h      []float64 // hidden state after step
		tau    []float64 // time constants
		preact []float64 // pre-activation (before tanh)
		x      []float64 // input to this layer
		hPrev  []float64 // hidden state before step
	}

	states := make([][]stepState, seqLen)
	hPrev := make([][]float64, numLayers)
	for l := 0; l < numLayers; l++ {
		hPrev[l] = make([]float64, hiddenSize)
	}

	ctx := context.Background()
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

			// Compute tau via Engine MatMul: sigmoid(Wtau * [x, h] + btau).
			tauX := cfcMatVecF64(ctx, x, layer.Wtau[:inSize], inSize, hiddenSize)
			tauH := cfcMatVecF64(ctx, hPrev[l], layer.Wtau[inSize:], hiddenSize, hiddenSize)
			ss.tau = make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				ss.tau[j] = sigmoid(layer.Btau[j] + tauX[j] + tauH[j])
			}

			// Compute pre-activation via Engine MatMul: Wx*x + Wh*h + bh.
			preX := cfcMatVecF64(ctx, x, layer.Wx, inSize, hiddenSize)
			preH := cfcMatVecF64(ctx, hPrev[l], layer.Wh, hiddenSize, hiddenSize)
			ss.preact = make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				ss.preact[j] = layer.Bh[j] + preX[j] + preH[j]
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

	grads := make([]float64, nParams)

	// Output projection backward: dOutW[i][j] += finalH[i] * dLoss[j], dOutB[j] += dLoss[j].
	// Use Engine: outer product finalH[hiddenSize,1] @ dLoss[1,outDim] -> [hiddenSize,outDim].
	outWOff := c.outWParamOffset()
	outBOff := c.outBParamOffset()
	hT, _ := tensor.New[float64]([]int{hiddenSize, 1}, finalH)
	dLT, _ := tensor.New[float64]([]int{1, outDim}, dLoss)
	dOutWT, outerErr := cpuEngine64.MatMul(ctx, hT, dLT)
	if outerErr == nil {
		dOutWData := dOutWT.Data()
		copy(grads[outWOff:outWOff+hiddenSize*outDim], dOutWData)
	} else {
		for i := 0; i < hiddenSize; i++ {
			for j := 0; j < outDim; j++ {
				grads[outWOff+i*outDim+j] = finalH[i] * dLoss[j]
			}
		}
	}
	for j := 0; j < outDim; j++ {
		grads[outBOff+j] = dLoss[j]
	}

	// dH = dLoss @ outW^T via Engine: dLoss[1,outDim] @ outW^T -> [1,hiddenSize].
	// Equivalently: outW[hiddenSize,outDim]^T @ dLoss^T, but simpler as matmul with transposed outW.
	outWFlat := make([]float64, hiddenSize*outDim)
	for i := 0; i < hiddenSize; i++ {
		copy(outWFlat[i*outDim:], c.outW[i])
	}
	// Transpose outW to [outDim, hiddenSize].
	outWtFlat := make([]float64, outDim*hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < outDim; j++ {
			outWtFlat[j*hiddenSize+i] = outWFlat[i*outDim+j]
		}
	}
	dh := make([]float64, hiddenSize)
	dLossVec, _ := tensor.New[float64]([]int{1, outDim}, dLoss)
	outWtT, _ := tensor.New[float64]([]int{outDim, hiddenSize}, outWtFlat)
	dhT, dhErr := cpuEngine64.MatMul(ctx, dLossVec, outWtT)
	if dhErr == nil {
		copy(dh, dhT.Data())
	} else {
		for i := 0; i < hiddenSize; i++ {
			for j := 0; j < outDim; j++ {
				dh[i] += c.outW[i][j] * dLoss[j]
			}
		}
	}

	// BPTT: backpropagate through time and layers.
	for t := seqLen - 1; t >= 0; t-- {
		for l := numLayers - 1; l >= 0; l-- {
			ss := states[t][l]
			layer := c.layers[l]
			inSize := len(layer.Wx)
			layerOff := c.layerParamOffset(l)

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

			// Gradient w.r.t. tau pre-activation.
			dZtau := make([]float64, hiddenSize)
			for j := 0; j < hiddenSize; j++ {
				tauClamped := math.Max(ss.tau[j], 1e-6)
				dfDtau := fVals[j] / (tauClamped * tauClamped)
				dhDf := ss.hPrev[j] - tanhPreact[j]
				dtauDz := ss.tau[j] * (1 - ss.tau[j])
				dZtau[j] = dh[j] * dhDf * dfDtau * dtauDz
			}

			// Accumulate parameter gradients using Engine outer products.
			// Wx gradients: dPreact[1,h] outer ss.x[1,inSize] -> [inSize,h].
			wxOff := layerOff + hiddenSize*hiddenSize
			xColT, _ := tensor.New[float64]([]int{inSize, 1}, ss.x)
			dPreactRowT, _ := tensor.New[float64]([]int{1, hiddenSize}, dPreact)
			dWxT, wxErr := cpuEngine64.MatMul(ctx, xColT, dPreactRowT)
			if wxErr == nil {
				dWxData := dWxT.Data()
				for i := range dWxData {
					grads[wxOff+i] += dWxData[i]
				}
			} else {
				for i := 0; i < inSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wxOff+i*hiddenSize+j] += dPreact[j] * ss.x[i]
					}
				}
			}

			// Wh gradients: dPreact[1,h] outer ss.hPrev[1,h] -> [h,h].
			whOff := layerOff
			hpColT, _ := tensor.New[float64]([]int{hiddenSize, 1}, ss.hPrev)
			dWhT, whErr := cpuEngine64.MatMul(ctx, hpColT, dPreactRowT)
			if whErr == nil {
				dWhData := dWhT.Data()
				for i := range dWhData {
					grads[whOff+i] += dWhData[i]
				}
			} else {
				for i := 0; i < hiddenSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[whOff+i*hiddenSize+j] += dPreact[j] * ss.hPrev[i]
					}
				}
			}

			// Bh gradients.
			bhOff := layerOff + hiddenSize*hiddenSize + inSize*hiddenSize
			for j := 0; j < hiddenSize; j++ {
				grads[bhOff+j] += dPreact[j]
			}

			// Wtau gradients: dZtau outer x -> [inSize, h] and dZtau outer hPrev -> [h, h].
			wtauOff := bhOff + hiddenSize
			dZtauRowT, _ := tensor.New[float64]([]int{1, hiddenSize}, dZtau)
			dWtauXT, wtxErr := cpuEngine64.MatMul(ctx, xColT, dZtauRowT)
			if wtxErr == nil {
				dWtauXData := dWtauXT.Data()
				for i := range dWtauXData {
					grads[wtauOff+i] += dWtauXData[i]
				}
			} else {
				for i := 0; i < inSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wtauOff+i*hiddenSize+j] += dZtau[j] * ss.x[i]
					}
				}
			}
			dWtauHT, wthErr := cpuEngine64.MatMul(ctx, hpColT, dZtauRowT)
			if wthErr == nil {
				dWtauHData := dWtauHT.Data()
				for i := range dWtauHData {
					grads[wtauOff+inSize*hiddenSize+i] += dWtauHData[i]
				}
			} else {
				for i := 0; i < hiddenSize; i++ {
					for j := 0; j < hiddenSize; j++ {
						grads[wtauOff+(inSize+i)*hiddenSize+j] += dZtau[j] * ss.hPrev[i]
					}
				}
			}

			// Btau gradients.
			btauOff := wtauOff + (inSize+hiddenSize)*hiddenSize
			for j := 0; j < hiddenSize; j++ {
				grads[btauOff+j] += dZtau[j]
			}

			// Propagate dh backward via Engine MatMul.
			// dhPrev[i] = sum_j(dPreact[j] * Wh[i][j]) + sum_j(dZtau[j] * Wtau[inSize+i][j]) + dh[i] * fVals[i]
			// = dPreact[1,h] @ Wh^T[h,h] + dZtau[1,h] @ Wtau_h^T[h,h] + dh * f (element-wise)
			dhPrevPreact := cfcMatVecF64TransposeW(ctx, dPreact, layer.Wh, hiddenSize, hiddenSize)
			dhPrevZtau := cfcMatVecF64TransposeW(ctx, dZtau, layer.Wtau[inSize:], hiddenSize, hiddenSize)
			dhPrev := make([]float64, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				dhPrev[i] = dhPrevPreact[i] + dhPrevZtau[i] + dh[i]*fVals[i]
			}

			if l > 0 {
				// dx[i] = sum_j(dPreact[j] * Wx[i][j]) + sum_j(dZtau[j] * Wtau[i][j])
				dxPreact := cfcMatVecF64TransposeW(ctx, dPreact, layer.Wx, inSize, hiddenSize)
				dxZtau := cfcMatVecF64TransposeW(ctx, dZtau, layer.Wtau[:inSize], inSize, hiddenSize)
				dx := make([]float64, inSize)
				for i := 0; i < inSize; i++ {
					dx[i] = dxPreact[i] + dxZtau[i]
				}
				dh = dx
			} else {
				dh = dhPrev
			}

			if l == 0 {
				dh = dhPrev
			}
		}
	}

	return grads
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

// cfcFlatten2D flattens a 2D float64 slice to a 1D slice (row-major).
func cfcFlatten2D(m [][]float64) []float64 {
	var out []float64
	for _, row := range m {
		out = append(out, row...)
	}
	return out
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
