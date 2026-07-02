package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zerfoo/training/scheduler"
)

// NHiTSConfig holds the configuration for an NHiTS model.
type NHiTSConfig struct {
	InputLength  int   // length of the lookback window
	OutputLength int   // forecast horizon
	Channels     int   // number of input channels (variates)
	PoolKernels  []int // downsampling factor per stack (e.g., [2, 4, 8])
	HiddenSize   int   // hidden dimension of MLP layers in each stack
	NumMLPLayers int   // number of hidden MLP layers per stack (default 2)
}

// nhitsStack is a single N-HiTS stack operating at a specific sampling rate.
type nhitsStack struct {
	poolKernel int        // maxpool kernel size for temporal downsampling
	mlpLayers  []mlpLayer // MLP hidden layers
	outputProj mlpLayer   // projects to outputLen coefficients
}

// NHiTS implements the N-HiTS (Neural Hierarchical Interpolation for
// Time Series Forecasting) model from AAAI 2023.
//
// N-HiTS extends N-BEATS by adding multi-rate temporal pooling. Each stack
// first downsamples the input via max-pooling with a different kernel, then
// processes through an MLP, and interpolates back to the forecast horizon.
// The hierarchical design lets different stacks capture patterns at different
// temporal granularities.
type NHiTS struct {
	config    NHiTSConfig
	engine    compute.Engine[float32]
	ops       numeric.Arithmetic[float32]
	stacks    []nhitsStack
	normMeans [][]float64 // per-channel normalization means from training
	normStds  [][]float64 // per-channel normalization stds from training

	// Float64 shadow parameters and gradient accumulator for TrainableBackend (CPU path).
	f64Params []float64 // flat float64 copy of all stack weights/biases
	grads     []float64 // gradient accumulator
}

// NewNHiTS creates a new N-HiTS model with the given configuration.
func NewNHiTS(config NHiTSConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*NHiTS, error) {
	if config.InputLength <= 0 {
		return nil, fmt.Errorf("nhits: InputLength must be positive, got %d", config.InputLength)
	}
	if config.OutputLength <= 0 {
		return nil, fmt.Errorf("nhits: OutputLength must be positive, got %d", config.OutputLength)
	}
	if config.Channels <= 0 {
		return nil, fmt.Errorf("nhits: Channels must be positive, got %d", config.Channels)
	}
	if len(config.PoolKernels) == 0 {
		return nil, fmt.Errorf("nhits: PoolKernels must have at least one element")
	}
	if config.HiddenSize <= 0 {
		return nil, fmt.Errorf("nhits: HiddenSize must be positive, got %d", config.HiddenSize)
	}
	for i, k := range config.PoolKernels {
		if k <= 0 {
			return nil, fmt.Errorf("nhits: PoolKernels[%d] must be positive, got %d", i, k)
		}
		if k > config.InputLength {
			return nil, fmt.Errorf("nhits: PoolKernels[%d]=%d exceeds InputLength=%d", i, k, config.InputLength)
		}
	}

	nMLPLayers := config.NumMLPLayers
	if nMLPLayers <= 0 {
		nMLPLayers = 2
	}

	m := &NHiTS{
		config: config,
		engine: engine,
		ops:    ops,
	}

	m.stacks = make([]nhitsStack, len(config.PoolKernels))
	for i, kernel := range config.PoolKernels {
		s, err := newNHiTSStack(config.InputLength, config.OutputLength, config.Channels, kernel, config.HiddenSize, nMLPLayers)
		if err != nil {
			return nil, fmt.Errorf("nhits: stack %d: %w", i, err)
		}
		m.stacks[i] = s
	}

	return m, nil
}

// pooledLen returns the output length after max-pooling with the given kernel.
// Always returns at least 1 to prevent zero-dimension tensors.
func pooledLen(inputLen, kernel int) int {
	p := inputLen / kernel
	if p == 0 {
		return 1
	}
	return p
}

// newNHiTSStack creates a single N-HiTS stack.
func newNHiTSStack(inputLen, outputLen, channels, kernel, hiddenSize, nMLPLayers int) (nhitsStack, error) {
	s := nhitsStack{poolKernel: kernel}

	pLen := pooledLen(inputLen, kernel)
	flatDim := pLen * channels
	if flatDim <= 0 {
		return nhitsStack{}, fmt.Errorf("invalid flatDim=%d (pooledLen=%d, channels=%d, kernel=%d, inputLen=%d)", flatDim, pLen, channels, kernel, inputLen)
	}

	// Build MLP: flatDim -> hidden -> ... -> hidden.
	layers := make([]mlpLayer, nMLPLayers)
	inDim := flatDim
	for i := 0; i < nMLPLayers; i++ {
		l, err := newMLPLayer(inDim, hiddenSize)
		if err != nil {
			return nhitsStack{}, fmt.Errorf("mlp layer %d: %w", i, err)
		}
		layers[i] = l
		inDim = hiddenSize
	}
	s.mlpLayers = layers

	// Output projection: hidden -> outputLen.
	proj, err := newMLPLayer(hiddenSize, outputLen)
	if err != nil {
		return nhitsStack{}, fmt.Errorf("output proj: %w", err)
	}
	s.outputProj = proj

	return s, nil
}

// maxPool1D performs 1D max-pooling over the last dimension.
// Input shape: [batch, length], output shape: [batch, length/kernel].
// Non-overlapping windows; trailing elements that don't fill a full window are dropped.
func maxPool1D(data []float32, length, kernel int) []float32 {
	batch := len(data) / length
	outLen := length / kernel
	if outLen == 0 {
		outLen = 1
	}
	out := make([]float32, batch*outLen)
	for b := 0; b < batch; b++ {
		for i := 0; i < outLen; i++ {
			start := i * kernel
			maxVal := float32(math.Inf(-1))
			end := start + kernel
			if end > length {
				end = length
			}
			for j := start; j < end; j++ {
				if data[b*length+j] > maxVal {
					maxVal = data[b*length+j]
				}
			}
			out[b*outLen+i] = maxVal
		}
	}
	return out
}

// Forward runs the N-HiTS forward pass.
// Input x has shape [batch, inputLen] (single channel) or [batch, channels * inputLen].
// Returns forecast tensor of shape [batch, outputLen].
func (m *NHiTS) Forward(ctx context.Context, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	expectedCols := m.config.Channels * m.config.InputLength
	if len(shape) != 2 || shape[1] != expectedCols {
		return nil, fmt.Errorf("nhits: expected input shape [batch, %d], got %v", expectedCols, shape)
	}

	batch := shape[0]
	xData := x.Data()

	// Initialize forecast accumulator.
	forecastData := make([]float32, batch*m.config.OutputLength)

	for _, stack := range m.stacks {
		stackOut, err := m.stackForward(ctx, xData, batch, stack)
		if err != nil {
			return nil, err
		}
		for i := range forecastData {
			forecastData[i] += stackOut[i]
		}
	}

	return tensor.New[float32]([]int{batch, m.config.OutputLength}, forecastData)
}

// stackForward processes one stack: pool -> flatten -> MLP -> interpolate.
func (m *NHiTS) stackForward(ctx context.Context, xData []float32, batch int, stack nhitsStack) ([]float32, error) {
	channels := m.config.Channels
	inputLen := m.config.InputLength

	// Per-channel max-pooling then flatten.
	pLen := pooledLen(inputLen, stack.poolKernel)
	flatDim := pLen * channels
	flatData := make([]float32, batch*flatDim)

	for c := 0; c < channels; c++ {
		// Extract channel c for all batches.
		chanData := make([]float32, batch*inputLen)
		for b := 0; b < batch; b++ {
			for t := 0; t < inputLen; t++ {
				chanData[b*inputLen+t] = xData[b*channels*inputLen+c*inputLen+t]
			}
		}

		pooled := maxPool1D(chanData, inputLen, stack.poolKernel)

		// Place pooled channel into flat buffer.
		for b := 0; b < batch; b++ {
			for t := 0; t < pLen; t++ {
				flatData[b*flatDim+c*pLen+t] = pooled[b*pLen+t]
			}
		}
	}

	// MLP forward pass with ReLU.
	h, err := tensor.New[float32]([]int{batch, flatDim}, flatData)
	if err != nil {
		return nil, err
	}

	for _, l := range stack.mlpLayers {
		h, err = m.linearForward(ctx, h, l)
		if err != nil {
			return nil, err
		}
		h, err = m.engine.UnaryOp(ctx, h, m.ops.ReLU)
		if err != nil {
			return nil, err
		}
	}

	// Output projection -> [batch, outputLen].
	h, err = m.linearForward(ctx, h, stack.outputProj)
	if err != nil {
		return nil, err
	}

	return h.Data(), nil
}

// linearForward computes x @ W + b.
func (m *NHiTS) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	if l.weights == nil || l.biases == nil {
		return nil, fmt.Errorf("nhits: nil weight/bias in linear layer")
	}
	wShape := l.weights.Shape()
	if len(wShape) < 2 || wShape[0] == 0 || wShape[1] == 0 {
		return nil, fmt.Errorf("nhits: invalid weight shape %v in linear layer", wShape)
	}
	return functional.Linear(ctx, m.engine, x, l.weights, l.biases)
}


// stackBackwardEngine computes gradients for a single stack using engine tensor
// operations (MatMul, Transpose, Sum) instead of manual triple-nested loops.
// Returns gradient slices in order: [mlp0_dW, mlp0_dB, mlp1_dW, mlp1_dB, ..., proj_dW, proj_dB].
func (n *NHiTS) stackBackwardEngine(ctx context.Context, dH []float32, batchN int, stack nhitsStack, intermediates [][]float32, pooledFlat []float32) ([][]float32, error) {
	nMLP := len(stack.mlpLayers)
	grads := make([][]float32, nMLP*2+2)

	// Output proj backward. Weights are [oDim, hDim] (functional.Linear layout).
	// dW = dH^T @ lastH: [oDim, batchN] @ [batchN, hDim] = [oDim, hDim]
	// dB = sum(dH, axis=0)
	// newDH = dH @ W: [batchN, oDim] @ [oDim, hDim] = [batchN, hDim]
	oDim := n.config.OutputLength
	hDim := stack.outputProj.weights.Shape()[1]

	lastH := intermediates[len(intermediates)-1]
	lastHTensor, err := tensor.New[float32]([]int{batchN, hDim}, lastH)
	if err != nil {
		return nil, err
	}
	dHTensor, err := tensor.New[float32]([]int{batchN, oDim}, dH)
	if err != nil {
		return nil, err
	}

	dHT, err := n.engine.Transpose(ctx, dHTensor, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dWTensor, err := n.engine.MatMul(ctx, dHT, lastHTensor)
	if err != nil {
		return nil, err
	}
	grads[nMLP*2] = make([]float32, len(dWTensor.Data()))
	copy(grads[nMLP*2], dWTensor.Data())

	dBTensor, err := n.engine.Sum(ctx, dHTensor, 0, false)
	if err != nil {
		return nil, err
	}
	grads[nMLP*2+1] = make([]float32, len(dBTensor.Data()))
	copy(grads[nMLP*2+1], dBTensor.Data())

	newDHTensor, err := n.engine.MatMul(ctx, dHTensor, stack.outputProj.weights)
	if err != nil {
		return nil, err
	}
	dHTensor = newDHTensor

	// MLP layers backward (reverse order). Weights are [outDim, inDim].
	for li := nMLP - 1; li >= 0; li-- {
		l := &stack.mlpLayers[li]
		lOutDim := l.weights.Shape()[0]
		lInDim := l.weights.Shape()[1]

		// ReLU gradient: zero out where post-ReLU activation <= 0.
		postReLU := intermediates[li]
		curDH := dHTensor.Data()
		maskedDH := make([]float32, len(curDH))
		copy(maskedDH, curDH)
		for b := 0; b < batchN; b++ {
			for i := 0; i < lOutDim; i++ {
				if postReLU[b*lOutDim+i] <= 0 {
					maskedDH[b*lOutDim+i] = 0
				}
			}
		}
		dHTensor, err = tensor.New[float32]([]int{batchN, lOutDim}, maskedDH)
		if err != nil {
			return nil, err
		}

		// Determine input to this layer.
		var hInput []float32
		if li == 0 {
			hInput = pooledFlat
		} else {
			hInput = intermediates[li-1]
		}
		hInputTensor, err := tensor.New[float32]([]int{batchN, lInDim}, hInput)
		if err != nil {
			return nil, err
		}

		// dW = dH^T @ hInput: [lOutDim, batchN] @ [batchN, lInDim] = [lOutDim, lInDim]
		dHT, err := n.engine.Transpose(ctx, dHTensor, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dWTensor, err := n.engine.MatMul(ctx, dHT, hInputTensor)
		if err != nil {
			return nil, err
		}
		grads[li*2] = make([]float32, len(dWTensor.Data()))
		copy(grads[li*2], dWTensor.Data())

		// dB = sum(dH, axis=0)
		dBTensor, err := n.engine.Sum(ctx, dHTensor, 0, false)
		if err != nil {
			return nil, err
		}
		grads[li*2+1] = make([]float32, len(dBTensor.Data()))
		copy(grads[li*2+1], dBTensor.Data())

		// newDH = dH @ W: [batchN, lOutDim] @ [lOutDim, lInDim] = [batchN, lInDim]
		dHTensor, err = n.engine.MatMul(ctx, dHTensor, l.weights)
		if err != nil {
			return nil, err
		}
	}

	return grads, nil
}

// TrainWindowed trains the N-HiTS model on pre-windowed data using AdamW with
// analytical gradient computation.
//
// windows: [numSamples][channels][inputLen] — input windows.
// labels: [numSamples * outputLen] — flattened target values.
func (n *NHiTS) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	if len(windows) == 0 {
		return nil, fmt.Errorf("nhits: no training windows provided")
	}
	numSamples := len(windows)
	if len(labels) != numSamples*n.config.OutputLength {
		return nil, fmt.Errorf("nhits: labels length %d != numSamples(%d) * outputLen(%d)",
			len(labels), numSamples, n.config.OutputLength)
	}

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, n.normMeans, n.normStds = normalizeWindows(windows)

	if n.engine != nil {
		return n.trainWindowedEngine(windows, labels, config)
	}

	// CPU path: use float64 shadow params and shared TrainLoop.
	n.initF64Params()
	result, err := TrainLoop(n, windows, labels, config)
	if err != nil {
		return nil, err
	}
	n.syncF64ToTensors()
	return result, nil
}

// trainWindowedEngine trains using the float32 engine-accelerated path.
func (n *NHiTS) trainWindowedEngine(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	numSamples := len(windows)

	baseLR := config.LR
	if baseLR == 0 {
		baseLR = 1e-3
	}
	batchSize := config.BatchSize
	if batchSize <= 0 {
		batchSize = numSamples
	}
	epochs := config.Epochs
	if epochs <= 0 {
		epochs = 100
	}

	ctx := context.Background()
	result := &TrainResult{}

	// Collect all parameters in order: per stack, mlp layers (w,b) then output proj (w,b).
	var graphParams []*graph.Parameter[float32]
	for si := range n.stacks {
		stack := &n.stacks[si]
		for li := range stack.mlpLayers {
			wParam, _ := graph.NewParameter(fmt.Sprintf("stack%d.mlp%d.w", si, li), stack.mlpLayers[li].weights, tensor.New[float32])
			bParam, _ := graph.NewParameter(fmt.Sprintf("stack%d.mlp%d.b", si, li), stack.mlpLayers[li].biases, tensor.New[float32])
			graphParams = append(graphParams, wParam, bParam)
		}
		wParam, _ := graph.NewParameter(fmt.Sprintf("stack%d.out.w", si), stack.outputProj.weights, tensor.New[float32])
		bParam, _ := graph.NewParameter(fmt.Sprintf("stack%d.out.b", si), stack.outputProj.biases, tensor.New[float32])
		graphParams = append(graphParams, wParam, bParam)
	}

	// AdamW optimizer from training/optimizer.
	beta1, beta2, eps := float32(0.9), float32(0.999), float32(1e-8)
	wd := float32(1e-4)
	opt := optimizer.NewAdamW(n.engine, float32(baseLR), beta1, beta2, eps, wd)
	if config.GradClip > 0 {
		opt.SetMaxGradNorm(config.GradClip)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < numSamples; start += batchSize {
			end := start + batchSize
			if end > numSamples {
				end = numSamples
			}
			batchN := end - start

			// Build input tensor [batchN, channels * inputLen].
			inputData := make([]float32, batchN*n.config.Channels*n.config.InputLength)
			for i := 0; i < batchN; i++ {
				w := windows[start+i]
				for c := 0; c < n.config.Channels; c++ {
					for t := 0; t < n.config.InputLength; t++ {
						inputData[i*n.config.Channels*n.config.InputLength+c*n.config.InputLength+t] = float32(w[c][t])
					}
				}
			}

			x, err := tensor.New[float32]([]int{batchN, n.config.Channels * n.config.InputLength}, inputData)
			if err != nil {
				return nil, err
			}

			// Forward pass.
			pred, err := n.Forward(ctx, x)
			if err != nil {
				return nil, fmt.Errorf("nhits train: forward: %w", err)
			}

			predData := pred.Data()

			// MSE loss and gradient.
			loss := 0.0
			lossGrad := make([]float32, batchN*n.config.OutputLength)
			for i := 0; i < batchN*n.config.OutputLength; i++ {
				target := float32(labels[(start+i/n.config.OutputLength)*n.config.OutputLength+i%n.config.OutputLength])
				diff := predData[i] - target
				loss += float64(diff * diff)
				lossGrad[i] = 2.0 * diff / float32(batchN*n.config.OutputLength)
			}
			loss /= float64(batchN * n.config.OutputLength)
			epochLoss += loss
			nBatches++

			// Backward pass per stack, collecting all gradients.
			allGrads := make([][]float32, len(graphParams))
			paramIdx := 0

			for si := range n.stacks {
				stack := &n.stacks[si]

				intermediates, pooledFlat, err := n.stackForwardWithIntermediates(ctx, x.Data(), batchN, *stack)
				if err != nil {
					return nil, err
				}

				dH := make([]float32, len(lossGrad))
				copy(dH, lossGrad)

				stackGrads, err := n.stackBackwardEngine(ctx, dH, batchN, *stack, intermediates, pooledFlat)
				if err != nil {
					return nil, fmt.Errorf("nhits train: engine backward stack %d: %w", si, err)
				}
				for i, g := range stackGrads {
					allGrads[paramIdx+i] = g
				}

				paramIdx += len(stack.mlpLayers)*2 + 2
			}

			// Set gradients on graph parameters and apply AdamW step.
			opt.SetLR(float32(scheduler.WarmupLR(baseLR, epoch, config.WarmupEpochs)))
			for i, gp := range graphParams {
				gradT, err := tensor.New[float32](gp.Value.Shape(), allGrads[i])
				if err != nil {
					return nil, fmt.Errorf("nhits train: gradient tensor: %w", err)
				}
				gp.Gradient = gradT
			}
			if err := opt.Step(ctx, graphParams); err != nil {
				return nil, fmt.Errorf("nhits train: adamw step: %w", err)
			}
		}

		avgLoss := epochLoss / float64(nBatches)
		result.LossHistory = append(result.LossHistory, avgLoss)

		// Early halt on NaN/Inf loss.
		if !isFinite(avgLoss) {
			return nil, fmt.Errorf("nhits: training diverged at epoch %d: loss=%v", epoch, avgLoss)
		}
	}

	if len(result.LossHistory) > 0 {
		result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	}
	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// stackForwardWithIntermediates re-runs the stack forward pass, returning
// intermediate activations (post-ReLU) for each MLP layer and the pooled flat input.
func (n *NHiTS) stackForwardWithIntermediates(ctx context.Context, xData []float32, batch int, stack nhitsStack) ([][]float32, []float32, error) {
	channels := n.config.Channels
	inputLen := n.config.InputLength

	pLen := pooledLen(inputLen, stack.poolKernel)
	flatDim := pLen * channels
	flatData := make([]float32, batch*flatDim)

	for c := 0; c < channels; c++ {
		chanData := make([]float32, batch*inputLen)
		for b := 0; b < batch; b++ {
			for t := 0; t < inputLen; t++ {
				chanData[b*inputLen+t] = xData[b*channels*inputLen+c*inputLen+t]
			}
		}
		pooled := maxPool1D(chanData, inputLen, stack.poolKernel)
		for b := 0; b < batch; b++ {
			for t := 0; t < pLen; t++ {
				flatData[b*flatDim+c*pLen+t] = pooled[b*pLen+t]
			}
		}
	}

	h, err := tensor.New[float32]([]int{batch, flatDim}, flatData)
	if err != nil {
		return nil, nil, err
	}

	var intermediates [][]float32
	for _, l := range stack.mlpLayers {
		h, err = n.linearForward(ctx, h, l)
		if err != nil {
			return nil, nil, err
		}
		h, err = n.engine.UnaryOp(ctx, h, n.ops.ReLU)
		if err != nil {
			return nil, nil, err
		}
		// Save post-ReLU activation.
		d := h.Data()
		saved := make([]float32, len(d))
		copy(saved, d)
		intermediates = append(intermediates, saved)
	}

	pooledFlat := make([]float32, len(flatData))
	copy(pooledFlat, flatData)

	return intermediates, pooledFlat, nil
}


// nhitsCache holds activations from a per-sample forward pass needed for backpropagation.
type nhitsCache struct {
	// Per-stack intermediates.
	stackPooled  [][]float64   // [stack] flattened pooled input
	stackPreReLU [][][]float64 // [stack][mlpLayer] pre-ReLU activations
	stackPostAct [][][]float64 // [stack][mlpLayer] post-ReLU activations
}

// nhitsF64ParamCount returns the total number of trainable parameters.
func (n *NHiTS) nhitsF64ParamCount() int {
	count := 0
	for _, stack := range n.stacks {
		for _, l := range stack.mlpLayers {
			s := l.weights.Shape()
			count += s[0] * s[1] // weights
			count += l.biases.Shape()[0] // biases
		}
		s := stack.outputProj.weights.Shape()
		count += s[0] * s[1]
		count += stack.outputProj.biases.Shape()[0]
	}
	return count
}

// initF64Params copies float32 tensor parameters into a flat float64 buffer.
func (n *NHiTS) initF64Params() {
	total := n.nhitsF64ParamCount()
	n.f64Params = make([]float64, total)
	idx := 0
	for _, stack := range n.stacks {
		for _, l := range stack.mlpLayers {
			for _, v := range l.weights.Data() {
				n.f64Params[idx] = float64(v)
				idx++
			}
			for _, v := range l.biases.Data() {
				n.f64Params[idx] = float64(v)
				idx++
			}
		}
		for _, v := range stack.outputProj.weights.Data() {
			n.f64Params[idx] = float64(v)
			idx++
		}
		for _, v := range stack.outputProj.biases.Data() {
			n.f64Params[idx] = float64(v)
			idx++
		}
	}
}

// syncF64ToTensors copies float64 parameters back to float32 tensors.
func (n *NHiTS) syncF64ToTensors() {
	idx := 0
	for si := range n.stacks {
		stack := &n.stacks[si]
		for li := range stack.mlpLayers {
			data := stack.mlpLayers[li].weights.Data()
			for i := range data {
				data[i] = float32(n.f64Params[idx])
				idx++
			}
			data = stack.mlpLayers[li].biases.Data()
			for i := range data {
				data[i] = float32(n.f64Params[idx])
				idx++
			}
		}
		data := stack.outputProj.weights.Data()
		for i := range data {
			data[i] = float32(n.f64Params[idx])
			idx++
		}
		data = stack.outputProj.biases.Data()
		for i := range data {
			data[i] = float32(n.f64Params[idx])
			idx++
		}
	}
}

// nhitsStackF64Forward runs a pure float64 forward pass for one stack on a single sample.
// input: [channels][inputLen], returns [outputLen] and per-layer intermediates.
func (n *NHiTS) nhitsStackF64Forward(input [][]float64, stackIdx int, paramOffset int) ([]float64, []float64, [][]float64, [][]float64) {
	stack := &n.stacks[stackIdx]
	channels := n.config.Channels
	inputLen := n.config.InputLength

	// Max-pool each channel.
	pLen := pooledLen(inputLen, stack.poolKernel)
	pooledFlat := make([]float64, channels*pLen)
	for c := 0; c < channels; c++ {
		for i := 0; i < pLen; i++ {
			start := i * stack.poolKernel
			maxVal := math.Inf(-1)
			end := start + stack.poolKernel
			if end > inputLen {
				end = inputLen
			}
			for j := start; j < end; j++ {
				if input[c][j] > maxVal {
					maxVal = input[c][j]
				}
			}
			pooledFlat[c*pLen+i] = maxVal
		}
	}

	// MLP layers with ReLU. Weights are stored [outDim, inDim].
	h := pooledFlat
	var preReLU, postAct [][]float64
	off := paramOffset

	for _, l := range stack.mlpLayers {
		wShape := l.weights.Shape()
		outDim, inDim := wShape[0], wShape[1]
		bDim := l.biases.Shape()[0]

		// Linear: y = x @ W^T + b, W is [outDim, inDim]
		out := make([]float64, outDim)
		for j := 0; j < outDim; j++ {
			out[j] = n.f64Params[off+outDim*inDim+j] // bias
		}
		for j := 0; j < outDim; j++ {
			for i := 0; i < inDim; i++ {
				out[j] += h[i] * n.f64Params[off+j*inDim+i]
			}
		}
		preReLU = append(preReLU, append([]float64(nil), out...))

		// ReLU
		for i := range out {
			if out[i] < 0 {
				out[i] = 0
			}
		}
		postAct = append(postAct, append([]float64(nil), out...))
		h = out
		off += outDim*inDim + bDim
	}

	// Output projection (no activation). Weights are [outDim, inDim].
	projShape := stack.outputProj.weights.Shape()
	outDim, inDim := projShape[0], projShape[1]
	result := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		result[j] = n.f64Params[off+outDim*inDim+j] // bias
	}
	for j := 0; j < outDim; j++ {
		for i := 0; i < inDim; i++ {
			result[j] += h[i] * n.f64Params[off+j*inDim+i]
		}
	}

	return result, pooledFlat, preReLU, postAct
}

// nhitsStackParamCount returns the number of parameters in a stack.
func (n *NHiTS) nhitsStackParamCount(stackIdx int) int {
	stack := &n.stacks[stackIdx]
	count := 0
	for _, l := range stack.mlpLayers {
		s := l.weights.Shape()
		count += s[0]*s[1] + l.biases.Shape()[0]
	}
	s := stack.outputProj.weights.Shape()
	count += s[0]*s[1] + stack.outputProj.biases.Shape()[0]
	return count
}

// ForwardSample runs the N-HiTS forward pass on a single sample using float64 params.
// Input: [channels][inputLen], returns (flatOutput [outputLen], cache, error).
func (n *NHiTS) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	output := make([]float64, n.config.OutputLength)
	cache := &nhitsCache{
		stackPooled:  make([][]float64, len(n.stacks)),
		stackPreReLU: make([][][]float64, len(n.stacks)),
		stackPostAct: make([][][]float64, len(n.stacks)),
	}

	paramOff := 0
	for si := range n.stacks {
		stackOut, pooled, preReLU, postAct := n.nhitsStackF64Forward(input, si, paramOff)
		for i := range output {
			output[i] += stackOut[i]
		}
		cache.stackPooled[si] = pooled
		cache.stackPreReLU[si] = preReLU
		cache.stackPostAct[si] = postAct
		paramOff += n.nhitsStackParamCount(si)
	}

	return output, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (n *NHiTS) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*nhitsCache)
	if !ok {
		return fmt.Errorf("nhits: invalid cache type")
	}

	if n.grads == nil {
		n.grads = make([]float64, n.nhitsF64ParamCount())
	}

	paramOff := 0
	for si := range n.stacks {
		stack := &n.stacks[si]

		// dH starts as dOutput (each stack's output is summed).
		dH := append([]float64(nil), dOutput...)
		off := paramOff

		// Compute parameter offsets for each layer in this stack.
		layerOffsets := make([]int, len(stack.mlpLayers)+1)
		tmpOff := paramOff
		for li, l := range stack.mlpLayers {
			layerOffsets[li] = tmpOff
			s := l.weights.Shape()
			tmpOff += s[0]*s[1] + l.biases.Shape()[0]
		}
		layerOffsets[len(stack.mlpLayers)] = tmpOff // output proj offset

		// Backward through output projection. Weights are [projOut, projIn].
		projOff := layerOffsets[len(stack.mlpLayers)]
		projShape := stack.outputProj.weights.Shape()
		projOut, projIn := projShape[0], projShape[1]

		// Last hidden layer activation (input to output proj).
		var lastH []float64
		if len(cache.stackPostAct[si]) > 0 {
			lastH = cache.stackPostAct[si][len(cache.stackPostAct[si])-1]
		} else {
			lastH = cache.stackPooled[si]
		}

		// dW_proj [projOut, projIn], dB_proj, dH_new
		newDH := make([]float64, projIn)
		for j := 0; j < projOut; j++ {
			for i := 0; i < projIn; i++ {
				n.grads[projOff+j*projIn+i] += lastH[i] * dH[j]
				newDH[i] += dH[j] * n.f64Params[projOff+j*projIn+i]
			}
		}
		for j := 0; j < projOut; j++ {
			n.grads[projOff+projOut*projIn+j] += dH[j]
		}
		dH = newDH

		// Backward through MLP layers in reverse. Weights are [lOut, lIn].
		for li := len(stack.mlpLayers) - 1; li >= 0; li-- {
			l := &stack.mlpLayers[li]
			lOff := layerOffsets[li]
			lShape := l.weights.Shape()
			lOut, lIn := lShape[0], lShape[1]

			// ReLU backward: mask where pre-ReLU <= 0.
			preReLU := cache.stackPreReLU[si][li]
			for i := range dH {
				if preReLU[i] <= 0 {
					dH[i] = 0
				}
			}

			// Input to this layer.
			var hInput []float64
			if li == 0 {
				hInput = cache.stackPooled[si]
			} else {
				hInput = cache.stackPostAct[si][li-1]
			}

			// dW [lOut, lIn], dB, dH_new
			newDH2 := make([]float64, lIn)
			for j := 0; j < lOut; j++ {
				for i := 0; i < lIn; i++ {
					n.grads[lOff+j*lIn+i] += hInput[i] * dH[j]
					newDH2[i] += dH[j] * n.f64Params[lOff+j*lIn+i]
				}
			}
			for j := 0; j < lOut; j++ {
				n.grads[lOff+lOut*lIn+j] += dH[j]
			}
			dH = newDH2
		}

		off = paramOff
		_ = off
		paramOff += n.nhitsStackParamCount(si)
	}

	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (n *NHiTS) FlatGrads() []float64 {
	if n.grads == nil {
		n.grads = make([]float64, n.nhitsF64ParamCount())
	}
	return n.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (n *NHiTS) ZeroGrads() {
	if n.grads == nil {
		n.grads = make([]float64, n.nhitsF64ParamCount())
		return
	}
	for i := range n.grads {
		n.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters.
func (n *NHiTS) FlatParams() []*float64 {
	if n.f64Params == nil {
		n.initF64Params()
	}
	ptrs := make([]*float64, len(n.f64Params))
	for i := range n.f64Params {
		ptrs[i] = &n.f64Params[i]
	}
	return ptrs
}

// ParamCount returns the total number of trainable parameters.
func (n *NHiTS) ParamCount() int {
	return n.nhitsF64ParamCount()
}

// Compile-time check that NHiTS implements TrainableBackend.
var _ TrainableBackend = (*NHiTS)(nil)

// nhitsModelFile is the JSON structure for saving/loading N-HiTS models.
type nhitsModelFile struct {
	Config    NHiTSConfig      `json:"config"`
	Stacks    []nhitsStackFile `json:"stacks"`
	NormMeans [][]float64      `json:"norm_means,omitempty"`
	NormStds  [][]float64      `json:"norm_stds,omitempty"`
}

type nhitsStackFile struct {
	PoolKernel int            `json:"pool_kernel"`
	MLPWeights [][]float64    `json:"mlp_weights"`
	MLPBiases  [][]float64    `json:"mlp_biases"`
	ProjWeight []float64      `json:"proj_weight"`
	ProjBias   []float64      `json:"proj_bias"`
}

// Save writes the N-HiTS model to a JSON file.
func (n *NHiTS) Save(path string) error {
	mf := nhitsModelFile{Config: n.config, NormMeans: n.normMeans, NormStds: n.normStds}
	for _, stack := range n.stacks {
		sf := nhitsStackFile{PoolKernel: stack.poolKernel}
		for _, l := range stack.mlpLayers {
			sf.MLPWeights = append(sf.MLPWeights, float32ToFloat64(l.weights.Data()))
			sf.MLPBiases = append(sf.MLPBiases, float32ToFloat64(l.biases.Data()))
		}
		sf.ProjWeight = float32ToFloat64(stack.outputProj.weights.Data())
		sf.ProjBias = float32ToFloat64(stack.outputProj.biases.Data())
		mf.Stacks = append(mf.Stacks, sf)
	}
	data, err := json.Marshal(mf)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// PredictWindowed loads a model from disk and runs inference.
// windows: [numSamples][channels][inputLen].
// Returns flattened predictions [numSamples * outputLen].
func (n *NHiTS) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := n.Load(modelPath); err != nil {
			return nil, fmt.Errorf("nhits predict: load: %w", err)
		}
	}

	if len(windows) == 0 {
		return nil, fmt.Errorf("nhits predict: no windows provided")
	}

	// Apply normalization from training if available.
	if n.normMeans != nil {
		windows = applyNormalization(windows, n.normMeans, n.normStds)
	}

	numSamples := len(windows)
	inputData := make([]float32, numSamples*n.config.Channels*n.config.InputLength)
	for i, w := range windows {
		for c := 0; c < n.config.Channels; c++ {
			for t := 0; t < n.config.InputLength; t++ {
				inputData[i*n.config.Channels*n.config.InputLength+c*n.config.InputLength+t] = float32(w[c][t])
			}
		}
	}

	x, err := tensor.New[float32]([]int{numSamples, n.config.Channels * n.config.InputLength}, inputData)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	pred, err := n.Forward(ctx, x)
	if err != nil {
		return nil, err
	}

	predData := pred.Data()
	result := make([]float64, len(predData))
	for i, v := range predData {
		result[i] = float64(v)
	}
	return result, nil
}

// Load reads model weights from a JSON file.
func (n *NHiTS) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var mf nhitsModelFile
	if err := json.Unmarshal(data, &mf); err != nil {
		return err
	}

	if len(mf.Stacks) != len(n.stacks) {
		return fmt.Errorf("nhits load: stack count mismatch: file=%d, model=%d", len(mf.Stacks), len(n.stacks))
	}

	for si, sf := range mf.Stacks {
		stack := &n.stacks[si]
		if len(sf.MLPWeights) != len(stack.mlpLayers) {
			return fmt.Errorf("nhits load: stack %d mlp layer count mismatch", si)
		}
		for li := range stack.mlpLayers {
			copy(stack.mlpLayers[li].weights.Data(), float64ToFloat32(sf.MLPWeights[li]))
			copy(stack.mlpLayers[li].biases.Data(), float64ToFloat32(sf.MLPBiases[li]))
		}
		copy(stack.outputProj.weights.Data(), float64ToFloat32(sf.ProjWeight))
		copy(stack.outputProj.biases.Data(), float64ToFloat32(sf.ProjBias))
	}

	n.normMeans = mf.NormMeans
	n.normStds = mf.NormStds
	return nil
}

func float32ToFloat64(data []float32) []float64 {
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = float64(v)
	}
	return out
}

func float64ToFloat32(data []float64) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = float32(v)
	}
	return out
}

// initWeightsSmall reinitializes all weights with a small scale for stable training.
func (n *NHiTS) initWeightsSmall() {
	for si := range n.stacks {
		stack := &n.stacks[si]
		for li := range stack.mlpLayers {
			initSmall(stack.mlpLayers[li].weights.Data())
			clearSlice(stack.mlpLayers[li].biases.Data())
		}
		initSmall(stack.outputProj.weights.Data())
		clearSlice(stack.outputProj.biases.Data())
	}
}

func initSmall(data []float32) {
	scale := float32(0.01)
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * scale
	}
}

func clearSlice(data []float32) {
	for i := range data {
		data[i] = 0
	}
}
