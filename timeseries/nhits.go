package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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
	config NHiTSConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]
	stacks []nhitsStack
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
	out, err := m.engine.MatMul(ctx, x, l.weights)
	if err != nil {
		return nil, err
	}
	return m.engine.Add(ctx, out, l.biases)
}

// adamState holds first and second moment estimates for AdamW.
type adamState struct {
	m []float32
	v []float32
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

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, _, _ = normalizeWindows(windows)

	ctx := context.Background()
	result := &TrainResult{}

	// Collect all parameters in order: per stack, mlp layers (w,b) then output proj (w,b).
	type paramRef struct {
		data []float32
	}
	var allParams []paramRef
	for si := range n.stacks {
		stack := &n.stacks[si]
		for li := range stack.mlpLayers {
			allParams = append(allParams,
				paramRef{stack.mlpLayers[li].weights.Data()},
				paramRef{stack.mlpLayers[li].biases.Data()},
			)
		}
		allParams = append(allParams,
			paramRef{stack.outputProj.weights.Data()},
			paramRef{stack.outputProj.biases.Data()},
		)
	}

	// AdamW state per parameter.
	paramStates := make([]adamState, len(allParams))
	for i, p := range allParams {
		paramStates[i] = adamState{m: make([]float32, len(p.data)), v: make([]float32, len(p.data))}
	}

	beta1, beta2, eps := float32(0.9), float32(0.999), float32(1e-8)
	wd := float32(1e-4)

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
			allGrads := make([][]float32, len(allParams))
			paramIdx := 0

			for si := range n.stacks {
				stack := &n.stacks[si]

				intermediates, pooledFlat, err := n.stackForwardWithIntermediates(ctx, x.Data(), batchN, *stack)
				if err != nil {
					return nil, err
				}

				// Backward from loss through output proj then MLP layers.
				dH := make([]float32, len(lossGrad))
				copy(dH, lossGrad)

				// Output proj backward: h is intermediates[last], output is [batchN, outputLen].
				{
					lastH := intermediates[len(intermediates)-1]
					wData := stack.outputProj.weights.Data()
					oDim := n.config.OutputLength
					hDim := stack.outputProj.weights.Shape()[0]

					dW := make([]float32, hDim*oDim)
					dB := make([]float32, oDim)
					for b := 0; b < batchN; b++ {
						for i := 0; i < hDim; i++ {
							for j := 0; j < oDim; j++ {
								dW[i*oDim+j] += lastH[b*hDim+i] * dH[b*oDim+j]
							}
						}
						for j := 0; j < oDim; j++ {
							dB[j] += dH[b*oDim+j]
						}
					}

					newDH := make([]float32, batchN*hDim)
					for b := 0; b < batchN; b++ {
						for i := 0; i < hDim; i++ {
							for j := 0; j < oDim; j++ {
								newDH[b*hDim+i] += dH[b*oDim+j] * wData[i*oDim+j]
							}
						}
					}
					dH = newDH

					// Store grads at correct indices (proj is after MLP layers).
					nMLP := len(stack.mlpLayers)
					allGrads[paramIdx+nMLP*2] = dW
					allGrads[paramIdx+nMLP*2+1] = dB
				}

				// MLP layers backward (reverse order).
				for li := len(stack.mlpLayers) - 1; li >= 0; li-- {
					l := &stack.mlpLayers[li]
					lWData := l.weights.Data()
					lInDim := l.weights.Shape()[0]
					lOutDim := l.weights.Shape()[1]

					// ReLU gradient.
					postReLU := intermediates[li]
					for b := 0; b < batchN; b++ {
						for i := 0; i < lOutDim; i++ {
							if postReLU[b*lOutDim+i] <= 0 {
								dH[b*lOutDim+i] = 0
							}
						}
					}

					var hInput []float32
					if li == 0 {
						hInput = pooledFlat
					} else {
						hInput = intermediates[li-1]
					}

					dW := make([]float32, lInDim*lOutDim)
					dB := make([]float32, lOutDim)
					for b := 0; b < batchN; b++ {
						for i := 0; i < lInDim; i++ {
							for j := 0; j < lOutDim; j++ {
								dW[i*lOutDim+j] += hInput[b*lInDim+i] * dH[b*lOutDim+j]
							}
						}
						for j := 0; j < lOutDim; j++ {
							dB[j] += dH[b*lOutDim+j]
						}
					}

					newDH := make([]float32, batchN*lInDim)
					for b := 0; b < batchN; b++ {
						for i := 0; i < lInDim; i++ {
							for j := 0; j < lOutDim; j++ {
								newDH[b*lInDim+i] += dH[b*lOutDim+j] * lWData[i*lOutDim+j]
							}
						}
					}
					dH = newDH

					allGrads[paramIdx+li*2] = dW
					allGrads[paramIdx+li*2+1] = dB
				}

				// Advance paramIdx past this stack's params.
				paramIdx += len(stack.mlpLayers)*2 + 2
			}

			// Apply all gradients with AdamW (with LR warmup).
			lr := float32(warmupLR(baseLR, epoch, config.WarmupEpochs))
			for i := range allParams {
				n.clipGradients(allGrads[i], config.GradClip)
				n.adamUpdate(allParams[i].data, allGrads[i], &paramStates[i], beta1, beta2, eps, lr, wd, epoch+1)
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

// clipGradients clips gradient vector by L2 norm.
func (n *NHiTS) clipGradients(grad []float32, maxNorm float64) {
	if maxNorm <= 0 {
		return
	}
	var norm float64
	for _, g := range grad {
		norm += float64(g) * float64(g)
	}
	norm = math.Sqrt(norm)
	if norm > maxNorm {
		scale := float32(maxNorm / norm)
		for i := range grad {
			grad[i] *= scale
		}
	}
}

// adamUpdate applies one AdamW step in-place.
func (n *NHiTS) adamUpdate(params, grads []float32, state *adamState, beta1, beta2, eps, lr, wd float32, t int) {
	bc1 := float32(1.0) - float32(math.Pow(float64(beta1), float64(t)))
	bc2 := float32(1.0) - float32(math.Pow(float64(beta2), float64(t)))

	for i := range params {
		state.m[i] = beta1*state.m[i] + (1-beta1)*grads[i]
		state.v[i] = beta2*state.v[i] + (1-beta2)*grads[i]*grads[i]
		mHat := state.m[i] / bc1
		vHat := state.v[i] / bc2
		params[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*params[i])
	}
}

// nhitsModelFile is the JSON structure for saving/loading N-HiTS models.
type nhitsModelFile struct {
	Config NHiTSConfig      `json:"config"`
	Stacks []nhitsStackFile `json:"stacks"`
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
	mf := nhitsModelFile{Config: n.config}
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
