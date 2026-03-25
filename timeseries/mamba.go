package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/ssm"
)

// scaleParams re-centers and scales parameter values in-place.
// Maps from [0,1) uniform to zero-centered [-scale, +scale) range.
func scaleParams(params []*graph.Parameter[float32], scale float32) {
	for _, p := range params {
		data := p.Value.Data()
		for i := range data {
			data[i] = (data[i] - 0.5) * scale * 2
		}
	}
}

// MambaConfig holds the configuration for a Mamba model.
type MambaConfig struct {
	Channels     int // number of input channels
	InputLen     int // lookback window
	OutputLen    int // forecast horizon
	DModel       int // model dimension
	DState       int // SSM state dimension (typically 16)
	DConv        int // convolution kernel size (typically 4)
	ExpandFactor int // expansion factor (typically 2)
	NLayers      int // number of Mamba blocks
}

// Mamba implements the Mamba selective state space model (NeurIPS 2023)
// for time-series forecasting. It wraps layers/ssm.MambaBlock[float32].
type Mamba struct {
	config MambaConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]

	// Input embedding: [channels -> dModel] applied per timestep.
	embed *core.Linear[float32]

	// Stacked Mamba blocks.
	blocks []*ssm.MambaBlock[float32]

	// Output head: [dModel -> channels*outputLen] applied on last timestep.
	head *core.Linear[float32]

	normMeans [][]float64
	normStds  [][]float64
}

// NewMamba creates a new Mamba model with the given configuration.
func NewMamba(config MambaConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Mamba, error) {
	if config.Channels <= 0 {
		return nil, fmt.Errorf("mamba: Channels must be positive, got %d", config.Channels)
	}
	if config.InputLen <= 0 {
		return nil, fmt.Errorf("mamba: InputLen must be positive, got %d", config.InputLen)
	}
	if config.OutputLen <= 0 {
		return nil, fmt.Errorf("mamba: OutputLen must be positive, got %d", config.OutputLen)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("mamba: DModel must be positive, got %d", config.DModel)
	}
	if config.DState <= 0 {
		return nil, fmt.Errorf("mamba: DState must be positive, got %d", config.DState)
	}
	if config.DConv <= 0 {
		return nil, fmt.Errorf("mamba: DConv must be positive, got %d", config.DConv)
	}
	if config.ExpandFactor <= 0 {
		return nil, fmt.Errorf("mamba: ExpandFactor must be positive, got %d", config.ExpandFactor)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("mamba: NLayers must be positive, got %d", config.NLayers)
	}

	dInner := config.ExpandFactor * config.DModel
	dtRank := config.DModel / 16
	if dtRank < 1 {
		dtRank = 1
	}

	m := &Mamba{
		config: config,
		engine: engine,
		ops:    ops,
	}

	// Input embedding: channels -> dModel.
	// Scale weights by 1/sqrt(channels) to keep activations normalized.
	embed, err := core.NewLinear[float32]("mamba_embed", engine, ops, config.Channels, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("mamba: creating embed: %w", err)
	}
	scaleParams(embed.Parameters(), float32(1.0/math.Sqrt(float64(config.Channels))))
	m.embed = embed

	// Mamba blocks.
	// Scale each block's output projection by 1/sqrt(NLayers) to prevent
	// residual accumulation from causing activation explosion. This follows
	// the GPT-2 / Mamba convention for deep residual networks.
	resScale := 1.0 / math.Sqrt(float64(config.NLayers))
	m.blocks = make([]*ssm.MambaBlock[float32], config.NLayers)
	for l := 0; l < config.NLayers; l++ {
		blk, err := ssm.NewMambaBlock[float32](
			fmt.Sprintf("mamba_block_%d", l),
			engine, ops,
			config.DModel, dInner, config.DState, dtRank, config.DConv,
		)
		if err != nil {
			return nil, fmt.Errorf("mamba: creating block %d: %w", l, err)
		}
		// Scale the output projection by residual scale.
		blk.ScaleOutputProj(resScale)
		m.blocks[l] = blk
	}

	// Output head: dModel -> channels*outputLen.
	// Scale weights by 1/sqrt(dModel).
	outDim := config.Channels * config.OutputLen
	head, err := core.NewLinear[float32]("mamba_head", engine, ops, config.DModel, outDim)
	if err != nil {
		return nil, fmt.Errorf("mamba: creating head: %w", err)
	}
	scaleParams(head.Parameters(), float32(1.0/math.Sqrt(float64(config.DModel))))
	m.head = head

	return m, nil
}

// allParams returns all graph parameters in order: embed, blocks, head.
func (m *Mamba) allParams() []*graph.Parameter[float32] {
	var params []*graph.Parameter[float32]
	params = append(params, m.embed.Parameters()...)
	for _, blk := range m.blocks {
		params = append(params, blk.Parameters()...)
	}
	params = append(params, m.head.Parameters()...)
	return params
}

// copyMambaParams copies parameter values from src to dst.
// Both models must have the same config (same number and size of parameters).
func copyMambaParams(src, dst *Mamba) {
	srcParams := src.allParams()
	dstParams := dst.allParams()
	for i := range srcParams {
		copy(dstParams[i].Value.Data(), srcParams[i].Value.Data())
	}
}

// zeroGrads zeroes all parameter gradients.
func (m *Mamba) zeroGrads() {
	for _, p := range m.allParams() {
		shape := p.Value.Shape()
		n := 1
		for _, s := range shape {
			n *= s
		}
		zeroData := make([]float32, n)
		p.Gradient, _ = tensor.New[float32](shape, zeroData)
	}
}

// forward runs the full forward pass.
// Input: [channels][inputLen], returns [channels][outputLen] and intermediate tensors for backward.
func (m *Mamba) forward(ctx context.Context, input [][]float64) ([]float32, *mambaCache, error) {
	channels := m.config.Channels
	seqLen := m.config.InputLen

	// Build input tensor [1, seqLen, channels] (batch=1).
	inData := make([]float32, seqLen*channels)
	for t := 0; t < seqLen; t++ {
		for c := 0; c < channels; c++ {
			inData[t*channels+c] = float32(input[c][t])
		}
	}
	inTensor, err := tensor.New[float32]([]int{1, seqLen, channels}, inData)
	if err != nil {
		return nil, nil, err
	}

	cache := &mambaCache{input: inTensor}

	// Embed: [1, seqLen, channels] -> [1, seqLen, dModel].
	embedded, err := m.embed.Forward(ctx, inTensor)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba: embed forward: %w", err)
	}
	cache.embedded = embedded

	// Process through Mamba blocks.
	x := embedded
	cache.blockInputs = make([]*tensor.TensorNumeric[float32], m.config.NLayers)
	cache.blockOutputs = make([]*tensor.TensorNumeric[float32], m.config.NLayers)
	for l := 0; l < m.config.NLayers; l++ {
		cache.blockInputs[l] = x
		out, err := m.blocks[l].Forward(ctx, x)
		if err != nil {
			return nil, nil, fmt.Errorf("mamba: block %d forward: %w", l, err)
		}
		// Residual connection: output = input + block(input).
		x, err = m.engine.Add(ctx, x, out)
		if err != nil {
			return nil, nil, fmt.Errorf("mamba: block %d residual: %w", l, err)
		}
		cache.blockOutputs[l] = x
	}

	// Extract last timestep: [1, seqLen, dModel] -> [1, dModel].
	xData := x.Data()
	dModel := m.config.DModel
	lastData := make([]float32, dModel)
	copy(lastData, xData[(seqLen-1)*dModel:seqLen*dModel])
	lastTensor, err := tensor.New[float32]([]int{1, dModel}, lastData)
	if err != nil {
		return nil, nil, err
	}
	cache.lastHidden = lastTensor

	// Head: [1, dModel] -> [1, channels*outputLen].
	headOut, err := m.head.Forward(ctx, lastTensor)
	if err != nil {
		return nil, nil, fmt.Errorf("mamba: head forward: %w", err)
	}
	cache.headOut = headOut

	return headOut.Data(), cache, nil
}

// mambaCache holds intermediate tensors for backward pass.
type mambaCache struct {
	input        *tensor.TensorNumeric[float32]
	embedded     *tensor.TensorNumeric[float32]
	blockInputs  []*tensor.TensorNumeric[float32]
	blockOutputs []*tensor.TensorNumeric[float32]
	lastHidden   *tensor.TensorNumeric[float32]
	headOut      *tensor.TensorNumeric[float32]
}

// backward computes gradients through the full model given output gradient.
func (m *Mamba) backward(ctx context.Context, dOut []float32, cache *mambaCache) error {
	dModel := m.config.DModel
	seqLen := m.config.InputLen

	// Head backward: dOut [1, outDim] -> dLastHidden [1, dModel].
	dOutTensor, err := tensor.New[float32]([]int{1, len(dOut)}, dOut)
	if err != nil {
		return err
	}
	dLastHidden, err := m.head.Backward(ctx, types.FullBackprop, dOutTensor, cache.lastHidden)
	if err != nil {
		return fmt.Errorf("mamba: head backward: %w", err)
	}

	// Scatter dLastHidden into [1, seqLen, dModel] gradient (only last timestep is nonzero).
	dSeqData := make([]float32, seqLen*dModel)
	copy(dSeqData[(seqLen-1)*dModel:], dLastHidden[0].Data())
	dSeq, err := tensor.New[float32]([]int{1, seqLen, dModel}, dSeqData)
	if err != nil {
		return err
	}

	// Backward through Mamba blocks (reverse order).
	for l := m.config.NLayers - 1; l >= 0; l-- {
		// Residual: dSeq flows both to block backward and through to the next layer.
		dBlock, err := m.blocks[l].Backward(ctx, types.FullBackprop, dSeq, cache.blockInputs[l])
		if err != nil {
			return fmt.Errorf("mamba: block %d backward: %w", l, err)
		}
		// Residual connection gradient: dInput = dSeq + dBlock[0].
		dSeq, err = m.engine.Add(ctx, dSeq, dBlock[0])
		if err != nil {
			return fmt.Errorf("mamba: block %d residual backward: %w", l, err)
		}
	}

	// Embed backward.
	_, err = m.embed.Backward(ctx, types.FullBackprop, dSeq, cache.input)
	if err != nil {
		return fmt.Errorf("mamba: embed backward: %w", err)
	}

	return nil
}

// TrainWindowed trains the Mamba model on windowed data using AdamW.
func (m *Mamba) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	// CPU fallback: if no engine is set, create a temporary Mamba with a
	// CPUEngine, train it, then copy learned weights back. This avoids side
	// effects on the original struct's engine/layers while still allowing
	// training without a GPU.
	if m.engine == nil {
		cpuEngine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
		cpuOps := numeric.Float32Ops{}
		tmp, err := NewMamba(m.config, cpuEngine, cpuOps)
		if err != nil {
			return nil, fmt.Errorf("mamba: creating CPU fallback model: %w", err)
		}
		// Copy current weights into the temporary model.
		copyMambaParams(m, tmp)
		result, err := tmp.TrainWindowed(windows, labels, config)
		if err != nil {
			return nil, err
		}
		// Copy trained weights back to the original model.
		copyMambaParams(tmp, m)
		m.normMeans = tmp.normMeans
		m.normStds = tmp.normStds
		return result, nil
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("mamba: empty training set")
	}

	expectedLabels := nSamples * m.config.Channels * m.config.OutputLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("mamba: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("mamba: window %d has %d channels, expected %d", i, len(w), m.config.Channels)
		}
		for c, ch := range w {
			if len(ch) != m.config.InputLen {
				return nil, fmt.Errorf("mamba: window %d channel %d has length %d, expected %d", i, c, len(ch), m.config.InputLen)
			}
		}
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

	// Z-score normalize inputs.
	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	ctx := context.Background()

	// AdamW state per parameter.
	params := m.allParams()
	type paramAdamState struct {
		m []float32
		v []float32
	}
	states := make([]paramAdamState, len(params))
	for i, p := range params {
		n := len(p.Value.Data())
		states[i] = paramAdamState{m: make([]float32, n), v: make([]float32, n)}
	}

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	outDim := m.config.Channels * m.config.OutputLen

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			bs := end - start

			m.zeroGrads()

			batchLoss := 0.0

			for s := 0; s < bs; s++ {
				predData, cache, err := m.forward(ctx, windows[start+s])
				if err != nil {
					return nil, fmt.Errorf("mamba: forward sample %d: %w", start+s, err)
				}

				// MSE loss and gradient.
				dOut := make([]float32, outDim)
				sampleLabels := labels[(start+s)*outDim : (start+s+1)*outDim]
				for i := 0; i < outDim; i++ {
					diff := predData[i] - float32(sampleLabels[i])
					batchLoss += float64(diff * diff)
					dOut[i] = 2.0 * diff / float32(bs*outDim)
				}

				if err := m.backward(ctx, dOut, cache); err != nil {
					return nil, fmt.Errorf("mamba: backward sample %d: %w", start+s, err)
				}
			}

			batchLoss /= float64(bs * outDim)
			epochLoss += batchLoss
			nBatches++

			// AdamW update with LR warmup.
			lr := float32(warmupLR(config.LR, epoch, config.WarmupEpochs))
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			beta1 := float32(config.Beta1)
			beta2 := float32(config.Beta2)
			eps := float32(config.Epsilon)
			wd := float32(config.WeightDecay)

			for i, p := range params {
				grad := p.Gradient
				if grad == nil {
					continue
				}
				gData := grad.Data()
				pData := p.Value.Data()

				// Gradient clipping.
				if config.GradClip > 0 {
					norm := float64(0)
					for _, g := range gData {
						norm += float64(g) * float64(g)
					}
					norm = math.Sqrt(norm)
					if norm > config.GradClip {
						scale := float32(config.GradClip / norm)
						for j := range gData {
							gData[j] *= scale
						}
					}
				}

				bc1 := float32(1.0 - math.Pow(float64(beta1), t))
				bc2 := float32(1.0 - math.Pow(float64(beta2), t))
				for j := range pData {
					g := gData[j]
					if !isFinite(float64(g)) {
						g = 0
					}
					states[i].m[j] = beta1*states[i].m[j] + (1-beta1)*g
					states[i].v[j] = beta2*states[i].v[j] + (1-beta2)*g*g
					mHat := states[i].m[j] / bc1
					vHat := states[i].v[j] / bc2
					pData[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*pData[j])
				}
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("mamba: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * channels * outputLen.
func (m *Mamba) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := m.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("mamba: load weights: %w", err)
		}
	}

	// CPU fallback: if no engine is set, create a temporary Mamba with a
	// CPUEngine, copy weights, run inference, and return. This mirrors the
	// TrainWindowed fallback and fixes issue #158 where PredictWindowed
	// would panic with nil pointer dereference on a nil-engine model.
	if m.engine == nil {
		cpuEngine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
		cpuOps := numeric.Float32Ops{}
		tmp, err := NewMamba(m.config, cpuEngine, cpuOps)
		if err != nil {
			return nil, fmt.Errorf("mamba: creating CPU fallback model: %w", err)
		}
		copyMambaParams(m, tmp)
		tmp.normMeans = m.normMeans
		tmp.normStds = m.normStds
		return tmp.PredictWindowed(modelPath, windows)
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("mamba: empty input")
	}

	if m.normMeans != nil {
		windows = applyNormalization(windows, m.normMeans, m.normStds)
	}

	ctx := context.Background()
	outDim := m.config.Channels * m.config.OutputLen
	out := make([]float64, 0, nSamples*outDim)

	for _, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("mamba: expected %d channels, got %d", m.config.Channels, len(w))
		}
		predData, _, err := m.forward(ctx, w)
		if err != nil {
			return nil, fmt.Errorf("mamba: forward: %w", err)
		}
		for _, v := range predData {
			out = append(out, float64(v))
		}
	}
	return out, nil
}

// mambaParamFile stores a single parameter's data for serialization.
type mambaParamFile struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}

// mambaWeights is the JSON-serializable form of Mamba parameters.
type mambaWeights struct {
	Config    MambaConfig      `json:"config"`
	Params    []mambaParamFile `json:"params"`
	NormMeans [][]float64      `json:"norm_means,omitempty"`
	NormStds  [][]float64      `json:"norm_stds,omitempty"`
}

// SaveWeights writes the model weights to a JSON file.
func (m *Mamba) SaveWeights(path string) error {
	w := mambaWeights{
		Config:    m.config,
		NormMeans: m.normMeans,
		NormStds:  m.normStds,
	}

	for _, p := range m.allParams() {
		pf := mambaParamFile{
			Name:  p.Name,
			Shape: p.Value.Shape(),
			Data:  float32ToFloat64(p.Value.Data()),
		}
		w.Params = append(w.Params, pf)
	}

	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("mamba: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (m *Mamba) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w mambaWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != m.config {
		return fmt.Errorf("mamba: config mismatch: file has %+v, model has %+v", w.Config, m.config)
	}

	params := m.allParams()
	if len(w.Params) != len(params) {
		return fmt.Errorf("mamba: param count mismatch: file has %d, model has %d", len(w.Params), len(params))
	}

	for i, pf := range w.Params {
		pData := params[i].Value.Data()
		fData := float64ToFloat32(pf.Data)
		if len(pData) != len(fData) {
			return fmt.Errorf("mamba: param %q size mismatch: file has %d, model has %d", pf.Name, len(fData), len(pData))
		}
		copy(pData, fData)
	}

	m.normMeans = w.NormMeans
	m.normStds = w.NormStds
	return nil
}
