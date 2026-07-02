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

	normMeans    [][]float64
	normStds     [][]float64
	grads        []float64 // gradient accumulator for TrainableBackend
	shadowParams []float64 // float64 shadow of float32 graph parameters
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

// forwardBatch runs the forward pass on a batch of samples simultaneously.
// Input: [batch][channels][inputLen], returns flat [batch * channels * outputLen].
// The SSM scan processes all batch samples in parallel via [batch, seqLen, dModel] tensors.
func (m *Mamba) forwardBatch(ctx context.Context, inputs [][][]float64) ([]float32, error) {
	batch := len(inputs)
	channels := m.config.Channels
	seqLen := m.config.InputLen
	dModel := m.config.DModel
	outDim := channels * m.config.OutputLen

	// Build input tensor [batch, seqLen, channels].
	inData := make([]float32, batch*seqLen*channels)
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				inData[(b*seqLen+t)*channels+c] = float32(inputs[b][c][t])
			}
		}
	}
	inTensor, err := tensor.New[float32]([]int{batch, seqLen, channels}, inData)
	if err != nil {
		return nil, err
	}

	// Embed: [batch, seqLen, channels] -> [batch, seqLen, dModel].
	x, err := m.embed.Forward(ctx, inTensor)
	if err != nil {
		return nil, fmt.Errorf("mamba: embed forward: %w", err)
	}

	// Process through Mamba blocks with residual connections.
	for l := 0; l < m.config.NLayers; l++ {
		out, err := m.blocks[l].Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("mamba: block %d forward: %w", l, err)
		}
		x, err = m.engine.Add(ctx, x, out)
		if err != nil {
			return nil, fmt.Errorf("mamba: block %d residual: %w", l, err)
		}
	}

	// Extract last timestep for each batch: [batch, seqLen, dModel] -> [batch, dModel].
	xData := x.Data()
	lastData := make([]float32, batch*dModel)
	for b := 0; b < batch; b++ {
		srcOff := (b*seqLen + seqLen - 1) * dModel
		copy(lastData[b*dModel:(b+1)*dModel], xData[srcOff:srcOff+dModel])
	}
	lastTensor, err := tensor.New[float32]([]int{batch, dModel}, lastData)
	if err != nil {
		return nil, err
	}

	// Head: [batch, dModel] -> [batch, channels*outputLen].
	headOut, err := m.head.Forward(ctx, lastTensor)
	if err != nil {
		return nil, fmt.Errorf("mamba: head forward: %w", err)
	}

	// Return flat output [batch * outDim].
	result := make([]float32, batch*outDim)
	copy(result, headOut.Data())
	return result, nil
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

// flatParamCount returns the total number of scalar float32 parameters.
func (m *Mamba) flatParamCount() int {
	n := 0
	for _, p := range m.allParams() {
		n += len(p.Value.Data())
	}
	return n
}

// ensureShadowParams lazily initializes the float64 shadow parameter array
// and syncs it from the float32 graph parameters. Only syncs on first call;
// subsequent calls preserve shadow values (which may have been updated by adamWUpdate).
func (m *Mamba) ensureShadowParams() {
	n := m.flatParamCount()
	if len(m.shadowParams) == n {
		return
	}
	m.shadowParams = make([]float64, n)
	idx := 0
	for _, p := range m.allParams() {
		for _, v := range p.Value.Data() {
			m.shadowParams[idx] = float64(v)
			idx++
		}
	}
}

// syncShadowToGraph copies float64 shadow params back to float32 graph parameters.
func (m *Mamba) syncShadowToGraph() {
	idx := 0
	for _, p := range m.allParams() {
		data := p.Value.Data()
		for i := range data {
			data[i] = float32(m.shadowParams[idx])
			idx++
		}
	}
}

// ForwardSample runs the Mamba forward pass on a single sample and returns
// a flat float64 output with cached activations for BackwardSample.
func (m *Mamba) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	// Sync shadow params (float64) back to graph params (float32) so the
	// forward pass uses the latest values after any adamWUpdate modifications.
	if len(m.shadowParams) > 0 {
		m.syncShadowToGraph()
	}

	ctx := context.Background()
	predData, cache, err := m.forward(ctx, input)
	if err != nil {
		return nil, nil, err
	}

	flat := make([]float64, len(predData))
	for i, v := range predData {
		flat[i] = float64(v)
	}
	return flat, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (m *Mamba) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*mambaCache)
	if !ok {
		return fmt.Errorf("mamba: invalid cache type")
	}

	if m.grads == nil {
		m.grads = make([]float64, m.flatParamCount())
	}

	// Zero graph gradients before backward so we get only this sample's contribution.
	m.zeroGrads()

	dOut := make([]float32, len(dOutput))
	for i, v := range dOutput {
		dOut[i] = float32(v)
	}

	ctx := context.Background()
	if err := m.backward(ctx, dOut, cache); err != nil {
		return err
	}

	// Accumulate graph parameter gradients into the float64 gradient buffer.
	idx := 0
	for _, p := range m.allParams() {
		if p.Gradient != nil {
			for _, g := range p.Gradient.Data() {
				if isFinite(float64(g)) {
					m.grads[idx] += float64(g)
				}
				idx++
			}
		} else {
			idx += len(p.Value.Data())
		}
	}

	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (m *Mamba) FlatGrads() []float64 {
	if m.grads == nil {
		m.grads = make([]float64, m.flatParamCount())
	}
	return m.grads
}

// ZeroGrads resets all accumulated gradients to zero (TrainableBackend).
func (m *Mamba) ZeroGrads() {
	if m.grads == nil {
		m.grads = make([]float64, m.flatParamCount())
		return
	}
	for i := range m.grads {
		m.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters as float64 pointers.
// Uses a shadow float64 array that is synced from/to the float32 graph parameters.
func (m *Mamba) FlatParams() []*float64 {
	m.ensureShadowParams()
	ptrs := make([]*float64, len(m.shadowParams))
	for i := range m.shadowParams {
		ptrs[i] = &m.shadowParams[i]
	}
	return ptrs
}

// ParamCount returns the total number of trainable scalar parameters.
func (m *Mamba) ParamCount() int {
	return m.flatParamCount()
}

// Compile-time check that Mamba implements TrainableBackend.
var _ TrainableBackend = (*Mamba)(nil)

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

	// Z-score normalize inputs.
	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	// Initialize shadow params from graph params before TrainLoop.
	m.ensureShadowParams()

	result, err := TrainLoop(m, windows, labels, config)
	if err != nil {
		return nil, err
	}

	// Sync final shadow params back to float32 graph parameters.
	m.syncShadowToGraph()

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

	// Validate channel counts.
	for i, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("mamba: window %d expected %d channels, got %d", i, m.config.Channels, len(w))
		}
	}

	ctx := context.Background()

	// Use batched forward pass for all samples simultaneously.
	predData, err := m.forwardBatch(ctx, windows)
	if err != nil {
		return nil, fmt.Errorf("mamba: forward batch: %w", err)
	}

	out := make([]float64, len(predData))
	for i, v := range predData {
		out[i] = float64(v)
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
