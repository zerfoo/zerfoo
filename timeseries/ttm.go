package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"math"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TTMTrainConfig holds configuration for a TTM (TinyTimeMixer) model.
type TTMTrainConfig struct {
	ContextLen     int     // length of input time series context window
	ForecastLen    int     // number of future steps to predict
	NumChannels    int     // number of input channels/features
	PatchLen       int     // length of each patch
	DModel         int     // model hidden dimension
	NumMixerLayers int     // number of TSMixer blocks per encoder/decoder
	ChannelMixing  bool    // enable feature-mixing MLPs (false = channel-independent)
	LearningRate   float64 // AdamW learning rate
	Epochs         int     // training epochs
	BatchSize      int     // mini-batch size (0 = full batch)
	FreezeEncoder  bool    // freeze encoder weights for few-shot fine-tuning
}

// NumPatches returns the number of patches from the context window.
func (c TTMTrainConfig) NumPatches() int {
	if c.PatchLen <= 0 {
		return 0
	}
	return c.ContextLen / c.PatchLen
}

// ForecastPatches returns the number of forecast patches.
func (c TTMTrainConfig) ForecastPatches() int {
	if c.PatchLen <= 0 {
		return 0
	}
	return c.ForecastLen / c.PatchLen
}

// ttmMixerLayerF64 holds the float64 parameters of a TSMixer block for training.
type ttmMixerLayerF64 struct {
	// Time-mixing MLP: [numPatches, numPatches]
	timeMLP1W []float64 // [numPatches * numPatches]
	timeMLP1B []float64 // [numPatches]
	timeMLP2W []float64 // [numPatches * numPatches]
	timeMLP2B []float64 // [numPatches]
	// Time norm: [dModel]
	timeNormScale []float64
	timeNormBias  []float64

	// Feature-mixing MLP: [dModel, ffnDim] and [ffnDim, dModel]
	featMLP1W []float64 // [dModel * ffnDim] (only if channelMixing)
	featMLP1B []float64 // [ffnDim]
	featMLP2W []float64 // [ffnDim * dModel]
	featMLP2B []float64 // [dModel]
	// Feature norm: [dModel]
	featNormScale []float64
	featNormBias  []float64
}

// ttmParamsF64 holds all float64 parameters for training.
type ttmParamsF64 struct {
	patchEmbW []float64 // [patchLen * dModel]
	patchEmbB []float64 // [dModel]
	encoder   []ttmMixerLayerF64
	decoder   []ttmMixerLayerF64
	headW     []float64 // [forecastPatches * dModel * forecastLen]
	headB     []float64 // [forecastLen]
}

func (p *ttmParamsF64) paramCount() int {
	n := len(p.patchEmbW) + len(p.patchEmbB)
	for _, l := range p.encoder {
		n += l.paramCount()
	}
	for _, l := range p.decoder {
		n += l.paramCount()
	}
	n += len(p.headW) + len(p.headB)
	return n
}

func (l *ttmMixerLayerF64) paramCount() int {
	n := len(l.timeMLP1W) + len(l.timeMLP1B) + len(l.timeMLP2W) + len(l.timeMLP2B)
	n += len(l.timeNormScale) + len(l.timeNormBias)
	n += len(l.featMLP1W) + len(l.featMLP1B) + len(l.featMLP2W) + len(l.featMLP2B)
	n += len(l.featNormScale) + len(l.featNormBias)
	return n
}

// flatParams returns pointers to all trainable parameters in order.
func (p *ttmParamsF64) flatParams() []*float64 {
	n := p.paramCount()
	out := make([]*float64, 0, n)
	out = appendPtrs(out, p.patchEmbW)
	out = appendPtrs(out, p.patchEmbB)
	for i := range p.encoder {
		out = p.encoder[i].appendPtrs(out)
	}
	for i := range p.decoder {
		out = p.decoder[i].appendPtrs(out)
	}
	out = appendPtrs(out, p.headW)
	out = appendPtrs(out, p.headB)
	return out
}

// flatParamsExcluding returns param pointers, skipping encoder params
// when freezeEncoder is true.
func (p *ttmParamsF64) flatParamsExcluding(freezeEncoder bool) []*float64 {
	n := p.paramCount()
	out := make([]*float64, 0, n)
	out = appendPtrs(out, p.patchEmbW)
	out = appendPtrs(out, p.patchEmbB)
	if !freezeEncoder {
		for i := range p.encoder {
			out = p.encoder[i].appendPtrs(out)
		}
	}
	for i := range p.decoder {
		out = p.decoder[i].appendPtrs(out)
	}
	out = appendPtrs(out, p.headW)
	out = appendPtrs(out, p.headB)
	return out
}

func (l *ttmMixerLayerF64) appendPtrs(out []*float64) []*float64 {
	out = appendPtrs(out, l.timeMLP1W)
	out = appendPtrs(out, l.timeMLP1B)
	out = appendPtrs(out, l.timeMLP2W)
	out = appendPtrs(out, l.timeMLP2B)
	out = appendPtrs(out, l.timeNormScale)
	out = appendPtrs(out, l.timeNormBias)
	out = appendPtrs(out, l.featMLP1W)
	out = appendPtrs(out, l.featMLP1B)
	out = appendPtrs(out, l.featMLP2W)
	out = appendPtrs(out, l.featMLP2B)
	out = appendPtrs(out, l.featNormScale)
	out = appendPtrs(out, l.featNormBias)
	return out
}

func appendPtrs(out []*float64, slice []float64) []*float64 {
	for i := range slice {
		out = append(out, &slice[i])
	}
	return out
}

// TTM implements the TinyTimeMixer model for time series forecasting.
// It supports zero-shot inference and few-shot fine-tuning.
type TTM struct {
	config    TTMTrainConfig
	engine    compute.Engine[float32]
	ops       numeric.Arithmetic[float32]
	patchEmb  linearLayer // patchLen -> dModel
	encoder   []ttmMixerBlockF32
	decoder   []ttmMixerBlockF32
	head      linearLayer // forecastPatches * dModel -> forecastLen
	normMeans   [][]float64
	normStds    [][]float64
	trainParams *ttmParamsF64 // extracted f64 params during training
	grads       []float64    // gradient accumulator for TrainableBackend
}

// ttmMixerBlockF32 holds the float32 weights of a TSMixer block for inference.
type ttmMixerBlockF32 struct {
	timeMLP1 linearLayer
	timeMLP2 linearLayer
	timeNorm struct {
		scale *tensor.TensorNumeric[float32]
		bias  *tensor.TensorNumeric[float32]
	}
	featMLP1 linearLayer
	featMLP2 linearLayer
	featNorm struct {
		scale *tensor.TensorNumeric[float32]
		bias  *tensor.TensorNumeric[float32]
	}
	channelMixing bool
	numPatches    int
	dModel        int
}

// NewTTM creates a new TTM model with the given configuration.
func NewTTM(config TTMTrainConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*TTM, error) {
	if config.ContextLen <= 0 {
		return nil, fmt.Errorf("ttm: ContextLen must be positive, got %d", config.ContextLen)
	}
	if config.ForecastLen <= 0 {
		return nil, fmt.Errorf("ttm: ForecastLen must be positive, got %d", config.ForecastLen)
	}
	if config.PatchLen <= 0 {
		return nil, fmt.Errorf("ttm: PatchLen must be positive, got %d", config.PatchLen)
	}
	if config.ContextLen%config.PatchLen != 0 {
		return nil, fmt.Errorf("ttm: ContextLen (%d) must be divisible by PatchLen (%d)", config.ContextLen, config.PatchLen)
	}
	if config.ForecastLen%config.PatchLen != 0 {
		return nil, fmt.Errorf("ttm: ForecastLen (%d) must be divisible by PatchLen (%d)", config.ForecastLen, config.PatchLen)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("ttm: DModel must be positive, got %d", config.DModel)
	}
	if config.NumMixerLayers <= 0 {
		return nil, fmt.Errorf("ttm: NumMixerLayers must be positive, got %d", config.NumMixerLayers)
	}
	if config.NumChannels <= 0 {
		config.NumChannels = 1
	}

	numPatches := config.NumPatches()
	forecastPatches := config.ForecastPatches()

	m := &TTM{
		config: config,
		engine: engine,
		ops:    ops,
	}

	var err error

	// Patch embedding: patchLen -> dModel.
	m.patchEmb, err = newLinearXavier(config.PatchLen, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("ttm: patch embedding: %w", err)
	}

	// Encoder: TSMixer blocks operating on numPatches.
	m.encoder = make([]ttmMixerBlockF32, config.NumMixerLayers)
	for i := range config.NumMixerLayers {
		m.encoder[i], err = newTTMMixerBlock(numPatches, config.DModel, config.ChannelMixing)
		if err != nil {
			return nil, fmt.Errorf("ttm: encoder block %d: %w", i, err)
		}
	}

	// Decoder: TSMixer blocks operating on forecastPatches.
	m.decoder = make([]ttmMixerBlockF32, config.NumMixerLayers)
	for i := range config.NumMixerLayers {
		m.decoder[i], err = newTTMMixerBlock(forecastPatches, config.DModel, config.ChannelMixing)
		if err != nil {
			return nil, fmt.Errorf("ttm: decoder block %d: %w", i, err)
		}
	}

	// Forecast head: forecastPatches * dModel -> forecastLen.
	m.head, err = newLinearXavier(forecastPatches*config.DModel, config.ForecastLen)
	if err != nil {
		return nil, fmt.Errorf("ttm: forecast head: %w", err)
	}

	return m, nil
}

// newTTMMixerBlock creates a single TSMixer block with Xavier-initialized weights.
func newTTMMixerBlock(numPatches, dModel int, channelMixing bool) (ttmMixerBlockF32, error) {
	var b ttmMixerBlockF32
	b.channelMixing = channelMixing
	b.numPatches = numPatches
	b.dModel = dModel

	var err error

	// Time-mixing MLP: numPatches -> numPatches.
	b.timeMLP1, err = newLinearXavier(numPatches, numPatches)
	if err != nil {
		return b, fmt.Errorf("time MLP1: %w", err)
	}
	b.timeMLP2, err = newLinearXavier(numPatches, numPatches)
	if err != nil {
		return b, fmt.Errorf("time MLP2: %w", err)
	}

	// Time norm parameters.
	ones := make([]float32, dModel)
	for i := range ones {
		ones[i] = 1.0
	}
	zeros := make([]float32, dModel)
	b.timeNorm.scale, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), ones...))
	if err != nil {
		return b, fmt.Errorf("time norm scale: %w", err)
	}
	b.timeNorm.bias, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), zeros...))
	if err != nil {
		return b, fmt.Errorf("time norm bias: %w", err)
	}

	if channelMixing {
		ffnDim := dModel * 2
		b.featMLP1, err = newLinearXavier(dModel, ffnDim)
		if err != nil {
			return b, fmt.Errorf("feat MLP1: %w", err)
		}
		b.featMLP2, err = newLinearXavier(ffnDim, dModel)
		if err != nil {
			return b, fmt.Errorf("feat MLP2: %w", err)
		}
		b.featNorm.scale, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), ones...))
		if err != nil {
			return b, fmt.Errorf("feat norm scale: %w", err)
		}
		b.featNorm.bias, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), zeros...))
		if err != nil {
			return b, fmt.Errorf("feat norm bias: %w", err)
		}
	}

	return b, nil
}

// extractParamsF64 copies model weights into float64 parameter struct.
func (m *TTM) extractParamsF64() *ttmParamsF64 {
	p := &ttmParamsF64{
		patchEmbW: f32ToF64(m.patchEmb.weights.Data()),
		patchEmbB: f32ToF64(m.patchEmb.biases.Data()),
		headW:     f32ToF64(m.head.weights.Data()),
		headB:     f32ToF64(m.head.biases.Data()),
	}
	p.encoder = make([]ttmMixerLayerF64, len(m.encoder))
	for i := range m.encoder {
		p.encoder[i] = extractMixerBlockF64(&m.encoder[i])
	}
	p.decoder = make([]ttmMixerLayerF64, len(m.decoder))
	for i := range m.decoder {
		p.decoder[i] = extractMixerBlockF64(&m.decoder[i])
	}
	return p
}

func extractMixerBlockF64(b *ttmMixerBlockF32) ttmMixerLayerF64 {
	l := ttmMixerLayerF64{
		timeMLP1W:     f32ToF64(b.timeMLP1.weights.Data()),
		timeMLP1B:     f32ToF64(b.timeMLP1.biases.Data()),
		timeMLP2W:     f32ToF64(b.timeMLP2.weights.Data()),
		timeMLP2B:     f32ToF64(b.timeMLP2.biases.Data()),
		timeNormScale: f32ToF64(b.timeNorm.scale.Data()),
		timeNormBias:  f32ToF64(b.timeNorm.bias.Data()),
	}
	if b.channelMixing {
		l.featMLP1W = f32ToF64(b.featMLP1.weights.Data())
		l.featMLP1B = f32ToF64(b.featMLP1.biases.Data())
		l.featMLP2W = f32ToF64(b.featMLP2.weights.Data())
		l.featMLP2B = f32ToF64(b.featMLP2.biases.Data())
		l.featNormScale = f32ToF64(b.featNorm.scale.Data())
		l.featNormBias = f32ToF64(b.featNorm.bias.Data())
	}
	return l
}

// writeBackF32 copies trained float64 parameters back to float32 model tensors.
func (m *TTM) writeBackF32(p *ttmParamsF64) {
	writeF64ToTensor(p.patchEmbW, m.patchEmb.weights)
	writeF64ToTensor(p.patchEmbB, m.patchEmb.biases)
	for i := range m.encoder {
		writeMixerBlockF32(&m.encoder[i], &p.encoder[i])
	}
	for i := range m.decoder {
		writeMixerBlockF32(&m.decoder[i], &p.decoder[i])
	}
	writeF64ToTensor(p.headW, m.head.weights)
	writeF64ToTensor(p.headB, m.head.biases)
}

func writeMixerBlockF32(b *ttmMixerBlockF32, l *ttmMixerLayerF64) {
	writeF64ToTensor(l.timeMLP1W, b.timeMLP1.weights)
	writeF64ToTensor(l.timeMLP1B, b.timeMLP1.biases)
	writeF64ToTensor(l.timeMLP2W, b.timeMLP2.weights)
	writeF64ToTensor(l.timeMLP2B, b.timeMLP2.biases)
	writeF64ToTensor(l.timeNormScale, b.timeNorm.scale)
	writeF64ToTensor(l.timeNormBias, b.timeNorm.bias)
	if b.channelMixing {
		writeF64ToTensor(l.featMLP1W, b.featMLP1.weights)
		writeF64ToTensor(l.featMLP1B, b.featMLP1.biases)
		writeF64ToTensor(l.featMLP2W, b.featMLP2.weights)
		writeF64ToTensor(l.featMLP2B, b.featMLP2.biases)
		writeF64ToTensor(l.featNormScale, b.featNorm.scale)
		writeF64ToTensor(l.featNormBias, b.featNorm.bias)
	}
}

func writeF64ToTensor(src []float64, dst *tensor.TensorNumeric[float32]) {
	data := dst.Data()
	for i, v := range src {
		data[i] = float32(v)
	}
}

func f32ToF64(src []float32) []float64 {
	out := make([]float64, len(src))
	for i, v := range src {
		out[i] = float64(v)
	}
	return out
}

// Forward runs the TTM forward pass on input time series.
// input shape: [batch, channels, contextLen] or [batch, contextLen].
// Returns predictions of shape [batch, forecastLen].
func (m *TTM) Forward(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	var batch, channels, length int

	switch len(shape) {
	case 2:
		batch, length = shape[0], shape[1]
		channels = 1
	case 3:
		batch, channels, length = shape[0], shape[1], shape[2]
	default:
		return nil, fmt.Errorf("ttm: input must be 2D or 3D, got shape %v", shape)
	}

	if length != m.config.ContextLen {
		return nil, fmt.Errorf("ttm: expected context length %d, got %d", m.config.ContextLen, length)
	}

	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()

	// Reshape to [batch*channels, contextLen].
	flat, err := m.engine.Reshape(ctx, input, []int{batch * channels, length})
	if err != nil {
		return nil, fmt.Errorf("ttm: reshape input: %w", err)
	}

	// Extract non-overlapping patches: [batch*channels, numPatches, patchLen].
	patches, err := m.extractPatchesTTM(ctx, flat, numPatches)
	if err != nil {
		return nil, fmt.Errorf("ttm: extract patches: %w", err)
	}

	// Patch embedding: [batch*channels*numPatches, patchLen] -> [batch*channels*numPatches, dModel].
	pFlat, err := m.engine.Reshape(ctx, patches, []int{batch * channels * numPatches, m.config.PatchLen})
	if err != nil {
		return nil, fmt.Errorf("ttm: reshape patches: %w", err)
	}
	embedded, err := m.linearF32(ctx, pFlat, m.patchEmb)
	if err != nil {
		return nil, fmt.Errorf("ttm: patch embed: %w", err)
	}

	// Reshape to [batch*channels, numPatches, dModel].
	x, err := m.engine.Reshape(ctx, embedded, []int{batch * channels, numPatches, m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("ttm: reshape embedded: %w", err)
	}

	// Encoder: TSMixer blocks.
	for i, block := range m.encoder {
		x, err = m.mixerForward(ctx, x, block)
		if err != nil {
			return nil, fmt.Errorf("ttm: encoder block %d: %w", i, err)
		}
	}

	// Map encoder output [batch*channels, numPatches, dModel]
	// to decoder input [batch*channels, forecastPatches, dModel] by taking last forecastPatches.
	if numPatches > forecastPatches {
		data := x.Data()
		n := batch * channels
		decData := make([]float32, n*forecastPatches*m.config.DModel)
		for i := 0; i < n; i++ {
			srcOff := i*numPatches*m.config.DModel + (numPatches-forecastPatches)*m.config.DModel
			dstOff := i * forecastPatches * m.config.DModel
			copy(decData[dstOff:dstOff+forecastPatches*m.config.DModel], data[srcOff:srcOff+forecastPatches*m.config.DModel])
		}
		x, err = tensor.New[float32]([]int{n, forecastPatches, m.config.DModel}, decData)
		if err != nil {
			return nil, fmt.Errorf("ttm: slice encoder output: %w", err)
		}
	}

	// Decoder: TSMixer blocks.
	for i, block := range m.decoder {
		x, err = m.mixerForward(ctx, x, block)
		if err != nil {
			return nil, fmt.Errorf("ttm: decoder block %d: %w", i, err)
		}
	}

	// Flatten: [batch*channels, forecastPatches * dModel].
	x, err = m.engine.Reshape(ctx, x, []int{batch * channels, forecastPatches * m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("ttm: flatten: %w", err)
	}

	// Forecast head.
	x, err = m.linearF32(ctx, x, m.head)
	if err != nil {
		return nil, fmt.Errorf("ttm: forecast head: %w", err)
	}

	// Average channels: [batch, forecastLen].
	if channels > 1 {
		x, err = m.engine.Reshape(ctx, x, []int{batch, channels, m.config.ForecastLen})
		if err != nil {
			return nil, fmt.Errorf("ttm: reshape channels: %w", err)
		}
		x, err = m.engine.ReduceMean(ctx, x, 1, false)
		if err != nil {
			return nil, fmt.Errorf("ttm: reduce channels: %w", err)
		}
	}

	return x, nil
}

// extractPatchesTTM extracts non-overlapping patches from the input.
func (m *TTM) extractPatchesTTM(_ context.Context, input *tensor.TensorNumeric[float32], numPatches int) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	n := shape[0]
	data := input.Data()

	outData := make([]float32, n*numPatches*m.config.PatchLen)
	for i := 0; i < n; i++ {
		for p := 0; p < numPatches; p++ {
			srcOff := i*m.config.ContextLen + p*m.config.PatchLen
			dstOff := (i*numPatches + p) * m.config.PatchLen
			copy(outData[dstOff:dstOff+m.config.PatchLen], data[srcOff:srcOff+m.config.PatchLen])
		}
	}

	return tensor.New[float32]([]int{n, numPatches, m.config.PatchLen}, outData)
}

// mixerForward runs one TSMixer block in float32 using the engine.
func (m *TTM) mixerForward(ctx context.Context, x *tensor.TensorNumeric[float32], block ttmMixerBlockF32) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	n, nP, dM := shape[0], shape[1], shape[2]

	// Time-mixing: norm -> transpose -> MLP -> transpose -> residual.
	residual := x

	normed, err := m.layerNormF32(ctx, x, block.timeNorm.scale, block.timeNorm.bias)
	if err != nil {
		return nil, fmt.Errorf("time norm: %w", err)
	}

	// Transpose: [n, nP, dM] -> [n, dM, nP].
	normed, err = m.engine.Transpose(ctx, normed, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("time transpose pre: %w", err)
	}

	// MLP: reshape to [n*dM, nP], linear, gelu, linear, reshape back.
	flat, err := m.engine.Reshape(ctx, normed, []int{n * dM, nP})
	if err != nil {
		return nil, err
	}
	flat, err = m.linearF32(ctx, flat, block.timeMLP1)
	if err != nil {
		return nil, fmt.Errorf("time MLP1: %w", err)
	}
	flat, err = functional.GELU(ctx, m.engine, m.ops, flat)
	if err != nil {
		return nil, fmt.Errorf("time GELU: %w", err)
	}
	flat, err = m.linearF32(ctx, flat, block.timeMLP2)
	if err != nil {
		return nil, fmt.Errorf("time MLP2: %w", err)
	}

	// Reshape back and transpose.
	x, err = m.engine.Reshape(ctx, flat, []int{n, dM, nP})
	if err != nil {
		return nil, err
	}
	x, err = m.engine.Transpose(ctx, x, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("time transpose post: %w", err)
	}

	// Residual.
	x, err = m.engine.Add(ctx, x, residual)
	if err != nil {
		return nil, fmt.Errorf("time residual: %w", err)
	}

	// Feature-mixing (if enabled).
	if block.channelMixing {
		residual = x

		normed, err = m.layerNormF32(ctx, x, block.featNorm.scale, block.featNorm.bias)
		if err != nil {
			return nil, fmt.Errorf("feat norm: %w", err)
		}

		flat, err = m.engine.Reshape(ctx, normed, []int{n * nP, dM})
		if err != nil {
			return nil, err
		}
		flat, err = m.linearF32(ctx, flat, block.featMLP1)
		if err != nil {
			return nil, fmt.Errorf("feat MLP1: %w", err)
		}
		flat, err = functional.GELU(ctx, m.engine, m.ops, flat)
		if err != nil {
			return nil, fmt.Errorf("feat GELU: %w", err)
		}
		flat, err = m.linearF32(ctx, flat, block.featMLP2)
		if err != nil {
			return nil, fmt.Errorf("feat MLP2: %w", err)
		}

		x, err = m.engine.Reshape(ctx, flat, []int{n, nP, dM})
		if err != nil {
			return nil, err
		}
		x, err = m.engine.Add(ctx, x, residual)
		if err != nil {
			return nil, fmt.Errorf("feat residual: %w", err)
		}
	}

	return x, nil
}

// layerNormF32 applies layer normalization over the last dimension.
func (m *TTM) layerNormF32(ctx context.Context, x *tensor.TensorNumeric[float32], scale, bias *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	dModel := shape[len(shape)-1]
	outerSize := 1
	for _, s := range shape[:len(shape)-1] {
		outerSize *= s
	}

	flat, err := m.engine.Reshape(ctx, x, []int{outerSize, dModel})
	if err != nil {
		return nil, err
	}
	mean, err := m.engine.ReduceMean(ctx, flat, 1, true)
	if err != nil {
		return nil, err
	}
	centered, err := m.engine.Sub(ctx, flat, mean)
	if err != nil {
		return nil, err
	}
	sq, err := m.engine.Mul(ctx, centered, centered)
	if err != nil {
		return nil, err
	}
	variance, err := m.engine.ReduceMean(ctx, sq, 1, true)
	if err != nil {
		return nil, err
	}
	variance, err = m.engine.AddScalar(ctx, variance, 1e-5)
	if err != nil {
		return nil, err
	}
	invStd, err := m.engine.Rsqrt(ctx, variance)
	if err != nil {
		return nil, err
	}
	normed, err := m.engine.Mul(ctx, centered, invStd)
	if err != nil {
		return nil, err
	}
	normed, err = m.engine.Mul(ctx, normed, scale)
	if err != nil {
		return nil, err
	}
	normed, err = m.engine.Add(ctx, normed, bias)
	if err != nil {
		return nil, err
	}
	return m.engine.Reshape(ctx, normed, shape)
}

// linearF32 computes x @ W + b.
func (m *TTM) linearF32(ctx context.Context, x *tensor.TensorNumeric[float32], l linearLayer) (*tensor.TensorNumeric[float32], error) {
	out, err := m.engine.MatMul(ctx, x, l.weights)
	if err != nil {
		return nil, err
	}
	return m.engine.Add(ctx, out, l.biases)
}

// TrainWindowed trains the TTM model on windowed time series data.
// windows: [nSamples][channels][contextLen].
// labels: flat slice of length nSamples * forecastLen.
func (m *TTM) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("ttm: empty training set")
	}

	expectedLabels := nSamples * m.config.ForecastLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("ttm: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) == 0 {
			return nil, fmt.Errorf("ttm: window %d has 0 channels", i)
		}
		for c, ch := range w {
			if len(ch) != m.config.ContextLen {
				return nil, fmt.Errorf("ttm: window %d channel %d has length %d, expected %d",
					i, c, len(ch), m.config.ContextLen)
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

	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	if m.engine != nil {
		return m.trainWindowedEngine(windows, labels, config)
	}
	return m.trainWindowedCPU(windows, labels, config)
}

// PredictWindowed runs inference on windowed data.
// Returns flat predictions of length nSamples * forecastLen.
func (m *TTM) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := m.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("ttm: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("ttm: empty input")
	}

	if m.normMeans != nil {
		windows = applyNormalization(windows, m.normMeans, m.normStds)
	}

	params := m.extractParamsF64()
	out := make([]float64, 0, nSamples*m.config.ForecastLen)
	for _, w := range windows {
		pred := m.forwardF64(w, params)
		out = append(out, pred...)
	}
	return out, nil
}

// forwardF64 runs the TTM forward pass in float64 for a single sample.
// input: [channels][contextLen], returns [forecastLen].
func (m *TTM) forwardF64(input [][]float64, params *ttmParamsF64) []float64 {
	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()
	dModel := m.config.DModel
	channels := len(input)

	chanOutputs := make([][]float64, channels)

	for ch := 0; ch < channels; ch++ {
		// Extract non-overlapping patches.
		patches := make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			start := p * m.config.PatchLen
			patches[p] = make([]float64, m.config.PatchLen)
			copy(patches[p], input[ch][start:start+m.config.PatchLen])
		}

		// Patch embedding.
		embedded := linearF64(patches, params.patchEmbW, params.patchEmbB, m.config.PatchLen, dModel)

		// Encoder.
		x := embedded
		for _, layer := range params.encoder {
			x = m.mixerBlockF64(x, &layer, numPatches, dModel)
		}

		// Slice to last forecastPatches.
		if numPatches > forecastPatches {
			x = x[numPatches-forecastPatches:]
		}

		// Decoder.
		for _, layer := range params.decoder {
			x = m.mixerBlockF64(x, &layer, forecastPatches, dModel)
		}

		// Flatten and forecast head.
		flatInput := make([]float64, forecastPatches*dModel)
		for p := 0; p < forecastPatches; p++ {
			copy(flatInput[p*dModel:(p+1)*dModel], x[p])
		}

		headIn := forecastPatches * dModel
		headOut := linearF64([][]float64{flatInput}, params.headW, params.headB, headIn, m.config.ForecastLen)
		chanOutputs[ch] = headOut[0]
	}

	// Average channels.
	result := make([]float64, m.config.ForecastLen)
	for ch := 0; ch < channels; ch++ {
		for j := 0; j < m.config.ForecastLen; j++ {
			result[j] += chanOutputs[ch][j]
		}
	}
	for j := range result {
		result[j] /= float64(channels)
	}
	return result
}

// mixerBlockF64 runs a single TSMixer block in float64.
func (m *TTM) mixerBlockF64(x [][]float64, layer *ttmMixerLayerF64, nPatches, dModel int) [][]float64 {
	// Time-mixing: norm -> transpose -> MLP -> transpose -> residual.
	residual := copyMatrix(x)
	normed := layerNormF64(x, layer.timeNormScale, layer.timeNormBias, dModel)

	// Transpose: [nPatches, dModel] -> [dModel, nPatches].
	transposed := transposeMatrix(normed, nPatches, dModel)

	// MLP: linear -> GELU -> linear.
	h := linearF64(transposed, layer.timeMLP1W, layer.timeMLP1B, nPatches, nPatches)
	h = geluMatrix(h)
	h = linearF64(h, layer.timeMLP2W, layer.timeMLP2B, nPatches, nPatches)

	// Transpose back: [dModel, nPatches] -> [nPatches, dModel].
	h = transposeMatrix(h, dModel, nPatches)

	// Residual.
	for p := 0; p < nPatches; p++ {
		for j := 0; j < dModel; j++ {
			h[p][j] += residual[p][j]
		}
	}

	// Feature-mixing (if enabled).
	if len(layer.featMLP1W) > 0 {
		residual = copyMatrix(h)
		normed = layerNormF64(h, layer.featNormScale, layer.featNormBias, dModel)

		ffnDim := len(layer.featMLP1B)
		f := linearF64(normed, layer.featMLP1W, layer.featMLP1B, dModel, ffnDim)
		f = geluMatrix(f)
		f = linearF64(f, layer.featMLP2W, layer.featMLP2B, ffnDim, dModel)

		for p := 0; p < nPatches; p++ {
			for j := 0; j < dModel; j++ {
				f[p][j] += residual[p][j]
			}
		}
		return f
	}
	return h
}

// forwardF64WithCache runs forward pass and caches activations for backward.
func (m *TTM) forwardF64WithCache(input [][]float64, params *ttmParamsF64) ([]float64, *ttmCacheF64) {
	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()
	dModel := m.config.DModel
	channels := len(input)

	cache := &ttmCacheF64{
		channels: make([]ttmChannelCache, channels),
	}
	chanOutputs := make([][]float64, channels)

	for ch := 0; ch < channels; ch++ {
		cc := &cache.channels[ch]

		// Extract patches.
		cc.patches = make([][]float64, numPatches)
		for p := 0; p < numPatches; p++ {
			start := p * m.config.PatchLen
			cc.patches[p] = make([]float64, m.config.PatchLen)
			copy(cc.patches[p], input[ch][start:start+m.config.PatchLen])
		}

		// Patch embedding.
		cc.embedded = linearF64(cc.patches, params.patchEmbW, params.patchEmbB, m.config.PatchLen, dModel)

		// Encoder.
		x := cc.embedded
		cc.encoderCaches = make([]ttmMixerCacheF64, len(params.encoder))
		for i, layer := range params.encoder {
			x, cc.encoderCaches[i] = m.mixerBlockF64WithCache(x, &layer, numPatches, dModel)
		}
		cc.encoderOutput = copyMatrix(x)

		// Slice to last forecastPatches.
		if numPatches > forecastPatches {
			x = x[numPatches-forecastPatches:]
		}

		// Decoder.
		cc.decoderCaches = make([]ttmMixerCacheF64, len(params.decoder))
		for i, layer := range params.decoder {
			x, cc.decoderCaches[i] = m.mixerBlockF64WithCache(x, &layer, forecastPatches, dModel)
		}

		// Flatten.
		cc.flatInput = make([]float64, forecastPatches*dModel)
		for p := 0; p < forecastPatches; p++ {
			copy(cc.flatInput[p*dModel:(p+1)*dModel], x[p])
		}

		headIn := forecastPatches * dModel
		headOut := linearF64([][]float64{cc.flatInput}, params.headW, params.headB, headIn, m.config.ForecastLen)
		chanOutputs[ch] = headOut[0]
	}

	result := make([]float64, m.config.ForecastLen)
	for ch := 0; ch < channels; ch++ {
		for j := 0; j < m.config.ForecastLen; j++ {
			result[j] += chanOutputs[ch][j]
		}
	}
	for j := range result {
		result[j] /= float64(channels)
	}
	return result, cache
}

// ttmCacheF64 stores intermediate activations for backpropagation.
type ttmCacheF64 struct {
	channels []ttmChannelCache
}

type ttmChannelCache struct {
	patches       [][]float64
	embedded      [][]float64
	encoderCaches []ttmMixerCacheF64
	encoderOutput [][]float64
	decoderCaches []ttmMixerCacheF64
	flatInput     []float64
}

type ttmMixerCacheF64 struct {
	input     [][]float64
	normed    [][]float64
	mean      []float64
	invStd    []float64
	centered  [][]float64
	transposed [][]float64
	mlp1Out   [][]float64
	mlp1Pre   [][]float64
	mlp2Out   [][]float64

	// Feature-mixing cache.
	featInput    [][]float64
	featNormed   [][]float64
	featMean     []float64
	featInvStd   []float64
	featCentered [][]float64
	featMLP1Pre  [][]float64
	featMLP1Out  [][]float64
	featMLP2Out  [][]float64
}

// mixerBlockF64WithCache runs one TSMixer block in float64 and caches activations.
func (m *TTM) mixerBlockF64WithCache(x [][]float64, layer *ttmMixerLayerF64, nPatches, dModel int) ([][]float64, ttmMixerCacheF64) {
	var mc ttmMixerCacheF64
	mc.input = copyMatrix(x)

	// Time-mixing.
	normed, mean, invStd, centered := layerNormF64WithCache(x, layer.timeNormScale, layer.timeNormBias, dModel)
	mc.normed = normed
	mc.mean = mean
	mc.invStd = invStd
	mc.centered = centered

	transposed := transposeMatrix(normed, nPatches, dModel)
	mc.transposed = transposed

	mlp1Pre := linearF64(transposed, layer.timeMLP1W, layer.timeMLP1B, nPatches, nPatches)
	mc.mlp1Pre = mlp1Pre
	mlp1Out := geluMatrix(mlp1Pre)
	mc.mlp1Out = mlp1Out
	mlp2Out := linearF64(mlp1Out, layer.timeMLP2W, layer.timeMLP2B, nPatches, nPatches)
	mc.mlp2Out = mlp2Out

	h := transposeMatrix(mlp2Out, dModel, nPatches)
	for p := 0; p < nPatches; p++ {
		for j := 0; j < dModel; j++ {
			h[p][j] += mc.input[p][j]
		}
	}

	// Feature-mixing.
	if len(layer.featMLP1W) > 0 {
		mc.featInput = copyMatrix(h)
		normed, mean, invStd, centered := layerNormF64WithCache(h, layer.featNormScale, layer.featNormBias, dModel)
		mc.featNormed = normed
		mc.featMean = mean
		mc.featInvStd = invStd
		mc.featCentered = centered

		ffnDim := len(layer.featMLP1B)
		mlp1Pre := linearF64(normed, layer.featMLP1W, layer.featMLP1B, dModel, ffnDim)
		mc.featMLP1Pre = mlp1Pre
		mlp1Out := geluMatrix(mlp1Pre)
		mc.featMLP1Out = mlp1Out
		mlp2Out := linearF64(mlp1Out, layer.featMLP2W, layer.featMLP2B, ffnDim, dModel)
		mc.featMLP2Out = mlp2Out

		for p := 0; p < nPatches; p++ {
			for j := 0; j < dModel; j++ {
				mlp2Out[p][j] += mc.featInput[p][j]
			}
		}
		return mlp2Out, mc
	}
	return h, mc
}

// backwardF64 computes gradients for a single sample via analytical backpropagation.
// dOutput: [forecastLen], returns gradient flat vector aligned with flatParams.
func (m *TTM) backwardF64(dOutput []float64, params *ttmParamsF64, cache *ttmCacheF64, freezeEncoder bool) []float64 {
	numPatches := m.config.NumPatches()
	forecastPatches := m.config.ForecastPatches()
	dModel := m.config.DModel
	forecastLen := m.config.ForecastLen
	channels := len(cache.channels)

	// Allocate gradient accumulators.
	grads := &ttmParamsF64{
		patchEmbW: make([]float64, len(params.patchEmbW)),
		patchEmbB: make([]float64, len(params.patchEmbB)),
		headW:     make([]float64, len(params.headW)),
		headB:     make([]float64, len(params.headB)),
	}
	grads.encoder = make([]ttmMixerLayerF64, len(params.encoder))
	for i := range params.encoder {
		grads.encoder[i] = allocMixerGrads(&params.encoder[i])
	}
	grads.decoder = make([]ttmMixerLayerF64, len(params.decoder))
	for i := range params.decoder {
		grads.decoder[i] = allocMixerGrads(&params.decoder[i])
	}

	dChanOutput := make([]float64, forecastLen)
	for j := range dChanOutput {
		dChanOutput[j] = dOutput[j] / float64(channels)
	}

	for ch := 0; ch < channels; ch++ {
		cc := &cache.channels[ch]

		// Backward through forecast head: dFlatInput and head grad.
		headIn := forecastPatches * dModel
		dFlatInput := make([]float64, headIn)
		for j := 0; j < forecastLen; j++ {
			grads.headB[j] += dChanOutput[j]
			for k := 0; k < headIn; k++ {
				grads.headW[k*forecastLen+j] += cc.flatInput[k] * dChanOutput[j]
				dFlatInput[k] += params.headW[k*forecastLen+j] * dChanOutput[j]
			}
		}

		// Unflatten dFlatInput -> [forecastPatches, dModel].
		dx := make([][]float64, forecastPatches)
		for p := 0; p < forecastPatches; p++ {
			dx[p] = make([]float64, dModel)
			copy(dx[p], dFlatInput[p*dModel:(p+1)*dModel])
		}

		// Backward through decoder blocks (reverse order).
		for i := len(params.decoder) - 1; i >= 0; i-- {
			dx = m.mixerBlockBackward(dx, &params.decoder[i], &cc.decoderCaches[i], &grads.decoder[i], forecastPatches, dModel)
		}

		if !freezeEncoder {
			// Expand dx back to numPatches (pad with zeros for encoder patches before forecast).
			if numPatches > forecastPatches {
				fullDx := make([][]float64, numPatches)
				for p := 0; p < numPatches-forecastPatches; p++ {
					fullDx[p] = make([]float64, dModel)
				}
				for p := 0; p < forecastPatches; p++ {
					fullDx[numPatches-forecastPatches+p] = dx[p]
				}
				dx = fullDx
			}

			// Backward through encoder blocks.
			for i := len(params.encoder) - 1; i >= 0; i-- {
				dx = m.mixerBlockBackward(dx, &params.encoder[i], &cc.encoderCaches[i], &grads.encoder[i], numPatches, dModel)
			}

			// Backward through patch embedding.
			for p := 0; p < numPatches; p++ {
				for j := 0; j < dModel; j++ {
					grads.patchEmbB[j] += dx[p][j]
					for k := 0; k < m.config.PatchLen; k++ {
						grads.patchEmbW[k*dModel+j] += cc.patches[p][k] * dx[p][j]
					}
				}
			}
		}
	}

	// Flatten gradients to match flatParams order.
	return grads.flatGrads(freezeEncoder)
}

func (p *ttmParamsF64) flatGrads(freezeEncoder bool) []float64 {
	n := len(p.patchEmbW) + len(p.patchEmbB)
	if !freezeEncoder {
		for _, l := range p.encoder {
			n += l.paramCount()
		}
	}
	for _, l := range p.decoder {
		n += l.paramCount()
	}
	n += len(p.headW) + len(p.headB)

	out := make([]float64, 0, n)
	out = append(out, p.patchEmbW...)
	out = append(out, p.patchEmbB...)
	if !freezeEncoder {
		for _, l := range p.encoder {
			out = l.appendGrads(out)
		}
	}
	for _, l := range p.decoder {
		out = l.appendGrads(out)
	}
	out = append(out, p.headW...)
	out = append(out, p.headB...)
	return out
}

func (l *ttmMixerLayerF64) appendGrads(out []float64) []float64 {
	out = append(out, l.timeMLP1W...)
	out = append(out, l.timeMLP1B...)
	out = append(out, l.timeMLP2W...)
	out = append(out, l.timeMLP2B...)
	out = append(out, l.timeNormScale...)
	out = append(out, l.timeNormBias...)
	out = append(out, l.featMLP1W...)
	out = append(out, l.featMLP1B...)
	out = append(out, l.featMLP2W...)
	out = append(out, l.featMLP2B...)
	out = append(out, l.featNormScale...)
	out = append(out, l.featNormBias...)
	return out
}

func allocMixerGrads(p *ttmMixerLayerF64) ttmMixerLayerF64 {
	g := ttmMixerLayerF64{
		timeMLP1W:     make([]float64, len(p.timeMLP1W)),
		timeMLP1B:     make([]float64, len(p.timeMLP1B)),
		timeMLP2W:     make([]float64, len(p.timeMLP2W)),
		timeMLP2B:     make([]float64, len(p.timeMLP2B)),
		timeNormScale: make([]float64, len(p.timeNormScale)),
		timeNormBias:  make([]float64, len(p.timeNormBias)),
		featMLP1W:     make([]float64, len(p.featMLP1W)),
		featMLP1B:     make([]float64, len(p.featMLP1B)),
		featMLP2W:     make([]float64, len(p.featMLP2W)),
		featMLP2B:     make([]float64, len(p.featMLP2B)),
		featNormScale: make([]float64, len(p.featNormScale)),
		featNormBias:  make([]float64, len(p.featNormBias)),
	}
	return g
}

// mixerBlockBackward computes gradients through a TSMixer block.
// Returns dx (gradient w.r.t. block input).
func (m *TTM) mixerBlockBackward(dOut [][]float64, layer *ttmMixerLayerF64, mc *ttmMixerCacheF64, grads *ttmMixerLayerF64, nPatches, dModel int) [][]float64 {
	dx := dOut

	// Feature-mixing backward (if enabled).
	if len(layer.featMLP1W) > 0 {
		ffnDim := len(layer.featMLP1B)

		// Residual: dx flows through to featInput.
		dResidual := copyMatrix(dx)

		// Backward through featMLP2: dFeatMLP1Out.
		dFeatMLP1Out := make([][]float64, nPatches)
		for p := 0; p < nPatches; p++ {
			dFeatMLP1Out[p] = make([]float64, ffnDim)
			for j := 0; j < dModel; j++ {
				grads.featMLP2B[j] += dx[p][j]
				for k := 0; k < ffnDim; k++ {
					grads.featMLP2W[k*dModel+j] += mc.featMLP1Out[p][k] * dx[p][j]
					dFeatMLP1Out[p][k] += layer.featMLP2W[k*dModel+j] * dx[p][j]
				}
			}
		}

		// Backward through GELU.
		dFeatMLP1Pre := make([][]float64, nPatches)
		for p := 0; p < nPatches; p++ {
			dFeatMLP1Pre[p] = make([]float64, ffnDim)
			for j := 0; j < ffnDim; j++ {
				xf := mc.featMLP1Pre[p][j]
					c := math.Sqrt(2.0 / math.Pi)
					innerVal := c * (xf + 0.044715*xf*xf*xf)
					th := math.Tanh(innerVal)
					dInner := c * (1 + 3*0.044715*xf*xf)
					dFeatMLP1Pre[p][j] = dFeatMLP1Out[p][j] * (0.5*(1+th) + 0.5*xf*(1-th*th)*dInner)
			}
		}

		// Backward through featMLP1: dNormed.
		dNormed := make([][]float64, nPatches)
		for p := 0; p < nPatches; p++ {
			dNormed[p] = make([]float64, dModel)
			for j := 0; j < ffnDim; j++ {
				grads.featMLP1B[j] += dFeatMLP1Pre[p][j]
				for k := 0; k < dModel; k++ {
					grads.featMLP1W[k*ffnDim+j] += mc.featNormed[p][k] * dFeatMLP1Pre[p][j]
					dNormed[p][k] += layer.featMLP1W[k*ffnDim+j] * dFeatMLP1Pre[p][j]
				}
			}
		}

		// Backward through feat layer norm.
		dNormInput := layerNormBackwardF64(dNormed, mc.featCentered, mc.featInvStd,
			layer.featNormScale, grads.featNormScale, grads.featNormBias, dModel)

		// Add residual gradient.
		for p := 0; p < nPatches; p++ {
			for j := 0; j < dModel; j++ {
				dx[p][j] = dNormInput[p][j] + dResidual[p][j]
			}
		}
	}

	// Time-mixing backward.
	dResidual := copyMatrix(dx)

	// Backward through transpose (post): [nPatches, dModel] -> [dModel, nPatches].
	dTransposed := transposeMatrix(dx, nPatches, dModel)

	// Backward through timeMLP2.
	dMLP1Out := make([][]float64, dModel)
	for p := 0; p < dModel; p++ {
		dMLP1Out[p] = make([]float64, nPatches)
		for j := 0; j < nPatches; j++ {
			grads.timeMLP2B[j] += dTransposed[p][j]
			for k := 0; k < nPatches; k++ {
				grads.timeMLP2W[k*nPatches+j] += mc.mlp1Out[p][k] * dTransposed[p][j]
				dMLP1Out[p][k] += layer.timeMLP2W[k*nPatches+j] * dTransposed[p][j]
			}
		}
	}

	// Backward through GELU.
	dMLP1Pre := make([][]float64, dModel)
	for p := 0; p < dModel; p++ {
		dMLP1Pre[p] = make([]float64, nPatches)
		for j := 0; j < nPatches; j++ {
			xf := mc.mlp1Pre[p][j]
				c := math.Sqrt(2.0 / math.Pi)
				innerVal := c * (xf + 0.044715*xf*xf*xf)
				th := math.Tanh(innerVal)
				dInner := c * (1 + 3*0.044715*xf*xf)
				dMLP1Pre[p][j] = dMLP1Out[p][j] * (0.5*(1+th) + 0.5*xf*(1-th*th)*dInner)
		}
	}

	// Backward through timeMLP1.
	dNormedT := make([][]float64, dModel)
	for p := 0; p < dModel; p++ {
		dNormedT[p] = make([]float64, nPatches)
		for j := 0; j < nPatches; j++ {
			grads.timeMLP1B[j] += dMLP1Pre[p][j]
			for k := 0; k < nPatches; k++ {
				grads.timeMLP1W[k*nPatches+j] += mc.transposed[p][k] * dMLP1Pre[p][j]
				dNormedT[p][k] += layer.timeMLP1W[k*nPatches+j] * dMLP1Pre[p][j]
			}
		}
	}

	// Backward through transpose (pre): [dModel, nPatches] -> [nPatches, dModel].
	dNormed := transposeMatrix(dNormedT, dModel, nPatches)

	// Backward through time layer norm.
	dNormInput := layerNormBackwardF64(dNormed, mc.centered, mc.invStd,
		layer.timeNormScale, grads.timeNormScale, grads.timeNormBias, dModel)

	// Add residual gradient.
	for p := 0; p < nPatches; p++ {
		for j := 0; j < dModel; j++ {
			dNormInput[p][j] += dResidual[p][j]
		}
	}

	return dNormInput
}

// trainWindowedCPU runs CPU-based training with analytical backpropagation.
func (m *TTM) trainWindowedCPU(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	// Extract float64 params for training, store on model for TrainableBackend methods.
	m.trainParams = m.extractParamsF64()

	result, err := TrainLoop(m, windows, labels, config)

	// Write trained params back to float32 tensors regardless of error,
	// then clear training state.
	if m.trainParams != nil {
		m.writeBackF32(m.trainParams)
		m.trainParams = nil
		m.grads = nil
	}

	return result, err
}

// ForwardSample runs the TTM forward pass on a single sample and returns
// a flat output [forecastLen] with cached activations for BackwardSample.
func (m *TTM) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	if m.trainParams == nil {
		return nil, nil, fmt.Errorf("ttm: ForwardSample called outside training context")
	}
	output, cache := m.forwardF64WithCache(input, m.trainParams)
	return output, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (m *TTM) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*ttmCacheF64)
	if !ok {
		return fmt.Errorf("ttm: invalid cache type")
	}
	if m.trainParams == nil {
		return fmt.Errorf("ttm: BackwardSample called outside training context")
	}

	nParams := m.ParamCount()
	if m.grads == nil {
		m.grads = make([]float64, nParams)
	}

	sampleGrads := m.backwardF64(dOutput, m.trainParams, cache, m.config.FreezeEncoder)
	for i := range sampleGrads {
		m.grads[i] += sampleGrads[i]
	}
	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (m *TTM) FlatGrads() []float64 {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
	}
	return m.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (m *TTM) ZeroGrads() {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
		return
	}
	for i := range m.grads {
		m.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
func (m *TTM) FlatParams() []*float64 {
	if m.trainParams == nil {
		return nil
	}
	return m.trainParams.flatParamsExcluding(m.config.FreezeEncoder)
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (m *TTM) ParamCount() int {
	if m.trainParams != nil {
		return len(m.trainParams.flatParamsExcluding(m.config.FreezeEncoder))
	}
	// Estimate from config without extracting.
	p := m.extractParamsF64()
	return len(p.flatParamsExcluding(m.config.FreezeEncoder))
}

// Compile-time check that TTM implements TrainableBackend.
var _ TrainableBackend = (*TTM)(nil)

// Helper functions for float64 matrix operations.

func transposeMatrix(x [][]float64, rows, cols int) [][]float64 {
	out := make([][]float64, cols)
	for j := 0; j < cols; j++ {
		out[j] = make([]float64, rows)
		for i := 0; i < rows; i++ {
			out[j][i] = x[i][j]
		}
	}
	return out
}

func geluMatrix(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = make([]float64, len(x[i]))
		for j, v := range x[i] {
			inner := math.Sqrt(2/math.Pi) * (v + 0.044715*v*v*v)
			out[i][j] = 0.5 * v * (1 + math.Tanh(inner))
		}
	}
	return out
}


// ttmWeights is the JSON-serializable form of TTM parameters.
type ttmWeights struct {
	Config    TTMTrainConfig      `json:"config"`
	PatchEmbW []float64           `json:"patch_emb_w"`
	PatchEmbB []float64           `json:"patch_emb_b"`
	Encoder   []ttmMixerLayerJSON `json:"encoder"`
	Decoder   []ttmMixerLayerJSON `json:"decoder"`
	HeadW     []float64           `json:"head_w"`
	HeadB     []float64           `json:"head_b"`
	NormMeans [][]float64         `json:"norm_means,omitempty"`
	NormStds  [][]float64         `json:"norm_stds,omitempty"`
}

type ttmMixerLayerJSON struct {
	TimeMLP1W     []float64 `json:"time_mlp1_w"`
	TimeMLP1B     []float64 `json:"time_mlp1_b"`
	TimeMLP2W     []float64 `json:"time_mlp2_w"`
	TimeMLP2B     []float64 `json:"time_mlp2_b"`
	TimeNormScale []float64 `json:"time_norm_scale"`
	TimeNormBias  []float64 `json:"time_norm_bias"`
	FeatMLP1W     []float64 `json:"feat_mlp1_w,omitempty"`
	FeatMLP1B     []float64 `json:"feat_mlp1_b,omitempty"`
	FeatMLP2W     []float64 `json:"feat_mlp2_w,omitempty"`
	FeatMLP2B     []float64 `json:"feat_mlp2_b,omitempty"`
	FeatNormScale []float64 `json:"feat_norm_scale,omitempty"`
	FeatNormBias  []float64 `json:"feat_norm_bias,omitempty"`
}

// SaveWeights writes the model weights to a JSON file.
func (m *TTM) SaveWeights(path string) error {
	p := m.extractParamsF64()
	w := ttmWeights{
		Config:    m.config,
		PatchEmbW: p.patchEmbW,
		PatchEmbB: p.patchEmbB,
		HeadW:     p.headW,
		HeadB:     p.headB,
		NormMeans: m.normMeans,
		NormStds:  m.normStds,
	}
	for _, e := range p.encoder {
		w.Encoder = append(w.Encoder, mixerLayerToJSON(e))
	}
	for _, d := range p.decoder {
		w.Decoder = append(w.Decoder, mixerLayerToJSON(d))
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("ttm: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

func mixerLayerToJSON(l ttmMixerLayerF64) ttmMixerLayerJSON {
	return ttmMixerLayerJSON{
		TimeMLP1W:     l.timeMLP1W,
		TimeMLP1B:     l.timeMLP1B,
		TimeMLP2W:     l.timeMLP2W,
		TimeMLP2B:     l.timeMLP2B,
		TimeNormScale: l.timeNormScale,
		TimeNormBias:  l.timeNormBias,
		FeatMLP1W:     l.featMLP1W,
		FeatMLP1B:     l.featMLP1B,
		FeatMLP2W:     l.featMLP2W,
		FeatMLP2B:     l.featMLP2B,
		FeatNormScale: l.featNormScale,
		FeatNormBias:  l.featNormBias,
	}
}

func jsonToMixerLayer(j ttmMixerLayerJSON) ttmMixerLayerF64 {
	return ttmMixerLayerF64{
		timeMLP1W:     j.TimeMLP1W,
		timeMLP1B:     j.TimeMLP1B,
		timeMLP2W:     j.TimeMLP2W,
		timeMLP2B:     j.TimeMLP2B,
		timeNormScale: j.TimeNormScale,
		timeNormBias:  j.TimeNormBias,
		featMLP1W:     j.FeatMLP1W,
		featMLP1B:     j.FeatMLP1B,
		featMLP2W:     j.FeatMLP2W,
		featMLP2B:     j.FeatMLP2B,
		featNormScale: j.FeatNormScale,
		featNormBias:  j.FeatNormBias,
	}
}

// loadWeights reads model weights from a JSON file.
func (m *TTM) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w ttmWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if len(w.Encoder) != len(m.encoder) {
		return fmt.Errorf("ttm: encoder layer count mismatch: file=%d, model=%d", len(w.Encoder), len(m.encoder))
	}
	if len(w.Decoder) != len(m.decoder) {
		return fmt.Errorf("ttm: decoder layer count mismatch: file=%d, model=%d", len(w.Decoder), len(m.decoder))
	}

	p := &ttmParamsF64{
		patchEmbW: w.PatchEmbW,
		patchEmbB: w.PatchEmbB,
		headW:     w.HeadW,
		headB:     w.HeadB,
	}
	p.encoder = make([]ttmMixerLayerF64, len(w.Encoder))
	for i, e := range w.Encoder {
		p.encoder[i] = jsonToMixerLayer(e)
	}
	p.decoder = make([]ttmMixerLayerF64, len(w.Decoder))
	for i, d := range w.Decoder {
		p.decoder[i] = jsonToMixerLayer(d)
	}
	m.writeBackF32(p)
	m.normMeans = w.NormMeans
	m.normStds = w.NormStds
	return nil
}

