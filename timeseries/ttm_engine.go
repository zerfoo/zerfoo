package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/core"
	tslayers "github.com/zerfoo/zerfoo/layers/timeseries"
)

// TTMEngine implements the TTM forward pass using typed tensor operations
// and TSMixerBlock layers for inference. This is the compute.Engine-based
// inference path, separate from the float64 analytical training path in ttm.go.
type TTMEngine[T tensor.Float] struct {
	encoder      []*tslayers.TSMixerBlock[T]
	decoder      []*tslayers.TSMixerBlock[T]
	patchEmbed   *core.Linear[T]
	forecastHead *core.Linear[T]
	engine       compute.Engine[T]
	ops          numeric.Arithmetic[T]
	config       TTMTrainConfig
}

// NewTTMEngine creates a new TTMEngine with initialized layers.
func NewTTMEngine[T tensor.Float](
	config TTMTrainConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*TTMEngine[T], error) {
	if config.ContextLen <= 0 {
		return nil, fmt.Errorf("ttm engine: ContextLen must be positive, got %d", config.ContextLen)
	}
	if config.ForecastLen <= 0 {
		return nil, fmt.Errorf("ttm engine: ForecastLen must be positive, got %d", config.ForecastLen)
	}
	if config.PatchLen <= 0 {
		return nil, fmt.Errorf("ttm engine: PatchLen must be positive, got %d", config.PatchLen)
	}
	if config.ContextLen%config.PatchLen != 0 {
		return nil, fmt.Errorf("ttm engine: ContextLen (%d) must be divisible by PatchLen (%d)", config.ContextLen, config.PatchLen)
	}
	if config.ForecastLen%config.PatchLen != 0 {
		return nil, fmt.Errorf("ttm engine: ForecastLen (%d) must be divisible by PatchLen (%d)", config.ForecastLen, config.PatchLen)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("ttm engine: DModel must be positive, got %d", config.DModel)
	}
	if config.NumMixerLayers <= 0 {
		return nil, fmt.Errorf("ttm engine: NumMixerLayers must be positive, got %d", config.NumMixerLayers)
	}

	numPatches := config.NumPatches()
	forecastPatches := config.ForecastPatches()
	expansion := 2

	// Patch embedding: patchLen -> dModel.
	patchEmbed, err := core.NewLinear[T]("ttm_patch_embed", engine, ops, config.PatchLen, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("ttm engine: patch embed: %w", err)
	}

	// Encoder TSMixer blocks.
	encoderBlocks := make([]*tslayers.TSMixerBlock[T], config.NumMixerLayers)
	for i := range config.NumMixerLayers {
		encoderBlocks[i], err = tslayers.NewTSMixerBlock[T](
			engine, ops, numPatches, config.DModel, expansion, config.ChannelMixing,
		)
		if err != nil {
			return nil, fmt.Errorf("ttm engine: encoder block %d: %w", i, err)
		}
	}

	// Decoder TSMixer blocks.
	decoderBlocks := make([]*tslayers.TSMixerBlock[T], config.NumMixerLayers)
	for i := range config.NumMixerLayers {
		decoderBlocks[i], err = tslayers.NewTSMixerBlock[T](
			engine, ops, forecastPatches, config.DModel, expansion, config.ChannelMixing,
		)
		if err != nil {
			return nil, fmt.Errorf("ttm engine: decoder block %d: %w", i, err)
		}
	}

	// Forecast head: forecastPatches * dModel -> forecastLen.
	forecastHead, err := core.NewLinear[T]("ttm_forecast_head", engine, ops, forecastPatches*config.DModel, config.ForecastLen)
	if err != nil {
		return nil, fmt.Errorf("ttm engine: forecast head: %w", err)
	}

	return &TTMEngine[T]{
		encoder:      encoderBlocks,
		decoder:      decoderBlocks,
		patchEmbed:   patchEmbed,
		forecastHead: forecastHead,
		engine:       engine,
		ops:          ops,
		config:       config,
	}, nil
}

// Forward runs the TTM forward pass through encoder and decoder TSMixer blocks.
//
// Input shape: [batch, numPatches, patchLen]
// Output shape: [batch, forecastLen]
//
// The input should already be patched. If you have raw time series of shape
// [batch, contextLen], use ExtractPatches first.
func (e *TTMEngine[T]) Forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("ttm engine: input must be 3D [batch, numPatches, patchLen], got shape %v", shape)
	}

	batch := shape[0]
	numPatches := shape[1]
	forecastPatches := e.config.ForecastPatches()

	// Patch embedding: [batch, numPatches, patchLen] -> [batch, numPatches, dModel].
	// Reshape to [batch*numPatches, patchLen] for linear, then reshape back.
	flat, err := e.engine.Reshape(ctx, input, []int{batch * numPatches, e.config.PatchLen})
	if err != nil {
		return nil, fmt.Errorf("ttm engine: reshape for embed: %w", err)
	}
	embedded, err := e.patchEmbed.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("ttm engine: patch embed: %w", err)
	}
	x, err := e.engine.Reshape(ctx, embedded, []int{batch, numPatches, e.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("ttm engine: reshape embedded: %w", err)
	}

	// Encoder blocks.
	for i, block := range e.encoder {
		x, err = block.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("ttm engine: encoder block %d: %w", i, err)
		}
	}

	// Slice to last forecastPatches patches for decoder input.
	if numPatches > forecastPatches {
		data := x.Data()
		n := batch
		decData := make([]T, n*forecastPatches*e.config.DModel)
		for i := 0; i < n; i++ {
			srcOff := i*numPatches*e.config.DModel + (numPatches-forecastPatches)*e.config.DModel
			dstOff := i * forecastPatches * e.config.DModel
			copy(decData[dstOff:dstOff+forecastPatches*e.config.DModel], data[srcOff:srcOff+forecastPatches*e.config.DModel])
		}
		x, err = tensor.New[T]([]int{n, forecastPatches, e.config.DModel}, decData)
		if err != nil {
			return nil, fmt.Errorf("ttm engine: slice encoder output: %w", err)
		}
	}

	// Decoder blocks.
	for i, block := range e.decoder {
		x, err = block.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("ttm engine: decoder block %d: %w", i, err)
		}
	}

	// Flatten: [batch, forecastPatches * dModel].
	x, err = e.engine.Reshape(ctx, x, []int{batch, forecastPatches * e.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("ttm engine: flatten: %w", err)
	}

	// Forecast head.
	x, err = e.forecastHead.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ttm engine: forecast head: %w", err)
	}

	return x, nil
}

// ExtractPatches converts raw time series into non-overlapping patches.
//
// Input shape: [batch, contextLen]
// Output shape: [batch, numPatches, patchLen]
func (e *TTMEngine[T]) ExtractPatches(_ context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("ttm engine: ExtractPatches input must be 2D [batch, contextLen], got shape %v", shape)
	}

	batch := shape[0]
	contextLen := shape[1]
	numPatches := e.config.NumPatches()

	if contextLen != e.config.ContextLen {
		return nil, fmt.Errorf("ttm engine: expected context length %d, got %d", e.config.ContextLen, contextLen)
	}

	data := input.Data()
	outData := make([]T, batch*numPatches*e.config.PatchLen)
	for i := 0; i < batch; i++ {
		for p := 0; p < numPatches; p++ {
			srcOff := i*contextLen + p*e.config.PatchLen
			dstOff := (i*numPatches + p) * e.config.PatchLen
			copy(outData[dstOff:dstOff+e.config.PatchLen], data[srcOff:srcOff+e.config.PatchLen])
		}
	}

	return tensor.New[T]([]int{batch, numPatches, e.config.PatchLen}, outData)
}

// EncoderParameters returns all trainable parameters from the encoder blocks.
func (e *TTMEngine[T]) EncoderParameters() []*tensor.TensorNumeric[T] {
	var params []*tensor.TensorNumeric[T]
	for _, block := range e.encoder {
		for _, p := range block.Parameters() {
			params = append(params, p.Value)
		}
	}
	return params
}

// DecoderParameters returns all trainable parameters from the decoder blocks.
func (e *TTMEngine[T]) DecoderParameters() []*tensor.TensorNumeric[T] {
	var params []*tensor.TensorNumeric[T]
	for _, block := range e.decoder {
		for _, p := range block.Parameters() {
			params = append(params, p.Value)
		}
	}
	return params
}
