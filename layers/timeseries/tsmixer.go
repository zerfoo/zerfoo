package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// TSMixerBlock implements a single TSMixer block with time-mixing and
// feature-mixing MLPs. This is the backbone layer used by IBM Granite
// TinyTimeMixer (TTM) for time series forecasting.
//
// Each block contains:
//   - Time-mixing MLP: mixes information across the patch/time dimension
//   - Feature-mixing MLP: mixes information across the feature/channel dimension
//   - LayerNorm after each mixing step
//   - Residual connections around each mixing step
//
// Time-mixing transposes the input so the MLP operates along the time axis,
// while feature-mixing applies the MLP along the last (feature) axis directly.
type TSMixerBlock[T tensor.Float] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Time-mixing MLP (operates across patch/time dimension).
	timeMLP1 *core.Linear[T] // [numPatches, numPatches]
	timeMLP2 *core.Linear[T] // [numPatches, numPatches]
	timeNorm *normalization.LayerNormalization[T]
	timeGelu *activations.Gelu[T]

	// Feature-mixing MLP (operates across feature/channel dimension).
	featMLP1 *core.Linear[T] // [dModel, dModel*expansion]
	featMLP2 *core.Linear[T] // [dModel*expansion, dModel]
	featNorm *normalization.LayerNormalization[T]
	featGelu *activations.Gelu[T]

	channelMixing bool // if false, skip feature-mixing (channel-independent mode)

	numPatches int
	dModel     int
}

// NewTSMixerBlock creates a new TSMixer block.
//
// Parameters:
//   - engine: the compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - numPatches: the number of time patches (time dimension size)
//   - dModel: the model/feature dimension size
//   - expansion: expansion factor for the feature-mixing MLP hidden dim
//   - channelMixing: if true, include the feature-mixing MLP; if false, channel-independent mode
func NewTSMixerBlock[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numPatches, dModel, expansion int,
	channelMixing bool,
) (*TSMixerBlock[T], error) {
	if numPatches <= 0 {
		return nil, fmt.Errorf("numPatches must be positive, got %d", numPatches)
	}
	if dModel <= 0 {
		return nil, fmt.Errorf("dModel must be positive, got %d", dModel)
	}
	if expansion <= 0 {
		return nil, fmt.Errorf("expansion must be positive, got %d", expansion)
	}

	// Time-mixing MLP: operates on transposed [batch, dModel, numPatches],
	// so the linear layers map numPatches -> numPatches.
	timeMLP1, err := core.NewLinear[T]("tsmixer_time_mlp1", engine, ops, numPatches, numPatches)
	if err != nil {
		return nil, fmt.Errorf("create time MLP1: %w", err)
	}
	timeMLP2, err := core.NewLinear[T]("tsmixer_time_mlp2", engine, ops, numPatches, numPatches)
	if err != nil {
		return nil, fmt.Errorf("create time MLP2: %w", err)
	}

	// Time-mixing LayerNorm normalizes over the feature dimension (dModel).
	timeNorm, err := normalization.NewLayerNormalization[T](engine, dModel)
	if err != nil {
		return nil, fmt.Errorf("create time norm: %w", err)
	}

	block := &TSMixerBlock[T]{
		engine:        engine,
		ops:           ops,
		timeMLP1:      timeMLP1,
		timeMLP2:      timeMLP2,
		timeNorm:      timeNorm,
		timeGelu:      activations.NewGelu[T](engine, ops),
		channelMixing: channelMixing,
		numPatches:    numPatches,
		dModel:        dModel,
	}

	if channelMixing {
		hiddenDim := dModel * expansion

		featMLP1, err := core.NewLinear[T]("tsmixer_feat_mlp1", engine, ops, dModel, hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("create feat MLP1: %w", err)
		}
		featMLP2, err := core.NewLinear[T]("tsmixer_feat_mlp2", engine, ops, hiddenDim, dModel)
		if err != nil {
			return nil, fmt.Errorf("create feat MLP2: %w", err)
		}
		featNorm, err := normalization.NewLayerNormalization[T](engine, dModel)
		if err != nil {
			return nil, fmt.Errorf("create feat norm: %w", err)
		}

		block.featMLP1 = featMLP1
		block.featMLP2 = featMLP2
		block.featNorm = featNorm
		block.featGelu = activations.NewGelu[T](engine, ops)
	}

	return block, nil
}

// OpType returns the operation type of the layer.
func (b *TSMixerBlock[T]) OpType() string {
	return "TSMixerBlock"
}

// Attributes returns the attributes of the layer.
func (b *TSMixerBlock[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_patches":    b.numPatches,
		"d_model":        b.dModel,
		"channel_mixing": b.channelMixing,
	}
}

// OutputShape returns the output shape of the layer.
func (b *TSMixerBlock[T]) OutputShape() []int {
	return []int{-1, b.numPatches, b.dModel}
}

// Forward computes the forward pass of the TSMixer block.
//
// Input shape: [batch, numPatches, dModel]
// Output shape: [batch, numPatches, dModel]
//
// Steps:
//  1. Time-mixing: LayerNorm -> transpose -> MLP(GELU) -> transpose -> residual add
//  2. Feature-mixing (if enabled): LayerNorm -> MLP(GELU) -> residual add
func (b *TSMixerBlock[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TSMixerBlock requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("TSMixerBlock input must be 3D [batch, num_patches, d_model], got shape %v", shape)
	}

	// --- Time-mixing ---
	// residual = input
	residual := input

	// x = timeNorm(input)
	x, err := b.timeNorm.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("time norm: %w", err)
	}

	// x = transpose(x, [0, 2, 1]) — swap patch and feature dims
	x, err = b.engine.Transpose(ctx, x, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("time transpose pre: %w", err)
	}

	// x = timeMLP2(gelu(timeMLP1(x)))
	x, err = b.timeMLP1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("time MLP1: %w", err)
	}
	x, err = b.timeGelu.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("time GELU: %w", err)
	}
	x, err = b.timeMLP2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("time MLP2: %w", err)
	}

	// x = transpose(x, [0, 2, 1]) — swap back
	x, err = b.engine.Transpose(ctx, x, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("time transpose post: %w", err)
	}

	// x = x + residual
	x, err = b.engine.Add(ctx, x, residual)
	if err != nil {
		return nil, fmt.Errorf("time residual add: %w", err)
	}

	// --- Feature-mixing (if enabled) ---
	if b.channelMixing {
		residual = x

		// x = featNorm(x)
		x, err = b.featNorm.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("feat norm: %w", err)
		}

		// x = featMLP2(gelu(featMLP1(x)))
		x, err = b.featMLP1.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("feat MLP1: %w", err)
		}
		x, err = b.featGelu.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("feat GELU: %w", err)
		}
		x, err = b.featMLP2.Forward(ctx, x)
		if err != nil {
			return nil, fmt.Errorf("feat MLP2: %w", err)
		}

		// x = x + residual
		x, err = b.engine.Add(ctx, x, residual)
		if err != nil {
			return nil, fmt.Errorf("feat residual add: %w", err)
		}
	}

	return x, nil
}

// Parameters returns all trainable parameters of the block.
func (b *TSMixerBlock[T]) Parameters() []*graph.Parameter[T] {
	params := b.timeMLP1.Parameters()
	params = append(params, b.timeMLP2.Parameters()...)
	params = append(params, b.timeNorm.Parameters()...)
	if b.channelMixing {
		params = append(params, b.featMLP1.Parameters()...)
		params = append(params, b.featMLP2.Parameters()...)
		params = append(params, b.featNorm.Parameters()...)
	}
	return params
}
