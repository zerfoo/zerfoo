package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// BatchNormalization implements batch normalization with optional training support.
// Forward expects 5 inputs: X, scale, B, mean, var (in that order).
// Formula per element at channel c:
//
//	y = scale[c] * (X - mean[c]) / sqrt(var[c] + epsilon) + B[c]
//
// scale, B, mean, and var must have shape [C] matching X's channel dimension.
// All operations use Engine primitives so they appear in the ExecutionPlan
// instruction tape.
//
// When scale and bias are set as graph.Parameter (via NewBatchNormalizationWithParams),
// Backward() computes and accumulates gradients for scale, bias, and input X.
type BatchNormalization[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	epsilon     T
	outputShape []int

	// Learnable parameters (nil for inference-only mode).
	scale *graph.Parameter[T]
	bias  *graph.Parameter[T]

	// Cache for backward pass (populated by Forward). Backward only
	// receives X (not scale/mean/var), so these cannot be recomputed from
	// its live inputs; they are registered with the save-for-backward
	// contract (ztensor ADR 006) every Forward so arena-backed storage
	// stays pinned until Backward consumes them.
	normalized *tensor.TensorNumeric[T] // (X - mean) / sqrt(var + eps)
	std        *tensor.TensorNumeric[T] // sqrt(var + eps), broadcast shape [1,C,1,...]
	scaleBC    *tensor.TensorNumeric[T] // scale reshaped to [1,C,1,...] for broadcasting
	inputShape []int                    // shape of X from the last Forward call
	saver      graph.Saver[T]           // wired by graph Builder (graph.SaverAware); nil outside a Graph
}

// SetSaver implements graph.SaverAware.
func (b *BatchNormalization[T]) SetSaver(sv graph.Saver[T]) {
	b.saver = sv
}

// NewBatchNormalization creates an inference-only BatchNormalization layer.
func NewBatchNormalization[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], epsilon T) *BatchNormalization[T] {
	return &BatchNormalization[T]{engine: engine, ops: ops, epsilon: epsilon}
}

// NewBatchNormalizationWithParams creates a BatchNormalization layer with
// learnable scale and bias parameters for training.
func NewBatchNormalizationWithParams[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], epsilon T, scale, bias *graph.Parameter[T]) *BatchNormalization[T] {
	return &BatchNormalization[T]{
		engine:  engine,
		ops:     ops,
		epsilon: epsilon,
		scale:   scale,
		bias:    bias,
	}
}

// Forward computes batch normalization in inference mode.
// inputs: [X, scale, B, mean, var].
func (b *BatchNormalization[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 5 {
		return nil, fmt.Errorf("BatchNormalization requires 5 inputs (X, scale, B, mean, var), got %d", len(inputs))
	}
	X, scale, bias, mean, variance := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

	xShape := X.Shape()
	if len(xShape) < 2 {
		return nil, fmt.Errorf("BatchNormalization: X must have at least rank 2, got shape %v", xShape)
	}

	c := xShape[1]

	// Validate parameter shapes
	for name, param := range map[string]*tensor.TensorNumeric[T]{"scale": scale, "B": bias, "mean": mean, "var": variance} {
		pShape := param.Shape()
		if len(pShape) != 1 || pShape[0] != c {
			return nil, fmt.Errorf("BatchNormalization: %s must have shape [%d], got %v", name, c, pShape)
		}
	}

	// Build broadcast shape [1, C, 1, 1, ...] matching X's rank.
	broadcastShape := make([]int, len(xShape))
	broadcastShape[0] = 1
	broadcastShape[1] = c
	for i := 2; i < len(xShape); i++ {
		broadcastShape[i] = 1
	}

	// Reshape [C] parameters to [1, C, 1, ...] for broadcasting with X.
	meanBC, err := b.engine.Reshape(ctx, mean, broadcastShape)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: reshape mean: %w", err)
	}
	varBC, err := b.engine.Reshape(ctx, variance, broadcastShape)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: reshape var: %w", err)
	}
	scaleBC, err := b.engine.Reshape(ctx, scale, broadcastShape)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: reshape scale: %w", err)
	}
	biasBC, err := b.engine.Reshape(ctx, bias, broadcastShape)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: reshape bias: %w", err)
	}

	// var + epsilon
	varEps, err := b.engine.AddScalar(ctx, varBC, b.epsilon)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: add epsilon: %w", err)
	}

	// sqrt(var + epsilon)
	std, err := b.engine.Sqrt(ctx, varEps)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: sqrt: %w", err)
	}

	// X - mean (broadcasts [1,C,1,...] to [N,C,H,W,...])
	centered, err := b.engine.Sub(ctx, X, meanBC)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: sub mean: %w", err)
	}

	// (X - mean) / sqrt(var + epsilon)
	normalized, err := b.engine.Div(ctx, centered, std)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: div std: %w", err)
	}

	// scale * normalized
	scaled, err := b.engine.Mul(ctx, normalized, scaleBC)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: mul scale: %w", err)
	}

	// scale * normalized + bias
	out, err := b.engine.Add(ctx, scaled, biasBC)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: add bias: %w", err)
	}

	// Cache intermediates for backward pass.
	b.normalized = normalized
	b.std = std
	b.scaleBC = scaleBC
	if b.saver != nil {
		b.saver.SaveForBackward(normalized, std, scaleBC)
	}
	b.inputShape = xShape
	b.outputShape = out.Shape()
	return out, nil
}

// Backward computes gradients for scale, bias, and input X.
//
// For inference-mode BatchNorm (pre-computed mean/var):
//
//	dBias  = sum(dOut, dims except channel)
//	dScale = sum(dOut * normalized, dims except channel)
//	dX     = dOut * scale / sqrt(var + eps)
//
// inputs must contain exactly one tensor (X, matching the Forward call).
// Forward must have been called first to populate cached intermediates.
func (b *BatchNormalization[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("BatchNormalization: %w, expected 1 input for backward, got %d", graph.ErrInvalidInputCount, len(inputs))
	}
	if b.normalized == nil || b.std == nil || b.scaleBC == nil {
		return nil, fmt.Errorf("BatchNormalization: backward called before forward: missing cached tensors")
	}

	// --- dX = dOut * scale / sqrt(var + eps) ---
	// scaleBC and std are already in broadcast shape [1,C,1,...].
	scaledDOut, err := b.engine.Mul(ctx, dOut, b.scaleBC)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization backward: mul dOut*scale: %w", err)
	}
	dInput, err := b.engine.Div(ctx, scaledDOut, b.std)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization backward: div by std: %w", err)
	}

	// --- Parameter gradients (only when parameters are set) ---
	if b.scale != nil {
		// dScale = sum(dOut * normalized, all dims except channel dim=1)
		dScaleFull, err := b.engine.Mul(ctx, dOut, b.normalized)
		if err != nil {
			return nil, fmt.Errorf("BatchNormalization backward: mul dOut*normalized: %w", err)
		}
		dScale := dScaleFull
		ndim := len(dScale.Shape())
		// Reduce over all dims except dim 1 (channel).
		// Reduce from highest dim to lowest, skipping dim 1.
		for dim := ndim - 1; dim >= 0; dim-- {
			if dim == 1 {
				continue
			}
			dScale, err = b.engine.ReduceSum(ctx, dScale, dim, true)
			if err != nil {
				return nil, fmt.Errorf("BatchNormalization backward: reduce dScale dim %d: %w", dim, err)
			}
		}
		dScale, err = dScale.Reshape(b.scale.Value.Shape())
		if err != nil {
			return nil, fmt.Errorf("BatchNormalization backward: reshape dScale: %w", err)
		}

		if b.scale.Gradient == nil {
			b.scale.Gradient = dScale
		} else {
			b.scale.Gradient, err = b.engine.Add(ctx, b.scale.Gradient, dScale, b.scale.Gradient)
			if err != nil {
				return nil, fmt.Errorf("BatchNormalization backward: accumulate scale gradient: %w", err)
			}
		}
	}

	if b.bias != nil {
		// dBias = sum(dOut, all dims except channel dim=1)
		dBias := dOut
		ndim := len(dBias.Shape())
		for dim := ndim - 1; dim >= 0; dim-- {
			if dim == 1 {
				continue
			}
			dBias, err = b.engine.ReduceSum(ctx, dBias, dim, true)
			if err != nil {
				return nil, fmt.Errorf("BatchNormalization backward: reduce dBias dim %d: %w", dim, err)
			}
		}
		dBias, err = dBias.Reshape(b.bias.Value.Shape())
		if err != nil {
			return nil, fmt.Errorf("BatchNormalization backward: reshape dBias: %w", err)
		}

		if b.bias.Gradient == nil {
			b.bias.Gradient = dBias
		} else {
			b.bias.Gradient, err = b.engine.Add(ctx, b.bias.Gradient, dBias, b.bias.Gradient)
			if err != nil {
				return nil, fmt.Errorf("BatchNormalization backward: accumulate bias gradient: %w", err)
			}
		}
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// OpType returns "BatchNormalization".
func (b *BatchNormalization[T]) OpType() string { return "BatchNormalization" }

// Attributes returns the layer configuration.
func (b *BatchNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": b.epsilon}
}

// OutputShape returns the output shape (populated after Forward).
func (b *BatchNormalization[T]) OutputShape() []int { return b.outputShape }

// Parameters returns the learnable parameters (scale and bias) when set,
// or nil for inference-only mode.
func (b *BatchNormalization[T]) Parameters() []*graph.Parameter[T] {
	if b.scale == nil && b.bias == nil {
		return nil
	}
	var params []*graph.Parameter[T]
	if b.scale != nil {
		params = append(params, b.scale)
	}
	if b.bias != nil {
		params = append(params, b.bias)
	}
	return params
}

// BuildBatchNormalization constructs a BatchNormalization layer for the registry.
// Reads "epsilon" (float32 or float64) from attributes; defaults to 1e-5.
func BuildBatchNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	eps := 1e-5
	if v, ok := attributes["epsilon"]; ok {
		switch e := v.(type) {
		case float64:
			eps = e
		case float32:
			eps = float64(e)
		}
	}
	return NewBatchNormalization(engine, ops, ops.FromFloat64(eps)), nil
}

// Statically assert that BatchNormalization implements graph.Node.
var _ graph.Node[float32] = (*BatchNormalization[float32])(nil)

// Statically assert that BatchNormalization participates in the save-for-backward contract.
var _ graph.SaverAware[float32] = (*BatchNormalization[float32])(nil)
