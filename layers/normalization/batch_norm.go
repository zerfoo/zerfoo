package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// BatchNormalization implements inference-mode batch normalization.
// Forward expects 5 inputs: X, scale, B, mean, var (in that order).
// Formula per element at channel c:
//
//	y = scale[c] * (X - mean[c]) / sqrt(var[c] + epsilon) + B[c]
//
// scale, B, mean, and var must have shape [C] matching X's channel dimension.
// All operations use Engine primitives so they appear in the ExecutionPlan
// instruction tape.
type BatchNormalization[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	epsilon     T
	outputShape []int
}

// NewBatchNormalization creates a BatchNormalization layer.
func NewBatchNormalization[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], epsilon T) *BatchNormalization[T] {
	return &BatchNormalization[T]{engine: engine, ops: ops, epsilon: epsilon}
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

	b.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (b *BatchNormalization[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "BatchNormalization".
func (b *BatchNormalization[T]) OpType() string { return "BatchNormalization" }

// Attributes returns the layer configuration.
func (b *BatchNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": b.epsilon}
}

// OutputShape returns the output shape (populated after Forward).
func (b *BatchNormalization[T]) OutputShape() []int { return b.outputShape }

// Parameters returns nil.
func (b *BatchNormalization[T]) Parameters() []*graph.Parameter[T] { return nil }

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
