// Package normalization provides normalization layers for the Zerfoo model.
package normalization

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SkipSimplifiedLayerNormalization applies SimplifiedLayerNormalization and adds a residual connection.
type SkipSimplifiedLayerNormalization[T tensor.Numeric] struct {
	normLayer *SimplifiedLayerNormalization[T]
	engine    compute.Engine[T]
}

// NewSkipSimplifiedLayerNormalization creates a new SkipSimplifiedLayerNormalization layer.
func NewSkipSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	gain *tensor.TensorNumeric[T],
	epsilon T,
) (*SkipSimplifiedLayerNormalization[T], error) {
	normLayer, err := NewSimplifiedLayerNormalization[T](engine, ops, gain, epsilon)
	if err != nil {
		return nil, err
	}

	return &SkipSimplifiedLayerNormalization[T]{
		normLayer: normLayer,
		engine:    engine,
	}, nil
}

// Forward applies the forward pass of the SkipSimplifiedLayerNormalization layer.
func (sln *SkipSimplifiedLayerNormalization[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	normalized, err := sln.normLayer.Forward(ctx, inputs...)
	if err != nil {
		return nil, err
	}

	return sln.engine.Add(ctx, inputs[0], normalized)
}

// Backward applies the backward pass of the SkipSimplifiedLayerNormalization layer.
func (sln *SkipSimplifiedLayerNormalization[T]) Backward(_ context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Parameters returns the learnable parameters of the layer.
func (sln *SkipSimplifiedLayerNormalization[T]) Parameters() []*graph.Parameter[T] {
	return sln.normLayer.Parameters()
}

// OutputShape returns the output shape of the layer.
func (sln *SkipSimplifiedLayerNormalization[T]) OutputShape() []int {
	return sln.normLayer.OutputShape()
}

// OpType returns the operation type of the SkipSimplifiedLayerNormalization layer.
func (sln *SkipSimplifiedLayerNormalization[T]) OpType() string {
	return "SkipSimplifiedLayerNormalization"
}

// Attributes returns the attributes of the underlying normalization layer.
func (sln *SkipSimplifiedLayerNormalization[T]) Attributes() map[string]interface{} {
	return sln.normLayer.Attributes()
}
