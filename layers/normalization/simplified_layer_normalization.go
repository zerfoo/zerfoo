// package normalization provides normalization layers for the Zerfoo model.
package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SimplifiedLayerNormalization applies layer normalization without centering the mean.
// It is equivalent to RMSNorm with a learnable gain parameter.
type SimplifiedLayerNormalization[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	gain    *graph.Parameter[T]
	epsilon T

	// Cached values for backward pass
	normalizedInput *tensor.TensorNumeric[T]
	invStdDev       *tensor.TensorNumeric[T]
	inputShape      []int
}

// NewSimplifiedLayerNormalization creates a new SimplifiedLayerNormalization layer.
func NewSimplifiedLayerNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	gain *tensor.TensorNumeric[T],
	epsilon T,
) (*SimplifiedLayerNormalization[T], error) {
	gainParam, err := graph.NewParameter[T]("gain", gain, tensor.New[T])
	if err != nil {
		return nil, err
	}
	return &SimplifiedLayerNormalization[T]{
		engine:  engine,
		ops:     ops,
		gain:    gainParam,
		epsilon: epsilon,
	}, nil
}

// Forward applies the forward pass of the SimplifiedLayerNormalization layer.
func (sln *SimplifiedLayerNormalization[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimplifiedLayerNormalization expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	sln.inputShape = input.Shape()

	// 1. Square the input
	squared, err := sln.engine.Mul(ctx, input, input)
	if err != nil {
		return nil, err
	}

	// 2. Mean of squares along the last dimension
	meanSquared, err := sln.engine.ReduceMean(ctx, squared, -1, true)
	if err != nil {
		return nil, err
	}

	// 3. Add epsilon
	withEpsilon, err := sln.engine.AddScalar(ctx, meanSquared, sln.epsilon)
	if err != nil {
		return nil, err
	}

	// 4. Inverse square root
	invStdDev, err := sln.engine.Rsqrt(ctx, withEpsilon)
	if err != nil {
		return nil, err
	}
	sln.invStdDev = invStdDev // Cache for backward pass

	// 5. Normalize
	normalized, err := sln.engine.Mul(ctx, input, invStdDev)
	if err != nil {
		return nil, err
	}
	sln.normalizedInput = normalized // Cache for backward pass

	// 6. Apply gain
	output, err := sln.engine.Mul(ctx, normalized, sln.gain.Value)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward applies the backward pass of the SimplifiedLayerNormalization layer.
func (sln *SimplifiedLayerNormalization[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("backward pass not implemented")
}

// Parameters returns the learnable parameters of the layer.
func (sln *SimplifiedLayerNormalization[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{sln.gain}
}

// OutputShape returns the output shape of the layer.
func (sln *SimplifiedLayerNormalization[T]) OutputShape() []int {
	return sln.inputShape
}
