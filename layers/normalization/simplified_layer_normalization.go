// package normalization provides normalization layers for the Zerfoo model.
package normalization

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SimplifiedLayerNormalization implements a simplified version of layer normalization.
type SimplifiedLayerNormalization[T tensor.Numeric] struct {
	engine          compute.Engine[T]
	ops             numeric.Arithmetic[T]
	gain            *graph.Parameter[T]
	epsilon         T
	inputShape      []int
	invStdDev       *tensor.TensorNumeric[T]
	normalizedInput *tensor.TensorNumeric[T]
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
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimplifiedLayerNormalization expects 1 input, got %d", len(inputs))
	}

	input := inputs[0]

	// Ensure forward caches are present
	if sln.invStdDev == nil || sln.normalizedInput == nil {
		return nil, errors.New("backward called before forward: missing cached tensors")
	}

	// dGain = sum(dOut * normalized) reduced to gain shape
	dGainFull, err := sln.engine.Mul(ctx, outputGradient, sln.normalizedInput)
	if err != nil {
		return nil, err
	}

	dGainReduced := dGainFull
	// Common case: gain has shape [D] (last dimension). Reduce over all other axes with keepDims=false.
	// Repeatedly reduce axis 0 until only the last dimension remains.
	for len(dGainReduced.Shape()) > 1 {
		dGainReduced, err = sln.engine.ReduceSum(ctx, dGainReduced, 0, false)
		if err != nil {
			return nil, err
		}
	}

	// Accumulate into parameter gradient
	sln.gain.Gradient, err = sln.engine.Add(ctx, sln.gain.Gradient, dGainReduced, sln.gain.Gradient)
	if err != nil {
		return nil, err
	}

	// dInput computation
	// dNormalized = dOut * gain
	dNormalized, err := sln.engine.Mul(ctx, outputGradient, sln.gain.Value)
	if err != nil {
		return nil, err
	}

	// term1 = dNormalized * invStdDev
	term1, err := sln.engine.Mul(ctx, dNormalized, sln.invStdDev)
	if err != nil {
		return nil, err
	}

	// rmsCubed = invStdDev^3
	rmsSq, err := sln.engine.Mul(ctx, sln.invStdDev, sln.invStdDev)
	if err != nil {
		return nil, err
	}
	rmsCubed, err := sln.engine.Mul(ctx, rmsSq, sln.invStdDev)
	if err != nil {
		return nil, err
	}

	// sumDNormX = ReduceSum(dNormalized * input, axis=-1, keepDims=true)
	dNormX, err := sln.engine.Mul(ctx, dNormalized, input)
	if err != nil {
		return nil, err
	}
	sumDNormX, err := sln.engine.ReduceSum(ctx, dNormX, -1, true)
	if err != nil {
		return nil, err
	}

	// invN = 1/N where N is last dimension size
	lastDim := input.Shape()[len(input.Shape())-1]
	invN, err := tensor.New[T]([]int{1}, []T{sln.ops.FromFloat64(1.0 / float64(lastDim))})
	if err != nil {
		return nil, err
	}

	// term2 = input * sumDNormX * rmsCubed * invN
	term2, err := sln.engine.Mul(ctx, input, sumDNormX)
	if err != nil {
		return nil, err
	}
	term2, err = sln.engine.Mul(ctx, term2, rmsCubed)
	if err != nil {
		return nil, err
	}
	term2, err = sln.engine.Mul(ctx, term2, invN)
	if err != nil {
		return nil, err
	}

	dInput, err := sln.engine.Sub(ctx, term1, term2)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// Parameters returns the learnable parameters of the layer.
func (sln *SimplifiedLayerNormalization[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{sln.gain}
}

// OutputShape returns the output shape of the layer.
func (sln *SimplifiedLayerNormalization[T]) OutputShape() []int {
	return sln.inputShape
}

// OpType returns the operation type of the SimplifiedLayerNormalization layer.
func (sln *SimplifiedLayerNormalization[T]) OpType() string {
	return "SimplifiedLayerNormalization"
}

// Attributes returns the attributes of the SimplifiedLayerNormalization layer.
func (sln *SimplifiedLayerNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": sln.epsilon}
}
