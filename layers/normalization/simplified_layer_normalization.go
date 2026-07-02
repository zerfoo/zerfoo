// Package normalization provides normalization layers for the Zerfoo model.
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

// SimplifiedLayerNormalization implements a simplified version of layer normalization.
type SimplifiedLayerNormalization[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	gain       *graph.Parameter[T]
	epsilon    T
	inputShape []int
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

	res, err := rmsNormalize(ctx, sln.engine, input, sln.gain.Value, sln.epsilon)
	if err != nil {
		return nil, err
	}

	return res.output, nil
}

// Backward applies the backward pass of the SimplifiedLayerNormalization layer.
func (sln *SimplifiedLayerNormalization[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimplifiedLayerNormalization expects 1 input, got %d", len(inputs))
	}

	input := inputs[0]

	// Recompute the RMS statistics from the live input instead of reading
	// tensors cached during Forward (ztensor ADR 006; zerfoo#842 bug class).
	invStdDev, normalizedInput, err := rmsRecomputeStats(ctx, sln.engine, input, sln.epsilon)
	if err != nil {
		return nil, err
	}

	// dGain = sum(dOut * normalized) reduced to gain shape
	dGainFull, err := sln.engine.Mul(ctx, outputGradient, normalizedInput)
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
	term1, err := sln.engine.Mul(ctx, dNormalized, invStdDev)
	if err != nil {
		return nil, err
	}

	// rmsCubed = invStdDev^3
	rmsSq, err := sln.engine.Mul(ctx, invStdDev, invStdDev)
	if err != nil {
		return nil, err
	}

	rmsCubed, err := sln.engine.Mul(ctx, rmsSq, invStdDev)
	if err != nil {
		return nil, err
	}

	// sumDNormX = ReduceSum(dNormalized * input, axis=-1, keepDims=true)
	dNormX, err := sln.engine.Mul(ctx, dNormalized, input)
	if err != nil {
		return nil, err
	}

	sumDNormX, err := sln.engine.ReduceSum(ctx, dNormX, len(dNormX.Shape())-1, true)
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

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*SimplifiedLayerNormalization[float32])(nil)
