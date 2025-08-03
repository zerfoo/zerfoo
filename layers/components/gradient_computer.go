package components

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// LinearGradientComputer handles gradient computation for linear layers.
type LinearGradientComputer[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewLinearGradientComputer creates a new linear gradient computer.
func NewLinearGradientComputer[T tensor.Numeric](engine compute.Engine[T]) *LinearGradientComputer[T] {
	return &LinearGradientComputer[T]{engine: engine}
}

// ComputeWeightGradient computes the gradient with respect to weights.
// Formula: weight_gradient = input^T * output_gradient
func (g *LinearGradientComputer[T]) ComputeWeightGradient(ctx context.Context, input, outputGradient *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// Transpose input: input^T
	transposedInput, err := g.engine.Transpose(ctx, input, []int{1, 0})
	if err != nil {
		return nil, err
	}

	// Multiply: input^T * output_gradient
	weightGradient, err := g.engine.MatMul(ctx, transposedInput, outputGradient)
	if err != nil {
		return nil, err
	}

	return weightGradient, nil
}

// ComputeInputGradient computes the gradient with respect to input.
// Formula: input_gradient = output_gradient * weights^T
func (g *LinearGradientComputer[T]) ComputeInputGradient(ctx context.Context, weights, outputGradient *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// Transpose weights: weights^T
	transposedWeights, err := g.engine.Transpose(ctx, weights, []int{1, 0})
	if err != nil {
		return nil, err
	}

	// Multiply: output_gradient * weights^T
	inputGradient, err := g.engine.MatMul(ctx, outputGradient, transposedWeights)
	if err != nil {
		return nil, err
	}

	return inputGradient, nil
}

// ComputeBothGradients computes both weight and input gradients in one call.
// This can be more efficient when both gradients are needed.
func (g *LinearGradientComputer[T]) ComputeBothGradients(ctx context.Context, input, weights, outputGradient *tensor.Tensor[T]) (*tensor.Tensor[T], *tensor.Tensor[T], error) {
	// Compute weight gradient: input^T * output_gradient
	weightGradient, err := g.ComputeWeightGradient(ctx, input, outputGradient)
	if err != nil {
		return nil, nil, err
	}

	// Compute input gradient: output_gradient * weights^T
	inputGradient, err := g.ComputeInputGradient(ctx, weights, outputGradient)
	if err != nil {
		return nil, nil, err
	}

	return weightGradient, inputGradient, nil
}
