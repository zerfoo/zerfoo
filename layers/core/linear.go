package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/components"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Linear performs a linear transformation: output = input * weights.
// Uses component-based architecture for better modularity and testability.
type Linear[T tensor.Numeric] struct {
	multiplier       *components.MatrixMultiplier[T]
	gradientComputer *components.LinearGradientComputer[T]
	weights          *graph.Parameter[T]
	lastInput        *tensor.Tensor[T]
	outputShape      []int
}

// NewLinear creates a new Linear layer with Xavier initialization (default).
func NewLinear[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int) (*Linear[T], error) {
	return NewLinearWithXavier(name, engine, ops, inputSize, outputSize)
}

// NewLinearWithXavier creates a Linear layer with Xavier initialization.
func NewLinearWithXavier[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int) (*Linear[T], error) {
	return NewLinearWithInitializer(name, engine, ops, inputSize, outputSize, components.NewXavierInitializer(ops))
}

// NewLinearWithHe creates a Linear layer with He initialization.
func NewLinearWithHe[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int) (*Linear[T], error) {
	return NewLinearWithInitializer(name, engine, ops, inputSize, outputSize, components.NewHeInitializer(ops))
}

// NewLinearWithUniform creates a Linear layer with uniform initialization.
func NewLinearWithUniform[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int, scale float64) (*Linear[T], error) {
	return NewLinearWithInitializer(name, engine, ops, inputSize, outputSize, components.NewUniformInitializer(ops, scale))
}

// NewLinearWithInitializer creates a Linear layer with a custom weight initializer.
func NewLinearWithInitializer[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int, initializer components.WeightInitializer[T]) (*Linear[T], error) {
	return NewLinearWithFactories(name, engine, ops, inputSize, outputSize, initializer, tensor.New[T], graph.NewParameter[T])
}

// NewLinearWithFactories creates a new Linear layer with custom tensor and parameter creation functions.
func NewLinearWithFactories[T tensor.Numeric](name string, engine compute.Engine[T], _ numeric.Arithmetic[T], inputSize, outputSize int, initializer components.WeightInitializer[T], newTensor func([]int, []T) (*tensor.Tensor[T], error), newParameter func(string, *tensor.Tensor[T], func([]int, []T) (*tensor.Tensor[T], error)) (*graph.Parameter[T], error)) (*Linear[T], error) {
	if name == "" {
		return nil, errors.New("layer name cannot be empty")
	}

	// Initialize weights using the provided initializer
	weightsData, err := initializer.Initialize(inputSize, outputSize)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize weights: %w", err)
	}

	weights, err := newTensor([]int{inputSize, outputSize}, weightsData)
	if err != nil {
		return nil, fmt.Errorf("failed to create weights tensor: %w", err)
	}

	weightsParam, err := newParameter(name+"_weights", weights, newTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to create weights parameter: %w", err)
	}

	return &Linear[T]{
		multiplier:       components.NewMatrixMultiplier(engine),
		gradientComputer: components.NewLinearGradientComputer(engine),
		weights:          weightsParam,
		outputShape:      []int{1, outputSize}, // Assuming batch size of 1 for now
	}, nil
}

// OutputShape returns the output shape of the Linear layer.
func (l *Linear[T]) OutputShape() []int {
	return l.outputShape
}

// Forward performs the forward pass: output = input * weights.
func (l *Linear[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	l.lastInput = inputs[0]

	output, err := l.multiplier.Multiply(ctx, l.lastInput, l.weights.Value)
	if err != nil {
		return nil, fmt.Errorf("forward pass failed: %w", err)
	}

	l.outputShape = output.Shape()

	return output, nil
}

// Backward computes the gradients using the gradient computer component.
func (l *Linear[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// Compute both gradients efficiently
	weightsGrad, inputGrad, err := l.gradientComputer.ComputeBothGradients(
		ctx, l.lastInput, l.weights.Value, outputGradient)
	if err != nil {
		return nil, err
	}

	l.weights.Gradient = weightsGrad

	return []*tensor.Tensor[T]{inputGrad}, nil
}

// Parameters returns the parameters of the Linear layer.
func (l *Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.weights}
}

// SetName sets the name of the Linear layer.
func (l *Linear[T]) SetName(name string) {
	l.weights.Name = name + "_weights"
}
