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
	lastInput        *tensor.TensorNumeric[T]
	outputShape      []int
}

// LinearOptions holds configuration options for the Linear layer.
type LinearOptions[T tensor.Numeric] struct {
	Initializer components.WeightInitializer[T]
}

// LinearOption is a function that applies an option to LinearOptions.
type LinearOption[T tensor.Numeric] func(*LinearOptions[T])

// WithInitializer sets a custom weight initializer for the Linear layer.
func WithInitializer[T tensor.Numeric](initializer components.WeightInitializer[T]) LinearOption[T] {
	return func(o *LinearOptions[T]) {
		o.Initializer = initializer
	}
}

// WithXavier is an option to use Xavier weight initialization.
func WithXavier[T tensor.Numeric](ops numeric.Arithmetic[T]) LinearOption[T] {
	return func(o *LinearOptions[T]) {
		o.Initializer = components.NewXavierInitializer(ops)
	}
}

// WithHe is an option to use He weight initialization.
func WithHe[T tensor.Numeric](ops numeric.Arithmetic[T]) LinearOption[T] {
	return func(o *LinearOptions[T]) {
		o.Initializer = components.NewHeInitializer(ops)
	}
}

// WithUniform is an option to use Uniform weight initialization.
func WithUniform[T tensor.Numeric](ops numeric.Arithmetic[T], scale float64) LinearOption[T] {
	return func(o *LinearOptions[T]) {
		o.Initializer = components.NewUniformInitializer(ops, components.WithScale[T](scale))
	}
}

// NewLinear creates a new Linear layer.
func NewLinear[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputSize, outputSize int,
	opts ...LinearOption[T],
) (*Linear[T], error) {
	// Default options
	options := &LinearOptions[T]{
		Initializer: components.NewXavierInitializer(ops),
	}
	for _, opt := range opts {
		opt(options)
	}

	return NewLinearWithFactories(name, engine, ops, inputSize, outputSize, options.Initializer, tensor.New[T], graph.NewParameter[T])
}

// NewLinearWithFactories creates a new Linear layer with custom tensor and parameter creation functions.
func NewLinearWithFactories[T tensor.Numeric](
	name string, engine compute.Engine[T], _ numeric.Arithmetic[T], inputSize, outputSize int,
	initializer components.WeightInitializer[T],
	newTensor func([]int, []T) (*tensor.TensorNumeric[T], error),
	newParameter func(string, *tensor.TensorNumeric[T], func([]int, []T) (*tensor.TensorNumeric[T], error)) (*graph.Parameter[T], error),
) (*Linear[T], error) {
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

// NewLinearFromParam creates a new Linear layer from an existing weights parameter.
func NewLinearFromParam[T tensor.Numeric](engine compute.Engine[T], weights *graph.Parameter[T]) *Linear[T] {
	outputSize := weights.Value.Shape()[1]

	return &Linear[T]{
		multiplier:       components.NewMatrixMultiplier(engine),
		gradientComputer: components.NewLinearGradientComputer(engine),
		weights:          weights,
		outputShape:      []int{1, outputSize},
	}
}

// OutputShape returns the output shape of the Linear layer.
func (l *Linear[T]) OutputShape() []int {
	return l.outputShape
}

// Forward performs the forward pass: output = input * weights.
func (l *Linear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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
func (l *Linear[T]) Backward(ctx context.Context, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Compute both gradients efficiently
	weightsGrad, inputGrad, err := l.gradientComputer.ComputeBothGradients(
		ctx, l.lastInput, l.weights.Value, outputGradient)
	if err != nil {
		return nil, err
	}

	l.weights.Gradient = weightsGrad

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// Parameters returns the parameters of the Linear layer.
func (l *Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.weights}
}

// SetName sets the name of the Linear layer.
func (l *Linear[T]) SetName(name string) {
	l.weights.Name = name + "_weights"
}

// OpType returns the operation type of the Linear layer.
func (l *Linear[T]) OpType() string {
	return "Linear"
}

// Attributes returns nil for the Linear layer.
func (l *Linear[T]) Attributes() map[string]interface{} {
	return nil
}
