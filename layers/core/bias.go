// Package core provides core neural network layer implementations.
package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Bias adds a bias vector to its input.
type Bias[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	biases      *graph.Parameter[T]
	outputShape []int
}

// BiasOptions holds configuration options for the Bias layer.
type BiasOptions[T tensor.Numeric] struct {
	Initializer func(size int) []T
}

// BiasOption is a function that applies an option to BiasOptions.
type BiasOption[T tensor.Numeric] func(*BiasOptions[T])

// WithBiasInitializer sets a custom initializer for the bias vector.
func WithBiasInitializer[T tensor.Numeric](initializer func(size int) []T) BiasOption[T] {
	return func(o *BiasOptions[T]) {
		o.Initializer = initializer
	}
}

// NewBias creates a new Bias layer with default tensor and parameter creation.
func NewBias[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], size int, opts ...BiasOption[T]) (*Bias[T], error) {
	// Default options
	options := &BiasOptions[T]{
		Initializer: func(size int) []T { return make([]T, size) }, // Default to zeros
	}
	for _, opt := range opts {
		opt(options)
	}

	return NewBiasWithFactories(name, engine, ops, size, tensor.New[T], graph.NewParameter[T], options.Initializer)
}

// NewBiasWithFactories creates a new Bias layer with custom tensor and parameter creation functions.
func NewBiasWithFactories[T tensor.Numeric](
	name string, engine compute.Engine[T], ops numeric.Arithmetic[T], size int,
	newTensor func([]int, []T) (*tensor.TensorNumeric[T], error),
	newParameter func(string, *tensor.TensorNumeric[T], func([]int, []T) (*tensor.TensorNumeric[T], error)) (*graph.Parameter[T], error),
	initializer func(size int) []T,
) (*Bias[T], error) {
	if name == "" {
		return nil, errors.New("layer name cannot be empty")
	}

	// Initialize biases.
	biasesData := initializer(size)

	biases, err := newTensor([]int{size}, biasesData)
	if err != nil {
		return nil, fmt.Errorf("failed to create biases tensor: %w", err)
	}

	biasesParam, err := newParameter(name+"_biases", biases, newTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to create biases parameter: %w", err)
	}

	return &Bias[T]{
		engine:      engine,
		ops:         ops,
		biases:      biasesParam,
		outputShape: []int{1, size}, // Assuming batch size of 1 for now
	}, nil
}

// NewBiasFromParam creates a new Bias layer from an existing biases parameter.
func NewBiasFromParam[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], biases *graph.Parameter[T]) *Bias[T] {
	size := biases.Value.Shape()[0]

	return &Bias[T]{
		engine:      engine,
		ops:         ops,
		biases:      biases,
		outputShape: []int{1, size},
	}
}

// OutputShape returns the output shape of the Bias layer.
func (b *Bias[T]) OutputShape() []int {
	return b.outputShape
}

// Forward performs the forward pass: output = input + biases.
func (b *Bias[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	output, err := b.engine.Add(ctx, inputs[0], b.biases.Value)
	if err != nil {
		return nil, err
	}

	b.outputShape = output.Shape()

	return output, nil
}

// Backward computes the gradients.
func (b *Bias[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Gradient with respect to biases: sum of output_gradient along batch dimension
	biasesGrad, err := b.engine.Sum(ctx, outputGradient, 0, false)
	if err != nil {
		return nil, err
	}

	b.biases.Gradient = biasesGrad

	// Gradient with respect to input is just the output gradient.
	return []*tensor.TensorNumeric[T]{outputGradient}, nil
}

// Parameters returns the parameters of the Bias layer.
func (b *Bias[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{b.biases}
}

// SetName sets the name of the Bias layer.
func (b *Bias[T]) SetName(name string) {
	b.biases.Name = name + "_biases"
}

// OpType returns the operation type of the Bias layer.
func (b *Bias[T]) OpType() string {
	return "Bias"
}

// Attributes returns nil for the Bias layer.
func (b *Bias[T]) Attributes() map[string]interface{} {
	return nil
}
