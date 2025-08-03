package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Bias adds a bias vector to its input.
type Bias[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	biases      *graph.Parameter[T]
	outputShape []int
}

// NewBias creates a new Bias layer with default tensor and parameter creation.
func NewBias[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], size int) (*Bias[T], error) {
	return NewBiasWithFactories(name, engine, ops, size, tensor.New[T], graph.NewParameter[T])
}

// NewBiasWithFactories creates a new Bias layer with custom tensor and parameter creation functions.
func NewBiasWithFactories[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], size int, newTensor func([]int, []T) (*tensor.Tensor[T], error), newParameter func(string, *tensor.Tensor[T], func([]int, []T) (*tensor.Tensor[T], error)) (*graph.Parameter[T], error)) (*Bias[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	// Initialize biases with zeros.
	biasesData := make([]T, size)
	biases, err := newTensor([]int{size}, biasesData)
	if err != nil {
		return nil, fmt.Errorf("failed to create biases tensor: %w", err)
	}

	biasesParam, err := newParameter(fmt.Sprintf("%s_biases", name), biases, newTensor)
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

func (b *Bias[T]) OutputShape() []int {
	return b.outputShape
}

// Forward performs the forward pass: output = input + biases.
func (b *Bias[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	ctx := context.Background()
	output, err := b.engine.Add(ctx, inputs[0], b.biases.Value)
	if err != nil {
		return nil, err
	}
	b.outputShape = output.Shape()
	return output, nil
}

// Backward computes the gradients.
func (b *Bias[T]) Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	ctx := context.Background()
	// Gradient with respect to biases: sum of output_gradient along batch dimension
	biasesGrad, err := b.engine.Sum(ctx, outputGradient, 0, false)
	if err != nil {
		return nil, err
	}
	b.biases.Gradient = biasesGrad

	// Gradient with respect to input is just the output gradient.
	return []*tensor.Tensor[T]{outputGradient}, nil
}

func (b *Bias[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{b.biases}
}

func (b *Bias[T]) SetName(name string) {
	b.biases.Name = fmt.Sprintf("%s_biases", name)
}
