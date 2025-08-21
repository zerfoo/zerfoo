package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Dense is a fully connected layer that combines a linear transformation and a bias.
type Dense[T tensor.Numeric] struct {
	linear *Linear[T]
	bias   *Bias[T]
}

// NewDense creates a new Dense layer.
func NewDense[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputSize, outputSize int) (*Dense[T], error) {
	linear, err := NewLinear[T](name, engine, ops, inputSize, outputSize)
	if err != nil {
		return nil, err
	}
	bias, err := NewBias[T](name, engine, ops, outputSize)
	if err != nil {
		return nil, err
	}

	return &Dense[T]{linear: linear, bias: bias}, nil
}

// NewDenseFromParams creates a new Dense layer from existing parameters.
func NewDenseFromParams[T tensor.Numeric](linear *Linear[T], bias *Bias[T]) *Dense[T] {
	return &Dense[T]{linear: linear, bias: bias}
}

// OutputShape returns the output shape of the Dense layer.
func (d *Dense[T]) OutputShape() []int {
	return d.linear.OutputShape()
}

// Forward performs the forward pass: output = input*weights + biases.
func (d *Dense[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	input := inputs[0]
	originalShape := input.Shape()
	inputSize := originalShape[len(originalShape)-1]

	// Reshape to 2D if input is N-D
	if len(originalShape) > 2 {
		batchSize := 1
		for i := 0; i < len(originalShape)-1; i++ {
			batchSize *= originalShape[i]
		}
		var err error
		input, err = input.Reshape([]int{batchSize, inputSize})
		if err != nil {
			return nil, err
		}
	}

	linearOutput, err := d.linear.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	biasOutput, err := d.bias.Forward(ctx, linearOutput)
	if err != nil {
		return nil, err
	}

	// Reshape back to original batch dimensions
	if len(originalShape) > 2 {
		outputShape := make([]int, len(originalShape))
		copy(outputShape, originalShape)
		outputShape[len(outputShape)-1] = d.linear.OutputShape()[1]
		var err error
		biasOutput, err = biasOutput.Reshape(outputShape)
		if err != nil {
			return nil, err
		}
	}

	return biasOutput, nil
}

// Backward computes the gradients.
func (d *Dense[T]) Backward(ctx context.Context, outputGradient *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	biasGrads, err := d.bias.Backward(ctx, outputGradient)
	if err != nil {
		return nil, err
	}
	linearGrads, err := d.linear.Backward(ctx, biasGrads[0])
	if err != nil {
		return nil, err
	}

	return linearGrads, nil
}

// Parameters returns the parameters of the Dense layer.
func (d *Dense[T]) Parameters() []*graph.Parameter[T] {
	return append(d.linear.Parameters(), d.bias.Parameters()...)
}

// SetName sets the name of the Dense layer.
func (d *Dense[T]) SetName(name string) {
	d.linear.SetName(name)
	d.bias.SetName(name)
}
