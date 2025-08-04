package core

import (
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

// OutputShape returns the output shape of the Dense layer.
func (d *Dense[T]) OutputShape() []int {
	return d.linear.OutputShape()
}

// Forward performs the forward pass: output = input*weights + biases.
func (d *Dense[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	linearOutput, err := d.linear.Forward(inputs...)
	if err != nil {
		return nil, err
	}
	biasOutput, err := d.bias.Forward(linearOutput)
	if err != nil {
		return nil, err
	}
	return biasOutput, nil
}

// Backward computes the gradients.
func (d *Dense[T]) Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	biasGrads, err := d.bias.Backward(outputGradient)
	if err != nil {
		return nil, err
	}
	linearGrads, err := d.linear.Backward(biasGrads[0])
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
