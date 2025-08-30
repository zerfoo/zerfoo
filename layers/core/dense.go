package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Dense is a fully connected layer with optional activation and bias.
type Dense[T tensor.Numeric] struct {
	name       string
	linear     *Linear[T]
	bias       *Bias[T]
	activation graph.Node[T]
}

// DenseOpt is a functional option for configuring a Dense layer.
type DenseOpt[T tensor.Numeric] func(*Dense[T])

// WithBias adds a bias to the Dense layer.
func WithBias[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], outputFeatures int) DenseOpt[T] {
	return func(d *Dense[T]) {
		b, err := NewBias[T](d.name+"_bias", engine, ops, outputFeatures)
		if err != nil {
			panic(fmt.Sprintf("failed to create bias: %v", err))
		}
		d.bias = b
	}
}

// WithoutBias disables bias for the Dense layer.
func WithoutBias[T tensor.Numeric]() DenseOpt[T] {
	return func(d *Dense[T]) {
		d.bias = nil
	}
}

// WithActivation adds an activation function to the Dense layer.
func WithActivation[T tensor.Numeric](activation graph.Node[T]) DenseOpt[T] {
	return func(d *Dense[T]) {
		d.activation = activation
	}
}

// NewDense creates a new Dense layer.
func NewDense[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputFeatures, outputFeatures int,
	opts ...DenseOpt[T],
) (*Dense[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputFeatures <= 0 || outputFeatures <= 0 {
		return nil, fmt.Errorf("input and output features must be positive")
	}

	linear, err := NewLinear[T](name+"_linear", engine, ops, inputFeatures, outputFeatures)
	if err != nil {
		return nil, err
	}

	d := &Dense[T]{
		name:   name,
		linear: linear,
	}

	// Add bias by default
	bias, err := NewBias[T](name+"_bias", engine, ops, outputFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias: %w", err)
	}
	d.bias = bias

	for _, opt := range opts {
		opt(d)
	}

	return d, nil
}

// OpType returns the operation type of the layer.
func (d *Dense[T]) OpType() string {
	return "Dense"
}

// Attributes returns the attributes of the layer.
func (d *Dense[T]) Attributes() map[string]interface{} {
	attrs := map[string]interface{}{}
	if d.bias != nil {
		attrs["bias"] = true
	}
	if d.activation != nil {
		attrs["activation"] = d.activation.OpType()
	}
	return attrs
}

// OutputShape returns the output shape of the layer.
func (d *Dense[T]) OutputShape() []int {
	return d.linear.OutputShape()
}

// Forward computes the forward pass of the layer.
func (d *Dense[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	linearOutput, err := d.linear.Forward(ctx, inputs...)
	if err != nil {
		return nil, err
	}

	biasOutput := linearOutput
	if d.bias != nil {
		biasOutput, err = d.bias.Forward(ctx, linearOutput)
		if err != nil {
			return nil, err
		}
	}

	if d.activation != nil {
		return d.activation.Forward(ctx, biasOutput)
	}

	return biasOutput, nil
}

// Backward computes the gradients.
func (d *Dense[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	activationGradient := outputGradient
	if d.activation != nil {
		grads, err := d.activation.Backward(ctx, mode, outputGradient, inputs...)
		if err != nil {
			return nil, err
		}
		activationGradient = grads[0]
	}

	biasGradient := activationGradient
	if d.bias != nil {
		grads, err := d.bias.Backward(ctx, mode, activationGradient)
		if err != nil {
			return nil, err
		}
		biasGradient = grads[0]
	}

	return d.linear.Backward(ctx, mode, biasGradient, inputs...)
}

// Parameters returns the parameters of the layer.
func (d *Dense[T]) Parameters() []*graph.Parameter[T] {
	params := d.linear.Parameters()
	if d.bias != nil {
		params = append(params, d.bias.Parameters()...)
	}
	return params
}

// SetName sets the name of the Dense layer.
func (d *Dense[T]) SetName(name string) {
	d.name = name
	d.linear.SetName(name + "_linear")
	if d.bias != nil {
		d.bias.SetName(name + "_bias")
	}
}

// Name returns the name of the Dense layer.
func (d *Dense[T]) Name() string {
	return d.name
}

// NewDenseFromParams creates a Dense layer from existing Linear and Bias components.
// This is used for constructing layers from pre-existing parameters during model loading.
func NewDenseFromParams[T tensor.Numeric](linear *Linear[T], bias *Bias[T]) *Dense[T] {
	return &Dense[T]{
		linear: linear,
		bias:   bias,
	}
}
