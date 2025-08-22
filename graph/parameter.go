package graph

import (
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// Parameter represents a trainable parameter in the graph.
type Parameter[T tensor.Numeric] struct {
	Name     string
	Value    *tensor.TensorNumeric[T]
	Gradient *tensor.TensorNumeric[T]
}

// NewParameter creates a new parameter.
func NewParameter[T tensor.Numeric](name string, value *tensor.TensorNumeric[T], newTensorFn func([]int, []T) (*tensor.TensorNumeric[T], error)) (*Parameter[T], error) {
	if name == "" {
		return nil, fmt.Errorf("parameter name cannot be empty")
	}
	if value == nil {
		return nil, fmt.Errorf("parameter value cannot be nil")
	}
	grad, err := newTensorFn(value.Shape(), nil)
	if err != nil {
		return nil, err
	}
	return &Parameter[T]{
		Name:     name,
		Value:    value,
		Gradient: grad,
	}, nil
}

// AddGradient adds the given gradient to the parameter's gradient.
func (p *Parameter[T]) AddGradient(grad *tensor.TensorNumeric[T]) error {
	if p.Gradient == nil {
		return fmt.Errorf("parameter gradient is nil")
	}
	if !tensor.ShapesEqual(p.Value.Shape(), grad.Shape()) {
		return fmt.Errorf("gradient shape mismatch")
	}
	for i := range p.Gradient.Data() {
		p.Gradient.Data()[i] += grad.Data()[i]
	}
	return nil
}

// ClearGradient resets the parameter's gradient to zero.
func (p *Parameter[T]) ClearGradient() {
	for i := range p.Gradient.Data() {
		p.Gradient.Data()[i] = 0
	}
}
