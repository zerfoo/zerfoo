package graph

import (
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/tensor"
)

// ErrInvalidInputCount is returned when the number of input tensors is invalid.
var ErrInvalidInputCount = errors.New("invalid number of input tensors")

// Node represents a node in the computation graph.
type Node[T tensor.Numeric] interface {
	// OutputShape returns the shape of the output tensor.
	OutputShape() []int
	// Forward computes the output of the node given the inputs.
	Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
	// Backward computes the gradients of the loss with respect to the inputs and parameters.
	Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error)
	// Parameters returns the parameters of the node.
	Parameters() []*Parameter[T]
}

// Parameter is a container for a trainable tensor (e.g., weights or biases).
// It holds both the tensor for its value and a tensor for its gradient.
type Parameter[T tensor.Numeric] struct {
	Name     string
	Value    *tensor.Tensor[T]
	Gradient *tensor.Tensor[T]
}

// NewParameter creates a new parameter, initializing its gradient tensor with the same shape.
// It takes a tensor creation function to allow for mocking in tests.
func NewParameter[T tensor.Numeric](name string, value *tensor.Tensor[T], newTensorFn func(shape []int, data []T) (*tensor.Tensor[T], error)) (*Parameter[T], error) {
	if name == "" {
		return nil, errors.New("parameter name cannot be empty")
	}
	if value == nil {
		return nil, errors.New("cannot create parameter from nil tensor")
	}
	// Gradient tensor is initialized with zeros and has the same shape as the value.
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

// AddGradient accumulates the given gradient to the parameter's gradient.
func (p *Parameter[T]) AddGradient(grad *tensor.Tensor[T]) error {
	if p.Gradient == nil {
		return fmt.Errorf("parameter %s gradient is nil", p.Name)
	}
	if !p.Gradient.ShapeEquals(grad) {
		return fmt.Errorf("gradient shape mismatch for parameter %s: expected %v, got %v", p.Name, p.Gradient.Shape(), grad.Shape())
	}

	// Assuming element-wise addition for gradients
	for i := range len(p.Gradient.Data()) {
		p.Gradient.Data()[i] += grad.Data()[i]
	}

	return nil
}

// ClearGradient sets the parameter's gradient to zero.
func (p *Parameter[T]) ClearGradient() {
	// Assuming the gradient tensor is already allocated and has the correct shape.
	// If not, it should be handled during parameter initialization.
	for i := range p.Gradient.Data() {
		p.Gradient.Data()[i] = 0
	}
}
