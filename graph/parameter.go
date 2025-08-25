package graph

import (
	"errors"

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
		return nil, errors.New("parameter name cannot be empty")
	}

	if value == nil {
		return nil, errors.New("parameter value cannot be nil")
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
		return errors.New("parameter gradient is nil")
	}

	if !tensor.ShapesEqual(p.Value.Shape(), grad.Shape()) {
		return errors.New("gradient shape mismatch")
	}

	gdst := p.Gradient.Data()
	gsrc := grad.Data()

	// Handle supported built-in numeric types via type switch.
	switch any(*new(T)).(type) {
	case float32:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(float32) + any(gsrc[i]).(float32)).(T)
		}
	case float64:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(float64) + any(gsrc[i]).(float64)).(T)
		}
	case int:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(int) + any(gsrc[i]).(int)).(T)
		}
	case int8:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(int8) + any(gsrc[i]).(int8)).(T)
		}
	case int16:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(int16) + any(gsrc[i]).(int16)).(T)
		}
	case int32:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(int32) + any(gsrc[i]).(int32)).(T)
		}
	case int64:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(int64) + any(gsrc[i]).(int64)).(T)
		}
	case uint:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(uint) + any(gsrc[i]).(uint)).(T)
		}
	case uint32:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(uint32) + any(gsrc[i]).(uint32)).(T)
		}
	case uint64:
		for i := range gdst {
			gdst[i] = any(any(gdst[i]).(uint64) + any(gsrc[i]).(uint64)).(T)
		}
	default:
		return errors.New("AddGradient unsupported for this numeric type; use engine ops instead")
	}

	return nil
}

// ClearGradient resets the parameter's gradient to zero.
func (p *Parameter[T]) ClearGradient() {
	gdst := p.Gradient.Data()
	switch any(*new(T)).(type) {
	case float32:
		for i := range gdst {
			gdst[i] = any(float32(0)).(T)
		}
	case float64:
		for i := range gdst {
			gdst[i] = any(float64(0)).(T)
		}
	case int:
		for i := range gdst {
			gdst[i] = any(int(0)).(T)
		}
	case int8:
		for i := range gdst {
			gdst[i] = any(int8(0)).(T)
		}
	case int16:
		for i := range gdst {
			gdst[i] = any(int16(0)).(T)
		}
	case int32:
		for i := range gdst {
			gdst[i] = any(int32(0)).(T)
		}
	case int64:
		for i := range gdst {
			gdst[i] = any(int64(0)).(T)
		}
	case uint:
		for i := range gdst {
			gdst[i] = any(uint(0)).(T)
		}
	case uint32:
		for i := range gdst {
			gdst[i] = any(uint32(0)).(T)
		}
	case uint64:
		for i := range gdst {
			gdst[i] = any(uint64(0)).(T)
		}
	default:
		// Unsupported numeric types: set via copy from a zeroed slice of same length if possible
		for i := range gdst {
			var z T
			gdst[i] = z
		}
	}
}
