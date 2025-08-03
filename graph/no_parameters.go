package graph

import "github.com/zerfoo/zerfoo/tensor"

// NoParameters is a helper struct for layers that have no parameters.
// It provides a default implementation of the Parameters() method.
type NoParameters[T tensor.Numeric] struct{}

// Parameters returns an empty slice of parameters.
func (np *NoParameters[T]) Parameters() []*Parameter[T] {
	return []*Parameter[T]{}
}
