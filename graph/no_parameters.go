package graph

import "github.com/zerfoo/zerfoo/tensor"

// NoParameters is a utility type for nodes that have no trainable parameters.
type NoParameters[T tensor.Numeric] struct{}

// Parameters returns an empty slice of parameters.
func (n *NoParameters[T]) Parameters() []*Parameter[T] {
	return nil
}
