package model

import "github.com/zerfoo/zerfoo/tensor"

// Exporter defines the interface for saving a zerfoo model to an external format.
type Exporter[T tensor.Numeric] interface {
	// Export saves the given model to the specified path.
	Export(model *Model[T], path string) error
}
