package model

// Exporter defines the interface for saving a zerfoo model to an external format.	ype Exporter[T Numeric] interface {
	// Export saves the given model to the specified path.
	Export(model *Model[T], path string) error
}
