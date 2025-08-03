package loss

import "github.com/zerfoo/zerfoo/tensor"

// Loss defines the interface for loss functions.
type Loss[T tensor.Numeric] interface {
	// Forward computes the loss and its gradient.
	Forward(predictions, targets *tensor.Tensor[T]) (T, *tensor.Tensor[T])
}
