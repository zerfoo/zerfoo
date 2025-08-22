package loss

import (
	"context"

	"github.com/zerfoo/zerfoo/tensor"
)

// Loss defines the interface for loss functions.
type Loss[T tensor.Numeric] interface {
	// Forward computes the loss and its gradient.
	Forward(ctx context.Context, predictions, targets *tensor.TensorNumeric[T]) (T, *tensor.TensorNumeric[T], error)
}
