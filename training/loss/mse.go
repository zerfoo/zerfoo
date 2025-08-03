package loss

import (
	"context"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// MSE calculates the mean squared error between predictions and targets.
type MSE[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
}

func NewMSE[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *MSE[T] {
	return &MSE[T]{engine: engine, ops: ops}
}

// Forward computes the loss value.
func (m *MSE[T]) Forward(predictions, targets *tensor.Tensor[T]) *tensor.Tensor[T] {
	ctx := context.Background()
	diff, _ := m.engine.Sub(ctx, predictions, targets, nil)
	squared, _ := m.engine.Mul(ctx, diff, diff, nil)

	data := squared.Data()
	var sum T
	for _, val := range data {
		sum = m.ops.Add(sum, val)
	}

	// For simplicity, we divide by N.
	n := m.ops.FromFloat32(float32(len(data)))
	loss, _ := tensor.New[T]([]int{1}, []T{m.ops.Div(sum, n)})
	return loss
}

// Backward computes the initial gradient of the loss.
func (m *MSE[T]) Backward(predictions, targets *tensor.Tensor[T]) *tensor.Tensor[T] {
	// Gradient is 2 * (predictions - targets) / N. We'll ignore the scaling factor.
	diff, _ := m.engine.Sub(context.Background(), predictions, targets, nil)
	return diff
}
