package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// MSE calculates the mean squared error between predictions and targets.
type MSE[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Cached tensors for backward pass
	predictions *tensor.TensorNumeric[T]
	targets     *tensor.TensorNumeric[T]
}

// NewMSE creates a new MSE loss function.
func NewMSE[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *MSE[T] {
	return &MSE[T]{engine: engine, ops: ops}
}

// Forward computes the loss value.
func (m *MSE[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MSE expects 2 inputs, got %d", len(inputs))
	}
	predictions := inputs[0]
	targets := inputs[1]
	// Cache inputs for backward
	m.predictions = predictions
	m.targets = targets

	diff, err := m.engine.Sub(ctx, predictions, targets, nil)
	if err != nil {
		return nil, err
	}

	squared, err := m.engine.Mul(ctx, diff, diff, nil)
	if err != nil {
		return nil, err
	}

	data := squared.Data()

	var sum T
	for _, val := range data {
		sum = m.ops.Add(sum, val)
	}

	// For simplicity, we divide by N.
	n := m.ops.FromFloat64(float64(len(data)))

	loss, err := tensor.New[T]([]int{1}, []T{m.ops.Div(sum, n)})
	if err != nil {
		return nil, err
	}

	return loss, nil
}

// Backward computes the gradients for MSE with respect to inputs.
// Returns gradients in the order of inputs: [dPredictions, dTargets(nil)].
func (m *MSE[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Determine sources: prefer provided inputs, else cached
	preds := m.predictions
	targs := m.targets
	if len(inputs) > 0 {
		preds = inputs[0]
		if len(inputs) > 1 {
			targs = inputs[1]
		}
	}
	// Base gradient: (predictions - targets)
	if preds == nil || targs == nil {
		return nil, graph.ErrInvalidInputCount
	}
	diff, err := m.engine.Sub(ctx, preds, targs, nil)
	if err != nil {
		return nil, err
	}

	// Chain with upstream gradient dOut.
	gradPred, err := m.engine.Mul(ctx, diff, dOut, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradPred, nil}, nil
}

// OutputShape returns the output shape of the MSE loss function.
func (m *MSE[T]) OutputShape() []int {
	return []int{1}
}

// OpType returns the operation type of the MSE loss function.
func (m *MSE[T]) OpType() string {
	return "MSE"
}

// Attributes returns the attributes of the MSE loss function.
func (m *MSE[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*MSE[float32])(nil)

// Parameters returns the parameters of the MSE loss function.
func (m *MSE[T]) Parameters() []*graph.Parameter[T] {
	return nil
}
