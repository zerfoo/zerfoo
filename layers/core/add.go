package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Statically assert that *Add[T] implements the graph.Node[T] interface.

// Add represents an element-wise addition node.
type Add[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

// NewAdd creates a new Add node.
func NewAdd[T tensor.Numeric](engine compute.Engine[T]) *Add[T] {
	return &Add[T]{engine: engine}
}

// Forward computes the forward pass of the Add node.
func (a *Add[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Add node requires 2 inputs, but got %d", len(inputs))
	}

	return a.engine.Add(ctx, inputs[0], inputs[1])
}

// Backward computes the backward pass of the Add node.
func (a *Add[T]) Backward(_ context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return []*tensor.TensorNumeric[T]{dOut, dOut}, nil
}

// Parameters returns the parameters of the Add node.
func (a *Add[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape of the Add node.
func (a *Add[T]) OutputShape() []int {
	// This is a simplified implementation. A more robust version would
	// calculate the broadcasted shape.
	return nil
}

// Attributes returns the attributes of the Add node.
func (a *Add[T]) Attributes() map[string]any {
	return nil
}

// OpType returns the operator type of the Add node.
func (a *Add[T]) OpType() string {
	return "Add"
}

// BuildAdd constructs an Add node from attributes.
func BuildAdd[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	return NewAdd[T](engine), nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Add[float32])(nil)
