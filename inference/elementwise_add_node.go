package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// elementwiseAddNode computes element-wise addition of two tensors: inputs[0] + inputs[1].
// This is used for residual connections across multiple architectures (GPT-2,
// Command R, BERT, etc.) replacing architecture-specific residual add nodes.
type elementwiseAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (n *elementwiseAddNode[T]) OpType() string                   { return "ElementwiseAdd" }
func (n *elementwiseAddNode[T]) Attributes() map[string]any        { return nil }
func (n *elementwiseAddNode[T]) OutputShape() []int                { return nil }
func (n *elementwiseAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *elementwiseAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("ElementwiseAdd: expected 2 inputs, got %d", len(inputs))
	}
	return n.engine.Add(ctx, inputs[0], inputs[1])
}

func (n *elementwiseAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
