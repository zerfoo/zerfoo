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

// GlobalAveragePool performs global average pooling over the spatial dimensions.
// Forward expects one 4D input [N, C, H, W] and produces [N, C, 1, 1].
type GlobalAveragePool[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	outputShape []int
}

// NewGlobalAveragePool creates a GlobalAveragePool layer.
func NewGlobalAveragePool[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *GlobalAveragePool[T] {
	return &GlobalAveragePool[T]{engine: engine, ops: ops}
}

// Forward computes global average pooling.
func (g *GlobalAveragePool[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GlobalAveragePool requires exactly 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("GlobalAveragePool requires 4D input [N,C,H,W], got shape %v", shape)
	}

	n, c, h, w := shape[0], shape[1], shape[2], shape[3]
	hw := h * w
	scale := g.ops.FromFloat64(1.0 / float64(hw))
	xData := x.Data()

	outData := make([]T, n*c)
	for ni := range n {
		for ci := range c {
			sum := g.ops.FromFloat64(0)
			base := ni*c*h*w + ci*h*w
			for i := range hw {
				sum = g.ops.Add(sum, xData[base+i])
			}
			outData[ni*c+ci] = g.ops.Mul(sum, scale)
		}
	}

	out, err := tensor.New[T]([]int{n, c, 1, 1}, outData)
	if err != nil {
		return nil, fmt.Errorf("GlobalAveragePool: failed to create output tensor: %w", err)
	}
	g.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (g *GlobalAveragePool[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "GlobalAveragePool".
func (g *GlobalAveragePool[T]) OpType() string { return "GlobalAveragePool" }

// Attributes returns nil.
func (g *GlobalAveragePool[T]) Attributes() map[string]interface{} { return nil }

// OutputShape returns the output shape (populated after Forward).
func (g *GlobalAveragePool[T]) OutputShape() []int { return g.outputShape }

// Parameters returns nil.
func (g *GlobalAveragePool[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildGlobalAveragePool constructs a GlobalAveragePool node for the layer registry.
func BuildGlobalAveragePool[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewGlobalAveragePool(engine, ops), nil
}

// Statically assert that GlobalAveragePool implements graph.Node.
var _ graph.Node[float32] = (*GlobalAveragePool[float32])(nil)
