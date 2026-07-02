package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Gemm implements the ONNX Gemm operator: Y = alpha * A' * B' + beta * C
// where A' = transpose(A) if transA, B' = transpose(B) if transB.
type Gemm[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	alpha  float64
	beta   float64
	transA bool
	transB bool
}

func (g *Gemm[T]) OpType() string { return "Gemm" }
func (g *Gemm[T]) Attributes() map[string]any {
	return map[string]any{
		"alpha":  g.alpha,
		"beta":   g.beta,
		"transA": g.transA,
		"transB": g.transB,
	}
}
func (g *Gemm[T]) OutputShape() []int               { return nil }
func (g *Gemm[T]) Parameters() []*graph.Parameter[T] { return nil }

func (g *Gemm[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("Gemm requires 2 or 3 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]

	if len(a.Shape()) != 2 || len(b.Shape()) != 2 {
		return nil, fmt.Errorf("Gemm: inputs must be 2D, got %v and %v", a.Shape(), b.Shape())
	}

	// Apply transposes if needed.
	var err error
	if g.transA {
		a, err = g.engine.Transpose(ctx, a, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("Gemm: transA: %w", err)
		}
	}
	if g.transB {
		b, err = g.engine.Transpose(ctx, b, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("Gemm: transB: %w", err)
		}
	}

	// Validate inner dimensions.
	if a.Shape()[1] != b.Shape()[0] {
		return nil, fmt.Errorf("Gemm: inner dims mismatch: %d vs %d", a.Shape()[1], b.Shape()[0])
	}

	// Compute A' * B'.
	result, err := g.engine.MatMul(ctx, a, b)
	if err != nil {
		return nil, fmt.Errorf("Gemm: matmul: %w", err)
	}

	// Scale by alpha if needed.
	if g.alpha != 1.0 {
		alpha := T(g.alpha)
		result, err = g.engine.UnaryOp(ctx, result, func(v T) T { return alpha * v })
		if err != nil {
			return nil, fmt.Errorf("Gemm: alpha scale: %w", err)
		}
	}

	// Add beta * C if provided.
	if len(inputs) == 3 {
		c := inputs[2]
		if g.beta != 1.0 {
			beta := T(g.beta)
			c, err = g.engine.UnaryOp(ctx, c, func(v T) T { return beta * v })
			if err != nil {
				return nil, fmt.Errorf("Gemm: beta scale: %w", err)
			}
		}
		result, err = g.engine.Add(ctx, result, c)
		if err != nil {
			return nil, fmt.Errorf("Gemm: add bias: %w", err)
		}
	}

	return result, nil
}

func (g *Gemm[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Gemm backward not implemented")
}

// BuildGemm constructs a Gemm node from attributes.
func BuildGemm[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	alpha := 1.0
	beta := 1.0
	var transA, transB bool

	if v, ok := attrs["alpha"]; ok {
		switch a := v.(type) {
		case float64:
			alpha = a
		case float32:
			alpha = float64(a)
		}
	}
	if v, ok := attrs["beta"]; ok {
		switch b := v.(type) {
		case float64:
			beta = b
		case float32:
			beta = float64(b)
		}
	}
	if v, ok := attrs["transA"]; ok {
		switch a := v.(type) {
		case int64:
			transA = a != 0
		case bool:
			transA = a
		}
	}
	if v, ok := attrs["transB"]; ok {
		switch b := v.(type) {
		case int64:
			transB = b != 0
		case bool:
			transB = b
		}
	}

	return &Gemm[T]{
		engine: engine,
		ops:    ops,
		alpha:  alpha,
		beta:   beta,
		transA: transA,
		transB: transB,
	}, nil
}

var _ graph.Node[float32] = (*Gemm[float32])(nil)
