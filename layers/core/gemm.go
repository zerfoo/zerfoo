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

func (g *Gemm[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 || len(inputs) > 3 {
		return nil, fmt.Errorf("Gemm requires 2 or 3 inputs, got %d", len(inputs))
	}

	aData := inputs[0].Data()
	bData := inputs[1].Data()
	aShape := inputs[0].Shape()
	bShape := inputs[1].Shape()

	if len(aShape) != 2 || len(bShape) != 2 {
		return nil, fmt.Errorf("Gemm: inputs must be 2D, got %v and %v", aShape, bShape)
	}

	M, K := aShape[0], aShape[1]
	if g.transA {
		M, K = K, M
	}
	K2, N := bShape[0], bShape[1]
	if g.transB {
		K2, N = N, K2
	}
	if K != K2 {
		return nil, fmt.Errorf("Gemm: inner dims mismatch: %d vs %d", K, K2)
	}

	alpha := T(g.alpha)
	out := make([]T, M*N)

	// Compute alpha * A' * B'
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum T
			for k := 0; k < K; k++ {
				var av, bv T
				if g.transA {
					av = aData[k*aShape[1]+i]
				} else {
					av = aData[i*aShape[1]+k]
				}
				if g.transB {
					bv = bData[j*bShape[1]+k]
				} else {
					bv = bData[k*bShape[1]+j]
				}
				sum += av * bv
			}
			out[i*N+j] = alpha * sum
		}
	}

	// Add beta * C if provided.
	if len(inputs) == 3 {
		beta := T(g.beta)
		cData := inputs[2].Data()
		cShape := inputs[2].Shape()
		switch {
		case len(cData) == 1:
			// Scalar broadcast.
			cv := beta * cData[0]
			for i := range out {
				out[i] += cv
			}
		case len(cShape) == 1 && cShape[0] == N:
			// Bias vector broadcast across rows.
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					out[i*N+j] += beta * cData[j]
				}
			}
		case len(cData) == M*N:
			// Full matrix.
			for i := range out {
				out[i] += beta * cData[i]
			}
		default:
			return nil, fmt.Errorf("Gemm: C shape %v incompatible with output [%d, %d]", cShape, M, N)
		}
	}

	return tensor.New([]int{M, N}, out)
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
