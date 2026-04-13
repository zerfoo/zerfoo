// Package core provides core layer implementations for the Zerfoo ML framework.
package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/internal/xblas"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// MatMul is a layer that performs matrix multiplication of two tensors.
type MatMul[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
}

// NewMatMul creates a new MatMul layer.
func NewMatMul[T tensor.Numeric](engine compute.Engine[T]) *MatMul[T] {
	return &MatMul[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the MatMul layer.
func (m *MatMul[T]) OutputShape() []int {
	return m.outputShape
}

// Parameters returns no trainable parameters for the MatMul layer.
func (m *MatMul[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the matrix multiplication.
func (m *MatMul[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MatMul layer requires exactly 2 inputs, got %d", len(inputs))
	}

	a, b := inputs[0], inputs[1]

	aShape := a.Shape()
	bShape := b.Shape()

	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, fmt.Errorf("MatMul requires at least 2D tensors, got %dD and %dD", len(aShape), len(bShape))
	}

	// For a @ b, a's last dim must match b's second-to-last dim.
	if aShape[len(aShape)-1] != bShape[len(bShape)-2] {
		// Check if this is a case where b needs to be transposed (2D only).
		if len(bShape) == 2 && aShape[len(aShape)-1] == bShape[1] {
			// Ternary fast path: compute C = A * B^T directly from packed
			// ternary weights using only additions and subtractions.
			if result, err := m.tryTernaryBTransposed(a, b, aShape, bShape); result != nil || err != nil {
				return result, err
			}

			// Q4 B fast path: compute C = A * B^T directly from Q4 blocks,
			// avoiding both the transpose and the dequantization of the weight matrix.
			if result, err := m.tryQ4BTransposed(a, b, aShape, bShape); result != nil || err != nil {
				return result, err
			}

			// Use MatMulTransposeB (C = A * B^T) when available, avoiding an
			// explicit Transpose allocation. Caching a transposed tensor caused
			// a use-after-free: the graph's arena pool reclaimed the GPU memory
			// between forward passes, leaving a null device pointer that caused
			// cuBLAS status 7 (INTERNAL_ERROR).
			if tb, ok := m.engine.(compute.TransposeBMatMuler[T]); ok {
				result, err := tb.MatMulTransposeB(ctx, a, b)
				if err != nil {
					return nil, err
				}
				m.outputShape = result.Shape()
				return result, nil
			}

			// Fallback: explicit transpose each pass (no caching).
			bTransposed, err := m.engine.Transpose(ctx, b, []int{1, 0})
			if err != nil {
				return nil, fmt.Errorf("failed to transpose second operand: %w", err)
			}

			result, err := m.engine.MatMul(ctx, a, bTransposed)
			if err != nil {
				return nil, err
			}

			m.outputShape = result.Shape()
			return result, nil
		}

		return nil, fmt.Errorf("incompatible dimensions for matrix multiplication: %v x %v", aShape, bShape)
	}

	result, err := m.engine.MatMul(ctx, a, b)
	if err != nil {
		return nil, err
	}

	m.outputShape = result.Shape()

	return result, nil
}

// tryQ4BTransposed checks if B has Q4 storage and computes C = A * B^T using
// the fused Q4 kernel that reads packed nibbles directly, avoiding both the
// expensive [N,K] → [K,N] transpose and the dequantization to float32.
// Returns (nil, nil) if B is not Q4-backed or T is not float32.
func (m *MatMul[T]) tryQ4BTransposed(a, b *tensor.TensorNumeric[T], aShape, bShape []int) (*tensor.TensorNumeric[T], error) {
	q4, ok := any(b.GetStorage()).(*tensor.Q4Storage)
	if !ok {
		return nil, nil
	}
	// Only handle float32 (Q4 kernel operates on float32).
	aData, ok := any(a.Data()).([]float32)
	if !ok {
		return nil, nil
	}

	// B is [N, K] in Q4. K must be a multiple of 32.
	bN, bK := bShape[0], bShape[1]
	if bK%32 != 0 {
		return nil, nil
	}

	// Compute batch dimensions from A.
	batchSize := 1
	for i := 0; i < len(aShape)-2; i++ {
		batchSize *= aShape[i]
	}
	mDim := aShape[len(aShape)-2]
	kDim := aShape[len(aShape)-1]

	// Build output shape: A's batch dims + [M, N].
	outputShape := make([]int, len(aShape))
	copy(outputShape, aShape[:len(aShape)-1])
	outputShape[len(outputShape)-1] = bN

	size := 1
	for _, d := range outputShape {
		size *= d
	}
	result, err := tensor.New[T](outputShape, make([]T, size))
	if err != nil {
		return nil, err
	}
	rData := any(result.Data()).([]float32)

	for i := range batchSize {
		aOff := i * mDim * kDim
		cOff := i * mDim * bN
		xblas.GemmF32Q4NT(mDim, bN, kDim, aData[aOff:aOff+mDim*kDim], q4, rData[cOff:cOff+mDim*bN])
	}

	m.outputShape = outputShape
	return result, nil
}

// tryTernaryBTransposed checks if B has TernaryStorage and computes C = A * B^T
// using the ternary GEMV kernel that operates on packed {-1, 0, 1} weights
// with only additions and subtractions (no floating-point multiply).
// Returns (nil, nil) if B is not ternary-backed or T is not float32.
func (m *MatMul[T]) tryTernaryBTransposed(a, b *tensor.TensorNumeric[T], aShape, bShape []int) (*tensor.TensorNumeric[T], error) {
	ts, ok := any(b.GetStorage()).(*tensor.TernaryStorage)
	if !ok {
		return nil, nil
	}
	aData, ok := any(a.Data()).([]float32)
	if !ok {
		return nil, nil
	}

	// B is [N, K] in ternary packed format.
	bN, bK := bShape[0], bShape[1]

	// Compute batch dimensions from A.
	batchSize := 1
	for i := 0; i < len(aShape)-2; i++ {
		batchSize *= aShape[i]
	}
	mDim := aShape[len(aShape)-2]
	kDim := aShape[len(aShape)-1]

	if kDim != bK {
		return nil, nil
	}

	// Build output shape: A's batch dims + [M, N].
	outputShape := make([]int, len(aShape))
	copy(outputShape, aShape[:len(aShape)-1])
	outputShape[len(outputShape)-1] = bN

	size := 1
	for _, d := range outputShape {
		size *= d
	}
	result, err := tensor.New[T](outputShape, make([]T, size))
	if err != nil {
		return nil, err
	}
	rData := any(result.Data()).([]float32)

	for i := range batchSize {
		for row := range mDim {
			aOff := (i*mDim + row) * kDim
			y := compute.TernaryGEMV(ts, aData[aOff:aOff+kDim], bN, bK)
			copy(rData[(i*mDim+row)*bN:], y)
		}
	}

	m.outputShape = outputShape
	return result, nil
}

// Backward computes the gradients for the MatMul layer.
func (m *MatMul[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MatMul layer requires exactly 2 inputs, got %d", len(inputs))
	}

	a := inputs[0]
	b := inputs[1]

	aShape := a.Shape()
	bShape := b.Shape()

	// transposeAxes returns axes that swap the last two dimensions.
	transposeAxes := func(ndim int) []int {
		axes := make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
		axes[ndim-2], axes[ndim-1] = axes[ndim-1], axes[ndim-2]
		return axes
	}

	// Detect whether Forward used the transposed-B path (C = A @ B^T).
	// This mirrors the shape check in Forward: inner dims don't match
	// but B is 2D and A's last dim equals B's last dim.
	bTransposedPath := aShape[len(aShape)-1] != bShape[len(bShape)-2] &&
		len(bShape) == 2 && aShape[len(aShape)-1] == bShape[1]

	var gradA, gradB *tensor.TensorNumeric[T]
	var err error

	if bTransposedPath {
		// Forward was: C = A @ B^T
		// dA = dOut @ B
		gradA, err = m.engine.MatMul(ctx, outputGradient, b)
		if err != nil {
			return nil, err
		}
		// dB = dOut^T @ A
		dOutT, err := m.engine.Transpose(ctx, outputGradient, transposeAxes(len(outputGradient.Shape())))
		if err != nil {
			return nil, fmt.Errorf("MatMul backward: failed to transpose outputGradient: %w", err)
		}
		gradB, err = m.engine.MatMul(ctx, dOutT, a)
		if err != nil {
			return nil, err
		}
	} else {
		// Forward was: C = A @ B
		// dA = dOut @ B^T
		bT, err := m.engine.Transpose(ctx, b, transposeAxes(len(bShape)))
		if err != nil {
			return nil, fmt.Errorf("MatMul backward: failed to transpose b: %w", err)
		}
		gradA, err = m.engine.MatMul(ctx, outputGradient, bT)
		if err != nil {
			return nil, err
		}
		// dB = A^T @ dOut
		aT, err := m.engine.Transpose(ctx, a, transposeAxes(len(aShape)))
		if err != nil {
			return nil, fmt.Errorf("MatMul backward: failed to transpose a: %w", err)
		}
		gradB, err = m.engine.MatMul(ctx, aT, outputGradient)
		if err != nil {
			return nil, err
		}
	}

	return []*tensor.TensorNumeric[T]{gradA, gradB}, nil
}

// OpType returns the operation type of the MatMul layer.
func (m *MatMul[T]) OpType() string {
	return "MatMul"
}

// Attributes returns nil for the MatMul layer.
func (m *MatMul[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*MatMul[float32])(nil)
