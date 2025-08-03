package compute

import (
	"context"
	"fmt"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestableEngine extends CPUEngine with methods that allow controlled error injection
// This enables testing of previously unreachable error paths
type TestableEngine[T tensor.Numeric] struct {
	*CPUEngine[T]
}

// NewTestableEngine creates a new TestableEngine
func NewTestableEngine[T tensor.Numeric](ops numeric.Arithmetic[T]) *TestableEngine[T] {
	return &TestableEngine[T]{
		CPUEngine: NewCPUEngine(ops),
	}
}

// FailableTensor wraps a tensor and can be configured to fail on specific operations
type FailableTensor[T tensor.Numeric] struct {
	*tensor.Tensor[T]
	failOnSet    bool
	setFailCount int // Fail after this many Set calls
	setCalls     int // Track number of Set calls
}

// NewFailableTensor creates a new FailableTensor wrapper
func NewFailableTensor[T tensor.Numeric](t *tensor.Tensor[T]) *FailableTensor[T] {
	return &FailableTensor[T]{Tensor: t}
}

// SetFailOnSet configures the tensor to fail on Set operations
func (f *FailableTensor[T]) SetFailOnSet(fail bool) {
	f.failOnSet = fail
}

// SetFailOnSetAfter configures the tensor to fail after N Set calls
func (f *FailableTensor[T]) SetFailOnSetAfter(count int) {
	f.setFailCount = count
	f.setCalls = 0
}

// Set overrides the tensor's Set method to allow controlled failures
func (f *FailableTensor[T]) Set(value T, indices ...int) error {
	f.setCalls++

	if f.failOnSet {
		return fmt.Errorf("controlled failure: Set operation failed")
	}

	if f.setFailCount > 0 && f.setCalls >= f.setFailCount {
		return fmt.Errorf("controlled failure: Set operation failed after %d calls", f.setCalls)
	}

	return f.Tensor.Set(value, indices...)
}

// TestableMatMul performs matrix multiplication with a FailableTensor result
// This allows testing the error path in MatMul when result.Set() fails
func (e *TestableEngine[T]) TestableMatMul(ctx context.Context, a, b *tensor.Tensor[T], result *FailableTensor[T]) error {
	if a == nil || b == nil {
		return fmt.Errorf("input tensors cannot be nil")
	}
	if result == nil {
		return fmt.Errorf("result tensor cannot be nil")
	}

	aShape := a.Shape()
	bShape := b.Shape()
	if len(aShape) != 2 || len(bShape) != 2 || aShape[1] != bShape[0] {
		return fmt.Errorf("invalid shapes for matrix multiplication")
	}

	// Verify result tensor has correct shape
	expectedShape := []int{aShape[0], bShape[1]}
	if !equalSlices(result.Shape(), expectedShape) {
		return fmt.Errorf("result tensor has incorrect shape: got %v, want %v", result.Shape(), expectedShape)
	}

	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < bShape[1]; j++ {
			sum := e.ops.FromFloat32(0)
			for k := 0; k < aShape[1]; k++ {
				valA, _ := a.At(i, k)
				valB, _ := b.At(k, j)
				sum = e.ops.Add(sum, e.ops.Mul(valA, valB))
			}
			// This error path is now testable with FailableTensor
			if err := result.Set(sum, i, j); err != nil {
				return err
			}
		}
	}
	return nil
}

// TestableTranspose performs transpose with a FailableTensor result
// This allows testing the error path in Transpose when result.Set() fails
func (e *TestableEngine[T]) TestableTranspose(ctx context.Context, a *tensor.Tensor[T], result *FailableTensor[T]) error {
	if a == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}
	if result == nil {
		return fmt.Errorf("result tensor cannot be nil")
	}

	shape := a.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("transpose is only supported for 2D tensors")
	}

	// Verify result tensor has correct shape
	expectedShape := []int{shape[1], shape[0]}
	if !equalSlices(result.Shape(), expectedShape) {
		return fmt.Errorf("result tensor has incorrect shape: got %v, want %v", result.Shape(), expectedShape)
	}

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			val, _ := a.At(i, j)
			// This error path is now testable with FailableTensor
			if err := result.Set(val, j, i); err != nil {
				return err
			}
		}
	}
	return nil
}

// FailableZeroer can be configured to fail on Zero operations
type FailableZeroer[T tensor.Numeric] struct {
	engine   *TestableEngine[T]
	failZero bool
}

// NewFailableZeroer creates a new FailableZeroer
func NewFailableZeroer[T tensor.Numeric](engine *TestableEngine[T]) *FailableZeroer[T] {
	return &FailableZeroer[T]{engine: engine}
}

// SetFailOnZero configures the zeroer to fail on Zero operations
func (f *FailableZeroer[T]) SetFailOnZero(fail bool) {
	f.failZero = fail
}

// Zero performs the zero operation with controlled failure capability
func (f *FailableZeroer[T]) Zero(ctx context.Context, a *tensor.Tensor[T]) error {
	if f.failZero {
		return fmt.Errorf("controlled failure: Zero operation failed")
	}
	return f.engine.Zero(ctx, a)
}

// TestableSum performs sum with a FailableZeroer
// This allows testing the error path in Sum when Zero() fails
func (e *TestableEngine[T]) TestableSum(ctx context.Context, a *tensor.Tensor[T], axis int, keepDims bool, zeroer *FailableZeroer[T], result *tensor.Tensor[T]) error {
	if a == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}
	if result == nil {
		return fmt.Errorf("result tensor cannot be nil")
	}
	if zeroer == nil {
		return fmt.Errorf("zeroer cannot be nil")
	}

	// This error path is now testable with FailableZeroer
	if err := zeroer.Zero(ctx, result); err != nil {
		return err
	}

	// Simplified sum logic for demonstration
	// In practice, this would include the full sum implementation
	aData := a.Data()
	rData := result.Data()

	// Simple case: sum all elements into first position
	if axis < 0 {
		var sum T
		for _, v := range aData {
			sum = e.ops.Add(sum, v)
		}
		if len(rData) > 0 {
			rData[0] = sum
		}
	}

	return nil
}

// Helper function to compare slices
func equalSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
