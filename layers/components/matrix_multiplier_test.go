package components

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestMatrixMultiplier_Multiply(t *testing.T) {
	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](ops)
	multiplier := NewMatrixMultiplier(engine)

	// Create test matrices: A (2x3) * B (3x2) = C (2x2)
	aData := []float32{1, 2, 3, 4, 5, 6}
	bData := []float32{1, 2, 3, 4, 5, 6}

	a, err := tensor.New([]int{2, 3}, aData)
	if err != nil {
		t.Fatalf("Failed to create tensor A: %v", err)
	}

	b, err := tensor.New([]int{3, 2}, bData)
	if err != nil {
		t.Fatalf("Failed to create tensor B: %v", err)
	}

	ctx := context.Background()
	result, err := multiplier.Multiply(ctx, a, b)
	if err != nil {
		t.Fatalf("Matrix multiplication failed: %v", err)
	}

	expectedShape := []int{2, 2}
	if !equalIntSlices(result.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
	}

	// Verify some values
	resultData := result.Data()
	if len(resultData) != 4 {
		t.Errorf("Expected 4 elements in result, got %d", len(resultData))
	}
}

func TestMatrixMultiplier_Transpose(t *testing.T) {
	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](ops)
	multiplier := NewMatrixMultiplier(engine)

	// Create test matrix: A (2x3)
	aData := []float32{1, 2, 3, 4, 5, 6}
	a, err := tensor.New([]int{2, 3}, aData)
	if err != nil {
		t.Fatalf("Failed to create tensor A: %v", err)
	}

	ctx := context.Background()
	result, err := multiplier.Transpose(ctx, a)
	if err != nil {
		t.Fatalf("Matrix transpose failed: %v", err)
	}

	expectedShape := []int{3, 2}
	if !equalIntSlices(result.Shape(), expectedShape) {
		t.Errorf("Expected shape %v, got %v", expectedShape, result.Shape())
	}
}

func equalIntSlices(a, b []int) bool {
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
