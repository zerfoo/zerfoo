package compute

import (
	"context"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"reflect"
	"testing"
)

func TestCPUEngine_Coverage(t *testing.T) {
	engine := NewCPUEngine[int](numeric.IntOps{})
	ctx := context.Background()

	// Test getOrCreateDest with nil dst
	a, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	dst, err := engine.getOrCreateDest(a.Shape(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dst == nil {
		t.Fatal("expected a new tensor to be created")
	}

	// Test Sum with axis 1 and 1D output
	b, _ := tensor.New[int]([]int{2, 1}, []int{1, 2})
	result, _ := engine.Sum(ctx, b, 1, false, nil)
	expected := []int{1, 2}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Sum with 3D tensor and axis 1
	c, _ := tensor.New[int]([]int{2, 2, 2}, []int{1, 2, 3, 4, 5, 6, 7, 8})
	result, _ = engine.Sum(ctx, c, 1, false, nil)
	expected = []int{4, 6, 12, 14}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Sum with 3D tensor and axis 2
	result, _ = engine.Sum(ctx, c, 2, false, nil)
	expected = []int{3, 7, 11, 15}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Add with broadcast
	d, _ := tensor.New[int]([]int{1, 2}, []int{1, 2})
	result, _ = engine.Add(ctx, c, d, nil)
	expected = []int{2, 4, 4, 6, 6, 8, 8, 10}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Sub with broadcast
	result, _ = engine.Sub(ctx, c, d, nil)
	expected = []int{0, 0, 2, 2, 4, 4, 6, 6}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Mul with broadcast
	result, _ = engine.Mul(ctx, c, d, nil)
	expected = []int{1, 4, 3, 8, 5, 12, 7, 16}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Div with broadcast
	result, _ = engine.Div(ctx, c, d, nil)
	expected = []int{1, 1, 3, 2, 5, 3, 7, 4}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test Pow with broadcast
	result, _ = engine.Pow(ctx, c, d, nil)
	expected = []int{1, 4, 3, 16, 5, 36, 7, 64}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test error case for getOrCreateDest
	dst, _ = tensor.New[int]([]int{1, 1}, nil)
	_, err = engine.getOrCreateDest(a.Shape(), dst)
	if err == nil {
		t.Error("expected error for mismatched dst shape")
	}

	// Test sum with axis 1 and 2D output
	e, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	result, _ = engine.Sum(ctx, e, 1, false, nil)
	expected = []int{3, 7}
	if !reflect.DeepEqual(result.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, result.Data())
	}

	// Test getOrCreateDest with nil dst slice
	dst, err = engine.getOrCreateDest(a.Shape())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dst == nil {
		t.Fatal("expected a new tensor to be created")
	}

	// Test Add with broadcast and dst
	result, _ = engine.Add(ctx, c, d, c)
	if result != c {
		t.Error("expected result to be c")
	}

	// Test Sub with broadcast and dst
	result, _ = engine.Sub(ctx, c, d, c)
	if result != c {
		t.Error("expected result to be c")
	}

	// Test Mul with broadcast and dst
	result, _ = engine.Mul(ctx, c, d, c)
	if result != c {
		t.Error("expected result to be c")
	}

	// Test Div with broadcast and dst
	result, _ = engine.Div(ctx, c, d, c)
	if result != c {
		t.Error("expected result to be c")
	}

	// Test Pow with broadcast and dst
	result, _ = engine.Pow(ctx, c, d, c)
	if result != c {
		t.Error("expected result to be c")
	}

	// Test UnaryOp with dst
	result, _ = engine.UnaryOp(ctx, a, func(v int) int { return v * 2 }, a)
	if result != a {
		t.Error("expected result to be a")
	}

	// Test Transpose with dst
	a, _ = tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	result, _ = engine.Transpose(ctx, a, []int{1, 0}, a)
	if result != a {
		t.Error("expected result to be a")
	}

	// Test Exp with dst
	f, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	engineFloat32 := NewCPUEngine[float32](numeric.Float32Ops{})
	resultFloat32, _ := engineFloat32.Exp(ctx, f, f)
	if resultFloat32 != f {
		t.Error("expected result to be f")
	}

	// Test Log with dst
	resultFloat32, _ = engineFloat32.Log(ctx, f, f)
	if resultFloat32 != f {
		t.Error("expected result to be f")
	}

	// Test Div with broadcast and zero divisor
	e, _ = tensor.New[int]([]int{1, 2}, []int{1, 0})
	_, err = engine.Div(ctx, c, e, nil)
	if err == nil {
		t.Error("expected error for division by zero")
	}
}
