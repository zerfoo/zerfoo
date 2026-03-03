package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestResize_Nearest_WithScales: 2x spatial upscale using scales attribute.
// Input [1,1,2,2] = [[1,2],[3,4]], scale 2x -> [1,1,4,4].
// Nearest-neighbor: each input pixel maps to a 2x2 block in output.
func TestResize_Nearest_WithScales(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	data := []float32{1, 2, 3, 4}
	x, _ := tensor.New[float32]([]int{1, 1, 2, 2}, data)

	attrs := map[string]interface{}{
		"mode":   "nearest",
		"scales": []float64{1, 1, 2, 2},
	}
	resize := NewResize[float32](engine, &ops, "nearest", []float64{1, 1, 2, 2}, nil)
	out, err := resize.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	_ = attrs

	wantShape := []int{1, 1, 4, 4}
	if !shapeEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}

	// Expected output: each pixel tiled 2x2.
	// [1,1,2,2, 1,1,2,2, 3,3,4,4, 3,3,4,4]
	want := []float32{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %v, want %v", i, got[i], v)
		}
	}
}

// TestResize_Nearest_WithSizes: upscale via explicit output sizes.
func TestResize_Nearest_WithSizes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	data := []float32{1, 2, 3, 4}
	x, _ := tensor.New[float32]([]int{1, 1, 2, 2}, data)

	resize := NewResize[float32](engine, &ops, "nearest", nil, []int64{1, 1, 4, 4})
	out, err := resize.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 1, 4, 4}
	if !shapeEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}
	want := []float32{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}
	got := out.Data()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %v, want %v", i, got[i], v)
		}
	}
}

// TestResize_Nearest_NoScaleNoSize: must return error when neither scales nor sizes given.
func TestResize_Nearest_NoScaleNoSize(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	resize := NewResize[float32](engine, &ops, "nearest", nil, nil)
	_, err := resize.Forward(context.Background(), x)
	if err == nil {
		t.Fatal("expected error when scales and sizes are both nil")
	}
}

func TestResize_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	resize := NewResize[float32](engine, &ops, "nearest", []float64{1, 1, 2, 2}, nil)

	_, err := resize.Forward(context.Background())
	if err == nil {
		t.Fatal("expected error for 0 inputs")
	}
}

func TestResize_OpTypeAndMeta(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	resize := NewResize[float32](engine, &ops, "nearest", []float64{1, 1, 2, 2}, nil)

	if resize.OpType() != "Resize" {
		t.Errorf("OpType = %q, want Resize", resize.OpType())
	}
	if resize.Parameters() != nil {
		t.Error("Parameters should be nil")
	}
	grads, err := resize.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("Backward should return nil")
	}
}

func TestBuildResize_NoScalesNoSizes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	_, err := BuildResize[float32](engine, &ops, "resize", nil, map[string]interface{}{})
	if err == nil {
		t.Fatal("expected error when no scales or sizes in attributes")
	}
}

func TestBuildResize_WithScales(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{
		"mode":   "nearest",
		"scales": []float64{1, 1, 2, 2},
	}
	node, err := BuildResize[float32](engine, &ops, "resize", nil, attrs)
	if err != nil {
		t.Fatalf("BuildResize failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildResize returned nil")
	}
	if node.OpType() != "Resize" {
		t.Errorf("OpType = %q, want Resize", node.OpType())
	}
}

func TestBuildResize_WithSizes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{
		"sizes": []int64{1, 1, 4, 4},
	}
	node, err := BuildResize[float32](engine, &ops, "resize", nil, attrs)
	if err != nil {
		t.Fatalf("BuildResize failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildResize returned nil")
	}
}
