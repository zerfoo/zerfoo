package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGlobalAveragePool_Basic: [1,2,2,2] input with two channels.
// Channel 0: [1,2,3,4] -> avg = 2.5. Channel 1: [5,6,7,8] -> avg = 6.5.
func TestGlobalAveragePool_Basic(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	data := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	x, _ := tensor.New[float32]([]int{1, 2, 2, 2}, data)

	gap := NewGlobalAveragePool[float32](engine, &ops)
	out, err := gap.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 2, 1, 1}
	if !shapeEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}

	got := out.Data()
	const tol = float32(1e-5)
	if diff := got[0] - 2.5; diff < -tol || diff > tol {
		t.Errorf("ch0 avg = %v, want 2.5", got[0])
	}
	if diff := got[1] - 6.5; diff < -tol || diff > tol {
		t.Errorf("ch1 avg = %v, want 6.5", got[1])
	}
}

// TestGlobalAveragePool_SingleSpatial: H=W=1, output should equal input.
func TestGlobalAveragePool_SingleSpatial(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x, _ := tensor.New[float32]([]int{1, 3, 1, 1}, []float32{7, 8, 9})
	gap := NewGlobalAveragePool[float32](engine, &ops)
	out, err := gap.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	got := out.Data()
	want := []float32{7, 8, 9}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestGlobalAveragePool_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	gap := NewGlobalAveragePool[float32](engine, &ops)

	_, err := gap.Forward(context.Background())
	if err == nil {
		t.Fatal("expected error for 0 inputs")
	}

	x1, _ := tensor.New[float32]([]int{1, 1, 2, 2}, nil)
	x2, _ := tensor.New[float32]([]int{1, 1, 2, 2}, nil)
	_, err = gap.Forward(context.Background(), x1, x2)
	if err == nil {
		t.Fatal("expected error for 2 inputs")
	}
}

func TestGlobalAveragePool_InvalidRank(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	gap := NewGlobalAveragePool[float32](engine, &ops)

	x, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, err := gap.Forward(context.Background(), x)
	if err == nil {
		t.Fatal("expected error for non-4D input")
	}
}

func TestGlobalAveragePool_OpTypeAndMeta(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	gap := NewGlobalAveragePool[float32](engine, &ops)

	if gap.OpType() != "GlobalAveragePool" {
		t.Errorf("OpType = %q, want GlobalAveragePool", gap.OpType())
	}
	if gap.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	grads, err := gap.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("Backward should return nil")
	}
}

func TestBuildGlobalAveragePool(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	node, err := BuildGlobalAveragePool[float32](engine, &ops, "gap", nil, nil)
	if err != nil {
		t.Fatalf("BuildGlobalAveragePool failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildGlobalAveragePool returned nil")
	}
	if node.OpType() != "GlobalAveragePool" {
		t.Errorf("OpType = %q, want GlobalAveragePool", node.OpType())
	}
}

// TestGlobalAveragePool_OutputShape verifies OutputShape is set after Forward.
func TestGlobalAveragePool_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x, _ := tensor.New[float32]([]int{2, 4, 3, 3}, nil)
	for _, v := range x.Data() {
		_ = v
	}
	// Fill with arbitrary values.
	data := make([]float32, 2*4*3*3)
	for i := range data {
		data[i] = float32(i)
	}
	x.SetData(data)

	gap := NewGlobalAveragePool[float32](engine, &ops)
	if _, err := gap.Forward(context.Background(), x); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	want := []int{2, 4, 1, 1}
	if !shapeEq(gap.OutputShape(), want) {
		t.Errorf("OutputShape = %v, want %v", gap.OutputShape(), want)
	}
}
