package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestWhere_ElementWise(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	w := &Where[float32]{engine: engine}

	cond, _ := tensor.New[float32]([]int{4}, []float32{1, 0, 1, 0})
	x, _ := tensor.New[float32]([]int{4}, []float32{10, 20, 30, 40})
	y, _ := tensor.New[float32]([]int{4}, []float32{-1, -2, -3, -4})

	out, err := w.Forward(context.Background(), cond, x, y)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	got := out.Data()
	want := []float32{10, -2, 30, -4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestWhere_ScalarX(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	w := &Where[float32]{engine: engine}

	cond, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{1, 0, 1, 0})
	x, _ := tensor.New[float32]([]int{}, []float32{99}) // scalar
	y, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{-1, -2, -3, -4})

	out, err := w.Forward(context.Background(), cond, x, y)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	got := out.Data()
	want := []float32{99, -2, 99, -4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestWhere_ScalarY(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	w := &Where[float32]{engine: engine}

	cond, _ := tensor.New[float32]([]int{3}, []float32{1, 0, 1})
	x, _ := tensor.New[float32]([]int{3}, []float32{10, 20, 30})
	y, _ := tensor.New[float32]([]int{}, []float32{-99}) // scalar

	out, err := w.Forward(context.Background(), cond, x, y)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	got := out.Data()
	want := []float32{10, -99, 30}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
