package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestCos_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	c := &Cos[float32]{engine: engine}

	in, _ := tensor.New[float32]([]int{3}, []float32{0, float32(math.Pi / 2), float32(math.Pi)})
	out, err := c.Forward(context.Background(), in)
	if err != nil {
		t.Fatalf("Cos Forward: %v", err)
	}

	got := out.Data()
	want := []float32{1, 0, -1}
	for i := range want {
		if diff := got[i] - want[i]; diff > 1e-5 || diff < -1e-5 {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCos_WrongInputCount(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	c := &Cos[float32]{engine: engine}

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	_, err := c.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for 2 inputs")
	}
}
