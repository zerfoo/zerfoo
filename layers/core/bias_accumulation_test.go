package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestBiasAccumulatesSharedParameterGradient(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	bias, err := NewBias("shared", engine, engine.Ops(), 2)
	if err != nil {
		t.Fatal(err)
	}
	gradient, err := tensor.New([]int{2, 2}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		if _, err := bias.Backward(context.Background(), types.FullBackprop, gradient); err != nil {
			t.Fatal(err)
		}
	}
	for i, want := range []float32{8, 12} {
		if got := bias.Parameters()[0].Gradient.Data()[i]; got != want {
			t.Fatalf("gradient[%d]=%g want %g", i, got, want)
		}
	}
}
