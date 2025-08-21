package activations

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestTanh_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	tanh := NewTanh[int](engine, numeric.IntOps{})
	testActivationForward(t, tanh)
}

func TestTanh_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tanh := NewTanh[float32](engine, numeric.Float32Ops{})
	testActivationBackward(t, tanh, []float32{-2, -1, 0, 1, 2}, []float32{0.070650816, 0.41997433, 1, 0.41997433, 0.070650816})
}
