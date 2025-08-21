package activations

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestSigmoid_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	sigmoid := NewSigmoid[int](engine, numeric.IntOps{})
	testActivationForward(t, sigmoid)
}

func TestSigmoid_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	sigmoid := NewSigmoid[float32](engine, numeric.Float32Ops{})
	testActivationBackward(t, sigmoid, []float32{-2, -1, 0, 1, 2}, []float32{0.10499358, 0.19661193, 0.25, 0.19661193, 0.10499358})
}
