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
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	tanh := NewTanh[int](engine, numeric.IntOps{})
	testActivationBackward(t, tanh)
}
