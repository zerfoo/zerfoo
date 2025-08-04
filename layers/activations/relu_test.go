package activations

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestReLU_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	relu := NewReLU[int](engine, numeric.IntOps{})
	testActivationForward(t, relu)
}

func TestReLU_Backward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	relu := NewReLU[int](engine, numeric.IntOps{})
	testActivationBackward(t, relu)
}
