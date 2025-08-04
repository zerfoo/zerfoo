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
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	sigmoid := NewSigmoid[int](engine, numeric.IntOps{})
	testActivationBackward(t, sigmoid)
}
