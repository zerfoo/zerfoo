package activations

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestReLU_Error(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test ReLU error
	_ = NewReLU[float32](engine, ops)
}

func TestReLU_Forward_Error(t *testing.T) {
	engine := &testutils.MockEngine[float32]{Err: errors.New("test error")}
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	// Test ReLU forward error
	relu := NewReLU[float32](engine, ops)
	_, err := relu.Forward(context.Background(), input)
	testutils.AssertError(t, err, "expected ReLU.Forward to return an error")
}

func TestReLU_OutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 5}, []float32{1, 2, 3, 4, 5})

	// Test ReLU output shape
	relu := NewReLU[float32](engine, ops)
	_, _ = relu.Forward(context.Background(), input)
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, relu.OutputShape()), "expected output shape to be equal")
}

func TestReLU_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test ReLU parameters
	relu := NewReLU[float32](engine, ops)
	testutils.AssertEqual(t, 0, len(relu.Parameters()), "expected parameters to be empty, got %v")
}
