package activations

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestSigmoid_Error(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test Sigmoid error
	_ = NewSigmoid[float32](engine, ops)
}

func TestSigmoid_Forward_Error(t *testing.T) {
	engine := &testutils.MockEngine[float32]{Err: fmt.Errorf("test error")}
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	// Test Sigmoid forward error
	sigmoid := NewSigmoid[float32](engine, ops)
	_, err := sigmoid.Forward(context.Background(), input)
	testutils.AssertError(t, err, "expected Sigmoid.Forward to return an error")
}

func TestSigmoid_OutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 5}, []float32{1, 2, 3, 4, 5})

	// Test Sigmoid output shape
	sigmoid := NewSigmoid[float32](engine, ops)
	_, _ = sigmoid.Forward(context.Background(), input)
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, sigmoid.OutputShape()), "expected output shape to be equal")
}

func TestSigmoid_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test Sigmoid parameters
	sigmoid := NewSigmoid[float32](engine, ops)
	testutils.AssertEqual(t, 0, len(sigmoid.Parameters()), "expected parameters to be empty, got %v")
}
