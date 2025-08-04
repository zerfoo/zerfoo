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

func TestTanh_Error(_ *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test Tanh error
	_ = NewTanh[float32](engine, ops)
}

func TestTanh_Forward_Error(t *testing.T) {
	engine := &testutils.MockEngine[float32]{Err: fmt.Errorf("test error")}
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	// Test Tanh forward error
	tanh := NewTanh[float32](engine, ops)
	_, err := tanh.Forward(context.Background(), input)
	testutils.AssertError(t, err, "expected Tanh.Forward to return an error")
}

func TestTanh_OutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 5}, []float32{1, 2, 3, 4, 5})

	// Test Tanh output shape
	tanh := NewTanh[float32](engine, ops)
	_, _ = tanh.Forward(context.Background(), input)
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, tanh.OutputShape()), "expected output shape to be equal")
}

func TestTanh_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test Tanh parameters
	tanh := NewTanh[float32](engine, ops)
	testutils.AssertEqual(t, 0, len(tanh.Parameters()), "expected parameters to be empty, got %v")
}
