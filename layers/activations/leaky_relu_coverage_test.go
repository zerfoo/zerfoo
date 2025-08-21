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

func TestLeakyReLU_Error(_ *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test LeakyReLU error
	_ = NewLeakyReLU[float32](engine, ops, WithAlpha[float32](0.01))
}

func TestLeakyReLU_Forward_Error(t *testing.T) {
	engine := &testutils.MockEngine[float32]{Err: errors.New("test error")}
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 1}, []float32{1})

	// Test LeakyReLU forward error
	leakyrelu := NewLeakyReLU[float32](engine, ops, WithAlpha[float32](0.01))
	_, err := leakyrelu.Forward(context.Background(), input)
	testutils.AssertError(t, err, "expected LeakyReLU.Forward to return an error")
}

func TestLeakyReLU_OutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	input, _ := tensor.New[float32]([]int{1, 5}, []float32{1, 2, 3, 4, 5})

	// Test LeakyReLU output shape
	leakyrelu := NewLeakyReLU[float32](engine, ops, WithAlpha[float32](0.01))
	_, _ = leakyrelu.Forward(context.Background(), input)
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, leakyrelu.OutputShape()), "expected output shape to be equal")
}

func TestLeakyReLU_Parameters(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Test LeakyReLU parameters
	leakyrelu := NewLeakyReLU[float32](engine, ops, WithAlpha[float32](0.01))
	testutils.AssertEqual(t, 0, len(leakyrelu.Parameters()), "expected parameters to be empty, got %v")
}
