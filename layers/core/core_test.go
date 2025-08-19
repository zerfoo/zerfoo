package core

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/components"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestLinear(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	// Use Xavier initialization as default
	layer, err := NewLinear("linear", engine, ops, 10, 5)
	testutils.AssertNoError(t, err, "expected no error when creating linear layer, got %v")

	inputData := make([]float32, 10)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{1, 10}, inputData)
	testutils.AssertNoError(t, err, "expected no error when creating input tensor, got %v")

	// Check forward pass
	output, _ := layer.Forward(context.Background(), input)
	testutils.AssertNotNil(t, output, "expected output to not be nil")

	// Check backward pass
	gradOutput, err := tensor.New[float32]([]int{1, 5}, []float32{1, 1, 1, 1, 1})
	testutils.AssertNoError(t, err, "expected no error when creating gradient output tensor, got %v")
	gradInput, _ := layer.Backward(context.Background(), gradOutput)
	testutils.AssertNotNil(t, gradInput, "expected gradient input to not be nil")

	// Test the SetName method of the linear layer
	layer.SetName("new_linear")
	testutils.AssertEqual(t, "new_linear_weights", layer.weights.Name, "expected weights name %q, got %q")

	// Test the OutputShape method of the linear layer
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 5}, layer.OutputShape()), "expected output shape to be equal")

	// Test the Parameters method of the linear layer
	testutils.AssertNotNil(t, layer.Parameters(), "expected parameters to not be nil")

	// Test the error case for NewLinear
	_, err = NewLinear("", engine, ops, 10, 5)
	testutils.AssertError(t, err, "expected an error for empty name, got nil")

	// Test the error case for NewLinear with a tensor error
	initializer := components.NewXavierInitializer(ops)
	_, err = NewLinearWithFactories("linear", engine, ops, 10, 5, initializer, func(_ []int, _ []float32) (*tensor.Tensor[float32], error) {
		return nil, errors.New("tensor error")
	}, graph.NewParameter[float32])
	testutils.AssertError(t, err, "expected an error for tensor creation failure, got nil")

	// Test the error case for NewLinear with a parameter error
	_, err = NewLinearWithFactories("linear", engine, ops, 10, 5, initializer, tensor.New[float32], func(_ string, _ *tensor.Tensor[float32], _ func(shape []int, data []float32) (*tensor.Tensor[float32], error)) (*graph.Parameter[float32], error) {
		return nil, errors.New("parameter error")
	})
	testutils.AssertError(t, err, "expected an error for parameter creation failure, got nil")

	// Test the panic case for the Forward method of the linear layer
	// Note: With component-based architecture, we can't easily mock the engine
	// This test would need to be redesigned to test error conditions properly
	_, err = layer.Forward(context.Background(), nil) // This will panic due to nil input
	testutils.AssertError(t, err, "expected Forward to panic on nil input")
}

type mockEngine struct {
	compute.Engine[float32]
	Err error
}

func (e *mockEngine) Add(_ context.Context, _, _ *tensor.Tensor[float32], _ ...*tensor.Tensor[float32]) (*tensor.Tensor[float32], error) {
	return nil, e.Err
}

func (e *mockEngine) MatMul(_ context.Context, _, _ *tensor.Tensor[float32], _ ...*tensor.Tensor[float32]) (*tensor.Tensor[float32], error) {
	return nil, e.Err
}
