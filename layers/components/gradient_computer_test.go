package components

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// setupGradientComputerTest creates a common test setup for gradient computer tests.
func setupGradientComputerTest(_ *testing.T) (*LinearGradientComputer[float32], context.Context) {
	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](ops)
	computer := NewLinearGradientComputer(engine)
	ctx := context.Background()

	return computer, ctx
}

func TestLinearGradientComputer_ComputeWeightGradient(t *testing.T) {
	computer, ctx := setupGradientComputerTest(t)

	// Create test tensors
	// Input: (2x3), Output gradient: (2x2)
	// Expected weight gradient shape: (3x2)
	inputData := []float32{1, 2, 3, 4, 5, 6}
	outputGradData := []float32{1, 1, 1, 1}

	input, err := tensor.New([]int{2, 3}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	outputGrad, err := tensor.New([]int{2, 2}, outputGradData)
	if err != nil {
		t.Fatalf("Failed to create output gradient tensor: %v", err)
	}

	weightGrad, err := computer.ComputeWeightGradient(ctx, input, outputGrad)
	if err != nil {
		t.Fatalf("Weight gradient computation failed: %v", err)
	}

	expectedShape := []int{3, 2}
	if !equalIntSlices(weightGrad.Shape(), expectedShape) {
		t.Errorf("Expected weight gradient shape %v, got %v", expectedShape, weightGrad.Shape())
	}
}

func TestLinearGradientComputer_ComputeInputGradient(t *testing.T) {
	computer, ctx := setupGradientComputerTest(t)

	// Create test tensors
	// Weights: (3x2), Output gradient: (2x2)
	// Expected input gradient shape: (2x3)
	weightsData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	outputGradData := []float32{1, 1, 1, 1}

	weights, err := tensor.New([]int{3, 2}, weightsData)
	if err != nil {
		t.Fatalf("Failed to create weights tensor: %v", err)
	}

	outputGrad, err := tensor.New([]int{2, 2}, outputGradData)
	if err != nil {
		t.Fatalf("Failed to create output gradient tensor: %v", err)
	}

	inputGrad, err := computer.ComputeInputGradient(ctx, weights, outputGrad)
	if err != nil {
		t.Fatalf("Input gradient computation failed: %v", err)
	}

	expectedShape := []int{2, 3}
	if !equalIntSlices(inputGrad.Shape(), expectedShape) {
		t.Errorf("Expected input gradient shape %v, got %v", expectedShape, inputGrad.Shape())
	}
}

func TestLinearGradientComputer_ComputeBothGradients(t *testing.T) {
	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](ops)
	computer := NewLinearGradientComputer(engine)

	// Create test tensors
	inputData := []float32{1, 2, 3, 4, 5, 6}
	weightsData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	outputGradData := []float32{1, 1, 1, 1}

	input, err := tensor.New([]int{2, 3}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	weights, err := tensor.New([]int{3, 2}, weightsData)
	if err != nil {
		t.Fatalf("Failed to create weights tensor: %v", err)
	}

	outputGrad, err := tensor.New([]int{2, 2}, outputGradData)
	if err != nil {
		t.Fatalf("Failed to create output gradient tensor: %v", err)
	}

	ctx := context.Background()
	weightGrad, inputGrad, err := computer.ComputeBothGradients(ctx, input, weights, outputGrad)
	if err != nil {
		t.Fatalf("Both gradients computation failed: %v", err)
	}

	expectedWeightShape := []int{3, 2}
	expectedInputShape := []int{2, 3}

	if !equalIntSlices(weightGrad.Shape(), expectedWeightShape) {
		t.Errorf("Expected weight gradient shape %v, got %v", expectedWeightShape, weightGrad.Shape())
	}

	if !equalIntSlices(inputGrad.Shape(), expectedInputShape) {
		t.Errorf("Expected input gradient shape %v, got %v", expectedInputShape, inputGrad.Shape())
	}
}
