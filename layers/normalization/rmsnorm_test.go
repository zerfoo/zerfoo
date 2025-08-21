package normalization

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestRMSNormLayer_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 4
	epsilon := float32(1e-6)

	rmsnorm, err := NewRMSNorm[float32]("test", engine, ops, 4, WithRMSNormEpsilon[float32](epsilon))
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	input, err := tensor.New[float32]([]int{2, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Manually calculate expected output
	// Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = 2.7386
	// Normalized: [0.3651, 0.7303, 1.0954, 1.4606]
	// Row 2: [5, 6, 7, 8], RMS = sqrt((25+36+49+64)/4) = 6.5954
	// Normalized: [0.7581, 0.9097, 1.0613, 1.2129]
	expectedData := []float32{
		0.36514837, 0.73029673, 1.0954452, 1.4605935,
		0.75809807, 0.9097177, 1.0613372, 1.2129569,
	}
	expected, err := tensor.New[float32]([]int{2, modelDim}, expectedData)
	if err != nil {
		t.Fatalf("Failed to create expected tensor: %v", err)
	}

	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.CompareTensorsApprox(t, output, expected, epsilon) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expected.Data())
	}
}

func TestRMSNorm_CustomEpsilon(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	
	modelDim := 4
	customEpsilon := float32(1e-3)
	
	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim, WithRMSNormEpsilon[float32](customEpsilon))
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}
	
	// Create input tensor
	inputData := []float32{1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0}
	input, err := tensor.New[float32]([]int{2, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	
	// Forward pass should work without error
	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	
	// Verify output shape
	expectedShape := []int{2, modelDim}
	if len(output.Shape()) != len(expectedShape) {
		t.Errorf("Output shape mismatch. Got %v, want %v", output.Shape(), expectedShape)
	}
	for i, dim := range output.Shape() {
		if dim != expectedShape[i] {
			t.Errorf("Output shape mismatch at dimension %d. Got %d, want %d", i, dim, expectedShape[i])
		}
	}
}

func TestRMSNorm_DefaultEpsilon(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	
	modelDim := 4
	
	// Create RMSNorm without epsilon option (should use default)
	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim)
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}
	
	// Create input tensor
	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	input, err := tensor.New[float32]([]int{1, modelDim}, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	
	// Forward pass should work without error
	output, err := rmsnorm.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}
	
	// Verify output shape
	expectedShape := []int{1, modelDim}
	if len(output.Shape()) != len(expectedShape) {
		t.Errorf("Output shape mismatch. Got %v, want %v", output.Shape(), expectedShape)
	}
	for i, dim := range output.Shape() {
		if dim != expectedShape[i] {
			t.Errorf("Output shape mismatch at dimension %d. Got %d, want %d", i, dim, expectedShape[i])
		}
	}
}

func TestRMSNorm_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	
	modelDim := 4
	
	rmsnorm, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim)
	if err != nil {
		t.Fatalf("Failed to create RMSNorm layer: %v", err)
	}
	
	params := rmsnorm.Parameters()
	if len(params) != 1 {
		t.Errorf("Expected 1 parameter, got %d", len(params))
	}
	
	// Check weight parameter
	weight := params[0]
	if weight.Name != "test_rmsnorm_gain" {
		t.Errorf("Expected parameter name 'test_rmsnorm_gain', got '%s'", weight.Name)
	}
}
