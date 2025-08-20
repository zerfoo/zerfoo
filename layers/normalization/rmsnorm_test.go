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

	layer, err := NewRMSNorm[float32]("test_rmsnorm", engine, ops, modelDim, epsilon)
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

	output, err := layer.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.CompareTensorsApprox(t, output, expected, epsilon) {
		t.Errorf("Output tensor does not match expected. Got %v, want %v", output.Data(), expected.Data())
	}
}
