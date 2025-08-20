package transformer

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestTransformerBlockForward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 8
	ffnDim := 256
	epsilon := float32(1e-6)
	base := 10000.0
	maxSeqLen := 512

	block, err := NewTransformerBlock[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, ffnDim, epsilon, base, maxSeqLen)
	if err != nil {
		t.Fatalf("Failed to create TransformerBlock: %v", err)
	}

	batchSize := 2
	seqLen := 10
	inputShape := []int{batchSize, seqLen, modelDim}
	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01 // Simple dummy data
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := block.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	// Create an expected output tensor with the same shape as input for initial shape check.
	// The actual values will be different, but for a basic forward pass test,
	// we primarily care about shape and no panics/errors.
	expectedOutputTensor, err := tensor.New[float32](inputShape, make([]float32, batchSize*seqLen*modelDim))
	if err != nil {
		t.Fatalf("Failed to create expected output tensor: %v", err)
	}

	// Compare shapes and ensure no panics.
	// CompareTensorsApprox already checks shapes internally.
	if !testutils.CompareTensorsApprox(t, output, expectedOutputTensor, epsilon) {
		// CompareTensorsApprox will log the shape mismatch if any.
		// We don't need to do anything here other than let the test fail.
	}

	// TODO: Add more comprehensive checks, e.g., against known good values or properties
	// For now, just ensure it runs without panicking and shape is correct.
}