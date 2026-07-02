package attention

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestAttentionHead(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{}) // Provide arithmetic implementation
	batchSize := 2
	seqLen := 5
	inputDim := 10
	headDim := 8

	// Create an AttentionHead instance
	attentionHead, err := NewAttentionHead[float32](engine, inputDim, headDim)
	if err != nil {
		t.Fatalf("NewAttentionHead failed: %v", err)
	}

	// Create a dummy input tensor
	inputShape := []int{batchSize, seqLen, inputDim}

	inputTensor, err := tensor.New[float32](inputShape, nil) // Corrected tensor creation
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	// Fill with some dummy data
	for i := range batchSize * seqLen * inputDim {
		inputTensor.Data()[i] = float32(i)
	}

	// Perform forward pass
	output, err := attentionHead.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("AttentionHead Forward failed: %v", err)
	}

	// Check output shape
	expectedOutputShape := []int{batchSize, seqLen, headDim}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedOutputShape, output.Shape()),
		fmt.Sprintf("Expected output shape %v, got %v", expectedOutputShape, output.Shape()))

	// Check if parameters are correctly exposed
	params := attentionHead.Parameters()
	testutils.AssertTrue(t, len(params) == 6, fmt.Sprintf("Expected 6 parameters (Q, K, V weights and biases), got %d", len(params)))

	// Test with different input dimensions
	attentionHead2, err := NewAttentionHead[float32](engine, 16, 8)
	if err != nil {
		t.Fatalf("NewAttentionHead failed: %v", err)
	}
	inputShape2 := []int{batchSize, seqLen, 16}

	inputTensor2, err := tensor.New[float32](inputShape2, nil) // Corrected tensor creation
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output2, err := attentionHead2.Forward(context.Background(), inputTensor2)
	if err != nil {
		t.Fatalf("AttentionHead2 Forward failed: %v", err)
	}

	expectedOutputShape2 := []int{batchSize, seqLen, 8}
	testutils.AssertTrue(t, testutils.IntSliceEqual(expectedOutputShape2, output2.Shape()),
		fmt.Sprintf("Expected output shape %v, got %v", expectedOutputShape2, output2.Shape()))
}
