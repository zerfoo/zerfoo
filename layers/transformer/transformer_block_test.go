package transformer

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestTransformerBlock_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 4
	ffnDim := 256
	epsilon := float32(1e-6)
	base := 10000.0
	maxSeqLen := 512

	gqa, err := attention.NewGroupedQueryAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
	if err != nil {
		t.Fatalf("Failed to create GroupedQueryAttention: %v", err)
	}

	block, err := NewTransformerBlock[float32](engine, ops, modelDim, ffnDim, epsilon, gqa)
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

	if !testutils.IntSliceEqual(output.Shape(), inputShape) {
		t.Errorf("Expected output shape %v, got %v", inputShape, output.Shape())
	}

	// Check number of parameters
	// rmsNorm1: 1
	// gqa: 8 (wq, wk, wv, wo weights and biases)
	// rmsNormPostAttention: 1
	// rmsNorm2: 1
	// ffn: 6 (w1, w3, w2 weights and biases)
	// Total: 17
	expectedNumParams := 17
	if len(block.Parameters()) != expectedNumParams {
		t.Errorf("Expected %d parameters, got %d", expectedNumParams, len(block.Parameters()))
	}
}
