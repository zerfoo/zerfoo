package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestGlobalAttention_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 4
	base := 10000.0
	maxSeqLen := 512

	globalAttn, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention: %v", err)
	}

	batchSize := 1
	seqLen := 5
	inputShape := []int{batchSize, seqLen, modelDim}
	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := globalAttn.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.IntSliceEqual(output.Shape(), inputShape) {
		t.Errorf("Expected output shape %v, got %v", inputShape, output.Shape())
	}
}