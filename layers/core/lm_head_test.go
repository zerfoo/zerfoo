package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestLMHead_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	vocabSize := 1000

	head, err := NewLMHead[float32]("test_lm_head", engine, ops, modelDim, vocabSize)
	if err != nil {
		t.Fatalf("Failed to create LMHead: %v", err)
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

	output, err := head.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	expectedShape := []int{batchSize, seqLen, vocabSize}
	if !testutils.IntSliceEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected output shape %v, got %v", expectedShape, output.Shape())
	}
}
