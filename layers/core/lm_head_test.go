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

	hiddenDim := 64
	vocabSize := 1000

	lmHead, err := NewLMHead[float32](engine, ops, hiddenDim, vocabSize)
	if err != nil {
		t.Fatalf("Failed to create LMHead: %v", err)
	}

	batchSize := 2
	seqLen := 10
	inputShape := []int{batchSize, seqLen, hiddenDim}
	inputData := make([]float32, batchSize*seqLen*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := lmHead.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	expectedShape := []int{batchSize, seqLen, vocabSize}
	if !testutils.IntSliceEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected output shape %v, got %v", expectedShape, output.Shape())
	}
}
