package training

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestGemmaModel_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	vocabSize := 1000
	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 4
	ffnDim := 256
	epsilon := float32(1e-6)
	base := 10000.0
	maxSeqLen := 512
	numLayers := 6
	localWindowSize := 3
	globalInterval := 5

	model, err := NewGemmaModel[float32](engine, ops, vocabSize, modelDim, numQueryHeads, numKeyValueHeads, ffnDim, epsilon, base, maxSeqLen, numLayers, localWindowSize, globalInterval)
	if err != nil {
		t.Fatalf("Failed to create GemmaModel: %v", err)
	}

	batchSize := 2
	seqLen := 10
	inputShape := []int{batchSize, seqLen}
	inputData := make([]int, batchSize*seqLen)
	for i := range inputData {
		inputData[i] = i % vocabSize // Simple dummy data
	}
	inputTensor, err := tensor.New[int](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := model.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	expectedShape := []int{batchSize, seqLen, vocabSize}
	if !testutils.IntSliceEqual(output.Shape(), expectedShape) {
		t.Errorf("Expected output shape %v, got %v", expectedShape, output.Shape())
	}
}
