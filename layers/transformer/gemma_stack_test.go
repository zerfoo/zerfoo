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

func TestGemmaStack_Forward(t *testing.T) {
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
	numLayers := 6
	localWindowSize := 3
	globalInterval := 5

	stack, err := NewGemmaStack[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, ffnDim, epsilon, base, maxSeqLen, numLayers, localWindowSize, globalInterval)
	if err != nil {
		t.Fatalf("Failed to create GemmaStack: %v", err)
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

	output, err := stack.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.IntSliceEqual(output.Shape(), inputShape) {
		t.Errorf("Expected output shape %v, got %v", inputShape, output.Shape())
	}

	// Check the types of attention layers
	numGlobal := 0
	numLocal := 0
	for _, layer := range stack.layers {
		block := layer.(*Block[float32])
		switch block.attention.(type) {
		case *attention.GlobalAttention[float32]:
			numGlobal++
		case *attention.LocalAttention[float32]:
			numLocal++
		}
	}

	expectedGlobal := numLayers / globalInterval
	expectedLocal := numLayers - expectedGlobal
	if numGlobal != expectedGlobal {
		t.Errorf("Expected %d global attention layers, got %d", expectedGlobal, numGlobal)
	}
	if numLocal != expectedLocal {
		t.Errorf("Expected %d local attention layers, got %d", expectedLocal, numLocal)
	}
}
