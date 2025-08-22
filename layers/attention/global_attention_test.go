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
	globalAttn, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithGlobalAttentionBase(10000.0), WithGlobalAttentionMaxSeqLen(512))
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

func TestGlobalAttention_FunctionalOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 4

	// Test default options
	globalAttnDefault, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads)
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention with defaults: %v", err)
	}
	if globalAttnDefault == nil {
		t.Error("Expected non-nil GlobalAttention with default options")
	}

	// Test custom base
	customBase := 5000.0
	globalAttnCustomBase, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithGlobalAttentionBase(customBase))
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention with custom base: %v", err)
	}
	if globalAttnCustomBase == nil {
		t.Error("Expected non-nil GlobalAttention with custom base")
	}

	// Test custom max sequence length
	customMaxSeqLen := 1024
	globalAttnCustomSeqLen, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithGlobalAttentionMaxSeqLen(customMaxSeqLen))
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention with custom max seq len: %v", err)
	}
	if globalAttnCustomSeqLen == nil {
		t.Error("Expected non-nil GlobalAttention with custom max seq len")
	}

	// Test both custom options
	globalAttnBoth, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithGlobalAttentionBase(customBase), WithGlobalAttentionMaxSeqLen(customMaxSeqLen))
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention with both custom options: %v", err)
	}
	if globalAttnBoth == nil {
		t.Error("Expected non-nil GlobalAttention with both custom options")
	}
}

func TestGlobalAttention_DefaultOptions(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 32
	numQueryHeads := 4
	numKeyValueHeads := 2

	// Test with default options (should work without any options)
	globalAttn, err := NewGlobalAttention[float32](engine, ops, modelDim, numQueryHeads, numKeyValueHeads)
	if err != nil {
		t.Fatalf("Failed to create GlobalAttention with defaults: %v", err)
	}

	// Test forward pass with defaults
	batchSize := 1
	seqLen := 3
	inputShape := []int{batchSize, seqLen, modelDim}
	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = 0.1
	}
	inputTensor, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := globalAttn.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass with defaults failed: %v", err)
	}

	if !testutils.IntSliceEqual(output.Shape(), inputShape) {
		t.Errorf("Expected output shape %v, got %v", inputShape, output.Shape())
	}
}
