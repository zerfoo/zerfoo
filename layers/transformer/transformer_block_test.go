package transformer

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

func TestTransformerBlock_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 64
	numQueryHeads := 8
	numKeyValueHeads := 4
	ffnDim := 64
	epsilon := float32(1e-6)
	base := 10000.0
	maxSeqLen := 512

	// First test the attention mechanism alone
	gqa, err := attention.NewGroupedQueryAttention[float32](
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		attention.WithRopeBase[float32](base),
		attention.WithMaxSeqLen[float32](maxSeqLen),
	)
	if err != nil {
		t.Fatalf("Failed to create GroupedQueryAttention: %v", err)
	}

	batchSize := 2
	seqLen := 10
	inputShape := []int{batchSize, seqLen, modelDim}

	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}

	inputTensor, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}

	// Test attention output shape
	attnOutput, err := gqa.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Attention forward failed: %v", err)
	}

	if !testutils.IntSliceEqual(attnOutput.Shape(), inputShape) {
		t.Fatalf("Attention output shape mismatch: expected %v, got %v", inputShape, attnOutput.Shape())
	}

	block, err := NewTransformerBlock[float32](engine, ops, modelDim, ffnDim, gqa, WithEpsilon[float32](epsilon))
	if err != nil {
		t.Fatalf("Failed to create TransformerBlock: %v", err)
	}

	blockOutput, err := block.Forward(ctx, inputTensor)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	if !testutils.IntSliceEqual(blockOutput.Shape(), inputShape) {
		t.Errorf("Expected output shape %v, got %v", inputShape, blockOutput.Shape())
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

// newTestBlock creates a Block with standard test configuration.
func newTestBlock(t *testing.T) (*Block[float32], []int) {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 64
	ffnDim := 64

	gqa, err := attention.NewGroupedQueryAttention[float32](
		engine, ops, modelDim, 8, 4,
		attention.WithRopeBase[float32](10000.0),
		attention.WithMaxSeqLen[float32](512),
	)
	if err != nil {
		t.Fatalf("Failed to create attention: %v", err)
	}

	block, err := NewTransformerBlock[float32](engine, ops, modelDim, ffnDim, gqa,
		WithEpsilon[float32](1e-6))
	if err != nil {
		t.Fatalf("Failed to create block: %v", err)
	}

	inputShape := []int{2, 10, modelDim}
	return block, inputShape
}

func TestTransformerBlock_Backward(t *testing.T) {
	ctx := context.Background()
	block, inputShape := newTestBlock(t)

	batchSize, seqLen, modelDim := inputShape[0], inputShape[1], inputShape[2]

	// Create input and run forward
	inputData := make([]float32, batchSize*seqLen*modelDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32](inputShape, inputData)
	if err != nil {
		t.Fatalf("Failed to create input: %v", err)
	}

	output, err := block.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Create gradient same shape as output (ones)
	gradData := make([]float32, batchSize*seqLen*modelDim)
	for i := range gradData {
		gradData[i] = 1.0
	}
	dOut, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		t.Fatalf("Failed to create gradient: %v", err)
	}

	// Run backward
	grads, err := block.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// Verify: single gradient tensor returned
	if len(grads) != 1 {
		t.Fatalf("Expected 1 gradient, got %d", len(grads))
	}

	// Verify: gradient is non-nil
	if grads[0] == nil {
		t.Fatal("Input gradient is nil")
	}

	// Verify: gradient shape matches input
	if !testutils.IntSliceEqual(grads[0].Shape(), inputShape) {
		t.Errorf("Gradient shape %v != input shape %v", grads[0].Shape(), inputShape)
	}

	// Verify: gradients are non-zero (at least some elements)
	gradValues := grads[0].Data()
	hasNonZero := false
	for _, v := range gradValues {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("All gradient values are zero; expected non-zero gradients")
	}
}

func TestTransformerBlock_BackwardShapes(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name  string
		batch int
		seq   int
		dim   int
	}{
		{"small", 1, 4, 64},
		{"medium", 2, 10, 64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			gqa, err := attention.NewGroupedQueryAttention[float32](
				engine, ops, tt.dim, 8, 4,
				attention.WithRopeBase[float32](10000.0),
				attention.WithMaxSeqLen[float32](512),
			)
			if err != nil {
				t.Fatalf("Failed to create attention: %v", err)
			}

			block, err := NewTransformerBlock[float32](engine, ops, tt.dim, tt.dim, gqa,
				WithEpsilon[float32](1e-6))
			if err != nil {
				t.Fatalf("Failed to create block: %v", err)
			}

			shape := []int{tt.batch, tt.seq, tt.dim}
			n := tt.batch * tt.seq * tt.dim
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i%100) * 0.001
			}
			input, err := tensor.New[float32](shape, data)
			if err != nil {
				t.Fatalf("Failed to create input: %v", err)
			}

			out, err := block.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			gradData := make([]float32, n)
			for i := range gradData {
				gradData[i] = 1.0
			}
			dOut, err := tensor.New[float32](out.Shape(), gradData)
			if err != nil {
				t.Fatalf("Failed to create dOut: %v", err)
			}

			grads, err := block.Backward(ctx, types.FullBackprop, dOut)
			if err != nil {
				t.Fatalf("Backward failed: %v", err)
			}

			if !testutils.IntSliceEqual(grads[0].Shape(), shape) {
				t.Errorf("Gradient shape %v != expected %v", grads[0].Shape(), shape)
			}
		})
	}
}
