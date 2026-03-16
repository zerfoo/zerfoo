package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestNewTiedLMHead(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	hiddenDim := 8
	vocabSize := 4

	// Simulate a token embedding weight [vocabSize, hiddenDim].
	embedData := make([]float32, vocabSize*hiddenDim)
	for i := range embedData {
		embedData[i] = float32(i%7+1) * 0.01
	}
	embedWeight, err := tensor.New[float32]([]int{vocabSize, hiddenDim}, embedData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	lmHead := NewTiedLMHead[float32](engine, embedWeight)

	// Forward: input [1, 2, hiddenDim] -> output [1, 2, vocabSize]
	inputData := make([]float32, 2*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i%5+1) * 0.1
	}
	input, err := tensor.New[float32]([]int{1, 2, hiddenDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	output, err := lmHead.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if output.Shape()[0] != 1 || output.Shape()[1] != 2 || output.Shape()[2] != vocabSize {
		t.Errorf("output shape = %v, want [1, 2, %d]", output.Shape(), vocabSize)
	}

	// Verify output matches manual matmul of input * embed^T.
	// embedWeight is [vocabSize, hiddenDim], so we need input * embed^T.
	transposed, err := engine.Transpose(ctx, embedWeight, []int{1, 0})
	if err != nil {
		t.Fatalf("Transpose failed: %v", err)
	}
	// Reshape input to [2, hiddenDim] for matmul
	reshapedInput, err := engine.Reshape(ctx, input, []int{2, hiddenDim})
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}
	expected, err := engine.MatMul(ctx, reshapedInput, transposed)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}
	expectedReshaped, err := engine.Reshape(ctx, expected, []int{1, 2, vocabSize})
	if err != nil {
		t.Fatalf("Reshape failed: %v", err)
	}

	for i, v := range output.Data() {
		diff := v - expectedReshaped.Data()[i]
		if diff > 1e-5 || diff < -1e-5 {
			t.Errorf("output[%d] = %f, want %f", i, v, expectedReshaped.Data()[i])
			break
		}
	}
}

func TestTiedLMHead_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	embedWeight, _ := tensor.New[float32]([]int{4, 8}, nil)
	lmHead := NewTiedLMHead[float32](engine, embedWeight)

	// Tied LMHead should have no trainable parameters of its own
	// (the embedding owns the weight).
	params := lmHead.Parameters()
	if len(params) != 0 {
		t.Errorf("TiedLMHead should have 0 parameters, got %d", len(params))
	}
}

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
