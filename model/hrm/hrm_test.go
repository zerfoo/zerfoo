// Package hrm_test contains tests for the HRM model.
package hrm_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/model/hrm"
	"github.com/zerfoo/ztensor/numeric"
)

func TestHRM_Build(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	inputDim := 16
	outputDim := 8
	N := 2
	T := 3
	numHeads := 2

	hAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create H-attention: %v", err)
	}

	lAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create L-attention: %v", err)
	}

	model, err := hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, inputDim, outputDim, hAttention, lAttention)
	if err != nil {
		t.Fatalf("failed to create HRM model: %v", err)
	}

	builder := graph.NewBuilder[float32](engine)
	inputShape := []int{1, inputDim}
	stateShape := []int{1, modelDim}

	input := builder.Input(inputShape)
	hStateIn := builder.Input(stateShape)
	lStateIn := builder.Input(stateShape)

	output, err := model.Build(builder, N, T, input, hStateIn, lStateIn)
	if err != nil {
		t.Fatalf("failed to build HRM graph: %v", err)
	}

	if output == nil {
		t.Fatal("output node is nil")
	}
}

func TestHRM_Forward_RecurrentLoop(t *testing.T) {
	// Setup
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	inputDim := 10   // 10 features
	outputDim := 1   // single output
	nSteps := 2
	tSteps := 3
	numHeads := 2
	batchSize := 32

	// Create attention layers
	hAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create H-attention: %v", err)
	}
	lAttention, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create L-attention: %v", err)
	}

	// Create HRM
	model, err := hrm.NewHRM[float32](engine, ops, modelDim, ffnDim, inputDim, outputDim, hAttention, lAttention)
	if err != nil {
		t.Fatalf("failed to create HRM: %v", err)
	}

	// Create batch input [32, 10]
	inputData := make([]float32, batchSize*inputDim)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batchSize, inputDim}, inputData)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	// Forward pass
	ctx := context.Background()
	output, err := model.Forward(ctx, nSteps, tSteps, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape [32, 1]
	shape := output.Shape()
	if len(shape) != 2 || shape[0] != batchSize || shape[1] != outputDim {
		t.Errorf("output shape = %v, want [%d, %d]", shape, batchSize, outputDim)
	}

	// Verify finite values
	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("output[%d] = %v, want finite value", i, v)
		}
	}
}
