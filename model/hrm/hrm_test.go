// Package hrm_test contains tests for the HRM model.
package hrm_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/model/hrm"
	"github.com/zerfoo/zerfoo/numeric"
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
