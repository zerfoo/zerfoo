package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestRotaryEmbedding_Forward_Shape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batch := 2
	seq := 5
	headDim := 8

	inp, err := tensor.New[float32]([]int{batch, seq, headDim}, nil)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%17) / 10.0
	}

	re := NewRotaryEmbedding[float32](engine)
	out, err := re.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	expected := []int{batch, seq, headDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Fatalf("unexpected output shape: got %v want %v", out.Shape(), expected)
	}
}

func TestRotaryEmbedding_Backward_Shape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batch := 3
	seq := 7
	headDim := 16

	inp, err := tensor.New[float32]([]int{batch, seq, headDim}, nil)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%11) / 10.0
	}

	re := NewRotaryEmbedding[float32](engine)
	out, err := re.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}

	dOut, err := tensor.New[float32](out.Shape(), nil)
	if err != nil {
		t.Fatalf("failed to create dOut: %v", err)
	}
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	dIn, err := re.Backward(context.Background(), dOut, inp)
	if err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	if len(dIn) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(dIn))
	}
	if !testutils.IntSliceEqual(inp.Shape(), dIn[0].Shape()) {
		t.Fatalf("unexpected grad shape: got %v want %v", dIn[0].Shape(), inp.Shape())
	}
}
