package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestLocalAttention_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 3)
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}
	if got := la.OpType(); got != "LocalAttention" {
		t.Errorf("OpType() = %q, want %q", got, "LocalAttention")
	}
}

func TestLocalAttention_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 5)
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}
	attrs := la.Attributes()
	if attrs["window_size"] != 5 {
		t.Errorf("window_size = %v, want 5", attrs["window_size"])
	}
	if attrs["model_dim"] != 16 {
		t.Errorf("model_dim = %v, want 16", attrs["model_dim"])
	}
}

func TestLocalAttention_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 3)
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}
	params := la.Parameters()
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}

func TestLocalAttention_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 3,
		WithLocalMaxSeqLen[float32](8))
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = 0.01
	}
	_, err = la.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	shape := la.OutputShape()
	if len(shape) == 0 {
		t.Error("expected non-empty OutputShape after Forward")
	}
}

func TestLocalAttention_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 3,
		WithLocalMaxSeqLen[float32](8))
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := la.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := la.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
}

func TestLocalAttention_DefaultOptions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 3)
	if err != nil {
		t.Fatalf("NewLocalAttention with defaults failed: %v", err)
	}
	if la == nil {
		t.Error("expected non-nil LocalAttention")
	}
}

func TestLocalAttention_InvalidGQA(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	// Invalid: numQueryHeads not divisible by numKeyValueHeads
	_, err := NewLocalAttention[float32](engine, ops, 16, 5, 2, 3)
	if err == nil {
		t.Error("expected error for invalid GQA dimensions")
	}
}
