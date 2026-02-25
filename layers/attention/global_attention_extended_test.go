package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestGlobalAttention_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}
	if got := ga.OpType(); got != "GlobalAttention" {
		t.Errorf("OpType() = %q, want %q", got, "GlobalAttention")
	}
}

func TestGlobalAttention_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}
	attrs := ga.Attributes()
	if attrs["embed_dim"] != 16 {
		t.Errorf("embed_dim = %v, want 16", attrs["embed_dim"])
	}
	if attrs["num_heads"] != 4 {
		t.Errorf("num_heads = %v, want 4", attrs["num_heads"])
	}
	if attrs["num_kv_heads"] != 2 {
		t.Errorf("num_kv_heads = %v, want 2", attrs["num_kv_heads"])
	}
}

func TestGlobalAttention_NewFromParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	gqa, err := NewGroupedQueryAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention failed: %v", err)
	}
	ga := NewGlobalAttentionFromParams(gqa)
	if ga == nil {
		t.Fatal("NewGlobalAttentionFromParams returned nil")
	}
	if ga.gqa != gqa {
		t.Error("inner GQA does not match")
	}
}

func TestGlobalAttention_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}
	params := ga.Parameters()
	// GQA has wq, wk, wv, wo; each Dense has weight + bias = 8 params
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}

func TestGlobalAttention_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}

	// OutputShape is nil before Forward; run Forward first
	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = 0.01
	}
	_, err = ga.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	shape := ga.OutputShape()
	if len(shape) == 0 {
		t.Error("expected non-empty OutputShape after Forward")
	}
}

func TestGlobalAttention_ScaleRope(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}

	if err := ga.ScaleRope(ctx, 2.0); err != nil {
		t.Errorf("ScaleRope failed: %v", err)
	}
}

func TestGlobalAttention_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2,
		WithGlobalAttentionMaxSeqLen(8))
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := ga.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 1.0
	}

	grads, err := ga.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
}

func TestBuildGlobalAttention(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name    string
		attrs   map[string]interface{}
		wantErr bool
	}{
		{
			name: "Valid",
			attrs: map[string]interface{}{
				"embed_dim":    16,
				"num_heads":    4,
				"num_kv_heads": 2,
			},
			wantErr: false,
		},
		{
			name: "MissingEmbedDim",
			attrs: map[string]interface{}{
				"num_heads":    4,
				"num_kv_heads": 2,
			},
			wantErr: true,
		},
		{
			name: "MissingNumHeads",
			attrs: map[string]interface{}{
				"embed_dim":    16,
				"num_kv_heads": 2,
			},
			wantErr: true,
		},
		{
			name: "MissingNumKVHeads",
			attrs: map[string]interface{}{
				"embed_dim": 16,
				"num_heads": 4,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildGlobalAttention[float32](engine, ops, "", nil, tt.attrs)
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildGlobalAttention() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGlobalAttention_InvalidDimensions(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// numQueryHeads not divisible by numKeyValueHeads
	_, err := NewGlobalAttention[float32](engine, ops, 16, 5, 2)
	if err == nil {
		t.Error("expected error for non-divisible head counts")
	}
}
