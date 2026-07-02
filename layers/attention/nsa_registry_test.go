package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestBuildNativeSparseAttention(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	numHeads := 2
	numKVHeads := 2
	modelDim := 8
	headDim := modelDim / numHeads

	attrs := map[string]interface{}{
		"model_dim":   modelDim,
		"num_heads":   numHeads,
		"num_kv_heads": numKVHeads,
		"block_size":  4,
		"top_blocks":  1,
		"top_tokens":  2,
		"window_size": 4,
	}

	node, err := BuildNativeSparseAttention(engine, ops, "nsa", nil, attrs)
	if err != nil {
		t.Fatalf("BuildNativeSparseAttention failed: %v", err)
	}

	if node.OpType() != "NativeSparseAttention" {
		t.Errorf("OpType() = %q, want %q", node.OpType(), "NativeSparseAttention")
	}

	// Run Forward with Q, K, V to verify the built layer works.
	batch := 1
	seqLen := 8
	Q := makeNonZeroInput(t, []int{batch, numHeads, seqLen, headDim})
	K := makeNonZeroInput(t, []int{batch, numKVHeads, seqLen, headDim})
	V := makeNonZeroInput(t, []int{batch, numKVHeads, seqLen, headDim})

	out, err := node.Forward(context.Background(), Q, K, V)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	expected := []int{batch, numHeads, seqLen, headDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Errorf("output shape = %v, want %v", out.Shape(), expected)
	}
}

func TestBuildNativeSparseAttention_WithGateParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	numHeads := 2
	attrs := map[string]interface{}{
		"model_dim":   8,
		"num_heads":   numHeads,
		"num_kv_heads": 2,
		"block_size":  4,
		"top_blocks":  1,
		"top_tokens":  2,
		"window_size": 4,
	}

	params := map[string]*graph.Parameter[float32]{
		"nsa_gate_coarse": makeNonZeroParam(t, "nsa_gate_coarse", []int{numHeads}),
		"nsa_gate_fine":   makeNonZeroParam(t, "nsa_gate_fine", []int{numHeads}),
		"nsa_gate_window": makeNonZeroParam(t, "nsa_gate_window", []int{numHeads}),
	}

	node, err := BuildNativeSparseAttention(engine, ops, "nsa", params, attrs)
	if err != nil {
		t.Fatalf("BuildNativeSparseAttention with gate params failed: %v", err)
	}

	nsa := node.(*NativeSparseAttention[float32])

	// Verify gate parameters were overridden from GGUF params.
	if nsa.gateCoarse != params["nsa_gate_coarse"] {
		t.Error("gateCoarse was not overridden from params")
	}
	if nsa.gateFine != params["nsa_gate_fine"] {
		t.Error("gateFine was not overridden from params")
	}
	if nsa.gateWindow != params["nsa_gate_window"] {
		t.Error("gateWindow was not overridden from params")
	}
}

func TestBuildNativeSparseAttention_MissingAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	fullAttrs := map[string]interface{}{
		"model_dim":   8,
		"num_heads":   2,
		"num_kv_heads": 2,
		"block_size":  4,
		"top_blocks":  1,
		"top_tokens":  2,
		"window_size": 4,
	}

	requiredAttrs := []string{"model_dim", "num_heads", "num_kv_heads", "block_size", "top_blocks", "top_tokens", "window_size"}
	for _, key := range requiredAttrs {
		t.Run("missing_"+key, func(t *testing.T) {
			attrs := make(map[string]interface{})
			for k, v := range fullAttrs {
				if k != key {
					attrs[k] = v
				}
			}
			_, err := BuildNativeSparseAttention(engine, ops, "nsa", nil, attrs)
			if err == nil {
				t.Errorf("expected error for missing attribute %q", key)
			}
		})
	}
}

func TestBuildNativeSparseAttention_InvalidHeads(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]interface{}{
		"model_dim":   8,
		"num_heads":   3,
		"num_kv_heads": 2, // 3 % 2 != 0
		"block_size":  4,
		"top_blocks":  1,
		"top_tokens":  2,
		"window_size": 4,
	}

	_, err := BuildNativeSparseAttention(engine, ops, "nsa", nil, attrs)
	if err == nil {
		t.Error("expected error for num_heads not divisible by num_kv_heads")
	}
}
