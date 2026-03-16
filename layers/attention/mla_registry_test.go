package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestBuildMultiHeadLatentAttention(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	numHeads := 2
	headDim := 4
	kvLoraDim := 3
	hiddenDim := 8
	qkDim := numHeads * headDim

	attrs := map[string]any{
		"num_heads":   numHeads,
		"head_dim":    headDim,
		"kv_lora_dim": kvLoraDim,
		"max_seq_len": 16,
	}

	params := map[string]*graph.Parameter[float32]{
		"mla_wq":   makeNonZeroParam(t, "mla_wq", []int{hiddenDim, qkDim}),
		"mla_wdkv": makeNonZeroParam(t, "mla_wdkv", []int{hiddenDim, kvLoraDim}),
		"mla_wuk":  makeNonZeroParam(t, "mla_wuk", []int{kvLoraDim, qkDim}),
		"mla_wuv":  makeNonZeroParam(t, "mla_wuv", []int{kvLoraDim, qkDim}),
		"mla_wo":   makeNonZeroParam(t, "mla_wo", []int{qkDim, hiddenDim}),
	}

	node, err := BuildMultiHeadLatentAttention(engine, ops, "mla", params, attrs)
	if err != nil {
		t.Fatalf("BuildMultiHeadLatentAttention failed: %v", err)
	}

	if node.OpType() != "MultiHeadLatentAttention" {
		t.Errorf("OpType() = %q, want %q", node.OpType(), "MultiHeadLatentAttention")
	}

	// Run Forward to verify the built layer works.
	input := makeNonZeroInput(t, []int{1, 3, hiddenDim})
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	expected := []int{1, 3, hiddenDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Errorf("output shape = %v, want %v", out.Shape(), expected)
	}
}

func TestBuildMultiHeadLatentAttention_MissingAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	fullAttrs := map[string]any{
		"num_heads":   2,
		"head_dim":    4,
		"kv_lora_dim": 3,
		"max_seq_len": 16,
	}

	params := map[string]*graph.Parameter[float32]{
		"mla_wq":   makeParam(t, "mla_wq", []int{8, 8}),
		"mla_wdkv": makeParam(t, "mla_wdkv", []int{8, 3}),
		"mla_wuk":  makeParam(t, "mla_wuk", []int{3, 8}),
		"mla_wuv":  makeParam(t, "mla_wuv", []int{3, 8}),
		"mla_wo":   makeParam(t, "mla_wo", []int{8, 8}),
	}

	requiredAttrs := []string{"num_heads", "head_dim", "kv_lora_dim", "max_seq_len"}
	for _, key := range requiredAttrs {
		t.Run("missing_"+key, func(t *testing.T) {
			attrs := make(map[string]any)
			for k, v := range fullAttrs {
				if k != key {
					attrs[k] = v
				}
			}
			_, err := BuildMultiHeadLatentAttention(engine, ops, "mla", params, attrs)
			if err == nil {
				t.Errorf("expected error for missing attribute %q", key)
			}
		})
	}
}

func TestBuildMultiHeadLatentAttention_MissingParams(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	attrs := map[string]any{
		"num_heads":   2,
		"head_dim":    4,
		"kv_lora_dim": 3,
		"max_seq_len": 16,
	}

	fullParams := map[string]*graph.Parameter[float32]{
		"mla_wq":   makeParam(t, "mla_wq", []int{8, 8}),
		"mla_wdkv": makeParam(t, "mla_wdkv", []int{8, 3}),
		"mla_wuk":  makeParam(t, "mla_wuk", []int{3, 8}),
		"mla_wuv":  makeParam(t, "mla_wuv", []int{3, 8}),
		"mla_wo":   makeParam(t, "mla_wo", []int{8, 8}),
	}

	requiredParams := []string{"mla_wq", "mla_wdkv", "mla_wuk", "mla_wuv", "mla_wo"}
	for _, key := range requiredParams {
		t.Run("missing_"+key, func(t *testing.T) {
			params := make(map[string]*graph.Parameter[float32])
			for k, v := range fullParams {
				if k != key {
					params[k] = v
				}
			}
			_, err := BuildMultiHeadLatentAttention(engine, ops, "mla", params, attrs)
			if err == nil {
				t.Errorf("expected error for missing parameter %q", key)
			}
		})
	}
}

func TestBuildMultiHeadLatentAttention_CustomRopeBase(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	attrs := map[string]any{
		"num_heads":   2,
		"head_dim":    4,
		"kv_lora_dim": 3,
		"max_seq_len": 16,
		"rope_base":   50000.0,
	}

	params := map[string]*graph.Parameter[float32]{
		"mla_wq":   makeNonZeroParam(t, "mla_wq", []int{8, 8}),
		"mla_wdkv": makeNonZeroParam(t, "mla_wdkv", []int{8, 3}),
		"mla_wuk":  makeNonZeroParam(t, "mla_wuk", []int{3, 8}),
		"mla_wuv":  makeNonZeroParam(t, "mla_wuv", []int{3, 8}),
		"mla_wo":   makeNonZeroParam(t, "mla_wo", []int{8, 8}),
	}

	node, err := BuildMultiHeadLatentAttention(engine, ops, "mla", params, attrs)
	if err != nil {
		t.Fatalf("BuildMultiHeadLatentAttention with custom rope_base failed: %v", err)
	}

	input := makeNonZeroInput(t, []int{1, 3, 8})
	_, err = node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
}

func makeNonZeroInput(t *testing.T, shape []int) *tensor.TensorNumeric[float32] {
	t.Helper()
	total := 1
	for _, d := range shape {
		total *= d
	}
	data := make([]float32, total)
	for i := range data {
		data[i] = float32(i%7+1) * 0.01
	}
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}
	return out
}
