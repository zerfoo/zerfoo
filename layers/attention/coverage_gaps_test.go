package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------------------------------------------------------------------------
// MLA Backward (multi_head_latent_attention.go:196)
// ---------------------------------------------------------------------------

func TestMultiHeadLatentAttention_Backward(t *testing.T) {
	mla := newTestMLA(t)
	ctx := context.Background()

	input, _ := tensor.New[float32]([]int{1, 3, 8}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := mla.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOut, _ := tensor.New[float32](out.Shape(), nil)
	for i := range dOut.Data() {
		dOut.Data()[i] = 0.1
	}

	grads, err := mla.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Errorf("expected 1 gradient, got %d", len(grads))
	}
}

// ---------------------------------------------------------------------------
// buildGlobalAttention (registry.go:118)
// ---------------------------------------------------------------------------

func TestBuildGlobalAttention_ViaRegistry(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	modelDim := 16
	numQ := 4
	numKV := 2
	headDim := modelDim / numQ
	kvDim := headDim * numKV

	params := map[string]*graph.Parameter[float32]{
		"attn_wq": makeNonZeroParam(t, "attn_wq", []int{modelDim, modelDim}),
		"attn_wk": makeNonZeroParam(t, "attn_wk", []int{modelDim, kvDim}),
		"attn_wv": makeNonZeroParam(t, "attn_wv", []int{modelDim, kvDim}),
		"attn_wo": makeNonZeroParam(t, "attn_wo", []int{modelDim, modelDim}),
	}

	attrs := map[string]interface{}{
		"model_dim":           modelDim,
		"num_query_heads":     numQ,
		"num_key_value_heads": numKV,
		"rope_base":           10000.0,
		"max_seq_len":         64,
	}

	node, err := buildGlobalAttention[float32](engine, ops, "attn", params, attrs)
	if err != nil {
		t.Fatalf("buildGlobalAttention failed: %v", err)
	}

	if node.OpType() != "GlobalAttention" {
		t.Errorf("OpType() = %q, want %q", node.OpType(), "GlobalAttention")
	}

	// Verify it works with Forward
	input := makeNonZeroInput(t, []int{1, 3, modelDim})
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out.Shape()[2] != modelDim {
		t.Errorf("output dim = %d, want %d", out.Shape()[2], modelDim)
	}
}

func TestBuildGlobalAttention_PropagatesError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// Missing required attributes should propagate error from buildGroupedQueryAttention
	_, err := buildGlobalAttention[float32](engine, ops, "attn", nil, map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing attributes")
	}
}

// ---------------------------------------------------------------------------
// SetLayerIndex (global_attention.go:117)
// ---------------------------------------------------------------------------

func TestGlobalAttention_SetLayerIndex(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	ga, err := NewGlobalAttention[float32](engine, ops, 16, 4, 2)
	if err != nil {
		t.Fatalf("NewGlobalAttention failed: %v", err)
	}

	ga.SetLayerIndex(42)
	if ga.gqa.LayerIndex != 42 {
		t.Errorf("LayerIndex = %d, want 42", ga.gqa.LayerIndex)
	}
}

// ---------------------------------------------------------------------------
// NewGroupedQueryAttention error branches
// (grouped_query_attention.go:114,122,127,132,141)
// ---------------------------------------------------------------------------

func TestNewGroupedQueryAttention_ModelDimNotDivisibleByKV(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// modelDim=17, numQ=1, numKV=1 -> headDim=17 (odd), RoPE fails
	_, err := NewGroupedQueryAttention[float32](engine, ops, 17, 1, 1)
	if err == nil {
		t.Error("expected error for odd headDim (RoPE requires even)")
	}
}

// ---------------------------------------------------------------------------
// LocalAttention Forward (local_attention.go:85)
// ---------------------------------------------------------------------------

func TestLocalAttention_Forward_MaskCreation(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	la, err := NewLocalAttention[float32](engine, ops, 16, 4, 2, 1,
		WithLocalMaxSeqLen[float32](16))
	if err != nil {
		t.Fatalf("NewLocalAttention failed: %v", err)
	}

	// Test with window size larger than sequence
	input, _ := tensor.New[float32]([]int{1, 2, 16}, nil)
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := la.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out.Shape()[1] != 2 {
		t.Errorf("seqLen = %d, want 2", out.Shape()[1])
	}
}

// ---------------------------------------------------------------------------
// GQA Forward with cache + engine errors (grouped_query_attention.go:335-362)
// ---------------------------------------------------------------------------

func TestGQA_Forward_CacheEngineErrors(t *testing.T) {
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Reshape failures inside cache block.
		// Pre-cache Reshape calls: 7 (qReshaped, kReshaped, vReshaped,
		// qForRoPE, kForRoPE, qHeadsRoPE back, kHeadsRoPE back).
		{"Cache_Reshape_kFlat", map[string]int{"Reshape": 8}},
		{"Cache_Reshape_vFlat", map[string]int{"Reshape": 9}},
		{"Cache_Reshape_unflatK", map[string]int{"Reshape": 10}},
		{"Cache_Reshape_unflatV", map[string]int{"Reshape": 11}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine(ops)

			gqa, err := NewGroupedQueryAttention[float32](engine, ops, 8, 2, 2,
				WithMaxSeqLen[float32](8))
			if err != nil {
				t.Fatalf("NewGQA failed: %v", err)
			}
			gqa.LayerIndex = 0

			input, _ := tensor.New[float32]([]int{1, 2, 8}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%7+1) * 0.01
			}

			// Set up KV cache context
			cache := generate.NewKVCache[float32](1, 128)
			ctx := generate.WithKVCache(context.Background(), cache)

			// Swap engine to failing one for the forward call
			fe := newFailingEngine(tc.failOn)
			gqa.engine = fe

			_, err = gqa.Forward(ctx, input)
			if err == nil {
				t.Errorf("expected error from %s", tc.name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// SDPA Forward headDim <= 0 fallback paths
// (scaled_dot_product_attention.go:81,86)
// ---------------------------------------------------------------------------

func TestSDPA_Forward_InvalidHeadDim(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	tests := []struct {
		name    string
		headDim int
	}{
		{"zero", 0},
		{"negative", -1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sdpa := NewScaledDotProductAttention[float32](engine, tc.headDim)

			q, _ := tensor.New[float32]([]int{1, 3, 4}, nil)
			k, _ := tensor.New[float32]([]int{1, 3, 4}, nil)
			v, _ := tensor.New[float32]([]int{1, 3, 4}, nil)
			for i := range q.Data() {
				q.Data()[i] = float32(i%5+1) * 0.01
				k.Data()[i] = float32(i%5+1) * 0.01
				v.Data()[i] = float32(i%5+1) * 0.01
			}

			out, err := sdpa.Forward(context.Background(), q, k, v, nil)
			if err != nil {
				t.Fatalf("Forward with headDim=%d fallback failed: %v", tc.headDim, err)
			}
			if out == nil {
				t.Error("expected non-nil output")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// MLA BuildMultiHeadLatentAttention RoPE failure (mla_registry.go:86-88)
// ---------------------------------------------------------------------------

func TestBuildMultiHeadLatentAttention_OddHeadDim(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// headDim=3 is odd, which should cause RoPE creation to fail
	attrs := map[string]any{
		"num_heads":   2,
		"head_dim":    3,
		"kv_lora_dim": 3,
		"max_seq_len": 16,
	}

	params := map[string]*graph.Parameter[float32]{
		"mla_wq":   makeParam(t, "mla_wq", []int{6, 6}),
		"mla_wdkv": makeParam(t, "mla_wdkv", []int{6, 3}),
		"mla_wuk":  makeParam(t, "mla_wuk", []int{3, 6}),
		"mla_wuv":  makeParam(t, "mla_wuv", []int{3, 6}),
		"mla_wo":   makeParam(t, "mla_wo", []int{6, 6}),
	}

	_, err := BuildMultiHeadLatentAttention(engine, ops, "mla", params, attrs)
	if err == nil {
		t.Error("expected error for odd headDim (RoPE requires even)")
	}
}
