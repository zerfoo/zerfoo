package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeGemma4_31BTestTensors creates a minimal set of Gemma 4 architecture tensors.
// Gemma 4 builds on the Gemma 3 base (tied embeddings, 4 norms per layer, Q/K norms)
// but uses per-layer varying KV head counts and head dimensions via the per-layer
// loop builder.
func makeGemma4_31BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	// Start from Gemma 3 tensors (which include post-norms, pre-FFN norms,
	// post-FFN norms, and Q/K norms).
	tensors := makeGemma3TestTensors(cfg)
	return tensors
}

func TestBuildGemma4Graph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemma4Graph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4Graph_TiedEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            1,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	// Verify no separate lm_head.weight exists (tied to embedding).
	if _, ok := tensors["lm_head.weight"]; ok {
		t.Fatal("Gemma 4 should not have separate lm_head.weight (tied embedding)")
	}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildGemma4Graph_HybridAttention(t *testing.T) {
	// 6 layers with SlidingWindowPattern=6: layers 0-4 are sliding, layer 5 is global.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            6,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}

	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads

	fill := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.02 * float32(i%7+1) * 0.1
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	ones := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		t, _ := tensor.New(shape, data)
		return t
	}

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	tensors["model.embed_tokens.weight"] = fill([]int{cfg.VocabSize, hidden})
	tensors["model.norm.weight"] = ones([]int{hidden})

	kvDim := headDim * cfg.NumKVHeads
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{cfg.IntermediateSize, hidden})
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{cfg.IntermediateSize, hidden})
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, cfg.IntermediateSize})
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph hybrid: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4Graph_KEqV(t *testing.T) {
	// Verify K=V on global layers by setting AttentionKEqV=true.
	// With SlidingWindowPattern=2 and NumLayers=2, layer 1 is global.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 2,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph K=V: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}
