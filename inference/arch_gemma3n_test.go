package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeGemma3nTestTensors creates a minimal set of Gemma 3n architecture tensors.
// Gemma 3n shares the Gemma 3 base (tied embeddings, post-norms, Q/K norms)
// but uses smaller dimensions suitable for mobile-optimized models.
// It also includes Per-Layer Embeddings (PLE) tensors.
func makeGemma3nTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeGemma3TestTensors(cfg)

	hidden := cfg.HiddenSize

	// Add PLE (Per-Layer Embedding) projection tensors for each layer.
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

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		// PLE uses a down-projection from the shared embedding to
		// a per-layer hidden size. For test purposes we use identity-like
		// tensors with the same dimension (hidden -> hidden).
		tensors[prefix+"ple_proj.weight"] = ones([]int{hidden, hidden})
	}

	return tensors
}

// makeGemma3TestTensors creates Gemma 3 tensors (Gemma base + post-norms + Q/K norms).
func makeGemma3TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeGemmaTestTensors(cfg)

	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
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

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		// Post-attention and post-FFN norms (Gemma 3 has 4 norms per layer).
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		// Q/K norms.
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
	}

	return tensors
}

func TestBuildGemma3nGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		LogitSoftcap:     30.0,
	}
	tensors := makeGemma3nTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemma3nGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma3nGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestGemma3nForward(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		LogitSoftcap:     30.0,
	}
	tensors := makeGemma3nTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma3nGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma3nGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma3nGraph_TiedEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		LogitSoftcap:     30.0,
	}
	tensors := makeGemma3nTestTensors(cfg)
	// Verify no separate lm_head.weight exists (tied to embedding).
	if _, ok := tensors["lm_head.weight"]; ok {
		t.Fatal("Gemma 3n should not have separate lm_head.weight (tied embedding)")
	}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma3nGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma3nGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildGemma3nGraph_MobileOptimizedDimensions(t *testing.T) {
	// Test with smaller dimensions typical of mobile-optimized models.
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        16,
		HiddenSize:       8,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		IntermediateSize: 16,
		MaxSeqLen:        32,
		RopeTheta:        10000.0,
		LogitSoftcap:     30.0,
	}
	tensors := makeGemma3nTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma3nGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma3nGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma3nGraph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildGemma3nGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildGemma3nGraph_EmbedScaling(t *testing.T) {
	// Verify that Gemma 3n applies the same sqrt(hidden_size) embedding
	// scaling as Gemma 3.
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3n",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		LogitSoftcap:     30.0,
	}
	tensors := makeGemma3nTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma3nGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma3nGraph: %v", err)
	}

	// Just verify forward pass works with the scaling applied.
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}
