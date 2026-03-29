package inference

import (
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeOLMo2TestTensors creates a minimal set of OLMo2-architecture tensors
// for testing. OLMo2 differs from Llama: no input_layernorm, instead uses
// post_attention_layernorm, post_feedforward_layernorm, and QK norms.
func makeOLMo2TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads
	headDim := hidden / cfg.NumHeads

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = scale * float32(math.Sin(float64(i)*0.01))
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

	// Global tensors.
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		// Attention projections.
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		// QK norms.
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
		// Post-attention norm (no input_layernorm in OLMo2).
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		// FFN.
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
		// Post-FFN norm.
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
	}

	return tensors
}

func TestBuildOLMo2Graph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "olmo2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeOLMo2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildOLMo2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildOLMo2Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildOLMo2Graph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "olmo2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeOLMo2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildOLMo2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildOLMo2Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildOLMo2Graph_TiedEmbeddings(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "olmo2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeOLMo2TestTensors(cfg)

	// Remove lm_head to test tied embeddings.
	delete(tensors, "lm_head.weight")

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildOLMo2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildOLMo2Graph with tied embeddings: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildOLMo2Graph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "olmo2",
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

	_, _, err := buildOLMo2Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildOLMo2Graph_NoInputLayernorm(t *testing.T) {
	// Verify OLMo2 does NOT require input_layernorm tensors.
	cfg := &gguf.ModelConfig{
		Architecture:     "olmo2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeOLMo2TestTensors(cfg)

	// Confirm no input_layernorm tensors exist (OLMo2 doesn't use them).
	for name := range tensors {
		if name == "model.layers.0.input_layernorm.weight" {
			t.Fatal("OLMo2 test tensors should not include input_layernorm")
		}
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildOLMo2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildOLMo2Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}
