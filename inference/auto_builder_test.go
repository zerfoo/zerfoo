package inference

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeTestTensors creates a minimal set of decoder-only transformer tensors
// for testing. If tieEmbed is true, no separate lm_head.weight is created.
// If extraNorms is true, Gemma 3-style post-attention and post-FFN norms are added.
// If attnBias is true, Q/K/V bias tensors are added.
func makeTestTensors(cfg *gguf.ModelConfig, tieEmbed, extraNorms, attnBias bool) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

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

	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	if !tieEmbed {
		tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)

		if extraNorms {
			tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{hidden / cfg.NumHeads})
			tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{hidden / cfg.NumHeads})
			tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
			tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		}

		if attnBias {
			tensors[prefix+"self_attn.q_proj.bias"] = fill([]int{hidden}, 0.01)
			tensors[prefix+"self_attn.k_proj.bias"] = fill([]int{kvDim}, 0.01)
			tensors[prefix+"self_attn.v_proj.bias"] = fill([]int{kvDim}, 0.01)
		}
	}

	return tensors
}

func TestAutoBuilder_LlamaMetadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        500000,
	}
	tensors := makeTestTensors(cfg, false, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild llama: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}

	// Verify forward pass produces valid output.
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_LlamaTiedEmbeddings(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        500000,
	}
	// No separate lm_head.weight -- should tie to embeddings.
	tensors := makeTestTensors(cfg, true, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild llama tied: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_GemmaMetadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
	}
	// Gemma always ties embeddings.
	tensors := makeTestTensors(cfg, true, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild gemma: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_Gemma3Metadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma3",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
		LogitSoftcap:     30.0,
	}
	// Gemma 3 ties embeddings and has extra norms.
	tensors := makeTestTensors(cfg, true, true, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild gemma3: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_QwenMetadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000,
	}
	// Qwen uses attention bias and ties embeddings when no lm_head.
	tensors := makeTestTensors(cfg, true, false, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild qwen2: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_MistralMetadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "mistral",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
		SlidingWindow:    4096,
	}
	tensors := makeTestTensors(cfg, true, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild mistral: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_PhiMetadata(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeTestTensors(cfg, true, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild phi3: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_UnknownArchFallback(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "futurearch_7b",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
	}
	tensors := makeTestTensors(cfg, false, false, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Unknown arch should still build a standard transformer graph via fallback.
	g, emb, err := AutoBuild(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("AutoBuild unknown arch: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestAutoBuilder_MissingEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
	}
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := AutoBuild(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing embedding tensor")
	}
}

func TestAutoBuilder_FeatureDetection(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want autoFeatures
	}{
		{
			name: "llama standard",
			cfg: &gguf.ModelConfig{
				Architecture: "llama",
				HiddenSize:   16,
				NumHeads:     4,
			},
			want: autoFeatures{},
		},
		{
			name: "gemma embedding scale",
			cfg: &gguf.ModelConfig{
				Architecture: "gemma",
				HiddenSize:   16,
				NumHeads:     4,
			},
			want: autoFeatures{
				embedScale: float32(math.Sqrt(16)),
			},
		},
		{
			name: "gemma3 all features",
			cfg: &gguf.ModelConfig{
				Architecture: "gemma3",
				HiddenSize:   16,
				NumHeads:     4,
				LogitSoftcap: 30.0,
			},
			want: autoFeatures{
				embedScale:   float32(math.Sqrt(16)),
				postNorm:     true,
				qkNorm:       true,
				logitSoftcap: 30.0,
			},
		},
		{
			name: "qwen2 attention bias",
			cfg: &gguf.ModelConfig{
				Architecture: "qwen2",
				HiddenSize:   16,
				NumHeads:     4,
			},
			want: autoFeatures{
				attnBias: true,
			},
		},
		{
			name: "mistral sliding window",
			cfg: &gguf.ModelConfig{
				Architecture:  "mistral",
				HiddenSize:    16,
				NumHeads:      4,
				SlidingWindow: 4096,
			},
			want: autoFeatures{
				slidingWindowSize: 4096,
			},
		},
		{
			name: "phi partial rotary",
			cfg: &gguf.ModelConfig{
				Architecture:        "phi3",
				HiddenSize:          16,
				NumHeads:            4,
				PartialRotaryFactor: 0.5,
			},
			want: autoFeatures{
				partialRotaryFactor: 0.5,
			},
		},
		{
			name: "unknown arch fallback",
			cfg: &gguf.ModelConfig{
				Architecture: "newarch_42",
				HiddenSize:   16,
				NumHeads:     4,
			},
			want: autoFeatures{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detectFeatures(tt.cfg)
			if got != tt.want {
				t.Errorf("detectFeatures(%q) =\n  %+v\nwant\n  %+v", tt.cfg.Architecture, got, tt.want)
			}
		})
	}
}
