package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
)

func TestBuildPhiGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           2,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildPhiGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           2,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildPhiGraph_FullRotaryFactor(t *testing.T) {
	// PartialRotaryFactor = 1.0 should behave like full RoPE (same as Llama).
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 1.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph with factor=1.0: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildPhiGraph_TiedEmbedding(t *testing.T) {
	// Phi should work without lm_head.weight (tied to embedding).
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	delete(tensors, "lm_head.weight")
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph with tied embedding: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildPhiGraph_PartialRotaryFactors(t *testing.T) {
	tests := []struct {
		name       string
		factor     float32
		hiddenSize int
		numHeads   int
	}{
		{"half", 0.5, 16, 4},
		{"three_quarters", 0.75, 32, 4},
		{"full", 1.0, 16, 4},
		{"zero_defaults_to_full", 0.0, 16, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:        "phi3",
				VocabSize:           32,
				HiddenSize:          tt.hiddenSize,
				NumLayers:           1,
				NumHeads:            tt.numHeads,
				NumKVHeads:          2,
				IntermediateSize:    32,
				MaxSeqLen:           64,
				RopeTheta:           10000.0,
				PartialRotaryFactor: tt.factor,
			}
			tensors := makeLlamaTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildPhiGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildPhiGraph with factor=%v: %v", tt.factor, err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

func TestBuildArchGraph_Phi(t *testing.T) {
	tests := []struct {
		arch string
	}{
		{"phi3"},
		{"phi"},
	}

	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:        tt.arch,
				VocabSize:           32,
				HiddenSize:          16,
				NumLayers:           1,
				NumHeads:            4,
				NumKVHeads:          2,
				IntermediateSize:    32,
				MaxSeqLen:           64,
				RopeTheta:           10000.0,
				PartialRotaryFactor: 0.5,
			}
			tensors := makeLlamaTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildArchGraph(tt.arch, tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildArchGraph(%q): %v", tt.arch, err)
			}
			if g == nil {
				t.Fatal("graph is nil")
			}
			if emb == nil {
				t.Fatal("embedding is nil")
			}
		})
	}
}
