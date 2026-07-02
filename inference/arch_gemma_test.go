package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func makeGemmaTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeLlamaTestTensors(cfg)
	// Gemma ties lm_head to embedding weights -- remove separate lm_head.weight.
	delete(tensors, "lm_head.weight")
	return tensors
}

func TestBuildGemmaGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemmaGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemmaGraph_TiedEmbedding(t *testing.T) {
	// Gemma should work without lm_head.weight (tied to embedding).
	cfg := &gguf.ModelConfig{
		Architecture:     "gemma",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGemmaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemmaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemmaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}
