package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBuildInternLM2Graph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "internlm2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildInternLM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildInternLM2Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildInternLM2Graph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "internlm2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildInternLM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildInternLM2Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildInternLM2Graph_TiedEmbeddings(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "internlm2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeLlamaTestTensors(cfg)

	// Remove lm_head.weight to test weight tying fallback.
	delete(tensors, "lm_head.weight")

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildInternLM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildInternLM2Graph with tied embeddings: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildInternLM2Graph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "internlm2",
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

	_, _, err := buildInternLM2Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}
