package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeGraniteTestTensors creates a minimal set of Granite-architecture tensors
// with canonical names for testing.
func makeGraniteTestTensors(cfg *gguf.ModelConfig, withBias bool) map[string]*tensor.TensorNumeric[float32] {
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
	zeros := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		t, _ := tensor.New(shape, data)
		return t
	}

	// Global tensors.
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		if withBias {
			tensors[prefix+"self_attn.q_proj.bias"] = zeros([]int{hidden})
			tensors[prefix+"self_attn.k_proj.bias"] = zeros([]int{kvDim})
			tensors[prefix+"self_attn.v_proj.bias"] = zeros([]int{kvDim})
		}
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func TestBuildGraniteGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGraniteTestTensors(cfg, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGraniteGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGraniteGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGraniteTestTensors(cfg, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGraniteGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGraniteGraph_TiedEmbeddings(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGraniteTestTensors(cfg, false)
	delete(tensors, "lm_head.weight")
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGraniteGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph with tied embeddings: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGraniteGraph_EmbeddingMultiplier(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "granite",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		EmbeddingMultiplier: 12.0,
	}
	tensors := makeGraniteTestTensors(cfg, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Build with embedding multiplier.
	gScaled, _, err := buildGraniteGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph (scaled): %v", err)
	}

	// Build without embedding multiplier.
	cfgNoScale := *cfg
	cfgNoScale.EmbeddingMultiplier = 0
	gUnscaled, _, err := buildGraniteGraph(tensors, &cfgNoScale, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph (unscaled): %v", err)
	}

	// Run forward passes and verify outputs differ.
	tokenIDs := []float32{1, 5, 10, 3}
	input, _ := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)

	outScaled, err := gScaled.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (scaled): %v", err)
	}
	outUnscaled, err := gUnscaled.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (unscaled): %v", err)
	}

	dataS := outScaled.Data()
	dataU := outUnscaled.Data()
	if len(dataS) != len(dataU) {
		t.Fatalf("output lengths differ: %d vs %d", len(dataS), len(dataU))
	}

	allSame := true
	for i := range dataS {
		if math.Abs(float64(dataS[i]-dataU[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("expected different outputs with embedding multiplier, but outputs are identical")
	}
}

func TestBuildGraniteGraph_AttentionBiasDetection(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}

	// With bias tensors.
	tensorsWithBias := makeGraniteTestTensors(cfg, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGraniteGraph(tensorsWithBias, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph (with bias): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}

	// Without bias tensors.
	tensorsNoBias := makeGraniteTestTensors(cfg, false)
	g2, _, err := buildGraniteGraph(tensorsNoBias, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph (no bias): %v", err)
	}
	if g2 == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildGraniteGraph_LogitSoftcap(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
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
	tensors := makeGraniteTestTensors(cfg, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGraniteGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGraniteGraph (softcap): %v", err)
	}

	// Forward pass should produce valid (non-NaN) logits capped by softcap.
	tokenIDs := []float32{1, 5, 10, 3}
	input, _ := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] = %v, want finite", i, v)
		}
		// With softcap=30, all logits should be in [-30, 30].
		if v > 30.0 || v < -30.0 {
			t.Errorf("output[%d] = %v, want in [-30, 30]", i, v)
		}
	}
}

func TestBuildGraniteGraph_MissingEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
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

	_, _, err := buildGraniteGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_Granite(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "granite",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	tensors := makeGraniteTestTensors(cfg, false)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("granite", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph granite: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestParseGraniteConfig(t *testing.T) {
	raw := map[string]interface{}{
		"model_type":           "granite",
		"vocab_size":           float64(49152),
		"hidden_size":          float64(2048),
		"num_hidden_layers":    float64(40),
		"num_attention_heads":  float64(32),
		"num_key_value_heads":  float64(8),
		"intermediate_size":    float64(8192),
		"max_position_embeddings": float64(4096),
		"rope_theta":           float64(10000),
		"attention_bias":       true,
		"embedding_multiplier": float64(12.0),
		"residual_multiplier":  float64(0.22),
		"logit_scale":          float64(13.0),
	}

	meta, err := parseGraniteConfig(raw)
	if err != nil {
		t.Fatalf("parseGraniteConfig: %v", err)
	}

	if meta.Architecture != "granite" {
		t.Errorf("architecture = %q, want %q", meta.Architecture, "granite")
	}
	if meta.VocabSize != 49152 {
		t.Errorf("vocab_size = %d, want 49152", meta.VocabSize)
	}
	if meta.HiddenSize != 2048 {
		t.Errorf("hidden_size = %d, want 2048", meta.HiddenSize)
	}
	if meta.NumLayers != 40 {
		t.Errorf("num_layers = %d, want 40", meta.NumLayers)
	}
	if meta.NumQueryHeads != 32 {
		t.Errorf("num_query_heads = %d, want 32", meta.NumQueryHeads)
	}
	if meta.NumKeyValueHeads != 8 {
		t.Errorf("num_kv_heads = %d, want 8", meta.NumKeyValueHeads)
	}
	if meta.IntermediateSize != 8192 {
		t.Errorf("intermediate_size = %d, want 8192", meta.IntermediateSize)
	}
	if !meta.AttentionBias {
		t.Error("attention_bias should be true")
	}
	if meta.EmbeddingMultiplier != 12.0 {
		t.Errorf("embedding_multiplier = %v, want 12.0", meta.EmbeddingMultiplier)
	}
	if meta.ResidualMultiplier != 0.22 {
		t.Errorf("residual_multiplier = %v, want 0.22", meta.ResidualMultiplier)
	}
	if meta.LogitScale != 13.0 {
		t.Errorf("logit_scale = %v, want 13.0", meta.LogitScale)
	}
	if meta.RopeTheta != 10000 {
		t.Errorf("rope_theta = %v, want 10000", meta.RopeTheta)
	}
}

func TestParseGraniteConfig_Defaults(t *testing.T) {
	raw := map[string]interface{}{
		"model_type":          "granite",
		"vocab_size":          float64(32000),
		"hidden_size":         float64(2048),
		"num_hidden_layers":   float64(24),
		"num_attention_heads": float64(16),
		"num_key_value_heads": float64(4),
		"intermediate_size":   float64(5632),
	}

	meta, err := parseGraniteConfig(raw)
	if err != nil {
		t.Fatalf("parseGraniteConfig: %v", err)
	}

	// Should default rope_theta to 10000.
	if meta.RopeTheta != 10000 {
		t.Errorf("rope_theta = %v, want 10000 (default)", meta.RopeTheta)
	}
	// No attention bias by default.
	if meta.AttentionBias {
		t.Error("attention_bias should be false by default")
	}
	// No scaling by default.
	if meta.EmbeddingMultiplier != 0 {
		t.Errorf("embedding_multiplier = %v, want 0 (default)", meta.EmbeddingMultiplier)
	}
}
