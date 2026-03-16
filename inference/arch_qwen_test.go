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

// makeQwenTestTensors creates a minimal set of Qwen 2-architecture tensors
// with canonical names for testing. Includes attention bias tensors.
func makeQwenTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
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
		// Qwen attention biases.
		tensors[prefix+"self_attn.q_proj.bias"] = zeros([]int{hidden})
		tensors[prefix+"self_attn.k_proj.bias"] = zeros([]int{kvDim})
		tensors[prefix+"self_attn.v_proj.bias"] = zeros([]int{kvDim})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func TestBuildQwenGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
	tensors := makeQwenTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildQwenGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwenGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildQwenGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
	tensors := makeQwenTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildQwenGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwenGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildQwenGraph_TiedEmbeddings(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
	tensors := makeQwenTestTensors(cfg)
	// Remove lm_head.weight to test tied embeddings.
	delete(tensors, "lm_head.weight")
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildQwenGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwenGraph with tied embeddings: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildQwenGraph_NonZeroBias(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}

	// Build two tensor sets: one with zero bias, one with non-zero bias.
	tensorsZero := makeQwenTestTensors(cfg)
	tensorsNonZero := makeQwenTestTensors(cfg)

	// Set non-zero biases for layer 0.
	prefix := "model.layers.0."
	qBiasData := make([]float32, cfg.HiddenSize)
	for i := range qBiasData {
		qBiasData[i] = 0.1
	}
	qBias, _ := tensor.New([]int{cfg.HiddenSize}, qBiasData)
	tensorsNonZero[prefix+"self_attn.q_proj.bias"] = qBias

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	gZero, _, err := buildQwenGraph(tensorsZero, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwenGraph (zero bias): %v", err)
	}

	gNonZero, _, err := buildQwenGraph(tensorsNonZero, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwenGraph (non-zero bias): %v", err)
	}

	// Run forward passes and verify outputs differ.
	tokenIDs := []float32{1, 5, 10, 3}
	input, _ := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)

	outZero, err := gZero.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (zero bias): %v", err)
	}
	outNonZero, err := gNonZero.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (non-zero bias): %v", err)
	}

	// Outputs should differ when bias is non-zero.
	dataZ := outZero.Data()
	dataNZ := outNonZero.Data()
	if len(dataZ) != len(dataNZ) {
		t.Fatalf("output lengths differ: %d vs %d", len(dataZ), len(dataNZ))
	}

	allSame := true
	for i := range dataZ {
		if math.Abs(float64(dataZ[i]-dataNZ[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("expected different outputs with non-zero bias, but outputs are identical")
	}
}

func TestBuildQwenGraph_MissingEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildQwenGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_Qwen2(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
	tensors := makeQwenTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("qwen2", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph qwen2: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}
