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

// makeBertTestTensors creates a minimal set of BERT-architecture tensors for
// testing. All weights use small deterministic values to produce non-NaN output.
func makeBertTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	maxPos := cfg.MaxSeqLen

	numLabels := cfg.NumLabels
	if numLabels <= 0 {
		numLabels = 2
	}

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

	// Global embedding tensors.
	tensors["token_embd.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["position_embd.weight"] = fill([]int{maxPos, hidden}, 0.02)
	tensors["token_type_embd.weight"] = fill([]int{2, hidden}, 0.02)

	// Embedding LayerNorm.
	tensors["token_embd_norm.weight"] = ones([]int{hidden})
	tensors["token_embd_norm.bias"] = zeros([]int{hidden})

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "blk." + itoa(i) + "."

		// Attention Q/K/V/O weights and biases.
		tensors[prefix+"attn_q.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"attn_q.bias"] = zeros([]int{hidden})
		tensors[prefix+"attn_k.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"attn_k.bias"] = zeros([]int{hidden})
		tensors[prefix+"attn_v.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"attn_v.bias"] = zeros([]int{hidden})
		tensors[prefix+"attn_output.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"attn_output.bias"] = zeros([]int{hidden})

		// Attention LayerNorm.
		tensors[prefix+"attn_norm.weight"] = ones([]int{hidden})
		tensors[prefix+"attn_norm.bias"] = zeros([]int{hidden})

		// FFN weights and biases.
		tensors[prefix+"ffn_up.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"ffn_up.bias"] = zeros([]int{inter})
		tensors[prefix+"ffn_down.weight"] = fill([]int{hidden, inter}, 0.02)
		tensors[prefix+"ffn_down.bias"] = zeros([]int{hidden})

		// FFN LayerNorm.
		tensors[prefix+"ffn_norm.weight"] = ones([]int{hidden})
		tensors[prefix+"ffn_norm.bias"] = zeros([]int{hidden})
	}

	// Pooler (CLS token projection + tanh).
	tensors["cls_pooler.weight"] = fill([]int{hidden, hidden}, 0.02)
	tensors["cls_pooler.bias"] = zeros([]int{hidden})

	// Classification head.
	tensors["cls.weight"] = fill([]int{numLabels, hidden}, 0.02)
	tensors["cls.bias"] = zeros([]int{numLabels})

	return tensors
}

func bertBaseConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "bert",
		VocabSize:        32,
		HiddenSize:       48, // divisible by 12 heads
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       4,
		IntermediateSize: 96,
		MaxSeqLen:        64,
		NumLabels:        3,
		LayerNormEps:     1e-12,
	}
}

func TestBuildBertGraph_Builds(t *testing.T) {
	cfg := bertBaseConfig()
	tensors := makeBertTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildBertGraph_BertBase(t *testing.T) {
	// BERT-base: 12 layers, 768 hidden, 12 heads.
	cfg := &gguf.ModelConfig{
		Architecture:     "bert",
		VocabSize:        32,
		HiddenSize:       768,
		NumLayers:        12,
		NumHeads:         12,
		NumKVHeads:       12,
		IntermediateSize: 3072,
		MaxSeqLen:        512,
		NumLabels:        3,
		LayerNormEps:     1e-12,
	}
	tensors := makeBertTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph BERT-base: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildBertGraph_BertLarge(t *testing.T) {
	// BERT-large: 24 layers, 1024 hidden, 16 heads.
	cfg := &gguf.ModelConfig{
		Architecture:     "bert",
		VocabSize:        32,
		HiddenSize:       1024,
		NumLayers:        24,
		NumHeads:         16,
		NumKVHeads:       16,
		IntermediateSize: 4096,
		MaxSeqLen:        512,
		NumLabels:        3,
		LayerNormEps:     1e-12,
	}
	tensors := makeBertTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph BERT-large: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildBertGraph_ForwardProducesClassLogits(t *testing.T) {
	cfg := bertBaseConfig()
	tensors := makeBertTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph: %v", err)
	}

	// Input: token IDs [1, 4].
	tokenIDs := []float32{1, 5, 10, 3}
	seqLen := len(tokenIDs)
	_ = seqLen

	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	// BERT classification: output is [batch, numClasses].
	if len(shape) != 2 || shape[0] != 1 || shape[1] != cfg.NumLabels {
		t.Fatalf("unexpected output shape: %v, want [1, %d]", shape, cfg.NumLabels)
	}

	// Check no NaN/Inf.
	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestBuildBertGraph_MissingTensor(t *testing.T) {
	cfg := bertBaseConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildBertGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildBertGraph_Registered(t *testing.T) {
	builder, ok := GetArchitecture("bert")
	if !ok {
		t.Fatal("bert architecture not registered")
	}
	if builder == nil {
		t.Fatal("bert builder is nil")
	}
}

func TestBuildBertGraph_DefaultNumLabels(t *testing.T) {
	cfg := bertBaseConfig()
	cfg.NumLabels = 0 // should default to 2
	tensors := makeBertTestTensors(cfg)
	// Need cls tensors for 2 labels.
	hidden := cfg.HiddenSize
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
	zeros := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		t, _ := tensor.New(shape, data)
		return t
	}
	tensors["cls.weight"] = fill([]int{2, hidden}, 0.02)
	tensors["cls.bias"] = zeros([]int{2})

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph with default numLabels: %v", err)
	}

	input, _ := tensor.New([]int{1, 4}, []float32{1, 2, 3, 4})
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 2 || shape[1] != 2 {
		t.Fatalf("expected [1, 2] output, got %v", shape)
	}
}
