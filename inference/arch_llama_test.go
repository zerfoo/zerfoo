package inference

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeLlamaTestTensors creates a minimal set of Llama-architecture tensors
// with canonical names for testing. All weights are initialized with small
// random-like values derived from position to produce non-NaN output.
func makeLlamaTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
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
			// Small deterministic values to avoid NaN.
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
	}

	return tensors
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}

func TestBuildLlamaGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
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

	g, emb, err := buildLlamaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlamaGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

// assertGraphForwardNonNaN runs a forward pass through a graph with test token IDs
// and asserts the output is non-NaN/Inf with the expected shape.
func assertGraphForwardNonNaN(t *testing.T, g interface {
	Forward(context.Context, ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
}, vocabSize int) {
	t.Helper()

	tokenIDs := []float32{1, 5, 10, 3}
	seqLen := len(tokenIDs)

	input, err := tensor.New([]int{1, seqLen}, tokenIDs)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != vocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, vocabSize)
	}

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

func TestBuildLlamaGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
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

	g, _, err := buildLlamaGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlamaGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestLMHeadNode_FP16WeightOutputIsF32(t *testing.T) {
	// When the LM head weight has Float16Storage, the MatMul output
	// should be converted to F32 for sampling compatibility.
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	vocabSize, hiddenDim := 8, 4

	// Create weight with Float16Storage.
	f32Data := make([]float32, vocabSize*hiddenDim)
	for i := range f32Data {
		f32Data[i] = float32(i%7+1) * 0.01
	}
	fp16Stor := tensor.NewFloat16StorageFromF32(f32Data)
	weight, err := tensor.NewWithStorage[float32]([]int{vocabSize, hiddenDim}, fp16Stor)
	if err != nil {
		t.Fatalf("create fp16 weight: %v", err)
	}

	head := &lmHeadNode[float32]{engine: engine, weight: weight}

	// Input: [1, 2, hiddenDim]
	inputData := make([]float32, 2*hiddenDim)
	for i := range inputData {
		inputData[i] = float32(i%5+1) * 0.1
	}
	input, err := tensor.New([]int{1, 2, hiddenDim}, inputData)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := head.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output should have shape [1, 2, vocabSize].
	if s := output.Shape(); len(s) != 3 || s[0] != 1 || s[1] != 2 || s[2] != vocabSize {
		t.Fatalf("shape = %v, want [1, 2, %d]", s, vocabSize)
	}

	// On CPU engine, Float16Storage goes through CPU matmul path which
	// produces regular float32 data. Verify output is valid (non-NaN).
	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("invalid value at %d: %v", i, v)
		}
	}
}

func TestBuildLlamaGraph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}

	// Empty tensors -- should fail with a clear error.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildLlamaGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}
