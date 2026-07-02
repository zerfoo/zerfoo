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

// makeCommandRTestTensors creates a minimal set of Command R architecture tensors.
// Command R uses LayerNorm (gamma + beta) instead of RMSNorm (gamma only).
func makeCommandRTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
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
		t, _ := tensor.New(shape, make([]float32, size))
		return t
	}

	// Global tensors.
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["model.norm.bias"] = zeros([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."

		// LayerNorm parameters (gamma + beta, unlike RMSNorm which has only gamma).
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"input_layernorm.bias"] = zeros([]int{hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_attention_layernorm.bias"] = zeros([]int{hidden})

		// Attention projections (standard GQA).
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		// FFN projections (SwiGLU: gate, up, down).
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

// commandRConfig returns a small Command R config for unit tests.
func commandRConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "command-r",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        128,
		RopeTheta:        10000.0,
	}
}

func TestCommandRForward(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"two_tokens", 2, []float32{1, 7}},
		{"four_tokens", 4, []float32{1, 5, 10, 3}},
	}

	cfg := commandRConfig()
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildCommandRGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.New([]int{1, tt.seqLen}, tt.tokens)
			if err != nil {
				t.Fatalf("create input: %v", err)
			}

			output, err := g.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}

			shape := output.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != tt.seqLen || shape[2] != cfg.VocabSize {
				t.Fatalf("output shape = %v, want [1, %d, %d]", shape, tt.seqLen, cfg.VocabSize)
			}

			for i, v := range output.Data() {
				if math.IsNaN(float64(v)) {
					t.Fatalf("NaN at index %d", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Fatalf("Inf at index %d", i)
				}
			}
		})
	}
}

func TestBuildCommandRGraph_Builds(t *testing.T) {
	cfg := commandRConfig()
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildCommandRGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildCommandRGraph_MissingTensor(t *testing.T) {
	cfg := commandRConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildCommandRGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestCommandRForward_Deterministic(t *testing.T) {
	cfg := commandRConfig()
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildCommandRGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph: %v", err)
	}

	tokenIDs := []float32{1, 5, 10}
	input, err := tensor.New([]int{1, 3}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()

	out1, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	data1 := make([]float32, len(out1.Data()))
	copy(data1, out1.Data())

	out2, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	data2 := out2.Data()

	if len(data1) != len(data2) {
		t.Fatalf("output length mismatch: %d vs %d", len(data1), len(data2))
	}
	const tol = 1e-5
	for i := range data1 {
		diff := float64(data1[i]) - float64(data2[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Fatalf("output differs at index %d: %v vs %v (diff=%v)", i, data1[i], data2[i], diff)
		}
	}
}

func TestCommandRForward_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := commandRConfig()
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildCommandRGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph: %v", err)
	}

	ctx := context.Background()
	input1, _ := tensor.New([]int{1, 3}, []float32{1, 2, 3})
	input2, _ := tensor.New([]int{1, 3}, []float32{4, 5, 6})

	out1, err := g.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("forward input1: %v", err)
	}
	out2, err := g.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("forward input2: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()
	identical := true
	for i := range d1 {
		if d1[i] != d2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("different inputs produced identical output; model may be collapsing")
	}
}

func TestCommandRForward_LongContext(t *testing.T) {
	// Command R is designed for 128K context; test that a 64-token sequence works.
	cfg := commandRConfig()
	cfg.MaxSeqLen = 131072
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildCommandRGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph: %v", err)
	}

	// Use 64 tokens to keep test fast.
	tokens := make([]float32, 64)
	for i := range tokens {
		tokens[i] = float32(i % cfg.VocabSize)
	}
	input, err := tensor.New([]int{1, 64}, tokens)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 64 || shape[2] != cfg.VocabSize {
		t.Fatalf("output shape = %v, want [1, 64, %d]", shape, cfg.VocabSize)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("invalid value at index %d: %v", i, v)
		}
	}
}

func TestCommandRForward_TwoLayerResidualFlow(t *testing.T) {
	cfg1 := commandRConfig()
	cfg1.NumLayers = 1

	cfg2 := commandRConfig()
	cfg2.NumLayers = 2

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g1, _, err := buildCommandRGraph(makeCommandRTestTensors(cfg1), cfg1, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph 1-layer: %v", err)
	}
	g2, _, err := buildCommandRGraph(makeCommandRTestTensors(cfg2), cfg2, engine)
	if err != nil {
		t.Fatalf("buildCommandRGraph 2-layer: %v", err)
	}

	input, _ := tensor.New([]int{1, 3}, []float32{1, 5, 10})
	ctx := context.Background()

	out1, err := g1.Forward(ctx, input)
	if err != nil {
		t.Fatalf("1-layer forward: %v", err)
	}
	out2, err := g2.Forward(ctx, input)
	if err != nil {
		t.Fatalf("2-layer forward: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()
	if len(d1) != len(d2) {
		t.Fatalf("output length mismatch: 1-layer=%d, 2-layer=%d", len(d1), len(d2))
	}

	identical := true
	for i := range d1 {
		if d1[i] != d2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("1-layer and 2-layer models produced identical output; second layer has no effect")
	}
}

func TestBuildArchGraph_CommandRDispatches(t *testing.T) {
	cfg := commandRConfig()
	tensors := makeCommandRTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("command-r", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(command-r): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}
