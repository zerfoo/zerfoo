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

// makeLlama4TestTensors creates a minimal set of Llama 4 architecture tensors
// for testing. Llama 4 uses MoE with a temperature-based router and iRoPE
// (interleaved RoPE with no-position-embedding layers).
func makeLlama4TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads
	numExperts := cfg.NumExperts

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
		prefix := "model.layers." + itoa(i) + "."
		blk := "blk." + itoa(i) + "."

		// Norms.
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})

		// Attention projections (standard GQA like Llama 3).
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		// MoE tensors.
		if numExperts > 0 {
			tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
			tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)

			// Shared expert tensors.
			if cfg.NumSharedExperts > 0 {
				tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
				tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
				tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
			}
		} else {
			// Standard FFN for non-MoE layers.
			tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
		}
	}

	return tensors
}

// llama4ParityConfig returns a small Llama 4 config suitable for parity tests.
func llama4ParityConfig(numExperts, numExpertsPerTok, numSharedExperts int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:       "llama4",
		VocabSize:          32,
		HiddenSize:         16,
		NumLayers:          2,
		NumHeads:           4,
		NumKVHeads:         2,
		IntermediateSize:   32,
		MaxSeqLen:          64,
		RopeTheta:          500000.0,
		NumExperts:         numExperts,
		NumExpertsPerToken: numExpertsPerTok,
		NumSharedExperts:   numSharedExperts,
	}
}

func TestBuildLlama4Graph_Builds(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{
			name:             "MoE with shared expert",
			numExperts:       4,
			numExpertsPerTok: 2,
			numSharedExperts: 1,
		},
		{
			name:             "MoE without shared expert",
			numExperts:       4,
			numExpertsPerTok: 2,
			numSharedExperts: 0,
		},
		{
			name:             "dense (no MoE)",
			numExperts:       0,
			numExpertsPerTok: 0,
			numSharedExperts: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := llama4ParityConfig(tt.numExperts, tt.numExpertsPerTok, tt.numSharedExperts)
			tensors := makeLlama4TestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildLlama4Graph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildLlama4Graph: %v", err)
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

func TestBuildLlama4Graph_MissingTensor(t *testing.T) {
	cfg := llama4ParityConfig(4, 2, 1)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildLlama4Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestLlama4Forward(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{"MoE_with_shared_expert", 4, 2, 1},
		{"MoE_without_shared_expert", 4, 2, 0},
		{"dense_FFN", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := llama4ParityConfig(tt.numExperts, tt.numExpertsPerTok, tt.numSharedExperts)
			tensors := makeLlama4TestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildLlama4Graph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildLlama4Graph: %v", err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

func TestLlama4Forward_Deterministic(t *testing.T) {
	cfg := llama4ParityConfig(4, 2, 1)
	tensors := makeLlama4TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLlama4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph: %v", err)
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

func TestLlama4Forward_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := llama4ParityConfig(4, 2, 1)
	tensors := makeLlama4TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLlama4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph: %v", err)
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

func TestLlama4Forward_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
		{"four_tokens", 4, []float32{1, 5, 10, 3}},
	}

	cfg := llama4ParityConfig(4, 2, 1)
	tensors := makeLlama4TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLlama4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph: %v", err)
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
		})
	}
}

func TestLlama4Forward_MoEVsDense(t *testing.T) {
	moeConfig := llama4ParityConfig(4, 2, 1)
	denseConfig := llama4ParityConfig(0, 0, 0)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	moeTensors := makeLlama4TestTensors(moeConfig)
	denseTensors := makeLlama4TestTensors(denseConfig)

	moeGraph, _, err := buildLlama4Graph(moeTensors, moeConfig, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph MoE: %v", err)
	}
	denseGraph, _, err := buildLlama4Graph(denseTensors, denseConfig, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph dense: %v", err)
	}

	tokenIDs := []float32{1, 5, 10}
	input, _ := tensor.New([]int{1, 3}, tokenIDs)
	ctx := context.Background()

	moeOut, err := moeGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("MoE forward: %v", err)
	}
	denseOut, err := denseGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("dense forward: %v", err)
	}

	moeData := moeOut.Data()
	denseData := denseOut.Data()

	if len(moeData) != len(denseData) {
		t.Fatalf("output length mismatch: MoE=%d, dense=%d", len(moeData), len(denseData))
	}

	identical := true
	for i := range moeData {
		if moeData[i] != denseData[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("MoE and dense FFN produced identical output; expected different outputs")
	}
}

func TestLlama4Forward_TwoLayerResidualFlow(t *testing.T) {
	cfg1 := llama4ParityConfig(4, 2, 1)
	cfg1.NumLayers = 1

	cfg2 := llama4ParityConfig(4, 2, 1)
	cfg2.NumLayers = 2

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	t1 := makeLlama4TestTensors(cfg1)
	t2 := makeLlama4TestTensors(cfg2)

	g1, _, err := buildLlama4Graph(t1, cfg1, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph 1-layer: %v", err)
	}
	g2, _, err := buildLlama4Graph(t2, cfg2, engine)
	if err != nil {
		t.Fatalf("buildLlama4Graph 2-layer: %v", err)
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

func TestBuildArchGraph_Llama4Dispatches(t *testing.T) {
	cfg := llama4ParityConfig(4, 2, 1)
	tensors := makeLlama4TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("llama4", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(llama4): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestParseLlama4Config(t *testing.T) {
	raw := map[string]interface{}{
		"model_type":              "llama4",
		"vocab_size":              float64(128256),
		"hidden_size":             float64(5120),
		"num_hidden_layers":       float64(48),
		"num_attention_heads":     float64(40),
		"num_key_value_heads":     float64(8),
		"intermediate_size":       float64(8192),
		"max_position_embeddings": float64(10485760),
		"rope_theta":              float64(500000),
		"num_local_experts":       float64(16),
		"num_experts_per_tok":     float64(2),
		"num_shared_experts":      float64(1),
	}

	registry := DefaultArchConfigRegistry()
	meta, err := registry.Parse(raw)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if meta.Architecture != "llama4" {
		t.Fatalf("architecture = %q, want llama4", meta.Architecture)
	}
	if meta.NumExperts != 16 {
		t.Fatalf("num_experts = %d, want 16", meta.NumExperts)
	}
	if meta.NumExpertsPerToken != 2 {
		t.Fatalf("num_experts_per_tok = %d, want 2", meta.NumExpertsPerToken)
	}
	if meta.NumSharedExperts != 1 {
		t.Fatalf("num_shared_experts = %d, want 1", meta.NumSharedExperts)
	}
	if meta.RopeTheta != 500000 {
		t.Fatalf("rope_theta = %v, want 500000", meta.RopeTheta)
	}
}
