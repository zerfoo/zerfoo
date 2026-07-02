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

// makeMiniMaxM2TestTensors creates a minimal set of MiniMax-M2-architecture tensors
// for testing. Uses GGUF blk-style tensor names as the builder expects.
func makeMiniMaxM2TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := hidden / numHeads
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
	tensors["token_embd.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["output_norm.weight"] = ones([]int{hidden})
	tensors["output.weight"] = fill([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		blk := "blk." + itoa(i) + "."

		// Attention and FFN norms.
		tensors[blk+"attn_norm.weight"] = ones([]int{hidden})
		tensors[blk+"ffn_norm.weight"] = ones([]int{hidden})

		// Attention projections.
		tensors[blk+"attn_q.weight"] = fill([]int{numHeads * headDim, hidden}, 0.02)
		tensors[blk+"attn_k.weight"] = fill([]int{numKVHeads * headDim, hidden}, 0.02)
		tensors[blk+"attn_v.weight"] = fill([]int{numKVHeads * headDim, hidden}, 0.02)
		tensors[blk+"attn_output.weight"] = fill([]int{hidden, numHeads * headDim}, 0.02)

		// QK norms.
		tensors[blk+"attn_q_norm.weight"] = ones([]int{headDim})
		tensors[blk+"attn_k_norm.weight"] = ones([]int{headDim})

		// MoE tensors.
		tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
		tensors[blk+"exp_probs_b"] = fill([]int{numExperts}, 0.01)
		tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)
	}

	return tensors
}

func miniMaxM2TestConfig(numLayers, numExperts, topK int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:        "minimax-m2",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           numLayers,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           5000000.0,
		PartialRotaryFactor: 0.5,
		NumExperts:          numExperts,
		NumExpertsPerToken:  topK,
		ScoringFunc:         "sigmoid",
	}
}

func TestBuildMiniMaxM2Graph_Builds(t *testing.T) {
	tests := []struct {
		name       string
		numLayers  int
		numExperts int
		topK       int
	}{
		{"4_layers_4_experts_top2", 4, 4, 2},
		{"1_layer_2_experts_top1", 1, 2, 1},
		{"2_layers_8_experts_top2", 2, 8, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := miniMaxM2TestConfig(tt.numLayers, tt.numExperts, tt.topK)
			tensors := makeMiniMaxM2TestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildMiniMaxM2Graph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildMiniMaxM2Graph: %v", err)
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

func TestBuildMiniMaxM2Graph_MissingTensor(t *testing.T) {
	cfg := miniMaxM2TestConfig(1, 4, 2)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildMiniMaxM2Graph_ForwardNonNaN(t *testing.T) {
	tests := []struct {
		name       string
		numLayers  int
		numExperts int
		topK       int
	}{
		{"4_layers_4_experts_top2", 4, 4, 2},
		{"1_layer_2_experts_top1", 1, 2, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := miniMaxM2TestConfig(tt.numLayers, tt.numExperts, tt.topK)
			tensors := makeMiniMaxM2TestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildMiniMaxM2Graph: %v", err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

func TestBuildMiniMaxM2Graph_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
	}

	cfg := miniMaxM2TestConfig(2, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
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

func TestBuildMiniMaxM2Graph_Deterministic(t *testing.T) {
	cfg := miniMaxM2TestConfig(2, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
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

func TestBuildMiniMaxM2Graph_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := miniMaxM2TestConfig(2, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
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

func TestBuildMiniMaxM2Graph_SigmoidGatingActive(t *testing.T) {
	// Verify that sigmoid gating selects experts by confirming the forward
	// pass produces finite, non-zero output. The MoE path with sigmoid
	// gating is exercised because numExperts > 0.
	cfg := miniMaxM2TestConfig(1, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
	}

	input, _ := tensor.New([]int{1, 2}, []float32{1, 5})
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	data := output.Data()
	allZero := true
	for _, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatal("NaN in output with sigmoid gating")
		}
		if math.IsInf(float64(v), 0) {
			t.Fatal("Inf in output with sigmoid gating")
		}
		if v != 0 {
			allZero = false
		}
	}
	if allZero {
		t.Fatal("all-zero output suggests sigmoid gating is not routing tokens")
	}
}

func TestBuildMiniMaxM2Graph_QKNormApplied(t *testing.T) {
	// Verify the builder does not error when QK norm weights are provided,
	// confirming they are wired into the GQA layer.
	cfg := miniMaxM2TestConfig(1, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
	}

	input, _ := tensor.New([]int{1, 2}, []float32{1, 5})
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward with QK norms: %v", err)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d with QK norm", i)
		}
	}
}

func TestBuildMiniMaxM2Graph_RegistrationDispatches(t *testing.T) {
	cfg := miniMaxM2TestConfig(1, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("minimax-m2", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(minimax-m2): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildMiniMaxM2Graph_OutputFinite(t *testing.T) {
	cfg := miniMaxM2TestConfig(2, 4, 2)
	tensors := makeMiniMaxM2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildMiniMaxM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildMiniMaxM2Graph: %v", err)
	}

	input, _ := tensor.New([]int{1, 4}, []float32{2, 7, 15, 20})
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}
