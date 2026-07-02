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

// makeDBRXTestTensors creates a minimal set of DBRX architecture tensors
// for testing. All layers are MoE with stacked expert weights.
func makeDBRXTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = 16
	}
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

		// Attention projections (GQA).
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		// MoE tensors (all layers are MoE in DBRX).
		tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
		tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)
	}

	return tensors
}

// dbrxTestConfig returns a small DBRX config suitable for testing.
func dbrxTestConfig(numExperts, topK int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:       "dbrx",
		VocabSize:          32,
		HiddenSize:         16,
		NumLayers:          2,
		NumHeads:           4,
		NumKVHeads:         2,
		IntermediateSize:   32,
		MaxSeqLen:          64,
		RopeTheta:          10000.0,
		NumExperts:         numExperts,
		NumExpertsPerToken: topK,
	}
}

func TestBuildDBRXGraph_Builds(t *testing.T) {
	tests := []struct {
		name       string
		numExperts int
		topK       int
	}{
		{"4_experts_top2", 4, 2},
		{"4_experts_top1", 4, 1},
		{"8_experts_top4", 8, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := dbrxTestConfig(tt.numExperts, tt.topK)
			tensors := makeDBRXTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildDBRXGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDBRXGraph: %v", err)
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

func TestBuildDBRXGraph_MissingTensor(t *testing.T) {
	cfg := dbrxTestConfig(4, 2)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildDBRXGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestDBRXForward(t *testing.T) {
	tests := []struct {
		name       string
		numExperts int
		topK       int
	}{
		{"4_experts_top2", 4, 2},
		{"4_experts_top1", 4, 1},
		{"8_experts_top4", 8, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := dbrxTestConfig(tt.numExperts, tt.topK)
			tensors := makeDBRXTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildDBRXGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDBRXGraph: %v", err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

func TestDBRXForward_Deterministic(t *testing.T) {
	cfg := dbrxTestConfig(4, 2)
	tensors := makeDBRXTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDBRXGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDBRXGraph: %v", err)
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

func TestDBRXForward_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
		{"four_tokens", 4, []float32{1, 5, 10, 3}},
	}

	cfg := dbrxTestConfig(4, 2)
	tensors := makeDBRXTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDBRXGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDBRXGraph: %v", err)
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

func TestDBRXForward_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := dbrxTestConfig(4, 2)
	tensors := makeDBRXTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDBRXGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDBRXGraph: %v", err)
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

func TestDBRXForward_MoERoutingTopK(t *testing.T) {
	cfgTop1 := dbrxTestConfig(4, 1)
	cfgTop2 := dbrxTestConfig(4, 2)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tensorsTop1 := makeDBRXTestTensors(cfgTop1)
	tensorsTop2 := makeDBRXTestTensors(cfgTop2)

	// Use same weights to test the routing difference.
	for k, v := range tensorsTop1 {
		tensorsTop2[k] = v
	}

	gTop1, _, err := buildDBRXGraph(tensorsTop1, cfgTop1, engine)
	if err != nil {
		t.Fatalf("buildDBRXGraph top-1: %v", err)
	}
	gTop2, _, err := buildDBRXGraph(tensorsTop2, cfgTop2, engine)
	if err != nil {
		t.Fatalf("buildDBRXGraph top-2: %v", err)
	}

	ctx := context.Background()
	input, _ := tensor.New([]int{1, 3}, []float32{1, 5, 10})

	out1, err := gTop1.Forward(ctx, input)
	if err != nil {
		t.Fatalf("forward top-1: %v", err)
	}
	out2, err := gTop2.Forward(ctx, input)
	if err != nil {
		t.Fatalf("forward top-2: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()
	if len(d1) != len(d2) {
		t.Fatalf("output length mismatch: top-1=%d, top-2=%d", len(d1), len(d2))
	}

	identical := true
	for i := range d1 {
		if d1[i] != d2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("top-1 and top-2 routing produced identical outputs; routing has no effect")
	}
}

func TestBuildArchGraph_DBRXDispatches(t *testing.T) {
	cfg := dbrxTestConfig(4, 2)
	tensors := makeDBRXTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("dbrx", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(dbrx): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}
