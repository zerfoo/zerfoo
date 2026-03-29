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

func lfm2TestConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "lfm2",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
}

func lfm2MoETestConfig(numExperts, topK int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:       "lfm2moe",
		VocabSize:          32,
		HiddenSize:         16,
		NumLayers:          4,
		NumHeads:           4,
		NumKVHeads:         2,
		IntermediateSize:   32,
		MaxSeqLen:          64,
		RopeTheta:          10000.0,
		NumExperts:         numExperts,
		NumExpertsPerToken: topK,
	}
}

// makeLFM2MoETestTensors creates tensors for the LFM2-MoE hybrid architecture.
// Even-indexed layers get MoE tensors; odd-indexed layers get dense FFN tensors.
func makeLFM2MoETestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = 8
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

	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		blk := "blk." + itoa(i) + "."

		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		if i%2 == 0 {
			// MoE layer.
			tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
			tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)
		} else {
			// Dense FFN layer.
			tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
		}
	}

	return tensors
}

func TestBuildLFM2Graph_Builds(t *testing.T) {
	cfg := lfm2TestConfig()
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildLFM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLFM2Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildLFM2Graph_ForwardNonNaN(t *testing.T) {
	cfg := lfm2TestConfig()
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLFM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLFM2Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildLFM2Graph_MissingTensor(t *testing.T) {
	cfg := lfm2TestConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildLFM2Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildLFM2MoEGraph_Builds(t *testing.T) {
	tests := []struct {
		name       string
		numExperts int
		topK       int
	}{
		{"8_experts_top2", 8, 2},
		{"4_experts_top1", 4, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := lfm2MoETestConfig(tt.numExperts, tt.topK)
			tensors := makeLFM2MoETestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildLFM2MoEGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildLFM2MoEGraph: %v", err)
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

func TestBuildLFM2MoEGraph_ForwardNonNaN(t *testing.T) {
	cfg := lfm2MoETestConfig(4, 2)
	tensors := makeLFM2MoETestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLFM2MoEGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLFM2MoEGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildLFM2MoEGraph_MissingTensor(t *testing.T) {
	cfg := lfm2MoETestConfig(8, 2)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildLFM2MoEGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_LFM2Dispatches(t *testing.T) {
	tests := []struct {
		name string
		arch string
	}{
		{"lfm2", "lfm2"},
		{"lfm2moe", "lfm2moe"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var g interface{ Forward(context.Context, ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) }
			var emb *tensor.TensorNumeric[float32]
			var err error

			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			if tt.arch == "lfm2" {
				cfg := lfm2TestConfig()
				tensors := makeLlamaTestTensors(cfg)
				g, emb, err = buildArchGraph(tt.arch, tensors, cfg, engine)
			} else {
				cfg := lfm2MoETestConfig(4, 2)
				tensors := makeLFM2MoETestTensors(cfg)
				g, emb, err = buildArchGraph(tt.arch, tensors, cfg, engine)
			}

			if err != nil {
				t.Fatalf("buildArchGraph(%s): %v", tt.arch, err)
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

func TestLFM2MoEGraph_HybridDispatch(t *testing.T) {
	// Verify the hybrid MoE/dense dispatch by checking that the graph builds
	// with alternating MoE and dense FFN layers (4 layers: MoE, dense, MoE, dense).
	cfg := lfm2MoETestConfig(4, 2)
	tensors := makeLFM2MoETestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLFM2MoEGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLFM2MoEGraph: %v", err)
	}

	// Forward pass should work through both MoE and dense layers.
	input, _ := tensor.New([]int{1, 3}, []float32{1, 5, 10})
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != cfg.VocabSize {
		t.Fatalf("output shape = %v, want [1, 3, %d]", shape, cfg.VocabSize)
	}
}

func TestLFM2Forward_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := lfm2TestConfig()
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildLFM2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildLFM2Graph: %v", err)
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
		t.Fatal("different inputs produced identical output")
	}
}
