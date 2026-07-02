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

func kimiTestConfig(numExperts, topK int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:       "kimi-linear",
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

// makeKimiTestTensors creates a minimal set of tensors for Kimi linear
// attention MoE testing.
func makeKimiTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
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

		// Q has numHeads * headDim = hidden. K/V have kvHeads * headDim = kvDim.
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
		tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
		tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)
	}

	return tensors
}

func TestBuildKimiLinearGraph_Builds(t *testing.T) {
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
			cfg := kimiTestConfig(tt.numExperts, tt.topK)
			tensors := makeKimiTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildKimiLinearGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildKimiLinearGraph: %v", err)
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

func TestBuildKimiLinearGraph_ForwardNonNaN(t *testing.T) {
	cfg := kimiTestConfig(4, 2)
	tensors := makeKimiTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildKimiLinearGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildKimiLinearGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildKimiLinearGraph_MissingTensor(t *testing.T) {
	cfg := kimiTestConfig(8, 2)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildKimiLinearGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_KimiLinearDispatches(t *testing.T) {
	cfg := kimiTestConfig(4, 2)
	tensors := makeKimiTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("kimi-linear", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(kimi-linear): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestKimiLinearAttention_OutputDiffers(t *testing.T) {
	cfg := kimiTestConfig(4, 2)
	tensors := makeKimiTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildKimiLinearGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildKimiLinearGraph: %v", err)
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
		t.Fatal("different inputs produced identical output; linear attention may be collapsing")
	}
}

func TestEluPlus1(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{"positive", 2.0, 3.0},
		{"zero", 0.0, 1.0},
		{"negative", -1.0, math.Exp(-1.0)},
		{"large_negative", -10.0, math.Exp(-10.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := eluPlus1(tt.x)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Fatalf("eluPlus1(%v) = %v, want %v", tt.x, got, tt.want)
			}
		})
	}
}

func TestKimiForward_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
	}

	cfg := kimiTestConfig(4, 2)
	tensors := makeKimiTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildKimiLinearGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildKimiLinearGraph: %v", err)
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
