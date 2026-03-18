package main

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// TestMultiArchBenchmark is a table-driven test that builds and runs a forward
// pass for all 6 supported architectures (Llama3, Gemma3, Mistral, Qwen2,
// Phi3, DeepSeek V3) using synthetic weights. It verifies each architecture
// produces non-NaN/Inf output with the expected shape [1, 4, vocabSize].
func TestMultiArchBenchmark(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		cfg     *gguf.ModelConfig
		tensors func(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32]
	}{
		{
			name: "Llama3",
			arch: "llama",
			cfg: &gguf.ModelConfig{
				Architecture:     "llama",
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        2,
				NumHeads:         4,
				NumKVHeads:       2,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        500000.0,
			},
			tensors: makeStandardTestTensors,
		},
		{
			name: "Gemma3",
			arch: "gemma3",
			cfg: &gguf.ModelConfig{
				Architecture:     "gemma3",
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        2,
				NumHeads:         4,
				NumKVHeads:       2,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        10000.0,
			},
			tensors: makeGemma3TestTensors,
		},
		{
			name: "Mistral",
			arch: "mistral",
			cfg: &gguf.ModelConfig{
				Architecture:     "mistral",
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        2,
				NumHeads:         4,
				NumKVHeads:       2,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        10000.0,
				SlidingWindow:    32,
			},
			tensors: makeStandardTestTensors,
		},
		{
			name: "Qwen2",
			arch: "qwen2",
			cfg: &gguf.ModelConfig{
				Architecture:     "qwen2",
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        2,
				NumHeads:         4,
				NumKVHeads:       2,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        1000000.0,
			},
			tensors: makeQwen2TestTensors,
		},
		{
			name: "Phi3",
			arch: "phi3",
			cfg: &gguf.ModelConfig{
				Architecture:        "phi3",
				VocabSize:           32,
				HiddenSize:          16,
				NumLayers:           2,
				NumHeads:            4,
				NumKVHeads:          2,
				IntermediateSize:    32,
				MaxSeqLen:           64,
				RopeTheta:           10000.0,
				PartialRotaryFactor: 0.5,
			},
			tensors: makeStandardTestTensors,
		},
		{
			name: "DeepSeekV3",
			arch: "deepseek_v3",
			cfg: &gguf.ModelConfig{
				Architecture:       "deepseek_v3",
				VocabSize:          32,
				HiddenSize:         16,
				NumLayers:          2,
				NumHeads:           4,
				NumKVHeads:         4,
				IntermediateSize:   32,
				MaxSeqLen:          64,
				RopeTheta:          10000.0,
				KVLoRADim:          8,
				NumExperts:         4,
				NumExpertsPerToken: 2,
				NumSharedExperts:   1,
			},
			tensors: makeDeepSeekV3TestTensors,
		},
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := tt.tensors(tt.cfg)

			g, _, err := inference.BuildArchGraph(tt.arch, tensors, tt.cfg, engine)
			if err != nil {
				t.Fatalf("BuildArchGraph(%q): %v", tt.arch, err)
			}
			if g == nil {
				t.Fatal("graph is nil")
			}

			// Forward pass with 4 tokens.
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

			// Assert output shape [1, 4, vocabSize].
			shape := output.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != tt.cfg.VocabSize {
				t.Fatalf("output shape = %v, want [1, %d, %d]", shape, seqLen, tt.cfg.VocabSize)
			}

			// Assert no NaN/Inf values.
			for i, v := range output.Data() {
				if math.IsNaN(float64(v)) {
					t.Fatalf("NaN at index %d", i)
				}
				if math.IsInf(float64(v), 0) {
					t.Fatalf("Inf at index %d", i)
				}
			}

			t.Logf("%s: output shape %v, all values finite", tt.name, shape)
		})
	}
}

// --- Synthetic weight tensor constructors ---

func fillTensor(shape []int, scale float32) *tensor.TensorNumeric[float32] {
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

func onesTensor(shape []int) *tensor.TensorNumeric[float32] {
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

func zerosTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	t, _ := tensor.New(shape, data)
	return t
}

func itoa(i int) string {
	return fmt.Sprintf("%d", i)
}

// makeStandardTestTensors creates synthetic tensors for Llama/Mistral/Phi
// architectures (standard transformer with SwiGLU FFN).
func makeStandardTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	tensors["model.embed_tokens.weight"] = fillTensor([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = onesTensor([]int{hidden})
	tensors["lm_head.weight"] = fillTensor([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fillTensor([]int{hidden, inter}, 0.02)
	}

	return tensors
}

// makeGemma3TestTensors creates synthetic tensors for Gemma 3 which requires
// additional post-norm, Q/K norm, and pre-feedforward layernorm tensors.
// Gemma ties lm_head to embedding weights.
func makeGemma3TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeStandardTestTensors(cfg)
	delete(tensors, "lm_head.weight") // Gemma ties lm_head to embedding

	hidden := cfg.HiddenSize
	headDim := hidden / cfg.NumHeads

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		// Q/K norms for Gemma 3.
		tensors[prefix+"self_attn.q_norm.weight"] = onesTensor([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = onesTensor([]int{headDim})
		// Post-attention norm (for postNorm path).
		// post_attention_layernorm.weight already exists from standard tensors.
		// Gemma 3 postNorm path needs pre_feedforward and post_feedforward norms.
		tensors[prefix+"pre_feedforward_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = onesTensor([]int{hidden})
	}

	return tensors
}

// makeQwen2TestTensors creates synthetic tensors for Qwen 2, which includes
// attention bias vectors for Q/K/V projections.
func makeQwen2TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeStandardTestTensors(cfg)

	kvDim := (cfg.HiddenSize / cfg.NumHeads) * cfg.NumKVHeads

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"self_attn.q_proj.bias"] = zerosTensor([]int{cfg.HiddenSize})
		tensors[prefix+"self_attn.k_proj.bias"] = zerosTensor([]int{kvDim})
		tensors[prefix+"self_attn.v_proj.bias"] = zerosTensor([]int{kvDim})
	}

	return tensors
}

// makeDeepSeekV3TestTensors creates synthetic tensors for DeepSeek V3 with
// MLA and MoE tensors.
func makeDeepSeekV3TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvLoraDim := cfg.KVLoRADim
	numHeads := cfg.NumHeads
	headDim := hidden / numHeads
	numExperts := cfg.NumExperts

	tensors["model.embed_tokens.weight"] = fillTensor([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = onesTensor([]int{hidden})
	tensors["lm_head.weight"] = fillTensor([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		blk := "blk." + itoa(i) + "."

		tensors[prefix+"input_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = onesTensor([]int{hidden})

		// MLA tensors.
		tensors[blk+"attn_kv_a_proj_with_mqa.weight"] = fillTensor([]int{kvLoraDim, hidden}, 0.02)
		tensors[blk+"attn_kv_b_proj.weight"] = fillTensor([]int{numHeads * headDim * 2, kvLoraDim}, 0.02)
		tensors[prefix+"self_attn.q_proj.weight"] = fillTensor([]int{numHeads * headDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fillTensor([]int{hidden, numHeads * headDim}, 0.02)

		// MoE tensors.
		if numExperts > 0 {
			tensors[blk+"ffn_gate_inp.weight"] = fillTensor([]int{numExperts, hidden}, 0.02)
			tensors[blk+"ffn_gate_exps.weight"] = fillTensor([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_up_exps.weight"] = fillTensor([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_down_exps.weight"] = fillTensor([]int{numExperts, hidden, inter}, 0.02)

			if cfg.NumSharedExperts > 0 {
				tensors[blk+"ffn_shared_expert_gate.weight"] = fillTensor([]int{inter, hidden}, 0.02)
				tensors[blk+"ffn_shared_expert_up.weight"] = fillTensor([]int{inter, hidden}, 0.02)
				tensors[blk+"ffn_shared_expert_down.weight"] = fillTensor([]int{hidden, inter}, 0.02)
			}
		} else {
			tensors[prefix+"mlp.gate_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.up_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.down_proj.weight"] = fillTensor([]int{hidden, inter}, 0.02)
		}
	}

	return tensors
}
