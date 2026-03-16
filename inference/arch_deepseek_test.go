package inference

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeDeepSeekTestTensors creates a minimal set of DeepSeek-architecture tensors
// for testing. Uses both HuggingFace-mapped names and GGUF names (for MLA/MoE
// tensors that pass through MapTensorName unchanged).
func makeDeepSeekTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvLoraDim := cfg.KVLoRADim
	numHeads := cfg.NumHeads
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

	// Global tensors (HuggingFace-mapped names).
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		blk := "blk." + itoa(i) + "."

		// LayerNorm weights (mapped names).
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})

		// MLA tensors (GGUF names, unmapped).
		tensors[blk+"attn_kv_a_proj_with_mqa.weight"] = fill([]int{kvLoraDim, hidden}, 0.02)
		// B projection contains both K and V up-projections concatenated.
		tensors[blk+"attn_kv_b_proj.weight"] = fill([]int{numHeads * headDim * 2, kvLoraDim}, 0.02)
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{numHeads * headDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, numHeads * headDim}, 0.02)

		// MoE tensors (GGUF names).
		if numExperts > 0 {
			tensors[blk+"ffn_gate_inp.weight"] = fill([]int{numExperts, hidden}, 0.02)
			tensors[blk+"ffn_gate_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_up_exps.weight"] = fill([]int{numExperts, inter, hidden}, 0.02)
			tensors[blk+"ffn_down_exps.weight"] = fill([]int{numExperts, hidden, inter}, 0.02)

			// Shared expert tensors.
			if cfg.NumSharedExperts > 0 {
				tensors[blk+"ffn_shared_expert_gate.weight"] = fill([]int{inter, hidden}, 0.02)
				tensors[blk+"ffn_shared_expert_up.weight"] = fill([]int{inter, hidden}, 0.02)
				tensors[blk+"ffn_shared_expert_down.weight"] = fill([]int{hidden, inter}, 0.02)
			}
		} else {
			// Standard FFN tensors for non-MoE layers.
			tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
			tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
		}
	}

	return tensors
}

func TestBuildDeepSeekGraph_Builds(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{
			name:             "with MoE",
			numExperts:       4,
			numExpertsPerTok: 2,
			numSharedExperts: 1,
		},
		{
			name:             "without MoE",
			numExperts:       0,
			numExpertsPerTok: 0,
			numSharedExperts: 0,
		},
		{
			name:             "MoE without shared experts",
			numExperts:       4,
			numExpertsPerTok: 2,
			numSharedExperts: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:     "deepseek_v3",
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        1,
				NumHeads:         4,
				NumKVHeads:       4,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        10000.0,
				KVLoRADim:        8,
				NumExperts:       tt.numExperts,
				NumExpertsPerToken: tt.numExpertsPerTok,
				NumSharedExperts: tt.numSharedExperts,
			}
			tensors := makeDeepSeekTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildDeepSeekGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDeepSeekGraph: %v", err)
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

func TestBuildDeepSeekGraph_MissingTensor(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:     "deepseek_v3",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       4,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		KVLoRADim:        8,
		NumExperts:       4,
		NumExpertsPerToken: 2,
	}

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildDeepSeekGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_DeepSeekDispatches(t *testing.T) {
	tests := []struct {
		arch string
	}{
		{"deepseek_v3"},
		{"deepseek2"},
	}

	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:     tt.arch,
				VocabSize:        32,
				HiddenSize:       16,
				NumLayers:        1,
				NumHeads:         4,
				NumKVHeads:       4,
				IntermediateSize: 32,
				MaxSeqLen:        64,
				RopeTheta:        10000.0,
				KVLoRADim:        8,
				NumExperts:       0,
			}
			tensors := makeDeepSeekTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildArchGraph(tt.arch, tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildArchGraph(%q): %v", tt.arch, err)
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

func TestExtractExpertSlice(t *testing.T) {
	tests := []struct {
		name       string
		shape      []int
		data       []float32
		expertIdx  int
		numExperts int
		wantShape  []int
	}{
		{
			name:       "3D stacked",
			shape:      []int{2, 3, 4},
			data:       make([]float32, 24),
			expertIdx:  1,
			numExperts: 2,
			wantShape:  []int{3, 4},
		},
		{
			name:       "2D stacked",
			shape:      []int{6, 4},
			data:       make([]float32, 24),
			expertIdx:  0,
			numExperts: 2,
			wantShape:  []int{3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i := range tt.data {
				tt.data[i] = float32(i)
			}
			stacked, err := tensor.New(tt.shape, tt.data)
			if err != nil {
				t.Fatalf("create tensor: %v", err)
			}
			slice, err := extractExpertSlice(stacked, tt.expertIdx, tt.numExperts)
			if err != nil {
				t.Fatalf("extractExpertSlice: %v", err)
			}
			gotShape := slice.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tt.wantShape)
			}
			for i, d := range gotShape {
				if d != tt.wantShape[i] {
					t.Fatalf("shape = %v, want %v", gotShape, tt.wantShape)
				}
			}
		})
	}
}
