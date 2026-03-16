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

// deepSeekParityConfig returns a small DeepSeek V3 config suitable for parity tests.
func deepSeekParityConfig(numExperts, numExpertsPerTok, numSharedExperts int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
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
		NumExperts:         numExperts,
		NumExpertsPerToken: numExpertsPerTok,
		NumSharedExperts:   numSharedExperts,
	}
}

// TestDeepSeekParity_ForwardNonNaN verifies the full forward pass produces
// non-NaN/Inf output for both MoE and non-MoE configurations.
func TestDeepSeekParity_ForwardNonNaN(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{"MoE_with_shared_expert", 4, 2, 1},
		{"MoE_without_shared_expert", 4, 2, 0},
		{"standard_FFN", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := deepSeekParityConfig(tt.numExperts, tt.numExpertsPerTok, tt.numSharedExperts)
			tensors := makeDeepSeekTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDeepSeekGraph: %v", err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

// TestDeepSeekParity_Deterministic verifies that the same input produces
// identical output across multiple forward passes.
func TestDeepSeekParity_Deterministic(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{"MoE", 4, 2, 1},
		{"standard_FFN", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := deepSeekParityConfig(tt.numExperts, tt.numExpertsPerTok, tt.numSharedExperts)
			tensors := makeDeepSeekTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDeepSeekGraph: %v", err)
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
			// Use a small tolerance because floating-point accumulation order
			// in the MoE batched path can cause LSB differences.
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
		})
	}
}

// TestDeepSeekParity_MoEVsStandardFFN verifies that MoE and standard FFN
// configurations produce different outputs given identical embeddings and
// attention weights. This confirms the router is actively selecting experts
// rather than degenerating to a single path.
func TestDeepSeekParity_MoEVsStandardFFN(t *testing.T) {
	moeConfig := deepSeekParityConfig(4, 2, 1)
	ffnConfig := deepSeekParityConfig(0, 0, 0)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	moeTensors := makeDeepSeekTestTensors(moeConfig)
	ffnTensors := makeDeepSeekTestTensors(ffnConfig)

	moeGraph, _, err := buildDeepSeekGraph(moeTensors, moeConfig, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph MoE: %v", err)
	}
	ffnGraph, _, err := buildDeepSeekGraph(ffnTensors, ffnConfig, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph FFN: %v", err)
	}

	tokenIDs := []float32{1, 5, 10}
	input, err := tensor.New([]int{1, 3}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	ctx := context.Background()

	moeOut, err := moeGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("MoE forward: %v", err)
	}
	ffnOut, err := ffnGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("FFN forward: %v", err)
	}

	moeData := moeOut.Data()
	ffnData := ffnOut.Data()

	if len(moeData) != len(ffnData) {
		t.Fatalf("output length mismatch: MoE=%d, FFN=%d", len(moeData), len(ffnData))
	}

	identical := true
	for i := range moeData {
		if moeData[i] != ffnData[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("MoE and standard FFN produced identical output; expected different outputs")
	}
}

// TestDeepSeekParity_DifferentInputsDifferentOutputs verifies that different
// token sequences produce different logits, confirming the model is not
// collapsing to a constant function.
func TestDeepSeekParity_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := deepSeekParityConfig(4, 2, 1)
	tensors := makeDeepSeekTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph: %v", err)
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

	data1 := out1.Data()
	data2 := out2.Data()

	identical := true
	for i := range data1 {
		if data1[i] != data2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("different inputs produced identical output; model may be collapsing")
	}
}

// TestDeepSeekParity_OutputShape verifies the output tensor shape is
// [batch, seqLen, vocabSize] for various sequence lengths.
func TestDeepSeekParity_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
		{"four_tokens", 4, []float32{1, 5, 10, 3}},
	}

	cfg := deepSeekParityConfig(4, 2, 1)
	tensors := makeDeepSeekTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph: %v", err)
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

// TestDeepSeekParity_RoPEPositionSensitivity verifies that RoPE makes the
// output position-dependent. Same tokens at different positions (achieved by
// padding) should produce different attention patterns and thus different output.
func TestDeepSeekParity_RoPEPositionSensitivity(t *testing.T) {
	cfg := deepSeekParityConfig(0, 0, 0) // standard FFN for simpler test
	tensors := makeDeepSeekTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph: %v", err)
	}

	ctx := context.Background()

	// Sequence [1, 5] vs [0, 1, 5] -- token 5 is at position 1 vs position 2.
	// The last-position logits should differ due to RoPE.
	input1, _ := tensor.New([]int{1, 2}, []float32{1, 5})
	input2, _ := tensor.New([]int{1, 3}, []float32{0, 1, 5})

	out1, err := g.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("forward short: %v", err)
	}
	out2, err := g.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("forward long: %v", err)
	}

	// Compare last-position logits.
	d1 := out1.Data()
	d2 := out2.Data()
	vocab := cfg.VocabSize

	// Last position of out1: index [0, 1, :] = d1[1*vocab : 2*vocab]
	lastLogits1 := d1[1*vocab : 2*vocab]
	// Last position of out2: index [0, 2, :] = d2[2*vocab : 3*vocab]
	lastLogits2 := d2[2*vocab : 3*vocab]

	identical := true
	for i := 0; i < vocab; i++ {
		if lastLogits1[i] != lastLogits2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("same tokens at different positions produced identical last-position logits; RoPE may not be applied")
	}
}

// TestDeepSeekParity_TwoLayerResidualFlow verifies the 2-layer model produces
// different output than a 1-layer model, confirming residual connections
// propagate through multiple layers.
func TestDeepSeekParity_TwoLayerResidualFlow(t *testing.T) {
	cfg1 := deepSeekParityConfig(4, 2, 1)
	cfg1.NumLayers = 1

	cfg2 := deepSeekParityConfig(4, 2, 1)
	cfg2.NumLayers = 2

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	t1 := makeDeepSeekTestTensors(cfg1)
	t2 := makeDeepSeekTestTensors(cfg2)

	g1, _, err := buildDeepSeekGraph(t1, cfg1, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph 1-layer: %v", err)
	}
	g2, _, err := buildDeepSeekGraph(t2, cfg2, engine)
	if err != nil {
		t.Fatalf("buildDeepSeekGraph 2-layer: %v", err)
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

// TestDeepSeekParity_OutputFinite verifies all output values are finite
// (no NaN, no Inf) across all configurations with multi-token input.
func TestDeepSeekParity_OutputFinite(t *testing.T) {
	tests := []struct {
		name             string
		numExperts       int
		numExpertsPerTok int
		numSharedExperts int
	}{
		{"MoE_4exp_2active_1shared", 4, 2, 1},
		{"MoE_4exp_2active_0shared", 4, 2, 0},
		{"standard_FFN", 0, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := deepSeekParityConfig(tt.numExperts, tt.numExpertsPerTok, tt.numSharedExperts)
			tensors := makeDeepSeekTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildDeepSeekGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildDeepSeekGraph: %v", err)
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
		})
	}
}
