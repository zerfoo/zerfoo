package inference

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeGemma4_31BTestTensors creates a minimal set of Gemma 4 architecture tensors.
// Gemma 4 builds on the Gemma 3 base (tied embeddings, 4 norms per layer, Q/K norms)
// but uses per-layer varying KV head counts and head dimensions via the per-layer
// loop builder.
func makeGemma4_31BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	// Start from Gemma 3 tensors (which include post-norms, pre-FFN norms,
	// post-FFN norms, and Q/K norms).
	tensors := makeGemma3TestTensors(cfg)
	return tensors
}

func TestBuildGemma4Graph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemma4Graph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4Graph_TiedEmbedding(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            1,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	// Verify no separate lm_head.weight exists (tied to embedding).
	if _, ok := tensors["lm_head.weight"]; ok {
		t.Fatal("Gemma 4 should not have separate lm_head.weight (tied embedding)")
	}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildGemma4Graph_HybridAttention(t *testing.T) {
	// 6 layers with SlidingWindowPattern=6: layers 0-4 are sliding, layer 5 is global.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            6,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}

	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads

	fill := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.02 * float32(i%7+1) * 0.1
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

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	tensors["model.embed_tokens.weight"] = fill([]int{cfg.VocabSize, hidden})
	tensors["model.norm.weight"] = ones([]int{hidden})

	kvDim := headDim * cfg.NumKVHeads
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{cfg.IntermediateSize, hidden})
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{cfg.IntermediateSize, hidden})
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, cfg.IntermediateSize})
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph hybrid: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

// makeGemma4_26BTestTensors creates a minimal set of Gemma 4 MoE architecture
// tensors. Layers at even indices (0, 2, ...) are MoE layers; odd indices are
// dense FFN layers. This mirrors the real Gemma 4 26B-A4B pattern where some
// layers have MoE and others use dense FFN.
func makeGemma4_26BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}
	inter := cfg.IntermediateSize

	fill := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.02 * float32(i%7+1) * 0.1
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

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	tensors["model.embed_tokens.weight"] = fill([]int{cfg.VocabSize, hidden})
	tensors["model.norm.weight"] = ones([]int{hidden})

	kvDim := headDim * cfg.NumKVHeads
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."

		// Attention weights (same for all layers).
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden})
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})

		// Even layers are MoE, odd layers are dense.
		if i%2 == 0 {
			// MoE layer: router + per-expert weights.
			tensors[prefix+"mlp.gate.weight"] = fill([]int{cfg.NumExperts, hidden})
			for j := 0; j < cfg.NumExperts; j++ {
				ep := prefix + "mlp.experts." + itoa(j) + "."
				tensors[ep+"gate_proj.weight"] = fill([]int{inter, hidden})
				tensors[ep+"up_proj.weight"] = fill([]int{inter, hidden})
				tensors[ep+"down_proj.weight"] = fill([]int{hidden, inter})
			}
		} else {
			// Dense FFN layer.
			tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden})
			tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden})
			tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter})
		}
	}

	return tensors
}

func TestBuildGemma4MoEGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4moe",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
		NumExperts:           4,
		NumExpertsPerToken:   2,
	}
	tensors := makeGemma4_26BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemma4MoEGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4MoEGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemma4MoEGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4moe",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
		NumExperts:           4,
		NumExpertsPerToken:   2,
	}
	tensors := makeGemma4_26BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4MoEGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4MoEGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4MoEGraph_MoELayerDetection(t *testing.T) {
	// Verify that layers with router weights are MoE and others are dense.
	// We build with 4 layers: even (0,2) = MoE, odd (1,3) = dense.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4moe",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		NumExperts:           4,
		NumExpertsPerToken:   2,
	}
	tensors := makeGemma4_26BTestTensors(cfg)

	// Verify MoE layers have router weights.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		_, hasRouter := tensors[prefix+"mlp.gate.weight"]
		if i%2 == 0 && !hasRouter {
			t.Errorf("layer %d should be MoE (has router), but router weight not found", i)
		}
		if i%2 != 0 && hasRouter {
			t.Errorf("layer %d should be dense (no router), but router weight found", i)
		}
	}

	// Build succeeds with mixed MoE/dense layers.
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, _, err := buildGemma4MoEGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4MoEGraph mixed layers: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4Graph_KEqV(t *testing.T) {
	// Verify K=V on global layers by setting AttentionKEqV=true.
	// With SlidingWindowPattern=2 and NumLayers=2, layer 1 is global.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            2,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 2,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
	}
	tensors := makeGemma4_31BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4Graph K=V: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

// --- Gemma 4 Edge variant tests (E4B, E2B) ---

// makeGemma4_E4BTestTensors creates tensors for the Gemma 4 E4B edge variant
// using the canonical post-mapping layout (ADR-086):
//   - Shared PLE table, projection, and norm at model.* level.
//   - Per-block input_gate, ple_layer_proj, post_layernorm, layer_output_scale.
func makeGemma4_E4BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeGemma4_31BTestTensors(cfg)

	hidden := cfg.HiddenSize
	pleHidden := cfg.PLEHiddenSize
	totalPLE := cfg.NumLayers * pleHidden

	fill := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.02 * float32(i%7+1) * 0.1
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

	// Shared PLE tensors.
	tensors["model.ple_embed_tokens.weight"] = fill([]int{cfg.VocabSize, totalPLE})
	tensors["model.ple_model_proj.weight"] = fill([]int{totalPLE, hidden})
	tensors["model.ple_proj_norm.weight"] = zeros([]int{pleHidden}) // Gemma uses (1+gain); 0 gain => identity.

	// Remove k_proj/v_proj/k_norm for shared layers (canonical layout skips them).
	firstSharedIdx := cfg.NumLayers - cfg.KVSharedLayers
	if firstSharedIdx < 0 {
		firstSharedIdx = 0
	}
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		if i >= firstSharedIdx {
			delete(tensors, prefix+"self_attn.k_proj.weight")
			delete(tensors, prefix+"self_attn.v_proj.weight")
			delete(tensors, prefix+"self_attn.k_norm.weight")
		}
		// Per-block PLE sub-block tensors.
		// GGUF post-mapping convention stores weights as [out, in]; the loader
		// flips to [in, out] for the framework. Fixtures author [out, in] to
		// match the real GGUF loader path.
		tensors[prefix+"input_gate.weight"] = fill([]int{pleHidden, hidden})
		tensors[prefix+"ple_layer_proj.weight"] = fill([]int{hidden, pleHidden})
		tensors[prefix+"post_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"layer_output_scale.weight"] = ones([]int{1})
	}

	return tensors
}

// makeGemma4_E2BTestTensors creates tensors for the Gemma 4 E2B edge variant
// using the canonical post-mapping layout (ADR-086): canonical PLE at
// model.* level plus per-block input_gate, ple_layer_proj, post_layernorm,
// layer_output_scale. E2B uses double-wide MLP.
func makeGemma4_E2BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}
	pleHidden := cfg.PLEHiddenSize
	totalPLE := cfg.NumLayers * pleHidden
	inter := cfg.IntermediateSize
	if cfg.DoubleWideMLP {
		inter = cfg.IntermediateSize * 2
	}

	fill := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 0.02 * float32(i%7+1) * 0.1
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

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	tensors["model.embed_tokens.weight"] = fill([]int{cfg.VocabSize, hidden})
	tensors["model.norm.weight"] = ones([]int{hidden})

	// Shared PLE tensors (ADR-086).
	tensors["model.ple_embed_tokens.weight"] = fill([]int{cfg.VocabSize, totalPLE})
	tensors["model.ple_model_proj.weight"] = fill([]int{totalPLE, hidden})
	tensors["model.ple_proj_norm.weight"] = zeros([]int{pleHidden})

	firstSharedIdx := cfg.NumLayers - cfg.KVSharedLayers
	if firstSharedIdx < 0 {
		firstSharedIdx = 0
	}

	kvDim := headDim * cfg.NumKVHeads
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."

		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"pre_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"post_feedforward_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})

		// Shared layers skip k_proj/v_proj/k_norm (HF modeling_gemma4.py 1167).
		if i < firstSharedIdx {
			tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden})
			tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden})
			tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
		}

		// Double-wide MLP.
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden})
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden})
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter})

		// PLE sub-block (per-layer).
		tensors[prefix+"input_gate.weight"] = fill([]int{pleHidden, hidden})
		tensors[prefix+"ple_layer_proj.weight"] = fill([]int{hidden, pleHidden})
		tensors[prefix+"post_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"layer_output_scale.weight"] = ones([]int{1})
	}

	return tensors
}

func TestBuildGemma4EdgeGraph_E4B_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4e",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
		PLEHiddenSize:        4,
		KVSharedLayers:       2,
	}
	tensors := makeGemma4_E4BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGemma4EdgeGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4EdgeGraph E4B: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGemma4EdgeGraph_E4B_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4e",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		AttentionKEqV:        true,
		PLEHiddenSize:        4,
		KVSharedLayers:       2,
	}
	tensors := makeGemma4_E4BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4EdgeGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4EdgeGraph E4B: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4EdgeGraph_E2B_DoubleWideMLP(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4e",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		PLEHiddenSize:        4,
		KVSharedLayers:       2,
		DoubleWideMLP:        true,
	}
	tensors := makeGemma4_E2BTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4EdgeGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4EdgeGraph E2B double-wide: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildGemma4EdgeGraph_KVSharing(t *testing.T) {
	// 4 layers with KVSharedLayers=2: layers 0-1 share KV, layers 2-3 share KV.
	// Verify that the shared layers reference the same underlying weight tensors.
	cfg := &gguf.ModelConfig{
		Architecture:         "gemma4e",
		VocabSize:            32,
		HiddenSize:           16,
		NumLayers:            4,
		NumHeads:             4,
		NumKVHeads:           2,
		IntermediateSize:     32,
		MaxSeqLen:            64,
		RopeTheta:            1000000.0,
		LocalRopeTheta:       10000.0,
		SlidingWindowPattern: 6,
		SlidingWindow:        4096,
		LogitSoftcap:         30.0,
		GlobalNumKVHeads:     2,
		GlobalHeadDim:        4,
		SlidingNumKVHeads:    2,
		SlidingHeadDim:       4,
		PLEHiddenSize:        4,
		KVSharedLayers:       2,
	}
	tensors := makeGemma4_E4BTestTensors(cfg)

	// In the canonical layout (ADR-086, ADR-087), the last KVSharedLayers
	// layers do not own k_proj/v_proj/k_norm tensors -- they reuse K/V at
	// runtime via KVReuseNode from the nearest non-shared layer of the same
	// attention type. With NumLayers=4 and KVSharedLayers=2, layers 0..1 are
	// donors and layers 2..3 are shared consumers.
	for i := 0; i < 2; i++ {
		prefix := "model.layers." + itoa(i) + "."
		if tensors[prefix+"self_attn.k_proj.weight"] == nil {
			t.Fatalf("donor layer %d k_proj.weight missing", i)
		}
		if tensors[prefix+"self_attn.v_proj.weight"] == nil {
			t.Fatalf("donor layer %d v_proj.weight missing", i)
		}
	}
	for i := 2; i < 4; i++ {
		prefix := "model.layers." + itoa(i) + "."
		if _, exists := tensors[prefix+"self_attn.k_proj.weight"]; exists {
			t.Fatalf("shared layer %d unexpectedly has k_proj.weight", i)
		}
		if _, exists := tensors[prefix+"self_attn.v_proj.weight"]; exists {
			t.Fatalf("shared layer %d unexpectedly has v_proj.weight", i)
		}
		if _, exists := tensors[prefix+"self_attn.k_norm.weight"]; exists {
			t.Fatalf("shared layer %d unexpectedly has k_norm.weight", i)
		}
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4EdgeGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4EdgeGraph KV sharing: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

// TestPLECombinedProducer_SliceBuffersStable verifies the ADR-088 invariant
// that pleCombinedProducer reuses the same per-layer slice tensor objects
// across Forward calls with the same shape, but refreshes their contents
// when the input tokens change. This property is required for CUDA graph
// capture to work: the captured graph reads from fixed device addresses,
// so the slice tensors (and their underlying storage) must outlive the
// capture and be updated in place on each decode step.
func TestPLECombinedProducer_SliceBuffersStable(t *testing.T) {
	const (
		numLayers = 3
		pleDim    = 4
		hidden    = 8
		vocab     = 16
	)
	totalPLE := numLayers * pleDim

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Fill tensor with deterministic values based on a seed so both Forward
	// calls produce valid outputs.
	fill := func(shape []int, seed float32) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = seed + 0.01*float32(i)
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New %v: %v", shape, err)
		}
		return tn
	}

	pleEmbed := fill([]int{vocab, totalPLE}, 1.0)
	pleModelProj := fill([]int{hidden, totalPLE}, 2.0)

	producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
	if err != nil {
		t.Fatalf("newPLECombinedProducer: %v", err)
	}

	// First pass: tokens 1, 2.
	ids1 := fill([]int{1, 2}, 1.0)
	// Round fill() output to ints in the valid token range.
	{
		d := ids1.Data()
		d[0], d[1] = 1, 2
	}
	hiddenIn := fill([]int{1, 2, hidden}, 0.5)

	if _, err := producer.Forward(context.Background(), ids1, hiddenIn); err != nil {
		t.Fatalf("first Forward: %v", err)
	}
	if producer.tokenPLESlices == nil || len(producer.tokenPLESlices) != numLayers {
		t.Fatalf("first Forward did not initialise tokenPLESlices: %v", producer.tokenPLESlices)
	}
	if producer.modelProjSlices == nil || len(producer.modelProjSlices) != numLayers {
		t.Fatalf("first Forward did not initialise modelProjSlices: %v", producer.modelProjSlices)
	}

	// Capture first-pass state: tensor-pointer identity and a copy of contents
	// so we can verify (1) the same tensor objects are reused next pass, and
	// (2) the contents differ after a second pass with different tokens.
	tokenPtrs := make([]*tensor.TensorNumeric[float32], numLayers)
	projPtrs := make([]*tensor.TensorNumeric[float32], numLayers)
	tokenSnapshot := make([][]float32, numLayers)
	projSnapshot := make([][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		tokenPtrs[i] = producer.tokenPLESlices[i]
		projPtrs[i] = producer.modelProjSlices[i]
		tokenSnapshot[i] = append([]float32(nil), producer.tokenPLESlices[i].Data()...)
		projSnapshot[i] = append([]float32(nil), producer.modelProjSlices[i].Data()...)
	}

	// Second pass: tokens 3, 4 (different from first pass -> different
	// tokenPLE contents). Same shape -> same buffers reused.
	ids2 := fill([]int{1, 2}, 0.0)
	{
		d := ids2.Data()
		d[0], d[1] = 3, 4
	}

	if _, err := producer.Forward(context.Background(), ids2, hiddenIn); err != nil {
		t.Fatalf("second Forward: %v", err)
	}

	// (1) Stable pointers: the CUDA-graph replay invariant.
	for i := 0; i < numLayers; i++ {
		if producer.tokenPLESlices[i] != tokenPtrs[i] {
			t.Fatalf("layer %d: tokenPLESlices tensor pointer changed (was %p, now %p); this breaks CUDA graph replay",
				i, tokenPtrs[i], producer.tokenPLESlices[i])
		}
		if producer.modelProjSlices[i] != projPtrs[i] {
			t.Fatalf("layer %d: modelProjSlices tensor pointer changed (was %p, now %p); this breaks CUDA graph replay",
				i, projPtrs[i], producer.modelProjSlices[i])
		}
	}

	// (2) Contents were refreshed: different tokens must produce different
	// tokenPLE data.
	anyTokenDiffered := false
	for i := 0; i < numLayers; i++ {
		newData := producer.tokenPLESlices[i].Data()
		if len(newData) != len(tokenSnapshot[i]) {
			t.Fatalf("layer %d: tokenPLESlices data length changed (was %d, now %d)",
				i, len(tokenSnapshot[i]), len(newData))
		}
		for j := range newData {
			if newData[j] != tokenSnapshot[i][j] {
				anyTokenDiffered = true
				break
			}
		}
	}
	if !anyTokenDiffered {
		t.Fatalf("second Forward with different token ids did not refresh tokenPLESlices contents")
	}
}

// TestPLECombinedProducer_SliceBuffersReallocateOnShapeChange verifies the
// producer drops its cached slice buffers and allocates fresh ones when the
// batch or seqLen changes (e.g. prefill [1,5] -> decode [1,1]). The stable
// address guarantee only applies within a given shape regime.
func TestPLECombinedProducer_SliceBuffersReallocateOnShapeChange(t *testing.T) {
	const (
		numLayers = 2
		pleDim    = 4
		hidden    = 8
		vocab     = 16
	)
	totalPLE := numLayers * pleDim

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	fill := func(shape []int, seed float32) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = seed + 0.01*float32(i)
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New %v: %v", shape, err)
		}
		return tn
	}

	pleEmbed := fill([]int{vocab, totalPLE}, 1.0)
	pleModelProj := fill([]int{hidden, totalPLE}, 2.0)

	producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
	if err != nil {
		t.Fatalf("newPLECombinedProducer: %v", err)
	}

	// Prefill-shaped pass: seqLen=5.
	idsPrefill := fill([]int{1, 5}, 0.0)
	for i := range idsPrefill.Data() {
		idsPrefill.Data()[i] = float32(i % vocab)
	}
	hiddenPrefill := fill([]int{1, 5, hidden}, 0.5)
	if _, err := producer.Forward(context.Background(), idsPrefill, hiddenPrefill); err != nil {
		t.Fatalf("prefill Forward: %v", err)
	}
	prefillToken0 := producer.tokenPLESlices[0]
	prefillShape := prefillToken0.Shape()
	if len(prefillShape) != 3 || prefillShape[1] != 5 {
		t.Fatalf("prefill slice unexpected shape: %v", prefillShape)
	}

	// Decode-shaped pass: seqLen=1 -> buffers must be reallocated.
	idsDecode := fill([]int{1, 1}, 0.0)
	idsDecode.Data()[0] = 1
	hiddenDecode := fill([]int{1, 1, hidden}, 0.5)
	if _, err := producer.Forward(context.Background(), idsDecode, hiddenDecode); err != nil {
		t.Fatalf("decode Forward: %v", err)
	}
	decodeToken0 := producer.tokenPLESlices[0]
	if decodeToken0 == prefillToken0 {
		t.Fatalf("slice tensor pointer should have been reallocated on shape change but was reused")
	}
	decodeShape := decodeToken0.Shape()
	if len(decodeShape) != 3 || decodeShape[1] != 1 {
		t.Fatalf("decode slice unexpected shape: %v", decodeShape)
	}
}

// TestPLECombinedProducer_DecodeFastPathSharesBackingStorage pins the
// T99.2.1 performance invariant: in the decode regime (seqLen == 1),
// per-layer slice tensors must alias a single shared full-width
// backing buffer, so refreshing them costs one memcpy instead of
// numLayers small memcpys. For CPU storage this means the per-layer
// slices' Slice() views all point into the same underlying []T; for
// GPU storage this is enforced implicitly by the code path that
// constructs NewGPUStorageView sub-slices of p.tokenPLEBuf and
// p.modelProjBuf. This test covers the CPU variant end-to-end; the
// GPU variant is verified by the DGX Spark gemma4 generate
// throughput bench required for T99.2.1 acceptance.
func TestPLECombinedProducer_DecodeFastPathSharesBackingStorage(t *testing.T) {
	const (
		numLayers = 4
		pleDim    = 3
		hidden    = 8
		vocab     = 16
	)
	totalPLE := numLayers * pleDim

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	fill := func(shape []int, seed float32) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = seed + 0.01*float32(i)
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New %v: %v", shape, err)
		}
		return tn
	}

	pleEmbed := fill([]int{vocab, totalPLE}, 1.0)
	pleModelProj := fill([]int{hidden, totalPLE}, 2.0)

	producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
	if err != nil {
		t.Fatalf("newPLECombinedProducer: %v", err)
	}

	// Decode-shaped pass (seqLen = 1): triggers the fast path.
	ids := fill([]int{1, 1}, 0.0)
	ids.Data()[0] = 2
	hiddenIn := fill([]int{1, 1, hidden}, 0.5)
	if _, err := producer.Forward(context.Background(), ids, hiddenIn); err != nil {
		t.Fatalf("first Forward: %v", err)
	}

	// Invariant 1: the per-layer CPU slices must share a single backing
	// []T of length totalPLE. Two different layers with the same
	// underlying array implies sliceLayer0.Data() and sliceLayer1.Data()
	// are contiguous segments within the same array.
	for _, pair := range []struct {
		name   string
		slices []*tensor.TensorNumeric[float32]
	}{
		{"tokenPLESlices", producer.tokenPLESlices},
		{"modelProjSlices", producer.modelProjSlices},
	} {
		if len(pair.slices) != numLayers {
			t.Fatalf("%s len=%d want %d", pair.name, len(pair.slices), numLayers)
		}
		cs0, ok := pair.slices[0].GetStorage().(*tensor.CPUStorage[float32])
		if !ok {
			t.Fatalf("%s[0] storage type %T, want *tensor.CPUStorage[float32]", pair.name, pair.slices[0].GetStorage())
		}
		base0 := cs0.Slice()
		// Sanity: layer 0's data is pleDim elements.
		if len(base0) != pleDim {
			t.Fatalf("%s[0] Slice len=%d want %d", pair.name, len(base0), pleDim)
		}
		// Capacity extends to the full backing array (totalPLE). Using
		// cap to peek at the shared array.
		if cap(base0) < totalPLE {
			t.Fatalf("%s[0] Slice cap=%d want >= %d (per-layer view must share a backing array of length totalPLE)",
				pair.name, cap(base0), totalPLE)
		}
		fullBacking := base0[:totalPLE]
		// Invariant 2: layer k's slice must be &fullBacking[k*pleDim].
		for layer := 1; layer < numLayers; layer++ {
			csK, ok := pair.slices[layer].GetStorage().(*tensor.CPUStorage[float32])
			if !ok {
				t.Fatalf("%s[%d] storage type %T, want *tensor.CPUStorage[float32]", pair.name, layer, csK.Slice())
			}
			baseK := csK.Slice()
			if len(baseK) != pleDim {
				t.Fatalf("%s[%d] Slice len=%d want %d", pair.name, layer, len(baseK), pleDim)
			}
			// Elementwise: mutate a byte in fullBacking at the layer
			// offset and verify it's visible through baseK.
			sentinel := float32(1000.0 + float32(layer))
			fullBacking[layer*pleDim] = sentinel
			if baseK[0] != sentinel {
				t.Fatalf("%s[%d] does not alias fullBacking at offset %d: want %f got %f",
					pair.name, layer, layer*pleDim, sentinel, baseK[0])
			}
		}
	}

	// Invariant 3: a subsequent Forward call with the same shape must
	// not allocate new slice tensor objects, and must refresh the
	// backing array's contents (not allocate a new one). This is the
	// per-step no-alloc property that makes the fast path fast.
	ptrToken := producer.tokenPLESlices[0]
	ptrProj := producer.modelProjSlices[0]
	cs0 := ptrToken.GetStorage().(*tensor.CPUStorage[float32])
	underlyingPre := &cs0.Slice()[:1][0] // pointer to first elem of the backing array

	ids.Data()[0] = 5 // different token so content changes
	if _, err := producer.Forward(context.Background(), ids, hiddenIn); err != nil {
		t.Fatalf("second Forward: %v", err)
	}
	if producer.tokenPLESlices[0] != ptrToken || producer.modelProjSlices[0] != ptrProj {
		t.Fatalf("fast-path decode allocated new slice tensor objects on second call")
	}
	cs0Post := producer.tokenPLESlices[0].GetStorage().(*tensor.CPUStorage[float32])
	underlyingPost := &cs0Post.Slice()[:1][0]
	if underlyingPre != underlyingPost {
		t.Fatalf("fast-path decode reallocated the backing array on second call")
	}
}

// TestPLESliceNode_ZeroAblation validates the T99.2.2 H16 / H19 ablation gates:
// ZERFOO_GEMMA4_PLE_ZERO=1/both zero the entire PLE output; =token zeros only
// the tokenSlice contribution; =proj zeros only the projNormed contribution.
// Unset keeps the normal path.
func TestPLESliceNode_ZeroAblation(t *testing.T) {
	const (
		numLayers = 2
		pleDim    = 4
		hidden    = 8
		vocab     = 16
	)
	totalPLE := numLayers * pleDim
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	fill := func(shape []int, seed float32) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = seed + 0.01*float32(i)
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New %v: %v", shape, err)
		}
		return tn
	}

	pleEmbed := fill([]int{vocab, totalPLE}, 1.0)
	pleModelProj := fill([]int{hidden, totalPLE}, 2.0)
	normGain := fill([]int{pleDim}, 1.0)

	// runForward rebuilds the pleSliceNode after setting the env var so the
	// constructor re-reads ZERFOO_GEMMA4_PLE_ZERO under the current subtest.
	runForward := func(t *testing.T, zeroEnv string) []float32 {
		t.Helper()
		t.Setenv("ZERFOO_GEMMA4_PLE_ZERO", zeroEnv)
		producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
		if err != nil {
			t.Fatalf("newPLECombinedProducer: %v", err)
		}
		ids := fill([]int{1, 1}, 0.0)
		ids.Data()[0] = 3
		hiddenIn := fill([]int{1, 1, hidden}, 0.5)
		if _, err := producer.Forward(context.Background(), ids, hiddenIn); err != nil {
			t.Fatalf("producer Forward: %v", err)
		}
		node, err := newPLESliceNode[float32](engine, ops, producer, normGain, 1e-6, 0)
		if err != nil {
			t.Fatalf("newPLESliceNode: %v", err)
		}
		out, err := node.Forward(context.Background())
		if err != nil {
			t.Fatalf("pleSliceNode Forward: %v", err)
		}
		gotShape := out.Shape()
		wantShape := []int{1, 1, pleDim}
		if len(gotShape) != len(wantShape) {
			t.Fatalf("output rank %d, want %d", len(gotShape), len(wantShape))
		}
		for i := range gotShape {
			if gotShape[i] != wantShape[i] {
				t.Fatalf("output shape %v, want %v", gotShape, wantShape)
			}
		}
		return append([]float32(nil), out.Data()...)
	}

	maxAbs := func(vs []float32) float32 {
		var m float32
		for _, v := range vs {
			if v < 0 {
				v = -v
			}
			if v > m {
				m = v
			}
		}
		return m
	}
	allZero := func(vs []float32) bool { return maxAbs(vs) == 0 }
	differs := func(a, b []float32) bool {
		if len(a) != len(b) {
			return true
		}
		for i := range a {
			d := a[i] - b[i]
			if d < 0 {
				d = -d
			}
			if d > 1e-6 {
				return true
			}
		}
		return false
	}

	// Capture each mode in its own subtest so t.Setenv scoping rebuilds the
	// node against a fresh env var each time.
	var baseline, tokenOnly []float32

	t.Run("unset", func(t *testing.T) {
		baseline = runForward(t, "")
		if allZero(baseline) {
			t.Fatalf("baseline all-zero; fixture cannot discriminate ablations")
		}
	})

	t.Run("one", func(t *testing.T) {
		// Regression guard for the H16 artifact (commit cca5ea3b): "1"
		// must zero the entire PLE output, byte-for-byte identical to "both".
		got := runForward(t, "1")
		if !allZero(got) {
			t.Fatalf("ZERFOO_GEMMA4_PLE_ZERO=1 max|out| = %v, want 0", maxAbs(got))
		}
	})

	t.Run("both", func(t *testing.T) {
		got := runForward(t, "both")
		if !allZero(got) {
			t.Fatalf("ZERFOO_GEMMA4_PLE_ZERO=both max|out| = %v, want 0", maxAbs(got))
		}
	})

	t.Run("token", func(t *testing.T) {
		tokenOnly = runForward(t, "token")
		if allZero(tokenOnly) {
			t.Fatalf("token mode produced all-zero output; expected projNormed contribution to remain")
		}
		if !differs(tokenOnly, baseline) {
			t.Fatalf("token mode output equals baseline; tokenSlice suppression not observable")
		}
	})

	t.Run("proj", func(t *testing.T) {
		projOnly := runForward(t, "proj")
		if allZero(projOnly) {
			t.Fatalf("proj mode produced all-zero output; expected tokenSlice contribution to remain")
		}
		if !differs(projOnly, baseline) {
			t.Fatalf("proj mode output equals baseline; projNormed suppression not observable")
		}
		if !differs(projOnly, tokenOnly) {
			t.Fatalf("proj output equals token output; modes are not independently discriminating")
		}
	})
}

// TestPLESliceNode_TokenNormAblation validates the T99.2.2 H20 fix candidate:
// ZERFOO_GEMMA4_PLE_TOKEN_NORM=1 inserts a per-layer RMSNorm on tokenSlice
// before the Add(projNormed, tokenSlice) combine. The normed path must:
//  1. Still produce a non-zero output (contract: normalization, not zeroing).
//  2. Differ from the baseline (observable behavior change).
//  3. Produce a strictly smaller max-abs output than the raw-token baseline
//     for a fixture whose token-identity contribution dominates, confirming
//     the RMSNorm is bounding the tokenSlice magnitude.
func TestPLESliceNode_TokenNormAblation(t *testing.T) {
	const (
		numLayers = 2
		pleDim    = 4
		hidden    = 8
		vocab     = 16
	)
	totalPLE := numLayers * pleDim
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	fill := func(shape []int, seed float32) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = seed + 0.01*float32(i)
		}
		tn, err := tensor.New[float32](shape, data)
		if err != nil {
			t.Fatalf("tensor.New %v: %v", shape, err)
		}
		return tn
	}

	// Amplify the tokenSlice contribution (seed=10) so its max-abs dominates
	// projNormed (whose magnitude is bounded by the existing RMSNorm). This
	// makes the H20 bound observable: enabling ple_token_norm should cut the
	// output max-abs significantly.
	pleEmbed := fill([]int{vocab, totalPLE}, 10.0)
	pleModelProj := fill([]int{hidden, totalPLE}, 0.1)
	normGain := fill([]int{pleDim}, 1.0)

	runForward := func(t *testing.T, tokenNormEnv string) []float32 {
		t.Helper()
		// Keep the H19 env unset so both runs share the same ablation state.
		t.Setenv("ZERFOO_GEMMA4_PLE_ZERO", "")
		t.Setenv("ZERFOO_GEMMA4_PLE_TOKEN_NORM", tokenNormEnv)
		producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
		if err != nil {
			t.Fatalf("newPLECombinedProducer: %v", err)
		}
		ids := fill([]int{1, 1}, 0.0)
		ids.Data()[0] = 3
		hiddenIn := fill([]int{1, 1, hidden}, 0.5)
		if _, err := producer.Forward(context.Background(), ids, hiddenIn); err != nil {
			t.Fatalf("producer Forward: %v", err)
		}
		node, err := newPLESliceNode[float32](engine, ops, producer, normGain, 1e-6, 0)
		if err != nil {
			t.Fatalf("newPLESliceNode: %v", err)
		}
		out, err := node.Forward(context.Background())
		if err != nil {
			t.Fatalf("pleSliceNode Forward: %v", err)
		}
		return append([]float32(nil), out.Data()...)
	}

	maxAbs := func(vs []float32) float32 {
		var m float32
		for _, v := range vs {
			if v < 0 {
				v = -v
			}
			if v > m {
				m = v
			}
		}
		return m
	}

	var baseline, normed []float32

	t.Run("baseline_unset", func(t *testing.T) {
		baseline = runForward(t, "")
		if maxAbs(baseline) == 0 {
			t.Fatalf("baseline all-zero; fixture cannot discriminate")
		}
	})

	t.Run("token_norm_enabled", func(t *testing.T) {
		normed = runForward(t, "1")
		if maxAbs(normed) == 0 {
			t.Fatalf("token-norm output all-zero; expected bounded non-zero value")
		}
		if len(normed) != len(baseline) {
			t.Fatalf("output length %d != baseline %d", len(normed), len(baseline))
		}
		differs := false
		for i := range normed {
			d := normed[i] - baseline[i]
			if d < 0 {
				d = -d
			}
			if d > 1e-6 {
				differs = true
				break
			}
		}
		if !differs {
			t.Fatalf("token-norm output equals baseline; ple_token_norm had no effect")
		}
		if maxAbs(normed) >= maxAbs(baseline) {
			t.Fatalf("token-norm max|out|=%v, want strictly < baseline %v (RMSNorm should bound tokenSlice magnitude)",
				maxAbs(normed), maxAbs(baseline))
		}
	})

	t.Run("frozen_registered", func(t *testing.T) {
		t.Setenv("ZERFOO_GEMMA4_PLE_ZERO", "")
		t.Setenv("ZERFOO_GEMMA4_PLE_TOKEN_NORM", "1")
		producer, err := newPLECombinedProducer[float32](engine, pleEmbed, pleModelProj, numLayers, pleDim, hidden)
		if err != nil {
			t.Fatalf("newPLECombinedProducer: %v", err)
		}
		node, err := newPLESliceNode[float32](engine, ops, producer, normGain, 1e-6, 0)
		if err != nil {
			t.Fatalf("newPLESliceNode: %v", err)
		}
		frozen := node.EmbeddedFrozen()
		if len(frozen) < 2 {
			t.Fatalf("EmbeddedFrozen returned %d tensors, want >= 2 (proj_norm + token_norm gains)", len(frozen))
		}
		// At least one frozen tensor must be [pleDim]-shaped and all-ones
		// (the freshly initialized ple_token_norm gain).
		foundOnes := false
		for _, f := range frozen {
			s := f.Shape()
			if len(s) != 1 || s[0] != pleDim {
				continue
			}
			d := f.Data()
			allOne := true
			for _, v := range d {
				if v != 1.0 {
					allOne = false
					break
				}
			}
			if allOne {
				foundOnes = true
				break
			}
		}
		if !foundOnes {
			t.Fatalf("EmbeddedFrozen did not include an all-ones [%d] gain tensor for ple_token_norm", pleDim)
		}
	})
}
