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
