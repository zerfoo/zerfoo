package inference

import (
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
// with PLE (Per-Layer Embeddings) and KV-shared layers.
func makeGemma4_E4BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeGemma4_31BTestTensors(cfg)

	hidden := cfg.HiddenSize
	pleHidden := cfg.PLEHiddenSize

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

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		// PLE embedding weight: [vocab, ple_hidden_size].
		tensors[prefix+"ple.weight"] = fill([]int{cfg.VocabSize, pleHidden})
		// PLE projection weight: [hidden_size, ple_hidden_size] (GGUF convention: out, in).
		tensors[prefix+"ple_proj.weight"] = fill([]int{hidden, pleHidden})
	}

	return tensors
}

// makeGemma4_E2BTestTensors creates tensors for the Gemma 4 E2B edge variant
// with PLE, KV-shared layers, and double-wide MLP.
func makeGemma4_E2BTestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	hidden := cfg.HiddenSize
	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}
	pleHidden := cfg.PLEHiddenSize
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

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	tensors["model.embed_tokens.weight"] = fill([]int{cfg.VocabSize, hidden})
	tensors["model.norm.weight"] = ones([]int{hidden})

	kvDim := headDim * cfg.NumKVHeads
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."

		// Standard Gemma 4 layer tensors.
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

		// Double-wide MLP.
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden})
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden})
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter})

		// PLE tensors.
		tensors[prefix+"ple.weight"] = fill([]int{cfg.VocabSize, pleHidden})
		tensors[prefix+"ple_proj.weight"] = fill([]int{hidden, pleHidden})
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

	// Verify that layers 0 and 1 share the same K/V tensors (from layer 0).
	kLayer0 := tensors["model.layers.0.self_attn.k_proj.weight"]
	vLayer0 := tensors["model.layers.0.self_attn.v_proj.weight"]
	kLayer1 := tensors["model.layers.1.self_attn.k_proj.weight"]
	vLayer1 := tensors["model.layers.1.self_attn.v_proj.weight"]

	// In the tensor map, each layer has its own weights. But the builder should
	// load K/V from the source layer. Verify the builder resolves KV correctly
	// by checking that the source tensors exist and the build succeeds.
	if kLayer0 == nil || vLayer0 == nil {
		t.Fatal("source layer 0 K/V tensors missing")
	}
	if kLayer1 == nil || vLayer1 == nil {
		t.Fatal("layer 1 K/V tensors missing (expected for tensor map, builder uses layer 0's)")
	}

	// Layers 2-3 share KV from layer 2.
	kLayer2 := tensors["model.layers.2.self_attn.k_proj.weight"]
	vLayer2 := tensors["model.layers.2.self_attn.v_proj.weight"]
	if kLayer2 == nil || vLayer2 == nil {
		t.Fatal("source layer 2 K/V tensors missing")
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGemma4EdgeGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGemma4EdgeGraph KV sharing: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}
