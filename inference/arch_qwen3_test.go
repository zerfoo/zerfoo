package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// qwen3TestConfig returns a small Qwen 3 config whose head dimension is
// deliberately decoupled from HiddenSize/NumHeads (8 rather than 16/4 = 4),
// mirroring the real Qwen3-0.6B GGUF where hiddenSize is 1024, NumHeads is 16
// and headDim is 128 rather than 64.
func qwen3TestConfig(numLayers int) *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "qwen3",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        numLayers,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
}

// makeQwen3TestTensors creates a minimal set of Qwen 3-architecture tensors
// with canonical names for testing. Unlike Qwen 2 there are no attention bias
// tensors, and every layer carries per-head Q/K RMSNorm weights.
func makeQwen3TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize

	headDim := hidden / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}
	qDim := headDim * cfg.NumHeads
	kvDim := headDim * cfg.NumKVHeads

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
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{qDim, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, qDim}, 0.02)
		// Qwen 3 QK RMSNorm weights, one scale per head dimension.
		tensors[prefix+"self_attn.q_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"self_attn.k_norm.weight"] = ones([]int{headDim})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func TestBuildQwen3Graph_Builds(t *testing.T) {
	cfg := qwen3TestConfig(2)
	tensors := makeQwen3TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildQwen3Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwen3Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildQwen3Graph_ForwardNonNaN(t *testing.T) {
	cfg := qwen3TestConfig(2)
	tensors := makeQwen3TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildQwen3Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwen3Graph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildQwen3Graph_TiedEmbeddings(t *testing.T) {
	cfg := qwen3TestConfig(1)
	tensors := makeQwen3TestTensors(cfg)
	// The real Qwen3-0.6B GGUF has no separate output projection.
	delete(tensors, "lm_head.weight")
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildQwen3Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwen3Graph with tied embeddings: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

// TestBuildQwen3Graph_RequiresQKNorm asserts that Qwen 3 genuinely depends on
// the QK norm tensors: dropping them must fail the build rather than silently
// producing a Qwen 2-shaped graph.
func TestBuildQwen3Graph_RequiresQKNorm(t *testing.T) {
	for _, name := range []string{"q_norm", "k_norm"} {
		t.Run(name, func(t *testing.T) {
			cfg := qwen3TestConfig(1)
			tensors := makeQwen3TestTensors(cfg)
			delete(tensors, "model.layers.0.self_attn."+name+".weight")
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			if _, _, err := buildQwen3Graph(tensors, cfg, engine); err == nil {
				t.Fatalf("expected error when %s.weight is missing", name)
			}
		})
	}
}

// TestBuildQwen3Graph_QKNormAffectsOutput asserts the QK norm weights are
// actually applied. A build that looked them up and ignored them would compile
// and produce identical logits for different norm scales.
func TestBuildQwen3Graph_QKNormAffectsOutput(t *testing.T) {
	cfg := qwen3TestConfig(1)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	headDim := cfg.HeadDim

	tensorsUnit := makeQwen3TestTensors(cfg)
	tensorsScaled := makeQwen3TestTensors(cfg)

	scaledData := make([]float32, headDim)
	for i := range scaledData {
		scaledData[i] = 2.5
	}
	scaled, _ := tensor.New([]int{headDim}, scaledData)
	tensorsScaled["model.layers.0.self_attn.q_norm.weight"] = scaled

	gUnit, _, err := buildQwen3Graph(tensorsUnit, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwen3Graph (unit q_norm): %v", err)
	}
	gScaled, _, err := buildQwen3Graph(tensorsScaled, cfg, engine)
	if err != nil {
		t.Fatalf("buildQwen3Graph (scaled q_norm): %v", err)
	}

	tokenIDs := []float32{1, 5, 10, 3}
	input, _ := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)

	outUnit, err := gUnit.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (unit q_norm): %v", err)
	}
	outScaled, err := gScaled.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward (scaled q_norm): %v", err)
	}

	dataU := outUnit.Data()
	dataS := outScaled.Data()
	if len(dataU) != len(dataS) {
		t.Fatalf("output lengths differ: %d vs %d", len(dataU), len(dataS))
	}

	for i := range dataU {
		if math.Abs(float64(dataU[i]-dataS[i])) > 1e-6 {
			return // outputs differ as expected
		}
	}
	t.Error("expected different logits when q_norm weights change, but outputs are identical")
}

// TestBuildQwen3Graph_NoAttentionBias asserts Qwen 3 builds cleanly with no
// bias tensors present at all -- the real GGUF contains none.
func TestBuildQwen3Graph_NoAttentionBias(t *testing.T) {
	cfg := qwen3TestConfig(1)
	tensors := makeQwen3TestTensors(cfg)

	for name := range tensors {
		if len(name) > 5 && name[len(name)-5:] == ".bias" {
			t.Fatalf("Qwen 3 test fixture unexpectedly contains a bias tensor: %s", name)
		}
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	if _, _, err := buildQwen3Graph(tensors, cfg, engine); err != nil {
		t.Fatalf("buildQwen3Graph without bias tensors: %v", err)
	}
}

func TestBuildQwen3Graph_MissingEmbedding(t *testing.T) {
	cfg := qwen3TestConfig(1)
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildQwen3Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildArchGraph_Qwen3(t *testing.T) {
	cfg := qwen3TestConfig(1)
	tensors := makeQwen3TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("qwen3", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph qwen3: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

// TestBuildArchGraph_Qwen3MoEUnsupported pins the deliberate scope boundary:
// the Qwen 3 mixture-of-experts GGUFs declare architecture "qwen3moe" and are
// not implemented, so they must still fail cleanly rather than being built as
// a dense Qwen 3 model.
func TestBuildArchGraph_Qwen3MoEUnsupported(t *testing.T) {
	cfg := qwen3TestConfig(1)
	cfg.Architecture = "qwen3moe"
	tensors := makeQwen3TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildArchGraph("qwen3moe", tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected qwen3moe to be unsupported")
	}
}
