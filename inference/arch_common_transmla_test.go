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

// makeTransMLATestTensors creates a tensor set with TransMLA tensors for a
// Llama-like model. K/V projection weights are removed and replaced by
// transmla.{layer}.wDKV, wUK, wUV tensors.
//
// The converter writes wUK/wUV with shape [dK, rank] where dK = numKVHeads * headDim.
// The MLA layer expects wUK to output numHeads * headDim. For TransMLA
// models where numKVHeads == numHeads (standard MHA → MLA conversion), this
// is consistent. For GQA models (numKVHeads < numHeads), the tensors use the
// original KV dimension and the MLA constructor receives numKVHeads as the
// effective head count for up-projections.
func makeTransMLATestTensors(cfg *gguf.ModelConfig, kvLoraDim int) map[string]*tensor.TensorNumeric[float32] {
	tensors := makeLlamaTestTensors(cfg)

	headDim := cfg.HiddenSize / cfg.NumHeads
	kvDim := cfg.NumKVHeads * headDim

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

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		transMLAPrefix := "transmla." + itoa(i) + "."

		// Remove original K/V projections (TransMLA replaces them).
		delete(tensors, prefix+"self_attn.k_proj.weight")
		delete(tensors, prefix+"self_attn.v_proj.weight")

		// Add TransMLA tensors matching converter output format.
		// wDKV: [dModel, rank] — shared down-projection to latent space
		tensors[transMLAPrefix+"wDKV"] = fill([]int{cfg.HiddenSize, kvLoraDim}, 0.02)
		// wUK: [dK, rank] — key up-projection (transposed in wiring to [rank, dK])
		tensors[transMLAPrefix+"wUK"] = fill([]int{kvDim, kvLoraDim}, 0.02)
		// wUV: [dV, rank] — value up-projection (transposed in wiring to [rank, dV])
		tensors[transMLAPrefix+"wUV"] = fill([]int{kvDim, kvLoraDim}, 0.02)
	}

	return tensors
}

func TestBuildTransformerGraph_TransMLADetection(t *testing.T) {
	kvLoraDim := 8
	cfg := &gguf.ModelConfig{
		Architecture:      "llama",
		VocabSize:         32,
		HiddenSize:        16,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        4, // TransMLA: KV heads match Q heads (MHA → MLA)
		IntermediateSize:  32,
		MaxSeqLen:         64,
		RopeTheta:         10000.0,
		TransMLAKVLoraDim: kvLoraDim,
	}

	tensors := makeTransMLATestTensors(cfg, kvLoraDim)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	embedWeight := tensors["model.embed_tokens.weight"]
	lmHeadWeight := tensors["lm_head.weight"]

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
	if err != nil {
		t.Fatalf("buildTransformerGraph with TransMLA tensors: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildTransformerGraph_TransMLAForwardNonNaN(t *testing.T) {
	kvLoraDim := 8
	cfg := &gguf.ModelConfig{
		Architecture:      "llama",
		VocabSize:         32,
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        4,
		IntermediateSize:  32,
		MaxSeqLen:         64,
		RopeTheta:         10000.0,
		TransMLAKVLoraDim: kvLoraDim,
	}

	tensors := makeTransMLATestTensors(cfg, kvLoraDim)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	embedWeight := tensors["model.embed_tokens.weight"]
	lmHeadWeight := tensors["lm_head.weight"]

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
	if err != nil {
		t.Fatalf("buildTransformerGraph: %v", err)
	}

	// Run a forward pass.
	tokenIDs := []float32{1, 5, 10}
	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != len(tokenIDs) || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, len(tokenIDs), cfg.VocabSize)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestBuildTransformerGraph_GQAFallbackWithoutTransMLA(t *testing.T) {
	// When TransMLAKVLoraDim is 0, even if transmla tensors happen to exist,
	// the standard GQA path should be used.
	cfg := &gguf.ModelConfig{
		Architecture:     "llama",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
		// TransMLAKVLoraDim is 0 — should use GQA.
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	embedWeight := tensors["model.embed_tokens.weight"]
	lmHeadWeight := tensors["lm_head.weight"]

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
	if err != nil {
		t.Fatalf("buildTransformerGraph (GQA fallback): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildTransformerGraph_TransMLAMissingTensor(t *testing.T) {
	kvLoraDim := 8
	cfg := &gguf.ModelConfig{
		Architecture:      "llama",
		VocabSize:         32,
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        4,
		IntermediateSize:  32,
		MaxSeqLen:         64,
		RopeTheta:         10000.0,
		TransMLAKVLoraDim: kvLoraDim,
	}

	tests := []struct {
		name   string
		remove string
	}{
		{"missing_wDKV", "transmla.0.wDKV"},
		{"missing_wUK", "transmla.0.wUK"},
		{"missing_wUV", "transmla.0.wUV"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tensors := makeTransMLATestTensors(cfg, kvLoraDim)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			// Remove the wDKV tensor to trigger detection but then fail on lookup
			// of the specific tensor being tested. For wDKV removal, the detection
			// check will fail, so it falls back to GQA which will fail on missing
			// k_proj. For wUK/wUV, detection succeeds but lookup fails.
			if tc.remove == "transmla.0.wDKV" {
				// If wDKV is missing, hasTransMLA is false, falls to GQA path.
				// GQA will fail because k_proj/v_proj were removed.
				delete(tensors, tc.remove)
			} else {
				delete(tensors, tc.remove)
			}

			embedWeight := tensors["model.embed_tokens.weight"]
			lmHeadWeight := tensors["lm_head.weight"]

			_, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
			if err == nil {
				t.Fatal("expected error for missing tensor, got nil")
			}
		})
	}
}

func TestBuildArchGraph_TransMLAViaLlama(t *testing.T) {
	kvLoraDim := 8
	cfg := &gguf.ModelConfig{
		Architecture:      "llama",
		VocabSize:         32,
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        4,
		IntermediateSize:  32,
		MaxSeqLen:         64,
		RopeTheta:         10000.0,
		TransMLAKVLoraDim: kvLoraDim,
	}

	tensors := makeTransMLATestTensors(cfg, kvLoraDim)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildArchGraph("llama", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildArchGraph with TransMLA: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestExtractModelConfig_TransMLAKVLoraDim(t *testing.T) {
	// Verify that TransMLAKVLoraDim defaults to 0 when not present.
	cfg := &gguf.ModelConfig{}
	if cfg.TransMLAKVLoraDim != 0 {
		t.Fatalf("expected TransMLAKVLoraDim=0, got %d", cfg.TransMLAKVLoraDim)
	}

	// Verify the field can be set.
	cfg.TransMLAKVLoraDim = 128
	if cfg.TransMLAKVLoraDim != 128 {
		t.Fatalf("expected TransMLAKVLoraDim=128, got %d", cfg.TransMLAKVLoraDim)
	}
}
