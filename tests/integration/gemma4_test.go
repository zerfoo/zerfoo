package integration

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/inference"
)

// TestGemma4E2B_EndToEnd loads a real Gemma 4 E2B GGUF and builds its graph,
// verifying every tensor name from the canonical layout resolves and the
// architecture router picks the correct sub-variant. When GEMMA4_RUN_FORWARD=1
// also runs a forward pass and asserts finite logits -- that portion must run
// on a GPU host (Spark on DGX) because CPU forward for a 2B-param model
// exceeds reasonable test timeouts (> 5 min). See docs/bench/manifests/
// gemma4-e2e.yaml for the Spark job manifest and scripts/gemma4-spark.sh
// for the submit wrapper. Skips entirely when GEMMA4_GGUF_PATH is unset.
func TestGemma4E2B_EndToEnd(t *testing.T) {
	path := os.Getenv("GEMMA4_GGUF_PATH")
	if path == "" {
		t.Skip("GEMMA4_GGUF_PATH not set; skipping Gemma 4 integration test")
	}

	mdl, err := inference.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	cfg := mdl.Config

	// Sub-variant routing: unsloth/Google GGUFs declare arch=gemma4 for all
	// variants. ExtractModelConfig should rewrite to gemma4e when PLE or
	// KV-sharing metadata is present.
	switch cfg.Architecture {
	case "gemma4", "gemma4e", "gemma4moe":
		// ok
	default:
		t.Fatalf("Architecture = %q, want gemma4/gemma4e/gemma4moe", cfg.Architecture)
	}

	if cfg.NumLayers == 0 || cfg.HiddenSize == 0 || cfg.VocabSize == 0 {
		t.Fatalf("config incomplete: layers=%d hidden=%d vocab=%d", cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, _, err := inference.BuildArchGraph(cfg.Architecture, mdl.Tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildArchGraph(%q): %v", cfg.Architecture, err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}

	t.Logf("Gemma 4 %s graph built: %d layers, hidden=%d, vocab=%d, tensors=%d",
		cfg.Architecture, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize, len(mdl.Tensors))

	if os.Getenv("GEMMA4_RUN_FORWARD") != "1" {
		t.Skip("forward pass skipped (set GEMMA4_RUN_FORWARD=1 on a GPU host; CPU 2B-param forward > 5min)")
	}

	tokenIDs := []float32{1, 2, 3, 4}
	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward pass: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != len(tokenIDs) || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape %v, want [1, %d, %d]", shape, len(tokenIDs), cfg.VocabSize)
	}

	data := output.Data()
	var hasNonZero bool
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at logit index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at logit index %d", i)
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		t.Fatal("all logits are zero -- graph not wired correctly")
	}

	t.Logf("Forward pass logits finite, shape=%v", shape)
}
