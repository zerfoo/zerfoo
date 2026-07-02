package inference

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBuildEAGLEHead(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	t.Run("valid config", func(t *testing.T) {
		head, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
			NumDraftTokens: 3,
			HiddenDim:      64,
		})
		if err != nil {
			t.Fatalf("BuildEAGLEHead returned error: %v", err)
		}
		if head == nil {
			t.Fatal("BuildEAGLEHead returned nil head")
		}
	})

	t.Run("zero hidden dim", func(t *testing.T) {
		_, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
			NumDraftTokens: 3,
			HiddenDim:      0,
		})
		if err == nil {
			t.Fatal("expected error for zero HiddenDim")
		}
	})

	t.Run("negative draft tokens", func(t *testing.T) {
		_, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
			NumDraftTokens: -1,
			HiddenDim:      64,
		})
		if err == nil {
			t.Fatal("expected error for negative NumDraftTokens")
		}
	})
}

func TestGenerateDraftTokens(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	const hiddenDim = 32
	const vocabSize = 16
	const numDrafts = 4

	head, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
		NumDraftTokens: numDrafts,
		HiddenDim:      hiddenDim,
	})
	if err != nil {
		t.Fatalf("BuildEAGLEHead: %v", err)
	}

	// Create penultimate features [1, 1, hidden].
	featData := make([]float32, hiddenDim)
	for i := range featData {
		featData[i] = float32(i%5-2) * 0.1
	}
	features, err := tensor.New[float32]([]int{1, 1, hiddenDim}, featData)
	if err != nil {
		t.Fatalf("create features: %v", err)
	}

	// Create LM head weight [vocab, hidden].
	lmData := make([]float32, vocabSize*hiddenDim)
	for i := range lmData {
		lmData[i] = float32(i%11-5) * 0.01
	}
	lmWeight, err := tensor.New[float32]([]int{vocabSize, hiddenDim}, lmData)
	if err != nil {
		t.Fatalf("create lm weight: %v", err)
	}

	tokens, err := GenerateDraftTokens(ctx, head, engine, features, lmWeight, numDrafts)
	if err != nil {
		t.Fatalf("GenerateDraftTokens: %v", err)
	}

	if len(tokens) != numDrafts {
		t.Fatalf("expected %d draft tokens, got %d", numDrafts, len(tokens))
	}

	for i, tok := range tokens {
		if tok < 0 || tok >= vocabSize {
			t.Errorf("draft token %d = %d, want 0 <= tok < %d", i, tok, vocabSize)
		}
	}
}

func TestGenerateDraftTokens_SingleDraft(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	const hiddenDim = 16
	const vocabSize = 8

	head, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
		NumDraftTokens: 1,
		HiddenDim:      hiddenDim,
	})
	if err != nil {
		t.Fatalf("BuildEAGLEHead: %v", err)
	}

	featData := make([]float32, hiddenDim)
	for i := range featData {
		featData[i] = 0.5
	}
	features, err := tensor.New[float32]([]int{1, 1, hiddenDim}, featData)
	if err != nil {
		t.Fatalf("create features: %v", err)
	}

	lmData := make([]float32, vocabSize*hiddenDim)
	for i := range lmData {
		lmData[i] = float32(i%7-3) * 0.02
	}
	lmWeight, err := tensor.New[float32]([]int{vocabSize, hiddenDim}, lmData)
	if err != nil {
		t.Fatalf("create lm weight: %v", err)
	}

	tokens, err := GenerateDraftTokens(ctx, head, engine, features, lmWeight, 1)
	if err != nil {
		t.Fatalf("GenerateDraftTokens: %v", err)
	}

	if len(tokens) != 1 {
		t.Fatalf("expected 1 draft token, got %d", len(tokens))
	}
	if tokens[0] < 0 || tokens[0] >= vocabSize {
		t.Errorf("draft token = %d, want 0 <= tok < %d", tokens[0], vocabSize)
	}
}

func TestGenerateDraftTokens_InvalidInputs(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}

	head, err := BuildEAGLEHead(engine, ops, EAGLEConfig{
		NumDraftTokens: 2,
		HiddenDim:      16,
	})
	if err != nil {
		t.Fatalf("BuildEAGLEHead: %v", err)
	}

	features, _ := tensor.New[float32]([]int{1, 1, 16}, make([]float32, 16))
	lmWeight, _ := tensor.New[float32]([]int{8, 16}, make([]float32, 128))

	t.Run("zero drafts", func(t *testing.T) {
		_, err := GenerateDraftTokens(ctx, head, engine, features, lmWeight, 0)
		if err == nil {
			t.Fatal("expected error for zero numDrafts")
		}
	})

	t.Run("2D features rejected", func(t *testing.T) {
		feat2D, _ := tensor.New[float32]([]int{1, 16}, make([]float32, 16))
		_, err := GenerateDraftTokens(ctx, head, engine, feat2D, lmWeight, 2)
		if err == nil {
			t.Fatal("expected error for 2D features")
		}
	})
}

func TestArgmaxLastPos(t *testing.T) {
	// Create a [1, 1, 4] tensor where index 2 has the max value.
	data := []float32{-1.0, 0.5, 3.0, 2.0}
	logits, err := tensor.New[float32]([]int{1, 1, 4}, data)
	if err != nil {
		t.Fatalf("create logits: %v", err)
	}

	got := argmaxLastPos(logits)
	if got != 2 {
		t.Errorf("argmaxLastPos = %d, want 2", got)
	}
}

func TestEAGLEConfig(t *testing.T) {
	cfg := EAGLEConfig{NumDraftTokens: 5, HiddenDim: 128}
	if cfg.NumDraftTokens != 5 {
		t.Errorf("NumDraftTokens = %d, want 5", cfg.NumDraftTokens)
	}
	if cfg.HiddenDim != 128 {
		t.Errorf("HiddenDim = %d, want 128", cfg.HiddenDim)
	}
}

// makeEAGLETensors creates a GGUF-style tensor map with EAGLE head weights
// using the given prefix (e.g. "eagle." or "").
func makeEAGLETensors(prefix string, hidden int) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	gamma, _ := tensor.New[float32]([]int{hidden}, make([]float32, hidden))
	beta, _ := tensor.New[float32]([]int{hidden}, make([]float32, hidden))
	fc1, _ := tensor.New[float32]([]int{hidden, hidden}, make([]float32, hidden*hidden))
	fc2, _ := tensor.New[float32]([]int{hidden, hidden}, make([]float32, hidden*hidden))

	// Set gamma to ones for a working LayerNorm.
	for i := range gamma.Data() {
		gamma.Data()[i] = 1.0
	}

	tensors[prefix+"norm.weight"] = gamma
	tensors[prefix+"norm.bias"] = beta
	tensors[prefix+"fc1.weight"] = fc1
	tensors[prefix+"fc2.weight"] = fc2
	return tensors
}

func TestLoadEAGLEWeights(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	ops := &numeric.Float32Ops{}
	const hidden = 32

	t.Run("prefixed eagle tensors", func(t *testing.T) {
		tensors := makeEAGLETensors("eagle.", hidden)
		head, err := LoadEAGLEWeights(tensors, engine, ops)
		if err != nil {
			t.Fatalf("LoadEAGLEWeights: %v", err)
		}
		if head == nil {
			t.Fatal("expected non-nil head")
		}
		params := head.Parameters()
		if len(params) != 4 {
			t.Fatalf("expected 4 parameters, got %d", len(params))
		}
	})

	t.Run("unprefixed eagle tensors", func(t *testing.T) {
		tensors := makeEAGLETensors("", hidden)
		head, err := LoadEAGLEWeights(tensors, engine, ops)
		if err != nil {
			t.Fatalf("LoadEAGLEWeights: %v", err)
		}
		if head == nil {
			t.Fatal("expected non-nil head")
		}
	})

	t.Run("no eagle tensors", func(t *testing.T) {
		tensors := map[string]*tensor.TensorNumeric[float32]{}
		_, err := LoadEAGLEWeights(tensors, engine, ops)
		if err == nil {
			t.Fatal("expected error for empty tensor map")
		}
	})

	t.Run("missing fc2 weight", func(t *testing.T) {
		tensors := makeEAGLETensors("eagle.", hidden)
		delete(tensors, "eagle.fc2.weight")
		_, err := LoadEAGLEWeights(tensors, engine, ops)
		if err == nil {
			t.Fatal("expected error for missing fc2.weight")
		}
	})

	t.Run("shape validation", func(t *testing.T) {
		tensors := makeEAGLETensors("eagle.", hidden)
		// Replace norm.weight with a 2D tensor to trigger shape validation.
		bad, _ := tensor.New[float32]([]int{1, hidden}, make([]float32, hidden))
		tensors["eagle.norm.weight"] = bad
		_, err := LoadEAGLEWeights(tensors, engine, ops)
		if err == nil {
			t.Fatal("expected error for 2D norm.weight")
		}
	})

	t.Run("forward after load", func(t *testing.T) {
		ctx := context.Background()
		tensors := makeEAGLETensors("eagle.", hidden)
		head, err := LoadEAGLEWeights(tensors, engine, ops)
		if err != nil {
			t.Fatalf("LoadEAGLEWeights: %v", err)
		}
		input, _ := tensor.New[float32]([]int{1, 1, hidden}, make([]float32, hidden))
		out, err := head.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		shape := out.Shape()
		if shape[0] != 1 || shape[1] != 1 || shape[2] != hidden {
			t.Errorf("output shape = %v, want [1, 1, %d]", shape, hidden)
		}
	})
}

func TestHasEAGLEWeights(t *testing.T) {
	const hidden = 16

	t.Run("prefixed tensors", func(t *testing.T) {
		tensors := makeEAGLETensors("eagle.", hidden)
		if !HasEAGLEWeights(tensors) {
			t.Error("expected HasEAGLEWeights=true for prefixed tensors")
		}
	})

	t.Run("unprefixed tensors", func(t *testing.T) {
		tensors := makeEAGLETensors("", hidden)
		if !HasEAGLEWeights(tensors) {
			t.Error("expected HasEAGLEWeights=true for unprefixed tensors")
		}
	})

	t.Run("empty map", func(t *testing.T) {
		if HasEAGLEWeights(map[string]*tensor.TensorNumeric[float32]{}) {
			t.Error("expected HasEAGLEWeights=false for empty map")
		}
	})
}
