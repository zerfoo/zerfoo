package speculative

import (
	"context"
	"math"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// mockDraftNode is a graph node that ignores input and returns logits where
// a specific token has the highest value. It cycles through a token sequence.
// The output includes a simple linear + softmax-like pattern: the target
// token gets logit +10.0, all others get -10.0.
type mockDraftNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	mu            sync.Mutex
	callCount     int
}

func (n *mockDraftNode) OpType() string                     { return "MockDraft" }
func (n *mockDraftNode) Attributes() map[string]interface{} { return nil }
func (n *mockDraftNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *mockDraftNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *mockDraftNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	n.mu.Lock()
	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}
	n.mu.Unlock()

	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

func buildMockGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &mockDraftNode{
		vocabSize:     vocabSize,
		tokenSequence: tokenSequence,
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestExternalDraft(t *testing.T) {
	t.Run("returns K tokens with valid log probs", func(t *testing.T) {
		vocabSize := 8
		// Mock draft model cycles through tokens: 4, 5, 6
		draftGraph := buildMockGraph(t, vocabSize, []int{4, 5, 6})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		cfg := generate.ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			NumLayers:  1,
		}

		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		K := 3
		tokens, logProbs, err := ed.Generate(context.Background(), []int32{1}, K)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		if len(tokens) != K {
			t.Fatalf("got %d tokens, want %d", len(tokens), K)
		}
		if len(logProbs) != K {
			t.Fatalf("got %d log probs, want %d", len(logProbs), K)
		}

		// Verify tokens match mock sequence.
		expectedTokens := []int32{4, 5, 6}
		for i, tok := range tokens {
			if tok != expectedTokens[i] {
				t.Errorf("tokens[%d] = %d, want %d", i, tok, expectedTokens[i])
			}
		}

		// Verify log probs are valid (negative, finite).
		for i, lp := range logProbs {
			if lp > 0 {
				t.Errorf("logProbs[%d] = %f, want <= 0", i, lp)
			}
			if math.IsNaN(float64(lp)) || math.IsInf(float64(lp), 0) {
				t.Errorf("logProbs[%d] = %f, want finite", i, lp)
			}
		}
	})

	t.Run("stops at EOS", func(t *testing.T) {
		vocabSize := 8
		eosToken := 2
		// Mock produces: 4, EOS — should stop after 2 tokens.
		draftGraph := buildMockGraph(t, vocabSize, []int{4, eosToken})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		cfg := generate.ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: eosToken,
			NumLayers:  1,
		}

		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		tokens, logProbs, err := ed.Generate(context.Background(), []int32{1}, 5)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		if len(tokens) != 2 {
			t.Fatalf("got %d tokens, want 2 (should stop at EOS)", len(tokens))
		}
		if tokens[1] != int32(eosToken) {
			t.Errorf("tokens[1] = %d, want EOS=%d", tokens[1], eosToken)
		}
		if len(logProbs) != len(tokens) {
			t.Errorf("logProbs length %d != tokens length %d", len(logProbs), len(tokens))
		}
	})

	t.Run("consistent with mock model distribution", func(t *testing.T) {
		vocabSize := 8
		// Single repeating token to verify consistency.
		draftGraph := buildMockGraph(t, vocabSize, []int{3})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})

		cfg := generate.ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  128,
			EOSTokenID: 2,
			NumLayers:  1,
		}

		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		tokens, logProbs, err := ed.Generate(context.Background(), []int32{1}, 4)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}

		// All tokens should be 3 (the mock always produces 3).
		for i, tok := range tokens {
			if tok != 3 {
				t.Errorf("tokens[%d] = %d, want 3", i, tok)
			}
		}

		// All log probs should be the same since the distribution is identical
		// at each step (logit 10.0 for token 3, -10.0 for others).
		for i := 1; i < len(logProbs); i++ {
			if logProbs[i] != logProbs[0] {
				t.Errorf("logProbs[%d] = %f, want %f (same distribution each step)",
					i, logProbs[i], logProbs[0])
			}
		}

		// The log prob should be close to 0 (the max token dominates).
		// With logit 10 vs -10 for 7 others: log(1 / (1 + 7*exp(-20)))
		// which is approximately -7*exp(-20) ~ 0.
		if logProbs[0] < -0.001 {
			t.Errorf("logProbs[0] = %f, expected close to 0 for dominant logit", logProbs[0])
		}
	})

	t.Run("errors on invalid K", func(t *testing.T) {
		vocabSize := 8
		draftGraph := buildMockGraph(t, vocabSize, []int{4})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		cfg := generate.ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, NumLayers: 1}
		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		_, _, err := ed.Generate(context.Background(), []int32{1}, 0)
		if err == nil {
			t.Error("expected error for K=0")
		}
	})

	t.Run("errors on empty tokens", func(t *testing.T) {
		vocabSize := 8
		draftGraph := buildMockGraph(t, vocabSize, []int{4})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		cfg := generate.ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, NumLayers: 1}
		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		_, _, err := ed.Generate(context.Background(), nil, 3)
		if err == nil {
			t.Error("expected error for empty tokens")
		}
	})

	t.Run("shares engine with target", func(t *testing.T) {
		vocabSize := 8
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		draftGraph := buildMockGraph(t, vocabSize, []int{4, 5})

		cfg := generate.ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, NumLayers: 1}
		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		// The external draft's engine should be the same instance as passed in.
		if ed.engine != engine {
			t.Error("ExternalDraft should share the provided engine instance")
		}
	})

	t.Run("context cancellation stops generation", func(t *testing.T) {
		vocabSize := 8
		// Never produces EOS, generates token 4 forever.
		draftGraph := buildMockGraph(t, vocabSize, []int{4})
		engine := compute.NewCPUEngine(numeric.Float32Ops{})
		cfg := generate.ModelConfig{VocabSize: vocabSize, MaxSeqLen: 128, EOSTokenID: 2, NumLayers: 1}
		ed := NewExternalDraft[float32](draftGraph, engine, nil, cfg)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // cancel immediately

		tokens, logProbs, err := ed.Generate(ctx, []int32{1}, 100)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		// Should produce at most 1 token (the prefill sample) since context
		// is already cancelled when the loop starts.
		if len(tokens) > 1 {
			t.Errorf("got %d tokens, expected at most 1 with cancelled context", len(tokens))
		}
		if len(logProbs) != len(tokens) {
			t.Errorf("logProbs length %d != tokens length %d", len(logProbs), len(tokens))
		}
	})
}
