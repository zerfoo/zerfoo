package generate

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	tokenizer "github.com/zerfoo/ztoken"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestNewGenerator(t *testing.T) {
	t.Run("creates generator with valid config", func(t *testing.T) {
		cfg := ModelConfig{
			VocabSize:  32000,
			MaxSeqLen:  2048,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  12,
		}
		g := NewGenerator[float32](nil, nil, nil, cfg)
		if g == nil {
			t.Fatal("expected non-nil generator")
		}
		if g.config.VocabSize != 32000 {
			t.Errorf("VocabSize = %d, want 32000", g.config.VocabSize)
		}
		if g.config.NumLayers != 12 {
			t.Errorf("NumLayers = %d, want 12", g.config.NumLayers)
		}
	})
}

func TestSamplingConfigDefaults(t *testing.T) {
	cfg := DefaultSamplingConfig()
	if cfg.Temperature != 1.0 {
		t.Errorf("Temperature = %f, want 1.0", cfg.Temperature)
	}
	if cfg.TopK != 0 {
		t.Errorf("TopK = %d, want 0", cfg.TopK)
	}
	if cfg.TopP != 1.0 {
		t.Errorf("TopP = %f, want 1.0", cfg.TopP)
	}
	if cfg.RepetitionPenalty != 1.0 {
		t.Errorf("RepetitionPenalty = %f, want 1.0", cfg.RepetitionPenalty)
	}
	if cfg.MaxNewTokens != 256 {
		t.Errorf("MaxNewTokens = %d, want 256", cfg.MaxNewTokens)
	}
}

func TestModelConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  ModelConfig
	}{
		{
			name: "gemma3",
			cfg: ModelConfig{
				VocabSize:  256000,
				MaxSeqLen:  8192,
				EOSTokenID: 1,
				BOSTokenID: 2,
				NumLayers:  26,
			},
		},
		{
			name: "small",
			cfg: ModelConfig{
				VocabSize:  100,
				MaxSeqLen:  128,
				EOSTokenID: 0,
				BOSTokenID: 1,
				NumLayers:  2,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenerator[float32](nil, nil, nil, tt.cfg)
			if g.config != tt.cfg {
				t.Errorf("config mismatch: got %+v, want %+v", g.config, tt.cfg)
			}
		})
	}
}

// fixedLogitsNode is a graph node that ignores input and always returns
// logits where a specific token index has the highest value.
// The output shape is [1, inputSeqLen, vocabSize].
type fixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize int
	// tokenSequence is the sequence of token IDs to produce on each call.
	// Wraps around if more calls than entries.
	tokenSequence []int
	mu            sync.Mutex
	callCount     int
}

func (n *fixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *fixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *fixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *fixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *fixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	n.mu.Lock()
	callCount := n.callCount
	data := make([]float32, seqLen*n.vocabSize)
	// For each position, set the target token to have the highest logit.
	for pos := range seqLen {
		targetToken := n.tokenSequence[callCount%len(n.tokenSequence)]
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

// buildTestGraph creates a simple graph with a fixedLogitsNode for testing.
func buildTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)

	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
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

// buildTestTokenizer creates a simple tokenizer with known vocabulary.
// Tokens: <unk>=0, <s>=1, </s>=2, <pad>=3, hello=4, world=5, foo=6, bar=7
func buildTestTokenizer() *tokenizer.WhitespaceTokenizer {
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7
	return tok
}

// buildErrorGraph creates a graph that always errors on Forward.
func buildErrorGraph(t *testing.T) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &errorNode{}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// buildBadLogitsGraph creates a graph that returns valid 3D logits on first call, 2D on second.
func buildBadLogitsGraph(t *testing.T) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &badLogitsNode{}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// buildErrorAfterPrefillGraph creates a graph that succeeds for prefill but errors on decode.
func buildErrorAfterPrefillGraph(t *testing.T, vocabSize int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &errorAfterPrefillNode{vocabSize: vocabSize}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestGenerate_Greedy(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// EOS = 2 (</s>). Generate tokens 6, 7, then EOS.
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0, // greedy
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Should produce "foo bar" (tokens 6, 7), then stop at EOS.
	if result != "foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar")
	}
}

func TestGenerate_MaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Never produce EOS -- always produce token 6.
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo foo foo" {
		t.Errorf("Generate = %q, want %q", result, "foo foo foo")
	}
}

func TestGenerate_StopTokenID(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce 6, 7, 6 (custom stop at 7 should stop after first 7).
	g := buildTestGraph(t, vocabSize, []int{6, 7, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopTokenIDs: []int{7}, // Stop when token 7 (bar) is generated.
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Should produce "foo" (token 6), then stop when 7 is sampled (not included).
	if result != "foo" {
		t.Errorf("Generate = %q, want %q", result, "foo")
	}
}

func TestGenerate_StopString(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce 6, 7, 6, 6 endlessly.
	g := buildTestGraph(t, vocabSize, []int{6, 7, 6, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Should produce "foo " then stop before "bar".
	if result != "foo " {
		t.Errorf("Generate = %q, want %q", result, "foo ")
	}
}

func TestGenerate_EmptyPrompt(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	_, err := gen.Generate(context.Background(), "", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error for empty prompt")
	}
}

func TestGenerate_ImmediateEOS(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// First token is EOS.
	g := buildTestGraph(t, vocabSize, []int{2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "" {
		t.Errorf("Generate = %q, want empty string", result)
	}
}

func TestGenerate_ContextCancellation(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	result, err := gen.Generate(ctx, "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 100,
	})
	// Should stop early -- either error or short result.
	// The prefill may succeed before context check, but the loop should stop.
	if err != nil {
		return // Error is acceptable for canceled context.
	}
	// If no error, result should be short (at most 1 token from prefill).
	tokens, _ := tok.Encode(result)
	if len(tokens) > 2 {
		t.Errorf("expected short result with canceled context, got %d tokens: %q", len(tokens), result)
	}
}

func TestGenerate_WithTemperature(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// With a dominant logit, temperature sampling should still pick the right token.
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0.5, // Low temperature sharpens distribution.
		MaxNewTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// With logit 10.0 vs -10.0 and temperature 0.5, token 6 has ~probability 1.
	if result != "foo" {
		t.Errorf("Generate = %q, want %q", result, "foo")
	}
}

func TestIdsToTensor(t *testing.T) {
	gen := &Generator[float32]{}
	got, err := gen.idsToTensor([]int{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	shape := got.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 3 {
		t.Errorf("shape = %v, want [1, 3]", shape)
	}
	data := got.Data()
	if data[0] != 1.0 || data[1] != 2.0 || data[2] != 3.0 {
		t.Errorf("data = %v, want [1, 2, 3]", data)
	}
}

func TestSampleFromLogits_InvalidShape(t *testing.T) {
	gen := &Generator[float32]{}

	// 2D tensor instead of 3D.
	logits2D, err := tensor.New([]int{1, 8}, make([]float32, 8))
	if err != nil {
		t.Fatal(err)
	}
	_, err = gen.sampleFromLogits(logits2D, SamplingConfig{Temperature: 0}, nil)
	if err == nil {
		t.Error("expected error for 2D logits")
	}
}

func TestGenerate_WithTopK(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  1.0,
		TopK:         2,
		MaxNewTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// With dominant logit (10.0 vs -10.0), top-k should still pick the right token.
	if result != "foo" {
		t.Errorf("Generate = %q, want %q", result, "foo")
	}
}

func TestGenerate_WithTopP(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  1.0,
		TopP:         0.9,
		MaxNewTokens: 5,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo" {
		t.Errorf("Generate = %q, want %q", result, "foo")
	}
}

func TestGenerate_WithRepetitionPenalty(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produces token 6 repeatedly, then EOS.
	g := buildTestGraph(t, vocabSize, []int{6, 6, 6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	// With a very dominant logit, even rep penalty won't change the result.
	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:       0.5,
		RepetitionPenalty: 1.5,
		MaxNewTokens:      10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result == "" {
		t.Error("expected non-empty result with repetition penalty")
	}
}

func TestGenerate_DefaultMaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	// MaxNewTokens=0 should use default (256).
	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 0,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if result != "foo" {
		t.Errorf("Generate = %q, want %q", result, "foo")
	}
}

func TestGenerate_StopStringFirstToken(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce bar (7) immediately.
	g := buildTestGraph(t, vocabSize, []int{7, 6, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// "bar" found at position 0, so result should be empty.
	if result != "" {
		t.Errorf("Generate = %q, want empty", result)
	}
}

// errorNode returns an error from Forward, simulating a graph failure.
type errorNode struct {
	graph.NoParameters[float32]
}

func (n *errorNode) OpType() string                     { return "Error" }
func (n *errorNode) Attributes() map[string]interface{} { return nil }
func (n *errorNode) OutputShape() []int                 { return []int{1, 1, 1} }
func (n *errorNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *errorNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return nil, fmt.Errorf("simulated forward error")
}

func TestGenerate_GraphForwardError(t *testing.T) {
	tok := buildTestTokenizer()
	g := buildErrorGraph(t)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: 8, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error from graph forward failure")
	}
}

// badLogitsNode returns a 2D tensor (wrong shape) to trigger sampleFromLogits error.
type badLogitsNode struct {
	graph.NoParameters[float32]
	first bool
}

func (n *badLogitsNode) OpType() string                     { return "BadLogits" }
func (n *badLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *badLogitsNode) OutputShape() []int                 { return []int{1, 8} }
func (n *badLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *badLogitsNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if !n.first {
		n.first = true
		// Return valid 3D logits with token 6 dominant for the prefill.
		data := make([]float32, 8)
		for i := range data {
			data[i] = -10.0
		}
		data[6] = 10.0
		return tensor.New([]int{1, 1, 8}, data)
	}
	// On second call, return 2D tensor to trigger error.
	return tensor.New([]int{1, 8}, make([]float32, 8))
}

func TestGenerate_BadLogitsFromDecodeStep(t *testing.T) {
	tok := buildTestTokenizer()
	g := buildBadLogitsGraph(t)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: 8, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error from bad logits shape in decode step")
	}
}

// errorAfterPrefillNode returns valid logits for prefill but errors on decode.
type errorAfterPrefillNode struct {
	graph.NoParameters[float32]
	callCount int
	vocabSize int
}

func (n *errorAfterPrefillNode) OpType() string                     { return "ErrorAfterPrefill" }
func (n *errorAfterPrefillNode) Attributes() map[string]interface{} { return nil }
func (n *errorAfterPrefillNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *errorAfterPrefillNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *errorAfterPrefillNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.callCount++
	if n.callCount > 1 {
		return nil, fmt.Errorf("simulated decode step error")
	}
	// Prefill: return valid logits with token 6 dominant.
	data := make([]float32, n.vocabSize)
	for i := range data {
		data[i] = -10.0
	}
	data[6] = 10.0
	return tensor.New([]int{1, 1, n.vocabSize}, data)
}

func TestGenerate_DecodeStepForwardError(t *testing.T) {
	tok := buildTestTokenizer()
	g := buildErrorAfterPrefillGraph(t, 8)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: 8, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error from decode step forward failure")
	}
}

func TestGenerate_AllSamplingFeatures(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	// Use all features at once.
	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:       0.8,
		TopK:              3,
		TopP:              0.95,
		RepetitionPenalty: 1.2,
		MaxNewTokens:      10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar")
	}
}

func TestCheckStop_NoStopStrings(t *testing.T) {
	gen := &Generator[float32]{
		tokenizer: buildTestTokenizer(),
	}
	var rd string
	var dc int
	stopped, _ := gen.checkStop([]int{6, 7}, nil, &rd, &dc)
	if stopped {
		t.Error("should not stop with no stop strings")
	}
}

func TestCheckStop_NoMatch(t *testing.T) {
	gen := &Generator[float32]{
		tokenizer: buildTestTokenizer(),
	}
	var rd string
	var dc int
	stopped, _ := gen.checkStop([]int{6, 7}, []string{"xyz"}, &rd, &dc)
	if stopped {
		t.Error("should not stop when no match found")
	}
}

// errorTokenizer is a tokenizer that returns errors for testing.
type errorTokenizer struct{}

func (e *errorTokenizer) Encode(_ string) ([]int, error) {
	return nil, fmt.Errorf("encode error")
}
func (e *errorTokenizer) Decode(_ []int) (string, error) {
	return "", fmt.Errorf("decode error")
}
func (e *errorTokenizer) VocabSize() int                            { return 0 }
func (e *errorTokenizer) GetToken(_ int) (string, bool)             { return "", false }
func (e *errorTokenizer) GetID(_ string) (int, bool)                { return 0, false }
func (e *errorTokenizer) SpecialTokens() tokenizer.SpecialTokens    { return tokenizer.SpecialTokens{} }

func TestGenerate_EncodeError(t *testing.T) {
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, &errorTokenizer{},
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error from tokenizer encode failure")
	}
}

func TestCheckStop_DecodeError(t *testing.T) {
	gen := &Generator[float32]{
		tokenizer: &errorTokenizer{},
	}
	var rd string
	var dc int
	stopped, _ := gen.checkStop([]int{1, 2}, []string{"test"}, &rd, &dc)
	if stopped {
		t.Error("should not stop when decode fails")
	}
}

// decodeErrorTokenizer encodes fine but returns error on Decode.
type decodeErrorTokenizer struct {
	inner tokenizer.Tokenizer
}

func (d *decodeErrorTokenizer) Encode(text string) ([]int, error) { return d.inner.Encode(text) }
func (d *decodeErrorTokenizer) Decode(_ []int) (string, error)    { return "", fmt.Errorf("decode error") }
func (d *decodeErrorTokenizer) VocabSize() int                    { return d.inner.VocabSize() }
func (d *decodeErrorTokenizer) GetToken(id int) (string, bool)    { return d.inner.GetToken(id) }
func (d *decodeErrorTokenizer) GetID(token string) (int, bool)    { return d.inner.GetID(token) }
func (d *decodeErrorTokenizer) SpecialTokens() tokenizer.SpecialTokens {
	return d.inner.SpecialTokens()
}

func TestGenerate_DecodeOutputError(t *testing.T) {
	tok := buildTestTokenizer()
	deTok := &decodeErrorTokenizer{inner: tok}

	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{5, 2})

	gen := NewGenerator[float32](
		g, deTok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	})
	if err == nil {
		t.Error("expected error from decode output failure")
	}
}

func TestGenerate_WithPagedKV(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Generate tokens 6, 7, then EOS (same as TestGenerate_Greedy).
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			BOSTokenID: 1,
			NumLayers:  2, // Must be > 0 for paged KV
		},
		WithPagedKV(1, 64), // 1 MB pool, headDim=64
	)

	if gen.blockPool == nil {
		t.Fatal("expected blockPool to be set with WithPagedKV")
	}

	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar")
	}
}

func TestGenerate_WithPagedKV_MaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6}) // always produce token 6

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  1,
		},
		WithPagedKV(1, 32),
	)

	result, err := gen.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo foo foo" {
		t.Errorf("Generate = %q, want %q", result, "foo foo foo")
	}
}

func TestNewGenerator_PoolWiring(t *testing.T) {
	t.Run("nil graph skips pool", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, ModelConfig{
			NumLayers: 2,
			MaxSeqLen: 32,
		})
		if gen.pool != nil {
			t.Error("expected nil pool when graph is nil")
		}
	})

	t.Run("non-nil graph creates pool", func(t *testing.T) {
		g := buildTestGraph(t, 8, []int{6, 2})
		gen := NewGenerator[float32](g, nil, nil, ModelConfig{
			NumLayers: 2,
			MaxSeqLen: 32,
		})
		if gen.pool == nil {
			t.Error("expected non-nil pool when graph is provided")
		}
	})
}

func TestGenerate_WithPoolCorrectness(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Generate 5 tokens then EOS -- enough decode steps to verify pool doesn't break output.
	g := buildTestGraph(t, vocabSize, []int{6, 7, 6, 7, 6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	if gen.pool == nil {
		t.Fatal("expected pool to be wired when graph is non-nil")
	}

	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Correctness: output should be unchanged by pool.
	if result != "foo bar foo bar foo" {
		t.Errorf("Generate = %q, want %q", result, "foo bar foo bar foo")
	}
}

func BenchmarkGenerate_Allocs(b *testing.B) {
	tok := buildBenchTokenizer(b)
	vocabSize := 8
	eng := compute.NewCPUEngine(numeric.Float32Ops{})

	// Build a graph that produces 3 tokens then EOS per Generate call.
	makeGraph := func(b *testing.B) *graph.Graph[float32] {
		b.Helper()
		builder := graph.NewBuilder[float32](eng)
		in := builder.Input([]int{1, 1, 1})
		node := &fixedLogitsNode{
			vocabSize:     vocabSize,
			tokenSequence: []int{6, 7, 6, 2},
		}
		builder.AddNode(node, in)
		g, err := builder.Build(node)
		if err != nil {
			b.Fatal(err)
		}
		return g
	}

	b.Run("with_pool", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			g := makeGraph(b)
			gen := NewGenerator[float32](g, tok, eng, ModelConfig{
				VocabSize:  vocabSize,
				MaxSeqLen:  32,
				EOSTokenID: 2,
				NumLayers:  0,
			})
			_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: 10,
			})
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("without_pool", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			g := makeGraph(b)
			gen := NewGenerator[float32](g, tok, eng, ModelConfig{
				VocabSize:  vocabSize,
				MaxSeqLen:  32,
				EOSTokenID: 2,
				NumLayers:  0,
			})
			// Disable pool to compare.
			gen.pool = nil
			g.WithPool(nil)
			_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: 10,
			})
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// buildBenchTokenizer creates a tokenizer for benchmarks (same as buildTestTokenizer).
func buildBenchTokenizer(b *testing.B) *tokenizer.WhitespaceTokenizer {
	b.Helper()
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")
	tok.AddToken("foo")
	tok.AddToken("bar")
	return tok
}

// cacheTrackingNode produces fixed logits AND tracks how many times the KV
// cache is updated. This lets us verify that compilation does not inject
// spurious cache updates.
type cacheTrackingNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	mu            sync.Mutex
	callCount     int
	cacheUpdates  int // number of times cache.Update was called through this node
}

func (n *cacheTrackingNode) OpType() string                     { return "CacheTracking" }
func (n *cacheTrackingNode) Attributes() map[string]interface{} { return nil }
func (n *cacheTrackingNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *cacheTrackingNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *cacheTrackingNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	// If context carries a KV cache, update it (simulating attention layer behavior).
	if cache, ok := GetCache[float32](ctx); ok {
		// Create dummy K/V tensors: [1, seqLen, 4] (4 = headDim placeholder).
		kv := make([]float32, seqLen*4)
		kvT, _ := tensor.New([]int{1, seqLen, 4}, kv)
		_ = cache.Update(0, kvT, kvT)
		n.mu.Lock()
		n.cacheUpdates++
		n.mu.Unlock()
	}

	n.mu.Lock()
	callCount := n.callCount
	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[callCount%len(n.tokenSequence)]
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

func TestCompileGraph_DoesNotCorruptKVCache(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	eng := compute.NewCPUEngine(numeric.Float32Ops{})

	// Create a graph with a cache-tracking node.
	node := &cacheTrackingNode{
		vocabSize:     vocabSize,
		tokenSequence: []int{6, 7, 6, 7, 2}, // produce a few tokens then EOS
	}
	b := graph.NewBuilder[float32](eng)
	in := b.Input([]int{1, 1, 1})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := NewGenerator[float32](
		g, tok, eng,
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  64,
			EOSTokenID: 2,
			NumLayers:  1, // 1 layer for KV cache
		},
	)

	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	if result != "foo bar foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar foo bar")
	}

	// Verify KV cache updates: prefill (1) + decode tokens (4, until EOS).
	// Without the fix, compilation would add extra cache updates via
	// CompileTraced or validation Run, corrupting the cache.
	// The compile context should not carry a KV cache, so no extra updates.
	node.mu.Lock()
	updates := node.cacheUpdates
	node.mu.Unlock()

	// Expected: 1 prefill + 4 decode steps (tokens 6,7,6,7) = 5 updates.
	// The 5th token (EOS=2) is sampled but we break before running Forward again.
	if updates != 5 {
		t.Errorf("cache updates = %d, want 5 (compilation should not add extra updates)", updates)
	}
}

// streamProviderEngine wraps CPUEngine and implements compute.StreamProvider
// to test CUDA graph wiring in compileGraph.
type streamProviderEngine struct {
	*compute.CPUEngine[float32]
	streamPtr unsafe.Pointer
}

func (e *streamProviderEngine) Stream() unsafe.Pointer { return e.streamPtr }

func TestCompileGraph_WiresCUDAGraphExecutor(t *testing.T) {
	tests := []struct {
		name      string
		streamPtr unsafe.Pointer
	}{
		{
			name:      "nil stream skips graph wiring",
			streamPtr: nil,
		},
	}

	// On CUDA-capable machines, also test with a real stream pointer.
	// We use a heap-allocated byte so checkptr is satisfied.
	if cuda.Available() && cuda.Lib().GraphAvailable() {
		sentinel := new(byte)
		tests = append(tests, struct {
			name      string
			streamPtr unsafe.Pointer
		}{
			name:      "non-nil stream wires graph executor",
			streamPtr: unsafe.Pointer(sentinel),
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tok := buildTestTokenizer()
			vocabSize := 8
			cpuEng := compute.NewCPUEngine(numeric.Float32Ops{})

			eng := &streamProviderEngine{
				CPUEngine: cpuEng,
				streamPtr: tt.streamPtr,
			}

			// Verify type assertion works.
			if _, ok := any(eng).(compute.StreamProvider); !ok {
				t.Fatal("streamProviderEngine should implement StreamProvider")
			}

			g := buildTestGraph(t, vocabSize, []int{6, 7, 6, 7, 2})

			gen := NewGenerator[float32](
				g, tok, eng,
				ModelConfig{
					VocabSize:  vocabSize,
					MaxSeqLen:  32,
					EOSTokenID: 2,
					NumLayers:  1,
				},
			)

			result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: 10,
			})
			if err != nil {
				t.Fatalf("Generate error: %v", err)
			}
			if result != "foo bar foo bar" {
				t.Errorf("Generate = %q, want %q", result, "foo bar foo bar")
			}

			plan := gen.plan.Load()
			if plan == nil {
				t.Fatal("expected plan to be stored after compilation")
			}
		})
	}
}

// inputTrackingNode wraps fixedLogitsNode and records the input tensor pointer
// and data value for each Forward call. This lets us verify that the decode
// loop reuses the same tensor object across steps.
type inputTrackingNode struct {
	fixedLogitsNode
	mu         sync.Mutex
	inputPtrs  []*tensor.TensorNumeric[float32]
	inputVals  []float32
}

func (n *inputTrackingNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		n.mu.Lock()
		n.inputPtrs = append(n.inputPtrs, inputs[0])
		// Record the value only for single-token inputs (decode steps).
		if len(shape) >= 2 && shape[1] == 1 {
			n.inputVals = append(n.inputVals, inputs[0].Data()[0])
		}
		n.mu.Unlock()
	}
	return n.fixedLogitsNode.Forward(ctx, inputs...)
}

func TestGenerate_TokenTensorReuse(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	eng := compute.NewCPUEngine(numeric.Float32Ops{})

	node := &inputTrackingNode{
		fixedLogitsNode: fixedLogitsNode{
			vocabSize:     vocabSize,
			tokenSequence: []int{6, 7, 6, 7, 6, 2}, // 5 decode steps then EOS
		},
	}
	b := graph.NewBuilder[float32](eng)
	in := b.Input([]int{1, 1, 1})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := NewGenerator[float32](
		g, tok, eng,
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	result, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Verify output correctness is unchanged with tensor reuse.
	if result != "foo bar foo bar foo" {
		t.Errorf("Generate = %q, want %q", result, "foo bar foo bar foo")
	}

	node.mu.Lock()
	defer node.mu.Unlock()

	// We expect 1 prefill call + 5 decode calls = 6 total Forward calls.
	if len(node.inputPtrs) != 6 {
		t.Fatalf("Forward call count = %d, want 6", len(node.inputPtrs))
	}

	// All decode calls (indices 1..5) should receive the same tensor pointer,
	// confirming a single allocation is reused across decode steps.
	decodeTensor := node.inputPtrs[1]
	for i := 2; i < len(node.inputPtrs); i++ {
		if node.inputPtrs[i] != decodeTensor {
			t.Errorf("decode step %d used different tensor pointer (not reused)", i)
		}
	}

	// Verify the input values were correctly updated in-place.
	// Expected decode tokens: 6 (from prefill sample), then 7, 6, 7, 6.
	// The first decode uses the initial value; subsequent ones update in-place.
	wantVals := []float32{6, 7, 6, 7, 6}
	if len(node.inputVals) != len(wantVals) {
		t.Fatalf("decode input values count = %d, want %d", len(node.inputVals), len(wantVals))
	}
	for i, want := range wantVals {
		if node.inputVals[i] != want {
			t.Errorf("decode step %d input value = %v, want %v", i, node.inputVals[i], want)
		}
	}
}

func TestNewGenerator_WithPagedKV_InvalidParams(t *testing.T) {
	// Zero headDim should fall back to regular KV cache.
	gen := NewGenerator[float32](nil, nil, nil, ModelConfig{
		NumLayers: 2,
		MaxSeqLen: 32,
	}, WithPagedKV(1, 0))

	if gen.blockPool != nil {
		t.Error("expected nil blockPool when headDim=0")
	}
}

// TestGenerate_ConcurrentSafety verifies that concurrent Generate calls on the
// same Generator do not race. Without the mu serialization lock, this would
// trigger data races on graph.ResetStatefulNodes() and graph.Forward() state.
func TestGenerate_ConcurrentSafety(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{
			VocabSize:  vocabSize,
			MaxSeqLen:  32,
			EOSTokenID: 2,
			NumLayers:  0,
		},
	)

	const goroutines = 4
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make([]error, goroutines)

	for i := range goroutines {
		go func(idx int) {
			defer wg.Done()
			_, err := gen.Generate(context.Background(), "hello", SamplingConfig{
				Temperature:  0,
				MaxNewTokens: 5,
			})
			errs[idx] = err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d: %v", i, err)
		}
	}
}

func TestCheckStop_Incremental(t *testing.T) {
	gen := &Generator[float32]{
		tokenizer: buildTestTokenizer(),
	}
	var rd string
	var dc int

	// Step 1: add token 6 ("foo"). No stop string match.
	stopped, _ := gen.checkStop([]int{6}, []string{"bar"}, &rd, &dc)
	if stopped {
		t.Fatal("unexpected stop after first token")
	}
	if rd != "foo" {
		t.Fatalf("running decoded = %q, want %q", rd, "foo")
	}

	// Step 2: add token 7 ("bar"). Should match stop string.
	stopped, text := gen.checkStop([]int{6, 7}, []string{"bar"}, &rd, &dc)
	if !stopped {
		t.Fatal("expected stop after second token")
	}
	// "foo bar" with stop at "bar" → text before stop is "foo ".
	if text != "foo " {
		t.Fatalf("text = %q, want %q", text, "foo ")
	}
}

func BenchmarkCheckStop_4096Tokens(b *testing.B) {
	tok := buildTestTokenizer()
	gen := &Generator[float32]{tokenizer: tok}

	// Build a 4096-token sequence cycling through tokens 4-7.
	ids := make([]int, 4096)
	for i := range ids {
		ids[i] = 4 + (i % 4)
	}
	stopStrings := []string{"zzz_never_match"}

	b.ResetTimer()
	for range b.N {
		var rd string
		var dc int
		// Simulate incremental generation: call checkStop after each token.
		for step := 1; step <= len(ids); step++ {
			gen.checkStop(ids[:step], stopStrings, &rd, &dc)
		}
	}
}

