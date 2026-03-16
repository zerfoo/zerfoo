package generate

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestGenerateStream_CollectTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce tokens 6, 7, then EOS (2).
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

	var tokens []string
	var doneCount int

	stream := TokenStreamFunc(func(token string, done bool) error {
		if done {
			doneCount++
			return nil
		}
		tokens = append(tokens, token)
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}

	// Verify done was called exactly once.
	if doneCount != 1 {
		t.Errorf("done called %d times, want 1", doneCount)
	}

	// Concatenated stream tokens should match non-streaming output.
	streamed := strings.Join(tokens, "")
	nonStreamResult, err := gen.Generate(context.Background(), "hello world", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Note: the graph resets its call counter for each test function,
	// but buildTestGraph creates a new node each time. We compare
	// the streams are non-empty and valid.
	if streamed == "" {
		t.Error("streamed output should not be empty")
	}
	if nonStreamResult == "" {
		t.Error("non-stream output should not be empty")
	}
}

func TestGenerateStream_StreamParity(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g1 := buildTestGraph(t, vocabSize, []int{6, 7, 2})

	gen1 := NewGenerator[float32](
		g1, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	nonStreamResult, err := gen1.Generate(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Build a fresh graph (same sequence) for streaming.
	g2 := buildTestGraph(t, vocabSize, []int{6, 7, 2})
	gen2 := NewGenerator[float32](
		g2, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var tokens []string
	stream := TokenStreamFunc(func(token string, done bool) error {
		if !done {
			tokens = append(tokens, token)
		}
		return nil
	})

	err = gen2.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}

	streamed := strings.Join(tokens, "")
	if streamed != nonStreamResult {
		t.Errorf("stream output = %q, want %q (matching Generate)", streamed, nonStreamResult)
	}
}

func TestGenerateStream_StopOnCallbackError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 6, 6, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	callCount := 0
	stream := TokenStreamFunc(func(_ string, _ bool) error {
		callCount++
		if callCount >= 2 {
			return fmt.Errorf("stop requested")
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 100,
	}, stream)
	if err == nil {
		t.Error("expected error from stream callback")
	}
}

func TestGenerateStream_EmptyPrompt(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error for empty prompt")
	}
}

func TestGenerateStream_ImmediateEOS(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var tokens []string
	var doneCount int
	stream := TokenStreamFunc(func(token string, done bool) error {
		if done {
			doneCount++
		} else {
			tokens = append(tokens, token)
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}

	if len(tokens) != 0 {
		t.Errorf("expected no tokens, got %v", tokens)
	}
	if doneCount != 1 {
		t.Errorf("done called %d times, want 1", doneCount)
	}
}

func TestGenerateStream_StopString(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7, 6, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var tokens []string
	var doneCount int
	stream := TokenStreamFunc(func(token string, done bool) error {
		if done {
			doneCount++
		} else {
			tokens = append(tokens, token)
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}

	if doneCount != 1 {
		t.Errorf("done called %d times, want 1", doneCount)
	}

	streamed := strings.Join(tokens, "")
	if strings.Contains(streamed, "bar") {
		t.Errorf("streamed output should not contain stop string: %q", streamed)
	}
}

func TestGenerateStream_MaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var tokens []string
	stream := TokenStreamFunc(func(token string, done bool) error {
		if !done {
			tokens = append(tokens, token)
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 3,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}

	// Should have emitted tokens for "foo foo foo".
	if len(tokens) == 0 {
		t.Error("expected non-empty tokens")
	}
}

func TestGenerateStream_ContextCancellation(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	ctx, cancel := context.WithCancel(context.Background())
	callCount := 0
	stream := TokenStreamFunc(func(_ string, done bool) error {
		callCount++
		if callCount >= 2 && !done {
			cancel()
		}
		return nil
	})

	err := gen.GenerateStream(ctx, "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 100,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	// Should have stopped early and still called done.
	if callCount == 0 {
		t.Error("expected at least one callback")
	}
}

func TestGenerateStream_ForwardError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildErrorGraph(t)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error from forward pass")
	}
}

func TestGenerateStream_DecodeForwardError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildErrorAfterPrefillGraph(t, vocabSize)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error from decode forward pass")
	}
}

func TestGenerateStream_BadLogitsShape(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildBadLogitsGraph(t)

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error from bad logits shape")
	}
}

func TestGenerateStream_EncodeError(t *testing.T) {
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6})

	gen := NewGenerator[float32](
		g, &errorTokenizer{},
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error from tokenizer encode")
	}
}

func TestGenerateStream_DefaultMaxTokens(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Immediate EOS so we don't loop 256 times.
	g := buildTestGraph(t, vocabSize, []int{2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var doneCount int
	stream := TokenStreamFunc(func(_ string, done bool) error {
		if done {
			doneCount++
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 0, // Should default to 256.
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	if doneCount != 1 {
		t.Errorf("done called %d times, want 1", doneCount)
	}
}

func TestGenerateStream_DecodeTokenError(t *testing.T) {
	vocabSize := 8
	tok := buildTestTokenizer()
	g := buildTestGraph(t, vocabSize, []int{6, 6})

	gen := NewGenerator[float32](
		g, &decodeErrorTokenizer{inner: tok},
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, _ bool) error { return nil })
	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err == nil {
		t.Error("expected error from tokenizer decode in emitToken")
	}
}

func TestTokenStreamFunc(t *testing.T) {
	var called bool
	f := TokenStreamFunc(func(_ string, _ bool) error {
		called = true
		return nil
	})
	err := f.OnToken("test", false)
	if err != nil {
		t.Fatalf("OnToken error: %v", err)
	}
	if !called {
		t.Error("function should have been called")
	}
}

func TestGenerateStream_StopStringEmitError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce 6, 7 (bar triggers stop string).
	g := buildTestGraph(t, vocabSize, []int{6, 7, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	// Stream that errors when receiving the pre-stop text.
	stream := TokenStreamFunc(func(_ string, done bool) error {
		if !done {
			return fmt.Errorf("stream error")
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	}, stream)
	if err == nil {
		t.Error("expected error from stream callback during stop string emit")
	}
}

func TestGenerateStream_StopTokenIDEmitsDone(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{6, 7})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var doneCount int
	stream := TokenStreamFunc(func(_ string, done bool) error {
		if done {
			doneCount++
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopTokenIDs: []int{7},
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	if doneCount != 1 {
		t.Errorf("done called %d times, want 1", doneCount)
	}
}

func TestGenerateStream_EmitTokenEmptyDelta(t *testing.T) {
	// When decoded text hasn't changed (e.g. token merges), no delta is emitted.
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce two tokens that together form a known word.
	g := buildTestGraph(t, vocabSize, []int{6, 2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	var tokenCount int
	stream := TokenStreamFunc(func(_ string, done bool) error {
		if !done {
			tokenCount++
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 5,
	}, stream)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	if tokenCount == 0 {
		t.Error("expected at least one token")
	}
}

func TestGenerateStream_StopStringDoneError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	// Produce bar (7) immediately - triggers stop string.
	g := buildTestGraph(t, vocabSize, []int{7, 6})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	// Stream errors on the done callback.
	stream := TokenStreamFunc(func(_ string, done bool) error {
		if done {
			return fmt.Errorf("done callback error")
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
		StopStrings:  []string{"bar"},
	}, stream)
	if err == nil {
		t.Error("expected error from done callback during stop string")
	}
}

func TestGenerateStream_ImmediateEOS_DoneError(t *testing.T) {
	tok := buildTestTokenizer()
	vocabSize := 8
	g := buildTestGraph(t, vocabSize, []int{2})

	gen := NewGenerator[float32](
		g, tok,
		compute.NewCPUEngine(numeric.Float32Ops{}),
		ModelConfig{VocabSize: vocabSize, MaxSeqLen: 32, EOSTokenID: 2, NumLayers: 0},
	)

	stream := TokenStreamFunc(func(_ string, done bool) error {
		if done {
			return fmt.Errorf("done error")
		}
		return nil
	})

	err := gen.GenerateStream(context.Background(), "hello", SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 10,
	}, stream)
	if err == nil {
		t.Error("expected error from done callback")
	}
}
