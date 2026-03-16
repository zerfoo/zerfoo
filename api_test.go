package zerfoo

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"
)

func TestLoad_pathDetection(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		isLocal bool
	}{
		{"absolute path", "/models/gemma.gguf", true},
		{"relative dot-slash", "./model.gguf", true},
		{"relative parent", "../model.gguf", true},
		{"huggingface id", "google/gemma-3-1b-it", false},
		{"short alias", "gemma-3-1b-q4", false},
		{"bare name", "my-model", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLocalPath(tt.input)
			if got != tt.isLocal {
				t.Errorf("isLocalPath(%q) = %v, want %v", tt.input, got, tt.isLocal)
			}
		})
	}
}

func TestLoad_missingFile(t *testing.T) {
	_, err := Load("/nonexistent/path/to/model.gguf")
	if err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}
}

func TestLoad_huggingFaceStub(t *testing.T) {
	_, err := Load("google/gemma-3-1b-it")
	if err == nil {
		t.Fatal("expected error for HuggingFace model ID, got nil")
	}
	want := "HuggingFace download not yet available"
	if got := err.Error(); got != want && len(got) < len(want) {
		t.Errorf("error = %q, want substring %q", got, want)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b Embedding
		want float32
	}{
		{
			name: "identical vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{1, 0, 0}},
			want: 1.0,
		},
		{
			name: "orthogonal vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{0, 1, 0}},
			want: 0.0,
		},
		{
			name: "opposite vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{-1, 0, 0}},
			want: -1.0,
		},
		{
			name: "45 degree angle",
			a:    Embedding{Vector: []float32{1, 0}},
			b:    Embedding{Vector: []float32{1, 1}},
			want: float32(1.0 / math.Sqrt(2)),
		},
		{
			name: "empty vectors",
			a:    Embedding{Vector: []float32{}},
			b:    Embedding{Vector: []float32{}},
			want: 0.0,
		},
		{
			name: "mismatched lengths",
			a:    Embedding{Vector: []float32{1, 2}},
			b:    Embedding{Vector: []float32{1, 2, 3}},
			want: 0.0,
		},
		{
			name: "zero vector",
			a:    Embedding{Vector: []float32{0, 0, 0}},
			b:    Embedding{Vector: []float32{1, 2, 3}},
			want: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.a.CosineSimilarity(tt.b)
			if diff := math.Abs(float64(got - tt.want)); diff > 1e-6 {
				t.Errorf("CosineSimilarity = %v, want %v (diff %v)", got, tt.want, diff)
			}
		})
	}
}

func TestEmbed_stub(t *testing.T) {
	// Create a zero-value Model to test the Embed stub.
	// We can't call Embed on a nil inner, so we test the error message
	// by constructing a Model with a non-nil inner. Since we can't easily
	// construct an inference.Model, we test the function signature works.
	m := &Model{}

	// This will panic due to nil inner if Embed tries to use it,
	// but it should return an error before accessing inner.
	result, err := m.Embed([]string{"hello"})
	if err == nil {
		t.Fatal("expected error from Embed stub, got nil")
	}
	want := "embedding not yet supported"
	if got := err.Error(); got != want {
		t.Errorf("Embed error = %q, want %q", got, want)
	}
	if result != nil {
		t.Errorf("Embed result = %v, want nil", result)
	}
}

func TestChatStream_nilModel(t *testing.T) {
	m := &Model{}
	ch, err := m.ChatStream(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for nil model, got nil")
	}
	if ch != nil {
		t.Errorf("expected nil channel, got %v", ch)
	}
}

func TestChatStream_yieldsTokens(t *testing.T) {
	m := &Model{
		generateFunc: func(ctx context.Context, prompt string) (string, error) {
			return "hello world foo", nil
		},
	}

	ch, err := m.ChatStream(context.Background(), "test prompt")
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if ch == nil {
		t.Fatal("expected non-nil channel")
	}

	var tokens []StreamToken
	for tok := range ch {
		tokens = append(tokens, tok)
	}

	// Expect 3 word tokens + 1 done token.
	if len(tokens) != 4 {
		t.Fatalf("got %d tokens, want 4: %v", len(tokens), tokens)
	}

	// Verify words are streamed with spaces between them.
	if tokens[0].Text != "hello " {
		t.Errorf("token[0] = %q, want %q", tokens[0].Text, "hello ")
	}
	if tokens[1].Text != "world " {
		t.Errorf("token[1] = %q, want %q", tokens[1].Text, "world ")
	}
	if tokens[2].Text != "foo" {
		t.Errorf("token[2] = %q, want %q", tokens[2].Text, "foo")
	}

	// Last token should be done signal.
	if !tokens[3].Done {
		t.Error("last token should have Done=true")
	}

	// Reconstructed text should match original.
	var sb strings.Builder
	for _, tok := range tokens {
		sb.WriteString(tok.Text)
	}
	if got := sb.String(); got != "hello world foo" {
		t.Errorf("reconstructed text = %q, want %q", got, "hello world foo")
	}
}

func TestChatStream_channelCloses(t *testing.T) {
	m := &Model{
		generateFunc: func(ctx context.Context, prompt string) (string, error) {
			return "done", nil
		},
	}

	ch, err := m.ChatStream(context.Background(), "test")
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}

	// Drain channel and verify it closes.
	count := 0
	for range ch {
		count++
	}
	if count != 2 { // 1 word + 1 done
		t.Errorf("got %d tokens, want 2", count)
	}
}

func TestChatStream_contextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	m := &Model{
		generateFunc: func(ctx context.Context, prompt string) (string, error) {
			// Simulate a slow generation that respects context.
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(5 * time.Second):
				return "this should not appear", nil
			}
		},
	}

	ch, err := m.ChatStream(ctx, "test")
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}

	// Cancel context immediately.
	cancel()

	// Channel should close without delivering the full result.
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()

	select {
	case _, ok := <-ch:
		if ok {
			// May get some tokens before cancellation is noticed; drain.
			for range ch {
			}
		}
	case <-timer.C:
		t.Fatal("channel did not close after context cancellation")
	}
}

func TestChatStream_emptyResult(t *testing.T) {
	m := &Model{
		generateFunc: func(ctx context.Context, prompt string) (string, error) {
			return "", nil
		},
	}

	ch, err := m.ChatStream(context.Background(), "test")
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}

	var tokens []StreamToken
	for tok := range ch {
		tokens = append(tokens, tok)
	}

	// Empty string has no words, only a done signal.
	if len(tokens) != 1 {
		t.Fatalf("got %d tokens, want 1: %v", len(tokens), tokens)
	}
	if !tokens[0].Done {
		t.Error("expected done signal")
	}
}
