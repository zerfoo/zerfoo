package zerfoo

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	ztoken "github.com/zerfoo/ztoken"
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

func TestParseModelID(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantRepo  string
		wantQuant string
	}{
		{"owner/model", "google/gemma-3-4b", "google/gemma-3-4b", defaultQuant},
		{"owner/model/quant", "google/gemma-3-4b/Q8_0", "google/gemma-3-4b", "Q8_0"},
		{"bare name", "my-model", "my-model", defaultQuant},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo, quant := parseModelID(tt.input)
			if repo != tt.wantRepo {
				t.Errorf("parseModelID(%q) repo = %q, want %q", tt.input, repo, tt.wantRepo)
			}
			if quant != tt.wantQuant {
				t.Errorf("parseModelID(%q) quant = %q, want %q", tt.input, quant, tt.wantQuant)
			}
		})
	}
}

func TestLoadFromHuggingFace_cacheMiss(t *testing.T) {
	// Calling loadFromHuggingFace with a non-existent model will fail
	// at the HuggingFace API call (no network in CI), confirming the
	// cache-miss path reaches the client.
	_, err := loadFromHuggingFace("nonexistent-org/nonexistent-model")
	if err == nil {
		t.Fatal("expected error for non-existent HuggingFace model, got nil")
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

func newTestModelWithEmbeddings(vocabTokens []string, dim int, weights []float32) *Model {
	tok := ztoken.NewWhitespaceTokenizer()
	for _, w := range vocabTokens {
		tok.AddToken(w)
	}
	inner := inference.NewTestModel(nil, tok, nil, inference.ModelMetadata{}, nil)
	inner.SetEmbeddingWeights(weights, dim)
	return &Model{inner: inner}
}

func TestEmbed_returnsCorrectShape(t *testing.T) {
	dim := 4
	vocab := 6 // 4 special + 2 real
	weights := make([]float32, vocab*dim)
	weights[4*dim+0] = 1 // "hello" = ID 4
	weights[5*dim+1] = 1 // "world" = ID 5

	m := newTestModelWithEmbeddings([]string{"hello", "world"}, dim, weights)

	results, err := m.Embed([]string{"hello", "world"})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Embed returned %d embeddings, want 2", len(results))
	}
	for i, emb := range results {
		if len(emb.Vector) != dim {
			t.Errorf("embedding[%d] has dim %d, want %d", i, len(emb.Vector), dim)
		}
	}
}

func TestEmbed_identicalInputsSimilarity(t *testing.T) {
	dim := 3
	vocab := 5
	weights := make([]float32, vocab*dim)
	weights[4*dim+0] = 1
	weights[4*dim+1] = 2
	weights[4*dim+2] = 3

	m := newTestModelWithEmbeddings([]string{"hello"}, dim, weights)

	results, err := m.Embed([]string{"hello", "hello"})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}

	sim := results[0].CosineSimilarity(results[1])
	if diff := math.Abs(float64(sim - 1.0)); diff > 1e-6 {
		t.Errorf("identical inputs: CosineSimilarity = %v, want 1.0", sim)
	}
}

func TestEmbed_orthogonalTokens(t *testing.T) {
	dim := 3
	vocab := 6
	weights := make([]float32, vocab*dim)
	weights[4*dim+0] = 1 // "cat" along x-axis
	weights[5*dim+1] = 1 // "dog" along y-axis

	m := newTestModelWithEmbeddings([]string{"cat", "dog"}, dim, weights)

	results, err := m.Embed([]string{"cat", "dog"})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}

	sim := results[0].CosineSimilarity(results[1])
	if diff := math.Abs(float64(sim)); diff > 1e-6 {
		t.Errorf("orthogonal tokens: CosineSimilarity = %v, want 0.0", sim)
	}
}

func TestEmbed_emptyInput(t *testing.T) {
	inner := inference.NewTestModel(nil, nil, nil, inference.ModelMetadata{}, nil)
	m := &Model{inner: inner}

	results, err := m.Embed([]string{})
	if err != nil {
		t.Fatalf("Embed(empty) returned error: %v", err)
	}
	if results != nil {
		t.Errorf("Embed(empty) = %v, want nil", results)
	}
}

func TestEmbed_noEmbeddingWeights(t *testing.T) {
	tok := ztoken.NewWhitespaceTokenizer()
	inner := inference.NewTestModel(nil, tok, nil, inference.ModelMetadata{}, nil)
	m := &Model{inner: inner}

	_, err := m.Embed([]string{"hello"})
	if err == nil {
		t.Fatal("expected error when embedding weights not set")
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
