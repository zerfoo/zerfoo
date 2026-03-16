package zerfoo

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/inference"
)

// Model is a loaded model ready for inference.
type Model struct {
	inner *inference.Model

	// generateFunc is an optional override for testing ChatStream.
	// When non-nil, ChatStream uses this instead of inner.Generate.
	generateFunc func(ctx context.Context, prompt string) (string, error)
}

// Load loads a model from a file path or HuggingFace model ID.
// Paths starting with "/", "./" or "../" are treated as local GGUF files.
// All other strings are treated as HuggingFace model IDs (not yet supported).
func Load(pathOrID string) (*Model, error) {
	if isLocalPath(pathOrID) {
		m, err := inference.LoadFile(pathOrID)
		if err != nil {
			return nil, fmt.Errorf("load %q: %w", pathOrID, err)
		}
		return &Model{inner: m}, nil
	}
	return nil, fmt.Errorf("HuggingFace download not yet available in this version; use a local file path")
}

// isLocalPath returns true if the string looks like a local file path.
func isLocalPath(s string) bool {
	return strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../")
}

// Chat runs a simple one-shot generation and returns the generated text.
func (m *Model) Chat(prompt string) (string, error) {
	result, err := m.Generate(context.Background(), prompt)
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// Generate runs generation with options.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (*GenerateResult, error) {
	var gopts generateOptions
	for _, o := range opts {
		o(&gopts)
	}

	var infOpts []inference.GenerateOption
	if gopts.maxTokens > 0 {
		infOpts = append(infOpts, inference.WithMaxTokens(gopts.maxTokens))
	}
	if gopts.temperature > 0 {
		infOpts = append(infOpts, inference.WithTemperature(float64(gopts.temperature)))
	}
	if gopts.topP > 0 && gopts.topP < 1.0 {
		infOpts = append(infOpts, inference.WithTopP(float64(gopts.topP)))
	}

	start := time.Now()
	text, err := m.inner.Generate(ctx, prompt, infOpts...)
	if err != nil {
		return nil, err
	}
	duration := time.Since(start)

	// Estimate token count from the tokenizer.
	tokenCount := 0
	tok := m.inner.Tokenizer()
	if tok != nil {
		if ids, encErr := tok.Encode(text); encErr == nil {
			tokenCount = len(ids)
		}
	}

	return &GenerateResult{
		Text:       text,
		TokenCount: tokenCount,
		Duration:   duration,
	}, nil
}

// Embed returns embeddings for the given texts. Each input string is tokenized,
// its token embeddings are looked up from the model's embedding table,
// mean-pooled, and L2-normalized.
func (m *Model) Embed(texts []string) ([]Embedding, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	results := make([]Embedding, len(texts))
	for i, text := range texts {
		vec, err := m.inner.Embed(text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		results[i] = Embedding{Vector: vec}
	}
	return results, nil
}

// Close releases model resources.
func (m *Model) Close() error {
	return m.inner.Close()
}

// GenerateResult holds the result of a generation.
type GenerateResult struct {
	Text       string
	TokenCount int
	Duration   time.Duration
}

// Embedding holds a text embedding vector.
type Embedding struct {
	Vector []float32
}

// CosineSimilarity computes cosine similarity with another embedding.
func (e Embedding) CosineSimilarity(other Embedding) float32 {
	if len(e.Vector) == 0 || len(e.Vector) != len(other.Vector) {
		return 0
	}
	var dot, normA, normB float64
	for i := range e.Vector {
		a := float64(e.Vector[i])
		b := float64(other.Vector[i])
		dot += a * b
		normA += a * a
		normB += b * b
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// generateOptions holds parsed generation options.
type generateOptions struct {
	maxTokens   int
	temperature float32
	topP        float32
}

// GenerateOption is an option for Generate.
type GenerateOption func(*generateOptions)

// WithGenMaxTokens sets the maximum number of tokens to generate.
func WithGenMaxTokens(n int) GenerateOption {
	return func(o *generateOptions) {
		o.maxTokens = n
	}
}

// WithGenTemperature sets the sampling temperature.
func WithGenTemperature(t float32) GenerateOption {
	return func(o *generateOptions) {
		o.temperature = t
	}
}

// WithGenTopP sets the top-p sampling parameter.
func WithGenTopP(p float32) GenerateOption {
	return func(o *generateOptions) {
		o.topP = p
	}
}

// StreamToken represents a token received during streaming generation.
type StreamToken struct {
	Text string
	Done bool
}

// ChatStream starts streaming generation and returns a receive-only channel
// that yields token strings as they are generated. The channel is closed when
// generation completes or ctx is cancelled. The error return is non-nil only
// if startup fails (e.g. the model is not loaded).
func (m *Model) ChatStream(ctx context.Context, prompt string, opts ...GenerateOption) (<-chan StreamToken, error) {
	if m.inner == nil && m.generateFunc == nil {
		return nil, fmt.Errorf("model not loaded")
	}

	// Build the generate function: use the override if set, otherwise wrap inner.Generate.
	genFn := m.generateFunc
	if genFn == nil {
		var gopts generateOptions
		for _, o := range opts {
			o(&gopts)
		}

		var infOpts []inference.GenerateOption
		if gopts.maxTokens > 0 {
			infOpts = append(infOpts, inference.WithMaxTokens(gopts.maxTokens))
		}
		if gopts.temperature > 0 {
			infOpts = append(infOpts, inference.WithTemperature(float64(gopts.temperature)))
		}
		if gopts.topP > 0 && gopts.topP < 1.0 {
			infOpts = append(infOpts, inference.WithTopP(float64(gopts.topP)))
		}

		genFn = func(ctx context.Context, prompt string) (string, error) {
			return m.inner.Generate(ctx, prompt, infOpts...)
		}
	}

	ch := make(chan StreamToken, 64)

	go func() {
		defer close(ch)

		text, err := genFn(ctx, prompt)
		if err != nil {
			return
		}

		// Split the result into words and stream them as tokens.
		words := strings.Fields(text)
		for i, word := range words {
			if ctx.Err() != nil {
				return
			}
			tok := word
			if i < len(words)-1 {
				tok += " "
			}
			select {
			case ch <- StreamToken{Text: tok}:
			case <-ctx.Done():
				return
			}
		}
		// Send done signal.
		select {
		case ch <- StreamToken{Done: true}:
		case <-ctx.Done():
		}
	}()

	return ch, nil
}
