package zerfoo

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/huggingface"
	"github.com/zerfoo/zerfoo/serve"
)

// Model is a loaded language model ready for inference.
//
// A Model is created via [Load] and used for text generation, embedding,
// and tool-call detection. [Model.Close] must be called when the model is no
// longer needed to release GPU and CPU resources.
//
// Stable.
type Model struct {
	inner *inference.Model

	// generateFunc is an optional override for testing ChatStream.
	// When non-nil, ChatStream uses this instead of inner.Generate.
	generateFunc func(ctx context.Context, prompt string) (string, error)
}

// NewModel creates a Model with a custom generate function for testing
// and demonstrations without loading a real GGUF model file. The provided
// function is called by Generate, Chat, and ChatStream.
//
// This is useful for writing pkg.go.dev examples and unit tests:
//
//	m := zerfoo.NewModel(func(ctx context.Context, prompt string) (string, error) {
//	    return "Hello from the model!", nil
//	})
//	result, _ := m.Generate(ctx, "Hi")
func NewModel(fn func(ctx context.Context, prompt string) (string, error)) *Model {
	return &Model{generateFunc: fn}
}

// defaultQuant is the preferred quantization when none is specified.
const defaultQuant = "Q4_K_M"

// Load loads a model from a file path or HuggingFace model ID.
//
// Paths starting with "/", "./" or "../" are treated as local GGUF files.
// All other strings are treated as HuggingFace model IDs (e.g. "google/gemma-3-4b"
// or "google/gemma-3-4b/Q8_0"). If the model is not cached locally it will be
// downloaded from HuggingFace.
//
// Stable.
func Load(pathOrID string) (*Model, error) {
	if isLocalPath(pathOrID) {
		m, err := inference.LoadFile(pathOrID)
		if err != nil {
			return nil, fmt.Errorf("load %q: %w", pathOrID, err)
		}
		return &Model{inner: m}, nil
	}
	return loadFromHuggingFace(pathOrID)
}

// parseModelID splits a model identifier into repo ID and optional quant.
// Accepted formats:
//
//	"owner/model"            → ("owner/model", "Q4_K_M")
//	"owner/model/Q8_0"       → ("owner/model", "Q8_0")
func parseModelID(id string) (repoID, quant string) {
	parts := strings.Split(id, "/")
	if len(parts) == 3 {
		return parts[0] + "/" + parts[1], parts[2]
	}
	return id, defaultQuant
}

// loadFromHuggingFace resolves a HuggingFace model ID through the cache,
// downloading the GGUF file if not already present.
func loadFromHuggingFace(id string) (*Model, error) {
	repoID, quant := parseModelID(id)

	manifest, err := huggingface.LoadManifest()
	if err != nil {
		return nil, fmt.Errorf("load %q: %w", id, err)
	}

	// Cache hit: model already downloaded.
	if cached, ok := manifest.FindByRepo(repoID); ok {
		if _, statErr := os.Stat(cached.Path); statErr == nil {
			m, loadErr := inference.LoadFile(cached.Path)
			if loadErr != nil {
				return nil, fmt.Errorf("load cached %q: %w", cached.Path, loadErr)
			}
			return &Model{inner: m}, nil
		}
		// File missing on disk — fall through to re-download.
	}

	// Cache miss: resolve the best GGUF file and download it.
	client := huggingface.NewClient()
	fileInfo, err := client.ResolveGGUF(repoID, quant)
	if err != nil {
		return nil, fmt.Errorf("load %q: %w", id, err)
	}

	cacheDir, err := huggingface.CacheDir()
	if err != nil {
		return nil, fmt.Errorf("load %q: %w", id, err)
	}

	destPath := filepath.Join(cacheDir, repoID, fileInfo.Filename)
	downloadURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, fileInfo.Filename)

	dl := huggingface.NewDownloader(nil)
	if err := dl.Download(context.Background(), downloadURL, destPath); err != nil {
		return nil, fmt.Errorf("load %q: download: %w", id, err)
	}

	// Update manifest.
	info, _ := os.Stat(destPath)
	manifest.Add(huggingface.CachedModel{
		RepoID:   repoID,
		Filename: fileInfo.Filename,
		Path:     destPath,
		Size:     info.Size(),
	})
	if err := huggingface.SaveManifest(manifest); err != nil {
		return nil, fmt.Errorf("load %q: save manifest: %w", id, err)
	}

	m, err := inference.LoadFile(destPath)
	if err != nil {
		return nil, fmt.Errorf("load %q: %w", id, err)
	}
	return &Model{inner: m}, nil
}

// isLocalPath returns true if the string looks like a local file path.
func isLocalPath(s string) bool {
	return strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../")
}

// Chat runs a simple one-shot generation and returns the generated text.
//
// Stable.
func (m *Model) Chat(prompt string) (string, error) {
	result, err := m.Generate(context.Background(), prompt)
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// Generate runs text generation with the given prompt and options.
//
// Stable.
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
	if gopts.schema != nil {
		g, err := grammar.Convert(gopts.schema)
		if err != nil {
			return nil, fmt.Errorf("convert schema to grammar: %w", err)
		}
		infOpts = append(infOpts, inference.WithGrammar(g))
	}

	start := time.Now()
	var text string
	var err error
	if m.generateFunc != nil {
		text, err = m.generateFunc(ctx, prompt)
	} else {
		text, err = m.inner.Generate(ctx, prompt, infOpts...)
	}
	if err != nil {
		return nil, err
	}
	duration := time.Since(start)

	// Estimate token count from the tokenizer.
	tokenCount := 0
	if m.inner != nil {
		tok := m.inner.Tokenizer()
		if tok != nil {
			if ids, encErr := tok.Encode(text); encErr == nil {
				tokenCount = len(ids)
			}
		}
	}

	result := &GenerateResult{
		Text:       text,
		TokenCount: tokenCount,
		Duration:   duration,
	}

	// Detect tool calls if tools are configured.
	if len(gopts.tools) > 0 {
		tc := serve.ToolChoice{Mode: "auto"}
		if gopts.toolChoice != nil {
			tc = *gopts.toolChoice
		}
		if tcResult, ok := serve.DetectToolCall(text, gopts.tools, tc); ok {
			result.ToolCalls = []ToolCall{{
				ID:           tcResult.ID,
				FunctionName: tcResult.FunctionName,
				Arguments:    tcResult.Arguments,
			}}
		}
	}

	return result, nil
}

// Embed returns embeddings for the given texts.
//
// Each input string is tokenized, its token embeddings are looked up from the
// model's embedding table, mean-pooled, and L2-normalized.
//
// Stable.
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
//
// Stable.
func (m *Model) Close() error {
	return m.inner.Close()
}

// GenerateResult holds the result of a text generation call.
//
// Stable.
type GenerateResult struct {
	Text       string
	TokenCount int
	Duration   time.Duration
	ToolCalls  []ToolCall
}

// Embedding holds a text embedding vector.
//
// Stable.
type Embedding struct {
	Vector []float32
}

// CosineSimilarity computes the cosine similarity between two embeddings.
//
// Stable.
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
	tools       []serve.Tool
	toolChoice  *serve.ToolChoice
	schema      *grammar.JSONSchema
}

// GenerateOption configures the behavior of [Model.Generate].
//
// Stable.
type GenerateOption func(*generateOptions)

// WithGenMaxTokens sets the maximum number of tokens to generate.
//
// Stable.
func WithGenMaxTokens(n int) GenerateOption {
	return func(o *generateOptions) {
		o.maxTokens = n
	}
}

// WithGenTemperature sets the sampling temperature.
//
// Stable.
func WithGenTemperature(t float32) GenerateOption {
	return func(o *generateOptions) {
		o.temperature = t
	}
}

// WithGenTopP sets the top-p (nucleus) sampling parameter.
//
// Stable.
func WithGenTopP(p float32) GenerateOption {
	return func(o *generateOptions) {
		o.topP = p
	}
}

// StreamToken represents a token received during streaming generation.
//
// Stable.
type StreamToken struct {
	Text string
	Done bool
}

// ChatStream starts streaming generation and returns a receive-only channel
// that yields [StreamToken] values as they are generated. The channel is closed
// when generation completes or ctx is canceled. The error return is non-nil
// only if startup fails (e.g. the model is not loaded).
//
// Stable.
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
			if i < len(words)-1 {
				word += " "
			}
			select {
			case ch <- StreamToken{Text: word}:
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

// ToolCall represents a tool invocation detected in model output.
//
// Experimental.
type ToolCall struct {
	ID           string
	FunctionName string
	Arguments    json.RawMessage
}

// WithTools configures the tools available for tool call detection.
//
// When tools are provided, [Model.Generate] will attempt to detect tool calls
// in the model output and populate [GenerateResult.ToolCalls].
//
// Experimental.
func WithTools(tools ...serve.Tool) GenerateOption {
	return func(o *generateOptions) {
		o.tools = tools
	}
}

// WithToolChoice sets the tool choice mode for tool call detection.
//
// Experimental.
func WithToolChoice(choice serve.ToolChoice) GenerateOption {
	return func(o *generateOptions) {
		o.toolChoice = &choice
	}
}

// WithSchema enables grammar-guided decoding.
//
// The model's output will be constrained to valid JSON matching the given schema.
//
// Experimental.
func WithSchema(schema grammar.JSONSchema) GenerateOption {
	return func(o *generateOptions) {
		o.schema = &schema
	}
}
