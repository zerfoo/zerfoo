// Package langchain provides an adapter that makes Zerfoo's OpenAI-compatible
// HTTP API compatible with LangChain-Go's LLM interface.
//
// The adapter does not import LangChain-Go directly. Instead it exposes the
// same method signatures that LangChain-Go expects (Call, Generate,
// GeneratePrompt) so that callers can use it as a drop-in LLM without taking
// a dependency on the langchain-go module.
//
// Usage:
//
//	llm := langchain.NewAdapter("http://localhost:8080", "llama3")
//	resp, err := llm.Call(ctx, "What is the capital of France?")
package langchain

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Adapter wraps Zerfoo's OpenAI-compatible HTTP API and exposes it with the
// method signatures expected by LangChain-Go's schema.LLM interface.
type Adapter struct {
	baseURL    string
	model      string
	httpClient *http.Client
	// Temperature controls randomness (0–1). Default 0.7.
	Temperature float32
	// MaxTokens limits the response length. 0 means server default.
	MaxTokens int
	// StopWords is an optional list of stop sequences.
	StopWords []string
}

// AdapterOption configures an Adapter.
type AdapterOption func(*Adapter)

// WithTemperature sets the sampling temperature.
func WithTemperature(t float32) AdapterOption {
	return func(a *Adapter) { a.Temperature = t }
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) AdapterOption {
	return func(a *Adapter) { a.MaxTokens = n }
}

// WithStopWords sets stop sequences that halt generation.
func WithStopWords(words ...string) AdapterOption {
	return func(a *Adapter) { a.StopWords = words }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(c *http.Client) AdapterOption {
	return func(a *Adapter) { a.httpClient = c }
}

// NewAdapter creates an Adapter pointing at a running Zerfoo serve instance.
//
// baseURL is the server root (e.g. "http://localhost:8080").
// model is the model identifier forwarded in the request body.
func NewAdapter(baseURL, model string, opts ...AdapterOption) *Adapter {
	a := &Adapter{
		baseURL:     strings.TrimRight(baseURL, "/"),
		model:       model,
		Temperature: 0.7,
		httpClient:  &http.Client{Timeout: 120 * time.Second},
	}
	for _, o := range opts {
		o(a)
	}
	return a
}

// --- OpenAI-compatible request / response types (internal) ---

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float32       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
	Stream      bool          `json:"stream"`
}

type chatChoice struct {
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatResponse struct {
	Choices []chatChoice `json:"choices"`
	Error   *apiError    `json:"error,omitempty"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// --- LLM interface methods ---

// Call sends a single prompt to the model and returns the generated text.
// This implements the LangChain-Go schema.LLM.Call signature.
func (a *Adapter) Call(ctx context.Context, prompt string, stop ...string) (string, error) {
	stopWords := a.StopWords
	if len(stop) > 0 {
		stopWords = stop
	}
	return a.generate(ctx, prompt, stopWords)
}

// Generate runs the model over each prompt in prompts and returns one
// generation per prompt. This mirrors the LangChain-Go schema.LLM.Generate
// signature (adapted to work without importing langchain types).
func (a *Adapter) Generate(ctx context.Context, prompts []string, stop ...string) ([]string, error) {
	stopWords := a.StopWords
	if len(stop) > 0 {
		stopWords = stop
	}
	results := make([]string, len(prompts))
	for i, p := range prompts {
		text, err := a.generate(ctx, p, stopWords)
		if err != nil {
			return nil, fmt.Errorf("prompt[%d]: %w", i, err)
		}
		results[i] = text
	}
	return results, nil
}

// generate sends a single chat completion request and returns the text.
func (a *Adapter) generate(ctx context.Context, prompt string, stop []string) (string, error) {
	req := chatRequest{
		Model:       a.model,
		Messages:    []chatMessage{{Role: "user", Content: prompt}},
		Temperature: a.Temperature,
		MaxTokens:   a.MaxTokens,
		Stop:        stop,
		Stream:      false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("langchain adapter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		a.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("langchain adapter: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("langchain adapter: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("langchain adapter: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("langchain adapter: server returned %d: %s", resp.StatusCode, raw)
	}

	var chatResp chatResponse
	if err := json.Unmarshal(raw, &chatResp); err != nil {
		return "", fmt.Errorf("langchain adapter: decode response: %w", err)
	}
	if chatResp.Error != nil {
		return "", fmt.Errorf("langchain adapter: api error (%s): %s", chatResp.Error.Type, chatResp.Error.Message)
	}
	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("langchain adapter: no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}

// Type returns a string identifier for this LLM type, satisfying the
// LangChain-Go schema.LLM interface.
func (a *Adapter) Type() string { return "zerfoo" }
