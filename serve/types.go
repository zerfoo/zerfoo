// Package serve request/response types and validation helpers for the
// OpenAI-compatible API server.
package serve

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// --- Request/Response types ---

// JSONSchemaFormat describes the json_schema object within a response_format request.
type JSONSchemaFormat struct {
	Name   string          `json:"name"`
	Strict bool            `json:"strict,omitempty"`
	Schema json.RawMessage `json:"schema"`
}

// ResponseFormat controls the output structure of a chat completion.
type ResponseFormat struct {
	Type       string            `json:"type"` // "text" | "json_object" | "json_schema"
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}

// ChatCompletionRequest represents the OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	TopK           *int            `json:"top_k,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     *ToolChoice     `json:"tool_choice,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ChatMessage is a single message in the chat.
// Content can be either a plain string or an array of content parts
// (for vision requests with type:"text" and type:"image_url").
// Custom JSON unmarshaling is in vision.go.
type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ImageURLs []ImageURL `json:"-"`
}

// CompletionRequest represents the OpenAI completion request.
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stream      bool     `json:"stream"`
}

// ChatCompletionResponse is the non-streaming response.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   UsageInfo              `json:"usage"`
}

// ToolCallFunction holds the function name and arguments in a tool call response.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCall represents a tool invocation in the assistant's response.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ChatCompletionChoice is a single choice in the response.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
	ToolCalls    []ToolCall  `json:"tool_calls,omitempty"`
}

// CompletionResponse is the non-streaming completion response.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   UsageInfo          `json:"usage"`
}

// CompletionChoice is a single choice in the completion response.
type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// UsageInfo reports token counts.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// EmbeddingRequest represents the OpenAI embeddings request.
type EmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"` // string or []string
}

// EmbeddingObject is a single embedding in the response.
type EmbeddingObject struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingResponse is the /v1/embeddings response.
type EmbeddingResponse struct {
	Object string            `json:"object"`
	Data   []EmbeddingObject `json:"data"`
	Model  string            `json:"model"`
	Usage  UsageInfo         `json:"usage"`
}

// ModelObject represents a model in the /v1/models response.
type ModelObject struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	Created      int64  `json:"created"`
	OwnedBy      string `json:"owned_by"`
	Architecture string `json:"architecture,omitempty"`
}

// ModelListResponse is the /v1/models response.
type ModelListResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

// ModelDeleteResponse is the DELETE /v1/models/:id response.
type ModelDeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// --- Validation and helper functions ---

// validateSamplingParams checks temperature, top_p, and top_k values.
// Temperature must be >= 0 (rejected otherwise). TopP is clamped to [0, 1].
// TopK must be >= 0 (rejected otherwise).
func validateSamplingParams(temperature *float64, topP *float64, topK *int) error {
	if temperature != nil && *temperature < 0 {
		return fmt.Errorf("temperature must be >= 0, got %g", *temperature)
	}
	if topP != nil {
		if *topP < 0 {
			*topP = 0
		} else if *topP > 1 {
			*topP = 1
		}
	}
	if topK != nil && *topK < 0 {
		return fmt.Errorf("top_k must be >= 0, got %d", *topK)
	}
	return nil
}

// isOOMError reports whether the error message indicates an out-of-memory condition.
func isOOMError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "out of memory") ||
		strings.Contains(msg, "oom") ||
		strings.Contains(msg, "cannot allocate") ||
		strings.Contains(msg, "cuda_error_out_of_memory") ||
		strings.Contains(msg, "cublas_status_alloc_failed")
}

// inferenceErrorStatus returns the appropriate HTTP status code for an inference error.
func inferenceErrorStatus(err error) int {
	if isOOMError(err) {
		return http.StatusServiceUnavailable
	}
	return http.StatusInternalServerError
}

// isFileNotFoundError reports whether the error indicates a missing file or model.
func isFileNotFoundError(err error) bool {
	if os.IsNotExist(err) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "no such file") ||
		strings.Contains(msg, "file not found") ||
		strings.Contains(msg, "model not found")
}

// sanitizeError returns a safe, client-facing error message and logs the
// original error details server-side. Internal details (file paths, stack
// traces, memory addresses) are never exposed to the caller.
func (s *Server) sanitizeError(err error) string {
	s.logger.Error("inference error", "error", err.Error())

	switch {
	case isOOMError(err):
		return "server temporarily overloaded, please retry"
	case isFileNotFoundError(err):
		return "model not available"
	default:
		return "inference failed"
	}
}

// isMaxBytesError reports whether the error (or any wrapped error) is an
// *http.MaxBytesError, which is returned when http.MaxBytesReader's limit
// is exceeded.
func isMaxBytesError(err error) bool {
	var mbe *http.MaxBytesError
	return errors.As(err, &mbe)
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]interface{}{
		"error": map[string]string{"message": message},
	})
}
