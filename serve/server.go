// Package serve provides an OpenAI-compatible HTTP API server for model inference.
package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

// Server wraps a loaded model and serves OpenAI-compatible HTTP endpoints.
type Server struct {
	model      *inference.Model
	draftModel *inference.Model // optional; enables speculative decoding
	mux        *http.ServeMux
	batch      *BatchScheduler // optional; nil means direct calls
	unloaded   bool            // true after DELETE /v1/models/:id
}

// ServerOption configures the server.
type ServerOption func(*Server)

// WithDraftModel enables speculative decoding using the given draft model.
// When set, completion requests use speculative decode with the draft model
// proposing tokens and the target model verifying them.
func WithDraftModel(draft *inference.Model) ServerOption {
	return func(s *Server) {
		s.draftModel = draft
	}
}

// WithBatchScheduler attaches a batch scheduler for non-streaming requests.
// When set, incoming completion requests are routed through the scheduler
// to be grouped into batches for higher throughput.
func WithBatchScheduler(bs *BatchScheduler) ServerOption {
	return func(s *Server) {
		s.batch = bs
	}
}

// NewServer creates a Server for the given model.
func NewServer(m *inference.Model, opts ...ServerOption) *Server {
	s := &Server{model: m, mux: http.NewServeMux()}
	for _, opt := range opts {
		opt(s)
	}
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/completions", s.handleCompletions)
	s.mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mux.HandleFunc("GET /v1/models/{id...}", s.handleModelInfo)
	s.mux.HandleFunc("DELETE /v1/models/{id...}", s.handleModelDelete)
	return s
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler { return s.mux }

// --- Request/Response types ---

// ChatCompletionRequest represents the OpenAI chat completion request.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Stream      bool          `json:"stream"`
}

// ChatMessage is a single message in the chat.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest represents the OpenAI completion request.
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Temperature *float64 `json:"temperature,omitempty"`
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

// ChatCompletionChoice is a single choice in the response.
type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
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

// --- Handlers ---

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required")
		return
	}

	// Build generation options.
	var opts []inference.GenerateOption
	if req.Temperature != nil {
		opts = append(opts, inference.WithTemperature(*req.Temperature))
	}
	if req.TopP != nil {
		opts = append(opts, inference.WithTopP(*req.TopP))
	}
	if req.MaxTokens != nil {
		opts = append(opts, inference.WithMaxTokens(*req.MaxTokens))
	}

	// Convert messages.
	messages := make([]inference.Message, len(req.Messages))
	for i, m := range req.Messages {
		messages[i] = inference.Message{Role: m.Role, Content: m.Content}
	}

	if req.Stream {
		s.streamChatCompletion(w, r.Context(), messages, opts)
		return
	}

	var resp inference.Response
	var err error

	if s.batch != nil {
		// Build prompt from messages for batching.
		var prompt strings.Builder
		for _, m := range messages {
			prompt.WriteString(m.Content)
			prompt.WriteString(" ")
		}
		var br BatchResult
		br, err = s.batch.Submit(r.Context(), BatchRequest{Prompt: prompt.String()})
		if err == nil {
			resp = inference.Response{Content: br.Value}
		}
	} else {
		resp, err = s.model.Chat(r.Context(), messages, opts...)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	writeJSON(w, http.StatusOK, ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []ChatCompletionChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: resp.Content},
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     resp.PromptTokens,
			CompletionTokens: resp.CompletionTokens,
			TotalTokens:      resp.TokensUsed,
		},
	})
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	var opts []inference.GenerateOption
	if req.Temperature != nil {
		opts = append(opts, inference.WithTemperature(*req.Temperature))
	}
	if req.MaxTokens != nil {
		opts = append(opts, inference.WithMaxTokens(*req.MaxTokens))
	}

	if req.Stream {
		s.streamCompletion(w, r.Context(), req.Prompt, opts)
		return
	}

	var result string
	var err error

	switch {
	case s.batch != nil:
		var br BatchResult
		br, err = s.batch.Submit(r.Context(), BatchRequest{Prompt: req.Prompt})
		result = br.Value
	case s.draftModel != nil:
		result, err = s.model.SpeculativeGenerate(r.Context(), s.draftModel, req.Prompt, 4, opts...)
	default:
		result, err = s.model.Generate(r.Context(), req.Prompt, opts...)
	}

	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	// Count prompt and completion tokens.
	var promptTokens, completionTokens int
	if tok := s.model.Tokenizer(); tok != nil {
		if ids, err := tok.Encode(req.Prompt); err == nil {
			promptTokens = len(ids)
		}
		if ids, err := tok.Encode(result); err == nil {
			completionTokens = len(ids)
		}
	}

	writeJSON(w, http.StatusOK, CompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []CompletionChoice{{
			Index:        0,
			Text:         result,
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	if s.unloaded {
		writeJSON(w, http.StatusOK, ModelListResponse{Object: "list"})
		return
	}

	obj := s.buildModelObject()
	writeJSON(w, http.StatusOK, ModelListResponse{
		Object: "list",
		Data:   []ModelObject{obj},
	})
}

func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	if s.unloaded {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	obj := s.buildModelObject()
	if obj.ID != id {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	writeJSON(w, http.StatusOK, obj)
}

func (s *Server) handleModelDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	if s.unloaded {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	obj := s.buildModelObject()
	if obj.ID != id {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	_ = s.model.Close()
	s.unloaded = true

	writeJSON(w, http.StatusOK, ModelDeleteResponse{
		ID:      id,
		Object:  "model",
		Deleted: true,
	})
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	// Parse input: can be a single string or an array of strings.
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			s, ok := item.(string)
			if !ok {
				writeError(w, http.StatusBadRequest, "input array must contain strings")
				return
			}
			inputs = append(inputs, s)
		}
	default:
		writeError(w, http.StatusBadRequest, "input must be a string or array of strings")
		return
	}

	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, "input is required")
		return
	}

	var data []EmbeddingObject
	var totalTokens int
	for i, text := range inputs {
		emb, err := s.model.Embed(r.Context(), text)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		data = append(data, EmbeddingObject{
			Object:    "embedding",
			Embedding: emb,
			Index:     i,
		})
		if tok := s.model.Tokenizer(); tok != nil {
			if ids, err := tok.Encode(text); err == nil {
				totalTokens += len(ids)
			}
		}
	}

	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	writeJSON(w, http.StatusOK, EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  modelID,
		Usage: UsageInfo{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	})
}

func (s *Server) buildModelObject() ModelObject {
	modelID := ""
	arch := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
		arch = info.Architecture
	}
	if arch == "" {
		arch = s.model.Config().Architecture
	}
	return ModelObject{
		ID:           modelID,
		Object:       "model",
		Created:      time.Now().Unix(),
		OwnedBy:      "local",
		Architecture: arch,
	}
}

// --- Streaming ---

func (s *Server) streamChatCompletion(w http.ResponseWriter, ctx context.Context, messages []inference.Message, opts []inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	// Format the prompt from messages.
	var prompt strings.Builder
	for _, m := range messages {
		prompt.WriteString(m.Content)
		prompt.WriteString(" ")
	}

	err := s.model.GenerateStream(ctx, prompt.String(), generate.TokenStreamFunc(func(token string, done bool) error {
		if done {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"delta": map[string]string{"content": token}},
			},
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return nil
	}), opts...)
	if err != nil {
		_, _ = fmt.Fprintf(w, "data: {\"error\": %q}\n\n", err.Error())
		flusher.Flush()
	}
}

func (s *Server) streamCompletion(w http.ResponseWriter, ctx context.Context, prompt string, opts []inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	err := s.model.GenerateStream(ctx, prompt, generate.TokenStreamFunc(func(token string, done bool) error {
		if done {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
		chunk := map[string]interface{}{
			"choices": []map[string]interface{}{
				{"text": token},
			},
		}
		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return nil
	}), opts...)
	if err != nil {
		_, _ = fmt.Fprintf(w, "data: {\"error\": %q}\n\n", err.Error())
		flusher.Flush()
	}
}

// --- Helpers ---

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

// Close implements shutdown.Closer for graceful shutdown integration.
func (s *Server) Close(_ context.Context) error {
	if s.batch != nil {
		s.batch.Stop()
	}
	return nil
}
