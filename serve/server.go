// Package serve provides an OpenAI-compatible HTTP API server for model inference.
package serve

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/security"
	"github.com/zerfoo/ztensor/log"
	"github.com/zerfoo/ztensor/metrics/runtime"
)

//go:embed openapi.yaml
var openapiSpec []byte

// Server wraps a loaded model and serves OpenAI-compatible HTTP endpoints.
type Server struct {
	model      *inference.Model
	draftModel *inference.Model // optional; enables speculative decoding
	mux        *http.ServeMux
	batch      *BatchScheduler // optional; nil means direct calls
	unloaded    atomic.Bool     // true after DELETE /v1/models/:id
	inflight    sync.WaitGroup  // tracks in-flight inference requests
	transcriber      Transcriber          // optional; enables /v1/audio/transcriptions
	classifier       Classifier           // optional; enables /v1/classify
	logger           log.Logger
	metrics          *ServerMetrics
	classifyMetrics  *ClassifyMetrics
	collector        runtime.Collector
	gpus        []int           // GPU IDs to distribute model across
	apiKey      string          // optional; enables Bearer token auth
	rateLimiter *security.RateLimiter // optional; enables per-IP rate limiting
	maxTokens   int             // server-side upper bound for max_tokens (default 8192)
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

// WithLogger sets the logger for request logging.
func WithLogger(l log.Logger) ServerOption {
	return func(s *Server) {
		s.logger = l
	}
}

// WithMetrics sets the metrics collector for token rate and request tracking.
func WithMetrics(c runtime.Collector) ServerOption {
	return func(s *Server) {
		s.collector = c
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

// WithAPIKey enables Bearer token authentication on all endpoints
// except health checks (/healthz, /readyz), metrics (/metrics), and
// the OpenAPI spec (/openapi.yaml).
func WithAPIKey(key string) ServerOption {
	return func(s *Server) {
		s.apiKey = key
	}
}

// WithGPUs sets the GPU IDs to distribute the model across.
func WithGPUs(ids []int) ServerOption {
	return func(s *Server) {
		s.gpus = ids
	}
}

// WithMaxTokens sets the server-side upper bound for max_tokens in completion
// requests. Any request asking for more tokens than this limit will be clamped.
// The default is 8192.
func WithMaxTokens(n int) ServerOption {
	return func(s *Server) {
		s.maxTokens = n
	}
}

// WithRateLimiter enables per-IP rate limiting using the provided RateLimiter.
// When set, requests that exceed the rate limit receive 429 Too Many Requests.
func WithRateLimiter(rl *security.RateLimiter) ServerOption {
	return func(s *Server) {
		s.rateLimiter = rl
	}
}

// GPUs returns the configured GPU IDs, or nil if not set.
func (s *Server) GPUs() []int {
	return s.gpus
}

// NewServer creates a Server for the given model.
func NewServer(m *inference.Model, opts ...ServerOption) *Server {
	s := &Server{model: m, mux: http.NewServeMux(), maxTokens: 8192}
	for _, opt := range opts {
		opt(s)
	}
	if s.logger == nil {
		s.logger = log.Nop()
	}
	if s.collector == nil {
		s.collector = runtime.Nop()
	}
	// Auto-wire batch handler to use GenerateBatch when a BatchScheduler
	// is attached but no handler has been configured.
	if s.batch != nil && s.batch.config.Handler == nil {
		s.batch.config.Handler = func(ctx context.Context, reqs []BatchRequest) []BatchResult {
			prompts := make([]string, len(reqs))
			for i, r := range reqs {
				prompts[i] = r.Prompt
			}
			outputs, err := s.model.GenerateBatch(ctx, prompts)
			results := make([]BatchResult, len(reqs))
			for i := range results {
				if err != nil {
					results[i] = BatchResult{Err: err}
				} else {
					results[i] = BatchResult{Value: outputs[i]}
				}
			}
			return results
		}
	}
	s.metrics = NewServerMetrics(s.collector)
	s.classifyMetrics = NewClassifyMetrics(s.collector)
	s.mux.HandleFunc("POST /v1/chat/completions", s.recoveryMiddleware(s.handleChatCompletions))
	s.mux.HandleFunc("POST /v1/completions", s.recoveryMiddleware(s.handleCompletions))
	s.mux.HandleFunc("POST /v1/embeddings", s.recoveryMiddleware(s.handleEmbeddings))
	s.mux.HandleFunc("GET /v1/models", s.recoveryMiddleware(s.handleModels))
	s.mux.HandleFunc("GET /v1/models/{id...}", s.recoveryMiddleware(s.handleModelInfo))
	s.mux.HandleFunc("DELETE /v1/models/{id...}", s.recoveryMiddleware(s.handleModelDelete))
	s.mux.HandleFunc("POST /v1/audio/transcriptions", s.recoveryMiddleware(s.handleAudioTranscriptions))
	s.mux.HandleFunc("POST /v1/classify", s.recoveryMiddleware(s.handleClassify))
	s.mux.HandleFunc("GET /openapi.yaml", s.recoveryMiddleware(handleOpenAPISpec))
	s.mux.HandleFunc("GET /metrics", handleMetrics(s.collector))
	return s
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler {
	var h http.Handler = s.mux
	if s.apiKey != "" {
		h = s.authMiddleware(h)
	}
	if s.rateLimiter != nil {
		h = s.rateLimitMiddleware(h)
	}
	h = s.requestIDMiddleware(h)
	h = s.logMiddleware(h)
	return s.securityHeadersMiddleware(h)
}

// securityHeadersMiddleware sets standard security headers on every response.
func (s *Server) securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(w, r)
	})
}

// requestIDKey is the context key for the request ID.
type requestIDKey struct{}

// RequestID returns the request ID from the context, or an empty string if not set.
func RequestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey{}).(string)
	return id
}

// requestIDMiddleware reads X-Request-Id from the request header; if absent,
// it generates a UUID v4. The ID is stored in the request context and echoed
// back in the X-Request-Id response header.
func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = generateRequestID()
		}
		w.Header().Set("X-Request-Id", id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// generateRequestID produces a UUID v4 string using crypto/rand.
func generateRequestID() string {
	var buf [16]byte
	_, _ = rand.Read(buf[:])
	buf[6] = (buf[6] & 0x0f) | 0x40 // version 4
	buf[8] = (buf[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		buf[0:4], buf[4:6], buf[6:8], buf[8:10], buf[10:16])
}

// authMiddleware rejects requests that do not carry a valid Bearer token.
// Health-check, metrics, and OpenAPI spec paths are exempt.
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics", "/healthz", "/readyz", "/openapi.yaml":
			next.ServeHTTP(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		const prefix = "Bearer "
		if !strings.HasPrefix(auth, prefix) {
			writeError(w, http.StatusUnauthorized, "missing or invalid authorization header")
			return
		}
		token := auth[len(prefix):]
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.apiKey)) != 1 {
			writeError(w, http.StatusUnauthorized, "invalid API key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// rateLimitMiddleware rejects requests that exceed the configured rate limit
// for the client IP. Returns 429 Too Many Requests when the limit is exceeded.
func (s *Server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := security.ClientIP(r)
		if !s.rateLimiter.Allow(ip) {
			writeError(w, http.StatusTooManyRequests, "rate limit exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// statusRecorder wraps http.ResponseWriter to capture the status code.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		latency := time.Since(start).Milliseconds()

		modelID := ""
		if info := s.model.Info(); info != nil {
			modelID = info.ID
		}

		fields := []string{
			"method", r.Method,
			"path", r.URL.Path,
			"model", modelID,
			"prompt_tokens", "0",
			"completion_tokens", "0",
			"latency_ms", strconv.FormatInt(latency, 10),
			"status_code", strconv.Itoa(rec.status),
		}

		switch {
		case rec.status >= 500:
			s.logger.Error("request completed", fields...)
		case rec.status >= 400:
			s.logger.Warn("request completed", fields...)
		default:
			s.logger.Info("request completed", fields...)
		}
	})
}

func (s *Server) recoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				msg := fmt.Sprintf("%v", rec)
				s.logger.Error("panic recovered", "error", msg, "method", r.Method, "path", r.URL.Path)
				fmt.Fprintf(os.Stderr, "panic recovered: %s %s: %s\n", r.Method, r.URL.Path, msg)
				writeError(w, http.StatusInternalServerError, "internal server error")
			}
		}()
		next(w, r)
	}
}

// isOOMError reports whether the error message indicates an out-of-memory condition.
func isOOMError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "out of memory") ||
		strings.Contains(msg, "oom") ||
		strings.Contains(msg, "cannot allocate")
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

// --- Handlers ---

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.inflight.Add(1)
	defer s.inflight.Done()

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required")
		return
	}

	// Validate tools and tool_choice.
	if len(req.Tools) > 0 {
		if err := validateTools(req.Tools); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.logger.Debug("chat request includes tools", "tool_count", strconv.Itoa(len(req.Tools)))
	}
	if err := validateToolChoice(req.ToolChoice, req.Tools); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Clamp max_tokens to server-side upper bound.
	if req.MaxTokens != nil && *req.MaxTokens > s.maxTokens {
		clamped := s.maxTokens
		req.MaxTokens = &clamped
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

	// Wire response_format json_schema into grammar-constrained decoding.
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" && req.ResponseFormat.JSONSchema != nil {
		var schema grammar.JSONSchema
		if err := json.Unmarshal(req.ResponseFormat.JSONSchema.Schema, &schema); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json_schema: "+err.Error())
			return
		}
		g, err := grammar.Convert(&schema)
		if err != nil {
			writeError(w, http.StatusBadRequest, "unsupported schema: "+err.Error())
			return
		}
		opts = append(opts, inference.WithGrammar(g))
	}

	// Convert messages and fetch any images.
	messages := make([]inference.Message, len(req.Messages))
	for i, m := range req.Messages {
		messages[i] = inference.Message{Role: m.Role, Content: m.Content}
		if len(m.ImageURLs) > 0 {
			images, err := fetchImages(r.Context(), m.ImageURLs)
			if err != nil {
				writeError(w, http.StatusBadRequest, "image fetch error: "+err.Error())
				return
			}
			messages[i].Images = images
		}
	}

	if req.Stream {
		s.streamChatCompletion(w, r.Context(), messages, opts)
		return
	}

	start := time.Now()
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
		writeError(w, inferenceErrorStatus(err), err.Error())
		return
	}

	s.metrics.RecordRequest(resp.CompletionTokens, time.Since(start))

	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	// Check for tool calls if tools were provided.
	choice := ChatCompletionChoice{
		Index:        0,
		Message:      ChatMessage{Role: "assistant", Content: resp.Content},
		FinishReason: "stop",
	}

	if len(req.Tools) > 0 {
		tc := ToolChoice{Mode: "auto"}
		if req.ToolChoice != nil {
			tc = *req.ToolChoice
		}

		var toolResult *ToolCallResult
		if result, ok := DetectToolCall(resp.Content, req.Tools, tc); ok {
			toolResult = result
		} else if tc.Mode == "function" && tc.Function != nil {
			// Forced tool choice: always return a tool call for the specified function.
			args := json.RawMessage("{}")
			if json.Valid([]byte(resp.Content)) {
				args = json.RawMessage(resp.Content)
			}
			toolResult = &ToolCallResult{
				ID:           generateCallID(),
				FunctionName: tc.Function.Name,
				Arguments:    args,
			}
		}

		if toolResult != nil {
			choice.Message.Content = ""
			choice.FinishReason = "tool_calls"
			choice.ToolCalls = []ToolCall{{
				ID:   toolResult.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      toolResult.FunctionName,
					Arguments: string(toolResult.Arguments),
				},
			}}
		}
	}

	writeJSON(w, http.StatusOK, ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelID,
		Choices: []ChatCompletionChoice{choice},
		Usage: UsageInfo{
			PromptTokens:     resp.PromptTokens,
			CompletionTokens: resp.CompletionTokens,
			TotalTokens:      resp.TokensUsed,
		},
	})
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.inflight.Add(1)
	defer s.inflight.Done()

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	// Clamp max_tokens to server-side upper bound.
	if req.MaxTokens != nil && *req.MaxTokens > s.maxTokens {
		clamped := s.maxTokens
		req.MaxTokens = &clamped
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

	start := time.Now()
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
		writeError(w, inferenceErrorStatus(err), err.Error())
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

	s.metrics.RecordRequest(completionTokens, time.Since(start))

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
	if s.unloaded.Load() {
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

	if s.unloaded.Load() {
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

	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	obj := s.buildModelObject()
	if obj.ID != id {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	_ = s.model.Close()
	s.unloaded.Store(true)

	writeJSON(w, http.StatusOK, ModelDeleteResponse{
		ID:      id,
		Object:  "model",
		Deleted: true,
	})
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
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
		emb, err := s.model.Embed(text)
		if err != nil {
			writeError(w, inferenceErrorStatus(err), err.Error())
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

func handleOpenAPISpec(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	w.WriteHeader(http.StatusOK)
	w.Write(openapiSpec) //nolint:errcheck
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

// Close implements shutdown.Closer for graceful shutdown integration.
func (s *Server) Close(_ context.Context) error {
	if s.batch != nil {
		s.batch.Stop()
	}
	return nil
}
