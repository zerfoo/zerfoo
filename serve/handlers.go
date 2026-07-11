// Package serve HTTP handler implementations for OpenAI-compatible API endpoints.
package serve

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
)

// --- Handlers ---

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required")
		return
	}

	// Validate sampling parameters.
	if err := validateSamplingParams(req.Temperature, req.TopP, req.TopK); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validateMaxTokens(req.MaxTokens); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
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

	// Parse adapter from model field ("base:adapter" format).
	_, adapterName := ParseModelAdapter(req.Model)
	if adapterName != "" && s.adapterCache == nil {
		writeError(w, http.StatusBadRequest, "adapter selection not enabled")
		return
	}
	if adapterName != "" {
		if _, err := s.adapterCache.resolveAdapter(adapterName); err != nil {
			writeError(w, http.StatusBadRequest, "adapter not found: "+adapterName)
			return
		}
	}

	// Build generation options.
	opts := buildGenerationOptions(samplingParams{
		AdapterName: adapterName,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		MaxTokens:   req.MaxTokens,
	})

	// Wire response_format json_schema into grammar-constrained decoding.
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_schema" && req.ResponseFormat.JSONSchema != nil {
		grammarOpt, err := parseAndApplyGrammar(req.ResponseFormat.JSONSchema.Schema)
		if err != nil {
			s.logger.Debug("grammar error", "error", err.Error())
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		opts = append(opts, grammarOpt)
	}

	// Convert messages and fetch any images.
	messages := make([]inference.Message, len(req.Messages))
	for i, m := range req.Messages {
		messages[i] = inference.Message{Role: m.Role, Content: m.Content}
		if len(m.ImageURLs) > 0 {
			images, err := fetchImages(r.Context(), m.ImageURLs)
			if err != nil {
				s.logger.Debug("image fetch error", "error", err.Error())
				writeError(w, http.StatusBadRequest, "image fetch failed")
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
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
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

	detectAndFormatToolCalls(&choice, resp.Content, req.Tools, req.ToolChoice)

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
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}

	// Validate sampling parameters.
	if err := validateSamplingParams(req.Temperature, req.TopP, req.TopK); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := validateMaxTokens(req.MaxTokens); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Clamp max_tokens to server-side upper bound.
	if req.MaxTokens != nil && *req.MaxTokens > s.maxTokens {
		clamped := s.maxTokens
		req.MaxTokens = &clamped
	}

	// Parse adapter from model field ("base:adapter" format).
	_, complAdapterName := ParseModelAdapter(req.Model)
	if complAdapterName != "" && s.adapterCache == nil {
		writeError(w, http.StatusBadRequest, "adapter selection not enabled")
		return
	}
	if complAdapterName != "" {
		if _, err := s.adapterCache.resolveAdapter(complAdapterName); err != nil {
			writeError(w, http.StatusBadRequest, "adapter not found: "+complAdapterName)
			return
		}
	}

	opts := buildGenerationOptions(samplingParams{
		AdapterName: complAdapterName,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		TopK:        req.TopK,
		MaxTokens:   req.MaxTokens,
	})

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
		writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
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

	// Take the write side of modelMu: this blocks until every handler
	// currently holding RLock (i.e. every in-flight request that already
	// passed its unloaded check) has released it, then excludes any new
	// RLock acquisition until unloaded is set and the model is closed. This
	// closes both CONC-H2 races structurally: no handler can observe a model
	// that Close() is concurrently tearing down, and there is no
	// WaitGroup-style counter for a request to race against.
	s.modelMu.Lock()
	defer s.modelMu.Unlock()

	// Re-check under the lock: a concurrent delete may have already won the
	// race between the initial unloaded.Load() above and acquiring the lock.
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model '"+id+"' not found")
		return
	}

	s.unloaded.Store(true)
	_ = s.model.Close()

	writeJSON(w, http.StatusOK, ModelDeleteResponse{
		ID:      id,
		Object:  "model",
		Deleted: true,
	})
}

// maxEmbeddingsBatch is the maximum number of inputs in a single embeddings
// request. Without a cap, a 10 MB body of short strings can drive tens of
// thousands of synchronous Embed calls (SERVE-3).
const maxEmbeddingsBatch = 256

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.modelMu.RLock()
	defer s.modelMu.RUnlock()
	if s.unloaded.Load() {
		writeError(w, http.StatusNotFound, "model not available")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if isMaxBytesError(err) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		s.logger.Debug("invalid request body", "error", err.Error())
		writeError(w, http.StatusBadRequest, "invalid request body")
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

	if len(inputs) > maxEmbeddingsBatch {
		writeError(w, http.StatusBadRequest, "input exceeds maximum batch size of 256")
		return
	}

	var data []EmbeddingObject
	var totalTokens int
	for i, text := range inputs {
		emb, err := s.model.Embed(text)
		if err != nil {
			writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
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

// samplingParams holds the common sampling parameters shared across chat
// completion and text completion requests.
type samplingParams struct {
	AdapterName string
	Temperature *float64
	TopP        *float64
	TopK        *int
	MaxTokens   *int
}

// buildGenerationOptions converts sampling parameters into a slice of
// inference.GenerateOption values suitable for passing to model methods.
func buildGenerationOptions(p samplingParams) []inference.GenerateOption {
	var opts []inference.GenerateOption
	if p.AdapterName != "" {
		opts = append(opts, inference.WithAdapter(p.AdapterName))
	}
	if p.Temperature != nil {
		opts = append(opts, inference.WithTemperature(*p.Temperature))
	}
	if p.TopP != nil {
		opts = append(opts, inference.WithTopP(*p.TopP))
	}
	if p.TopK != nil {
		opts = append(opts, inference.WithTopK(*p.TopK))
	}
	if p.MaxTokens != nil {
		opts = append(opts, inference.WithMaxTokens(*p.MaxTokens))
	}
	return opts
}

// parseAndApplyGrammar unmarshals a JSON Schema from raw bytes and converts
// it into a grammar-constrained decoding option.
func parseAndApplyGrammar(schemaBytes json.RawMessage) (inference.GenerateOption, error) {
	var schema grammar.JSONSchema
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil, fmt.Errorf("invalid json_schema: %w", err)
	}
	g, err := grammar.Convert(&schema)
	if err != nil {
		return nil, fmt.Errorf("unsupported schema: %w", err)
	}
	return inference.WithGrammar(g), nil
}

func handleOpenAPISpec(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	w.WriteHeader(http.StatusOK)
	w.Write(openapiSpec) //nolint:errcheck
}

// detectAndFormatToolCalls examines generated content for tool calls and, if
// detected, rewrites the choice to use finish_reason "tool_calls" with the
// appropriate ToolCall slice. It handles auto-detection via DetectToolCall and
// the forced tool_choice fallback.
func detectAndFormatToolCalls(choice *ChatCompletionChoice, content string, tools []Tool, choicePtr *ToolChoice) {
	if len(tools) == 0 {
		return
	}

	tc := ToolChoice{Mode: "auto"}
	if choicePtr != nil {
		tc = *choicePtr
	}

	var toolResult *ToolCallResult
	if result, ok := DetectToolCall(content, tools, tc); ok {
		toolResult = result
	} else if tc.Mode == "function" && tc.Function != nil {
		// Forced tool choice: always return a tool call for the specified function.
		args := json.RawMessage("{}")
		if json.Valid([]byte(content)) {
			args = json.RawMessage(content)
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

// handleHealthz returns 200 with {"status":"ok"} to indicate the server is alive.
func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"}) //nolint:errcheck
}

// handleReadyz returns 200 with {"status":"ready"} if the model is loaded and
// has not been unloaded, or 503 with {"status":"not ready"} otherwise.
func (s *Server) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if s.model == nil || s.unloaded.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{"status": "not ready"}) //nolint:errcheck
		return
	}
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ready"}) //nolint:errcheck
}
