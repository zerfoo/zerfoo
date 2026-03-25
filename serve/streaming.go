// Package serve SSE streaming implementations for chat and text completions.
package serve

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

func (s *Server) streamChatCompletion(w http.ResponseWriter, ctx context.Context, messages []inference.Message, opts []inference.GenerateOption) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	// Use the model's chat template to format messages, matching the
	// non-streaming Chat() path which calls model.formatMessages().
	err := s.model.ChatStream(ctx, messages, generate.TokenStreamFunc(func(token string, done bool) error {
		if done {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
		chunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   modelID,
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
		_, _ = fmt.Fprintf(w, "data: {\"error\": %q}\n\n", s.sanitizeError(err))
		flusher.Flush()
	}
}

func (s *Server) streamCompletion(w http.ResponseWriter, ctx context.Context, prompt string, opts []inference.GenerateOption) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	modelID := ""
	if info := s.model.Info(); info != nil {
		modelID = info.ID
	}

	err := s.model.GenerateStream(ctx, prompt, generate.TokenStreamFunc(func(token string, done bool) error {
		if done {
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return nil
		}
		chunk := map[string]interface{}{
			"id":      id,
			"object":  "text_completion",
			"created": created,
			"model":   modelID,
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
		_, _ = fmt.Fprintf(w, "data: {\"error\": %q}\n\n", s.sanitizeError(err))
		flusher.Flush()
	}
}
