package cloud

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
)

func TestNDJSONRecorder(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	event := UsageEvent{
		TenantID:         "tenant-1",
		Model:            "llama3-8b",
		PromptTokens:     100,
		CompletionTokens: 50,
		Timestamp:        1700000000,
	}

	if err := rec.Record(event); err != nil {
		t.Fatalf("Record() error = %v", err)
	}

	line := strings.TrimSpace(buf.String())
	var got UsageEvent
	if err := json.Unmarshal([]byte(line), &got); err != nil {
		t.Fatalf("unmarshal recorded line: %v", err)
	}
	if got.TenantID != "tenant-1" {
		t.Errorf("TenantID = %q, want %q", got.TenantID, "tenant-1")
	}
	if got.Model != "llama3-8b" {
		t.Errorf("Model = %q, want %q", got.Model, "llama3-8b")
	}
	if got.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", got.PromptTokens)
	}
	if got.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", got.CompletionTokens)
	}
}

func TestNDJSONRecorderMultipleEvents(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	for i := 0; i < 3; i++ {
		if err := rec.Record(UsageEvent{
			TenantID:         "tenant-1",
			Model:            "llama3-8b",
			PromptTokens:     10 * (i + 1),
			CompletionTokens: 5 * (i + 1),
			Timestamp:        1700000000 + int64(i),
		}); err != nil {
			t.Fatalf("Record() error = %v", err)
		}
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3", len(lines))
	}

	for i, line := range lines {
		var got UsageEvent
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("line %d: unmarshal error: %v", i, err)
		}
		if got.PromptTokens != 10*(i+1) {
			t.Errorf("line %d: PromptTokens = %d, want %d", i, got.PromptTokens, 10*(i+1))
		}
	}
}

func TestNDJSONRecorderConcurrent(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			rec.Record(UsageEvent{
				TenantID:     "tenant-1",
				PromptTokens: n,
				Timestamp:    1700000000,
			})
		}(i)
	}
	wg.Wait()

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 50 {
		t.Errorf("got %d lines, want 50", len(lines))
	}
}

// fakeUsageHandler returns a handler that responds with a JSON body containing usage info.
func fakeUsageHandler(promptTokens, completionTokens int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "llama3-8b",
			"choices": []map[string]interface{}{},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": completionTokens,
				"total_tokens":      promptTokens + completionTokens,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})
}

// newTestTenantContext creates a context with a Tenant for testing BillingMiddleware.
func newTestTenantContext(ctx context.Context) context.Context {
	tenant := &Tenant{ID: "test-tenant"}
	tenant.rateLimit.Store(100)
	tenant.tokenBudget.Store(100000)
	return context.WithValue(ctx, tenantKey{}, tenant)
}

func TestBillingMiddleware(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(100, 50)
	handler := BillingMiddleware(rec)(inner)

	reqBody := `{"model":"llama3-8b","messages":[{"role":"user","content":"hello"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer key-abc")
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	// Verify the usage event was recorded.
	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("no usage event recorded")
	}

	var event UsageEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	wantHash := sha256Hex("key-abc")
	if event.TenantID != wantHash {
		t.Errorf("TenantID = %q, want SHA-256 hash %q", event.TenantID, wantHash)
	}
	if event.TenantID == "key-abc" {
		t.Error("TenantID must not contain the raw API key")
	}
	if event.Model != "llama3-8b" {
		t.Errorf("Model = %q, want %q", event.Model, "llama3-8b")
	}
	if event.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", event.PromptTokens)
	}
	if event.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", event.CompletionTokens)
	}
	if event.Timestamp == 0 {
		t.Error("Timestamp should not be zero")
	}
}

func TestBillingMiddlewareNoTenant(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(100, 50)
	handler := BillingMiddleware(rec)(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3-8b"}`))
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Fail closed: missing tenant must return 500, not pass through.
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusInternalServerError)
	}

	// Verify the error body is a JSON object with an error message.
	var errResp struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("response body is not valid JSON: %v", err)
	}
	if errResp.Error.Message != "billing: tenant context required" {
		t.Errorf("error message = %q, want %q", errResp.Error.Message, "billing: tenant context required")
	}

	// No event should be recorded when there's no tenant.
	if buf.Len() != 0 {
		t.Errorf("expected no event without tenant, got: %s", buf.String())
	}
}

func TestBillingMiddlewareZeroUsage(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(0, 0)
	handler := BillingMiddleware(rec)(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3-8b"}`))
	req.Header.Set("Authorization", "Bearer key-abc")
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// No event for zero usage.
	if buf.Len() != 0 {
		t.Errorf("expected no event for zero usage, got: %s", buf.String())
	}
}

func TestBillingMiddlewareResponsePassthrough(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(20, 10)
	handler := BillingMiddleware(rec)(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3-8b"}`))
	req.Header.Set("Authorization", "Bearer key-abc")
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	// Verify the original response body is passed through to the client.
	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response body is not valid JSON: %v", err)
	}
	if resp["id"] != "chatcmpl-123" {
		t.Errorf("response id = %v, want chatcmpl-123", resp["id"])
	}
}

func TestBillingMiddlewareStreamingSSE(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	// Simulate a streaming handler that writes SSE chunks and records
	// token usage via the context-based TokenUsage.
	sseHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := range 5 {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"word%d\"}}]}\n\n", i)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")

		// Generation layer records usage via context.
		if usage := generate.TokenUsageFromContext(r.Context()); usage != nil {
			usage.SetPromptTokens(128)
			usage.SetCompletionTokens(5)
		}
	})

	handler := BillingMiddleware(rec)(sseHandler)

	reqBody := `{"model":"llama3-8b","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer key-stream")
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("no usage event recorded for streaming request")
	}

	var event UsageEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	wantHash := sha256Hex("key-stream")
	if event.TenantID != wantHash {
		t.Errorf("TenantID = %q, want SHA-256 hash %q", event.TenantID, wantHash)
	}
	if event.TenantID == "key-stream" {
		t.Error("TenantID must not contain the raw API key")
	}
	if event.Model != "llama3-8b" {
		t.Errorf("Model = %q, want %q", event.Model, "llama3-8b")
	}
	if event.PromptTokens != 128 {
		t.Errorf("PromptTokens = %d, want 128", event.PromptTokens)
	}
	if event.CompletionTokens != 5 {
		t.Errorf("CompletionTokens = %d, want 5", event.CompletionTokens)
	}
}

func TestBillingMiddlewareNonStreamingJSONFallback(t *testing.T) {
	// Verify that non-streaming JSON responses still produce billing records
	// via the JSON body fallback path (handler does not use context-based usage).
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(200, 75)
	handler := BillingMiddleware(rec)(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gemma-3b"}`))
	req.Header.Set("Authorization", "Bearer key-fallback")
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("no usage event recorded for non-streaming request")
	}

	var event UsageEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}
	if event.PromptTokens != 200 {
		t.Errorf("PromptTokens = %d, want 200", event.PromptTokens)
	}
	if event.CompletionTokens != 75 {
		t.Errorf("CompletionTokens = %d, want 75", event.CompletionTokens)
	}
}

func TestBillingMiddlewareBodyLimit(t *testing.T) {
	t.Run("oversized body is truncated at 10MB", func(t *testing.T) {
		var bodyLen int
		inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			bodyLen = len(b)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"usage": map[string]int{
					"prompt_tokens":     10,
					"completion_tokens": 5,
					"total_tokens":      15,
				},
			})
		})

		var buf bytes.Buffer
		rec := NewNDJSONRecorder(&buf)
		handler := BillingMiddleware(rec)(inner)

		bigBody := strings.NewReader(strings.Repeat("x", 11<<20))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bigBody)
		req.Header.Set("Authorization", "Bearer key-big")
		req = req.WithContext(newTestTenantContext(req.Context()))

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if bodyLen > 10<<20 {
			t.Errorf("downstream saw %d bytes, want <= %d", bodyLen, 10<<20)
		}
	})

	t.Run("normal request body still works", func(t *testing.T) {
		var buf bytes.Buffer
		rec := NewNDJSONRecorder(&buf)

		inner := fakeUsageHandler(10, 5)
		handler := BillingMiddleware(rec)(inner)

		reqBody := `{"model":"llama3-8b","messages":[{"role":"user","content":"hello"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody))
		req.Header.Set("Authorization", "Bearer key-normal")
		req = req.WithContext(newTestTenantContext(req.Context()))

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
		}

		line := strings.TrimSpace(buf.String())
		if line == "" {
			t.Fatal("no usage event recorded")
		}

		var event UsageEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("unmarshal event: %v", err)
		}
		if event.PromptTokens != 10 {
			t.Errorf("PromptTokens = %d, want 10", event.PromptTokens)
		}
	})
}

// sha256Hex returns the hex-encoded SHA-256 hash of s.
func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func TestBillingMiddlewareTenantIDIsHashed(t *testing.T) {
	var buf bytes.Buffer
	rec := NewNDJSONRecorder(&buf)

	inner := fakeUsageHandler(42, 18)
	handler := BillingMiddleware(rec)(inner)

	const rawKey = "sk-secret-api-key-12345"
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3-8b"}`))
	req.Header.Set("Authorization", "Bearer "+rawKey)
	req = req.WithContext(newTestTenantContext(req.Context()))

	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("no usage event recorded")
	}

	var event UsageEvent
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		t.Fatalf("unmarshal event: %v", err)
	}

	// The tenant ID must NOT be the raw API key.
	if event.TenantID == rawKey {
		t.Fatal("TenantID contains raw API key — credentials must be hashed before recording")
	}

	// The tenant ID must be the SHA-256 hex digest of the raw key.
	want := sha256Hex(rawKey)
	if event.TenantID != want {
		t.Errorf("TenantID = %q, want SHA-256 hash %q", event.TenantID, want)
	}

	// Verify it is a valid 64-character hex string (SHA-256 output).
	if len(event.TenantID) != 64 {
		t.Errorf("TenantID length = %d, want 64 (hex-encoded SHA-256)", len(event.TenantID))
	}
}

// failingRecorder is a UsageRecorder that always returns an error.
type failingRecorder struct {
	err error
}

func (f *failingRecorder) Record(event UsageEvent) error {
	return f.err
}

func TestBillingMiddlewareRecordErrorIsLogged(t *testing.T) {
	rec := &failingRecorder{err: errors.New("kafka: connection refused")}

	inner := fakeUsageHandler(100, 50)
	handler := BillingMiddleware(rec)(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3-8b"}`))
	req.Header.Set("Authorization", "Bearer key-fail")
	req = req.WithContext(newTestTenantContext(req.Context()))

	// Capture stderr to verify the error is logged.
	origStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	w.Close()
	os.Stderr = origStderr

	var stderrBuf bytes.Buffer
	io.Copy(&stderrBuf, r)
	r.Close()

	output := stderrBuf.String()
	if !strings.Contains(output, "billing: failed to record usage event") {
		t.Errorf("expected billing error log on stderr, got: %q", output)
	}
	if !strings.Contains(output, "kafka: connection refused") {
		t.Errorf("expected error message in log, got: %q", output)
	}
	if !strings.Contains(output, "model=llama3-8b") {
		t.Errorf("expected model in log, got: %q", output)
	}
	if !strings.Contains(output, "prompt=100") {
		t.Errorf("expected prompt token count in log, got: %q", output)
	}
	if !strings.Contains(output, "completion=50") {
		t.Errorf("expected completion token count in log, got: %q", output)
	}
}
