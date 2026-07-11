package serve

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/inference/guardian"
)

// echoHandler writes a fixed ChatCompletionResponse so the middleware has
// something to inspect when CheckOutput is enabled.
func echoHandler(content string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := ChatCompletionResponse{
			ID:     "chatcmpl-test",
			Object: "chat.completion",
			Model:  "test-model",
			Choices: []ChatCompletionChoice{
				{Index: 0, Message: ChatMessage{Role: "assistant", Content: content}, FinishReason: "stop"},
			},
		}
		writeJSON(w, http.StatusOK, resp)
	})
}

func chatBody(t *testing.T, userMsg string) *bytes.Buffer {
	t.Helper()
	req := ChatCompletionRequest{
		Model:    "test-model",
		Messages: []ChatMessage{{Role: "user", Content: userMsg}},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.NewBuffer(b)
}

func TestGuardianMiddleware_SafeInputPassesThrough(t *testing.T) {
	eval := &mockGuardEvaluator{} // default: all safe
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: true,
	})

	inner := echoHandler("hello world")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "hi"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp ChatCompletionResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Choices[0].Message.Content != "hello world" {
		t.Fatalf("unexpected content: %s", resp.Choices[0].Message.Content)
	}
}

func TestGuardianMiddleware_HarmfulInputBlocked(t *testing.T) {
	eval := &mockGuardEvaluator{
		verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: true, Confidence: 0.95, Reasoning: "harmful content detected"},
		},
	}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: true,
	})

	inner := echoHandler("should not reach")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "bad stuff"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp guardianFlaggedResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Error != "content_flagged" {
		t.Fatalf("expected error content_flagged, got %q", resp.Error)
	}
	if len(resp.Verdicts) != 1 || !resp.Verdicts[0].Unsafe {
		t.Fatalf("expected one unsafe verdict, got %+v", resp.Verdicts)
	}
}

func TestGuardianMiddleware_CheckInputFalseSkipsScanning(t *testing.T) {
	eval := &mockGuardEvaluator{
		verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: true, Confidence: 0.95},
		},
	}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  false,
		BlockOnFlag: true,
	})

	inner := echoHandler("passed through")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "bad stuff"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 (input scanning disabled), got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestGuardianMiddleware_CheckOutputScansAssistantResponse(t *testing.T) {
	// Input is safe, but output evaluator will flag it.
	callCount := 0
	eval := &mockGuardEvaluator{}
	// Override Evaluate to flag only when assistant content is present (output check).
	origEval := &outputFlaggingEvaluator{callCount: &callCount}

	mw := GuardianMiddleware(origEval, GuardianMiddlewareConfig{
		CheckInput:  true,
		CheckOutput: true,
		BlockOnFlag: true,
	})

	inner := echoHandler("toxic response")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "hello"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	// The output check should have flagged it.
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 from output check, got %d: %s", rec.Code, rec.Body.String())
	}

	_ = eval // used for reference only
}

// outputFlaggingEvaluator returns safe for input checks (no assistant) and
// unsafe for output checks (assistant present).
type outputFlaggingEvaluator struct {
	callCount *int
}

func (e *outputFlaggingEvaluator) Evaluate(_ context.Context, req guardian.GuardianRequest) ([]guardian.Verdict, error) {
	*e.callCount++
	if req.Input.Assistant != "" {
		return []guardian.Verdict{{Risk: "harm", Unsafe: true, Confidence: 0.99}}, nil
	}
	return []guardian.Verdict{{Risk: "harm", Unsafe: false, Confidence: 0.01}}, nil
}

func (e *outputFlaggingEvaluator) EvaluateBatch(_ context.Context, _ []guardian.GuardianInput, _ []string) (*guardian.BatchResult, error) {
	return &guardian.BatchResult{}, nil
}

func (e *outputFlaggingEvaluator) Scan(_ context.Context, _ guardian.GuardianInput) (*guardian.ScanResult, error) {
	return &guardian.ScanResult{}, nil
}

func TestGuardianMiddleware_BlockOnFlagFalseAllowsFlaggedContent(t *testing.T) {
	eval := &mockGuardEvaluator{
		verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: true, Confidence: 0.95},
		},
	}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: false,
	})

	inner := echoHandler("allowed through")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "bad stuff"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 (BlockOnFlag=false), got %d: %s", rec.Code, rec.Body.String())
	}

	if got := rec.Header().Get("X-Guardian-Flagged"); got != "true" {
		t.Fatalf("expected X-Guardian-Flagged: true, got %q", got)
	}
}

func TestGuardianMiddleware_NilEvaluatorPassesThrough(t *testing.T) {
	mw := GuardianMiddleware(nil, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: true,
	})

	inner := echoHandler("no guardian")
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "anything"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 (nil evaluator), got %d: %s", rec.Code, rec.Body.String())
	}
}

// TestGuardianMiddleware_OverCapRequestBodyRejected verifies SERVE-4: the
// middleware bounds its request body read with http.MaxBytesReader instead
// of doing an unbounded io.ReadAll, so an oversized request is rejected
// rather than buffered into memory.
func TestGuardianMiddleware_OverCapRequestBodyRejected(t *testing.T) {
	eval := &mockGuardEvaluator{}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: true,
	})

	called := false
	inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})
	handler := mw(inner)

	// Pad well past the 10 MB cap shared with the rest of serve/.
	bigValue := strings.Repeat("x", guardianMaxRequestBodyBytes+(1<<20))
	oversized := `{"model":"test-model","messages":[{"role":"user","content":"` + bigValue + `"}]}`

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(oversized))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected %d, got %d: %s", http.StatusRequestEntityTooLarge, rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("inner handler should not run for an over-cap request body")
	}
}

// trackingFlushWriter is a minimal http.ResponseWriter + http.Flusher that
// records every Write/Flush as it happens, so tests can observe whether
// bytes reached the "client" before a streaming handler finished -- i.e.
// that GuardianMiddleware isn't buffering the whole response first.
type trackingFlushWriter struct {
	mu      sync.Mutex
	header  http.Header
	buf     bytes.Buffer
	status  int
	flushes int
}

func newTrackingFlushWriter() *trackingFlushWriter {
	return &trackingFlushWriter{header: make(http.Header)}
}

func (w *trackingFlushWriter) Header() http.Header { return w.header }

func (w *trackingFlushWriter) WriteHeader(code int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.status = code
}

func (w *trackingFlushWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(b)
}

func (w *trackingFlushWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.flushes++
}

func (w *trackingFlushWriter) snapshot() (string, int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.String(), w.flushes
}

// TestGuardianMiddleware_SSEStreamsThroughWithoutBuffering verifies SERVE-4:
// when the inner handler emits a text/event-stream response, the middleware
// must not swallow/delay it by buffering the entire response before
// forwarding -- each chunk must reach the client (and be flushed) as soon as
// the inner handler writes it.
func TestGuardianMiddleware_SSEStreamsThroughWithoutBuffering(t *testing.T) {
	firstChunkWritten := make(chan struct{})
	proceed := make(chan struct{})

	inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		if _, err := w.Write([]byte("data: chunk1\n\n")); err != nil {
			t.Errorf("write chunk1: %v", err)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(firstChunkWritten)

		<-proceed

		if _, err := w.Write([]byte("data: chunk2\n\n")); err != nil {
			t.Errorf("write chunk2: %v", err)
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	})

	eval := &mockGuardEvaluator{}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		CheckOutput: true,
		BlockOnFlag: true,
	})
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, "hi"))
	rw := newTrackingFlushWriter()

	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(rw, req)
		close(done)
	}()

	select {
	case <-firstChunkWritten:
	case <-done:
		t.Fatal("handler returned before writing the first SSE chunk")
	}

	// The first chunk (and a flush) must already be visible on the real
	// ResponseWriter even though the handler hasn't returned yet -- proof
	// the middleware is not buffering the whole response.
	got, flushes := rw.snapshot()
	if !strings.Contains(got, "chunk1") {
		t.Fatalf("expected first SSE chunk to stream through immediately, got %q", got)
	}
	if strings.Contains(got, "chunk2") {
		t.Fatalf("second chunk should not have been written yet, got %q", got)
	}
	if flushes == 0 {
		t.Fatal("expected the flush to propagate to the underlying ResponseWriter")
	}

	close(proceed)
	<-done

	final, _ := rw.snapshot()
	if !strings.Contains(final, "chunk1") || !strings.Contains(final, "chunk2") {
		t.Fatalf("expected both SSE chunks in the final output, got %q", final)
	}
}

func TestGuardianMiddleware_NonChatPathPassesThrough(t *testing.T) {
	eval := &mockGuardEvaluator{
		verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: true, Confidence: 0.95},
		},
	}
	mw := GuardianMiddleware(eval, GuardianMiddlewareConfig{
		CheckInput:  true,
		BlockOnFlag: true,
	})

	called := false
	inner := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})
	handler := mw(inner)

	req := httptest.NewRequest(http.MethodPost, "/v1/completions", chatBody(t, "bad stuff"))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if !called {
		t.Fatal("inner handler was not called for non-chat path")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for non-chat path, got %d", rec.Code)
	}
}
