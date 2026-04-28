package langchain_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zerfoo/zerfoo/sdk/integrations/langchain"
)

// fakeChatServer returns a test HTTP server that responds with a fixed
// chat completion payload.
func fakeChatServer(t *testing.T, reply string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": reply}, "finish_reason": "stop"},
			},
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
}

func TestLangChainAdapter_Generate(t *testing.T) {
	want := "Paris is the capital of France."
	srv := fakeChatServer(t, want)
	defer srv.Close()

	adapter := langchain.NewAdapter(srv.URL, "test-model")

	got, err := adapter.Call(context.Background(), "What is the capital of France?")
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestLangChainAdapter_Generate_MultiplePrompts(t *testing.T) {
	srv := fakeChatServer(t, "pong")
	defer srv.Close()

	adapter := langchain.NewAdapter(srv.URL, "test-model")

	prompts := []string{"ping", "ping", "ping"}
	results, err := adapter.Generate(context.Background(), prompts)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(results) != len(prompts) {
		t.Fatalf("want %d results, got %d", len(prompts), len(results))
	}
	for i, r := range results {
		if r != "pong" {
			t.Errorf("result[%d]: got %q, want %q", i, r, "pong")
		}
	}
}

func TestLangChainAdapter_Generate_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"type":"server_error","message":"oops"}}`, http.StatusInternalServerError)
	}))
	defer srv.Close()

	adapter := langchain.NewAdapter(srv.URL, "test-model")
	_, err := adapter.Call(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error from server 500, got nil")
	}
}

func TestLangChainAdapter_Generate_StopWords(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	adapter := langchain.NewAdapter(srv.URL, "test-model")
	_, err := adapter.Call(context.Background(), "prompt", "\n", "END")
	if err != nil {
		t.Fatalf("Call: %v", err)
	}

	stopRaw, ok := capturedBody["stop"]
	if !ok {
		t.Fatal("stop field missing from request")
	}
	stops, _ := stopRaw.([]any)
	if len(stops) != 2 {
		t.Errorf("expected 2 stop words, got %v", stops)
	}
}

func TestLangChainAdapter_Type(t *testing.T) {
	a := langchain.NewAdapter("http://localhost:8080", "model")
	if got := a.Type(); got != "zerfoo" {
		t.Errorf("Type() = %q, want %q", got, "zerfoo")
	}
}
