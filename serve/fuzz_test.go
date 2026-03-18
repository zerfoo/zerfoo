package serve

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func FuzzChatCompletionRequest(f *testing.F) {
	// Seed 1: minimal valid chat completion request.
	f.Add(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)

	// Seed 2: request with optional fields.
	f.Add(`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.9,"max_tokens":100,"stream":false}`)

	// Seed 3: empty object.
	f.Add(`{}`)

	// Seed 4: empty messages array.
	f.Add(`{"model":"m","messages":[]}`)

	// Seed 5: not JSON.
	f.Add(`not json at all`)

	// Seed 6: deeply nested.
	f.Add(`{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"test","schema":{"type":"object"}}}}`)

	// Seed 7: with tools.
	f.Add(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}`)

	// Seed 8: very large max_tokens.
	f.Add(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":999999999}`)

	f.Fuzz(func(t *testing.T, data string) {
		srv := buildTestModel(t)
		s := NewServer(srv)
		ts := httptest.NewServer(s.Handler())
		defer ts.Close()

		resp, err := http.Post(
			ts.URL+"/v1/chat/completions",
			"application/json",
			strings.NewReader(data),
		)
		if err != nil {
			t.Fatalf("http post: %v", err)
		}
		defer resp.Body.Close()
		io.Copy(io.Discard, resp.Body)

		// The server must never return a 5xx for any input body.
		if resp.StatusCode >= 500 {
			t.Errorf("server returned %d for input %q", resp.StatusCode, data)
		}
	})
}

func FuzzCompletionRequest(f *testing.F) {
	// Seed 1: minimal valid completion request.
	f.Add(`{"model":"m","prompt":"hello"}`)

	// Seed 2: empty prompt.
	f.Add(`{"model":"m","prompt":""}`)

	// Seed 3: not JSON.
	f.Add(`{broken`)

	// Seed 4: empty object.
	f.Add(`{}`)

	// Seed 5: with optional fields.
	f.Add(`{"model":"m","prompt":"hello","temperature":0.5,"max_tokens":10}`)

	f.Fuzz(func(t *testing.T, data string) {
		srv := buildTestModel(t)
		s := NewServer(srv)
		ts := httptest.NewServer(s.Handler())
		defer ts.Close()

		resp, err := http.Post(
			ts.URL+"/v1/completions",
			"application/json",
			strings.NewReader(data),
		)
		if err != nil {
			t.Fatalf("http post: %v", err)
		}
		defer resp.Body.Close()
		io.Copy(io.Discard, resp.Body)

		if resp.StatusCode >= 500 {
			t.Errorf("server returned %d for input %q", resp.StatusCode, data)
		}
	})
}

func FuzzEmbeddingRequest(f *testing.F) {
	// Seed 1: string input.
	f.Add(`{"model":"m","input":"hello"}`)

	// Seed 2: array input.
	f.Add(`{"model":"m","input":["hello","world"]}`)

	// Seed 3: empty object.
	f.Add(`{}`)

	// Seed 4: not JSON.
	f.Add(`xxx`)

	// Seed 5: numeric input (invalid type).
	f.Add(`{"model":"m","input":42}`)

	f.Fuzz(func(t *testing.T, data string) {
		srv := buildTestModel(t)
		s := NewServer(srv)
		ts := httptest.NewServer(s.Handler())
		defer ts.Close()

		resp, err := http.Post(
			ts.URL+"/v1/embeddings",
			"application/json",
			strings.NewReader(data),
		)
		if err != nil {
			t.Fatalf("http post: %v", err)
		}
		defer resp.Body.Close()
		io.Copy(io.Discard, resp.Body)

		// The test model does not support embeddings, so 500 from Embed()
		// is expected. We verify the server returns a valid HTTP response
		// (no panics, no connection resets) for any input body.
		if resp.StatusCode < 200 || resp.StatusCode > 599 {
			t.Errorf("server returned invalid status %d", resp.StatusCode)
		}
	})
}

func FuzzChatMessageUnmarshal(f *testing.F) {
	// Seed 1: plain string content.
	f.Add(`{"role":"user","content":"hello"}`)

	// Seed 2: vision content parts.
	f.Add(`{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBOR"}}]}`)

	// Seed 3: empty object.
	f.Add(`{}`)

	// Seed 4: not JSON.
	f.Add(`[[[`)

	f.Fuzz(func(t *testing.T, data string) {
		var msg ChatMessage
		_ = json.Unmarshal([]byte(data), &msg)
	})
}
