//go:build integration

package serve

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// newIntegrationServer creates a test server with the standard test model.
func newIntegrationServer(t *testing.T) (*httptest.Server, func()) {
	t.Helper()
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	return ts, ts.Close
}

// --- T1101.1: Chat Completions ---

func TestIntegration_ChatCompletions_NonStreaming(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	tests := []struct {
		name        string
		body        string
		wantStatus  int
		checkResult func(t *testing.T, body []byte)
	}{
		{
			name:       "basic request",
			body:       `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`,
			wantStatus: http.StatusOK,
			checkResult: func(t *testing.T, body []byte) {
				var resp ChatCompletionResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("decode: %v", err)
				}
				if resp.Object != "chat.completion" {
					t.Errorf("Object = %q, want %q", resp.Object, "chat.completion")
				}
				if !strings.HasPrefix(resp.ID, "chatcmpl-") {
					t.Errorf("ID = %q, want prefix %q", resp.ID, "chatcmpl-")
				}
				if resp.Model != "test-model" {
					t.Errorf("Model = %q, want %q", resp.Model, "test-model")
				}
				if len(resp.Choices) != 1 {
					t.Fatalf("Choices len = %d, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Message.Role != "assistant" {
					t.Errorf("Role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
				}
				if resp.Choices[0].Message.Content == "" {
					t.Error("Content should not be empty")
				}
				if resp.Choices[0].FinishReason != "stop" {
					t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "stop")
				}
				if resp.Usage.PromptTokens == 0 {
					t.Error("PromptTokens should be > 0")
				}
				if resp.Usage.CompletionTokens == 0 {
					t.Error("CompletionTokens should be > 0")
				}
				if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
					t.Errorf("TotalTokens = %d, want %d",
						resp.Usage.TotalTokens, resp.Usage.PromptTokens+resp.Usage.CompletionTokens)
				}
				if resp.Created == 0 {
					t.Error("Created should not be zero")
				}
			},
		},
		{
			name:       "with temperature and top_p",
			body:       `{"messages":[{"role":"user","content":"hello"}],"temperature":0.5,"top_p":0.9,"max_tokens":3}`,
			wantStatus: http.StatusOK,
			checkResult: func(t *testing.T, body []byte) {
				var resp ChatCompletionResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("decode: %v", err)
				}
				if len(resp.Choices) != 1 {
					t.Fatalf("Choices len = %d, want 1", len(resp.Choices))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", tt.body)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != tt.wantStatus {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			if tt.checkResult != nil {
				data, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("read body: %v", err)
				}
				tt.checkResult(t, data)
			}
		})
	}
}

func TestIntegration_ChatCompletions_Streaming(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}
	if cc := resp.Header.Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("Cache-Control = %q, want %q", cc, "no-cache")
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	body2 := string(raw)

	// Verify SSE format: non-empty lines start with "data: ".
	var dataLines int
	for _, line := range strings.Split(body2, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			t.Errorf("unexpected non-data line: %q", line)
		}
		dataLines++
	}
	if dataLines == 0 {
		t.Fatal("no SSE data lines found")
	}

	// Verify [DONE] terminator.
	if !strings.Contains(body2, "data: [DONE]") {
		t.Error("SSE stream should end with data: [DONE]")
	}

	// Verify JSON chunks have choices[].delta.content.
	content := extractSSEChatContent(t, body2)
	if content == "" {
		t.Error("streamed content should not be empty")
	}
}

// --- T1101.2: Completions ---

func TestIntegration_Completions_NonStreaming(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	tests := []struct {
		name        string
		body        string
		wantStatus  int
		checkResult func(t *testing.T, body []byte)
	}{
		{
			name:       "basic request",
			body:       `{"prompt":"hello","max_tokens":5}`,
			wantStatus: http.StatusOK,
			checkResult: func(t *testing.T, body []byte) {
				var resp CompletionResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("decode: %v", err)
				}
				if resp.Object != "text_completion" {
					t.Errorf("Object = %q, want %q", resp.Object, "text_completion")
				}
				if !strings.HasPrefix(resp.ID, "cmpl-") {
					t.Errorf("ID = %q, want prefix %q", resp.ID, "cmpl-")
				}
				if resp.Model != "test-model" {
					t.Errorf("Model = %q, want %q", resp.Model, "test-model")
				}
				if len(resp.Choices) != 1 {
					t.Fatalf("Choices len = %d, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Text == "" {
					t.Error("Text should not be empty")
				}
				if resp.Choices[0].FinishReason != "stop" {
					t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "stop")
				}
				if resp.Usage.PromptTokens == 0 {
					t.Error("PromptTokens should be > 0")
				}
				if resp.Usage.CompletionTokens == 0 {
					t.Error("CompletionTokens should be > 0")
				}
				if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
					t.Errorf("TotalTokens = %d, want %d",
						resp.Usage.TotalTokens, resp.Usage.PromptTokens+resp.Usage.CompletionTokens)
				}
			},
		},
		{
			name:       "with temperature",
			body:       `{"prompt":"hello","temperature":0.5,"max_tokens":3}`,
			wantStatus: http.StatusOK,
			checkResult: func(t *testing.T, body []byte) {
				var resp CompletionResponse
				if err := json.Unmarshal(body, &resp); err != nil {
					t.Fatalf("decode: %v", err)
				}
				if len(resp.Choices) != 1 {
					t.Fatalf("Choices len = %d, want 1", len(resp.Choices))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+"/v1/completions", "application/json", tt.body)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != tt.wantStatus {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			if tt.checkResult != nil {
				data, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("read body: %v", err)
				}
				tt.checkResult(t, data)
			}
		})
	}
}

func TestIntegration_Completions_Streaming(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	if !strings.Contains(string(raw), "data: [DONE]") {
		t.Error("SSE stream should end with data: [DONE]")
	}

	// Verify JSON chunks have choices[].text.
	text := extractSSECompletionText(t, string(raw))
	if text == "" {
		t.Error("streamed text should not be empty")
	}
}

// --- T1101.3: Models Endpoint ---

func TestIntegration_Models(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	t.Run("list models", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/v1/models")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}

		var result ModelListResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if result.Object != "list" {
			t.Errorf("Object = %q, want %q", result.Object, "list")
		}
		if len(result.Data) != 1 {
			t.Fatalf("Data len = %d, want 1", len(result.Data))
		}
		if result.Data[0].ID != "test-model" {
			t.Errorf("model ID = %q, want %q", result.Data[0].ID, "test-model")
		}
		if result.Data[0].Object != "model" {
			t.Errorf("model Object = %q, want %q", result.Data[0].Object, "model")
		}
		if result.Data[0].OwnedBy != "local" {
			t.Errorf("OwnedBy = %q, want %q", result.Data[0].OwnedBy, "local")
		}
	})

	t.Run("get model info", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/v1/models/test-model")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}

		var result ModelObject
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if result.ID != "test-model" {
			t.Errorf("ID = %q, want %q", result.ID, "test-model")
		}
		if result.Object != "model" {
			t.Errorf("Object = %q, want %q", result.Object, "model")
		}
		if result.Created == 0 {
			t.Error("Created should not be zero")
		}
	})

	t.Run("get model info not found", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/v1/models/nonexistent-model")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("status = %d, want 404", resp.StatusCode)
		}
	})

	t.Run("delete model", func(t *testing.T) {
		resp := doDelete(t, ts.URL+"/v1/models/test-model")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}

		var result ModelDeleteResponse
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if result.ID != "test-model" {
			t.Errorf("ID = %q, want %q", result.ID, "test-model")
		}
		if !result.Deleted {
			t.Error("Deleted should be true")
		}
		if result.Object != "model" {
			t.Errorf("Object = %q, want %q", result.Object, "model")
		}

		// Verify model is gone from list.
		listResp := doGet(t, ts.URL+"/v1/models")
		defer func() { _ = listResp.Body.Close() }()

		var listResult ModelListResponse
		if err := json.NewDecoder(listResp.Body).Decode(&listResult); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(listResult.Data) != 0 {
			t.Errorf("Data len = %d, want 0 after deletion", len(listResult.Data))
		}

		// Verify model info returns 404 after deletion.
		infoResp := doGet(t, ts.URL+"/v1/models/test-model")
		defer func() { _ = infoResp.Body.Close() }()

		if infoResp.StatusCode != http.StatusNotFound {
			t.Errorf("model info after delete: status = %d, want 404", infoResp.StatusCode)
		}
	})
}

// --- T1101.4: Error Handling and Concurrent Requests ---

func TestIntegration_ErrorHandling(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	tests := []struct {
		name       string
		endpoint   string
		body       string
		wantStatus int
	}{
		{
			name:       "chat completions invalid JSON",
			endpoint:   "/v1/chat/completions",
			body:       "not json",
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "chat completions empty messages",
			endpoint:   "/v1/chat/completions",
			body:       `{"messages":[]}`,
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "completions invalid JSON",
			endpoint:   "/v1/completions",
			body:       "{bad",
			wantStatus: http.StatusBadRequest,
		},
		{
			name:       "completions empty prompt",
			endpoint:   "/v1/completions",
			body:       `{"prompt":""}`,
			wantStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+tt.endpoint, "application/json", tt.body)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != tt.wantStatus {
				t.Errorf("status = %d, want %d", resp.StatusCode, tt.wantStatus)
			}

			// Verify error response has an error object.
			data, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			var errResp map[string]interface{}
			if err := json.Unmarshal(data, &errResp); err != nil {
				t.Fatalf("decode error response: %v", err)
			}
			if _, ok := errResp["error"]; !ok {
				t.Error("response should contain an 'error' field")
			}
		})
	}
}

func TestIntegration_ModelNotFound(t *testing.T) {
	ts, cleanup := newIntegrationServer(t)
	defer cleanup()

	t.Run("GET nonexistent model", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/v1/models/wrong-model-id")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("status = %d, want 404", resp.StatusCode)
		}
	})

	t.Run("DELETE nonexistent model", func(t *testing.T) {
		resp := doDelete(t, ts.URL+"/v1/models/wrong-model-id")
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("status = %d, want 404", resp.StatusCode)
		}
	})
}

func TestIntegration_ConcurrentRequests(t *testing.T) {
	// Pre-create all servers sequentially to avoid data races in
	// NewCPUEngine (writes to package-level state).
	const concurrency = 10
	servers := make([]*httptest.Server, concurrency)
	for i := range concurrency {
		ts, _ := newIntegrationServer(t)
		servers[i] = ts
	}
	defer func() {
		for _, ts := range servers {
			ts.Close()
		}
	}()

	var wg sync.WaitGroup
	wg.Add(concurrency)

	for i := range concurrency {
		go func(ts *httptest.Server) {
			defer wg.Done()
			resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json",
				`{"messages":[{"role":"user","content":"hello"}],"max_tokens":3}`)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != http.StatusOK {
				t.Errorf("concurrent request: status = %d, want 200", resp.StatusCode)
			}
			_, _ = io.ReadAll(resp.Body)
		}(servers[i])
	}

	wg.Wait()
}
