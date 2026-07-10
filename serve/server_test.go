package serve

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/inference/lora"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/zerfoo/serve/security"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	tokenizer "github.com/zerfoo/ztoken"
)

// fixedLogitsNode always returns logits where a specific token has the highest value.
type fixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
}

func (n *fixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *fixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *fixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *fixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *fixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}
	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}
	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

func buildTestModel(t *testing.T) *inference.Model {
	t.Helper()
	vocabSize := 8
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: []int{6, 7, 2}, // foo, bar, EOS
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}

// doPost sends a POST request with context.
func doPost(t *testing.T, url, contentType, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", contentType)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request error: %v", err)
	}
	return resp
}

// doGet sends a GET request with context.
func doGet(t *testing.T, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, http.NoBody)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request error: %v", err)
	}
	return resp
}

// errorNode always fails during Forward.
type errorNode struct {
	graph.NoParameters[float32]
}

func (n *errorNode) OpType() string                     { return "Error" }
func (n *errorNode) Attributes() map[string]interface{} { return nil }
func (n *errorNode) OutputShape() []int                 { return []int{1, 1, 8} }
func (n *errorNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *errorNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return nil, errors.New("forward error")
}

// buildModelWithNode builds a minimal inference.Model using the given node as the sole graph op.
func buildModelWithNode(t *testing.T, node graph.Node[float32]) *inference.Model {
	t.Helper()
	vocabSize := 8
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")
	tok.AddToken("foo")
	tok.AddToken("bar")

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}

func buildErrorModel(t *testing.T) *inference.Model {
	t.Helper()
	return buildModelWithNode(t, &errorNode{})
}

// --- Chat Completions ---

func TestHandleChatCompletions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", result.Object, "chat.completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
	if result.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", result.Choices[0].Message.Role, "assistant")
	}
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", result.Choices[0].FinishReason, "stop")
	}
}

func TestHandleChatCompletions_EmptyMessages(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_InvalidJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", "not json")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

// TestHandleChatCompletions_AdapterPathTraversalRejected is the API-level
// regression test for SERVE-2: a request whose "model" field encodes a LoRA
// adapter name with path-traversal sequences must be rejected with a
// 400-class error, and must never open a file outside the configured
// adapter directory.
func TestHandleChatCompletions_AdapterPathTraversalRejected(t *testing.T) {
	mdl := buildTestModel(t)

	base := t.TempDir()
	dir := filepath.Join(base, "adapters")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Plant a real, loadable adapter GGUF outside the adapter directory.
	// If the traversal succeeded, this is the file it would open.
	writeLoRAGGUFFile(t, filepath.Join(base, "secret.gguf"), 4, 8.0)

	srv := NewServer(mdl, WithAdapterCache(dir, 5))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"model":"base:../secret","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body = %s", resp.StatusCode, b)
	}
	if srv.adapterCache.cache.Size() != 0 {
		t.Errorf("adapter cache poisoned by traversal name, size = %d, want 0", srv.adapterCache.cache.Size())
	}
}

// TestHandleChatCompletions_AdapterValidNameLoads confirms the traversal fix
// does not regress the legitimate path: a real adapter with a well-formed
// name still resolves via the adapter cache and the request completes.
func TestHandleChatCompletions_AdapterValidNameLoads(t *testing.T) {
	mdl := buildTestModel(t)

	dir := t.TempDir()
	writeLoRAGGUFFile(t, filepath.Join(dir, "my-adapter.gguf"), 4, 8.0)

	srv := NewServer(mdl, WithAdapterCache(dir, 5))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"model":"test-model:my-adapter","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, b)
	}
	if srv.adapterCache.cache.Size() != 1 {
		t.Errorf("adapter cache size = %d, want 1 (legitimate adapter should load)", srv.adapterCache.cache.Size())
	}
}

func TestHandleChatCompletions_WithOptions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"temperature":0.5,"top_p":0.9,"max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// --- Completions ---

func TestHandleCompletions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "text_completion" {
		t.Errorf("Object = %q, want %q", result.Object, "text_completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestHandleCompletions_EmptyPrompt(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":""}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleCompletions_InvalidJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/completions", "application/json", "{bad")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleCompletions_WithOptions(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","temperature":0.5,"max_tokens":3}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// --- Models ---

func TestHandleModels(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
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
}

// --- Streaming ---

func TestHandleChatCompletions_Stream(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}
	// Drain the body to ensure no errors.
	_, _ = io.ReadAll(resp.Body)
}

func TestHandleCompletions_Stream(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// Drain the body to ensure no errors.
	_, _ = io.ReadAll(resp.Body)
}

// --- Error paths ---

func TestHandleChatCompletions_GenerateError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestHandleCompletions_GenerateError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestHandleChatCompletions_StreamError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	// Streaming starts with 200 but body contains the error.
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(data), "error") {
		t.Errorf("body should contain error, got %q", string(data))
	}
}

func TestHandleCompletions_StreamError(t *testing.T) {
	mdl := buildErrorModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	data, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(data), "error") {
		t.Errorf("body should contain error, got %q", string(data))
	}
}

// --- Integration: SSE parity ---

func TestChatCompletion_StreamParity(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Non-streaming request.
	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Verify JSON structure.
	if result.ID == "" {
		t.Error("response ID should not be empty")
	}
	if result.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", result.Object, "chat.completion")
	}
	if result.Model != "test-model" {
		t.Errorf("Model = %q, want %q", result.Model, "test-model")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
	nonStreamContent := result.Choices[0].Message.Content

	// Streaming request.
	streamBody := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	streamResp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", streamBody)
	defer func() { _ = streamResp.Body.Close() }()

	if streamResp.StatusCode != http.StatusOK {
		t.Fatalf("stream status = %d, want 200", streamResp.StatusCode)
	}

	// Parse SSE events to extract concatenated content.
	raw, _ := io.ReadAll(streamResp.Body)
	streamContent := extractSSEChatContent(t, string(raw))

	if nonStreamContent != streamContent {
		t.Errorf("stream/non-stream content mismatch:\n  non-stream: %q\n  stream:     %q",
			nonStreamContent, streamContent)
	}
}

func TestCompletion_StreamParity(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Non-streaming.
	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Object != "text_completion" {
		t.Errorf("Object = %q, want %q", result.Object, "text_completion")
	}
	if result.Model != "test-model" {
		t.Errorf("Model = %q, want %q", result.Model, "test-model")
	}
	nonStreamText := result.Choices[0].Text

	// Streaming.
	streamBody := `{"prompt":"hello","stream":true,"max_tokens":5}`
	streamResp := doPost(t, ts.URL+"/v1/completions", "application/json", streamBody)
	defer func() { _ = streamResp.Body.Close() }()

	if streamResp.StatusCode != http.StatusOK {
		t.Fatalf("stream status = %d, want 200", streamResp.StatusCode)
	}

	raw, _ := io.ReadAll(streamResp.Body)
	streamText := extractSSECompletionText(t, string(raw))

	if nonStreamText != streamText {
		t.Errorf("stream/non-stream text mismatch:\n  non-stream: %q\n  stream:     %q",
			nonStreamText, streamText)
	}
}

// extractSSEChatContent parses SSE data lines and concatenates delta.content fields.
func extractSSEChatContent(t *testing.T, raw string) string {
	t.Helper()
	var sb strings.Builder
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Logf("skip unparseable SSE chunk: %s", data)
			continue
		}
		if len(chunk.Choices) > 0 {
			sb.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	return sb.String()
}

// extractSSECompletionText parses SSE data lines and concatenates text fields.
func extractSSECompletionText(t *testing.T, raw string) string {
	t.Helper()
	var sb strings.Builder
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Logf("skip unparseable SSE chunk: %s", data)
			continue
		}
		if len(chunk.Choices) > 0 {
			sb.WriteString(chunk.Choices[0].Text)
		}
	}
	return sb.String()
}

// doDelete sends a DELETE request with context.
func doDelete(t *testing.T, url string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(context.Background(), http.MethodDelete, url, http.NoBody)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request error: %v", err)
	}
	return resp
}

// --- Embeddings ---

func TestHandleEmbeddings(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"input":"hello","model":"test-model"}`
	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	// Embed is not supported in the test model, so expect 500.
	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 (embeddings not supported)", resp.StatusCode)
	}
}

func TestHandleEmbeddings_InvalidJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", "not json")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleEmbeddings_EmptyInput(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"input":[]}`
	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleEmbeddings_InvalidInputType(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"input":42}`
	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleEmbeddings_BatchInput(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"input":["hello","world"]}`
	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	// Embed is not supported in the test model, so expect 500.
	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

// --- Model Info ---

func TestHandleModelInfo(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models/test-model")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ModelObject
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.ID != "test-model" {
		t.Errorf("ID = %q, want %q", result.ID, "test-model")
	}
	if result.Object != "model" {
		t.Errorf("Object = %q, want %q", result.Object, "model")
	}
	if result.OwnedBy != "local" {
		t.Errorf("OwnedBy = %q, want %q", result.OwnedBy, "local")
	}
	if result.Created == 0 {
		t.Error("Created should not be zero")
	}
}

func TestHandleModelInfo_NotFound(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/v1/models/nonexistent")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

// --- Model Delete ---

func TestHandleModelDelete(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doDelete(t, ts.URL+"/v1/models/test-model")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ModelDeleteResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
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

	// After deletion, listing models should return empty.
	listResp := doGet(t, ts.URL+"/v1/models")
	defer func() { _ = listResp.Body.Close() }()

	var listResult ModelListResponse
	if err := json.NewDecoder(listResp.Body).Decode(&listResult); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(listResult.Data) != 0 {
		t.Errorf("Data len = %d, want 0 after deletion", len(listResult.Data))
	}
}

func TestHandleModelDelete_NotFound(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doDelete(t, ts.URL+"/v1/models/nonexistent")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestHandleModelDelete_AlreadyDeleted(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Delete once.
	resp := doDelete(t, ts.URL+"/v1/models/test-model")
	_ = resp.Body.Close()

	// Delete again - should get 404.
	resp2 := doDelete(t, ts.URL+"/v1/models/test-model")
	defer func() { _ = resp2.Body.Close() }()

	if resp2.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp2.StatusCode)
	}
}

// --- Usage token counting ---

func TestChatCompletions_UsageTokens(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Usage.PromptTokens == 0 {
		t.Error("PromptTokens should not be zero")
	}
	if result.Usage.CompletionTokens == 0 {
		t.Error("CompletionTokens should not be zero")
	}
	if result.Usage.TotalTokens != result.Usage.PromptTokens+result.Usage.CompletionTokens {
		t.Errorf("TotalTokens = %d, want %d",
			result.Usage.TotalTokens, result.Usage.PromptTokens+result.Usage.CompletionTokens)
	}
}

func TestCompletions_UsageTokens(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Usage.PromptTokens == 0 {
		t.Error("PromptTokens should not be zero")
	}
	if result.Usage.CompletionTokens == 0 {
		t.Error("CompletionTokens should not be zero")
	}
	if result.Usage.TotalTokens != result.Usage.PromptTokens+result.Usage.CompletionTokens {
		t.Errorf("TotalTokens = %d, want %d",
			result.Usage.TotalTokens, result.Usage.PromptTokens+result.Usage.CompletionTokens)
	}
}

// --- Close ---

func TestServer_Close(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	if err := srv.Close(context.Background()); err != nil {
		t.Errorf("Close error: %v", err)
	}
}

// --- OpenAPI spec ---

func TestHandleOpenAPISpec(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/openapi.yaml")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/yaml" {
		t.Errorf("Content-Type = %q, want %q", ct, "application/yaml")
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(body) != string(openapiSpec) {
		t.Errorf("response body does not match embedded openapi.yaml")
	}
	if !strings.Contains(string(body), "openapi:") {
		t.Errorf("response body does not look like valid OpenAPI YAML")
	}
}

// --- Response format compliance ---

func TestChatCompletions_ResponseIDPrefix(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if !strings.HasPrefix(result.ID, "chatcmpl-") {
		t.Errorf("ID = %q, want prefix %q", result.ID, "chatcmpl-")
	}
	if result.Created == 0 {
		t.Error("Created should not be zero")
	}
}

func TestCompletions_ResponseIDPrefix(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var result CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if !strings.HasPrefix(result.ID, "cmpl-") {
		t.Errorf("ID = %q, want prefix %q", result.ID, "cmpl-")
	}
	if result.Created == 0 {
		t.Error("Created should not be zero")
	}
}

func TestChatCompletions_ContentTypeJSON(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want %q", ct, "application/json")
	}
}

// --- SSE format validation ---

func TestChatCompletions_SSEFormat(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	// Verify streaming headers.
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}
	if cc := resp.Header.Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("Cache-Control = %q, want %q", cc, "no-cache")
	}

	raw, _ := io.ReadAll(resp.Body)
	lines := strings.Split(string(raw), "\n")

	// Every non-empty line must start with "data: ".
	var dataLines int
	for _, line := range lines {
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

	// Last data line should be [DONE].
	if !strings.Contains(string(raw), "data: [DONE]") {
		t.Error("SSE stream should end with data: [DONE]")
	}
}

func TestCompletions_SSEFormat(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","stream":true,"max_tokens":5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want %q", ct, "text/event-stream")
	}

	raw, _ := io.ReadAll(resp.Body)

	// Verify each SSE chunk is valid JSON with choices[].text.
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Errorf("invalid JSON in SSE chunk: %s", data)
			continue
		}
		choices, ok := chunk["choices"]
		if !ok {
			t.Errorf("SSE chunk missing 'choices' field: %s", data)
			continue
		}
		arr, ok := choices.([]interface{})
		if !ok || len(arr) == 0 {
			t.Errorf("SSE chunk 'choices' is not a non-empty array: %s", data)
		}
	}

	if !strings.Contains(string(raw), "data: [DONE]") {
		t.Error("SSE stream should end with data: [DONE]")
	}
}

// --- Model info after delete ---

func TestHandleModelInfo_AfterDelete(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Delete the model.
	delResp := doDelete(t, ts.URL+"/v1/models/test-model")
	_ = delResp.Body.Close()

	if delResp.StatusCode != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", delResp.StatusCode)
	}

	// Model info should return 404 after deletion.
	infoResp := doGet(t, ts.URL+"/v1/models/test-model")
	defer func() { _ = infoResp.Body.Close() }()

	if infoResp.StatusCode != http.StatusNotFound {
		t.Errorf("model info after delete: status = %d, want 404", infoResp.StatusCode)
	}
}

// --- Embeddings non-string array items ---

func TestHandleEmbeddings_NonStringArrayItems(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"input":["hello", 42]}`
	resp := doPost(t, ts.URL+"/v1/embeddings", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

// --- Full integration: all endpoints on a single server ---

func TestIntegration_AllEndpoints(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// 1. GET /v1/models - list models.
	listResp := doGet(t, ts.URL+"/v1/models")
	var listResult ModelListResponse
	if err := json.NewDecoder(listResp.Body).Decode(&listResult); err != nil {
		t.Fatalf("models list decode: %v", err)
	}
	_ = listResp.Body.Close()
	if listResult.Object != "list" {
		t.Errorf("models Object = %q, want %q", listResult.Object, "list")
	}
	if len(listResult.Data) != 1 {
		t.Fatalf("models Data len = %d, want 1", len(listResult.Data))
	}
	modelID := listResult.Data[0].ID

	// 2. GET /v1/models/{id} - model info.
	infoResp := doGet(t, ts.URL+"/v1/models/"+modelID)
	var infoResult ModelObject
	if err := json.NewDecoder(infoResp.Body).Decode(&infoResult); err != nil {
		t.Fatalf("model info decode: %v", err)
	}
	_ = infoResp.Body.Close()
	if infoResult.ID != modelID {
		t.Errorf("model info ID = %q, want %q", infoResult.ID, modelID)
	}
	if infoResult.Object != "model" {
		t.Errorf("model info Object = %q, want %q", infoResult.Object, "model")
	}

	// 3. POST /v1/chat/completions - non-streaming.
	chatBody := `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`
	chatResp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", chatBody)
	var chatResult ChatCompletionResponse
	if err := json.NewDecoder(chatResp.Body).Decode(&chatResult); err != nil {
		t.Fatalf("chat decode: %v", err)
	}
	_ = chatResp.Body.Close()
	if chatResult.Object != "chat.completion" {
		t.Errorf("chat Object = %q, want %q", chatResult.Object, "chat.completion")
	}
	if chatResult.Model != modelID {
		t.Errorf("chat Model = %q, want %q", chatResult.Model, modelID)
	}
	if len(chatResult.Choices) != 1 || chatResult.Choices[0].Message.Role != "assistant" {
		t.Error("chat response should have 1 choice with assistant role")
	}

	// 4. POST /v1/chat/completions - streaming.
	chatStreamBody := `{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":5}`
	chatStreamResp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", chatStreamBody)
	chatStreamRaw, _ := io.ReadAll(chatStreamResp.Body)
	_ = chatStreamResp.Body.Close()
	if !strings.Contains(string(chatStreamRaw), "data: [DONE]") {
		t.Error("chat stream should contain [DONE]")
	}

	// 5. POST /v1/completions - non-streaming.
	complBody := `{"prompt":"hello","max_tokens":5}`
	complResp := doPost(t, ts.URL+"/v1/completions", "application/json", complBody)
	var complResult CompletionResponse
	if err := json.NewDecoder(complResp.Body).Decode(&complResult); err != nil {
		t.Fatalf("completion decode: %v", err)
	}
	_ = complResp.Body.Close()
	if complResult.Object != "text_completion" {
		t.Errorf("completion Object = %q, want %q", complResult.Object, "text_completion")
	}
	if complResult.Model != modelID {
		t.Errorf("completion Model = %q, want %q", complResult.Model, modelID)
	}

	// 6. POST /v1/completions - streaming.
	complStreamBody := `{"prompt":"hello","stream":true,"max_tokens":5}`
	complStreamResp := doPost(t, ts.URL+"/v1/completions", "application/json", complStreamBody)
	complStreamRaw, _ := io.ReadAll(complStreamResp.Body)
	_ = complStreamResp.Body.Close()
	if !strings.Contains(string(complStreamRaw), "data: [DONE]") {
		t.Error("completion stream should contain [DONE]")
	}

	// 7. POST /v1/embeddings (expected to fail on test model, but verifies routing).
	embBody := `{"input":"hello"}`
	embResp := doPost(t, ts.URL+"/v1/embeddings", "application/json", embBody)
	_ = embResp.Body.Close()
	// 500 is expected since test model does not support embeddings.
	if embResp.StatusCode != http.StatusInternalServerError {
		t.Errorf("embeddings status = %d, want 500", embResp.StatusCode)
	}

	// 8. GET /openapi.yaml
	specResp := doGet(t, ts.URL+"/openapi.yaml")
	specBody, _ := io.ReadAll(specResp.Body)
	_ = specResp.Body.Close()
	if specResp.StatusCode != http.StatusOK {
		t.Fatalf("openapi status = %d, want 200", specResp.StatusCode)
	}
	if !strings.Contains(string(specBody), "openapi:") {
		t.Error("openapi.yaml should contain 'openapi:' field")
	}

	// 9. DELETE /v1/models/{id} - delete the model.
	delResp := doDelete(t, ts.URL+"/v1/models/"+modelID)
	var delResult ModelDeleteResponse
	if err := json.NewDecoder(delResp.Body).Decode(&delResult); err != nil {
		t.Fatalf("delete decode: %v", err)
	}
	_ = delResp.Body.Close()
	if !delResult.Deleted {
		t.Error("delete Deleted should be true")
	}

	// 10. Verify model is gone.
	postDelList := doGet(t, ts.URL+"/v1/models")
	var postDelResult ModelListResponse
	if err := json.NewDecoder(postDelList.Body).Decode(&postDelResult); err != nil {
		t.Fatalf("post-delete list decode: %v", err)
	}
	_ = postDelList.Body.Close()
	if len(postDelResult.Data) != 0 {
		t.Errorf("after delete, models Data len = %d, want 0", len(postDelResult.Data))
	}
}

// --- Panic recovery ---

// panicNode panics during Forward.
type panicNode struct {
	graph.NoParameters[float32]
}

func (n *panicNode) OpType() string                     { return "Panic" }
func (n *panicNode) Attributes() map[string]interface{} { return nil }
func (n *panicNode) OutputShape() []int                 { return []int{1, 1, 8} }
func (n *panicNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *panicNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	panic("test panic in handler")
}

func buildPanicModel(t *testing.T) *inference.Model {
	t.Helper()
	return buildModelWithNode(t, &panicNode{})
}

// oomErrorNode returns an OOM error during Forward.
type oomErrorNode struct {
	graph.NoParameters[float32]
}

func (n *oomErrorNode) OpType() string                     { return "OOMError" }
func (n *oomErrorNode) Attributes() map[string]interface{} { return nil }
func (n *oomErrorNode) OutputShape() []int                 { return []int{1, 1, 8} }
func (n *oomErrorNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *oomErrorNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return nil, errors.New("CUDA out of memory: tried to allocate 2.00 GiB")
}

func buildOOMModel(t *testing.T) *inference.Model {
	t.Helper()
	return buildModelWithNode(t, &oomErrorNode{})
}

func TestRecoveryMiddleware_PanicReturns500(t *testing.T) {
	mdl := buildPanicModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	tests := []struct {
		name string
		path string
		body string
	}{
		{
			name: "chat completions panic",
			path: "/v1/chat/completions",
			body: `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`,
		},
		{
			name: "completions panic",
			path: "/v1/completions",
			body: `{"prompt":"hello","max_tokens":5}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+tt.path, "application/json", tt.body)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != http.StatusInternalServerError {
				t.Fatalf("status = %d, want 500", resp.StatusCode)
			}

			var result map[string]interface{}
			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				t.Fatalf("decode error: %v", err)
			}
			errObj, ok := result["error"].(map[string]interface{})
			if !ok {
				t.Fatal("response missing error object")
			}
			msg, _ := errObj["message"].(string)
			if msg != "internal server error" {
				t.Errorf("error message = %q, want %q", msg, "internal server error")
			}
		})
	}
}

func TestOOMError_Returns503(t *testing.T) {
	mdl := buildOOMModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	tests := []struct {
		name string
		path string
		body string
	}{
		{
			name: "chat completions OOM",
			path: "/v1/chat/completions",
			body: `{"messages":[{"role":"user","content":"hello"}],"max_tokens":5}`,
		},
		{
			name: "completions OOM",
			path: "/v1/completions",
			body: `{"prompt":"hello","max_tokens":5}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+tt.path, "application/json", tt.body)
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != http.StatusServiceUnavailable {
				t.Fatalf("status = %d, want 503", resp.StatusCode)
			}
		})
	}
}

func TestIsOOMError(t *testing.T) {
	tests := []struct {
		msg  string
		want bool
	}{
		{"CUDA out of memory: tried to allocate 2.00 GiB", true},
		{"OOM killed", true},
		{"cannot allocate memory", true},
		{"CUDA_ERROR_OUT_OF_MEMORY", true},
		{"CUBLAS_STATUS_ALLOC_FAILED", true},
		{"cuda driver version mismatch", false},
		{"forward error", false},
		{"invalid request body", false},
	}

	for _, tt := range tests {
		t.Run(tt.msg, func(t *testing.T) {
			err := errors.New(tt.msg)
			if got := isOOMError(err); got != tt.want {
				t.Errorf("isOOMError(%q) = %v, want %v", tt.msg, got, tt.want)
			}
		})
	}
}

// --- Response Format ---

func TestHandleChatCompletions_ResponseFormatJSONSchema(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Valid json_schema response_format — server should accept and return 200.
	body := `{
		"messages":[{"role":"user","content":"hello"}],
		"max_tokens":5,
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"test_schema",
				"strict":true,
				"schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
			}
		}
	}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", result.Object, "chat.completion")
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestHandleChatCompletions_ResponseFormatInvalidSchema(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{
		"messages":[{"role":"user","content":"hello"}],
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"bad",
				"schema":"not a json object"
			}
		}
	}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_ResponseFormatUnsupportedSchema(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Schema with $ref is unsupported and should return 400.
	body := `{
		"messages":[{"role":"user","content":"hello"}],
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"unsupported",
				"schema":{"$ref":"#/definitions/Foo"}
			}
		}
	}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_ResponseFormatText(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// response_format type "text" should behave like no response_format (normal response).
	body := `{
		"messages":[{"role":"user","content":"hello"}],
		"max_tokens":5,
		"response_format":{"type":"text"}
	}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
	if result.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", result.Choices[0].FinishReason, "stop")
	}
}

func TestHandleChatCompletions_ResponseFormatJSONObject(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// json_object type is a pass-through (no grammar), should return 200.
	body := `{
		"messages":[{"role":"user","content":"hello"}],
		"max_tokens":5,
		"response_format":{"type":"json_object"}
	}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("Choices len = %d, want 1", len(result.Choices))
	}
}

func TestAuthMiddleware(t *testing.T) {
	mdl := buildTestModel(t)
	const apiKey = "test-secret-key"
	srv := NewServer(mdl, WithAPIKey(apiKey))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	t.Run("no key returns 401", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/v1/models")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", resp.StatusCode)
		}
	})

	t.Run("wrong key returns 401", func(t *testing.T) {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/v1/models", http.NoBody)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer wrong-key")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("status = %d, want 401", resp.StatusCode)
		}
	})

	t.Run("correct key returns 200", func(t *testing.T) {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/v1/models", http.NoBody)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})

	t.Run("metrics skips auth", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/metrics")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})

	t.Run("openapi.yaml skips auth", func(t *testing.T) {
		resp := doGet(t, ts.URL+"/openapi.yaml")
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
	})
}

func TestRequestBodySizeLimit(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Build a valid JSON body that exceeds 10 MB.
	// Use a JSON object with a large string value so the decoder must
	// read past the 10 MB limit before completing the parse.
	bigValue := strings.Repeat("x", 11<<20) // 11 MB string
	oversized := `{"prompt":"` + bigValue + `"}`

	endpoints := []string{
		"/v1/chat/completions",
		"/v1/completions",
		"/v1/embeddings",
	}

	for _, ep := range endpoints {
		t.Run(ep, func(t *testing.T) {
			resp := doPost(t, ts.URL+ep, "application/json", oversized)
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusRequestEntityTooLarge {
				body, _ := io.ReadAll(resp.Body)
				t.Fatalf("status = %d, want %d; body = %s", resp.StatusCode, http.StatusRequestEntityTooLarge, body)
			}
		})
	}
}

func TestMaxTokensClamp(t *testing.T) {
	mdl := buildTestModel(t)

	t.Run("default_clamps_chat", func(t *testing.T) {
		srv := NewServer(mdl)
		if srv.maxTokens != 8192 {
			t.Fatalf("default maxTokens = %d, want 8192", srv.maxTokens)
		}
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":100000}`
		resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, b)
		}
	})

	t.Run("default_clamps_completions", func(t *testing.T) {
		srv := NewServer(mdl)
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{"model":"test-model","prompt":"hello","max_tokens":100000}`
		resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, b)
		}
	})

	t.Run("custom_max_tokens_option", func(t *testing.T) {
		srv := NewServer(mdl, WithMaxTokens(256))
		if srv.maxTokens != 256 {
			t.Fatalf("maxTokens = %d, want 256", srv.maxTokens)
		}
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":100000}`
		resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, b)
		}
	})

	t.Run("within_limit_not_clamped", func(t *testing.T) {
		srv := NewServer(mdl)
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":100}`
		resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, want 200; body = %s", resp.StatusCode, b)
		}
	})
}

func TestRateLimitMiddleware(t *testing.T) {
	m := buildTestModel(t)

	// Allow 2 requests with burst=2, rate=0 (no refill).
	rl := security.NewRateLimiter(0, 2)
	srv := NewServer(m, WithRateLimiter(rl))
	defer srv.Close(context.Background())
	handler := srv.Handler()

	// First two requests should succeed.
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("request %d: got status %d, want %d", i+1, rec.Code, http.StatusOK)
		}
	}

	// Third request should be rate limited.
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("got status %d, want %d", rec.Code, http.StatusTooManyRequests)
	}

	var errResp map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&errResp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	errObj, ok := errResp["error"].(map[string]interface{})
	if !ok {
		t.Fatal("expected error object in response")
	}
	if msg, _ := errObj["message"].(string); msg != "rate limit exceeded" {
		t.Fatalf("got error message %q, want %q", msg, "rate limit exceeded")
	}
}

func TestSecurityHeaders(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	srv.Handler().ServeHTTP(rec, req)

	expected := map[string]string{
		"X-Content-Type-Options": "nosniff",
		"X-Frame-Options":        "DENY",
		"Cache-Control":          "no-store",
	}
	for header, want := range expected {
		got := rec.Header().Get(header)
		if got != want {
			t.Errorf("header %s = %q, want %q", header, got, want)
		}
	}
}

// --- Sampling parameter validation ---

func TestHandleChatCompletions_NegativeTemperature(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hi"}],"temperature":-1}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleChatCompletions_TopPClampedTo1(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// TopP=1.5 should be clamped to 1.0 and succeed.
	body := `{"messages":[{"role":"user","content":"hi"}],"top_p":1.5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}

func TestHandleChatCompletions_NegativeTopK(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"messages":[{"role":"user","content":"hi"}],"top_k":-5}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestHandleCompletions_NegativeTemperature(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt":"hello","temperature":-0.5}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestValidateSamplingParams(t *testing.T) {
	t.Run("valid params", func(t *testing.T) {
		temp := 0.7
		topP := 0.9
		topK := 40
		if err := validateSamplingParams(&temp, &topP, &topK); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("nil params", func(t *testing.T) {
		if err := validateSamplingParams(nil, nil, nil); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("negative temperature", func(t *testing.T) {
		temp := -1.0
		if err := validateSamplingParams(&temp, nil, nil); err == nil {
			t.Error("expected error for negative temperature")
		}
	})

	t.Run("topP clamped high", func(t *testing.T) {
		topP := 1.5
		if err := validateSamplingParams(nil, &topP, nil); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if topP != 1.0 {
			t.Errorf("topP = %g, want 1.0", topP)
		}
	})

	t.Run("topP clamped low", func(t *testing.T) {
		topP := -0.5
		if err := validateSamplingParams(nil, &topP, nil); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if topP != 0.0 {
			t.Errorf("topP = %g, want 0.0", topP)
		}
	})

	t.Run("negative topK", func(t *testing.T) {
		topK := -1
		if err := validateSamplingParams(nil, nil, &topK); err == nil {
			t.Error("expected error for negative topK")
		}
	})

	t.Run("zero temperature allowed", func(t *testing.T) {
		temp := 0.0
		if err := validateSamplingParams(&temp, nil, nil); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("zero topK allowed", func(t *testing.T) {
		topK := 0
		if err := validateSamplingParams(nil, nil, &topK); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
}
func TestHandleHealthz(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/healthz")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("status = %q, want %q", body["status"], "ok")
	}
}

func TestHandleReadyz(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp := doGet(t, ts.URL+"/readyz")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if body["status"] != "ready" {
		t.Errorf("status = %q, want %q", body["status"], "ready")
	}
}

func TestHandleReadyz_AfterUnload(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Unload the model via DELETE /v1/models/{id}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodDelete, ts.URL+"/v1/models/test-model", http.NoBody)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	delResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("delete request: %v", err)
	}
	_ = delResp.Body.Close()

	resp := doGet(t, ts.URL+"/readyz")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}

	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if body["status"] != "not ready" {
		t.Errorf("status = %q, want %q", body["status"], "not ready")
	}
}

// noFlushResponseRecorder implements http.ResponseWriter but NOT http.Flusher,
// so streaming handlers can detect the absence of flush support.
type noFlushResponseRecorder struct {
	code   int
	header http.Header
	body   strings.Builder
}

func newNoFlushResponseRecorder() *noFlushResponseRecorder {
	return &noFlushResponseRecorder{header: http.Header{}}
}

func (r *noFlushResponseRecorder) Header() http.Header         { return r.header }
func (r *noFlushResponseRecorder) Write(b []byte) (int, error) { return r.body.Write(b) }
func (r *noFlushResponseRecorder) WriteHeader(code int)        { r.code = code }

func TestStreamChatCompletion_NoFlusher(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	rec := newNoFlushResponseRecorder()

	srv.streamChatCompletion(rec, context.Background(), nil, nil)

	if rec.code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d", rec.code, http.StatusInternalServerError)
	}
	if !strings.Contains(rec.body.String(), "streaming not supported") {
		t.Errorf("body = %q, want error about streaming not supported", rec.body.String())
	}
}

func TestStreamCompletion_NoFlusher(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	rec := newNoFlushResponseRecorder()

	srv.streamCompletion(rec, context.Background(), "hello", nil)

	if rec.code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d", rec.code, http.StatusInternalServerError)
	}
	if !strings.Contains(rec.body.String(), "streaming not supported") {
		t.Errorf("body = %q, want error about streaming not supported", rec.body.String())
	}
}

// TestMalformedJSON_NoInternalDetails verifies that malformed JSON requests
// return a generic error message without leaking Go internal type information.
func TestMalformedJSON_NoInternalDetails(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Each case must trigger a JSON decode error so we can verify
	// the raw Go error string is not leaked to the client.
	cases := []struct {
		endpoint string
		name     string
		body     string
	}{
		// Type mismatch: string where float64 expected leaks "Go struct field" info.
		{"/v1/chat/completions", "type_mismatch", `{"temperature":"not_a_number"}`},
		{"/v1/completions", "type_mismatch", `{"temperature":"not_a_number"}`},
		// Syntax error: triggers "invalid character" JSON errors.
		{"/v1/chat/completions", "syntax_error", `{invalid json`},
		{"/v1/completions", "syntax_error", `{invalid json`},
		{"/v1/embeddings", "syntax_error", `{invalid json`},
	}

	for _, tc := range cases {
		t.Run(tc.endpoint+"/"+tc.name, func(t *testing.T) {
			resp := doPost(t, ts.URL+tc.endpoint, "application/json", tc.body)
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			bodyStr := string(body)

			// The response must NOT contain Go-internal type information.
			leakPatterns := []string{
				"Go struct field",
				"json: cannot unmarshal",
				"reflect.",
				"runtime.",
			}
			for _, pattern := range leakPatterns {
				if strings.Contains(bodyStr, pattern) {
					t.Errorf("response body leaks internal details: contains %q\nbody: %s", pattern, bodyStr)
				}
			}

			// Verify it contains the generic error message.
			var errResp struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(body, &errResp); err != nil {
				t.Fatalf("unmarshal error response: %v", err)
			}
			if errResp.Error.Message != "invalid request body" {
				t.Errorf("error message = %q, want %q", errResp.Error.Message, "invalid request body")
			}
		})
	}
}

func TestParseModelAdapter(t *testing.T) {
	tests := []struct {
		input       string
		wantBase    string
		wantAdapter string
	}{
		{"gemma3-1b", "gemma3-1b", ""},
		{"gemma3-1b:my-lora", "gemma3-1b", "my-lora"},
		{"base:adapter:extra", "base", "adapter:extra"},
		{"", "", ""},
		{":adapter", "", "adapter"},
	}
	for _, tt := range tests {
		base, adapter := ParseModelAdapter(tt.input)
		if base != tt.wantBase || adapter != tt.wantAdapter {
			t.Errorf("ParseModelAdapter(%q) = (%q, %q), want (%q, %q)",
				tt.input, base, adapter, tt.wantBase, tt.wantAdapter)
		}
	}
}

func TestChatCompletions_AdapterNotEnabled(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m) // no adapter cache
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model:my-lora","messages":[{"role":"user","content":"hello"}]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}
	var errResp struct {
		Error struct{ Message string } `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&errResp)
	if errResp.Error.Message != "adapter selection not enabled" {
		t.Errorf("error = %q, want %q", errResp.Error.Message, "adapter selection not enabled")
	}
}

func TestChatCompletions_AdapterNotFound(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m, WithAdapterCache(t.TempDir(), 4))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model:nonexistent","messages":[{"role":"user","content":"hello"}]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}
	var errResp struct {
		Error struct{ Message string } `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&errResp)
	if !strings.Contains(errResp.Error.Message, "adapter not found") {
		t.Errorf("error = %q, want containing %q", errResp.Error.Message, "adapter not found")
	}
}

func TestChatCompletions_WithAdapter(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m, WithAdapterCache(t.TempDir(), 4))

	// Pre-populate the adapter cache with a mock adapter.
	srv.adapterCache.cache.Put("my-lora", &lora.Adapter{
		Rank:        4,
		Alpha:       4.0,
		ScaleFactor: 1.0,
		Layers:      map[string]*lora.Layer{},
	})

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model:my-lora","messages":[{"role":"user","content":"hello"}]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, respBody)
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(chatResp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
}

func TestChatCompletions_NoAdapterUsesBaseModel(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m, WithAdapterCache(t.TempDir(), 4))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	// Request without adapter separator should use base model.
	body := `{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`
	resp := doPost(t, ts.URL+"/v1/chat/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, respBody)
	}
}

func TestCompletions_AdapterNotEnabled(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m) // no adapter cache
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model:my-lora","prompt":"hello"}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}
}

func TestCompletions_WithAdapter(t *testing.T) {
	m := buildTestModel(t)
	srv := NewServer(m, WithAdapterCache(t.TempDir(), 4))

	srv.adapterCache.cache.Put("my-lora", &lora.Adapter{
		Rank:        4,
		Alpha:       4.0,
		ScaleFactor: 1.0,
		Layers:      map[string]*lora.Layer{},
	})

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"test-model:my-lora","prompt":"hello"}`
	resp := doPost(t, ts.URL+"/v1/completions", "application/json", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, body = %s", resp.StatusCode, respBody)
	}
}
