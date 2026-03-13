package serve

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/registry"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
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

func buildErrorModel(t *testing.T) *inference.Model {
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
	node := &errorNode{}
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
