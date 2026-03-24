package serve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference/sentiment"
)

// mockClassifier implements Classifier for testing.
type mockClassifier struct {
	results []sentiment.SentimentResult
	err     error
}

func (m *mockClassifier) Classify(_ context.Context, texts []string) ([]sentiment.SentimentResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.results != nil {
		return m.results, nil
	}
	// Default: return one result per input text.
	out := make([]sentiment.SentimentResult, len(texts))
	for i := range texts {
		out[i] = sentiment.SentimentResult{
			Label: "positive",
			Score: 0.92,
		}
	}
	return out, nil
}

func newClassifyServer(t *testing.T, c Classifier) *httptest.Server {
	t.Helper()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithClassifier(c))
	return httptest.NewServer(srv.Handler())
}

func TestClassifyEndpoint(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	body := `{"model":"finbert","input":["stocks are up today"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Data) != 1 {
		t.Fatalf("data length = %d, want 1", len(result.Data))
	}
	if result.Data[0].Label != "positive" {
		t.Errorf("label = %q, want %q", result.Data[0].Label, "positive")
	}
	if result.Data[0].Score != 0.92 {
		t.Errorf("score = %f, want 0.92", result.Data[0].Score)
	}
	if result.Data[0].Index != 0 {
		t.Errorf("index = %d, want 0", result.Data[0].Index)
	}
	if result.Model != "finbert" {
		t.Errorf("model = %q, want %q", result.Model, "finbert")
	}
}

func TestClassifyEndpoint_Batch(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	inputs := make([]string, 5)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("text %d", i)
	}
	reqBody, _ := json.Marshal(ClassifyRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Data) != 5 {
		t.Fatalf("data length = %d, want 5", len(result.Data))
	}
	for i, d := range result.Data {
		if d.Index != i {
			t.Errorf("data[%d].Index = %d, want %d", i, d.Index, i)
		}
	}
}

func TestClassifyEndpoint_EmptyInput(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	body := `{"model":"finbert","input":[]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestClassifyEndpoint_ExceedMaxBatch(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	inputs := make([]string, 257)
	for i := range inputs {
		inputs[i] = "text"
	}
	reqBody, _ := json.Marshal(ClassifyRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestClassifyEndpoint_NotConfigured(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl) // no classifier
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", resp.StatusCode)
	}
}

func TestClassifyEndpoint_ClassifyError(t *testing.T) {
	sc := &mockClassifier{err: errors.New("model error")}
	ts := newClassifyServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestClassifyEndpoint_InvalidJSON(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/classify", "application/json", "not json")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestClassifyEndpoint_ModelFallback(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	body := `{"input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Model != "test-model" {
		t.Errorf("model = %q, want %q", result.Model, "test-model")
	}
}

func TestClassifyEndpoint_CustomResults(t *testing.T) {
	sc := &mockClassifier{
		results: []sentiment.SentimentResult{
			{Label: "negative", Score: 0.85},
			{Label: "neutral", Score: 0.60},
		},
	}
	ts := newClassifyServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["bad news","nothing special"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Data) != 2 {
		t.Fatalf("data length = %d, want 2", len(result.Data))
	}
	if result.Data[0].Label != "negative" {
		t.Errorf("data[0].label = %q, want %q", result.Data[0].Label, "negative")
	}
	if result.Data[1].Label != "neutral" {
		t.Errorf("data[1].label = %q, want %q", result.Data[1].Label, "neutral")
	}
}

func TestClassifyEndpoint_OOMError(t *testing.T) {
	sc := &mockClassifier{err: errors.New("out of memory")}
	ts := newClassifyServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestClassifyEndpoint_MethodNotAllowed(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/v1/classify", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusMethodNotAllowed {
		data, _ := io.ReadAll(resp.Body)
		t.Errorf("status = %d, want 405; body: %s", resp.StatusCode, data)
	}
}

func TestClassifyEndpoint_MaxBatchBoundary(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	// Exactly 256 should succeed.
	inputs := make([]string, 256)
	for i := range inputs {
		inputs[i] = "text"
	}
	reqBody, _ := json.Marshal(ClassifyRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d for 256 inputs, want 200; body: %s", resp.StatusCode, data)
	}

	var result ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Data) != 256 {
		t.Errorf("data length = %d, want 256", len(result.Data))
	}
}

func TestClassifyEndpoint_OversizedBody(t *testing.T) {
	ts := newClassifyServer(t, &mockClassifier{})
	defer ts.Close()

	bigValue := strings.Repeat("x", 11<<20) // 11 MB string
	oversized := `{"model":"finbert","input":["` + bigValue + `"]}`

	resp := doPost(t, ts.URL+"/v1/classify", "application/json", oversized)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want %d; body = %s", resp.StatusCode, http.StatusRequestEntityTooLarge, body)
	}
}

func TestClassifyEndpoint_ErrorNotLeaked(t *testing.T) {
	secret := "/var/data/models/secret-path/weights.bin: no such file"
	sc := &mockClassifier{err: errors.New(secret)}
	ts := newClassifyServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/classify", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	data, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(data), secret) {
		t.Fatalf("internal error details leaked to client: %s", data)
	}
	if strings.Contains(string(data), "/var/data") {
		t.Fatalf("file path leaked to client: %s", data)
	}
}
