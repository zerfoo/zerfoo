package serve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zerfoo/zerfoo/inference/sentiment"
)

// mockSentiment implements SentimentClassifier for testing.
type mockSentiment struct {
	results []sentiment.SentimentResult
	err     error
}

func (m *mockSentiment) Classify(_ context.Context, texts []string) ([]sentiment.SentimentResult, error) {
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

func newSentimentServer(t *testing.T, sc SentimentClassifier) *httptest.Server {
	t.Helper()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithSentiment(sc))
	return httptest.NewServer(srv.Handler())
}

func TestSentimentEndpoint(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	body := `{"model":"finbert","input":["stocks are up today"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result SentimentResponse
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

func TestSentimentEndpoint_Batch(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	inputs := make([]string, 5)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("text %d", i)
	}
	reqBody, _ := json.Marshal(SentimentRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result SentimentResponse
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

func TestSentimentEndpoint_EmptyInput(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	body := `{"model":"finbert","input":[]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestSentimentEndpoint_ExceedMaxBatch(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	inputs := make([]string, 257)
	for i := range inputs {
		inputs[i] = "text"
	}
	reqBody, _ := json.Marshal(SentimentRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestSentimentEndpoint_NotConfigured(t *testing.T) {
	mdl := buildTestModel(t)
	srv := NewServer(mdl) // no sentiment classifier
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", resp.StatusCode)
	}
}

func TestSentimentEndpoint_ClassifyError(t *testing.T) {
	sc := &mockSentiment{err: errors.New("model error")}
	ts := newSentimentServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
}

func TestSentimentEndpoint_InvalidJSON(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", "not json")
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestSentimentEndpoint_ModelFallback(t *testing.T) {
	// When model field is empty, should fall back to loaded model's ID.
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	body := `{"input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result SentimentResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if result.Model != "test-model" {
		t.Errorf("model = %q, want %q", result.Model, "test-model")
	}
}

func TestSentimentEndpoint_CustomResults(t *testing.T) {
	sc := &mockSentiment{
		results: []sentiment.SentimentResult{
			{Label: "negative", Score: 0.85},
			{Label: "neutral", Score: 0.60},
		},
	}
	ts := newSentimentServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["bad news","nothing special"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result SentimentResponse
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

func TestSentimentEndpoint_OOMError(t *testing.T) {
	sc := &mockSentiment{err: errors.New("out of memory")}
	ts := newSentimentServer(t, sc)
	defer ts.Close()

	body := `{"model":"finbert","input":["hello"]}`
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestSentimentEndpoint_MethodNotAllowed(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, ts.URL+"/v1/sentiment", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusMethodNotAllowed {
		// Go 1.22+ mux returns 405 for wrong method on registered pattern.
		data, _ := io.ReadAll(resp.Body)
		t.Errorf("status = %d, want 405; body: %s", resp.StatusCode, data)
	}
}

func TestSentimentEndpoint_MaxBatchBoundary(t *testing.T) {
	ts := newSentimentServer(t, &mockSentiment{})
	defer ts.Close()

	// Exactly 256 should succeed.
	inputs := make([]string, 256)
	for i := range inputs {
		inputs[i] = "text"
	}
	reqBody, _ := json.Marshal(SentimentRequest{Model: "finbert", Input: inputs})
	resp := doPost(t, ts.URL+"/v1/sentiment", "application/json", string(reqBody))
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d for 256 inputs, want 200; body: %s", resp.StatusCode, data)
	}

	var result SentimentResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(result.Data) != 256 {
		t.Errorf("data length = %d, want 256", len(result.Data))
	}
}

