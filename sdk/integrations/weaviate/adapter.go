// Package weaviate provides an adapter for generating embeddings via Zerfoo's
// OpenAI-compatible HTTP API and inserting them into a Weaviate vector
// database client.
//
// The adapter does not import the Weaviate Go client directly. It produces
// []float32 embedding vectors from text using Zerfoo's /v1/embeddings
// endpoint. Callers can then pass those vectors to Weaviate's batch import or
// search API.
//
// Usage:
//
//	emb := weaviate.NewAdapter("http://localhost:8080", "nomic-embed-text")
//	vecs, err := emb.EmbedDocuments(ctx, []string{"doc one", "doc two"})
package weaviate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Adapter wraps Zerfoo's /v1/embeddings endpoint and produces float32 vectors
// suitable for insertion into Weaviate.
type Adapter struct {
	baseURL    string
	model      string
	httpClient *http.Client
	// BatchSize controls how many texts are sent per API request. Default 32.
	BatchSize int
}

// AdapterOption configures an Adapter.
type AdapterOption func(*Adapter)

// WithBatchSize sets the number of texts to embed in a single request.
func WithBatchSize(n int) AdapterOption {
	return func(a *Adapter) { a.BatchSize = n }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(c *http.Client) AdapterOption {
	return func(a *Adapter) { a.httpClient = c }
}

// NewAdapter creates an Adapter pointing at a running Zerfoo serve instance.
//
// baseURL is the server root (e.g. "http://localhost:8080").
// model is the embedding model identifier forwarded in the request body.
func NewAdapter(baseURL, model string, opts ...AdapterOption) *Adapter {
	a := &Adapter{
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      model,
		BatchSize:  32,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
	for _, o := range opts {
		o(a)
	}
	return a
}

// --- OpenAI-compatible embedding request / response types (internal) ---

type embeddingRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string or []string
}

type embeddingObject struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type embeddingResponse struct {
	Data  []embeddingObject `json:"data"`
	Error *apiError         `json:"error,omitempty"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// EmbedQuery embeds a single query string and returns its vector.
// This mirrors the LangChain-Go schema.Embedder.EmbedQuery signature.
func (a *Adapter) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	vecs, err := a.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedDocuments embeds multiple texts and returns one vector per text.
// Texts are sent in batches of BatchSize to avoid oversized requests.
// This mirrors the LangChain-Go schema.Embedder.EmbedDocuments signature.
func (a *Adapter) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += a.BatchSize {
		end := i + a.BatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]
		vecs, err := a.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("weaviate adapter: batch [%d:%d]: %w", i, end, err)
		}
		results = append(results, vecs...)
	}
	return results, nil
}

// embedBatch sends a single /v1/embeddings request for the given texts.
func (a *Adapter) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := embeddingRequest{
		Model: a.model,
		Input: texts,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("weaviate adapter: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		a.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("weaviate adapter: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("weaviate adapter: http: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("weaviate adapter: read body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("weaviate adapter: server returned %d: %s", resp.StatusCode, raw)
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(raw, &embResp); err != nil {
		return nil, fmt.Errorf("weaviate adapter: decode response: %w", err)
	}
	if embResp.Error != nil {
		return nil, fmt.Errorf("weaviate adapter: api error (%s): %s", embResp.Error.Type, embResp.Error.Message)
	}
	if len(embResp.Data) != len(texts) {
		return nil, fmt.Errorf("weaviate adapter: expected %d embeddings, got %d", len(texts), len(embResp.Data))
	}

	// The API may return embeddings in any order; sort by index.
	ordered := make([][]float32, len(texts))
	for _, obj := range embResp.Data {
		if obj.Index < 0 || obj.Index >= len(texts) {
			return nil, fmt.Errorf("weaviate adapter: invalid embedding index %d", obj.Index)
		}
		ordered[obj.Index] = obj.Embedding
	}
	return ordered, nil
}
