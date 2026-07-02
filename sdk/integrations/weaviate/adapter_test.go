package weaviate_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zerfoo/zerfoo/sdk/integrations/weaviate"
)

// fakeEmbedServer returns a test HTTP server that echoes back fake embedding
// vectors for each text in the request.
func fakeEmbedServer(t *testing.T, dims int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		data := make([]map[string]any, len(req.Input))
		vec := make([]float32, dims)
		for i := range vec {
			vec[i] = float32(i) * 0.01
		}
		for i := range req.Input {
			data[i] = map[string]any{
				"index":     i,
				"embedding": vec,
			}
		}

		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{"data": data}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
}

func TestWeaviateAdapter_Embed(t *testing.T) {
	const dims = 64
	srv := fakeEmbedServer(t, dims)
	defer srv.Close()

	adapter := weaviate.NewAdapter(srv.URL, "nomic-embed-text")

	vec, err := adapter.EmbedQuery(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if len(vec) != dims {
		t.Errorf("want vector of length %d, got %d", dims, len(vec))
	}
}

func TestWeaviateAdapter_EmbedDocuments(t *testing.T) {
	const dims = 16
	srv := fakeEmbedServer(t, dims)
	defer srv.Close()

	adapter := weaviate.NewAdapter(srv.URL, "nomic-embed-text")

	docs := []string{"doc one", "doc two", "doc three"}
	vecs, err := adapter.EmbedDocuments(context.Background(), docs)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(vecs) != len(docs) {
		t.Fatalf("want %d vectors, got %d", len(docs), len(vecs))
	}
	for i, v := range vecs {
		if len(v) != dims {
			t.Errorf("vec[%d]: want length %d, got %d", i, dims, len(v))
		}
	}
}

func TestWeaviateAdapter_Embed_Batching(t *testing.T) {
	requestCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		var req struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		data := make([]map[string]any, len(req.Input))
		for i := range req.Input {
			data[i] = map[string]any{"index": i, "embedding": []float32{1.0}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer srv.Close()

	// batchSize=2, 5 docs → 3 requests
	adapter := weaviate.NewAdapter(srv.URL, "model",
		weaviate.WithBatchSize(2))

	docs := make([]string, 5)
	for i := range docs {
		docs[i] = "text"
	}
	if _, err := adapter.EmbedDocuments(context.Background(), docs); err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if requestCount != 3 {
		t.Errorf("want 3 HTTP requests (batch 2+2+1), got %d", requestCount)
	}
}

func TestWeaviateAdapter_Embed_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"type":"server_error","message":"oops"}}`, http.StatusInternalServerError)
	}))
	defer srv.Close()

	adapter := weaviate.NewAdapter(srv.URL, "model")
	_, err := adapter.EmbedQuery(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error from server 500, got nil")
	}
}
