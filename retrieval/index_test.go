package retrieval

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type stubEmbedder struct{ vectors map[string][]float32 }

func (s stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i] = s.vectors[text]
	}
	return out, nil
}

type reverseReranker struct{}

func (reverseReranker) Rerank(_ context.Context, _ string, docs []Document) ([]float32, error) {
	scores := make([]float32, len(docs))
	for i := range docs {
		scores[i] = float32(i)
		if docs[i].Body != "secret" {
			return nil, errors.New("reranker missing document body")
		}
	}
	return scores, nil
}

func TestLexicalSearchAndFetch(t *testing.T) {
	idx, err := NewIndex(context.Background(), []Document{
		{ID: "go", Text: "Create a Go HTTP server", Body: "full Go instructions"},
		{ID: "react", Text: "Build a React application", Body: "full React instructions"},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), "Go server", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Document.ID != "go" || got[0].Document.Body != "" || got[0].LexicalRank != 1 {
		t.Fatalf("unexpected results: %+v", got)
	}
	if d, ok := idx.Get("go"); !ok || d.Body != "full Go instructions" {
		t.Fatalf("unexpected fetch: %+v, %v", d, ok)
	}
	if got, err := idx.Search(context.Background(), "unknown", 2); err != nil || len(got) != 0 {
		t.Fatalf("no-match query: %+v, %v", got, err)
	}
}

func TestHybridRetrievalAndReranking(t *testing.T) {
	docs := []Document{{ID: "a", Text: "write API", Body: "secret"}, {ID: "b", Text: "draw chart", Body: "secret"}}
	embed := stubEmbedder{map[string][]float32{"write API": {1, 0}, "draw chart": {0, 1}, "write plot data": {0, 2}}}
	idx, err := NewIndex(context.Background(), docs, Options{Embedder: embed, Reranker: reverseReranker{}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), "write plot data", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Document.ID != "b" || got[1].Document.ID != "a" || got[0].DenseRank != 1 || got[1].LexicalRank != 1 {
		t.Fatalf("unexpected reranking: %+v", got)
	}
}

func TestInvalidIndexAndEmbeddings(t *testing.T) {
	ctx := context.Background()
	if _, err := NewIndex(ctx, []Document{{ID: "x"}, {ID: "x"}}, Options{}); err == nil {
		t.Fatal("accepted duplicate ID")
	}
	if _, err := NewIndex(ctx, []Document{{ID: "x", Text: "x"}}, Options{Embedder: stubEmbedder{map[string][]float32{"x": {0, 0}}}}); err == nil {
		t.Fatal("accepted zero vector")
	}
	if _, err := NewIndex(ctx, []Document{{ID: "x", Text: "x"}}, Options{Embedder: stubEmbedder{map[string][]float32{"x": {float32(math.NaN())}}}}); err == nil {
		t.Fatal("accepted NaN")
	}
	idx, err := NewIndex(ctx, []Document{{ID: "x", Text: "x"}}, Options{Embedder: stubEmbedder{map[string][]float32{"x": {1, 0}, "query": {1}}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := idx.Search(ctx, "query", 1); err == nil {
		t.Fatal("accepted query dimension mismatch")
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := idx.Search(canceled, "query", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestDeterministicTies(t *testing.T) {
	idx, err := NewIndex(context.Background(), []Document{{ID: "b", Text: "same"}, {ID: "a", Text: "same"}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := idx.Search(context.Background(), "same", 2)
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{got[0].Document.ID, got[1].Document.ID}
	if !reflect.DeepEqual(ids, []string{"a", "b"}) {
		t.Fatal(ids)
	}
}
