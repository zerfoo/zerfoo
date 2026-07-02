// Command weaviate-search demonstrates using the Zerfoo Weaviate adapter to
// embed a corpus of documents and perform cosine-similarity semantic search
// without requiring a live Weaviate instance.
//
// Start a Zerfoo server first:
//
//	zerfoo serve --model path/to/embed-model.gguf --port 8080
//
// Then run this example:
//
//	go run ./examples/weaviate-search/ --server http://localhost:8080 --model nomic-embed-text
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/zerfoo/zerfoo/sdk/integrations/weaviate"
)

// corpus is a small set of documents to embed and search.
var corpus = []string{
	"Go is a statically typed, compiled programming language designed at Google.",
	"Machine learning models can be deployed as HTTP microservices.",
	"Zerfoo is a production-grade ML inference framework written in Go.",
	"Vector databases store embeddings for semantic similarity search.",
	"The transformer architecture revolutionised natural language processing.",
	"GGUF is a binary model format compatible with llama.cpp and Zerfoo.",
	"Weaviate is an open-source vector database with a GraphQL API.",
	"Embeddings map text into a high-dimensional semantic space.",
}

func main() {
	server := flag.String("server", "http://localhost:8080", "Zerfoo server URL")
	model := flag.String("model", "nomic-embed-text", "Embedding model name")
	query := flag.String("query", "How do I run ML inference in Go?", "Search query")
	topK := flag.Int("top-k", 3, "Number of results to return")
	flag.Parse()

	ctx := context.Background()
	emb := weaviate.NewAdapter(*server, *model)

	fmt.Printf("Embedding %d documents...\n", len(corpus))
	docVecs, err := emb.EmbedDocuments(ctx, corpus)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed documents: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Embedding query: %q\n", *query)
	queryVec, err := emb.EmbedQuery(ctx, *query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed query: %v\n", err)
		os.Exit(1)
	}

	// Rank documents by cosine similarity to the query.
	type result struct {
		idx   int
		score float64
	}
	results := make([]result, len(corpus))
	for i, dv := range docVecs {
		results[i] = result{i, cosineSimilarity(queryVec, dv)}
	}
	sort.Slice(results, func(a, b int) bool { return results[a].score > results[b].score })

	k := *topK
	if k > len(results) {
		k = len(results)
	}

	fmt.Printf("\nTop %d results:\n", k)
	for rank, r := range results[:k] {
		fmt.Printf("  %d. [score=%.4f] %s\n", rank+1, r.score, corpus[r.idx])
	}
}

// cosineSimilarity returns the cosine similarity between two float32 vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
