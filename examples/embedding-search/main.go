// Command embedding-search demonstrates semantic search using model embeddings.
//
// It embeds a corpus of documents and a query, then ranks documents by cosine
// similarity to find the most relevant matches. This is the retrieval half of
// a RAG (retrieval-augmented generation) system.
//
// Usage:
//
//	go build -o embedding-search ./examples/embedding-search/
//	./embedding-search --model path/to/model.gguf
//	./embedding-search --model path/to/model.gguf --query "memory management"
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/zerfoo/zerfoo"
)

// corpus is a small set of documents to search over.
var corpus = []string{
	"Go's garbage collector is a concurrent, tri-color, mark-sweep collector.",
	"Goroutines are multiplexed onto a small number of OS threads by the Go scheduler.",
	"The sync.Mutex type provides mutual exclusion for protecting shared state.",
	"Go modules use go.mod and go.sum files to manage versioned dependencies.",
	"The net/http package provides HTTP client and server implementations.",
	"Channels are the primary mechanism for communication between goroutines.",
	"The context package carries deadlines, cancellation signals, and request-scoped values.",
	"Go interfaces are satisfied implicitly -- no 'implements' keyword is needed.",
	"The testing package provides support for automated unit and benchmark tests.",
	"Defer statements schedule a function call to run when the enclosing function returns.",
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "How does Go handle memory?", "search query")
	topN := flag.Int("top", 3, "number of results to show")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: embedding-search --model <path> [--query <text>] [--top N]")
		os.Exit(1)
	}

	// Load the model.
	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Embed all corpus documents.
	fmt.Fprintf(os.Stderr, "Embedding %d documents...\n", len(corpus))
	corpusEmbeds, err := m.Embed(corpus)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed corpus: %v\n", err)
		os.Exit(1)
	}

	// Embed the query.
	queryEmbeds, err := m.Embed([]string{*query})
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed query: %v\n", err)
		os.Exit(1)
	}
	queryEmbed := queryEmbeds[0]

	// Rank by cosine similarity.
	type result struct {
		index int
		doc   string
		score float32
	}
	results := make([]result, len(corpus))
	for i, emb := range corpusEmbeds {
		results[i] = result{
			index: i,
			doc:   corpus[i],
			score: queryEmbed.CosineSimilarity(emb),
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	// Print the top-N results.
	fmt.Printf("Query: %q\n\n", *query)
	n := *topN
	if n > len(results) {
		n = len(results)
	}
	for i := 0; i < n; i++ {
		r := results[i]
		fmt.Printf("  %d. [score=%.4f] %s\n", i+1, r.score, r.doc)
	}
}
