// Command rag demonstrates retrieval-augmented generation using Zerfoo.
//
// It embeds a hardcoded 5-document corpus, accepts a query, finds the top-3
// most similar documents via cosine similarity, and passes them as context
// to the model for generation.
//
// Usage:
//
//	go build -o rag ./examples/rag/
//	./rag --model path/to/model.gguf
//	./rag --model path/to/model.gguf --query "How does Go handle concurrency?"
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo"
)

var corpus = []string{
	"Go was created at Google in 2009 by Robert Griesemer, Rob Pike, and Ken Thompson.",
	"Go uses goroutines for concurrency, which are lightweight threads managed by the Go runtime.",
	"The Go standard library includes packages for HTTP, JSON, cryptography, and testing.",
	"Go modules were introduced in Go 1.11 to manage dependencies via go.mod files.",
	"Go's garbage collector uses a concurrent tri-color mark-and-sweep algorithm.",
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "How does Go handle concurrency?", "search query")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: rag --model <path> --query <text>")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer m.Close()

	// Embed the corpus documents.
	corpusEmbeds, err := m.Embed(corpus)
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed corpus:", err)
		os.Exit(1)
	}

	// Embed the query.
	queryEmbeds, err := m.Embed([]string{*query})
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed query:", err)
		os.Exit(1)
	}
	queryEmbed := queryEmbeds[0]

	// Rank corpus documents by cosine similarity to the query.
	type ranked struct {
		doc   string
		score float32
	}
	results := make([]ranked, len(corpus))
	for i, e := range corpusEmbeds {
		results[i] = ranked{corpus[i], queryEmbed.CosineSimilarity(e)}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	// Build a prompt from the top-3 most relevant documents.
	top := results[:3]
	fmt.Fprintf(os.Stderr, "Top-3 documents (by similarity):\n")
	for i, r := range top {
		fmt.Fprintf(os.Stderr, "  %d. [%.4f] %s\n", i+1, r.score, r.doc)
	}
	fmt.Fprintln(os.Stderr)

	var contextParts []string
	for _, r := range top {
		contextParts = append(contextParts, r.doc)
	}
	prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s\nAnswer:", strings.Join(contextParts, "\n"), *query)

	response, err := m.Chat(prompt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Println(response)
}
