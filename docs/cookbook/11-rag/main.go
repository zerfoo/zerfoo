// Recipe 11: Retrieval-Augmented Generation (RAG)
//
// Combine embedding-based retrieval with text generation. The pipeline:
//  1. Embed a corpus of documents
//  2. Embed the user's query
//  3. Find the most relevant documents by cosine similarity
//  4. Pass them as context to the model for grounded generation
//
// This is the standard RAG pattern used in knowledge-base chatbots,
// documentation assistants, and question-answering systems.
//
// Usage:
//
//	go run ./docs/cookbook/11-rag/ --model path/to/model.gguf
//	go run ./docs/cookbook/11-rag/ --model path/to/model.gguf --query "How do I use channels?"
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo"
)

// corpus simulates a knowledge base. In production, this would be loaded
// from a database, vector store, or file system.
var corpus = []string{
	"Goroutines are lightweight threads managed by the Go runtime. Launch one with the go keyword.",
	"Channels provide typed, thread-safe communication between goroutines. Use make(chan T) to create one.",
	"The select statement lets a goroutine wait on multiple channel operations simultaneously.",
	"sync.WaitGroup coordinates goroutines: call Add before launching, Done when finished, Wait to block.",
	"The context package carries deadlines, cancellation signals, and request-scoped values across API boundaries.",
	"sync.Mutex and sync.RWMutex protect shared state when channels are not the right fit.",
	"Go's race detector (go test -race) finds data races at runtime during testing.",
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	query := flag.String("query", "How do I coordinate multiple goroutines?", "user question")
	topK := flag.Int("top-k", 3, "number of documents to retrieve")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: rag --model <path> [--query <text>]")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Step 1: Embed the corpus.
	corpusEmbeds, err := m.Embed(corpus)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed corpus: %v\n", err)
		os.Exit(1)
	}

	// Step 2: Embed the query.
	queryEmbeds, err := m.Embed([]string{*query})
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed query: %v\n", err)
		os.Exit(1)
	}
	qe := queryEmbeds[0]

	// Step 3: Retrieve the top-K most relevant documents.
	type ranked struct {
		doc   string
		score float32
	}
	results := make([]ranked, len(corpus))
	for i, e := range corpusEmbeds {
		results[i] = ranked{corpus[i], qe.CosineSimilarity(e)}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })

	k := *topK
	if k > len(results) {
		k = len(results)
	}
	top := results[:k]

	fmt.Fprintf(os.Stderr, "Retrieved %d documents:\n", k)
	for i, r := range top {
		fmt.Fprintf(os.Stderr, "  %d. [%.4f] %s\n", i+1, r.score, r.doc)
	}
	fmt.Fprintln(os.Stderr)

	// Step 4: Build a grounded prompt and generate.
	var docs []string
	for _, r := range top {
		docs = append(docs, "- "+r.doc)
	}
	prompt := fmt.Sprintf(
		"Answer the question using only the provided context.\n\nContext:\n%s\n\nQuestion: %s\nAnswer:",
		strings.Join(docs, "\n"), *query,
	)

	response, err := m.Chat(prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(response)
}
