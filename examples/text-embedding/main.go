// Command text-embedding demonstrates extracting text embedding vectors
// from a loaded GGUF model using the inference package.
//
// This example loads a model, generates embeddings for several texts,
// and computes cosine similarity between them. Useful for semantic search,
// clustering, and retrieval-augmented generation (RAG).
//
// Usage:
//
//	go build -o text-embedding ./examples/text-embedding/
//	./text-embedding path/to/model.gguf
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0", "rocm"`)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <model.gguf>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	modelPath := flag.Arg(0)

	// --- Step 1: Load the model ---
	fmt.Println("=== Text Embedding Example ===")
	fmt.Printf("Loading model: %s\n", modelPath)
	fmt.Printf("Device: %s\n\n", *device)

	model, err := inference.LoadFile(modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Printf("Model: %s (%d layers, hidden=%d, vocab=%d)\n\n",
		cfg.Architecture, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize)

	// --- Step 2: Generate embeddings ---
	// The Embed method tokenizes the text, looks up token embeddings from
	// the model's embedding table, mean-pools across tokens, and L2-normalizes
	// the result to produce a unit vector.
	texts := []string{
		"The cat sat on the mat.",
		"A kitten rested on the rug.",
		"Stock prices rose sharply today.",
		"The financial markets surged in early trading.",
	}

	fmt.Println("--- Generating Embeddings ---")
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := model.Embed(text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error embedding text %d: %v\n", i, err)
			os.Exit(1)
		}
		embeddings[i] = emb
		fmt.Printf("Text %d: %q -> %d-dim embedding (first 5: [%.4f, %.4f, %.4f, %.4f, %.4f])\n",
			i, text, len(emb),
			safeGet(emb, 0), safeGet(emb, 1), safeGet(emb, 2),
			safeGet(emb, 3), safeGet(emb, 4))
	}

	// --- Step 3: Compute pairwise cosine similarity ---
	// Since embeddings are L2-normalized, cosine similarity is just the dot product.
	fmt.Println("\n--- Cosine Similarity Matrix ---")
	fmt.Printf("%25s", "")
	for i := range texts {
		fmt.Printf("  Text %d", i)
	}
	fmt.Println()

	for i := range texts {
		fmt.Printf("  Text %d (%20s)", i, truncate(texts[i], 20))
		for j := range texts {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			fmt.Printf("  %6.4f", sim)
		}
		fmt.Println()
	}

	fmt.Println("\n--- Interpretation ---")
	fmt.Println("Texts 0-1 (cats) should have high similarity.")
	fmt.Println("Texts 2-3 (finance) should have high similarity.")
	fmt.Println("Cross-topic pairs should have lower similarity.")

	fmt.Println("\n=== Done ===")
}

// cosineSimilarity computes the cosine similarity between two vectors.
// For L2-normalized vectors, this is equivalent to the dot product.
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

// safeGet returns the value at index i, or 0 if out of bounds.
func safeGet(s []float32, i int) float32 {
	if i < len(s) {
		return s[i]
	}
	return 0
}

// truncate shortens a string to maxLen characters with "..." suffix.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
