// Recipe 01: Basic Text Generation
//
// Load a GGUF model and generate a text completion with a single function call.
// This is the simplest way to run inference with Zerfoo.
//
// Usage:
//
//	go run ./docs/cookbook/01-basic-text-generation/ --model path/to/model.gguf
//	go run ./docs/cookbook/01-basic-text-generation/ --model google/gemma-3-1b
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	prompt := flag.String("prompt", "Explain goroutines in one paragraph.", "generation prompt")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: basic-text-generation --model <path-or-id> [--prompt <text>]")
		os.Exit(1)
	}

	// Load the model. Accepts a local GGUF path or a HuggingFace model ID
	// like "google/gemma-3-1b". Remote models are downloaded and cached.
	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	// Generate a completion. The result includes the generated text,
	// token count, and wall-clock duration.
	result, err := m.Generate(context.Background(), *prompt,
		zerfoo.WithGenMaxTokens(256),
		zerfoo.WithGenTemperature(0.7),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result.Text)
	fmt.Fprintf(os.Stderr, "\n[%d tokens in %s]\n", result.TokenCount, result.Duration)
}
