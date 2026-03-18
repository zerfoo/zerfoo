// Recipe 05: Custom Sampling Parameters
//
// Demonstrate how temperature, top-K, top-P, and repetition penalty affect
// text generation. The program generates the same prompt three times with
// different sampling configurations so you can compare the outputs.
//
// Usage:
//
//	go run ./docs/cookbook/05-custom-sampling/ --model path/to/model.gguf
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	prompt := flag.String("prompt", "Write a haiku about concurrency.", "generation prompt")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: custom-sampling --model <model.gguf>")
		os.Exit(1)
	}

	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	ctx := context.Background()

	// Configuration 1: Greedy decoding (temperature=0).
	// Deterministic output -- always picks the highest-probability token.
	fmt.Println("=== Greedy (temperature=0) ===")
	text, err := model.Generate(ctx, *prompt,
		inference.WithMaxTokens(64),
		inference.WithTemperature(0),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)

	// Configuration 2: Creative (high temperature + top-P nucleus sampling).
	// Produces more varied, surprising output.
	fmt.Println("\n=== Creative (temp=1.2, top-P=0.9) ===")
	text, err = model.Generate(ctx, *prompt,
		inference.WithMaxTokens(64),
		inference.WithTemperature(1.2),
		inference.WithTopP(0.9),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)

	// Configuration 3: Focused (low temperature + top-K).
	// Picks from a narrow set of likely tokens for coherent output.
	fmt.Println("\n=== Focused (temp=0.3, top-K=10) ===")
	text, err = model.Generate(ctx, *prompt,
		inference.WithMaxTokens(64),
		inference.WithTemperature(0.3),
		inference.WithTopK(10),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)
}
