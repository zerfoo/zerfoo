package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file")
	prompt := flag.String("prompt", "What is 2+2?", "Prompt text")
	maxTokens := flag.Int("max-tokens", 20, "Max tokens to generate")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("--model is required")
	}

	log.Printf("Loading model from %s...", *modelPath)
	start := time.Now()
	mdl, err := inference.LoadFile(*modelPath)
	if err != nil {
		log.Fatalf("LoadFile: %v", err)
	}
	log.Printf("Model loaded in %v", time.Since(start))

	log.Printf("Prompt: %q", *prompt)
	log.Printf("Generating %d tokens...", *maxTokens)

	genStart := time.Now()
	tokenCount := 0
	err = mdl.GenerateStream(context.Background(), *prompt, generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			fmt.Print(token)
			tokenCount++
		}
		return nil
	}), inference.WithMaxTokens(*maxTokens), inference.WithTemperature(0))
	fmt.Println()

	elapsed := time.Since(genStart)
	if err != nil {
		log.Fatalf("Generate: %v", err)
	}
	log.Printf("Generated %d tokens in %v (%.1f tok/s)", tokenCount, elapsed, float64(tokenCount)/elapsed.Seconds())
}
