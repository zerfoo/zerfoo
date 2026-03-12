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

	// Load GGUF separately to inspect config.
	log.Printf("Loading GGUF from %s...", *modelPath)
	gm, err := inference.LoadGGUF(*modelPath)
	if err != nil {
		log.Fatalf("LoadGGUF: %v", err)
	}
	cfg := gm.Config
	log.Printf("Architecture: %s", cfg.Architecture)
	log.Printf("HiddenSize: %d, NumLayers: %d", cfg.HiddenSize, cfg.NumLayers)
	log.Printf("NumHeads: %d, NumKVHeads: %d, HeadDim: %d", cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	log.Printf("IntermediateSize: %d, VocabSize: %d", cfg.IntermediateSize, cfg.VocabSize)
	log.Printf("RopeTheta: %f, LocalRopeTheta: %f", cfg.RopeTheta, cfg.LocalRopeTheta)
	log.Printf("SlidingWindow: %d, SlidingWindowPattern: %d", cfg.SlidingWindow, cfg.SlidingWindowPattern)
	log.Printf("LogitSoftcap: %f, RMSNormEps: %e", cfg.LogitSoftcap, cfg.RMSNormEps)
	log.Printf("MaxSeqLen: %d", cfg.MaxSeqLen)

	// Print tensor names and shapes.
	log.Printf("Loaded %d tensors:", len(gm.Tensors))
	for name, t := range gm.Tensors {
		log.Printf("  %s: shape=%v", name, t.Shape())
	}

	// Print first few values of embedding weight to verify dequantization.
	if emb, ok := gm.Tensors["model.embed_tokens.weight"]; ok {
		data := emb.Data()
		n := 10
		if len(data) < n {
			n = len(data)
		}
		log.Printf("Embedding weight first %d values: %v", n, data[:n])
	}

	log.Printf("Building model...")
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
