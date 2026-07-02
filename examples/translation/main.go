// Command translation demonstrates text translation using a GGUF language model.
//
// It translates text between languages by prompting the model with a translation
// instruction. Works best with multilingual models (Qwen, Gemma, Llama 3).
//
// Usage:
//
//	go build -o translation ./examples/translation/
//	./translation --model path/to/model.gguf --text "Hello, world!" --target French
//	./translation --model path/to/model.gguf --text "Bonjour le monde" --source French --target English
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
	text := flag.String("text", "The quick brown fox jumps over the lazy dog.", "text to translate")
	source := flag.String("source", "English", "source language")
	target := flag.String("target", "French", "target language")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <model.gguf> --text <text> --target <language>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Load the model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Build the translation prompt.
	prompt := fmt.Sprintf(
		"Translate the following text from %s to %s.\n\n%s: %s\n\n%s:",
		*source, *target, *source, *text, *target,
	)

	result, err := model.Generate(context.Background(), prompt,
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(0.3),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s -> %s\n", *source, *target)
	fmt.Printf("Input:  %s\n", *text)
	fmt.Printf("Output: %s\n", result)
}
