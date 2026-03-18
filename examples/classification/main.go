// Command classification demonstrates text classification using grammar-constrained
// decoding to guarantee a valid JSON response with a category label.
//
// The model is prompted to classify text into one of several categories, and
// grammar-guided decoding ensures the output is always valid JSON matching the
// expected schema. This eliminates the need for fragile output parsing.
//
// Usage:
//
//	go build -o classification ./examples/classification/
//	./classification --model path/to/model.gguf --text "I love this product!"
//	./classification --model path/to/model.gguf --text "The package arrived damaged."
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	text := flag.String("text", "I absolutely love this product, it exceeded my expectations!", "text to classify")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <model.gguf> [--text <text>]\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Define a JSON schema for the classification output.
	// The model must respond with a JSON object containing "label" and "confidence".
	schema := grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"label": {
				Type: "string",
				Enum: []any{"positive", "negative", "neutral"},
			},
			"confidence": {
				Type: "number",
			},
		},
		Required: []string{"label", "confidence"},
	}

	// Convert the schema to a grammar state machine for constrained decoding.
	g, err := grammar.Convert(&schema)
	if err != nil {
		fmt.Fprintf(os.Stderr, "grammar convert: %v\n", err)
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

	// Build the classification prompt.
	prompt := fmt.Sprintf(
		`Classify the sentiment of the following text as "positive", "negative", or "neutral".
Respond with a JSON object containing "label" and "confidence" (0 to 1).

Text: %s

JSON:`, *text)

	// Generate with grammar constraint -- the output is guaranteed to be valid JSON.
	result, err := model.Generate(context.Background(), prompt,
		inference.WithMaxTokens(64),
		inference.WithTemperature(0.1),
		inference.WithGrammar(g),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Text:   %s\n", *text)
	fmt.Printf("Result: %s\n", result)
}
