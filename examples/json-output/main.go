// Command json-output demonstrates grammar-guided decoding with a JSON schema.
//
// The model's output is constrained to valid JSON matching the given schema,
// ensuring well-formed structured output without post-processing.
//
// Two approaches are shown:
//  1. High-level: zerfoo.WithSchema (one-line, recommended for most use cases)
//  2. Low-level: grammar.Convert + inference.WithGrammar (for custom pipelines)
//
// Usage:
//
//	go build -o json-output ./examples/json-output/
//	./json-output --model path/to/model.gguf
//	./json-output --model google/gemma-3-1b --prompt "Generate a person named Bob who is 25"
//	./json-output --model path/to/model.gguf --low-level
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo"
	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	prompt := flag.String("prompt", "Generate a JSON object for a person named Alice who is 30 years old.", "generation prompt")
	lowLevel := flag.Bool("low-level", false, "use low-level grammar.Convert + inference.WithGrammar API")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: json-output --model <path>")
		os.Exit(1)
	}

	// Define the JSON schema that constrains model output.
	schema := grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "number"},
		},
		Required: []string{"name", "age"},
	}

	if *lowLevel {
		runLowLevel(*modelPath, *prompt, &schema)
	} else {
		runHighLevel(*modelPath, *prompt, schema)
	}
}

// runHighLevel uses the one-line zerfoo.WithSchema API.
func runHighLevel(modelPath, prompt string, schema grammar.JSONSchema) {
	m, err := zerfoo.Load(modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer m.Close()

	result, err := m.Generate(context.Background(), prompt, zerfoo.WithSchema(schema))
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Println(result.Text)
}

// runLowLevel demonstrates grammar.Convert and inference.WithGrammar
// for use cases that need direct control over the inference pipeline.
func runLowLevel(modelPath, prompt string, schema *grammar.JSONSchema) {
	// Step 1: Convert the JSON schema into a grammar state machine.
	g, err := grammar.Convert(schema)
	if err != nil {
		fmt.Fprintln(os.Stderr, "grammar convert:", err)
		os.Exit(1)
	}

	// Step 2: Load the model using the inference package directly.
	model, err := inference.LoadFile(modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer model.Close()

	// Step 3: Generate with the grammar constraint applied.
	text, err := model.Generate(context.Background(), prompt, inference.WithGrammar(g))
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Println(text)
}
