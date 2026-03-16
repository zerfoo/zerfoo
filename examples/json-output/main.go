// Command json-output demonstrates grammar-guided decoding with a JSON schema.
//
// The model's output is constrained to valid JSON matching the given schema,
// ensuring well-formed structured output without post-processing.
//
// Usage:
//
//	go build -o json-output ./examples/json-output/
//	./json-output --model path/to/model.gguf
//	./json-output --model google/gemma-3-1b --prompt "Generate a person named Bob who is 25"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo"
	"github.com/zerfoo/zerfoo/generate/grammar"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")
	prompt := flag.String("prompt", "Generate a JSON object for a person named Alice who is 30 years old.", "generation prompt")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: json-output --model <path>")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	defer m.Close()

	schema := grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "number"},
		},
		Required: []string{"name", "age"},
	}

	result, err := m.Generate(context.Background(), *prompt, zerfoo.WithSchema(schema))
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Println(result.Text)
}
