// Command summarization demonstrates text summarization using a GGUF language model.
//
// It reads text from a file or flag, prompts the model to produce a concise summary,
// and prints the result. Useful for condensing long documents, articles, or logs.
//
// Usage:
//
//	go build -o summarization ./examples/summarization/
//	./summarization --model path/to/model.gguf --text "Long text to summarize..."
//	./summarization --model path/to/model.gguf --file article.txt
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
	text := flag.String("text", "", "text to summarize")
	filePath := flag.String("file", "", "path to a text file to summarize")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens in the summary")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <model.gguf> [--text <text> | --file <path>]\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Get the input text from flag or file.
	input := *text
	if input == "" && *filePath != "" {
		data, err := os.ReadFile(*filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read file: %v\n", err)
			os.Exit(1)
		}
		input = string(data)
	}
	if input == "" {
		// Use a built-in example text for demonstration.
		input = exampleText
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

	// Build the summarization prompt.
	prompt := fmt.Sprintf(
		"Summarize the following text in a concise paragraph:\n\n%s\n\nSummary:",
		input,
	)

	result, err := model.Generate(context.Background(), prompt,
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(0.3),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)
}

const exampleText = `Transformers have become the dominant architecture in natural language processing
since their introduction in the 2017 paper "Attention Is All You Need." The key
innovation is the self-attention mechanism, which allows the model to weigh the
importance of different parts of the input when producing each output element.
Unlike recurrent neural networks, transformers process all positions in parallel,
making them significantly faster to train on modern hardware. The architecture
consists of an encoder and decoder, each built from stacked layers of multi-head
attention and feed-forward networks. Pre-training on large text corpora followed
by task-specific fine-tuning has proven remarkably effective, leading to models
like BERT, GPT, and T5 that achieve state-of-the-art results across a wide range
of language tasks including translation, summarization, and question answering.`
