// Command inference demonstrates loading a GGUF model and generating text.
//
// Usage:
//
//	go build -o inference ./examples/inference/
//	./inference path/to/model.gguf "Your prompt here"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0", "rocm"`)
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")
	temperature := flag.Float64("temperature", 0.7, "sampling temperature (0 = greedy)")
	topK := flag.Int("top-k", 0, "top-K sampling (0 = disabled)")
	topP := flag.Float64("top-p", 1.0, "top-P nucleus sampling (1.0 = disabled)")
	stream := flag.Bool("stream", false, "stream tokens as they are generated")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <model.gguf> <prompt>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 2 {
		flag.Usage()
		os.Exit(1)
	}

	modelPath := flag.Arg(0)
	prompt := flag.Arg(1)

	// Load the GGUF model file.
	model, err := inference.LoadFile(modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s model (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Build generation options.
	var genOpts []inference.GenerateOption
	genOpts = append(genOpts, inference.WithMaxTokens(*maxTokens))
	if *temperature > 0 {
		genOpts = append(genOpts, inference.WithTemperature(*temperature))
	}
	if *topK > 0 {
		genOpts = append(genOpts, inference.WithTopK(*topK))
	}
	if *topP < 1.0 {
		genOpts = append(genOpts, inference.WithTopP(*topP))
	}

	ctx := context.Background()

	if *stream {
		// Stream tokens to stdout as they are generated.
		err = model.GenerateStream(ctx, prompt, generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				fmt.Print(token)
			}
			return nil
		}), genOpts...)
		fmt.Println()
	} else {
		// Generate the full response at once.
		var result string
		result, err = model.Generate(ctx, prompt, genOpts...)
		if err == nil {
			fmt.Println(result)
		}
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating: %v\n", err)
		os.Exit(1)
	}
}
