// Command vision-analysis demonstrates multimodal inference with image input.
//
// It loads a vision-capable GGUF model, reads an image file, and asks the model
// to describe or analyze the image. This uses the same inference.Message API
// that the OpenAI-compatible server uses for vision requests.
//
// Usage:
//
//	go build -o vision-analysis ./examples/vision-analysis/
//	./vision-analysis --model path/to/vision-model.gguf --image photo.jpg
//	./vision-analysis --model path/to/vision-model.gguf --image photo.jpg --prompt "What objects are in this image?"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to a vision-capable GGUF model file")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	imagePath := flag.String("image", "", "path to an image file (JPEG or PNG)")
	prompt := flag.String("prompt", "Describe this image in detail.", "question or instruction about the image")
	maxTokens := flag.Int("max-tokens", 512, "maximum tokens to generate")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <model.gguf> --image <image.jpg>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Read the image file.
	imageData, err := os.ReadFile(*imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read image: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Image: %s (%d bytes)\n", *imagePath, len(imageData))

	// Load the vision-capable model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Build a chat request with the image embedded in the message.
	// The inference.Message.Images field carries raw image bytes, matching
	// the same format used by the OpenAI-compatible /v1/chat/completions
	// endpoint for vision requests.
	messages := []inference.Message{
		{
			Role:    "user",
			Content: *prompt,
			Images:  [][]byte{imageData},
		},
	}

	resp, err := model.Chat(context.Background(), messages,
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(0.5),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(resp.Content)
}
