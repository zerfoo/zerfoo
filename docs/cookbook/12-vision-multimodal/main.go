// Recipe 12: Vision / Multimodal Inference
//
// Analyze images using a vision-capable GGUF model. The image is passed
// alongside a text prompt using the inference.Message API, the same format
// used by the OpenAI-compatible /v1/chat/completions endpoint.
//
// Requirements:
//   - A vision-capable GGUF model (e.g. LLaVA, Gemma 3 with vision encoder)
//
// Usage:
//
//	go run ./docs/cookbook/12-vision-multimodal/ --model path/to/vision-model.gguf --image photo.jpg
//	go run ./docs/cookbook/12-vision-multimodal/ --model path/to/vision-model.gguf --image photo.jpg --prompt "Count the objects"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to a vision-capable GGUF model")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	imagePath := flag.String("image", "", "path to an image file (JPEG or PNG)")
	prompt := flag.String("prompt", "Describe this image in detail.", "question about the image")
	maxTokens := flag.Int("max-tokens", 512, "maximum tokens to generate")
	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		fmt.Fprintln(os.Stderr, "usage: vision-multimodal --model <model.gguf> --image <image.jpg>")
		os.Exit(1)
	}

	// Read the image file into memory.
	imageData, err := os.ReadFile(*imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read image: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Image: %s (%d bytes)\n", *imagePath, len(imageData))

	// Load the vision-capable model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Model: %s (%d layers)\n", cfg.Architecture, cfg.NumLayers)

	// Build a chat message with the image embedded.
	// The Images field carries raw image bytes, matching the OpenAI vision API.
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
