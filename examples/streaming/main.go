// Command streaming demonstrates streaming chat generation using the zerfoo API.
//
// Unlike the chat example which waits for the full response, this example
// prints each token to the terminal as it arrives using [zerfoo.Model.ChatStream].
//
// Usage:
//
//	go build -o streaming ./examples/streaming/
//	./streaming --model path/to/model.gguf
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file or HuggingFace model ID")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <path-or-id>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	fmt.Println("Streaming chat started. Type your message and press Enter. Type 'quit' to exit.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		prompt := strings.TrimSpace(scanner.Text())
		if prompt == "" {
			continue
		}
		if prompt == "quit" {
			break
		}

		stream, err := m.ChatStream(context.Background(), prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		for tok := range stream {
			if tok.Done {
				break
			}
			fmt.Print(tok.Text)
		}
		fmt.Println()
	}
}
