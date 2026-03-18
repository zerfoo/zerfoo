// Recipe 02: Streaming Chat
//
// Stream tokens to the terminal as they are generated. This provides a
// responsive, ChatGPT-like experience where text appears incrementally.
//
// Usage:
//
//	go run ./docs/cookbook/02-streaming-chat/ --model path/to/model.gguf
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
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: streaming-chat --model <path-or-id>")
		os.Exit(1)
	}

	m, err := zerfoo.Load(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	fmt.Println("Type a message and press Enter. Type 'quit' to exit.")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
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

		// ChatStream returns a channel that yields tokens as they arrive.
		stream, err := m.ChatStream(context.Background(), prompt,
			zerfoo.WithGenMaxTokens(512),
			zerfoo.WithGenTemperature(0.8),
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "stream: %v\n", err)
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
