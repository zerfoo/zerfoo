// Command chat demonstrates a simple interactive chatbot using the zerfoo one-line API.
//
// Usage:
//
//	go build -o chat ./examples/chat/
//	./chat --model path/to/model.gguf
package main

import (
	"bufio"
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

	fmt.Println("Chat started. Type your message and press Enter. Type 'quit' to exit.")

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
		response, err := m.Chat(prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Println(response)
	}
}
