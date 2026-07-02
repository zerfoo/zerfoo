// Command langchain-chatbot demonstrates using the Zerfoo LangChain adapter
// as a drop-in LLM for a simple interactive chatbot loop.
//
// Start a Zerfoo server first:
//
//	zerfoo serve --model path/to/model.gguf --port 8080
//
// Then run this example:
//
//	go run ./examples/langchain-chatbot/ --server http://localhost:8080 --model llama3
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/sdk/integrations/langchain"
)

func main() {
	server := flag.String("server", "http://localhost:8080", "Zerfoo server URL")
	model := flag.String("model", "llama3", "Model name")
	temp := flag.Float64("temperature", 0.7, "Sampling temperature (0–1)")
	maxTok := flag.Int("max-tokens", 512, "Maximum tokens to generate")
	flag.Parse()

	llm := langchain.NewAdapter(
		*server,
		*model,
		langchain.WithTemperature(float32(*temp)),
		langchain.WithMaxTokens(*maxTok),
	)

	fmt.Printf("LangChain chatbot connected to %s (model: %s)\n", *server, *model)
	fmt.Println("Type your message and press Enter. Type 'quit' to exit.")

	ctx := context.Background()
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
		if prompt == "quit" || prompt == "exit" {
			break
		}

		reply, err := llm.Call(ctx, prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
		fmt.Println(reply)
		fmt.Println()
	}
}
