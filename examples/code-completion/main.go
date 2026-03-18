// Command code-completion demonstrates using a language model for code completion.
//
// It reads a code snippet (from a flag or stdin), appends a completion prompt,
// and generates the continuation. This pattern works with any code-capable GGUF
// model (e.g., CodeLlama, DeepSeek Coder, Qwen 2.5 Coder).
//
// Usage:
//
//	go build -o code-completion ./examples/code-completion/
//	./code-completion --model path/to/model.gguf --code "func fibonacci(n int) int {"
//	echo "func add(a, b int) int {" | ./code-completion --model path/to/model.gguf
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda"`)
	code := flag.String("code", "", "code prefix to complete (reads from stdin if empty)")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")
	temperature := flag.Float64("temperature", 0.2, "sampling temperature (lower = more deterministic)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s --model <model.gguf> [--code <prefix>]\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if *modelPath == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Get the code prefix from the flag or stdin.
	codePrefix := *code
	if codePrefix == "" {
		codePrefix = readStdin()
	}
	if codePrefix == "" {
		fmt.Fprintln(os.Stderr, "error: no code provided via --code flag or stdin")
		os.Exit(1)
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

	// Build the prompt. We include the code as-is and let the model continue it.
	prompt := fmt.Sprintf("Complete the following code:\n\n%s", codePrefix)

	result, err := model.Generate(context.Background(), prompt,
		inference.WithMaxTokens(*maxTokens),
		inference.WithTemperature(*temperature),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}

	// Print the original code followed by the completion.
	fmt.Println(codePrefix)
	fmt.Println(result)
}

// readStdin reads all available input from stdin (non-blocking if a pipe is attached).
func readStdin() string {
	info, _ := os.Stdin.Stat()
	if (info.Mode() & os.ModeCharDevice) != 0 {
		// stdin is a terminal, not a pipe -- nothing to read.
		return ""
	}
	var sb strings.Builder
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		sb.WriteString(scanner.Text())
		sb.WriteByte('\n')
	}
	return strings.TrimSpace(sb.String())
}
