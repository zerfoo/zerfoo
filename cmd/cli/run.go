package cli

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

// RunCommand implements the "run" CLI command for interactive
// prompt-response generation.
type RunCommand struct {
	in  io.Reader
	out io.Writer
	// loadFn allows injection of a custom model loader for testing.
	loadFn func(modelID string, opts ...inference.Option) (*inference.Model, error)
}

// NewRunCommand creates a new RunCommand using the given I/O streams.
func NewRunCommand(in io.Reader, out io.Writer) *RunCommand {
	return &RunCommand{in: in, out: out, loadFn: inference.Load}
}

// Name implements Command.Name.
func (c *RunCommand) Name() string { return "run" }

// Description implements Command.Description.
func (c *RunCommand) Description() string {
	return "Run interactive chat with a model"
}

// Run implements Command.Run.
func (c *RunCommand) Run(ctx context.Context, args []string) error {
	var modelID, systemPrompt, cacheDir string
	var temperature float64
	var topK, maxTokens int
	var topP, repetitionPenalty float64

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--temperature":
			if i+1 >= len(args) {
				return errors.New("--temperature requires a value")
			}
			v, err := strconv.ParseFloat(args[i+1], 64)
			if err != nil {
				return fmt.Errorf("--temperature: %w", err)
			}
			temperature = v
			i++
		case "--top-k":
			if i+1 >= len(args) {
				return errors.New("--top-k requires a value")
			}
			v, err := strconv.Atoi(args[i+1])
			if err != nil {
				return fmt.Errorf("--top-k: %w", err)
			}
			topK = v
			i++
		case "--top-p":
			if i+1 >= len(args) {
				return errors.New("--top-p requires a value")
			}
			v, err := strconv.ParseFloat(args[i+1], 64)
			if err != nil {
				return fmt.Errorf("--top-p: %w", err)
			}
			topP = v
			i++
		case "--max-tokens":
			if i+1 >= len(args) {
				return errors.New("--max-tokens requires a value")
			}
			v, err := strconv.Atoi(args[i+1])
			if err != nil {
				return fmt.Errorf("--max-tokens: %w", err)
			}
			maxTokens = v
			i++
		case "--system":
			if i+1 >= len(args) {
				return errors.New("--system requires a value")
			}
			systemPrompt = args[i+1]
			i++
		case "--repetition-penalty":
			if i+1 >= len(args) {
				return errors.New("--repetition-penalty requires a value")
			}
			v, err := strconv.ParseFloat(args[i+1], 64)
			if err != nil {
				return fmt.Errorf("--repetition-penalty: %w", err)
			}
			repetitionPenalty = v
			i++
		case "--cache-dir":
			if i+1 >= len(args) {
				return errors.New("--cache-dir requires a value")
			}
			cacheDir = args[i+1]
			i++
		default:
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			modelID = args[i]
		}
	}

	if modelID == "" {
		return errors.New("model ID is required")
	}

	// Build load options.
	var loadOpts []inference.Option
	if cacheDir != "" {
		loadOpts = append(loadOpts, inference.WithCacheDir(cacheDir))
	}

	li := startLoading(c.out)
	mdl, err := c.loadFn(modelID, loadOpts...)
	li.stop()
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	// Build generate options.
	var genOpts []inference.GenerateOption
	if temperature > 0 {
		genOpts = append(genOpts, inference.WithTemperature(temperature))
	}
	if topK > 0 {
		genOpts = append(genOpts, inference.WithTopK(topK))
	}
	if topP > 0 {
		genOpts = append(genOpts, inference.WithTopP(topP))
	}
	if maxTokens > 0 {
		genOpts = append(genOpts, inference.WithMaxTokens(maxTokens))
	}
	if repetitionPenalty > 0 {
		genOpts = append(genOpts, inference.WithRepetitionPenalty(repetitionPenalty))
	}

	_, _ = fmt.Fprintf(c.out, "Model loaded. Type your message (Ctrl-D to quit).\n\n")

	scanner := bufio.NewScanner(c.in)
	for {
		_, _ = fmt.Fprint(c.out, "> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if systemPrompt != "" {
			// Use Chat to pass the system prompt via chat template formatting.
			messages := []inference.Message{
				{Role: "system", Content: systemPrompt},
				{Role: "user", Content: line},
			}
			resp, err := mdl.Chat(ctx, messages, genOpts...)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}
			_, _ = fmt.Fprintln(c.out, resp.Content)
		} else {
			// Stream the response when no system prompt is needed.
			err := mdl.GenerateStream(ctx, line, generate.TokenStreamFunc(func(token string, done bool) error {
				if !done {
					_, _ = fmt.Fprint(c.out, token)
				}
				return nil
			}), genOpts...)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}
			_, _ = fmt.Fprintln(c.out)
		}
	}

	return scanner.Err()
}

// Usage implements Command.Usage.
func (c *RunCommand) Usage() string {
	return `run [OPTIONS] <model-id>

Start an interactive chat session with a model.

OPTIONS:
  --temperature <float>          Sampling temperature (default: 1.0)
  --top-k <int>                  Top-K sampling (default: disabled)
  --top-p <float>                Top-P nucleus sampling (default: 1.0)
  --repetition-penalty <float>   Penalize repeated tokens (default: 1.0)
  --max-tokens <int>             Maximum tokens to generate (default: 256)
  --system <prompt>              System prompt for context
  --cache-dir <dir>              Override model cache directory`
}

// Examples implements Command.Examples.
func (c *RunCommand) Examples() []string {
	return []string{
		"run google/gemma-3-1b",
		"run google/gemma-3-1b --temperature 0.7 --max-tokens 512",
		`run google/gemma-3-1b --system "You are a helpful assistant"`,
	}
}

// Static interface assertion.
var _ Command = (*RunCommand)(nil)
