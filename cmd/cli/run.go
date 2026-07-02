package cli

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/generate/grammar"
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
	var jsonSchemaStr, prompt string
	var temperature float64
	var topK, maxTokens int
	var topP, repetitionPenalty float64
	var quarot bool
	var pjrtPlugin string

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--temperature":
			s, err := nextVal("--temperature")
			if err != nil {
				return err
			}
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return fmt.Errorf("--temperature: %w", err)
			}
			temperature = v
		case "--top-k":
			s, err := nextVal("--top-k")
			if err != nil {
				return err
			}
			v, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("--top-k: %w", err)
			}
			topK = v
		case "--top-p":
			s, err := nextVal("--top-p")
			if err != nil {
				return err
			}
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return fmt.Errorf("--top-p: %w", err)
			}
			topP = v
		case "--max-tokens":
			s, err := nextVal("--max-tokens")
			if err != nil {
				return err
			}
			v, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("--max-tokens: %w", err)
			}
			maxTokens = v
		case "--system":
			s, err := nextVal("--system")
			if err != nil {
				return err
			}
			systemPrompt = s
		case "--repetition-penalty":
			s, err := nextVal("--repetition-penalty")
			if err != nil {
				return err
			}
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return fmt.Errorf("--repetition-penalty: %w", err)
			}
			repetitionPenalty = v
		case "--cache-dir":
			s, err := nextVal("--cache-dir")
			if err != nil {
				return err
			}
			cacheDir = s
		case "--json-schema":
			s, err := nextVal("--json-schema")
			if err != nil {
				return err
			}
			jsonSchemaStr = s
		case "--prompt":
			s, err := nextVal("--prompt")
			if err != nil {
				return err
			}
			prompt = s
		case "--pjrt":
			s, err := nextVal("--pjrt")
			if err != nil {
				return err
			}
			pjrtPlugin = s
		case "--quarot":
			quarot = true
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

	// Parse and convert JSON schema if provided.
	var grammarState *grammar.Grammar
	if jsonSchemaStr != "" {
		var schema grammar.JSONSchema
		if err := json.Unmarshal([]byte(jsonSchemaStr), &schema); err != nil {
			return fmt.Errorf("--json-schema: invalid JSON: %w", err)
		}
		g, err := grammar.Convert(&schema)
		if err != nil {
			return fmt.Errorf("--json-schema: %w", err)
		}
		grammarState = g
	}

	// Build load options.
	var loadOpts []inference.Option
	if cacheDir != "" {
		loadOpts = append(loadOpts, inference.WithCacheDir(cacheDir))
	}
	if quarot {
		loadOpts = append(loadOpts, inference.WithQuaRot(true))
	}
	if pjrtPlugin != "" {
		loadOpts = append(loadOpts, inference.WithPJRT(pjrtPlugin))
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
	if grammarState != nil {
		genOpts = append(genOpts, inference.WithGrammar(grammarState))
	}

	// Non-interactive mode: when --json-schema is set, generate once and exit.
	if jsonSchemaStr != "" {
		if prompt == "" {
			return errors.New("--prompt is required when using --json-schema")
		}
		result, err := mdl.Generate(ctx, prompt, genOpts...)
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		_, _ = fmt.Fprint(c.out, result)
		return nil
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
  --cache-dir <dir>              Override model cache directory
  --json-schema <schema>         JSON Schema for structured output (non-interactive)
  --prompt <text>                Prompt text (required with --json-schema)
  --quarot                       Fuse QuaRot Hadamard rotation into weights at load time
  --pjrt <path>                  Path to PJRT plugin .so for accelerator backend`
}

// Examples implements Command.Examples.
func (c *RunCommand) Examples() []string {
	return []string{
		"run google/gemma-3-1b",
		"run google/gemma-3-1b --temperature 0.7 --max-tokens 512",
		`run google/gemma-3-1b --system "You are a helpful assistant"`,
		`run google/gemma-3-1b --json-schema '{"type":"object","properties":{"name":{"type":"string"}}}' --prompt "Generate a name"`,
		"run --quarot model.gguf",
		"run --pjrt /usr/lib/pjrt_cpu.so google/gemma-3-1b",
	}
}

// Static interface assertion.
var _ Command = (*RunCommand)(nil)
