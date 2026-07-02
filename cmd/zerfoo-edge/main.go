// Package main provides a minimal edge/embedded inference binary for Zerfoo.
//
// Build: go build -tags edge ./cmd/zerfoo-edge/
//
// The edge binary supports CPU-only inference with GGUF models. It excludes
// training, distributed, serve/API, GPU backends, AutoML, and NAS to produce
// a small, self-contained binary suitable for edge and embedded deployments.
package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

// version is set at build time via -ldflags "-X main.version=...".
var version string

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	args := os.Args[1:]

	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" {
		printUsage()
		return nil
	}

	if args[0] == "--version" || args[0] == "-v" {
		v := version
		if v == "" {
			v = "dev"
		}
		fmt.Fprintf(os.Stdout, "zerfoo-edge %s\n", v)
		return nil
	}

	return runInference(ctx, args)
}

func runInference(ctx context.Context, args []string) error {
	var modelID, prompt, systemPrompt, cacheDir string
	var temperature float64
	var topK, maxTokens int
	var topP, repetitionPenalty float64

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
		case "--prompt":
			s, err := nextVal("--prompt")
			if err != nil {
				return err
			}
			prompt = s
		default:
			if strings.HasPrefix(arg, "--") {
				return fmt.Errorf("unknown flag: %s", arg)
			}
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", arg)
			}
			modelID = arg
		}
	}

	if modelID == "" {
		return errors.New("model ID is required; usage: zerfoo-edge <model-id> [options]")
	}

	// Build load options (CPU only for edge).
	var loadOpts []inference.Option
	if cacheDir != "" {
		loadOpts = append(loadOpts, inference.WithCacheDir(cacheDir))
	}

	fmt.Fprintf(os.Stderr, "Loading model %s...\n", modelID)
	mdl, err := inference.Load(modelID, loadOpts...)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Model loaded.\n")

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

	// Single-shot mode: if --prompt is provided, generate once and exit.
	if prompt != "" {
		if systemPrompt != "" {
			messages := []inference.Message{
				{Role: "system", Content: systemPrompt},
				{Role: "user", Content: prompt},
			}
			resp, err := mdl.Chat(ctx, messages, genOpts...)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}
			fmt.Fprintln(os.Stdout, resp.Content)
			return nil
		}
		return mdl.GenerateStream(ctx, prompt, generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				fmt.Fprint(os.Stdout, token)
			} else {
				fmt.Fprintln(os.Stdout)
			}
			return nil
		}), genOpts...)
	}

	// Interactive mode.
	fmt.Fprintln(os.Stderr, "Type your message (Ctrl-D to quit).")
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Fprint(os.Stderr, "> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if systemPrompt != "" {
			messages := []inference.Message{
				{Role: "system", Content: systemPrompt},
				{Role: "user", Content: line},
			}
			resp, err := mdl.Chat(ctx, messages, genOpts...)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}
			fmt.Fprintln(os.Stdout, resp.Content)
		} else {
			err := mdl.GenerateStream(ctx, line, generate.TokenStreamFunc(func(token string, done bool) error {
				if !done {
					fmt.Fprint(os.Stdout, token)
				}
				return nil
			}), genOpts...)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}
			fmt.Fprintln(os.Stdout)
		}
	}

	return scanner.Err()
}

// splitFlag checks whether arg contains "=" (e.g. "--flag=value") and returns
// the flag name and value separately.
func splitFlag(arg string) (flag, value string, ok bool) {
	if idx := strings.Index(arg, "="); idx >= 0 {
		return arg[:idx], arg[idx+1:], true
	}
	return arg, "", false
}

func printUsage() {
	fmt.Fprint(os.Stdout, `zerfoo-edge — minimal edge/embedded inference binary

USAGE:
  zerfoo-edge <model-id> [OPTIONS]
  zerfoo-edge --prompt "Hello" <model-id>

OPTIONS:
  --prompt <text>              Single-shot prompt (non-interactive)
  --system <text>              System prompt
  --temperature <float>        Sampling temperature (default: 1.0)
  --top-k <int>                Top-K sampling
  --top-p <float>              Top-P nucleus sampling
  --repetition-penalty <float> Penalize repeated tokens
  --max-tokens <int>           Maximum tokens to generate
  --cache-dir <dir>            Override model cache directory
  --version                    Print version and exit
  --help                       Print this help

EXAMPLES:
  zerfoo-edge google/gemma-3-1b
  zerfoo-edge google/gemma-3-1b --prompt "What is 2+2?"
  zerfoo-edge google/gemma-3-1b --temperature 0.7 --max-tokens 512
`)
}
