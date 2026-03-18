// Command bench_spec benchmarks speculative decoding speedup by comparing
// standalone target model decode against speculative decode (target + draft).
//
// Usage:
//
//	bench_spec --model-target /path/to/27B.gguf --model-draft /path/to/1B.gguf [--tokens 200] [--prompts 10] [--backend cuda] [--warmup 2] [--draft-len 4]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// SpecBenchConfig holds the CLI configuration for the benchmark.
type SpecBenchConfig struct {
	ModelTarget string
	ModelDraft  string
	Backend     string
	Tokens      int
	Prompts     int
	Warmup      int
	DraftLen    int
	Output      string
}

// SpecBenchResult holds results for a single benchmark mode.
type SpecBenchResult struct {
	Mode           string  `json:"mode"`
	TotalTokens    int     `json:"total_tokens"`
	TotalDurationS float64 `json:"total_duration_s"`
	TokPerSec      float64 `json:"tok_per_sec"`
	AvgTTFTMs      float64 `json:"avg_ttft_ms"`
}

// SpecBenchReport is the full benchmark output.
type SpecBenchReport struct {
	ModelTarget    string          `json:"model_target"`
	ModelDraft     string          `json:"model_draft"`
	Backend        string          `json:"backend"`
	TokensPerRun   int             `json:"tokens_per_run"`
	NumPrompts     int             `json:"num_prompts"`
	DraftLen       int             `json:"draft_len"`
	Standalone     SpecBenchResult `json:"standalone"`
	Speculative    SpecBenchResult `json:"speculative"`
	AcceptanceRate float64         `json:"acceptance_rate"`
	Speedup        float64         `json:"speedup"`
	Commit         string          `json:"commit"`
	Timestamp      string          `json:"timestamp"`
}

// defaultPrompts returns a standard set of prompts for benchmarking.
func defaultPrompts(n int) []string {
	base := []string{
		"Explain the theory of general relativity in simple terms.",
		"Write a short story about a robot learning to paint.",
		"What are the key differences between TCP and UDP?",
		"Describe the process of photosynthesis step by step.",
		"List five important principles of software engineering.",
		"How does a neural network learn from data?",
		"What is the significance of the Turing test?",
		"Explain how a compiler transforms source code into machine code.",
		"Describe the water cycle and its importance to ecosystems.",
		"What are the main causes of climate change?",
	}
	prompts := make([]string, 0, n)
	for i := range n {
		prompts = append(prompts, base[i%len(base)])
	}
	return prompts
}

// BenchRunner abstracts model generation for testing.
type BenchRunner interface {
	Generate(ctx context.Context, prompt string, maxTokens int) (tokensGenerated int, durationMs float64, err error)
}

// standaloneRunner benchmarks standard autoregressive decoding.
type standaloneRunner struct {
	model *inference.Model
}

func (r *standaloneRunner) Generate(ctx context.Context, prompt string, maxTokens int) (int, float64, error) {
	var tokenCount atomic.Int64
	start := time.Now()

	handler := generate.TokenStreamFunc(func(token string, done bool) error {
		if !done {
			tokenCount.Add(1)
		}
		return nil
	})

	err := r.model.GenerateStream(ctx, prompt, handler,
		inference.WithMaxTokens(maxTokens),
		inference.WithTemperature(0),
	)
	if err != nil {
		return 0, 0, err
	}

	dur := time.Since(start)
	return int(tokenCount.Load()), float64(dur.Milliseconds()), nil
}

// speculativeRunner benchmarks speculative decoding (target + draft).
type speculativeRunner struct {
	target   *inference.Model
	draft    *inference.Model
	draftLen int
}

func (r *speculativeRunner) Generate(ctx context.Context, prompt string, maxTokens int) (int, float64, error) {
	start := time.Now()

	result, err := r.target.SpeculativeGenerate(ctx, r.draft, prompt, r.draftLen,
		inference.WithMaxTokens(maxTokens),
	)
	if err != nil {
		return 0, 0, err
	}

	dur := time.Since(start)

	// Count tokens by encoding the result.
	tok := r.target.Tokenizer()
	ids, encErr := tok.Encode(result)
	if encErr != nil {
		// Fall back to word count estimate.
		return len(strings.Fields(result)), float64(dur.Milliseconds()), nil
	}
	return len(ids), float64(dur.Milliseconds()), nil
}

// runMode benchmarks a single mode across all prompts.
func runMode(ctx context.Context, runner BenchRunner, prompts []string, tokens int, warmup int, mode string) (SpecBenchResult, error) {
	// Warmup.
	for i := range warmup {
		_, _, err := runner.Generate(ctx, prompts[0], 4)
		if err != nil {
			return SpecBenchResult{}, fmt.Errorf("warmup %d: %w", i, err)
		}
	}

	var totalTokens int
	var totalDurMs float64

	for i, prompt := range prompts {
		fmt.Printf("  [%s] prompt %d/%d...\n", mode, i+1, len(prompts))
		tc, durMs, err := runner.Generate(ctx, prompt, tokens)
		if err != nil {
			return SpecBenchResult{}, fmt.Errorf("prompt %d: %w", i, err)
		}
		totalTokens += tc
		totalDurMs += durMs
	}

	result := SpecBenchResult{
		Mode:        mode,
		TotalTokens: totalTokens,
	}
	if totalDurMs > 0 {
		result.TotalDurationS = totalDurMs / 1000.0
		result.TokPerSec = float64(totalTokens) / (totalDurMs / 1000.0)
	}
	return result, nil
}

// gitCommitHash returns the short git commit hash.
func gitCommitHash() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// printReport prints the benchmark report to stdout.
func printReport(r SpecBenchReport) {
	fmt.Println()
	fmt.Println("=== Speculative Decoding Benchmark ===")
	fmt.Printf("Target model: %s\n", r.ModelTarget)
	fmt.Printf("Draft model:  %s\n", r.ModelDraft)
	fmt.Printf("Backend:      %s\n", r.Backend)
	fmt.Printf("Tokens/run:   %d\n", r.TokensPerRun)
	fmt.Printf("Prompts:      %d\n", r.NumPrompts)
	fmt.Printf("Draft len:    %d\n", r.DraftLen)
	fmt.Println()
	fmt.Printf("Standalone:   %.2f tok/s\n", r.Standalone.TokPerSec)
	fmt.Printf("Speculative:  %.2f tok/s\n", r.Speculative.TokPerSec)
	fmt.Printf("Acceptance:   %.2f (alpha)\n", r.AcceptanceRate)
	fmt.Printf("Speedup:      %.2fx\n", r.Speedup)
	fmt.Println()

	if r.Speedup >= 2.0 {
		fmt.Println("PASS: >= 2x speedup achieved")
	} else {
		fmt.Printf("NOTE: %.2fx speedup (target: >= 2x)\n", r.Speedup)
	}
}

// writeJSON writes the report to a JSON file.
func writeJSON(path string, r SpecBenchReport) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// parseFlags parses CLI flags into a config. Uses the provided args slice
// so tests can inject flags without touching os.Args.
func parseFlags(args []string) (SpecBenchConfig, error) {
	fs := flag.NewFlagSet("bench_spec", flag.ContinueOnError)
	cfg := SpecBenchConfig{}
	fs.StringVar(&cfg.ModelTarget, "model-target", "", "path to target model GGUF file (e.g. 27B)")
	fs.StringVar(&cfg.ModelDraft, "model-draft", "", "path to draft model GGUF file (e.g. 1B)")
	fs.StringVar(&cfg.Backend, "backend", "cpu", "backend: cpu|cuda|rocm")
	fs.IntVar(&cfg.Tokens, "tokens", 200, "tokens to generate per prompt")
	fs.IntVar(&cfg.Prompts, "prompts", 10, "number of prompts to benchmark")
	fs.IntVar(&cfg.Warmup, "warmup", 2, "warmup iterations per mode")
	fs.IntVar(&cfg.DraftLen, "draft-len", 4, "draft tokens per speculative step")
	fs.StringVar(&cfg.Output, "output", "bench_spec_results.json", "path for JSON results file")

	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// validateConfig checks that the config has all required fields.
func validateConfig(cfg SpecBenchConfig) error {
	if cfg.ModelTarget == "" {
		return fmt.Errorf("--model-target is required")
	}
	if cfg.ModelDraft == "" {
		return fmt.Errorf("--model-draft is required")
	}
	if cfg.Tokens <= 0 {
		return fmt.Errorf("--tokens must be positive")
	}
	if cfg.Prompts <= 0 {
		return fmt.Errorf("--prompts must be positive")
	}
	if cfg.DraftLen <= 0 {
		return fmt.Errorf("--draft-len must be positive")
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func run() error {
	layerreg.RegisterAll()

	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		return err
	}
	if err := validateConfig(cfg); err != nil {
		return fmt.Errorf("usage: bench_spec --model-target /path/to/27B.gguf --model-draft /path/to/1B.gguf\n%w", err)
	}

	prompts := defaultPrompts(cfg.Prompts)
	ctx := context.Background()

	// Load target model.
	fmt.Printf("Loading target model: %s (backend=%s)...\n", cfg.ModelTarget, cfg.Backend)
	targetModel, err := inference.LoadFile(cfg.ModelTarget, inference.WithDevice(cfg.Backend))
	if err != nil {
		return fmt.Errorf("load target: %w", err)
	}

	// Benchmark standalone target.
	fmt.Println("\n--- Standalone target decode ---")
	standalone, err := runMode(ctx, &standaloneRunner{model: targetModel}, prompts, cfg.Tokens, cfg.Warmup, "standalone")
	if err != nil {
		return fmt.Errorf("standalone benchmark: %w", err)
	}

	// Load draft model.
	fmt.Printf("\nLoading draft model: %s (backend=%s)...\n", cfg.ModelDraft, cfg.Backend)
	draftModel, err := inference.LoadFile(cfg.ModelDraft, inference.WithDevice(cfg.Backend))
	if err != nil {
		return fmt.Errorf("load draft: %w", err)
	}

	// Benchmark speculative decoding.
	fmt.Println("\n--- Speculative decode (target + draft) ---")
	specRunner := &speculativeRunner{target: targetModel, draft: draftModel, draftLen: cfg.DraftLen}
	speculative, err := runMode(ctx, specRunner, prompts, cfg.Tokens, cfg.Warmup, "speculative")
	if err != nil {
		return fmt.Errorf("speculative benchmark: %w", err)
	}

	// Compute speedup.
	var speedup float64
	if standalone.TokPerSec > 0 {
		speedup = speculative.TokPerSec / standalone.TokPerSec
	}

	// Estimate acceptance rate from token throughput ratio.
	// In speculative decoding, if alpha is the acceptance rate and K is draftLen,
	// the expected tokens per step ≈ 1 + alpha*K (approximately).
	// We report speedup as the primary metric; alpha is estimated.
	estimatedAlpha := 0.0
	if speedup > 1 && cfg.DraftLen > 0 {
		// speedup ≈ (1 + alpha*K) / (1 + K * cost_ratio)
		// For same-family models, cost_ratio ≈ params_draft/params_target.
		// Without exact cost_ratio, we estimate alpha from observed speedup.
		estimatedAlpha = math.Min(1.0, (speedup-1)/float64(cfg.DraftLen))
	}

	report := SpecBenchReport{
		ModelTarget:    cfg.ModelTarget,
		ModelDraft:     cfg.ModelDraft,
		Backend:        cfg.Backend,
		TokensPerRun:   cfg.Tokens,
		NumPrompts:     cfg.Prompts,
		DraftLen:       cfg.DraftLen,
		Standalone:     standalone,
		Speculative:    speculative,
		AcceptanceRate: math.Round(estimatedAlpha*100) / 100,
		Speedup:        math.Round(speedup*100) / 100,
		Commit:         gitCommitHash(),
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
	}

	printReport(report)

	if err := writeJSON(cfg.Output, report); err != nil {
		return fmt.Errorf("write results: %w", err)
	}
	fmt.Printf("Results written to %s\n", cfg.Output)
	return nil
}
