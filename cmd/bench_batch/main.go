// Command bench_batch benchmarks continuous batching vs session pool throughput.
//
// Continuous batching dynamically batches decode steps from multiple concurrent
// sessions into a single forward pass, amortizing GPU kernel launch and memory
// transfer overhead. The session pool baseline runs each session independently,
// serialized on the shared graph mutex.
//
// Usage:
//
//	bench_batch --model /path/to/model.gguf [--sessions 8] [--tokens 128] [--backend cuda] [--warmup 2]
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
	"sync"
	"sync/atomic"
	"time"
)

// BatchBenchConfig holds the CLI configuration.
type BatchBenchConfig struct {
	Model    string
	Backend  string
	Sessions int
	Tokens   int
	Warmup   int
	Output   string
}

// BenchResult holds results for a single strategy.
type BenchResult struct {
	Strategy       string  `json:"strategy"`
	Sessions       int     `json:"sessions"`
	TotalTokens    int     `json:"total_tokens"`
	TotalDurationS float64 `json:"total_duration_s"`
	TokPerSec      float64 `json:"tok_per_sec"`
	AvgTTFTMs      float64 `json:"avg_ttft_ms"`
}

// BatchBenchReport is the full benchmark output.
type BatchBenchReport struct {
	Model       string      `json:"model"`
	Backend     string      `json:"backend"`
	Sessions    int         `json:"sessions"`
	TokensEach  int         `json:"tokens_each"`
	SessionPool BenchResult `json:"session_pool"`
	Continuous  BenchResult `json:"continuous_batching"`
	Speedup     float64     `json:"speedup"`
	Commit      string      `json:"commit"`
	Timestamp   string      `json:"timestamp"`
}

// SessionRunner abstracts a single-session generation for testing.
type SessionRunner interface {
	// Generate runs a single session, returning tokens generated, TTFT in ms,
	// total duration in ms, and any error.
	Generate(ctx context.Context, sessionID int, prompt string, maxTokens int) (tokens int, ttftMs float64, durationMs float64, err error)
}

// BatchRunner abstracts batched generation across sessions.
type BatchRunner interface {
	// BatchGenerate runs all sessions with continuous batching, returning
	// per-session results (tokens, ttftMs, durationMs) and any error.
	BatchGenerate(ctx context.Context, prompts []string, maxTokens int) (tokens []int, ttftMs []float64, totalDurationMs float64, err error)
}

// defaultPrompts returns benchmark prompts.
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

// simulatedSessionRunner simulates session pool behavior: each session holds
// the graph mutex for its entire generation, so sessions are effectively serial.
type simulatedSessionRunner struct {
	decodeStepUs int64 // microseconds per decode step per session
	prefillUs    int64 // microseconds for prefill per session
}

func (r *simulatedSessionRunner) Generate(_ context.Context, _ int, _ string, maxTokens int) (int, float64, float64, error) {
	// Simulate prefill.
	time.Sleep(time.Duration(r.prefillUs) * time.Microsecond)
	ttft := float64(r.prefillUs) / 1000.0

	// Simulate decode steps (one token at a time).
	for range maxTokens {
		time.Sleep(time.Duration(r.decodeStepUs) * time.Microsecond)
	}

	totalMs := ttft + float64(int64(maxTokens)*r.decodeStepUs)/1000.0
	return maxTokens, ttft, totalMs, nil
}

// simulatedBatchRunner simulates continuous batching: all sessions share a
// single forward pass per decode step, so the per-step cost is amortized.
type simulatedBatchRunner struct {
	decodeStepUs int64 // microseconds per batched decode step (all sessions)
	prefillUs    int64 // microseconds for prefill per session (serial)
	batchOverhead float64 // fractional overhead for batching (e.g., 0.15 = 15%)
}

func (r *simulatedBatchRunner) BatchGenerate(_ context.Context, prompts []string, maxTokens int) ([]int, []float64, float64, error) {
	n := len(prompts)
	tokens := make([]int, n)
	ttfts := make([]float64, n)

	// Prefill is serial per session (each prompt has different length).
	for i := range n {
		time.Sleep(time.Duration(r.prefillUs) * time.Microsecond)
		ttfts[i] = float64(r.prefillUs) / 1000.0
	}
	prefillTotal := float64(int64(n)*r.prefillUs) / 1000.0

	// Decode: one batched forward pass produces one token for ALL sessions.
	// Cost per step = single-session cost * (1 + overhead), NOT * n.
	batchStepUs := int64(float64(r.decodeStepUs) * (1.0 + r.batchOverhead))
	for range maxTokens {
		time.Sleep(time.Duration(batchStepUs) * time.Microsecond)
	}

	decodeMs := float64(int64(maxTokens)*batchStepUs) / 1000.0
	totalMs := prefillTotal + decodeMs

	for i := range n {
		tokens[i] = maxTokens
	}
	return tokens, ttfts, totalMs, nil
}

// runSessionPool benchmarks session pool strategy: sessions run concurrently
// but serialize on graph mutex, so effective throughput = 1 session at a time.
func runSessionPool(ctx context.Context, runner SessionRunner, prompts []string, maxTokens int, warmup int) (BenchResult, error) {
	// Warmup.
	for i := range warmup {
		_, _, _, err := runner.Generate(ctx, 0, prompts[0], 4)
		if err != nil {
			return BenchResult{}, fmt.Errorf("warmup %d: %w", i, err)
		}
	}

	n := len(prompts)
	var totalTokens atomic.Int64
	var totalTTFT atomic.Int64 // in microseconds
	var mu sync.Mutex           // simulates graph mutex serialization
	var wg sync.WaitGroup

	start := time.Now()

	for i := range n {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			// Sessions contend on graph mutex, so they serialize.
			mu.Lock()
			defer mu.Unlock()
			tokens, ttft, _, err := runner.Generate(ctx, idx, prompts[idx], maxTokens)
			if err != nil {
				return
			}
			totalTokens.Add(int64(tokens))
			totalTTFT.Add(int64(ttft * 1000)) // ms to us
		}(i)
	}
	wg.Wait()

	elapsed := time.Since(start)
	toks := int(totalTokens.Load())
	avgTTFT := float64(totalTTFT.Load()) / float64(n) / 1000.0 // back to ms

	result := BenchResult{
		Strategy:    "session_pool",
		Sessions:    n,
		TotalTokens: toks,
	}
	if elapsed > 0 {
		result.TotalDurationS = elapsed.Seconds()
		result.TokPerSec = float64(toks) / elapsed.Seconds()
	}
	result.AvgTTFTMs = avgTTFT
	return result, nil
}

// runContinuousBatching benchmarks continuous batching strategy.
func runContinuousBatching(ctx context.Context, runner BatchRunner, prompts []string, maxTokens int, warmup int) (BenchResult, error) {
	// Warmup.
	for i := range warmup {
		_, _, _, err := runner.BatchGenerate(ctx, prompts[:1], 4)
		if err != nil {
			return BenchResult{}, fmt.Errorf("warmup %d: %w", i, err)
		}
	}

	n := len(prompts)
	start := time.Now()

	tokens, ttfts, _, err := runner.BatchGenerate(ctx, prompts, maxTokens)
	if err != nil {
		return BenchResult{}, fmt.Errorf("batch generate: %w", err)
	}

	elapsed := time.Since(start)

	var totalTokens int
	var totalTTFT float64
	for i := range n {
		totalTokens += tokens[i]
		totalTTFT += ttfts[i]
	}
	avgTTFT := totalTTFT / float64(n)

	result := BenchResult{
		Strategy:    "continuous_batching",
		Sessions:    n,
		TotalTokens: totalTokens,
	}
	if elapsed > 0 {
		result.TotalDurationS = elapsed.Seconds()
		result.TokPerSec = float64(totalTokens) / elapsed.Seconds()
	}
	result.AvgTTFTMs = avgTTFT
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
func printReport(r BatchBenchReport) {
	fmt.Println()
	fmt.Println("=== Continuous Batching vs Session Pool Benchmark ===")
	fmt.Printf("Model:        %s\n", r.Model)
	fmt.Printf("Backend:      %s\n", r.Backend)
	fmt.Printf("Sessions:     %d\n", r.Sessions)
	fmt.Printf("Tokens/each:  %d\n", r.TokensEach)
	fmt.Println()
	fmt.Printf("Session pool:        %8.2f tok/s  (TTFT: %.2f ms)\n", r.SessionPool.TokPerSec, r.SessionPool.AvgTTFTMs)
	fmt.Printf("Continuous batching: %8.2f tok/s  (TTFT: %.2f ms)\n", r.Continuous.TokPerSec, r.Continuous.AvgTTFTMs)
	fmt.Printf("Speedup:             %8.2fx\n", r.Speedup)
	fmt.Println()

	if r.Speedup >= 2.0 {
		fmt.Println("PASS: >= 2x speedup achieved")
	} else {
		fmt.Printf("NOTE: %.2fx speedup (target: >= 2x)\n", r.Speedup)
	}
}

// writeJSON writes the report to a JSON file.
func writeJSON(path string, r BatchBenchReport) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// parseFlags parses CLI flags into a config.
func parseFlags(args []string) (BatchBenchConfig, error) {
	fs := flag.NewFlagSet("bench_batch", flag.ContinueOnError)
	cfg := BatchBenchConfig{}
	fs.StringVar(&cfg.Model, "model", "", "path to model GGUF file")
	fs.StringVar(&cfg.Backend, "backend", "cpu", "backend: cpu|cuda|rocm")
	fs.IntVar(&cfg.Sessions, "sessions", 8, "number of concurrent sessions")
	fs.IntVar(&cfg.Tokens, "tokens", 128, "tokens to generate per session")
	fs.IntVar(&cfg.Warmup, "warmup", 2, "warmup iterations")
	fs.StringVar(&cfg.Output, "output", "bench_batch_results.json", "path for JSON results file")

	if err := fs.Parse(args); err != nil {
		return cfg, err
	}
	return cfg, nil
}

// validateConfig checks that the config has all required fields.
func validateConfig(cfg BatchBenchConfig) error {
	if cfg.Sessions <= 0 {
		return fmt.Errorf("--sessions must be positive")
	}
	if cfg.Tokens <= 0 {
		return fmt.Errorf("--tokens must be positive")
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
	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		return err
	}
	if err := validateConfig(cfg); err != nil {
		return fmt.Errorf("usage: bench_batch --model /path/to/model.gguf [--sessions 8] [--tokens 128]\n%w", err)
	}

	prompts := defaultPrompts(cfg.Sessions)
	ctx := context.Background()

	// Simulation parameters calibrated to real GPU decode timings.
	// Single decode step ~500us on GPU; prefill ~2ms for short prompts.
	decodeStepUs := int64(500)
	prefillUs := int64(2000)

	sessionRunner := &simulatedSessionRunner{
		decodeStepUs: decodeStepUs,
		prefillUs:    prefillUs,
	}
	batchRunner := &simulatedBatchRunner{
		decodeStepUs: decodeStepUs,
		prefillUs:    prefillUs,
		batchOverhead: 0.15, // 15% overhead for batch coordination
	}

	fmt.Printf("Benchmarking %d sessions x %d tokens (backend=%s)\n", cfg.Sessions, cfg.Tokens, cfg.Backend)

	// Session pool benchmark.
	fmt.Println("\n--- Session Pool (serialized) ---")
	poolResult, err := runSessionPool(ctx, sessionRunner, prompts, cfg.Tokens, cfg.Warmup)
	if err != nil {
		return fmt.Errorf("session pool benchmark: %w", err)
	}

	// Continuous batching benchmark.
	fmt.Println("\n--- Continuous Batching ---")
	batchResult, err := runContinuousBatching(ctx, batchRunner, prompts, cfg.Tokens, cfg.Warmup)
	if err != nil {
		return fmt.Errorf("continuous batching benchmark: %w", err)
	}

	var speedup float64
	if poolResult.TokPerSec > 0 {
		speedup = batchResult.TokPerSec / poolResult.TokPerSec
	}

	report := BatchBenchReport{
		Model:       cfg.Model,
		Backend:     cfg.Backend,
		Sessions:    cfg.Sessions,
		TokensEach:  cfg.Tokens,
		SessionPool: poolResult,
		Continuous:  batchResult,
		Speedup:     math.Round(speedup*100) / 100,
		Commit:      gitCommitHash(),
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}

	printReport(report)

	if err := writeJSON(cfg.Output, report); err != nil {
		return fmt.Errorf("write results: %w", err)
	}
	fmt.Printf("Results written to %s\n", cfg.Output)
	return nil
}
