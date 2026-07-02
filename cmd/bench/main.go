// Command bench runs a standardized benchmark harness for zerfoo models.
//
// Usage:
//
//	bench --model /path/to/model.gguf [--backend cpu] [--tokens 100] [--concurrent 1] [--warmup 3] [--output bench_results.json] [--prompt "Once upon a time"]
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// BenchmarkRunner abstracts the generation backend so it can be mocked in tests.
type BenchmarkRunner interface {
	// Run generates tokens from the given prompt and returns metrics.
	Run(ctx context.Context, prompt string, tokens int) (result RunResult, err error)
}

// RunResult holds the raw metrics from a single generation run.
type RunResult struct {
	TokensGenerated int
	TTFTMs          float64
	LatenciesMs     []float64
	GPUMemoryMB     float64
}

// BenchmarkResult holds the aggregated benchmark output.
type BenchmarkResult struct {
	Model        string  `json:"model"`
	Backend      string  `json:"backend"`
	Tokens       int     `json:"tokens"`
	Concurrent   int     `json:"concurrent"`
	ThroughputTs float64 `json:"throughput_toks"`
	TTFTMs       float64 `json:"ttft_ms"`
	P99LatencyMs float64 `json:"p99_latency_ms"`
	GPUMemoryMB  float64 `json:"gpu_memory_mb"`
	Timestamp    string  `json:"timestamp"`
	Commit       string  `json:"commit"`
}

// ComputeP99 returns the 99th percentile value from a sorted slice of latencies.
// Returns 0 if the slice is empty.
func ComputeP99(latencies []float64) float64 {
	n := len(latencies)
	if n == 0 {
		return 0
	}
	sorted := make([]float64, n)
	copy(sorted, latencies)
	sort.Float64s(sorted)
	idx := int(math.Ceil(0.99*float64(n))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return sorted[idx]
}

// gitCommitHash returns the short git commit hash, or "unknown" on failure.
func gitCommitHash() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// modelRunner wraps an inference.Model to implement BenchmarkRunner.
type modelRunner struct {
	model *inference.Model
}

func (r *modelRunner) Run(ctx context.Context, prompt string, tokens int) (RunResult, error) {
	var result RunResult
	var firstTokenTime time.Time
	var tokenCount atomic.Int64
	var latencies []float64
	var mu sync.Mutex
	var lastTokenTime time.Time

	start := time.Now()
	lastTokenTime = start

	handler := generate.TokenStreamFunc(func(token string, done bool) error {
		if done {
			return nil
		}
		now := time.Now()
		tokenCount.Add(1)
		if tokenCount.Load() == 1 {
			firstTokenTime = now
		}
		mu.Lock()
		latencies = append(latencies, float64(now.Sub(lastTokenTime).Microseconds())/1000.0)
		lastTokenTime = now
		mu.Unlock()
		return nil
	})

	err := r.model.GenerateStream(ctx, prompt, handler, inference.WithMaxTokens(tokens), inference.WithTemperature(0))
	if err != nil {
		return result, err
	}

	result.TokensGenerated = int(tokenCount.Load())
	if !firstTokenTime.IsZero() {
		result.TTFTMs = float64(firstTokenTime.Sub(start).Microseconds()) / 1000.0
	}
	result.LatenciesMs = latencies
	return result, nil
}

// runBenchmark executes the benchmark with the given configuration.
func runBenchmark(ctx context.Context, runner BenchmarkRunner, cfg BenchmarkResult, prompt string, warmup int) (BenchmarkResult, error) {
	// Warmup runs.
	for i := 0; i < warmup; i++ {
		_, err := runner.Run(ctx, prompt, 4)
		if err != nil {
			return cfg, fmt.Errorf("warmup %d: %w", i, err)
		}
	}

	// Actual benchmark run(s).
	type sessionResult struct {
		result RunResult
		err    error
		dur    time.Duration
	}

	results := make([]sessionResult, cfg.Concurrent)
	var wg sync.WaitGroup
	for i := range cfg.Concurrent {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			start := time.Now()
			res, err := runner.Run(ctx, prompt, cfg.Tokens)
			results[idx] = sessionResult{result: res, err: err, dur: time.Since(start)}
		}(i)
	}
	wg.Wait()

	// Aggregate results.
	var totalTokens int
	var totalDur time.Duration
	var allLatencies []float64
	var sumTTFT float64
	var ttftCount int
	var maxGPUMem float64

	for _, sr := range results {
		if sr.err != nil {
			return cfg, fmt.Errorf("session error: %w", sr.err)
		}
		totalTokens += sr.result.TokensGenerated
		totalDur += sr.dur
		allLatencies = append(allLatencies, sr.result.LatenciesMs...)
		if sr.result.TTFTMs > 0 {
			sumTTFT += sr.result.TTFTMs
			ttftCount++
		}
		if sr.result.GPUMemoryMB > maxGPUMem {
			maxGPUMem = sr.result.GPUMemoryMB
		}
	}

	// For concurrent runs, throughput is total tokens / average wall time.
	avgDur := totalDur / time.Duration(cfg.Concurrent)
	if avgDur > 0 {
		cfg.ThroughputTs = float64(totalTokens) / avgDur.Seconds()
	}
	if ttftCount > 0 {
		cfg.TTFTMs = sumTTFT / float64(ttftCount)
	}
	cfg.P99LatencyMs = ComputeP99(allLatencies)
	cfg.GPUMemoryMB = maxGPUMem
	cfg.Timestamp = time.Now().UTC().Format(time.RFC3339)
	cfg.Commit = gitCommitHash()

	return cfg, nil
}

// printResults writes the benchmark results to stdout.
func printResults(r BenchmarkResult) {
	fmt.Printf("Model: %s\n", r.Model)
	fmt.Printf("Backend: %s\n", r.Backend)
	fmt.Printf("Tokens: %d\n", r.Tokens)
	fmt.Printf("Concurrent: %d\n", r.Concurrent)
	fmt.Println("Results:")
	fmt.Printf("  Throughput: %.2f tok/s\n", r.ThroughputTs)
	fmt.Printf("  TTFT: %.2f ms\n", r.TTFTMs)
	fmt.Printf("  P99 Latency: %.2f ms\n", r.P99LatencyMs)
	fmt.Printf("  GPU Memory: %.2f MB\n", r.GPUMemoryMB)
}

// writeJSON writes the benchmark results to a JSON file.
func writeJSON(path string, r BenchmarkResult) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
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

	modelPath := flag.String("model", "", "path to GGUF model file")
	backend := flag.String("backend", "cpu", "backend: cpu|cuda|rocm")
	tokens := flag.Int("tokens", 100, "number of tokens to generate")
	concurrent := flag.Int("concurrent", 1, "number of concurrent sessions")
	warmup := flag.Int("warmup", 3, "warmup iterations")
	output := flag.String("output", "bench_results.json", "path for JSON results file")
	prompt := flag.String("prompt", "Once upon a time", "input prompt")
	flag.Parse()

	if *modelPath == "" {
		return fmt.Errorf("usage: bench --model /path/to/model.gguf")
	}

	fmt.Printf("Loading model from %s (backend=%s)...\n", *modelPath, *backend)
	mdl, err := inference.LoadFile(*modelPath, inference.WithDevice(*backend))
	if err != nil {
		return fmt.Errorf("load error: %w", err)
	}

	runner := &modelRunner{model: mdl}
	cfg := BenchmarkResult{
		Model:      *modelPath,
		Backend:    *backend,
		Tokens:     *tokens,
		Concurrent: *concurrent,
	}

	result, err := runBenchmark(context.Background(), runner, cfg, *prompt, *warmup)
	if err != nil {
		return fmt.Errorf("benchmark error: %w", err)
	}

	printResults(result)

	if err := writeJSON(*output, result); err != nil {
		return err
	}
	fmt.Printf("\nResults written to %s\n", *output)
	return nil
}
