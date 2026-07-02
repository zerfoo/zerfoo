// Package benchmark provides a standardized benchmark suite for measuring
// ML model inference performance: tok/s decode, tok/s prefill, memory usage,
// and time to first token.
package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"
)

// ModelSpec identifies a model to benchmark.
type ModelSpec struct {
	Path         string `json:"path"`
	Name         string `json:"name"`
	Architecture string `json:"architecture"`
}

// Config controls what the benchmark suite measures.
type Config struct {
	Models        []ModelSpec `json:"models"`
	Quantizations []string   `json:"quantizations"`
	BatchSizes    []int      `json:"batch_sizes"`
	WarmupRuns    int        `json:"warmup_runs"`
	BenchmarkRuns int        `json:"benchmark_runs"`
}

// Validate checks that the configuration is well-formed.
func (c Config) Validate() error {
	if len(c.Models) == 0 {
		return fmt.Errorf("benchmark: at least one model is required")
	}
	if len(c.Quantizations) == 0 {
		return fmt.Errorf("benchmark: at least one quantization is required")
	}
	if len(c.BatchSizes) == 0 {
		return fmt.Errorf("benchmark: at least one batch size is required")
	}
	for _, bs := range c.BatchSizes {
		if bs < 1 {
			return fmt.Errorf("benchmark: batch size must be >= 1, got %d", bs)
		}
	}
	if c.WarmupRuns < 0 {
		return fmt.Errorf("benchmark: warmup runs must be >= 0, got %d", c.WarmupRuns)
	}
	if c.BenchmarkRuns < 1 {
		return fmt.Errorf("benchmark: benchmark runs must be >= 1, got %d", c.BenchmarkRuns)
	}
	return nil
}

// BenchmarkResult holds the metrics from a single benchmark configuration.
type BenchmarkResult struct {
	ModelName           string  `json:"model_name"`
	Quantization        string  `json:"quantization"`
	BatchSize           int     `json:"batch_size"`
	DecodeTokensPerSec  float64 `json:"decode_tokens_per_sec"`
	PrefillTokensPerSec float64 `json:"prefill_tokens_per_sec"`
	MemoryUsageMB       float64 `json:"memory_usage_mb"`
	TimeToFirstTokenMS  float64 `json:"time_to_first_token_ms"`
	Timestamp           string  `json:"timestamp"`
}

// RunMetrics holds the raw measurements from a single inference run.
type RunMetrics struct {
	DecodeTokensPerSec  float64
	PrefillTokensPerSec float64
	MemoryUsageMB       float64
	TimeToFirstTokenMS  float64
}

// InferenceFunc is the function signature that the suite calls to run a single
// inference benchmark. Implementations should return metrics for one run of
// the given model, quantization, and batch size.
type InferenceFunc func(ctx context.Context, model ModelSpec, quantization string, batchSize int) (RunMetrics, error)

// Suite orchestrates running standardized benchmarks across all combinations
// of models, quantizations, and batch sizes.
type Suite struct {
	config Config
	infer  InferenceFunc
}

// NewSuite creates a benchmark suite with the given configuration and inference function.
func NewSuite(cfg Config, infer InferenceFunc) (*Suite, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if infer == nil {
		return nil, fmt.Errorf("benchmark: inference function must not be nil")
	}
	return &Suite{config: cfg, infer: infer}, nil
}

// Run executes all model x quantization x batch_size combinations, performing
// warmup runs followed by benchmark runs. It returns one BenchmarkResult per
// combination with mean metrics across the benchmark runs.
func (s *Suite) Run(ctx context.Context) ([]BenchmarkResult, error) {
	var results []BenchmarkResult

	for _, model := range s.config.Models {
		for _, quant := range s.config.Quantizations {
			for _, bs := range s.config.BatchSizes {
				result, err := s.runSingle(ctx, model, quant, bs)
				if err != nil {
					return results, fmt.Errorf("benchmark %s/%s/bs%d: %w", model.Name, quant, bs, err)
				}
				results = append(results, result)
			}
		}
	}

	return results, nil
}

// runSingle runs warmup + benchmark iterations for a single combination and
// returns the mean result.
func (s *Suite) runSingle(ctx context.Context, model ModelSpec, quant string, batchSize int) (BenchmarkResult, error) {
	// Warmup runs (results discarded).
	for i := 0; i < s.config.WarmupRuns; i++ {
		if err := ctx.Err(); err != nil {
			return BenchmarkResult{}, err
		}
		if _, err := s.infer(ctx, model, quant, batchSize); err != nil {
			return BenchmarkResult{}, fmt.Errorf("warmup %d: %w", i, err)
		}
	}

	// Benchmark runs.
	metrics := make([]RunMetrics, 0, s.config.BenchmarkRuns)
	for i := 0; i < s.config.BenchmarkRuns; i++ {
		if err := ctx.Err(); err != nil {
			return BenchmarkResult{}, err
		}
		m, err := s.infer(ctx, model, quant, batchSize)
		if err != nil {
			return BenchmarkResult{}, fmt.Errorf("run %d: %w", i, err)
		}
		metrics = append(metrics, m)
	}

	return BenchmarkResult{
		ModelName:           model.Name,
		Quantization:        quant,
		BatchSize:           batchSize,
		DecodeTokensPerSec:  mean(metrics, func(m RunMetrics) float64 { return m.DecodeTokensPerSec }),
		PrefillTokensPerSec: mean(metrics, func(m RunMetrics) float64 { return m.PrefillTokensPerSec }),
		MemoryUsageMB:       mean(metrics, func(m RunMetrics) float64 { return m.MemoryUsageMB }),
		TimeToFirstTokenMS:  mean(metrics, func(m RunMetrics) float64 { return m.TimeToFirstTokenMS }),
		Timestamp:           time.Now().UTC().Format(time.RFC3339),
	}, nil
}

// ResultsJSON returns the benchmark results as a JSON byte slice.
func ResultsJSON(results []BenchmarkResult) ([]byte, error) {
	return json.MarshalIndent(results, "", "  ")
}

// RunB is a helper for integrating with Go's testing.B. It creates a suite
// and runs it within the benchmark function, reporting decode tok/s as the
// benchmark metric.
func RunB(b *testing.B, cfg Config, infer InferenceFunc) []BenchmarkResult {
	b.Helper()
	suite, err := NewSuite(cfg, infer)
	if err != nil {
		b.Fatal(err)
	}

	var allResults []BenchmarkResult
	b.ResetTimer()
	for range b.N {
		results, err := suite.Run(b.Context())
		if err != nil {
			b.Fatal(err)
		}
		allResults = results
	}
	b.StopTimer()

	// Report the mean decode tok/s across all results as a custom metric.
	if len(allResults) > 0 {
		var total float64
		for _, r := range allResults {
			total += r.DecodeTokensPerSec
		}
		b.ReportMetric(total/float64(len(allResults)), "decode-tok/s")
	}

	return allResults
}

// mean computes the arithmetic mean of a field extracted from a slice of RunMetrics.
func mean(metrics []RunMetrics, field func(RunMetrics) float64) float64 {
	if len(metrics) == 0 {
		return 0
	}
	var sum float64
	for _, m := range metrics {
		sum += field(m)
	}
	return sum / float64(len(metrics))
}

// stddev computes the population standard deviation of a field extracted from a
// slice of RunMetrics.
func stddev(metrics []RunMetrics, field func(RunMetrics) float64) float64 {
	if len(metrics) == 0 {
		return 0
	}
	avg := mean(metrics, field)
	var sumSq float64
	for _, m := range metrics {
		d := field(m) - avg
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(metrics)))
}

// percentile returns the p-th percentile (0-100) from a slice of float64 values.
func percentile(values []float64, p float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)
	idx := int(math.Ceil(p/100.0*float64(n))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return sorted[idx]
}
