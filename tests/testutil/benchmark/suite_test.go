package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
)

// mockInference returns a fixed set of metrics for testing.
func mockInference(decode, prefill, mem, ttft float64) InferenceFunc {
	return func(_ context.Context, _ ModelSpec, _ string, _ int) (RunMetrics, error) {
		return RunMetrics{
			DecodeTokensPerSec:  decode,
			PrefillTokensPerSec: prefill,
			MemoryUsageMB:       mem,
			TimeToFirstTokenMS:  ttft,
		}, nil
	}
}

// countingInference tracks invocation count for verifying warmup/benchmark calls.
func countingInference(count *int, m RunMetrics) InferenceFunc {
	return func(_ context.Context, _ ModelSpec, _ string, _ int) (RunMetrics, error) {
		*count++
		return m, nil
	}
}

func TestBenchmarkSuite_Config(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		wantErr string
	}{
		{
			name: "valid config",
			config: Config{
				Models:        []ModelSpec{{Path: "/tmp/m.gguf", Name: "test", Architecture: "llama"}},
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    []int{1},
				WarmupRuns:    2,
				BenchmarkRuns: 5,
			},
		},
		{
			name: "no models",
			config: Config{
				Models:        nil,
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    []int{1},
				BenchmarkRuns: 1,
			},
			wantErr: "at least one model",
		},
		{
			name: "no quantizations",
			config: Config{
				Models:        []ModelSpec{{Name: "m"}},
				Quantizations: nil,
				BatchSizes:    []int{1},
				BenchmarkRuns: 1,
			},
			wantErr: "at least one quantization",
		},
		{
			name: "no batch sizes",
			config: Config{
				Models:        []ModelSpec{{Name: "m"}},
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    nil,
				BenchmarkRuns: 1,
			},
			wantErr: "at least one batch size",
		},
		{
			name: "zero batch size",
			config: Config{
				Models:        []ModelSpec{{Name: "m"}},
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    []int{0},
				BenchmarkRuns: 1,
			},
			wantErr: "batch size must be >= 1",
		},
		{
			name: "negative warmup",
			config: Config{
				Models:        []ModelSpec{{Name: "m"}},
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    []int{1},
				WarmupRuns:    -1,
				BenchmarkRuns: 1,
			},
			wantErr: "warmup runs must be >= 0",
		},
		{
			name: "zero benchmark runs",
			config: Config{
				Models:        []ModelSpec{{Name: "m"}},
				Quantizations: []string{"Q4_K_M"},
				BatchSizes:    []int{1},
				BenchmarkRuns: 0,
			},
			wantErr: "benchmark runs must be >= 1",
		},
		{
			name: "multiple models and batch sizes",
			config: Config{
				Models: []ModelSpec{
					{Path: "/m1.gguf", Name: "model-a", Architecture: "llama"},
					{Path: "/m2.gguf", Name: "model-b", Architecture: "gemma"},
				},
				Quantizations: []string{"Q4_K_M", "Q8_0"},
				BatchSizes:    []int{1, 4, 8},
				WarmupRuns:    1,
				BenchmarkRuns: 3,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if got := err.Error(); !contains(got, tt.wantErr) {
					t.Fatalf("error %q does not contain %q", got, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Valid configs should also successfully create a suite.
			suite, err := NewSuite(tt.config, mockInference(100, 200, 512, 10))
			if err != nil {
				t.Fatalf("NewSuite: %v", err)
			}
			if suite == nil {
				t.Fatal("NewSuite returned nil suite")
			}
		})
	}
}

func TestBenchmarkSuite_NilInference(t *testing.T) {
	cfg := Config{
		Models:        []ModelSpec{{Name: "m"}},
		Quantizations: []string{"Q4_K_M"},
		BatchSizes:    []int{1},
		BenchmarkRuns: 1,
	}
	_, err := NewSuite(cfg, nil)
	if err == nil {
		t.Fatal("expected error for nil inference function")
	}
}

func TestBenchmarkSuite_JSONOutput(t *testing.T) {
	cfg := Config{
		Models: []ModelSpec{
			{Path: "/models/llama.gguf", Name: "llama-3-1b", Architecture: "llama"},
			{Path: "/models/gemma.gguf", Name: "gemma-3-1b", Architecture: "gemma"},
		},
		Quantizations: []string{"Q4_K_M", "FP16"},
		BatchSizes:    []int{1, 4},
		WarmupRuns:    1,
		BenchmarkRuns: 3,
	}

	suite, err := NewSuite(cfg, mockInference(245.5, 1200.0, 512.0, 12.3))
	if err != nil {
		t.Fatalf("NewSuite: %v", err)
	}

	results, err := suite.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 2 models x 2 quantizations x 2 batch sizes = 8 results.
	if got := len(results); got != 8 {
		t.Fatalf("len(results) = %d, want 8", got)
	}

	// Marshal to JSON and back.
	data, err := ResultsJSON(results)
	if err != nil {
		t.Fatalf("ResultsJSON: %v", err)
	}

	var decoded []BenchmarkResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if len(decoded) != len(results) {
		t.Fatalf("decoded length %d != original %d", len(decoded), len(results))
	}

	// Spot-check first result.
	r := decoded[0]
	if r.ModelName != "llama-3-1b" {
		t.Errorf("ModelName = %q, want %q", r.ModelName, "llama-3-1b")
	}
	if r.Quantization != "Q4_K_M" {
		t.Errorf("Quantization = %q, want %q", r.Quantization, "Q4_K_M")
	}
	if r.BatchSize != 1 {
		t.Errorf("BatchSize = %d, want 1", r.BatchSize)
	}
	if r.DecodeTokensPerSec != 245.5 {
		t.Errorf("DecodeTokensPerSec = %v, want 245.5", r.DecodeTokensPerSec)
	}
	if r.PrefillTokensPerSec != 1200.0 {
		t.Errorf("PrefillTokensPerSec = %v, want 1200.0", r.PrefillTokensPerSec)
	}
	if r.MemoryUsageMB != 512.0 {
		t.Errorf("MemoryUsageMB = %v, want 512.0", r.MemoryUsageMB)
	}
	if diff := r.TimeToFirstTokenMS - 12.3; diff < -0.01 || diff > 0.01 {
		t.Errorf("TimeToFirstTokenMS = %v, want ~12.3", r.TimeToFirstTokenMS)
	}
	if r.Timestamp == "" {
		t.Error("Timestamp is empty")
	}

	// Verify all JSON fields are present by checking raw JSON keys.
	var rawSlice []map[string]any
	if err := json.Unmarshal(data, &rawSlice); err != nil {
		t.Fatalf("raw unmarshal: %v", err)
	}
	raw := rawSlice[0]
	requiredKeys := []string{
		"model_name", "quantization", "batch_size",
		"decode_tokens_per_sec", "prefill_tokens_per_sec",
		"memory_usage_mb", "time_to_first_token_ms", "timestamp",
	}
	for _, key := range requiredKeys {
		if _, ok := raw[key]; !ok {
			t.Errorf("JSON missing required key %q", key)
		}
	}
}

func TestBenchmarkSuite_Metrics(t *testing.T) {
	// Track call count to verify warmup + benchmark run counts.
	var callCount int
	metrics := RunMetrics{
		DecodeTokensPerSec:  200.0,
		PrefillTokensPerSec: 1000.0,
		MemoryUsageMB:       256.0,
		TimeToFirstTokenMS:  8.5,
	}

	cfg := Config{
		Models:        []ModelSpec{{Name: "test-model", Path: "/m.gguf", Architecture: "llama"}},
		Quantizations: []string{"Q4_K_M"},
		BatchSizes:    []int{1},
		WarmupRuns:    3,
		BenchmarkRuns: 5,
	}

	suite, err := NewSuite(cfg, countingInference(&callCount, metrics))
	if err != nil {
		t.Fatalf("NewSuite: %v", err)
	}

	results, err := suite.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 3 warmup + 5 benchmark = 8 total calls.
	if callCount != 8 {
		t.Errorf("callCount = %d, want 8 (3 warmup + 5 benchmark)", callCount)
	}

	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1", len(results))
	}

	r := results[0]
	if r.DecodeTokensPerSec != 200.0 {
		t.Errorf("DecodeTokensPerSec = %v, want 200.0", r.DecodeTokensPerSec)
	}
	if r.PrefillTokensPerSec != 1000.0 {
		t.Errorf("PrefillTokensPerSec = %v, want 1000.0", r.PrefillTokensPerSec)
	}
	if r.MemoryUsageMB != 256.0 {
		t.Errorf("MemoryUsageMB = %v, want 256.0", r.MemoryUsageMB)
	}
	if r.TimeToFirstTokenMS != 8.5 {
		t.Errorf("TimeToFirstTokenMS = %v, want 8.5", r.TimeToFirstTokenMS)
	}
}

func TestBenchmarkSuite_MultiCombination(t *testing.T) {
	// Verify that inference func receives the correct parameters.
	type call struct {
		model string
		quant string
		bs    int
	}
	var calls []call

	infer := func(_ context.Context, m ModelSpec, q string, bs int) (RunMetrics, error) {
		calls = append(calls, call{model: m.Name, quant: q, bs: bs})
		return RunMetrics{DecodeTokensPerSec: float64(bs) * 100}, nil
	}

	cfg := Config{
		Models: []ModelSpec{
			{Name: "alpha"},
			{Name: "beta"},
		},
		Quantizations: []string{"Q4", "Q8"},
		BatchSizes:    []int{1, 2},
		WarmupRuns:    0,
		BenchmarkRuns: 1,
	}

	suite, err := NewSuite(cfg, infer)
	if err != nil {
		t.Fatalf("NewSuite: %v", err)
	}

	results, err := suite.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 2 models x 2 quants x 2 batch sizes = 8 combinations, 0 warmup + 1 run each = 8 calls.
	if len(calls) != 8 {
		t.Errorf("len(calls) = %d, want 8", len(calls))
	}
	if len(results) != 8 {
		t.Errorf("len(results) = %d, want 8", len(results))
	}

	// Verify batch-size-dependent metric.
	for _, r := range results {
		expected := float64(r.BatchSize) * 100
		if r.DecodeTokensPerSec != expected {
			t.Errorf("%s/%s/bs%d: DecodeTokensPerSec = %v, want %v",
				r.ModelName, r.Quantization, r.BatchSize, r.DecodeTokensPerSec, expected)
		}
	}
}

func TestBenchmarkSuite_InferenceError(t *testing.T) {
	infer := func(_ context.Context, _ ModelSpec, _ string, _ int) (RunMetrics, error) {
		return RunMetrics{}, fmt.Errorf("gpu out of memory")
	}

	cfg := Config{
		Models:        []ModelSpec{{Name: "m"}},
		Quantizations: []string{"Q4"},
		BatchSizes:    []int{1},
		WarmupRuns:    0,
		BenchmarkRuns: 1,
	}

	suite, err := NewSuite(cfg, infer)
	if err != nil {
		t.Fatalf("NewSuite: %v", err)
	}

	_, err = suite.Run(context.Background())
	if err == nil {
		t.Fatal("expected error from failing inference function")
	}
	if !contains(err.Error(), "gpu out of memory") {
		t.Errorf("error %q does not contain %q", err.Error(), "gpu out of memory")
	}
}

func TestBenchmarkSuite_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	cfg := Config{
		Models:        []ModelSpec{{Name: "m"}},
		Quantizations: []string{"Q4"},
		BatchSizes:    []int{1},
		WarmupRuns:    1,
		BenchmarkRuns: 1,
	}

	suite, err := NewSuite(cfg, mockInference(100, 200, 512, 10))
	if err != nil {
		t.Fatalf("NewSuite: %v", err)
	}

	_, err = suite.Run(ctx)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}

func TestMean(t *testing.T) {
	tests := []struct {
		name    string
		metrics []RunMetrics
		want    float64
	}{
		{name: "empty", metrics: nil, want: 0},
		{name: "single", metrics: []RunMetrics{{DecodeTokensPerSec: 100}}, want: 100},
		{name: "average", metrics: []RunMetrics{
			{DecodeTokensPerSec: 100},
			{DecodeTokensPerSec: 200},
			{DecodeTokensPerSec: 300},
		}, want: 200},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mean(tt.metrics, func(m RunMetrics) float64 { return m.DecodeTokensPerSec })
			if got != tt.want {
				t.Errorf("mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStddev(t *testing.T) {
	metrics := []RunMetrics{
		{DecodeTokensPerSec: 10},
		{DecodeTokensPerSec: 10},
		{DecodeTokensPerSec: 10},
	}
	got := stddev(metrics, func(m RunMetrics) float64 { return m.DecodeTokensPerSec })
	if got != 0 {
		t.Errorf("stddev of identical values = %v, want 0", got)
	}
}

func TestPercentile(t *testing.T) {
	values := make([]float64, 100)
	for i := range values {
		values[i] = float64(i + 1)
	}
	got := percentile(values, 99)
	if got != 99.0 {
		t.Errorf("percentile(99) = %v, want 99.0", got)
	}
	got = percentile(nil, 99)
	if got != 0 {
		t.Errorf("percentile(nil, 99) = %v, want 0", got)
	}
}

// contains reports whether s contains substr.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

