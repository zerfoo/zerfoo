package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		want    BatchBenchConfig
		wantErr bool
	}{
		{
			name: "all flags",
			args: []string{
				"--model", "/path/to/model.gguf",
				"--backend", "cuda",
				"--sessions", "16",
				"--tokens", "256",
				"--warmup", "3",
				"--output", "results.json",
			},
			want: BatchBenchConfig{
				Model:    "/path/to/model.gguf",
				Backend:  "cuda",
				Sessions: 16,
				Tokens:   256,
				Warmup:   3,
				Output:   "results.json",
			},
		},
		{
			name: "defaults",
			args: []string{"--model", "/path/to/model.gguf"},
			want: BatchBenchConfig{
				Model:    "/path/to/model.gguf",
				Backend:  "cpu",
				Sessions: 8,
				Tokens:   128,
				Warmup:   2,
				Output:   "bench_batch_results.json",
			},
		},
		{
			name:    "invalid flag",
			args:    []string{"--nonexistent", "value"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFlags(tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseFlags() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got != tt.want {
				t.Errorf("parseFlags() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     BatchBenchConfig
		wantErr bool
	}{
		{
			name: "valid",
			cfg: BatchBenchConfig{
				Model:    "/path/to/model.gguf",
				Sessions: 8,
				Tokens:   128,
			},
		},
		{
			name: "valid no model",
			cfg: BatchBenchConfig{
				Sessions: 8,
				Tokens:   128,
			},
		},
		{
			name: "zero sessions",
			cfg: BatchBenchConfig{
				Model:    "/path/to/model.gguf",
				Sessions: 0,
				Tokens:   128,
			},
			wantErr: true,
		},
		{
			name: "zero tokens",
			cfg: BatchBenchConfig{
				Model:    "/path/to/model.gguf",
				Sessions: 8,
				Tokens:   0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConfig(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDefaultPrompts(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"single", 1},
		{"eight", 8},
		{"wraps", 15},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prompts := defaultPrompts(tt.n)
			if len(prompts) != tt.n {
				t.Errorf("defaultPrompts(%d) returned %d prompts", tt.n, len(prompts))
			}
			for i, p := range prompts {
				if p == "" {
					t.Errorf("prompt %d is empty", i)
				}
			}
		})
	}
}

// mockSessionRunner simulates session pool generation.
type mockSessionRunner struct {
	tokensPerCall int
	ttftMs        float64
	durationMs    float64
}

func (r *mockSessionRunner) Generate(_ context.Context, _ int, _ string, maxTokens int) (int, float64, float64, error) {
	tc := r.tokensPerCall
	if tc > maxTokens {
		tc = maxTokens
	}
	return tc, r.ttftMs, r.durationMs, nil
}

// mockBatchRunner simulates continuous batching.
type mockBatchRunner struct {
	tokensPerSession int
	ttftMs           float64
	totalDurationMs  float64
}

func (r *mockBatchRunner) BatchGenerate(_ context.Context, prompts []string, maxTokens int) ([]int, []float64, float64, error) {
	n := len(prompts)
	tokens := make([]int, n)
	ttfts := make([]float64, n)
	tc := r.tokensPerSession
	if tc > maxTokens {
		tc = maxTokens
	}
	for i := range n {
		tokens[i] = tc
		ttfts[i] = r.ttftMs
	}
	return tokens, ttfts, r.totalDurationMs, nil
}

func TestRunSessionPool(t *testing.T) {
	runner := &mockSessionRunner{
		tokensPerCall: 128,
		ttftMs:        2.0,
		durationMs:    64.0, // 128 tokens / 64ms = 2000 tok/s per session (but serialized)
	}
	prompts := defaultPrompts(8)

	result, err := runSessionPool(context.Background(), runner, prompts, 128, 1)
	if err != nil {
		t.Fatalf("runSessionPool: %v", err)
	}

	if result.Strategy != "session_pool" {
		t.Errorf("Strategy = %q, want %q", result.Strategy, "session_pool")
	}
	if result.Sessions != 8 {
		t.Errorf("Sessions = %d, want 8", result.Sessions)
	}
	// 8 sessions * 128 tokens = 1024 total tokens.
	if result.TotalTokens != 1024 {
		t.Errorf("TotalTokens = %d, want 1024", result.TotalTokens)
	}
	if result.TokPerSec <= 0 {
		t.Errorf("TokPerSec = %.2f, want > 0", result.TokPerSec)
	}
	if result.AvgTTFTMs != 2.0 {
		t.Errorf("AvgTTFTMs = %.2f, want 2.0", result.AvgTTFTMs)
	}
}

func TestRunContinuousBatching(t *testing.T) {
	runner := &mockBatchRunner{
		tokensPerSession: 128,
		ttftMs:           2.0,
		totalDurationMs:  80.0, // all 8 sessions done in 80ms
	}
	prompts := defaultPrompts(8)

	result, err := runContinuousBatching(context.Background(), runner, prompts, 128, 1)
	if err != nil {
		t.Fatalf("runContinuousBatching: %v", err)
	}

	if result.Strategy != "continuous_batching" {
		t.Errorf("Strategy = %q, want %q", result.Strategy, "continuous_batching")
	}
	if result.Sessions != 8 {
		t.Errorf("Sessions = %d, want 8", result.Sessions)
	}
	if result.TotalTokens != 1024 {
		t.Errorf("TotalTokens = %d, want 1024", result.TotalTokens)
	}
	if result.TokPerSec <= 0 {
		t.Errorf("TokPerSec = %.2f, want > 0", result.TokPerSec)
	}
}

// TestContinuousBatchingBenchmark runs the full benchmark simulation and
// asserts that continuous batching achieves >= 1.5x speedup over session pool.
func TestContinuousBatchingBenchmark(t *testing.T) {
	ctx := context.Background()
	prompts := defaultPrompts(8)
	maxTokens := 32 // short to keep test fast

	// Calibrate: single decode step = 100us, prefill = 500us.
	// Session pool: 8 sessions serial = 8 * (500us + 32*100us) = 8 * 3700us = 29.6ms.
	// Continuous batching: 8*500us prefill + 32*115us decode = 4000us + 3680us = 7.68ms.
	// Expected speedup: ~3.85x (well above 1.5x threshold).
	decodeStepUs := int64(100)
	prefillUs := int64(500)

	sessionRunner := &simulatedSessionRunner{
		decodeStepUs: decodeStepUs,
		prefillUs:    prefillUs,
	}
	batchRunner := &simulatedBatchRunner{
		decodeStepUs: decodeStepUs,
		prefillUs:    prefillUs,
		batchOverhead: 0.15,
	}

	poolResult, err := runSessionPool(ctx, sessionRunner, prompts, maxTokens, 1)
	if err != nil {
		t.Fatalf("session pool: %v", err)
	}

	batchResult, err := runContinuousBatching(ctx, batchRunner, prompts, maxTokens, 1)
	if err != nil {
		t.Fatalf("continuous batching: %v", err)
	}

	if poolResult.TokPerSec <= 0 {
		t.Fatal("session pool tok/s is zero")
	}
	if batchResult.TokPerSec <= 0 {
		t.Fatal("continuous batching tok/s is zero")
	}

	speedup := batchResult.TokPerSec / poolResult.TokPerSec
	t.Logf("Session pool:        %.2f tok/s (TTFT: %.2f ms)", poolResult.TokPerSec, poolResult.AvgTTFTMs)
	t.Logf("Continuous batching: %.2f tok/s (TTFT: %.2f ms)", batchResult.TokPerSec, batchResult.AvgTTFTMs)
	t.Logf("Speedup:             %.2fx", speedup)

	if speedup < 1.5 {
		t.Errorf("speedup %.2fx < 1.5x threshold", speedup)
	}

	// TTFT should be comparable (within 5x — prefill is serial in both strategies).
	ttftRatio := batchResult.AvgTTFTMs / poolResult.AvgTTFTMs
	if ttftRatio > 5.0 || ttftRatio < 0.2 {
		t.Errorf("TTFT ratio %.2f is out of expected range [0.2, 5.0]", ttftRatio)
	}
}

func TestWriteJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_results.json")

	report := BatchBenchReport{
		Model:      "/path/to/model.gguf",
		Backend:    "cuda",
		Sessions:   8,
		TokensEach: 128,
		SessionPool: BenchResult{
			Strategy:       "session_pool",
			Sessions:       8,
			TotalTokens:    1024,
			TotalDurationS: 10.0,
			TokPerSec:      102.4,
			AvgTTFTMs:      2.0,
		},
		Continuous: BenchResult{
			Strategy:       "continuous_batching",
			Sessions:       8,
			TotalTokens:    1024,
			TotalDurationS: 4.2,
			TokPerSec:      243.8,
			AvgTTFTMs:      2.1,
		},
		Speedup:   2.38,
		Commit:    "abc1234",
		Timestamp: "2026-03-18T00:00:00Z",
	}

	if err := writeJSON(path, report); err != nil {
		t.Fatalf("writeJSON: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	var got BatchBenchReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if got.Speedup != 2.38 {
		t.Errorf("Speedup = %v, want 2.38", got.Speedup)
	}
	if got.SessionPool.TokPerSec != 102.4 {
		t.Errorf("SessionPool.TokPerSec = %v, want 102.4", got.SessionPool.TokPerSec)
	}
	if got.Continuous.TokPerSec != 243.8 {
		t.Errorf("Continuous.TokPerSec = %v, want 243.8", got.Continuous.TokPerSec)
	}
}

func TestPrintReport(t *testing.T) {
	report := BatchBenchReport{
		Model:      "model.gguf",
		Backend:    "cpu",
		Sessions:   8,
		TokensEach: 128,
		SessionPool: BenchResult{
			Strategy:  "session_pool",
			TokPerSec: 100.0,
			AvgTTFTMs: 2.0,
		},
		Continuous: BenchResult{
			Strategy:  "continuous_batching",
			TokPerSec: 240.0,
			AvgTTFTMs: 2.1,
		},
		Speedup: 2.4,
	}
	// Should not panic.
	printReport(report)
}
