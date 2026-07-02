package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// mockRunner implements BenchmarkRunner for testing.
type mockRunner struct {
	tokensGenerated int
	ttftMs          float64
	latenciesMs     []float64
	gpuMemoryMB     float64
	err             error
	callCount       atomic.Int64
}

func (m *mockRunner) Run(_ context.Context, _ string, tokens int) (RunResult, error) {
	m.callCount.Add(1)
	if m.err != nil {
		return RunResult{}, m.err
	}
	gen := m.tokensGenerated
	if gen == 0 {
		gen = tokens
	}
	return RunResult{
		TokensGenerated: gen,
		TTFTMs:          m.ttftMs,
		LatenciesMs:     m.latenciesMs,
		GPUMemoryMB:     m.gpuMemoryMB,
	}, nil
}

func TestComputeP99(t *testing.T) {
	tests := []struct {
		name      string
		latencies []float64
		want      float64
	}{
		{name: "empty", latencies: nil, want: 0},
		{name: "single", latencies: []float64{5.0}, want: 5.0},
		{name: "two", latencies: []float64{1.0, 10.0}, want: 10.0},
		{name: "hundred", latencies: func() []float64 {
			ls := make([]float64, 100)
			for i := range ls {
				ls[i] = float64(i + 1)
			}
			return ls
		}(), want: 99.0},
		{name: "unsorted", latencies: []float64{10.0, 1.0, 5.0, 3.0, 8.0}, want: 10.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeP99(tt.latencies)
			if got != tt.want {
				t.Errorf("ComputeP99() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBenchResultsJSON(t *testing.T) {
	r := BenchmarkResult{
		Model:        "test-model.gguf",
		Backend:      "cpu",
		Tokens:       100,
		Concurrent:   2,
		ThroughputTs: 245.5,
		TTFTMs:       12.3,
		P99LatencyMs: 4.5,
		GPUMemoryMB:  1024.0,
		Timestamp:    "2026-03-17T00:00:00Z",
		Commit:       "abc1234",
	}

	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded BenchmarkResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Model != r.Model {
		t.Errorf("Model = %q, want %q", decoded.Model, r.Model)
	}
	if decoded.ThroughputTs != r.ThroughputTs {
		t.Errorf("ThroughputTs = %v, want %v", decoded.ThroughputTs, r.ThroughputTs)
	}
	if decoded.TTFTMs != r.TTFTMs {
		t.Errorf("TTFTMs = %v, want %v", decoded.TTFTMs, r.TTFTMs)
	}
	if decoded.P99LatencyMs != r.P99LatencyMs {
		t.Errorf("P99LatencyMs = %v, want %v", decoded.P99LatencyMs, r.P99LatencyMs)
	}
	if decoded.GPUMemoryMB != r.GPUMemoryMB {
		t.Errorf("GPUMemoryMB = %v, want %v", decoded.GPUMemoryMB, r.GPUMemoryMB)
	}
	if decoded.Concurrent != r.Concurrent {
		t.Errorf("Concurrent = %v, want %v", decoded.Concurrent, r.Concurrent)
	}

	// Verify JSON output file writing.
	dir := t.TempDir()
	path := filepath.Join(dir, "results.json")
	if err := writeJSON(path, r); err != nil {
		t.Fatalf("writeJSON: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	var fileResult BenchmarkResult
	if err := json.Unmarshal(raw, &fileResult); err != nil {
		t.Fatalf("unmarshal file: %v", err)
	}
	if fileResult.Model != r.Model {
		t.Errorf("file Model = %q, want %q", fileResult.Model, r.Model)
	}
}

func TestBenchHarness(t *testing.T) {
	runner := &mockRunner{
		tokensGenerated: 50,
		ttftMs:          15.0,
		latenciesMs:     []float64{2.0, 3.0, 2.5, 4.0, 1.5, 3.5, 2.0, 5.0, 2.5, 3.0},
		gpuMemoryMB:     512.0,
	}

	cfg := BenchmarkResult{
		Model:      "test.gguf",
		Backend:    "cpu",
		Tokens:     50,
		Concurrent: 1,
	}

	result, err := runBenchmark(context.Background(), runner, cfg, "hello", 2)
	if err != nil {
		t.Fatalf("runBenchmark: %v", err)
	}

	// Warmup should have been called (2 iterations).
	// Plus 1 actual run = 3 total calls.
	if runner.callCount.Load() != 3 {
		t.Errorf("callCount = %d, want 3", runner.callCount.Load())
	}

	if result.Model != "test.gguf" {
		t.Errorf("Model = %q, want %q", result.Model, "test.gguf")
	}
	if result.Backend != "cpu" {
		t.Errorf("Backend = %q, want %q", result.Backend, "cpu")
	}
	if result.ThroughputTs <= 0 {
		t.Errorf("ThroughputTs = %v, want > 0", result.ThroughputTs)
	}
	if result.TTFTMs != 15.0 {
		t.Errorf("TTFTMs = %v, want 15.0", result.TTFTMs)
	}
	if result.P99LatencyMs <= 0 {
		t.Errorf("P99LatencyMs = %v, want > 0", result.P99LatencyMs)
	}
	if result.GPUMemoryMB != 512.0 {
		t.Errorf("GPUMemoryMB = %v, want 512.0", result.GPUMemoryMB)
	}
	if result.Timestamp == "" {
		t.Error("Timestamp is empty")
	}
	if result.Commit == "" {
		t.Error("Commit is empty")
	}
}

func TestBenchHarnessConcurrent(t *testing.T) {
	runner := &mockRunner{
		tokensGenerated: 25,
		ttftMs:          10.0,
		latenciesMs:     []float64{2.0, 3.0, 4.0},
		gpuMemoryMB:     256.0,
	}

	cfg := BenchmarkResult{
		Model:      "test.gguf",
		Backend:    "cuda",
		Tokens:     25,
		Concurrent: 4,
	}

	result, err := runBenchmark(context.Background(), runner, cfg, "hello", 1)
	if err != nil {
		t.Fatalf("runBenchmark: %v", err)
	}

	// 1 warmup + 4 concurrent = 5 total calls.
	if runner.callCount.Load() != 5 {
		t.Errorf("callCount = %d, want 5", runner.callCount.Load())
	}

	if result.Concurrent != 4 {
		t.Errorf("Concurrent = %d, want 4", result.Concurrent)
	}
	// 4 sessions * 25 tokens = 100 total tokens.
	if result.ThroughputTs <= 0 {
		t.Errorf("ThroughputTs = %v, want > 0", result.ThroughputTs)
	}
	// All 4 sessions have same GPU mem, so max = 256.
	if result.GPUMemoryMB != 256.0 {
		t.Errorf("GPUMemoryMB = %v, want 256.0", result.GPUMemoryMB)
	}
}
