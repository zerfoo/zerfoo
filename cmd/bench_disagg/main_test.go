package main

import (
	"bytes"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestBenchDisaggregated(t *testing.T) {
	result, err := benchDisaggregated(4, 8, 10)
	if err != nil {
		t.Fatalf("benchDisaggregated: %v", err)
	}
	if result.Mode != "disaggregated" {
		t.Errorf("Mode = %q, want %q", result.Mode, "disaggregated")
	}
	if result.ReqPerSec <= 0 {
		t.Errorf("ReqPerSec = %v, want > 0", result.ReqPerSec)
	}
	if result.MeanTTFTMs <= 0 {
		t.Errorf("MeanTTFTMs = %v, want > 0", result.MeanTTFTMs)
	}
	if result.P99Ms <= 0 {
		t.Errorf("P99Ms = %v, want > 0", result.P99Ms)
	}
}

func TestBenchCollocated(t *testing.T) {
	result, err := benchCollocated(4, 8, 10)
	if err != nil {
		t.Fatalf("benchCollocated: %v", err)
	}
	if result.Mode != "collocated" {
		t.Errorf("Mode = %q, want %q", result.Mode, "collocated")
	}
	if result.ReqPerSec <= 0 {
		t.Errorf("ReqPerSec = %v, want > 0", result.ReqPerSec)
	}
	if result.MeanTTFTMs <= 0 {
		t.Errorf("MeanTTFTMs = %v, want > 0", result.MeanTTFTMs)
	}
	if result.P99Ms <= 0 {
		t.Errorf("P99Ms = %v, want > 0", result.P99Ms)
	}
}

func TestDisaggFasterThanCollocated(t *testing.T) {
	disaggResult, err := benchDisaggregated(8, 16, 10)
	if err != nil {
		t.Fatalf("benchDisaggregated: %v", err)
	}
	collocResult, err := benchCollocated(8, 16, 10)
	if err != nil {
		t.Fatalf("benchCollocated: %v", err)
	}

	speedup := disaggResult.ReqPerSec / collocResult.ReqPerSec
	t.Logf("disaggregated: %.2f req/s, collocated: %.2f req/s, speedup: %.2fx",
		disaggResult.ReqPerSec, collocResult.ReqPerSec, speedup)

	if speedup < 1.0 {
		t.Errorf("disaggregated should be faster than collocated, got speedup %.2fx", speedup)
	}
}

func TestBinaryBuilds(t *testing.T) {
	cmd := exec.Command("go", "build", "-o", os.DevNull, ".")
	// Build from the package directory (where this test file lives).
	cmd.Dir = "."
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("go build failed: %v\n%s", err, stderr.String())
	}
}

func TestBuildResult(t *testing.T) {
	latencies := []time.Duration{
		10 * time.Millisecond,
		20 * time.Millisecond,
		15 * time.Millisecond,
		25 * time.Millisecond,
	}
	r := buildResult("test", 2, 4, 10, latencies, "abc123")

	if r.Mode != "test" {
		t.Errorf("Mode = %q, want %q", r.Mode, "test")
	}
	if r.Concurrent != 2 {
		t.Errorf("Concurrent = %d, want 2", r.Concurrent)
	}
	if r.Requests != 4 {
		t.Errorf("Requests = %d, want 4", r.Requests)
	}
	if r.ReqPerSec <= 0 {
		t.Errorf("ReqPerSec = %v, want > 0", r.ReqPerSec)
	}
	if r.MeanTTFTMs <= 0 {
		t.Errorf("MeanTTFTMs = %v, want > 0", r.MeanTTFTMs)
	}
	if r.P99Ms <= 0 {
		t.Errorf("P99Ms = %v, want > 0", r.P99Ms)
	}
	if r.Commit != "abc123" {
		t.Errorf("Commit = %q, want %q", r.Commit, "abc123")
	}
}
