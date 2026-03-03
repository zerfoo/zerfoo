package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseBenchmarks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bench.txt")

	content := `goos: darwin
goarch: arm64
pkg: github.com/zerfoo/zerfoo/compute
BenchmarkMatMul-10       1000       500000 ns/op       1024 B/op       4 allocs/op
BenchmarkAdd-10         10000        15000 ns/op        256 B/op       2 allocs/op
BenchmarkSoftmax-10      5000        80000 ns/op        512 B/op       3 allocs/op
PASS
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}

	tests := []struct {
		name    string
		wantNS  float64
		epsilon float64
	}{
		{"BenchmarkMatMul-10", 500000, 1},
		{"BenchmarkAdd-10", 15000, 1},
		{"BenchmarkSoftmax-10", 80000, 1},
	}

	for _, tt := range tests {
		got, ok := results[tt.name]
		if !ok {
			t.Errorf("expected benchmark %s in results", tt.name)
			continue
		}
		if !approxEqual(got, tt.wantNS, tt.epsilon) {
			t.Errorf("%s = %.0f ns/op, want %.0f", tt.name, got, tt.wantNS)
		}
	}
}

func TestParseBenchmarks_MultipleRuns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.txt")

	content := `BenchmarkMatMul-10       1000       500000 ns/op
BenchmarkMatMul-10       1000       510000 ns/op
BenchmarkMatMul-10       1000       490000 ns/op
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}

	// Median of [490000, 500000, 510000] = 500000
	got := results["BenchmarkMatMul-10"]
	if !approxEqual(got, 500000, 1) {
		t.Errorf("median = %.0f, want 500000", got)
	}
}

func TestParseBenchmarks_FileNotFound(t *testing.T) {
	_, err := parseBenchmarks("/nonexistent/bench.txt")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestParseBenchmarks_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(path, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseBenchmarks(path)
	if err != nil {
		t.Fatalf("parseBenchmarks: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results, got %d", len(results))
	}
}

func TestMedian(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   float64
	}{
		{"empty", nil, 0},
		{"single", []float64{42}, 42},
		{"odd", []float64{3, 1, 2}, 2},
		{"even", []float64{4, 1, 3, 2}, 2.5},
	}

	for _, tt := range tests {
		got := median(tt.values)
		if !approxEqual(got, tt.want, 0.01) {
			t.Errorf("median(%s) = %f, want %f", tt.name, got, tt.want)
		}
	}
}
