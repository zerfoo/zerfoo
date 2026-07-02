package main

import (
	"os"
	"strings"
	"testing"
)

func TestCompare_Regression(t *testing.T) {
	prev := map[string]BenchResult{
		"matmul_ns_op": {Metric: "matmul_ns_op", Value: 1000, Unit: "ns/op"},
	}
	curr := map[string]BenchResult{
		"matmul_ns_op": {Metric: "matmul_ns_op", Value: 1100, Unit: "ns/op"},
	}

	report, regressed := compare(prev, curr)
	if !regressed {
		t.Error("expected regression for 10% increase in timing metric")
	}
	if !strings.Contains(report, "REGRESSION") {
		t.Errorf("report should contain REGRESSION, got:\n%s", report)
	}
}

func TestCompare_NoRegression(t *testing.T) {
	prev := map[string]BenchResult{
		"matmul_ns_op": {Metric: "matmul_ns_op", Value: 1000, Unit: "ns/op"},
	}
	curr := map[string]BenchResult{
		"matmul_ns_op": {Metric: "matmul_ns_op", Value: 1040, Unit: "ns/op"},
	}

	_, regressed := compare(prev, curr)
	if regressed {
		t.Error("expected no regression for <5% change")
	}
}

func TestCompare_MissingPrevious(t *testing.T) {
	curr := map[string]BenchResult{
		"matmul_ns_op": {Metric: "matmul_ns_op", Value: 1000, Unit: "ns/op"},
	}

	report, regressed := compare(nil, curr)
	if regressed {
		t.Error("expected no regression when previous is empty")
	}
	if !strings.Contains(report, "No previous baseline") {
		t.Errorf("expected 'No previous baseline' message, got: %s", report)
	}
}

func TestCompare_NewMetric(t *testing.T) {
	prev := map[string]BenchResult{
		"old_metric": {Metric: "old_metric", Value: 100, Unit: "ns/op"},
	}
	curr := map[string]BenchResult{
		"old_metric": {Metric: "old_metric", Value: 100, Unit: "ns/op"},
		"new_metric": {Metric: "new_metric", Value: 200, Unit: "tok/s"},
	}

	report, regressed := compare(prev, curr)
	if regressed {
		t.Error("new metric should not trigger regression")
	}
	if !strings.Contains(report, "NEW") {
		t.Errorf("report should contain NEW for new metric, got:\n%s", report)
	}
}

func TestCompare_FormatMarkdown(t *testing.T) {
	prev := map[string]BenchResult{
		"throughput_tok_s": {Metric: "throughput_tok_s", Value: 10.0, Unit: "tok/s"},
		"latency_ns_op":   {Metric: "latency_ns_op", Value: 5000, Unit: "ns/op"},
	}
	curr := map[string]BenchResult{
		"throughput_tok_s": {Metric: "throughput_tok_s", Value: 10.5, Unit: "tok/s"},
		"latency_ns_op":   {Metric: "latency_ns_op", Value: 5100, Unit: "ns/op"},
	}

	report, _ := compare(prev, curr)

	if !strings.Contains(report, "| Metric | Previous | Current | Change | Status |") {
		t.Errorf("missing markdown header in report:\n%s", report)
	}
	if !strings.Contains(report, "|--------|") {
		t.Errorf("missing markdown separator in report:\n%s", report)
	}
	if !strings.Contains(report, "throughput_tok_s") {
		t.Errorf("missing metric name in report:\n%s", report)
	}
}

func TestCompare_ThroughputRegression(t *testing.T) {
	prev := map[string]BenchResult{
		"inference_tok_s": {Metric: "inference_tok_s", Value: 10.0, Unit: "tok/s"},
	}
	curr := map[string]BenchResult{
		"inference_tok_s": {Metric: "inference_tok_s", Value: 8.0, Unit: "tok/s"},
	}

	_, regressed := compare(prev, curr)
	if !regressed {
		t.Error("expected regression for 20% decrease in throughput metric")
	}
}

func TestCompare_ThroughputImproved(t *testing.T) {
	prev := map[string]BenchResult{
		"inference_tok_s": {Metric: "inference_tok_s", Value: 10.0, Unit: "tok/s"},
	}
	curr := map[string]BenchResult{
		"inference_tok_s": {Metric: "inference_tok_s", Value: 12.0, Unit: "tok/s"},
	}

	report, regressed := compare(prev, curr)
	if regressed {
		t.Error("throughput increase should not be regression")
	}
	if !strings.Contains(report, "IMPROVED") {
		t.Errorf("report should show IMPROVED for throughput increase:\n%s", report)
	}
}

func TestReadResults(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    int
	}{
		{
			name:    "valid ndjson",
			content: "{\"metric\":\"a\",\"value\":1.0}\n{\"metric\":\"b\",\"value\":2.0}\n",
			want:    2,
		},
		{
			name:    "empty file",
			content: "",
			want:    0,
		},
		{
			name:    "blank lines",
			content: "\n\n{\"metric\":\"a\",\"value\":1.0}\n\n",
			want:    1,
		},
		{
			name:    "invalid json lines skipped",
			content: "not json\n{\"metric\":\"a\",\"value\":1.0}\n",
			want:    1,
		},
		{
			name:    "missing metric field skipped",
			content: "{\"value\":1.0}\n{\"metric\":\"a\",\"value\":2.0}\n",
			want:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := dir + "/test.json"
			if err := os.WriteFile(path, []byte(tt.content), 0o600); err != nil {
				t.Fatal(err)
			}

			results, err := readResults(path)
			if err != nil {
				t.Fatalf("readResults: %v", err)
			}
			if len(results) != tt.want {
				t.Errorf("got %d results, want %d", len(results), tt.want)
			}
		})
	}
}

func TestReadResults_NonexistentFile(t *testing.T) {
	results, err := readResults("/nonexistent/file.json")
	if err != nil {
		t.Fatalf("expected nil error for nonexistent file, got: %v", err)
	}
	if results != nil {
		t.Errorf("expected nil results for nonexistent file, got: %v", results)
	}
}

func TestIsRegressionChange(t *testing.T) {
	tests := []struct {
		name   string
		metric string
		unit   string
		change float64
		want   bool
	}{
		{"timing increase is regression", "matmul", "ns/op", 10.0, true},
		{"timing decrease is not regression", "matmul", "ns/op", -10.0, false},
		{"throughput decrease is regression", "tps", "tok/s", -10.0, true},
		{"throughput increase is not regression", "tps", "tok/s", 10.0, false},
		{"gflops decrease is regression", "cuda_q4_gemm_gflops", "GFLOPS", -10.0, true},
		{"default increase is regression", "unknown", "", 10.0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRegressionChange(tt.metric, tt.unit, tt.change)
			if got != tt.want {
				t.Errorf("isRegressionChange(%q, %q, %.1f) = %v, want %v",
					tt.metric, tt.unit, tt.change, got, tt.want)
			}
		})
	}
}
