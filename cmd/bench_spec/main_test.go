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
		want    SpecBenchConfig
		wantErr bool
	}{
		{
			name: "all flags",
			args: []string{
				"--model-target", "/path/to/27B.gguf",
				"--model-draft", "/path/to/1B.gguf",
				"--backend", "cuda",
				"--tokens", "100",
				"--prompts", "5",
				"--warmup", "3",
				"--draft-len", "6",
				"--output", "results.json",
			},
			want: SpecBenchConfig{
				ModelTarget: "/path/to/27B.gguf",
				ModelDraft:  "/path/to/1B.gguf",
				Backend:     "cuda",
				Tokens:      100,
				Prompts:     5,
				Warmup:      3,
				DraftLen:    6,
				Output:      "results.json",
			},
		},
		{
			name: "defaults",
			args: []string{
				"--model-target", "/path/to/target.gguf",
				"--model-draft", "/path/to/draft.gguf",
			},
			want: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				ModelDraft:  "/path/to/draft.gguf",
				Backend:     "cpu",
				Tokens:      200,
				Prompts:     10,
				Warmup:      2,
				DraftLen:    4,
				Output:      "bench_spec_results.json",
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
		cfg     SpecBenchConfig
		wantErr bool
	}{
		{
			name: "valid",
			cfg: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				ModelDraft:  "/path/to/draft.gguf",
				Tokens:      200,
				Prompts:     10,
				DraftLen:    4,
			},
		},
		{
			name: "missing target",
			cfg: SpecBenchConfig{
				ModelDraft: "/path/to/draft.gguf",
				Tokens:     200,
				Prompts:    10,
				DraftLen:   4,
			},
			wantErr: true,
		},
		{
			name: "missing draft",
			cfg: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				Tokens:      200,
				Prompts:     10,
				DraftLen:    4,
			},
			wantErr: true,
		},
		{
			name: "zero tokens",
			cfg: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				ModelDraft:  "/path/to/draft.gguf",
				Tokens:      0,
				Prompts:     10,
				DraftLen:    4,
			},
			wantErr: true,
		},
		{
			name: "zero prompts",
			cfg: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				ModelDraft:  "/path/to/draft.gguf",
				Tokens:      200,
				Prompts:     0,
				DraftLen:    4,
			},
			wantErr: true,
		},
		{
			name: "zero draft-len",
			cfg: SpecBenchConfig{
				ModelTarget: "/path/to/target.gguf",
				ModelDraft:  "/path/to/draft.gguf",
				Tokens:      200,
				Prompts:     10,
				DraftLen:    0,
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
		{"five", 5},
		{"ten", 10},
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

// mockRunner simulates model generation for testing runMode.
type mockRunner struct {
	tokensPerCall int
	durationMs    float64
	callCount     int
}

func (r *mockRunner) Generate(_ context.Context, _ string, maxTokens int) (int, float64, error) {
	r.callCount++
	tc := r.tokensPerCall
	if tc > maxTokens {
		tc = maxTokens
	}
	return tc, r.durationMs, nil
}

func TestRunMode(t *testing.T) {
	runner := &mockRunner{
		tokensPerCall: 50,
		durationMs:    500.0, // 500ms per call = 100 tok/s
	}
	prompts := []string{"prompt1", "prompt2", "prompt3"}

	result, err := runMode(context.Background(), runner, prompts, 50, 1, "test")
	if err != nil {
		t.Fatalf("runMode: %v", err)
	}

	if result.Mode != "test" {
		t.Errorf("Mode = %q, want %q", result.Mode, "test")
	}
	// 1 warmup + 3 prompts = 4 calls.
	if runner.callCount != 4 {
		t.Errorf("callCount = %d, want 4", runner.callCount)
	}
	// 3 prompts * 50 tokens = 150 total tokens.
	if result.TotalTokens != 150 {
		t.Errorf("TotalTokens = %d, want 150", result.TotalTokens)
	}
	// 150 tokens / 1.5s = 100 tok/s.
	expectedTokS := 100.0
	if result.TokPerSec < expectedTokS-1 || result.TokPerSec > expectedTokS+1 {
		t.Errorf("TokPerSec = %.2f, want ~%.2f", result.TokPerSec, expectedTokS)
	}
}

func TestWriteJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_results.json")

	report := SpecBenchReport{
		ModelTarget:  "/path/to/target.gguf",
		ModelDraft:   "/path/to/draft.gguf",
		Backend:      "cuda",
		TokensPerRun: 200,
		NumPrompts:   10,
		DraftLen:     4,
		Standalone: SpecBenchResult{
			Mode:           "standalone",
			TotalTokens:    2000,
			TotalDurationS: 100.0,
			TokPerSec:      20.0,
		},
		Speculative: SpecBenchResult{
			Mode:           "speculative",
			TotalTokens:    2000,
			TotalDurationS: 40.0,
			TokPerSec:      50.0,
		},
		AcceptanceRate: 0.75,
		Speedup:        2.5,
		Commit:         "abc1234",
		Timestamp:      "2026-03-18T00:00:00Z",
	}

	if err := writeJSON(path, report); err != nil {
		t.Fatalf("writeJSON: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	var got SpecBenchReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}

	if got.Speedup != 2.5 {
		t.Errorf("Speedup = %v, want 2.5", got.Speedup)
	}
	if got.AcceptanceRate != 0.75 {
		t.Errorf("AcceptanceRate = %v, want 0.75", got.AcceptanceRate)
	}
	if got.Standalone.TokPerSec != 20.0 {
		t.Errorf("Standalone.TokPerSec = %v, want 20.0", got.Standalone.TokPerSec)
	}
	if got.Speculative.TokPerSec != 50.0 {
		t.Errorf("Speculative.TokPerSec = %v, want 50.0", got.Speculative.TokPerSec)
	}
}

func TestPrintReport(t *testing.T) {
	// Verify printReport doesn't panic. Output goes to stdout.
	report := SpecBenchReport{
		ModelTarget:  "target.gguf",
		ModelDraft:   "draft.gguf",
		Backend:      "cpu",
		TokensPerRun: 100,
		NumPrompts:   5,
		DraftLen:     4,
		Standalone: SpecBenchResult{
			Mode:      "standalone",
			TokPerSec: 10.0,
		},
		Speculative: SpecBenchResult{
			Mode:      "speculative",
			TokPerSec: 25.0,
		},
		AcceptanceRate: 0.80,
		Speedup:        2.5,
	}
	// Should not panic.
	printReport(report)
}
