package parity_test

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// TestZerfooVsOllamaThroughput compares Zerfoo decode throughput against Ollama
// on the same model (Gemma 3 1B Q4_K_M). It asserts Zerfoo >= Ollama * 1.3.
//
// Prerequisites (skipped gracefully if missing):
//   - GEMMA3_1B_GGUF env var pointing to the Q4_K_M GGUF file
//   - ollama binary in PATH with gemma3:1b pulled
func TestZerfooVsOllamaThroughput(t *testing.T) {
	ggufPath := os.Getenv("GEMMA3_1B_GGUF")
	if ggufPath == "" {
		t.Skip("GEMMA3_1B_GGUF not set; skipping throughput parity test")
	}
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skipf("GEMMA3_1B_GGUF file not found: %v", err)
	}

	ollamaPath, err := exec.LookPath("ollama")
	if err != nil {
		t.Skip("ollama not in PATH; skipping throughput parity test")
	}
	_ = ollamaPath

	const (
		prompt    = "Explain the theory of relativity in simple terms."
		maxTokens = 64
		margin    = 1.3 // Zerfoo must be at least 30% faster
	)

	// --- Zerfoo benchmark ---
	zerfooTPS := benchmarkZerfoo(t, ggufPath, prompt, maxTokens)

	// --- Ollama benchmark ---
	ollamaTPS := benchmarkOllama(t, prompt, maxTokens)

	t.Logf("Zerfoo: %.2f tok/s", zerfooTPS)
	t.Logf("Ollama: %.2f tok/s", ollamaTPS)
	t.Logf("Ratio:  %.2fx", zerfooTPS/ollamaTPS)

	threshold := ollamaTPS * margin
	if zerfooTPS < threshold {
		t.Errorf("throughput regression: Zerfoo %.2f tok/s < %.2f tok/s (Ollama %.2f * %.1f)",
			zerfooTPS, threshold, ollamaTPS, margin)
	}
}

// benchmarkZerfoo loads a GGUF model and measures decode throughput.
func benchmarkZerfoo(t *testing.T, ggufPath, prompt string, maxTokens int) float64 {
	t.Helper()
	layerreg.RegisterAll()

	mdl, err := inference.LoadFile(ggufPath)
	if err != nil {
		t.Fatalf("inference.LoadFile failed: %v", err)
	}

	// Warmup run.
	ctx := context.Background()
	_, err = mdl.Generate(ctx, prompt, inference.WithTemperature(0), inference.WithMaxTokens(4))
	if err != nil {
		t.Fatalf("warmup Generate failed: %v", err)
	}

	// Timed run: count tokens via streaming.
	var tokenCount int
	start := time.Now()
	err = mdl.GenerateStream(ctx, prompt,
		generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				tokenCount++
			}
			return nil
		}),
		inference.WithTemperature(0),
		inference.WithMaxTokens(maxTokens),
	)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("GenerateStream failed: %v", err)
	}
	if tokenCount == 0 {
		t.Fatal("Zerfoo generated zero tokens")
	}

	return float64(tokenCount) / elapsed.Seconds()
}

// benchmarkOllama runs Ollama on gemma3:1b and measures decode throughput.
// It parses the eval rate from Ollama's verbose output.
func benchmarkOllama(t *testing.T, prompt string, maxTokens int) float64 {
	t.Helper()

	ollamaModel := os.Getenv("OLLAMA_PARITY_MODEL")
	if ollamaModel == "" {
		ollamaModel = "gemma3:1b"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ollama", "run", ollamaModel,
		"--verbose",
		"--nowordwrap",
		prompt,
	)
	cmd.Env = append(os.Environ(), "OLLAMA_NUM_PREDICT="+strconv.Itoa(maxTokens))

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		t.Fatalf("ollama run failed: %v\nstderr: %s", err, stderr.String())
	}

	// Parse eval rate from ollama --verbose output.
	// Format: "eval rate:       204.32 tokens/s"
	tps := parseOllamaEvalRate(t, stderr.String())
	if tps <= 0 {
		t.Fatalf("could not parse Ollama eval rate from stderr:\n%s", stderr.String())
	}

	return tps
}

// parseOllamaEvalRate extracts the eval rate (tokens/s) from Ollama verbose output.
func parseOllamaEvalRate(t *testing.T, output string) float64 {
	t.Helper()
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "eval rate:") {
			continue
		}
		// "eval rate:       204.32 tokens/s"
		parts := strings.Fields(line)
		for i, p := range parts {
			if p == "tokens/s" && i > 0 {
				v, err := strconv.ParseFloat(parts[i-1], 64)
				if err != nil {
					t.Logf("failed to parse eval rate value %q: %v", parts[i-1], err)
					return 0
				}
				return v
			}
		}
	}
	return 0
}
