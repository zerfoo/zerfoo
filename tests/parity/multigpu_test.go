//go:build cuda

package parity_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/registry"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"
)

// TestMultiGPU_DualDeviceInference loads the same model on two GPUs and verifies
// that greedy decode produces identical output on both devices.
func TestMultiGPU_DualDeviceInference(t *testing.T) {
	count, err := cuda.GetDeviceCount()
	if err != nil || count < 2 {
		t.Skip("requires at least 2 CUDA devices")
	}

	modelID := "gemma-3-1b-it"
	modelPath := findModelDir(t, modelID)

	reg := &testutil.DirRegistry{
		Models: map[string]*registry.ModelInfo{
			modelID: {ID: modelID, Path: modelPath},
		},
	}

	// Load on device 0.
	m0, err := inference.Load(modelID,
		inference.WithRegistry(reg),
		inference.WithDevice("cuda:0"),
	)
	if err != nil {
		t.Fatalf("load on cuda:0: %v", err)
	}
	defer m0.Close()

	// Load on device 1.
	m1, err := inference.Load(modelID,
		inference.WithRegistry(reg),
		inference.WithDevice("cuda:1"),
	)
	if err != nil {
		t.Fatalf("load on cuda:1: %v", err)
	}
	defer m1.Close()

	prompt := "The capital of France is"
	ctx := context.Background()

	out0, err := m0.Generate(ctx, prompt,
		inference.WithTemperature(0),
		inference.WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("generate on cuda:0: %v", err)
	}

	out1, err := m1.Generate(ctx, prompt,
		inference.WithTemperature(0),
		inference.WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("generate on cuda:1: %v", err)
	}

	if out0 != out1 {
		t.Errorf("device outputs differ:\n  cuda:0: %q\n  cuda:1: %q", out0, out1)
	}
}

// findModelDir returns the model directory path or skips the test.
func findModelDir(t *testing.T, modelID string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	candidates := []string{
		filepath.Join(home, ".cache", "zerfoo", "models", modelID),
		filepath.Join(home, ".zerfoo", "models", modelID),
	}
	for _, dir := range candidates {
		if _, err := os.Stat(filepath.Join(dir, "config.json")); err == nil {
			return dir
		}
	}
	t.Skipf("model %q not found in cache", modelID)
	return ""
}
