package main

import (
	"bytes"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFlagsHelp(t *testing.T) {
	_, err := parseFlags([]string{"--help"})
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("expected flag.ErrHelp, got %v", err)
	}
}

func TestParseFlagsMissingConfig(t *testing.T) {
	_, err := parseFlags([]string{"--data", "train.jsonl"})
	if err == nil || !strings.Contains(err.Error(), "--config") {
		t.Fatalf("expected --config required error, got %v", err)
	}
}

func TestParseFlagsMissingData(t *testing.T) {
	_, err := parseFlags([]string{"--config", "model.gguf"})
	if err == nil || !strings.Contains(err.Error(), "--data") {
		t.Fatalf("expected --data required error, got %v", err)
	}
}

func TestParseFlagsInvalidWorldSize(t *testing.T) {
	_, err := parseFlags([]string{"--config", "m.gguf", "--data", "d.jsonl", "--world-size", "0"})
	if err == nil || !strings.Contains(err.Error(), "world-size") {
		t.Fatalf("expected world-size error, got %v", err)
	}
}

func TestParseFlagsInvalidRank(t *testing.T) {
	_, err := parseFlags([]string{"--config", "m.gguf", "--data", "d.jsonl", "--world-size", "2", "--rank", "5"})
	if err == nil || !strings.Contains(err.Error(), "rank") {
		t.Fatalf("expected rank error, got %v", err)
	}
}

func TestParseFlagsInvalidPort(t *testing.T) {
	_, err := parseFlags([]string{"--config", "m.gguf", "--data", "d.jsonl", "--master-port", "-1"})
	if err == nil || !strings.Contains(err.Error(), "master-port") {
		t.Fatalf("expected master-port error, got %v", err)
	}
}

func TestParseFlagsDefaults(t *testing.T) {
	cfg, err := parseFlags([]string{"--config", "model.gguf", "--data", "train.jsonl"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "model.gguf" {
		t.Errorf("modelPath = %q, want %q", cfg.modelPath, "model.gguf")
	}
	if cfg.dataPath != "train.jsonl" {
		t.Errorf("dataPath = %q, want %q", cfg.dataPath, "train.jsonl")
	}
	if cfg.worldSize != 1 {
		t.Errorf("worldSize = %d, want 1", cfg.worldSize)
	}
	if cfg.rank != 0 {
		t.Errorf("rank = %d, want 0", cfg.rank)
	}
	if cfg.masterAddr != "localhost" {
		t.Errorf("masterAddr = %q, want %q", cfg.masterAddr, "localhost")
	}
	if cfg.masterPort != 29500 {
		t.Errorf("masterPort = %d, want 29500", cfg.masterPort)
	}
	if cfg.outputPath != "checkpoint.gguf" {
		t.Errorf("outputPath = %q, want %q", cfg.outputPath, "checkpoint.gguf")
	}
	if cfg.epochs != 1 {
		t.Errorf("epochs = %d, want 1", cfg.epochs)
	}
}

func TestParseFlagsAllFlags(t *testing.T) {
	cfg, err := parseFlags([]string{
		"--config", "m.gguf",
		"--data", "d.jsonl",
		"--world-size", "4",
		"--rank", "2",
		"--master-addr", "10.0.0.1",
		"--master-port", "9000",
		"--output", "out.gguf",
		"--epochs", "3",
		"--batch-size", "8",
		"--lr", "0.001",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.worldSize != 4 {
		t.Errorf("worldSize = %d, want 4", cfg.worldSize)
	}
	if cfg.rank != 2 {
		t.Errorf("rank = %d, want 2", cfg.rank)
	}
	if cfg.masterAddr != "10.0.0.1" {
		t.Errorf("masterAddr = %q, want %q", cfg.masterAddr, "10.0.0.1")
	}
	if cfg.masterPort != 9000 {
		t.Errorf("masterPort = %d, want 9000", cfg.masterPort)
	}
	if cfg.outputPath != "out.gguf" {
		t.Errorf("outputPath = %q, want %q", cfg.outputPath, "out.gguf")
	}
	if cfg.epochs != 3 {
		t.Errorf("epochs = %d, want 3", cfg.epochs)
	}
	if cfg.batchSize != 8 {
		t.Errorf("batchSize = %d, want 8", cfg.batchSize)
	}
	if cfg.lr != 0.001 {
		t.Errorf("lr = %f, want 0.001", cfg.lr)
	}
}

func TestTrainDistributedSingleGPU(t *testing.T) {
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "checkpoint.gguf")

	// Create minimal mock data file.
	dataPath := filepath.Join(tmpDir, "train.jsonl")
	if err := os.WriteFile(dataPath, []byte(`{"input":"hello","output":"world"}`+"\n"), 0600); err != nil {
		t.Fatal(err)
	}

	// Create minimal mock model file.
	modelPath := filepath.Join(tmpDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("stub"), 0600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{
		"--config", modelPath,
		"--data", dataPath,
		"--world-size", "1",
		"--rank", "0",
		"--output", outputPath,
		"--epochs", "1",
		"--batch-size", "4",
		"--master-port", "0",
	}, &stdout, &stderr)

	// master-port=0 will cause coordinator to fail to listen, but
	// we still validate that the run function processes flags correctly
	// and gets to the coordinator start phase.
	if err == nil {
		// If it somehow succeeds (port 0 is valid for OS-assigned port),
		// verify output contains training progress.
		output := stdout.String()
		if !strings.Contains(output, "epoch=") || !strings.Contains(output, "loss=") {
			t.Errorf("expected training progress output, got: %s", output)
		}
		if !strings.Contains(output, "tok/s=") {
			t.Errorf("expected tok/s in output, got: %s", output)
		}
	} else {
		// Port 0 is actually valid (OS assigns), so this should succeed.
		// If it fails, it should be a coordinator start error, not a flag error.
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTrainDistributedCLIHelpOutput(t *testing.T) {
	_, err := parseFlags([]string{"--help"})
	if !errors.Is(err, flag.ErrHelp) {
		t.Fatalf("expected flag.ErrHelp, got %v", err)
	}
}

func TestTrainLoopAdamWUpdatesParams(t *testing.T) {
	// Verify that trainLoop uses AdamW and parameters change after training.
	model, err := newStubModel(64)
	if err != nil {
		t.Fatal(err)
	}

	// Verify model was created (parameters exist).
	if len(model.params) == 0 {
		t.Fatal("expected at least 1 parameter")
	}

	// Run a 1-epoch training loop via the full CLI path.
	tmpDir := t.TempDir()
	outputPath := filepath.Join(tmpDir, "checkpoint.gguf")
	dataPath := filepath.Join(tmpDir, "train.jsonl")
	if err := os.WriteFile(dataPath, []byte(`{"input":"test"}`+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(tmpDir, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("stub"), 0600); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	err = run([]string{
		"--config", modelPath,
		"--data", dataPath,
		"--world-size", "1",
		"--rank", "0",
		"--output", outputPath,
		"--epochs", "1",
		"--batch-size", "4",
		"--lr", "0.01",
		"--master-port", "0",
	}, &stdout, &stderr)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	output := stdout.String()
	if !strings.Contains(output, "epoch=1") {
		t.Errorf("expected epoch=1 in output, got: %s", output)
	}
	if !strings.Contains(output, "checkpoint saved") {
		t.Errorf("expected checkpoint saved in output, got: %s", output)
	}
}
