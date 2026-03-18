package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestFinetuneSmokeTest(t *testing.T) {
	dir := t.TempDir()

	// Create a small synthetic JSONL dataset.
	datasetPath := filepath.Join(dir, "train.jsonl")
	lines := []string{
		`{"input": "What is Go?", "output": "A programming language."}`,
		`{"input": "What is LoRA?", "output": "Low-rank adaptation."}`,
		`{"input": "What is QLoRA?", "output": "Quantized LoRA."}`,
		`{"input": "What is GGUF?", "output": "A model format."}`,
	}
	f, err := os.Create(datasetPath)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	for _, line := range lines {
		if _, err := f.WriteString(line + "\n"); err != nil {
			t.Fatalf("write line: %v", err)
		}
	}
	f.Close()

	adapterPath := filepath.Join(dir, "adapter.gguf")

	// Set flags for the run() function.
	os.Args = []string{
		"finetune",
		"--dataset", datasetPath,
		"--rank", "4",
		"--alpha", "8",
		"--epochs", "2",
		"--batch-size", "2",
		"--lr", "0.001",
		"--output", adapterPath,
	}

	// Reset flags for re-parsing.
	resetFlags()

	if err := run(); err != nil {
		t.Fatalf("run() failed: %v", err)
	}

	// Verify adapter file was created.
	info, err := os.Stat(adapterPath)
	if err != nil {
		t.Fatalf("adapter file not created: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("adapter file is empty")
	}
	t.Logf("adapter file size: %d bytes", info.Size())
}

func TestFinetuneNoDataset(t *testing.T) {
	os.Args = []string{"finetune"}
	resetFlags()

	err := run()
	if err == nil {
		t.Fatal("expected error when no dataset provided")
	}
}

func TestFinetuneModelNotFound(t *testing.T) {
	dir := t.TempDir()
	datasetPath := filepath.Join(dir, "train.jsonl")
	if err := os.WriteFile(datasetPath, []byte(`{"input":"a","output":"b"}`+"\n"), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}

	os.Args = []string{
		"finetune",
		"--model", filepath.Join(dir, "nonexistent.gguf"),
		"--dataset", datasetPath,
	}
	resetFlags()

	// Should exit gracefully (no error) when model not found.
	if err := run(); err != nil {
		t.Fatalf("expected graceful exit, got: %v", err)
	}
}

func TestLoadDatasetEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.jsonl")
	if err := os.WriteFile(path, []byte{}, 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	_, err := loadDataset(path)
	if err == nil {
		t.Fatal("expected error for empty dataset")
	}
}

func TestLoadDatasetInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.jsonl")
	if err := os.WriteFile(path, []byte("not json\n"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	_, err := loadDataset(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// resetFlags resets the flag package state so run() can re-parse.
func resetFlags() {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
}
