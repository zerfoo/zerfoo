package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/training/automl"
)

// fakeWorker returns a deterministic score based on the "lr" param.
type fakeWorker struct{}

func (w *fakeWorker) RunTrial(config automl.Config) (automl.Metric, error) {
	lr := config.Params["lr"]
	// Simulate a score that peaks near lr=0.001.
	score := 1.0 / (1.0 + (lr-0.001)*(lr-0.001)*1e6)
	return automl.Metric{Score: score}, nil
}

func TestAutoMLCommand_Name(t *testing.T) {
	cmd := NewAutoMLCommand(nil)
	if cmd.Name() != "automl" {
		t.Errorf("expected name 'automl', got %q", cmd.Name())
	}
}

func TestAutoMLCommand_MissingModel(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	err := cmd.Run(context.Background(), []string{"--dataset", "data.jsonl"})
	if err == nil || !strings.Contains(err.Error(), "--model is required") {
		t.Fatalf("expected --model required error, got %v", err)
	}
}

func TestAutoMLCommand_MissingDataset(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	err := cmd.Run(context.Background(), []string{"--model", "m.gguf"})
	if err == nil || !strings.Contains(err.Error(), "--dataset is required") {
		t.Fatalf("expected --dataset required error, got %v", err)
	}
}

func TestAutoMLCommand_UnknownFlag(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	err := cmd.Run(context.Background(), []string{"--model", "m.gguf", "--dataset", "d.jsonl", "--bogus"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Fatalf("expected unknown flag error, got %v", err)
	}
}

func TestAutoMLCommand_RunWithFakeWorker(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	cmd.workerFactory = func(cfg autoMLRunConfig) automl.Worker {
		return &fakeWorker{}
	}

	outFile := filepath.Join(t.TempDir(), "best.json")

	err := cmd.Run(context.Background(), []string{
		"--model", "test.gguf",
		"--dataset", "test.jsonl",
		"--trials", "10",
		"--metric", "sharpe",
		"--output", outFile,
		"--seed", "123",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify NDJSON output: should have 10 lines of trial logs + 1 "Best config saved" line.
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	ndjsonLines := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "{") {
			ndjsonLines++
			var entry trialLog
			if err := json.Unmarshal([]byte(line), &entry); err != nil {
				t.Errorf("invalid NDJSON line: %v", err)
			}
		}
	}
	if ndjsonLines != 10 {
		t.Errorf("expected 10 NDJSON trial lines, got %d", ndjsonLines)
	}

	// Verify best config file was written.
	data, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("failed to read output file: %v", err)
	}
	var best bestConfigOutput
	if err := json.Unmarshal(data, &best); err != nil {
		t.Fatalf("failed to unmarshal best config: %v", err)
	}
	if best.Model != "test.gguf" {
		t.Errorf("expected model 'test.gguf', got %q", best.Model)
	}
	if best.Metric != "sharpe" {
		t.Errorf("expected metric 'sharpe', got %q", best.Metric)
	}
	if best.Score <= 0 {
		t.Errorf("expected positive score, got %f", best.Score)
	}
}

func TestAutoMLCommand_RandomStrategy(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	cmd.workerFactory = func(cfg autoMLRunConfig) automl.Worker {
		return &fakeWorker{}
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "test.gguf",
		"--dataset", "test.jsonl",
		"--trials", "5",
		"--strategy", "random",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	ndjsonLines := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "{") {
			ndjsonLines++
		}
	}
	if ndjsonLines != 5 {
		t.Errorf("expected 5 NDJSON trial lines, got %d", ndjsonLines)
	}
}

func TestAutoMLCommand_EarlyStopping(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	cmd.workerFactory = func(cfg autoMLRunConfig) automl.Worker {
		return &constantWorker{score: 0.5}
	}

	err := cmd.Run(context.Background(), []string{
		"--model", "test.gguf",
		"--dataset", "test.jsonl",
		"--trials", "100",
		"--patience", "3",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// With patience=3 and constant scores, should stop after 4 trials
	// (1 initial best + 3 without improvement).
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	ndjsonLines := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "{") {
			ndjsonLines++
		}
	}
	if ndjsonLines >= 100 {
		t.Errorf("expected early stopping before 100 trials, got %d", ndjsonLines)
	}
}

func TestAutoMLCommand_FlagEqualsForm(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	cmd.workerFactory = func(cfg autoMLRunConfig) automl.Worker {
		return &fakeWorker{}
	}

	err := cmd.Run(context.Background(), []string{
		"--model=test.gguf",
		"--dataset=test.jsonl",
		"--trials=3",
		"--metric=sharpe",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAutoMLCommand_Usage(t *testing.T) {
	cmd := NewAutoMLCommand(nil)
	usage := cmd.Usage()
	if !strings.Contains(usage, "--model") {
		t.Error("usage should mention --model")
	}
	if !strings.Contains(usage, "--dataset") {
		t.Error("usage should mention --dataset")
	}
}

func TestAutoMLCommand_Examples(t *testing.T) {
	cmd := NewAutoMLCommand(nil)
	examples := cmd.Examples()
	if len(examples) == 0 {
		t.Error("expected at least one example")
	}
}

// constantWorker always returns the same score, useful for testing early stopping.
type constantWorker struct {
	score float64
}

func (w *constantWorker) RunTrial(_ automl.Config) (automl.Metric, error) {
	return automl.Metric{Score: w.score}, nil
}
