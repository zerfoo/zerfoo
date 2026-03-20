package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func TestTabularWorker_RunTrial(t *testing.T) {
	// Create a synthetic CSV with 3 features and a label column (0, 1, or 2).
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "f1,f2,f3,label\n"
	// Generate enough rows for training with validation split.
	for i := 0; i < 30; i++ {
		label := i % 3
		f1 := float64(label) * 0.5
		f2 := float64(label) * 0.3
		f3 := float64(label) * 0.7
		content += fmt.Sprintf("%.1f,%.1f,%.1f,%d\n", f1, f2, f3, label)
	}
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatalf("failed to write test CSV: %v", err)
	}

	w, err := newTabularWorker(csvFile, "accuracy")
	if err != nil {
		t.Fatalf("newTabularWorker failed: %v", err)
	}

	cfg := automl.Config{
		Params: map[string]float64{
			"lr":         0.01,
			"batch_size": 16,
		},
	}
	metric, err := w.RunTrial(cfg)
	if err != nil {
		t.Fatalf("RunTrial failed: %v", err)
	}
	// Score should be between 0 and 1.
	if metric.Score < 0 || metric.Score > 1 {
		t.Errorf("expected score in [0, 1], got %f", metric.Score)
	}
}

func TestTabularWorker_MissingFile(t *testing.T) {
	_, err := newTabularWorker("/nonexistent/data.csv", "accuracy")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestReadTabularCSV(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr bool
		rows    int
	}{
		{
			name:    "valid",
			content: "f1,f2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,2\n",
			rows:    3,
		},
		{
			name:    "header only",
			content: "f1,f2,label\n",
			wantErr: true,
		},
		{
			name:    "single column",
			content: "label\n0\n1\n",
			wantErr: true,
		},
		{
			name:    "bad feature value",
			content: "f1,f2,label\nbad,2.0,0\n",
			wantErr: true,
		},
		{
			name:    "bad label value",
			content: "f1,f2,label\n1.0,2.0,bad\n",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			csvFile := filepath.Join(dir, "data.csv")
			if err := os.WriteFile(csvFile, []byte(tt.content), 0600); err != nil {
				t.Fatalf("write: %v", err)
			}

			data, labels, err := readTabularCSV(csvFile)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(data) != tt.rows {
				t.Errorf("expected %d rows, got %d", tt.rows, len(data))
			}
			if len(labels) != tt.rows {
				t.Errorf("expected %d labels, got %d", tt.rows, len(labels))
			}
		})
	}
}

func TestAutoMLCommand_TabularModel(t *testing.T) {
	// Create a small CSV dataset.
	dir := t.TempDir()
	csvFile := filepath.Join(dir, "data.csv")
	content := "f1,f2,label\n"
	for i := 0; i < 30; i++ {
		label := i % 3
		content += fmt.Sprintf("%.1f,%.1f,%d\n", float64(label)*0.5, float64(label)*0.3, label)
	}
	if err := os.WriteFile(csvFile, []byte(content), 0600); err != nil {
		t.Fatalf("write CSV: %v", err)
	}

	var buf bytes.Buffer
	cmd := NewAutoMLCommand(&buf)
	err := cmd.Run(context.Background(), []string{
		"--model", "tabular",
		"--dataset", csvFile,
		"--trials", "2",
		"--metric", "accuracy",
		"--strategy", "random",
		"--patience", "0",
		"--seed", "42",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have 2 NDJSON lines.
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	ndjsonLines := 0
	for _, line := range lines {
		if strings.HasPrefix(line, "{") {
			ndjsonLines++
			var entry trialLog
			if err := json.Unmarshal([]byte(line), &entry); err != nil {
				t.Errorf("invalid NDJSON line: %v", err)
			}
			if entry.Error != "" {
				t.Errorf("trial %d had error: %s", entry.TrialID, entry.Error)
			}
		}
	}
	if ndjsonLines != 2 {
		t.Errorf("expected 2 NDJSON trial lines, got %d", ndjsonLines)
	}
}

// constantWorker always returns the same score, useful for testing early stopping.
type constantWorker struct {
	score float64
}

func (w *constantWorker) RunTrial(_ automl.Config) (automl.Metric, error) {
	return automl.Metric{Score: w.score}, nil
}
