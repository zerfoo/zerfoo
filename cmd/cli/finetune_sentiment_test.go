package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestFineTuneSentimentCommand_Name(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	if got := cmd.Name(); got != "finetune-sentiment" {
		t.Errorf("Name() = %q, want %q", got, "finetune-sentiment")
	}
}

func TestFineTuneSentimentCommand_Description(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	if got := cmd.Description(); got == "" {
		t.Error("Description() should not be empty")
	}
}

func TestFineTuneSentimentCommand_Usage(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	usage := cmd.Usage()
	if usage == "" {
		t.Fatal("Usage() should not be empty")
	}
	for _, flag := range []string{"--model", "--data", "--output", "--epochs", "--lr", "--batch-size", "--lora-rank", "--val-split", "--device"} {
		if !strings.Contains(usage, flag) {
			t.Errorf("Usage() missing flag %s", flag)
		}
	}
}

func TestFineTuneSentimentCommand_Examples(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	if got := cmd.Examples(); len(got) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestFineTuneSentimentCommand_Interface(t *testing.T) {
	var _ Command = (*FineTuneSentimentCommand)(nil)
}

func TestFineTuneSentimentCommand_HelpViaRegistry(t *testing.T) {
	var buf bytes.Buffer
	app := NewCLI()
	app.out = &buf
	app.RegisterCommand(NewFineTuneSentimentCommand(&buf))

	err := app.Run(context.Background(), []string{"finetune-sentiment", "--help"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "--model") {
		t.Error("--help output should contain --model")
	}
}

func TestFineTuneSentimentCommand_ParseArgs_ModelRequired(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--data", "train.csv"})
	if err == nil || !strings.Contains(err.Error(), "--model is required") {
		t.Errorf("expected --model required error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_DataRequired(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf"})
	if err == nil || !strings.Contains(err.Error(), "--data is required") {
		t.Errorf("expected --data required error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_Defaults(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "train.csv"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "m.gguf" {
		t.Errorf("modelPath = %q, want %q", cfg.modelPath, "m.gguf")
	}
	if cfg.dataPath != "train.csv" {
		t.Errorf("dataPath = %q, want %q", cfg.dataPath, "train.csv")
	}
	if cfg.outputPath != "finetuned.gguf" {
		t.Errorf("outputPath = %q, want %q", cfg.outputPath, "finetuned.gguf")
	}
	if cfg.epochs != 3 {
		t.Errorf("epochs = %d, want 3", cfg.epochs)
	}
	if cfg.lr != 2e-5 {
		t.Errorf("lr = %g, want 2e-5", cfg.lr)
	}
	if cfg.batchSize != 16 {
		t.Errorf("batchSize = %d, want 16", cfg.batchSize)
	}
	if cfg.loraRank != 8 {
		t.Errorf("loraRank = %d, want 8", cfg.loraRank)
	}
	if cfg.valSplit != 0.1 {
		t.Errorf("valSplit = %f, want 0.1", cfg.valSplit)
	}
	if cfg.device != "cpu" {
		t.Errorf("device = %q, want %q", cfg.device, "cpu")
	}
}

func TestFineTuneSentimentCommand_ParseArgs_AllFlags(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{
		"--model", "finbert.gguf",
		"--data", "train.jsonl",
		"--output", "out.gguf",
		"--epochs", "5",
		"--lr", "1e-4",
		"--batch-size", "32",
		"--lora-rank", "16",
		"--val-split", "0.2",
		"--device", "cuda",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "finbert.gguf" {
		t.Errorf("modelPath = %q", cfg.modelPath)
	}
	if cfg.dataPath != "train.jsonl" {
		t.Errorf("dataPath = %q", cfg.dataPath)
	}
	if cfg.outputPath != "out.gguf" {
		t.Errorf("outputPath = %q", cfg.outputPath)
	}
	if cfg.epochs != 5 {
		t.Errorf("epochs = %d", cfg.epochs)
	}
	if cfg.lr != 1e-4 {
		t.Errorf("lr = %g", cfg.lr)
	}
	if cfg.batchSize != 32 {
		t.Errorf("batchSize = %d", cfg.batchSize)
	}
	if cfg.loraRank != 16 {
		t.Errorf("loraRank = %d", cfg.loraRank)
	}
	if cfg.valSplit != 0.2 {
		t.Errorf("valSplit = %f", cfg.valSplit)
	}
	if cfg.device != "cuda" {
		t.Errorf("device = %q", cfg.device)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_EqualsSyntax(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--model=m.gguf", "--data=train.csv", "--epochs=10"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "m.gguf" {
		t.Errorf("modelPath = %q", cfg.modelPath)
	}
	if cfg.dataPath != "train.csv" {
		t.Errorf("dataPath = %q", cfg.dataPath)
	}
	if cfg.epochs != 10 {
		t.Errorf("epochs = %d", cfg.epochs)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_InvalidEpochs(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--epochs", "0"})
	if err == nil || !strings.Contains(err.Error(), "--epochs must be a positive integer") {
		t.Errorf("expected epochs error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_InvalidLR(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--lr", "-1"})
	if err == nil || !strings.Contains(err.Error(), "--lr must be a positive number") {
		t.Errorf("expected lr error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_InvalidBatchSize(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--batch-size", "0"})
	if err == nil || !strings.Contains(err.Error(), "--batch-size must be a positive integer") {
		t.Errorf("expected batch-size error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_InvalidLoraRank(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--lora-rank", "-1"})
	if err == nil || !strings.Contains(err.Error(), "--lora-rank must be a non-negative integer") {
		t.Errorf("expected lora-rank error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_InvalidValSplit(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--val-split", "1.0"})
	if err == nil || !strings.Contains(err.Error(), "--val-split must be in [0, 1)") {
		t.Errorf("expected val-split error, got: %v", err)
	}
}

func TestFineTuneSentimentCommand_ParseArgs_UnknownFlag(t *testing.T) {
	cmd := NewFineTuneSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--data", "d.csv", "--bogus"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("expected unknown flag error, got: %v", err)
	}
}
