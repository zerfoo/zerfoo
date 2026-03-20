package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTrainCommand_Name(t *testing.T) {
	cmd := NewTrainCommand(&bytes.Buffer{})
	if got := cmd.Name(); got != "train" {
		t.Errorf("Name() = %q, want %q", got, "train")
	}
}

func TestTrainCommand_Description(t *testing.T) {
	cmd := NewTrainCommand(&bytes.Buffer{})
	if got := cmd.Description(); got == "" {
		t.Error("Description() should not be empty")
	}
}

func TestTrainCommand_Usage(t *testing.T) {
	cmd := NewTrainCommand(&bytes.Buffer{})
	usage := cmd.Usage()
	if usage == "" {
		t.Fatal("Usage() should not be empty")
	}
	for _, flag := range []string{"--config", "--data", "--output", "--world-size", "--rank", "--master-addr", "--master-port", "--epochs", "--batch-size", "--lr"} {
		if !strings.Contains(usage, flag) {
			t.Errorf("Usage() missing flag %s", flag)
		}
	}
}

func TestTrainCommand_Examples(t *testing.T) {
	cmd := NewTrainCommand(&bytes.Buffer{})
	if got := cmd.Examples(); len(got) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestTrainCommand_Interface(t *testing.T) {
	var _ Command = (*TrainCommand)(nil)
}

func TestTrainCommand_HelpViaRegistry(t *testing.T) {
	var buf bytes.Buffer
	app := NewCLI()
	app.out = &buf
	app.RegisterCommand(NewTrainCommand(&buf))

	err := app.Run(context.Background(), []string{"train", "--help"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "--config") {
		t.Error("--help output should contain --config")
	}
}

func TestTrainCommand_MissingRequired(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"missing both", []string{}, "--config is required"},
		{"missing data", []string{"--config", "model.gguf"}, "--data is required"},
		{"missing config", []string{"--data", "train.jsonl"}, "--config is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTrainCommand(&bytes.Buffer{})
			err := cmd.Run(context.Background(), tt.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestTrainCommand_FlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"defaults", []string{"--config", "m.gguf", "--data", "d.jsonl"}},
		{"all flags", []string{
			"--config", "m.gguf",
			"--data", "d.jsonl",
			"--output", "out.gguf",
			"--world-size", "1",
			"--rank", "0",
			"--master-addr", "127.0.0.1",
			"--master-port", "30000",
			"--epochs", "2",
			"--batch-size", "8",
			"--lr", "5e-5",
		}},
		{"equals syntax", []string{
			"--config=m.gguf",
			"--data=d.jsonl",
			"--epochs=3",
			"--lr=1e-3",
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTrainCommand(&bytes.Buffer{})
			cfg, err := cmd.parseArgs(tt.args)
			if err != nil {
				t.Fatalf("parseArgs() error: %v", err)
			}
			if cfg.modelPath == "" {
				t.Error("modelPath should be set")
			}
			if cfg.dataPath == "" {
				t.Error("dataPath should be set")
			}
		})
	}
}

func TestTrainCommand_InvalidFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"unknown flag", []string{"--config", "m.gguf", "--data", "d.jsonl", "--bogus"}, "unknown flag"},
		{"world-size zero", []string{"--config", "m.gguf", "--data", "d.jsonl", "--world-size", "0"}, "--world-size must be >= 1"},
		{"world-size negative", []string{"--config", "m.gguf", "--data", "d.jsonl", "--world-size", "-1"}, "--world-size must be >= 1"},
		{"rank negative", []string{"--config", "m.gguf", "--data", "d.jsonl", "--rank", "-1"}, "--rank must be >= 0"},
		{"rank >= world-size", []string{"--config", "m.gguf", "--data", "d.jsonl", "--world-size", "2", "--rank", "2"}, "--rank must be in [0, world-size)"},
		{"port too high", []string{"--config", "m.gguf", "--data", "d.jsonl", "--master-port", "99999"}, "--master-port must be in [0, 65535]"},
		{"epochs zero", []string{"--config", "m.gguf", "--data", "d.jsonl", "--epochs", "0"}, "--epochs must be >= 1"},
		{"batch-size zero", []string{"--config", "m.gguf", "--data", "d.jsonl", "--batch-size", "0"}, "--batch-size must be >= 1"},
		{"lr zero", []string{"--config", "m.gguf", "--data", "d.jsonl", "--lr", "0"}, "--lr must be a positive number"},
		{"lr negative", []string{"--config", "m.gguf", "--data", "d.jsonl", "--lr", "-1"}, "--lr must be a positive number"},
		{"lr non-numeric", []string{"--config", "m.gguf", "--data", "d.jsonl", "--lr", "abc"}, "--lr must be a positive number"},
		{"config no value", []string{"--config"}, "--config requires a value"},
		{"data no value", []string{"--config", "m.gguf", "--data"}, "--data requires a value"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTrainCommand(&bytes.Buffer{})
			err := cmd.Run(context.Background(), tt.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tt.want)
			}
		})
	}
}

func TestTrainCommand_LocalRun(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTrainCommand(&buf)
	err := cmd.Run(context.Background(), []string{
		"--config", "model.gguf",
		"--data", "train.jsonl",
		"--epochs", "1",
		"--batch-size", "4",
	})
	if err != nil {
		t.Fatalf("local run failed: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "world-size=1") {
		t.Error("output should indicate single-process mode")
	}
	if !strings.Contains(out, "checkpoint saved") {
		t.Error("output should indicate checkpoint was saved")
	}
}

func TestTrainCommand_Defaults(t *testing.T) {
	cmd := NewTrainCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--config", "m.gguf", "--data", "d.jsonl"})
	if err != nil {
		t.Fatalf("parseArgs() error: %v", err)
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
	if cfg.batchSize != 4 {
		t.Errorf("batchSize = %d, want 4", cfg.batchSize)
	}
	if cfg.lr != 1e-4 {
		t.Errorf("lr = %v, want 1e-4", cfg.lr)
	}
}
