package cli

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model"
)

func float32From(f float64) float32 { return float32(f) }
func float32To(v float32) float64   { return float64(v) }

func TestCLI(t *testing.T) {
	cliApp := NewCLI()

	predictCmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	cliApp.RegisterCommand(predictCmd)

	tokenizeCmd := NewTokenizeCommand()
	cliApp.RegisterCommand(tokenizeCmd)

	commands := cliApp.registry.List()
	if len(commands) != 2 {
		t.Errorf("Expected 2 commands, got %d", len(commands))
	}

	for _, name := range []string{"predict", "tokenize"} {
		if _, ok := cliApp.registry.Get(name); !ok {
			t.Errorf("Expected command %q to be registered", name)
		}
	}
}

func TestTokenizeCommand_NoVocab(t *testing.T) {
	cmd := NewTokenizeCommand()
	ctx := context.Background()

	// Without a vocab file, all words map to <unk>
	err := cmd.Run(ctx, []string{"--text", "Hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the tokenizer returns <unk> IDs for unknown words
	ids := cmd.Tok().Encode("Hello world")
	if len(ids) != 2 {
		t.Fatalf("expected 2 token IDs, got %d", len(ids))
	}
	for i, id := range ids {
		if id != 0 {
			t.Errorf("token %d: expected <unk> ID=0, got %d", i, id)
		}
	}
}

func TestTokenizeCommand_WithVocab(t *testing.T) {
	dir := t.TempDir()
	vocabFile := filepath.Join(dir, "vocab.txt")
	err := os.WriteFile(vocabFile, []byte("hello\nworld\nfoo\nbar\n"), 0600)
	if err != nil {
		t.Fatalf("failed to write vocab file: %v", err)
	}

	cmd := NewTokenizeCommand()
	ctx := context.Background()

	err = cmd.Run(ctx, []string{"--vocab", vocabFile, "--text", "hello world baz"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify the tokenizer loaded the vocabulary and produces correct IDs
	tok := cmd.Tok()
	ids := tok.Encode("hello world baz")
	if len(ids) != 3 {
		t.Fatalf("expected 3 token IDs, got %d", len(ids))
	}
	// "hello" added after special tokens (<unk>=0, <s>=1, </s>=2) → ID 3
	if ids[0] != 3 {
		t.Errorf("expected hello ID=3, got %d", ids[0])
	}
	// "world" → ID 4
	if ids[1] != 4 {
		t.Errorf("expected world ID=4, got %d", ids[1])
	}
	// "baz" is OOV → ID 0 (<unk>)
	if ids[2] != 0 {
		t.Errorf("expected baz ID=0 (<unk>), got %d", ids[2])
	}
}

func TestTokenizeCommand_MissingText(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{})
	if err == nil {
		t.Error("expected error for missing text argument")
	}
}

func TestTokenizeCommand_MissingVocabFile(t *testing.T) {
	cmd := NewTokenizeCommand()
	err := cmd.Run(context.Background(), []string{"--vocab", "/nonexistent/vocab.txt", "--text", "hello"})
	if err == nil {
		t.Error("expected error for missing vocab file")
	}
}

func TestPredictCommand_MissingArgs(t *testing.T) {
	cmd := NewPredictCommand(model.Float32ModelRegistry, float32From, float32To)
	ctx := context.Background()

	err := cmd.Run(ctx, []string{})
	if err == nil {
		t.Error("expected error for missing arguments")
	}

	// Missing data-path
	err = cmd.Run(ctx, []string{"--model-path", "m.zmf", "--output", "o.csv"})
	if err == nil {
		t.Error("expected error for missing data-path")
	}

	// Missing output
	err = cmd.Run(ctx, []string{"--model-path", "m.zmf", "--data-path", "d.csv"})
	if err == nil {
		t.Error("expected error for missing output")
	}
}
