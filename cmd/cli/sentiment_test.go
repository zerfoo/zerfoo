package cli

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference/sentiment"
)

func TestSentimentCommand_Name(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	if got := cmd.Name(); got != "sentiment" {
		t.Errorf("Name() = %q, want %q", got, "sentiment")
	}
}

func TestSentimentCommand_Description(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	if got := cmd.Description(); got == "" {
		t.Error("Description() should not be empty")
	}
}

func TestSentimentCommand_Usage(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	usage := cmd.Usage()
	if usage == "" {
		t.Fatal("Usage() should not be empty")
	}
	for _, flag := range []string{"--model", "--text", "--file", "--batch-size", "--format", "--device", "--continuous"} {
		if !strings.Contains(usage, flag) {
			t.Errorf("Usage() missing flag %s", flag)
		}
	}
}

func TestSentimentCommand_Examples(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	if got := cmd.Examples(); len(got) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestSentimentCommand_Interface(t *testing.T) {
	var _ Command = (*SentimentCommand)(nil)
}

func TestSentimentCommand_HelpViaRegistry(t *testing.T) {
	var buf bytes.Buffer
	app := NewCLI()
	app.out = &buf
	app.RegisterCommand(NewSentimentCommand(&buf))

	err := app.Run(context.Background(), []string{"sentiment", "--help"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(buf.String(), "--model") {
		t.Error("--help output should contain --model")
	}
}

func TestSentimentCommand_ParseArgs_ModelRequired(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--text", "hello"})
	if err == nil || !strings.Contains(err.Error(), "--model is required") {
		t.Errorf("expected --model required error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_TextOrFileRequired(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf"})
	if err == nil || !strings.Contains(err.Error(), "either --text or --file is required") {
		t.Errorf("expected text/file required error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_MutuallyExclusive(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--text", "hi", "--file", "f.txt"})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected mutually exclusive error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_InvalidFormat(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--text", "hi", "--format", "xml"})
	if err == nil || !strings.Contains(err.Error(), "text, json, or csv") {
		t.Errorf("expected format error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_InvalidBatchSize(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--text", "hi", "--batch-size", "0"})
	if err == nil || !strings.Contains(err.Error(), "--batch-size must be >= 1") {
		t.Errorf("expected batch-size error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_UnknownFlag(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	_, err := cmd.parseArgs([]string{"--model", "m.gguf", "--text", "hi", "--bogus"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("expected unknown flag error, got: %v", err)
	}
}

func TestSentimentCommand_ParseArgs_Defaults(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--model", "m.gguf", "--text", "hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "m.gguf" {
		t.Errorf("modelPath = %q, want %q", cfg.modelPath, "m.gguf")
	}
	if cfg.text != "hello world" {
		t.Errorf("text = %q, want %q", cfg.text, "hello world")
	}
	if cfg.batchSize != 64 {
		t.Errorf("batchSize = %d, want 64", cfg.batchSize)
	}
	if cfg.format != "text" {
		t.Errorf("format = %q, want %q", cfg.format, "text")
	}
	if cfg.device != "cpu" {
		t.Errorf("device = %q, want %q", cfg.device, "cpu")
	}
	if cfg.continuous {
		t.Error("continuous should default to false")
	}
}

func TestSentimentCommand_ParseArgs_AllFlags(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{
		"--model", "finbert.gguf",
		"--file", "data.csv",
		"--batch-size", "32",
		"--format", "json",
		"--device", "cuda",
		"--continuous",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "finbert.gguf" {
		t.Errorf("modelPath = %q", cfg.modelPath)
	}
	if cfg.filePath != "data.csv" {
		t.Errorf("filePath = %q", cfg.filePath)
	}
	if cfg.batchSize != 32 {
		t.Errorf("batchSize = %d", cfg.batchSize)
	}
	if cfg.format != "json" {
		t.Errorf("format = %q", cfg.format)
	}
	if cfg.device != "cuda" {
		t.Errorf("device = %q", cfg.device)
	}
	if !cfg.continuous {
		t.Error("continuous should be true")
	}
}

func TestSentimentCommand_ParseArgs_EqualsSyntax(t *testing.T) {
	cmd := NewSentimentCommand(&bytes.Buffer{})
	cfg, err := cmd.parseArgs([]string{"--model=m.gguf", "--text=hello", "--batch-size=16"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.modelPath != "m.gguf" {
		t.Errorf("modelPath = %q", cfg.modelPath)
	}
	if cfg.text != "hello" {
		t.Errorf("text = %q", cfg.text)
	}
	if cfg.batchSize != 16 {
		t.Errorf("batchSize = %d", cfg.batchSize)
	}
}

func TestSentimentCommand_ReadTexts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "input.txt")
	if err := os.WriteFile(path, []byte("line one\nline two\n\nline three\n"), 0600); err != nil {
		t.Fatal(err)
	}

	cmd := NewSentimentCommand(&bytes.Buffer{})
	texts, err := cmd.readTexts(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(texts) != 3 {
		t.Fatalf("got %d texts, want 3", len(texts))
	}
	if texts[0] != "line one" || texts[1] != "line two" || texts[2] != "line three" {
		t.Errorf("texts = %v", texts)
	}
}

func TestSentimentCommand_WriteText(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewSentimentCommand(&buf)

	texts := []string{"good news", "bad news"}
	results := []sentiment.SentimentResult{
		{Label: "positive", Score: 0.92},
		{Label: "negative", Score: 0.78},
	}
	if err := cmd.writeText(texts, results); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, `positive (0.92)  "good news"`) {
		t.Errorf("unexpected text output: %s", out)
	}
	if !strings.Contains(out, `negative (0.78)  "bad news"`) {
		t.Errorf("unexpected text output: %s", out)
	}
}

func TestSentimentCommand_WriteJSON(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewSentimentCommand(&buf)

	texts := []string{"hello"}
	results := []sentiment.SentimentResult{
		{Label: "positive", Score: 0.95},
	}
	if err := cmd.writeJSON(texts, results); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, `"label": "positive"`) {
		t.Errorf("unexpected JSON output: %s", out)
	}
	if !strings.Contains(out, `"score": 0.95`) {
		t.Errorf("unexpected JSON output: %s", out)
	}
}

func TestSentimentCommand_WriteCSV(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewSentimentCommand(&buf)

	texts := []string{"hello"}
	results := []sentiment.SentimentResult{
		{Label: "positive", Score: 0.95},
	}
	if err := cmd.writeCSV(texts, results); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "text,label,score") {
		t.Errorf("missing CSV header: %s", out)
	}
	if !strings.Contains(out, "hello,positive,0.9500") {
		t.Errorf("unexpected CSV output: %s", out)
	}
}
