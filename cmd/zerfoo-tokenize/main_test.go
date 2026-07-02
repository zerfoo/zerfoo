package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()

	// Create a vocab file.
	vocabContent := "hello\nworld\nfoo\nbar\n"
	vocabPath := filepath.Join(dir, "vocab.txt")
	if err := os.WriteFile(vocabPath, []byte(vocabContent), 0o600); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		text        string
		vocab       string
		wantErr     bool
		errContains string
		outContains string
	}{
		{
			name:        "empty text",
			text:        "",
			wantErr:     true,
			errContains: "-text flag is required",
		},
		{
			name:        "simple tokenization",
			text:        "hello world",
			outContains: "Token IDs for 'hello world':",
		},
		{
			name:        "with vocab file",
			text:        "hello world",
			vocab:       vocabPath,
			outContains: "Token IDs for 'hello world':",
		},
		{
			name:        "nonexistent vocab",
			text:        "hello",
			vocab:       "/nonexistent/vocab.txt",
			wantErr:     true,
			errContains: "failed to load vocabulary",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := run(tt.text, tt.vocab, &buf)

			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.errContains != "" && err != nil {
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
			}
			if tt.outContains != "" {
				if !strings.Contains(buf.String(), tt.outContains) {
					t.Errorf("output does not contain %q:\n%s", tt.outContains, buf.String())
				}
			}
		})
	}
}

func TestLoadVocab(t *testing.T) {
	dir := t.TempDir()

	vocabContent := "hello\nworld\n\n  \nfoo\n"
	vocabPath := filepath.Join(dir, "vocab.txt")
	if err := os.WriteFile(vocabPath, []byte(vocabContent), 0o600); err != nil {
		t.Fatal(err)
	}

	// Test that loadVocab works without error.
	var buf bytes.Buffer
	err := run("hello world foo", vocabPath, &buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Output should contain token IDs.
	if !strings.Contains(buf.String(), "Token IDs") {
		t.Errorf("expected Token IDs in output, got: %s", buf.String())
	}
}

func TestLoadVocab_FileNotFound(t *testing.T) {
	err := run("test", "/nonexistent/vocab.txt", &bytes.Buffer{})
	if err == nil {
		t.Error("expected error for nonexistent vocab file")
	}
}
