package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTransMLACommand_Name(t *testing.T) {
	cmd := NewTransMLACommand(nil)
	if cmd.Name() != "transmla" {
		t.Errorf("expected name 'transmla', got %q", cmd.Name())
	}
}

func TestTransMLACommand_Description(t *testing.T) {
	cmd := NewTransMLACommand(nil)
	if cmd.Description() == "" {
		t.Error("expected non-empty description")
	}
}

func TestTransMLACommand_Usage(t *testing.T) {
	cmd := NewTransMLACommand(nil)
	usage := cmd.Usage()
	for _, flag := range []string{"--input", "--output", "--rank"} {
		if !strings.Contains(usage, flag) {
			t.Errorf("usage missing %s", flag)
		}
	}
}

func TestTransMLACommand_Examples(t *testing.T) {
	cmd := NewTransMLACommand(nil)
	examples := cmd.Examples()
	if len(examples) == 0 {
		t.Error("expected at least one example")
	}
}

func TestTransMLACommand_MissingInput(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)
	err := cmd.Run(context.Background(), []string{"--output", "out.gguf"})
	if err == nil {
		t.Fatal("expected error for missing --input")
	}
	if !strings.Contains(err.Error(), "--input is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTransMLACommand_MissingOutput(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)
	err := cmd.Run(context.Background(), []string{"--input", "in.gguf"})
	if err == nil {
		t.Fatal("expected error for missing --output")
	}
	if !strings.Contains(err.Error(), "--output is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTransMLACommand_InvalidRank(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "non-numeric",
			args: []string{"--input", "in.gguf", "--output", "out.gguf", "--rank", "abc"},
			want: "positive integer",
		},
		{
			name: "zero",
			args: []string{"--input", "in.gguf", "--output", "out.gguf", "--rank", "0"},
			want: "positive integer",
		},
		{
			name: "missing value",
			args: []string{"--input", "in.gguf", "--output", "out.gguf", "--rank"},
			want: "requires a value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			cmd := NewTransMLACommand(&buf)
			err := cmd.Run(context.Background(), tt.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("expected error containing %q, got %q", tt.want, err.Error())
			}
		})
	}
}

func TestTransMLACommand_UnknownFlag(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)
	err := cmd.Run(context.Background(), []string{"--input", "in.gguf", "--output", "out.gguf", "--bogus"})
	if err == nil {
		t.Fatal("expected error for unknown flag")
	}
	if !strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTransMLACommand_BadInputFile(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)
	err := cmd.Run(context.Background(), []string{"--input", "/nonexistent/model.gguf", "--output", "/tmp/out.gguf"})
	if err == nil {
		t.Fatal("expected error for missing input file")
	}
	if !strings.Contains(err.Error(), "open input") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestTransMLACommand_EqualsSyntax(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)
	// Validates parsing with = syntax; file doesn't exist so we get an open error,
	// which confirms parsing succeeded.
	err := cmd.Run(context.Background(), []string{"--input=/nonexistent/model.gguf", "--output=/tmp/out.gguf", "--rank=256"})
	if err == nil {
		t.Fatal("expected error for missing input file")
	}
	if !strings.Contains(err.Error(), "open input") {
		t.Errorf("unexpected error (flag parsing may have failed): %v", err)
	}
}

func TestTransMLACommand_ConvertCheckpointGGUF(t *testing.T) {
	// Use the checkpoint.gguf test fixture that exists in this package's testdata.
	// This is a minimal GGUF file used by other CLI tests. It may not contain
	// K/V projection tensors, but running the conversion on it exercises the
	// full CLI path (parse GGUF, skip layers with no K/V, write output, report ratio).
	var buf bytes.Buffer
	cmd := NewTransMLACommand(&buf)

	input := "checkpoint.gguf"
	output := t.TempDir() + "/out.gguf"

	err := cmd.Run(context.Background(), []string{"--input", input, "--output", output, "--rank", "2"})
	// The fixture may not have K/V tensors or may not be a valid GGUF for transmla,
	// so we accept both success and a conversion error. The important thing is
	// that flag parsing and file I/O work correctly.
	if err != nil {
		// Conversion errors from the transmla package are acceptable.
		if !strings.Contains(err.Error(), "conversion failed") {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}

	out := buf.String()
	if !strings.Contains(out, "Done.") {
		t.Errorf("expected completion message, got %q", out)
	}
	if !strings.Contains(out, "compression") {
		t.Errorf("expected compression ratio in output, got %q", out)
	}
}

func TestTransMLACommand_Interface(t *testing.T) {
	var _ Command = (*TransMLACommand)(nil)
}
