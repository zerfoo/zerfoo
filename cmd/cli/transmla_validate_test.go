package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTransMLAValidateCommand_Name(t *testing.T) {
	cmd := NewTransMLAValidateCommand(nil)
	if got := cmd.Name(); got != "transmla-validate" {
		t.Errorf("Name() = %q, want %q", got, "transmla-validate")
	}
}

func TestTransMLAValidateCommand_MissingFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"no flags", nil, "--original is required"},
		{"only original", []string{"--original", "a.gguf"}, "--converted is required"},
		{"only converted", []string{"--converted", "b.gguf"}, "--original is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTransMLAValidateCommand(&bytes.Buffer{})
			err := cmd.Run(context.Background(), tt.args)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want containing %q", err.Error(), tt.want)
			}
		})
	}
}

func TestTransMLAValidateCommand_UnknownFlag(t *testing.T) {
	cmd := NewTransMLAValidateCommand(&bytes.Buffer{})
	err := cmd.Run(context.Background(), []string{"--original", "a.gguf", "--converted", "b.gguf", "--unknown"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("expected unknown flag error, got %v", err)
	}
}

func TestTransMLAValidateCommand_FlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"equals syntax", []string{"--original=a.gguf", "--converted=b.gguf"}, ""},
		{"max-tokens", []string{"--original", "a.gguf", "--converted", "b.gguf", "--max-tokens", "128"}, ""},
		{"bad max-tokens", []string{"--original", "a.gguf", "--converted", "b.gguf", "--max-tokens", "abc"}, "must be a positive integer"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTransMLAValidateCommand(&bytes.Buffer{})
			err := cmd.Run(context.Background(), tt.args)
			if tt.want == "" {
				// Will fail on LoadFile (file doesn't exist) — that's fine for flag parsing test
				if err != nil && strings.Contains(err.Error(), "must be") {
					t.Errorf("unexpected parse error: %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Errorf("error = %v, want containing %q", err, tt.want)
				}
			}
		})
	}
}
