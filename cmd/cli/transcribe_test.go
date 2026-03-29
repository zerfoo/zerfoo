package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestTranscribeCommand_Name(t *testing.T) {
	cmd := NewTranscribeCommand(nil)
	if got := cmd.Name(); got != "transcribe" {
		t.Errorf("Name() = %q, want %q", got, "transcribe")
	}
}

func TestTranscribeCommand_MissingFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"no flags", nil, "--model is required"},
		{"only model", []string{"--model", "m.gguf"}, "--audio is required"},
		{"only audio", []string{"--audio", "a.wav"}, "--model is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := NewTranscribeCommand(&bytes.Buffer{})
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

func TestTranscribeCommand_UnknownFlag(t *testing.T) {
	cmd := NewTranscribeCommand(&bytes.Buffer{})
	err := cmd.Run(context.Background(), []string{"--model", "m", "--audio", "a", "--unknown"})
	if err == nil || !strings.Contains(err.Error(), "unknown flag") {
		t.Errorf("expected unknown flag error, got %v", err)
	}
}
