package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestVersionCommand_DefaultDevel(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewVersionCommand("", &buf)

	if err := cmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := buf.String()
	if !strings.Contains(got, "(devel)") {
		t.Errorf("expected output to contain (devel), got %q", got)
	}
}

func TestVersionCommand_SetVersion(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewVersionCommand("v1.2.3", &buf)

	if err := cmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := buf.String()
	if !strings.Contains(got, "v1.2.3") {
		t.Errorf("expected output to contain v1.2.3, got %q", got)
	}
	if !strings.HasPrefix(got, "zerfoo version ") {
		t.Errorf("expected output to start with 'zerfoo version ', got %q", got)
	}
}

func TestVersionCommand_Name(t *testing.T) {
	cmd := NewVersionCommand("", nil)
	if cmd.Name() != "version" {
		t.Errorf("expected name 'version', got %q", cmd.Name())
	}
}
