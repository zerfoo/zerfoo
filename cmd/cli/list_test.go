package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model/registry"
)

func TestListCommand_Name(t *testing.T) {
	cmd := NewListCommand(nil, nil)
	if cmd.Name() != "list" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "list")
	}
}

func TestListCommand_Description(t *testing.T) {
	cmd := NewListCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestListCommand_Usage(t *testing.T) {
	cmd := NewListCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "list") {
		t.Error("Usage() should contain 'list'")
	}
}

func TestListCommand_Examples(t *testing.T) {
	cmd := NewListCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestListCommand_EmptyRegistry(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewListCommand(reg, &buf)
	if err := cmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "No cached models") {
		t.Errorf("output = %q, want 'No cached models'", buf.String())
	}
}

func TestListCommand_WithModels(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"org/model-a": {ID: "org/model-a", Path: "/cache/org/model-a", Size: 1024 * 1024 * 500},
			"org/model-b": {ID: "org/model-b", Path: "/cache/org/model-b", Size: 1024 * 1024 * 1024 * 2},
		},
	}
	cmd := NewListCommand(reg, &buf)
	if err := cmd.Run(context.Background(), nil); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "REPO") {
		t.Error("output should contain header 'REPO'")
	}
	if !strings.Contains(output, "org/model-a") {
		t.Errorf("output should contain 'org/model-a', got: %s", output)
	}
	if !strings.Contains(output, "org/model-b") {
		t.Errorf("output should contain 'org/model-b', got: %s", output)
	}
}

func TestListCommand_UnexpectedArgument(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewListCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"extra"})
	if err == nil {
		t.Error("expected error for unexpected argument")
	}
}

func TestListCommand_Interface(t *testing.T) {
	var _ Command = (*ListCommand)(nil)
}

func TestFormatSize(t *testing.T) {
	tests := []struct {
		bytes int64
		want  string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1024 * 1024, "1.0 MB"},
		{1024 * 1024 * 1024, "1.0 GB"},
		{1024*1024*1024*2 + 1024*1024*500, "2.5 GB"},
	}
	for _, tc := range tests {
		got := formatSize(tc.bytes)
		if got != tc.want {
			t.Errorf("formatSize(%d) = %q, want %q", tc.bytes, got, tc.want)
		}
	}
}
