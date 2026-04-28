package cli

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model/registry"
)

func TestRmCommand_Name(t *testing.T) {
	cmd := NewRmCommand(nil, nil)
	if cmd.Name() != "rm" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "rm")
	}
}

func TestRmCommand_Description(t *testing.T) {
	cmd := NewRmCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestRmCommand_Usage(t *testing.T) {
	cmd := NewRmCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "rm") {
		t.Error("Usage() should contain 'rm'")
	}
}

func TestRmCommand_Examples(t *testing.T) {
	cmd := NewRmCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestRmCommand_MissingModelID(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewRmCommand(reg, &buf)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
}

func TestRmCommand_RemovesModel(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"org/model": {ID: "org/model", Path: "/cache/org/model"},
		},
	}
	cmd := NewRmCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"org/model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "Removed org/model") {
		t.Errorf("output = %q, want 'Removed org/model'", buf.String())
	}
	if _, ok := reg.models["org/model"]; ok {
		t.Error("model should have been deleted from registry")
	}
}

func TestRmCommand_ModelNotFound(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewRmCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"nonexistent/model"})
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestRmCommand_UnexpectedArgument(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewRmCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"model1", "model2"})
	if err == nil {
		t.Error("expected error for extra argument")
	}
}

func TestRmCommand_Interface(t *testing.T) {
	var _ Command = (*RmCommand)(nil)
}
