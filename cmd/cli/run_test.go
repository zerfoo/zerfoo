package cli

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
)

func TestRunCommand_Name(t *testing.T) {
	cmd := NewRunCommand(nil, nil)
	if cmd.Name() != "run" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "run")
	}
}

func TestRunCommand_Description(t *testing.T) {
	cmd := NewRunCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestRunCommand_Usage(t *testing.T) {
	cmd := NewRunCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "run") {
		t.Error("Usage() should contain 'run'")
	}
}

func TestRunCommand_Examples(t *testing.T) {
	cmd := NewRunCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestRunCommand_MissingModelID(t *testing.T) {
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
}

func TestRunCommand_LoadError(t *testing.T) {
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("load failed")
	}
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err == nil {
		t.Error("expected error from load")
	}
	if !strings.Contains(err.Error(), "load model") {
		t.Errorf("error = %q, want 'load model'", err.Error())
	}
}

func TestRunCommand_FlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
		err  string
	}{
		{"temperature missing value", []string{"--temperature"}, "--temperature requires a value"},
		{"temperature invalid", []string{"--temperature", "abc", "m"}, "--temperature:"},
		{"top-k missing value", []string{"--top-k"}, "--top-k requires a value"},
		{"top-k invalid", []string{"--top-k", "abc", "m"}, "--top-k:"},
		{"top-p missing value", []string{"--top-p"}, "--top-p requires a value"},
		{"top-p invalid", []string{"--top-p", "abc", "m"}, "--top-p:"},
		{"max-tokens missing value", []string{"--max-tokens"}, "--max-tokens requires a value"},
		{"max-tokens invalid", []string{"--max-tokens", "abc", "m"}, "--max-tokens:"},
		{"system missing value", []string{"--system"}, "--system requires a value"},
		{"repetition-penalty missing value", []string{"--repetition-penalty"}, "--repetition-penalty requires a value"},
		{"repetition-penalty invalid", []string{"--repetition-penalty", "abc", "m"}, "--repetition-penalty:"},
		{"cache-dir missing value", []string{"--cache-dir"}, "--cache-dir requires a value"},
		{"unexpected arg", []string{"model1", "model2"}, "unexpected argument"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewRunCommand(strings.NewReader(""), &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Error("expected error")
			}
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.err)
			}
		})
	}
}

func TestRunCommand_REPL(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	// Simulate user typing "hello" then EOF.
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(out.String(), "Model loaded") {
		t.Errorf("output should contain 'Model loaded', got %q", out.String())
	}
}

func TestRunCommand_REPL_EmptyLines(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	// Empty lines should be skipped.
	cmd := NewRunCommand(strings.NewReader("\n\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestRunCommand_REPL_WithOptions(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{
		"--temperature", "0.5",
		"--top-k", "10",
		"--top-p", "0.9",
		"--repetition-penalty", "1.2",
		"--max-tokens", "5",
		"--system", "You are helpful",
		"--cache-dir", "/tmp/cache",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestRunCommand_SystemPrompt(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{
		"--system", "You are a helpful assistant",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "Model loaded") {
		t.Errorf("output should contain 'Model loaded', got %q", output)
	}
	// When --system is set, Chat is used (non-streaming), so the response
	// should appear as a complete line after the prompt marker.
	if !strings.Contains(output, "> ") {
		t.Errorf("output should contain prompt marker '> ', got %q", output)
	}
}

func TestRunCommand_EqualsSyntax(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{
		"--temperature=0.7",
		"--top-k=40",
		"--max-tokens=256",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run with --flag=value syntax failed: %v", err)
	}
	if !strings.Contains(out.String(), "Model loaded") {
		t.Errorf("output should contain 'Model loaded', got %q", out.String())
	}
}

func TestRunCommand_MixedEqualsSyntax(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{
		"--temperature=0.7",
		"--top-k", "40",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run with mixed syntax failed: %v", err)
	}
}

func TestRunCommand_Interface(t *testing.T) {
	var _ Command = (*RunCommand)(nil)
}

func TestRunCommand_JSONSchemaParses(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	schema := `{"type":"object","properties":{"name":{"type":"string"}}}`
	err := cmd.Run(context.Background(), []string{
		"--json-schema", schema,
		"--prompt", "Generate a name",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run with --json-schema failed: %v", err)
	}
	// In non-interactive mode, output should be raw generation (no "Model loaded" banner).
	if strings.Contains(out.String(), "Model loaded") {
		t.Errorf("non-interactive mode should not print 'Model loaded', got %q", out.String())
	}
}

func TestRunCommand_JSONSchemaInvalid(t *testing.T) {
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("should not be called")
	}
	err := cmd.Run(context.Background(), []string{
		"--json-schema", "not valid json",
		"--prompt", "test",
		"test-model",
	})
	if err == nil {
		t.Fatal("expected error for invalid JSON schema")
	}
	if !strings.Contains(err.Error(), "--json-schema") {
		t.Errorf("error = %q, want to contain '--json-schema'", err.Error())
	}
}

func TestRunCommand_JSONSchemaUnsupported(t *testing.T) {
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return nil, errors.New("should not be called")
	}
	// $ref is unsupported by the grammar converter.
	schema := `{"type":"object","$ref":"#/defs/foo"}`
	err := cmd.Run(context.Background(), []string{
		"--json-schema", schema,
		"--prompt", "test",
		"test-model",
	})
	if err == nil {
		t.Fatal("expected error for unsupported schema feature")
	}
	if !strings.Contains(err.Error(), "--json-schema") {
		t.Errorf("error = %q, want to contain '--json-schema'", err.Error())
	}
}

func TestRunCommand_JSONSchemaRequiresPrompt(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader(""), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	schema := `{"type":"object","properties":{"name":{"type":"string"}}}`
	err := cmd.Run(context.Background(), []string{
		"--json-schema", schema,
		"test-model",
	})
	if err == nil {
		t.Fatal("expected error when --prompt is missing with --json-schema")
	}
	if !strings.Contains(err.Error(), "--prompt") {
		t.Errorf("error = %q, want to contain '--prompt'", err.Error())
	}
}

func TestRunCommand_JSONSchemaFlagParsing(t *testing.T) {
	tests := []struct {
		name string
		args []string
		err  string
	}{
		{"json-schema missing value", []string{"--json-schema"}, "--json-schema requires a value"},
		{"prompt missing value", []string{"--prompt"}, "--prompt requires a value"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			cmd := NewRunCommand(strings.NewReader(""), &out)
			cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
				return nil, errors.New("should not be called")
			}
			err := cmd.Run(context.Background(), tc.args)
			if err == nil {
				t.Error("expected error")
			}
			if !strings.Contains(err.Error(), tc.err) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.err)
			}
		})
	}
}

func TestRunCommand_QuaRotFlag(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	var called bool
	cmd.loadFn = func(_ string, opts ...inference.Option) (*inference.Model, error) {
		called = true
		// --quarot should produce at least one load option.
		if len(opts) == 0 {
			t.Error("expected load options to include WithQuaRot, got none")
		}
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{"--quarot", "test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !called {
		t.Error("loadFn was not called")
	}
}

func TestRunCommand_QuaRotUsage(t *testing.T) {
	cmd := NewRunCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "--quarot") {
		t.Errorf("Usage() should contain '--quarot', got %q", cmd.Usage())
	}
}

func TestRunCommand_PromptWithoutSchema(t *testing.T) {
	mdl := buildCLITestModel(t)
	var out bytes.Buffer
	// --prompt without --json-schema falls through to interactive mode.
	cmd := NewRunCommand(strings.NewReader("hello\n"), &out)
	cmd.loadFn = func(_ string, _ ...inference.Option) (*inference.Model, error) {
		return mdl, nil
	}
	err := cmd.Run(context.Background(), []string{
		"--prompt", "ignored in interactive mode",
		"test-model",
	})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(out.String(), "Model loaded") {
		t.Errorf("should be interactive mode, got %q", out.String())
	}
}
