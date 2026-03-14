package cli

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/registry"
)

// mockPullRegistry is a test double for registry.ModelRegistry.
type mockPullRegistry struct {
	models  map[string]*registry.ModelInfo
	pulled  bool
	pullErr error
}

func (r *mockPullRegistry) Get(id string) (*registry.ModelInfo, bool) {
	info, ok := r.models[id]
	return info, ok
}

func (r *mockPullRegistry) Pull(_ context.Context, id string) (*registry.ModelInfo, error) {
	r.pulled = true
	if r.pullErr != nil {
		return nil, r.pullErr
	}
	if info, ok := r.models[id]; ok {
		return info, nil
	}
	return &registry.ModelInfo{ID: id, Path: "/cache/" + id, Size: 1024}, nil
}

func (r *mockPullRegistry) List() []registry.ModelInfo { return nil }
func (r *mockPullRegistry) Delete(_ string) error      { return nil }

func TestPullCommand_Name(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Name() != "pull" {
		t.Errorf("Name() = %q, want %q", cmd.Name(), "pull")
	}
}

func TestPullCommand_Description(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if cmd.Description() == "" {
		t.Error("Description() should not be empty")
	}
}

func TestPullCommand_Usage(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if !strings.Contains(cmd.Usage(), "pull") {
		t.Error("Usage() should contain 'pull'")
	}
}

func TestPullCommand_Examples(t *testing.T) {
	cmd := NewPullCommand(nil, nil)
	if len(cmd.Examples()) == 0 {
		t.Error("Examples() should not be empty")
	}
}

func TestPullCommand_MissingModelID(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), nil)
	if err == nil {
		t.Error("expected error for missing model ID")
	}
	if !strings.Contains(err.Error(), "model ID is required") {
		t.Errorf("error = %q, want 'model ID is required'", err.Error())
	}
}

func TestPullCommand_AlreadyCached(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"test-model": {ID: "test-model", Path: "/cache/test-model"},
		},
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"test-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "Already up to date") {
		t.Errorf("output = %q, want 'Already up to date'", buf.String())
	}
}

func TestPullCommand_PullsNewModel(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"new-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
	if !strings.Contains(buf.String(), "Model saved to") {
		t.Errorf("output = %q, want 'Model saved to'", buf.String())
	}
}

func TestPullCommand_CacheDirFlag(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir", "/tmp/cache", "test"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
}

func TestPullCommand_CacheDirMissingValue(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"--cache-dir"})
	if err == nil {
		t.Error("expected error for --cache-dir without value")
	}
}

func TestPullCommand_UnexpectedArgument(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewPullCommand(&mockPullRegistry{models: map[string]*registry.ModelInfo{}}, &buf)
	err := cmd.Run(context.Background(), []string{"model1", "model2"})
	if err == nil {
		t.Error("expected error for extra argument")
	}
}

func TestPullCommand_NilRegistry(t *testing.T) {
	// When no registry is provided, it creates a default one.
	var buf bytes.Buffer
	cmd := NewPullCommand(nil, &buf)
	// This will fail to pull (no real model) but exercises the nil-registry code path.
	err := cmd.Run(context.Background(), []string{"nonexistent"})
	if err == nil {
		t.Error("expected error")
	}
}

func TestPullCommand_PullError(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models:  map[string]*registry.ModelInfo{},
		pullErr: errors.New("network timeout"),
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"fail-model"})
	if err == nil {
		t.Fatal("expected error when Pull fails")
	}
	if !strings.Contains(err.Error(), "network timeout") {
		t.Errorf("error = %q, want it to contain 'network timeout'", err.Error())
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
}

func TestPullCommand_PullOutputIncludesSize(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{models: map[string]*registry.ModelInfo{}}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"size-model"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	output := buf.String()
	if !strings.Contains(output, "Pulling size-model") {
		t.Errorf("output = %q, want it to contain 'Pulling size-model'", output)
	}
	if !strings.Contains(output, "Size: 1024 bytes") {
		t.Errorf("output = %q, want it to contain 'Size: 1024 bytes'", output)
	}
}

func TestPullCommand_AlreadyCachedOutputIncludesPath(t *testing.T) {
	var buf bytes.Buffer
	reg := &mockPullRegistry{
		models: map[string]*registry.ModelInfo{
			"cached": {ID: "cached", Path: "/models/cached"},
		},
	}
	cmd := NewPullCommand(reg, &buf)
	err := cmd.Run(context.Background(), []string{"cached"})
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if !strings.Contains(buf.String(), "/models/cached") {
		t.Errorf("output = %q, want it to contain path '/models/cached'", buf.String())
	}
	if reg.pulled {
		t.Error("Pull should not be called for cached model")
	}
}

func TestPullCommand_NilRegistryCreatesLocalRegistry(t *testing.T) {
	tmp := t.TempDir()
	var buf bytes.Buffer
	cmd := NewPullCommand(nil, &buf)
	// Use --cache-dir so the LocalRegistry targets a temp directory.
	// Pull will fail because no pullFunc is set, but we verify the error
	// comes from the registry (not a nil-pointer panic).
	err := cmd.Run(context.Background(), []string{"--cache-dir", tmp, "org/test-model"})
	if err == nil {
		t.Fatal("expected error from LocalRegistry.Pull (no pull function configured)")
	}
	if !strings.Contains(err.Error(), "no pull function configured") {
		t.Errorf("error = %q, want 'no pull function configured'", err.Error())
	}
}

// Ensure *PullCommand satisfies Command interface at compile time.
func TestPullCommand_Interface(t *testing.T) {
	var _ Command = (*PullCommand)(nil)
	_ = os.Stderr // use os to avoid unused import
}
