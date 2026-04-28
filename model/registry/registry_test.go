package registry

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestNewLocalRegistry(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatalf("NewLocalRegistry error: %v", err)
	}
	if r.CacheDir() != dir {
		t.Errorf("CacheDir() = %q, want %q", r.CacheDir(), dir)
	}
}

func TestNewLocalRegistry_DefaultDir(t *testing.T) {
	// Test that empty cacheDir uses home directory.
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("cannot determine home directory")
	}
	expected := filepath.Join(home, ".zerfoo", "models")

	r, err := NewLocalRegistry("")
	if err != nil {
		t.Fatalf("NewLocalRegistry error: %v", err)
	}
	if r.CacheDir() != expected {
		t.Errorf("CacheDir() = %q, want %q", r.CacheDir(), expected)
	}
}

func TestLocalRegistry_GetMissing(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	_, ok := r.Get("nonexistent/model")
	if ok {
		t.Error("Get for missing model should return false")
	}
}

func TestLocalRegistry_ListEmpty(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	models := r.List()
	if len(models) != 0 {
		t.Errorf("List() on empty registry = %d items, want 0", len(models))
	}
}

func TestLocalRegistry_PullAndGet(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Set a mock pull function.
	r.SetPullFunc(func(_ context.Context, modelID string, targetDir string) (*ModelInfo, error) {
		// Simulate writing a model file.
		if err := os.WriteFile(filepath.Join(targetDir, "model.zmf"), []byte("fake"), 0o600); err != nil {
			return nil, err
		}
		return &ModelInfo{
			ID:           modelID,
			Architecture: "transformer",
			VocabSize:    32000,
			MaxSeqLen:    2048,
			Size:         1024,
		}, nil
	})

	info, err := r.Pull(context.Background(), "test-org/test-model")
	if err != nil {
		t.Fatalf("Pull error: %v", err)
	}
	if info.ID != "test-org/test-model" {
		t.Errorf("ID = %q, want %q", info.ID, "test-org/test-model")
	}
	if info.Architecture != "transformer" {
		t.Errorf("Architecture = %q, want %q", info.Architecture, "transformer")
	}

	// Get should now find the model.
	got, ok := r.Get("test-org/test-model")
	if !ok {
		t.Fatal("Get should return true after Pull")
	}
	if got.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", got.VocabSize)
	}
}

func TestLocalRegistry_PullNoPullFunc(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	_, err = r.Pull(context.Background(), "org/model")
	if err == nil {
		t.Error("Pull without pullFunc should return error")
	}
}

func TestLocalRegistry_List(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	r.SetPullFunc(func(_ context.Context, modelID string, targetDir string) (*ModelInfo, error) {
		return &ModelInfo{
			ID:           modelID,
			Architecture: "transformer",
			VocabSize:    32000,
		}, nil
	})

	for _, id := range []string{"org/model-a", "org/model-b"} {
		if _, err := r.Pull(context.Background(), id); err != nil {
			t.Fatalf("Pull(%s) error: %v", id, err)
		}
	}

	models := r.List()
	if len(models) != 2 {
		t.Errorf("List() = %d items, want 2", len(models))
	}
}

func TestLocalRegistry_Delete(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	r.SetPullFunc(func(_ context.Context, modelID string, targetDir string) (*ModelInfo, error) {
		return &ModelInfo{ID: modelID}, nil
	})

	if _, err := r.Pull(context.Background(), "org/model"); err != nil {
		t.Fatalf("Pull error: %v", err)
	}

	if err := r.Delete("org/model"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}

	_, ok := r.Get("org/model")
	if ok {
		t.Error("Get should return false after Delete")
	}
}

func TestLocalRegistry_DeleteMissing(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	if err := r.Delete("nonexistent/model"); err == nil {
		t.Error("Delete missing model should return error")
	}
}

func TestLocalRegistry_PullError(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	r.SetPullFunc(func(_ context.Context, _ string, _ string) (*ModelInfo, error) {
		return nil, fmt.Errorf("download failed")
	})

	_, err = r.Pull(context.Background(), "org/model")
	if err == nil {
		t.Error("Pull should return error when pullFunc fails")
	}
}

func TestLocalRegistry_ReadInvalidConfig(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Create model dir with invalid config.json
	modelDir := filepath.Join(dir, "org", "broken")
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, ok := r.Get("org/broken")
	if ok {
		t.Error("Get should return false for invalid config.json")
	}

	// List should skip invalid configs.
	models := r.List()
	if len(models) != 0 {
		t.Errorf("List() should skip invalid configs, got %d items", len(models))
	}
}

func TestLocalRegistry_ModelDir(t *testing.T) {
	dir := t.TempDir()
	r, err := NewLocalRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		modelID string
		want    string
	}{
		{"org/model", filepath.Join(dir, "org", "model")},
		{"single", filepath.Join(dir, "single")},
	}
	for _, tc := range tests {
		got, err := r.modelDir(tc.modelID)
		if err != nil {
			t.Errorf("modelDir(%q) unexpected error: %v", tc.modelID, err)
			continue
		}
		if got != tc.want {
			t.Errorf("modelDir(%q) = %q, want %q", tc.modelID, got, tc.want)
		}
	}

	// Verify path traversal is rejected.
	traversalIDs := []string{"../etc", "org/../../etc", "org/../../../passwd"}
	for _, id := range traversalIDs {
		_, err := r.modelDir(id)
		if err == nil {
			t.Errorf("modelDir(%q) should have returned error for path traversal", id)
		}
	}
}
