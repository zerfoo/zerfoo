package huggingface

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestManifestPath(t *testing.T) {
	p, err := ManifestPath()
	if err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".cache", "zerfoo", "manifest.json")
	if p != want {
		t.Errorf("ManifestPath() = %q, want %q", p, want)
	}
}

func TestLoadManifestMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	m, err := loadManifestFrom(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Models) != 0 {
		t.Errorf("expected empty manifest, got %d models", len(m.Models))
	}
}

func TestAddAndList(t *testing.T) {
	m := &CacheManifest{}
	now := time.Now()

	entry := CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma-3-4b-Q4_K_M.gguf",
		Path:     "/tmp/zerfoo/gemma.gguf",
		Size:     1024,
		SHA256:   "abc123",
		AddedAt:  now,
	}

	m.Add(entry)
	list := m.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 model, got %d", len(list))
	}
	if list[0].RepoID != "google/gemma-3-4b" {
		t.Errorf("unexpected repo ID: %s", list[0].RepoID)
	}
	if list[0].Size != 1024 {
		t.Errorf("unexpected size: %d", list[0].Size)
	}
}

func TestAddDeduplication(t *testing.T) {
	m := &CacheManifest{}
	now := time.Now()

	entry1 := CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma.gguf",
		Path:     "/tmp/zerfoo/gemma.gguf",
		Size:     1024,
		AddedAt:  now,
	}
	entry2 := CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma.gguf",
		Path:     "/tmp/zerfoo/gemma.gguf",
		Size:     2048,
		AddedAt:  now.Add(time.Hour),
	}

	m.Add(entry1)
	m.Add(entry2)

	if len(m.Models) != 1 {
		t.Fatalf("expected 1 model after dedup, got %d", len(m.Models))
	}
	if m.Models[0].Size != 2048 {
		t.Errorf("expected updated size 2048, got %d", m.Models[0].Size)
	}
}

func TestRemoveExisting(t *testing.T) {
	dir := t.TempDir()
	fpath := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(fpath, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := &CacheManifest{}
	m.Add(CachedModel{
		RepoID:   "test/model",
		Filename: "model.gguf",
		Path:     fpath,
		Size:     4,
		AddedAt:  time.Now(),
	})

	if err := m.Remove(fpath); err != nil {
		t.Fatal(err)
	}
	if len(m.Models) != 0 {
		t.Errorf("expected 0 models after remove, got %d", len(m.Models))
	}
	if _, err := os.Stat(fpath); !os.IsNotExist(err) {
		t.Error("expected file to be deleted")
	}
}

func TestRemoveMissing(t *testing.T) {
	m := &CacheManifest{}
	err := m.Remove("/nonexistent/path")
	if err == nil {
		t.Error("expected error when removing missing entry")
	}
}

func TestFindByRepoAndFile(t *testing.T) {
	m := &CacheManifest{}
	m.Add(CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma.gguf",
		Path:     "/tmp/gemma.gguf",
		Size:     1024,
		AddedAt:  time.Now(),
	})

	tests := []struct {
		name     string
		repoID   string
		filename string
		found    bool
	}{
		{"found", "google/gemma-3-4b", "gemma.gguf", true},
		{"wrong repo", "meta/llama-3", "gemma.gguf", false},
		{"wrong file", "google/gemma-3-4b", "other.gguf", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := m.FindByRepoAndFile(tt.repoID, tt.filename)
			if ok != tt.found {
				t.Errorf("FindByRepoAndFile(%q, %q) found = %v, want %v",
					tt.repoID, tt.filename, ok, tt.found)
			}
		})
	}
}

func TestFindByRepo(t *testing.T) {
	m := &CacheManifest{}
	m.Add(CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma.gguf",
		Path:     "/tmp/gemma.gguf",
		Size:     1024,
		AddedAt:  time.Now(),
	})

	tests := []struct {
		name   string
		repoID string
		found  bool
	}{
		{"found", "google/gemma-3-4b", true},
		{"not found", "meta/llama-3", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ok := m.FindByRepo(tt.repoID)
			if ok != tt.found {
				t.Errorf("FindByRepo(%q) found = %v, want %v",
					tt.repoID, ok, tt.found)
			}
		})
	}
}

func TestCacheDir(t *testing.T) {
	dir, err := CacheDir()
	if err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	want := filepath.Join(home, ".cache", "zerfoo", "models")
	if dir != want {
		t.Errorf("CacheDir() = %q, want %q", dir, want)
	}
}

func TestSaveAndLoadManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	original := &CacheManifest{}
	original.Add(CachedModel{
		RepoID:   "google/gemma-3-4b",
		Filename: "gemma.gguf",
		Path:     "/tmp/gemma.gguf",
		Size:     4096,
		SHA256:   "deadbeef",
		AddedAt:  time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
	})

	if err := saveManifestTo(original, path); err != nil {
		t.Fatal(err)
	}

	loaded, err := loadManifestFrom(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(loaded.Models))
	}
	got := loaded.Models[0]
	if got.RepoID != "google/gemma-3-4b" || got.Size != 4096 || got.SHA256 != "deadbeef" {
		t.Errorf("round-trip mismatch: %+v", got)
	}
}

func TestAtomicWriteCleanup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	m := &CacheManifest{}
	if err := saveManifestTo(m, path); err != nil {
		t.Fatal(err)
	}

	// After successful save, no .tmp file should remain.
	tmp := path + ".tmp"
	if _, err := os.Stat(tmp); !os.IsNotExist(err) {
		t.Error("expected .tmp file to be cleaned up after atomic write")
	}

	// The manifest file should exist.
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected manifest file to exist: %v", err)
	}
}
