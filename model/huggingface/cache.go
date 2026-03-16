package huggingface

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// CacheManifest tracks downloaded models.
type CacheManifest struct {
	Models []CachedModel `json:"models"`
}

// CachedModel is one entry in the cache.
type CachedModel struct {
	RepoID   string    `json:"repo_id"`
	Filename string    `json:"filename"`
	Path     string    `json:"path"`
	Size     int64     `json:"size"`
	SHA256   string    `json:"sha256,omitempty"`
	AddedAt  time.Time `json:"added_at"`
}

// ManifestPath returns the path to the manifest JSON file (~/.cache/zerfoo/manifest.json).
func ManifestPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("huggingface: home directory: %w", err)
	}
	return filepath.Join(home, ".cache", "zerfoo", "manifest.json"), nil
}

// LoadManifest reads the manifest from disk. Returns an empty manifest if the file does not exist.
func LoadManifest() (*CacheManifest, error) {
	p, err := ManifestPath()
	if err != nil {
		return nil, err
	}
	return loadManifestFrom(p)
}

func loadManifestFrom(path string) (*CacheManifest, error) {
	data, err := os.ReadFile(path) //nolint:gosec // path from ManifestPath()
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &CacheManifest{}, nil
		}
		return nil, fmt.Errorf("huggingface: read manifest: %w", err)
	}
	var m CacheManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("huggingface: decode manifest: %w", err)
	}
	return &m, nil
}

// SaveManifest writes the manifest atomically using a temp file and rename.
func SaveManifest(m *CacheManifest) error {
	p, err := ManifestPath()
	if err != nil {
		return err
	}
	return saveManifestTo(m, p)
}

func saveManifestTo(m *CacheManifest, path string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("huggingface: encode manifest: %w", err)
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return fmt.Errorf("huggingface: create cache dir: %w", err)
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("huggingface: write temp manifest: %w", err)
	}

	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp) // best-effort cleanup
		return fmt.Errorf("huggingface: rename manifest: %w", err)
	}
	return nil
}

// Add adds or updates a model entry (upsert by path).
func (m *CacheManifest) Add(model CachedModel) {
	for i, existing := range m.Models {
		if existing.Path == model.Path {
			m.Models[i] = model
			return
		}
	}
	m.Models = append(m.Models, model)
}

// Remove removes a model entry by path and deletes the file from disk.
// Returns an error if the entry is not found.
func (m *CacheManifest) Remove(path string) error {
	idx := -1
	for i, model := range m.Models {
		if model.Path == path {
			idx = i
			break
		}
	}
	if idx < 0 {
		return fmt.Errorf("huggingface: cache entry not found: %s", path)
	}

	// Delete the file from disk. Ignore "not exist" — the entry still gets removed.
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("huggingface: remove file: %w", err)
	}

	m.Models = append(m.Models[:idx], m.Models[idx+1:]...)
	return nil
}

// List returns all cached models.
func (m *CacheManifest) List() []CachedModel {
	return m.Models
}

// FindByRepoAndFile finds a cached model by repo ID and filename.
func (m *CacheManifest) FindByRepoAndFile(repoID, filename string) (*CachedModel, bool) {
	for i, model := range m.Models {
		if model.RepoID == repoID && model.Filename == filename {
			return &m.Models[i], true
		}
	}
	return nil, false
}

// FindByRepo finds the first cached model matching the given repo ID.
func (m *CacheManifest) FindByRepo(repoID string) (*CachedModel, bool) {
	for i, model := range m.Models {
		if model.RepoID == repoID {
			return &m.Models[i], true
		}
	}
	return nil, false
}

// CacheDir returns the directory where model files are stored (~/.cache/zerfoo/models/).
func CacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("huggingface: home directory: %w", err)
	}
	return filepath.Join(home, ".cache", "zerfoo", "models"), nil
}
