// Package repository provides a model repository for storing and managing GGUF model files.
package repository

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var (
	// ErrNotFound is returned when a model is not found.
	ErrNotFound = errors.New("repository: model not found")
	// ErrAlreadyExists is returned when a model with the same ID already exists.
	ErrAlreadyExists = errors.New("repository: model already exists")
	// ErrPathTraversal is returned when a model ID attempts to escape the base directory.
	ErrPathTraversal = errors.New("repository: path traversal detected")
)

// ModelMetadata describes a stored model.
type ModelMetadata struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Version   string    `json:"version"`
	Format    string    `json:"format"`
	Size      int64     `json:"size"`
	SHA256    string    `json:"sha256"`
	CreatedAt time.Time `json:"created_at"`
}

// ModelRepository defines the interface for managing model storage.
type ModelRepository interface {
	// List returns metadata for all stored models.
	List() ([]ModelMetadata, error)
	// Get returns metadata for a specific model.
	Get(id string) (ModelMetadata, error)
	// Upload stores a model file and its metadata. The reader provides the model data.
	Upload(meta ModelMetadata, r io.Reader) error
	// Delete removes a model and its metadata.
	Delete(id string) error
}

// FileSystemRepository implements ModelRepository using local filesystem storage.
// Models are stored as {baseDir}/{modelID}/model.gguf with a metadata.json sidecar.
type FileSystemRepository struct {
	baseDir string
	mu      sync.RWMutex
}

// NewFileSystemRepository creates a new FileSystemRepository rooted at baseDir.
// The directory is created if it does not exist.
func NewFileSystemRepository(baseDir string) (*FileSystemRepository, error) {
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("repository: create base dir: %w", err)
	}
	return &FileSystemRepository{baseDir: baseDir}, nil
}

func (r *FileSystemRepository) modelDir(id string) (string, error) {
	joined := filepath.Clean(filepath.Join(r.baseDir, id))
	base := filepath.Clean(r.baseDir)
	if joined == base || !strings.HasPrefix(joined, base+string(filepath.Separator)) {
		return "", ErrPathTraversal
	}
	return joined, nil
}

func (r *FileSystemRepository) modelPath(id string) (string, error) {
	dir, err := r.modelDir(id)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "model.gguf"), nil
}

func (r *FileSystemRepository) metadataPath(id string) (string, error) {
	dir, err := r.modelDir(id)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "metadata.json"), nil
}

// List returns metadata for all stored models.
func (r *FileSystemRepository) List() ([]ModelMetadata, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	entries, err := os.ReadDir(r.baseDir)
	if err != nil {
		return nil, fmt.Errorf("repository: list: %w", err)
	}

	var models []ModelMetadata
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		meta, err := r.readMetadata(entry.Name())
		if err != nil {
			continue // skip directories without valid metadata
		}
		models = append(models, meta)
	}
	return models, nil
}

// Get returns metadata for a specific model.
func (r *FileSystemRepository) Get(id string) (ModelMetadata, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.readMetadata(id)
}

// Upload stores a model file and writes its metadata sidecar.
func (r *FileSystemRepository) Upload(meta ModelMetadata, data io.Reader) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dir, err := r.modelDir(meta.ID)
	if err != nil {
		return err
	}
	if _, err := os.Stat(dir); err == nil {
		return ErrAlreadyExists
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("repository: create model dir: %w", err)
	}

	// Write model file and compute SHA256.
	mp, err := r.modelPath(meta.ID)
	if err != nil {
		os.RemoveAll(dir)
		return err
	}
	f, err := os.Create(mp)
	if err != nil {
		os.RemoveAll(dir)
		return fmt.Errorf("repository: create model file: %w", err)
	}

	h := sha256.New()
	w := io.MultiWriter(f, h)
	n, err := io.Copy(w, data)
	f.Close()
	if err != nil {
		os.RemoveAll(dir)
		return fmt.Errorf("repository: write model file: %w", err)
	}

	meta.Size = n
	meta.SHA256 = hex.EncodeToString(h.Sum(nil))
	if meta.Format == "" {
		meta.Format = "gguf"
	}
	if meta.CreatedAt.IsZero() {
		meta.CreatedAt = time.Now().UTC()
	}

	if err := r.writeMetadata(meta); err != nil {
		os.RemoveAll(dir)
		return err
	}
	return nil
}

// Delete removes a model and its metadata from disk.
func (r *FileSystemRepository) Delete(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dir, err := r.modelDir(id)
	if err != nil {
		return err
	}
	mdPath, err := r.metadataPath(id)
	if err != nil {
		return err
	}
	if _, err := os.Stat(mdPath); errors.Is(err, os.ErrNotExist) {
		return ErrNotFound
	}
	return os.RemoveAll(dir)
}

func (r *FileSystemRepository) readMetadata(id string) (ModelMetadata, error) {
	mdPath, err := r.metadataPath(id)
	if err != nil {
		return ModelMetadata{}, err
	}
	data, err := os.ReadFile(mdPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ModelMetadata{}, ErrNotFound
		}
		return ModelMetadata{}, fmt.Errorf("repository: read metadata: %w", err)
	}
	var meta ModelMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return ModelMetadata{}, fmt.Errorf("repository: unmarshal metadata: %w", err)
	}
	return meta, nil
}

func (r *FileSystemRepository) writeMetadata(meta ModelMetadata) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("repository: marshal metadata: %w", err)
	}
	mdPath, err2 := r.metadataPath(meta.ID)
	if err2 != nil {
		return err2
	}
	if err := os.WriteFile(mdPath, data, 0o644); err != nil {
		return fmt.Errorf("repository: write metadata: %w", err)
	}
	return nil
}
