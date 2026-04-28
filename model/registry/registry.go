package registry

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ModelInfo describes a locally cached model.
type ModelInfo struct {
	ID           string `json:"id"`
	Path         string `json:"path"`
	Architecture string `json:"architecture"`
	VocabSize    int    `json:"vocab_size"`
	MaxSeqLen    int    `json:"max_seq_len"`
	Size         int64  `json:"size"`
}

// ModelRegistry manages local model storage and retrieval.
type ModelRegistry interface {
	// Pull downloads a model by ID and caches it locally.
	Pull(ctx context.Context, modelID string) (*ModelInfo, error)
	// Get returns a locally cached model by ID.
	Get(modelID string) (*ModelInfo, bool)
	// List returns all locally cached models.
	List() []ModelInfo
	// Delete removes a locally cached model.
	Delete(modelID string) error
}

// LocalRegistry implements ModelRegistry with a local filesystem cache.
// Cache layout: <cacheDir>/<org>/<model>/ containing model.zmf, tokenizer.json, config.json.
type LocalRegistry struct {
	cacheDir string
	mu       sync.RWMutex
	// pullFunc is called by Pull to download a model. It receives the model ID
	// and the target directory. If nil, Pull returns an error.
	pullFunc PullFunc
}

// PullFunc downloads or converts a model into the target directory.
// The target directory is guaranteed to exist when this function is called.
type PullFunc func(ctx context.Context, modelID string, targetDir string) (*ModelInfo, error)

// NewLocalRegistry creates a LocalRegistry with the given cache directory.
// If cacheDir is empty, it defaults to ~/.zerfoo/models/.
func NewLocalRegistry(cacheDir string) (*LocalRegistry, error) {
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("determine home directory: %w", err)
		}
		cacheDir = filepath.Join(home, ".zerfoo", "models")
	}
	if err := os.MkdirAll(cacheDir, 0o750); err != nil {
		return nil, fmt.Errorf("create cache directory: %w", err)
	}
	return &LocalRegistry{cacheDir: cacheDir}, nil
}

// SetPullFunc sets the function used by Pull to download models.
func (r *LocalRegistry) SetPullFunc(fn PullFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pullFunc = fn
}

// CacheDir returns the path to the local model cache directory.
func (r *LocalRegistry) CacheDir() string {
	return r.cacheDir
}

// Pull downloads a model and caches it locally.
func (r *LocalRegistry) Pull(ctx context.Context, modelID string) (*ModelInfo, error) {
	r.mu.Lock()
	pullFn := r.pullFunc
	r.mu.Unlock()

	if pullFn == nil {
		return nil, fmt.Errorf("no pull function configured")
	}

	targetDir, err := r.modelDir(modelID)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(targetDir, 0o750); err != nil {
		return nil, fmt.Errorf("create model directory: %w", err)
	}

	info, err := pullFn(ctx, modelID, targetDir)
	if err != nil {
		return nil, fmt.Errorf("pull %s: %w", modelID, err)
	}

	// Ensure the model ID and path are set.
	info.ID = modelID
	info.Path = targetDir

	// Write config.json with model info.
	if err := r.writeModelInfo(targetDir, info); err != nil {
		return nil, fmt.Errorf("write model info: %w", err)
	}
	return info, nil
}

// Get returns a locally cached model by ID.
func (r *LocalRegistry) Get(modelID string) (*ModelInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	targetDir, err := r.modelDir(modelID)
	if err != nil {
		return nil, false
	}
	info, err := r.readModelInfo(targetDir)
	if err != nil {
		return nil, false
	}
	return info, true
}

// List returns all locally cached models.
func (r *LocalRegistry) List() []ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var models []ModelInfo
	// Walk the cache directory looking for config.json files.
	_ = filepath.Walk(r.cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip errors.
		}
		if info.Name() == "config.json" && !info.IsDir() {
			modelInfo, readErr := r.readModelInfo(filepath.Dir(path))
			if readErr == nil {
				models = append(models, *modelInfo)
			}
		}
		return nil
	})
	return models
}

// Delete removes a locally cached model.
func (r *LocalRegistry) Delete(modelID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	targetDir, err := r.modelDir(modelID)
	if err != nil {
		return err
	}
	if _, err := os.Stat(targetDir); os.IsNotExist(err) {
		return fmt.Errorf("model %q not found", modelID)
	}
	return os.RemoveAll(targetDir)
}

// modelDir returns the cache directory path for a given model ID.
// Model ID format: "org/model" -> cacheDir/org/model/
// Returns an error if the resolved path escapes the cache directory.
func (r *LocalRegistry) modelDir(modelID string) (string, error) {
	// Sanitize the model ID to create a valid directory path.
	parts := strings.SplitN(modelID, "/", 2)
	var resolved string
	if len(parts) == 2 {
		resolved = filepath.Join(r.cacheDir, parts[0], parts[1])
	} else {
		resolved = filepath.Join(r.cacheDir, modelID)
	}

	// Resolve to absolute and verify containment within cacheDir.
	cleaned := filepath.Clean(resolved)
	cachePrefix := filepath.Clean(r.cacheDir) + string(filepath.Separator)
	if !strings.HasPrefix(cleaned+string(filepath.Separator), cachePrefix) {
		return "", fmt.Errorf("model ID %q resolves outside cache directory", modelID)
	}
	return cleaned, nil
}

// writeModelInfo writes a config.json with model metadata.
func (r *LocalRegistry) writeModelInfo(dir string, info *ModelInfo) error {
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "config.json"), data, 0o600)
}

// readModelInfo reads a config.json with model metadata.
func (r *LocalRegistry) readModelInfo(dir string) (*ModelInfo, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json")) //nolint:gosec
	if err != nil {
		return nil, err
	}
	var info ModelInfo
	if err := json.Unmarshal(data, &info); err != nil {
		return nil, err
	}
	return &info, nil
}
