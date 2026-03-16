package inference

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// trtCacheDirOverride allows tests to redirect the cache directory.
var trtCacheDirOverride string

// trtCacheDir returns the TensorRT engine cache directory.
func trtCacheDir() string {
	if trtCacheDirOverride != "" {
		return trtCacheDirOverride
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "zerfoo", "tensorrt")
	}
	return filepath.Join(home, ".cache", "zerfoo", "tensorrt")
}

// TRTCacheKey builds a deterministic cache key from model ID, precision, and GPU
// architecture. The key is a hex SHA-256 hash to avoid filesystem issues with
// long or special-character model IDs.
func TRTCacheKey(modelID, precision string) (string, error) {
	arch, err := gpuArchString()
	if err != nil {
		return "", fmt.Errorf("tensorrt cache: %w", err)
	}
	raw := fmt.Sprintf("%s|%s|%s", modelID, precision, arch)
	hash := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", hash[:16]), nil
}

// SaveTRTEngine writes a serialized TensorRT engine to the cache directory.
func SaveTRTEngine(key string, data []byte) error {
	dir := trtCacheDir()
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return fmt.Errorf("tensorrt cache: mkdir: %w", err)
	}
	path := filepath.Join(dir, key+".engine")
	return os.WriteFile(path, data, 0o600)
}

// LoadTRTEngine reads a serialized TensorRT engine from the cache.
// Returns nil, nil on cache miss (file not found).
func LoadTRTEngine(key string) ([]byte, error) {
	path := filepath.Join(trtCacheDir(), key+".engine")
	data, err := os.ReadFile(path) //nolint:gosec // path is constructed from cache dir + sanitized key
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("tensorrt cache: read: %w", err)
	}
	return data, nil
}

// gpuArchString returns a string identifying the GPU architecture, e.g., "sm_75".
func gpuArchString() (string, error) {
	major, minor, err := cuda.DeviceComputeCapability(0)
	if err != nil {
		return "", fmt.Errorf("get compute capability: %w", err)
	}
	return fmt.Sprintf("sm_%d%d", major, minor), nil
}
