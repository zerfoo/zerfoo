package codegen

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// NvccPath returns the path to the nvcc compiler, or an error if not found.
func NvccPath() (string, error) {
	// Check common locations first.
	paths := []string{
		"/usr/local/cuda/bin/nvcc",
		"/usr/bin/nvcc",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	// Fall back to PATH lookup.
	p, err := exec.LookPath("nvcc")
	if err != nil {
		return "", fmt.Errorf("nvcc not found: %w", err)
	}
	return p, nil
}

// gpuArch returns the nvcc -arch flag for the current platform.
// Falls back to a safe default if detection fails.
func gpuArch() string {
	if runtime.GOARCH == "arm64" {
		return "sm_121" // DGX Spark GB10 (Blackwell)
	}
	return "sm_80" // safe default for Ampere
}

// sourceHash returns a hex-encoded SHA-256 of the CUDA source.
func sourceHash(source string) string {
	h := sha256.Sum256([]byte(source))
	return fmt.Sprintf("%x", h)
}

// CachedCompile compiles a CUDA source string to a shared library (.so),
// caching the result alongside the model. If the cached .so exists and
// the source hash matches, the cached version is returned immediately.
//
// Parameters:
//   - source: the CUDA source code string
//   - cacheDir: directory for the cached .so and hash file
//   - modelName: base name for the output files
//
// Returns the path to the compiled .so file.
func CachedCompile(source, cacheDir, modelName string) (string, error) {
	soName := modelName + ".megakernel.so"
	hashName := modelName + ".megakernel.hash"
	soPath := filepath.Join(cacheDir, soName)
	hashPath := filepath.Join(cacheDir, hashName)

	currentHash := sourceHash(source)

	// Check cache: if .so exists and hash matches, skip compilation.
	if existingHash, err := os.ReadFile(hashPath); err == nil { //nolint:gosec // path is constructed from known cacheDir+modelName
		if string(existingHash) == currentHash {
			if _, err := os.Stat(soPath); err == nil {
				return soPath, nil
			}
		}
	}

	// Find nvcc.
	nvcc, err := NvccPath()
	if err != nil {
		return "", err
	}

	// Write source to temp file.
	cuPath := filepath.Join(cacheDir, modelName+".megakernel.cu")
	if err := os.WriteFile(cuPath, []byte(source), 0o600); err != nil {
		return "", fmt.Errorf("write .cu: %w", err)
	}

	// Compile to shared library.
	arch := gpuArch()
	ctx := context.Background()
	cmd := exec.CommandContext(ctx, nvcc, //nolint:gosec // nvcc path is from known locations
		"-arch="+arch,
		"--shared",
		"-o", soPath,
		cuPath,
	)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("nvcc compilation failed: %w", err)
	}

	// Write hash for cache validation.
	if err := os.WriteFile(hashPath, []byte(currentHash), 0o600); err != nil {
		return "", fmt.Errorf("write hash: %w", err)
	}

	return soPath, nil
}
