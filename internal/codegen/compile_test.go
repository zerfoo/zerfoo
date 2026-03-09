package codegen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCachedCompile(t *testing.T) {
	// Skip if nvcc is not available (only runs on machines with CUDA).
	if _, err := NvccPath(); err != nil {
		t.Skipf("nvcc not available: %v", err)
	}

	tmpDir := t.TempDir()
	cuSource := `
#include <cuda_runtime.h>
__global__ void test_kernel(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = in[tid] * 2.0f;
}
`

	// First compile: should create .so
	soPath, err := CachedCompile(cuSource, tmpDir, "test_model")
	if err != nil {
		t.Fatalf("CachedCompile: %v", err)
	}
	if soPath == "" {
		t.Fatal("soPath is empty")
	}
	if _, err := os.Stat(soPath); err != nil {
		t.Fatalf("compiled .so not found: %v", err)
	}

	// Verify hash file exists.
	hashPath := filepath.Join(tmpDir, "test_model.megakernel.hash")
	if _, err := os.Stat(hashPath); err != nil {
		t.Fatalf("hash file not found: %v", err)
	}

	// Second compile with same source: should use cache.
	soPath2, err := CachedCompile(cuSource, tmpDir, "test_model")
	if err != nil {
		t.Fatalf("CachedCompile (cached): %v", err)
	}
	if soPath2 != soPath {
		t.Errorf("cached path mismatch: %q vs %q", soPath2, soPath)
	}

	// Third compile with different source: should recompile.
	differentSource := cuSource + "\n// modified"
	soPath3, err := CachedCompile(differentSource, tmpDir, "test_model")
	if err != nil {
		t.Fatalf("CachedCompile (recompile): %v", err)
	}
	if soPath3 != soPath {
		t.Errorf("recompile path should be same location: %q vs %q", soPath3, soPath)
	}
}

func TestNvccPath(t *testing.T) {
	path, err := NvccPath()
	if err != nil {
		t.Skipf("nvcc not available: %v", err)
	}
	if !strings.Contains(path, "nvcc") {
		t.Errorf("unexpected nvcc path: %q", path)
	}
}
