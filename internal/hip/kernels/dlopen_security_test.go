package kernels

import (
	"os"
	"path/filepath"
	"testing"
)

// TestHipKernelPathsAreAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md, mirroring CUDA-1's
// TestKernelLibPathsAreAbsolute in internal/cuda/purego_test.go): the HIP
// kernel shim is a zerfoo-built artifact with exactly one vetted install
// location, like libkernels.so, so every dlopen candidate must be an
// absolute path -- never a bare soname or CWD-relative entry.
func TestHipKernelPathsAreAbsolute(t *testing.T) {
	if len(hipKernelPaths) == 0 {
		t.Fatal("expected hipKernelPaths to contain at least the trusted default")
	}
	for _, p := range hipKernelPaths {
		if !filepath.IsAbs(p) {
			t.Fatalf("hipKernelPaths contains a non-absolute (CWD-relative or bare-soname) entry: %q", p)
		}
	}
}

func TestHipKernelPathsContainsTrustedDefault(t *testing.T) {
	found := false
	for _, p := range hipKernelPaths {
		if p == trustedHipKernelLibPath {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected hipKernelPaths to contain trusted default %q, got %v", trustedHipKernelLibPath, hipKernelPaths)
	}
}

func TestBuildHipKernelLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(hipKernelLibPathOverrideEnv, "./libhipkernels.so")
	paths := buildHipKernelLibPaths()
	if len(paths) != 1 || paths[0] != trustedHipKernelLibPath {
		t.Fatalf("expected invalid override to fall through to trusted default only, got %v", paths)
	}
}

func TestBuildHipKernelLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libhipkernels.so")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(hipKernelLibPathOverrideEnv, path)
	paths := buildHipKernelLibPaths()
	if len(paths) != 2 || paths[0] != path || paths[1] != trustedHipKernelLibPath {
		t.Fatalf("expected [override, trustedDefault], got %v", paths)
	}
}
