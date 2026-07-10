package cublas

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCublasLibPathsPreferAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md): the trusted absolute CUDA
// toolkit locations must always be tried before the documented bare-soname
// fallback.
func TestCublasLibPathsPreferAbsolute(t *testing.T) {
	if len(cublasLibPaths) < len(trustedCublasLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedCublasLibPaths), cublasLibPaths)
	}
	for i, want := range trustedCublasLibPaths {
		if cublasLibPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, cublasLibPaths)
		}
	}
	for _, p := range cublasLibPaths[len(trustedCublasLibPaths):] {
		if p == "./libcublas.so" || p == "." {
			t.Fatalf("cublasLibPaths fallback must not contain a CWD-relative entry: %q", p)
		}
	}
}

func TestBuildCublasLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libcublas.so.12")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(cublasLibPathOverrideEnv, path)
	paths := buildCublasLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildCublasLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(cublasLibPathOverrideEnv, "./libcublas.so")
	paths := buildCublasLibPaths()
	if len(paths) == 0 || paths[0] != trustedCublasLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}

// TestCublasLtLibPathsPreferAbsolute mirrors TestCublasLibPathsPreferAbsolute
// for cublasLt.
func TestCublasLtLibPathsPreferAbsolute(t *testing.T) {
	if len(cublasLtLibPaths) < len(trustedCublasLtLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedCublasLtLibPaths), cublasLtLibPaths)
	}
	for i, want := range trustedCublasLtLibPaths {
		if cublasLtLibPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, cublasLtLibPaths)
		}
	}
}

func TestBuildCublasLtLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libcublasLt.so.12")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(cublasLtLibPathOverrideEnv, path)
	paths := buildCublasLtLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildCublasLtLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(cublasLtLibPathOverrideEnv, "./libcublasLt.so")
	paths := buildCublasLtLibPaths()
	if len(paths) == 0 || paths[0] != trustedCublasLtLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
