package rocblas

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRocblasPathsPreferAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md): the trusted absolute ROCm
// install locations must always be tried before the documented bare-soname
// fallback.
func TestRocblasPathsPreferAbsolute(t *testing.T) {
	if len(rocblasPaths) < len(trustedRocblasLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedRocblasLibPaths), rocblasPaths)
	}
	for i, want := range trustedRocblasLibPaths {
		if rocblasPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, rocblasPaths)
		}
	}
	for _, p := range rocblasPaths[len(trustedRocblasLibPaths):] {
		if p == "./librocblas.so" || p == "." {
			t.Fatalf("rocblasPaths fallback must not contain a CWD-relative entry: %q", p)
		}
	}
}

func TestBuildRocblasLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "librocblas.so.4")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(rocblasLibPathOverrideEnv, path)
	paths := buildRocblasLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildRocblasLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(rocblasLibPathOverrideEnv, "./librocblas.so")
	paths := buildRocblasLibPaths()
	if len(paths) == 0 || paths[0] != trustedRocblasLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
