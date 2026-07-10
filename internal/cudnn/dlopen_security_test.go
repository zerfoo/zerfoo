package cudnn

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCudnnPathsPreferAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md): the trusted absolute CUDA
// toolkit locations must always be tried before the documented bare-soname
// fallback.
func TestCudnnPathsPreferAbsolute(t *testing.T) {
	if len(cudnnPaths) < len(trustedCudnnLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedCudnnLibPaths), cudnnPaths)
	}
	for i, want := range trustedCudnnLibPaths {
		if cudnnPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, cudnnPaths)
		}
	}
	for _, p := range cudnnPaths[len(trustedCudnnLibPaths):] {
		if p == "./libcudnn.so" || p == "." {
			t.Fatalf("cudnnPaths fallback must not contain a CWD-relative entry: %q", p)
		}
	}
}

func TestBuildCudnnLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libcudnn.so.9")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(cudnnLibPathOverrideEnv, path)
	paths := buildCudnnLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildCudnnLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(cudnnLibPathOverrideEnv, "./libcudnn.so")
	paths := buildCudnnLibPaths()
	if len(paths) == 0 || paths[0] != trustedCudnnLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
