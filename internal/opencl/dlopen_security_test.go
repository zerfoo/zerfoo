package opencl

import (
	"os"
	"path/filepath"
	"testing"
)

// TestOpenclPathsNoCWDRelativeEntry is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md). OpenCL has no trusted absolute
// default -- see the SECURITY comment on openclPaths for why the ICD
// design makes a bare soname the correct residual choice here -- but a
// CWD-relative entry would still be an unambiguous regression, so this
// guards against exactly that.
func TestOpenclPathsNoCWDRelativeEntry(t *testing.T) {
	if len(openclPaths) == 0 {
		t.Fatal("expected openclPaths to contain at least the bare-soname fallback")
	}
	for _, p := range openclPaths {
		if p == "./libOpenCL.so" || p == "." {
			t.Fatalf("openclPaths must not contain a CWD-relative entry: %q", p)
		}
		if filepath.IsAbs(p) {
			continue
		}
		if filepath.Dir(p) != "." {
			t.Fatalf("openclPaths entry %q resolves outside the bare-soname search path", p)
		}
	}
}

func TestBuildOpenclLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libOpenCL.so.1")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(openclLibPathOverrideEnv, path)
	paths := buildOpenclLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildOpenclLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(openclLibPathOverrideEnv, "./libOpenCL.so")
	paths := buildOpenclLibPaths()
	if len(paths) == 0 || paths[0] != "libOpenCL.so.1" {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
