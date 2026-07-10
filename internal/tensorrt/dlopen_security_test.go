package tensorrt

import (
	"os"
	"path/filepath"
	"testing"
)

// TestTrtLibPathsAreAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md, mirroring CUDA-1's
// TestKernelLibPathsAreAbsolute in internal/cuda/purego_test.go): the
// TensorRT C shim is a zerfoo-built artifact with exactly one vetted
// install location, like libkernels.so, so every dlopen candidate must be
// an absolute path -- never a bare soname or CWD-relative entry.
func TestTrtLibPathsAreAbsolute(t *testing.T) {
	if len(trtLibPaths) == 0 {
		t.Fatal("expected trtLibPaths to contain at least the trusted default")
	}
	for _, p := range trtLibPaths {
		if !filepath.IsAbs(p) {
			t.Fatalf("trtLibPaths contains a non-absolute (CWD-relative or bare-soname) entry: %q", p)
		}
	}
}

func TestTrtLibPathsContainsTrustedDefault(t *testing.T) {
	found := false
	for _, p := range trtLibPaths {
		if p == trustedTrtCapiLibPath {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected trtLibPaths to contain trusted default %q, got %v", trustedTrtCapiLibPath, trtLibPaths)
	}
}

func TestBuildTrtLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(trtCapiLibPathOverrideEnv, "./libtrt_capi.so")
	paths := buildTrtLibPaths()
	if len(paths) != 1 || paths[0] != trustedTrtCapiLibPath {
		t.Fatalf("expected invalid override to fall through to trusted default only, got %v", paths)
	}
}

func TestBuildTrtLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libtrt_capi.so")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(trtCapiLibPathOverrideEnv, path)
	paths := buildTrtLibPaths()
	if len(paths) != 2 || paths[0] != path || paths[1] != trustedTrtCapiLibPath {
		t.Fatalf("expected [override, trustedDefault], got %v", paths)
	}
}
