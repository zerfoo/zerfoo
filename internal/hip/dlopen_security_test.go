package hip

import (
	"os"
	"path/filepath"
	"testing"
)

// TestHipPathsPreferAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md): the trusted absolute ROCm
// install locations must always be tried before the documented bare-soname
// fallback.
func TestHipPathsPreferAbsolute(t *testing.T) {
	if len(hipPaths) < len(trustedHipLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedHipLibPaths), hipPaths)
	}
	for i, want := range trustedHipLibPaths {
		if hipPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, hipPaths)
		}
	}
	for _, p := range hipPaths[len(trustedHipLibPaths):] {
		if p == "./libamdhip64.so" || p == "." {
			t.Fatalf("hipPaths fallback must not contain a CWD-relative entry: %q", p)
		}
	}
}

func TestBuildHipLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libamdhip64.so.6")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(hipLibPathOverrideEnv, path)
	paths := buildHipLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildHipLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(hipLibPathOverrideEnv, "./libamdhip64.so")
	paths := buildHipLibPaths()
	if len(paths) == 0 || paths[0] != trustedHipLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
