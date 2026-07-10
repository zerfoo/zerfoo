package miopen

import (
	"os"
	"path/filepath"
	"testing"
)

// TestMiopenPathsPreferAbsolute is the CUDA-2 regression guard
// (docs/deep-reviews/002-full-codebase.md): the trusted absolute ROCm
// install locations must always be tried before the documented bare-soname
// fallback.
func TestMiopenPathsPreferAbsolute(t *testing.T) {
	if len(miopenPaths) < len(trustedMiopenLibPaths) {
		t.Fatalf("expected at least %d candidates, got %v", len(trustedMiopenLibPaths), miopenPaths)
	}
	for i, want := range trustedMiopenLibPaths {
		if miopenPaths[i] != want {
			t.Fatalf("expected trusted absolute path %q at index %d, got %v", want, i, miopenPaths)
		}
	}
	for _, p := range miopenPaths[len(trustedMiopenLibPaths):] {
		if p == "./libMIOpen.so" || p == "." {
			t.Fatalf("miopenPaths fallback must not contain a CWD-relative entry: %q", p)
		}
	}
}

func TestBuildMiopenLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libMIOpen.so.1")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(miopenLibPathOverrideEnv, path)
	paths := buildMiopenLibPaths()
	if len(paths) == 0 || paths[0] != path {
		t.Fatalf("expected override %q first, got %v", path, paths)
	}
}

func TestBuildMiopenLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(miopenLibPathOverrideEnv, "./libMIOpen.so")
	paths := buildMiopenLibPaths()
	if len(paths) == 0 || paths[0] != trustedMiopenLibPaths[0] {
		t.Fatalf("expected invalid override to be dropped, got %v", paths)
	}
}
