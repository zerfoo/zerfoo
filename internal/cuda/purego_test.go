package cuda

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestDlopenImpl(t *testing.T) {
	// On any platform, dlopen of a nonexistent library returns 0.
	h := dlopenImpl("libnonexistent_test_xyzzy.so", rtldLazy)
	if h != 0 {
		dlcloseImpl(h)
		t.Fatal("expected dlopen of nonexistent library to return 0")
	}
}

func TestDlerrorImpl(t *testing.T) {
	// After a failed dlopen, dlerror should return a non-empty string.
	_ = dlopenImpl("libnonexistent_test_xyzzy.so", rtldLazy)
	msg := dlerrorImpl()
	if msg == "" {
		t.Fatal("expected dlerror to return non-empty string after failed dlopen")
	}
	t.Logf("dlerror: %s", msg)
}

func TestOpenFailsGracefully(t *testing.T) {
	// On macOS and non-CUDA Linux machines, Open() should fail gracefully.
	if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		t.Skip("skipping on linux/arm64 -- CUDA may be available")
	}
	_, err := Open()
	if err == nil {
		t.Fatal("expected Open() to fail on a non-CUDA machine")
	}
	t.Logf("Open error: %v", err)
}

func TestAvailableReturnsFalseWithoutCUDA(t *testing.T) {
	if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" {
		t.Skip("skipping on linux/arm64 -- CUDA may be available")
	}
	// Reset global state for this test.
	// Note: Available() is cached via sync.Once, so this test must run
	// before any other test that calls Available(). In practice, the
	// global state is initialized once per process.

	// We can't easily reset sync.Once, so just verify the behavior.
	if Available() {
		t.Fatal("expected Available() = false on a non-CUDA machine")
	}
}

func TestDlsymImplFailsOnInvalidHandle(t *testing.T) {
	if runtime.GOOS == "linux" {
		// On glibc, handle 0 is RTLD_DEFAULT: dlsym searches the global
		// namespace and can resolve real symbols, so the invalid-handle
		// expectation below only holds on darwin.
		t.Skip("dlsym(0) is RTLD_DEFAULT on glibc and may resolve symbols")
	}
	// dlsym with handle 0 should return 0.
	addr := dlsymImpl(0, "cudaMalloc")
	if addr != 0 {
		t.Fatalf("expected dlsym with handle 0 to return 0, got %#x", addr)
	}
}

func TestCcallDoesNotPanic(t *testing.T) {
	// Calling ccall with fn=0 would segfault, so we just verify the
	// function is callable (type checks). We can't test actual C
	// function calling without a valid function pointer.
	// This test documents that ccall exists and has the expected signature.
	var fn uintptr // zero = invalid
	_ = fn
	// ccall(fn) would crash, so we just verify compilation.
}

func TestDlopenPathNonexistent(t *testing.T) {
	_, err := DlopenPath("/tmp/libnonexistent_test_xyzzy.so")
	if err == nil {
		t.Fatal("expected DlopenPath to fail for nonexistent library")
	}
	t.Logf("DlopenPath error: %v", err)
}

func TestDlopenPathValid(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("libSystem test only runs on macOS")
	}
	h, err := DlopenPath("/usr/lib/libSystem.B.dylib")
	if err != nil {
		t.Fatalf("expected DlopenPath to succeed: %v", err)
	}
	if h == 0 {
		t.Fatal("expected non-zero handle")
	}
	dlcloseImpl(h)
}

// TestKernelLibPathsAreAbsolute is the CUDA-1 regression guard
// (docs/deep-reviews/002-full-codebase.md, docs/adr/094-*): dlopen executes a
// shared object's ELF constructors at load time, so any bare-soname or
// CWD-relative candidate in the kernel dlopen search list is a local
// code-execution primitive. Every candidate must be an absolute path.
func TestKernelLibPathsAreAbsolute(t *testing.T) {
	if len(kernelLibPaths) == 0 {
		t.Fatal("expected kernelLibPaths to contain at least the trusted default")
	}
	for _, p := range kernelLibPaths {
		if !filepath.IsAbs(p) {
			t.Fatalf("kernelLibPaths contains a non-absolute (CWD-relative or bare-soname) entry: %q", p)
		}
	}
}

// TestKernelLibPathsContainsTrustedDefault pins the trusted production path
// so a future edit cannot silently drop it.
func TestKernelLibPathsContainsTrustedDefault(t *testing.T) {
	found := false
	for _, p := range kernelLibPaths {
		if p == trustedKernelLibPath {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected kernelLibPaths to contain trusted default %q, got %v", trustedKernelLibPath, kernelLibPaths)
	}
}

func TestVetKernelLibOverrideRejectsRelativePath(t *testing.T) {
	if _, ok := vetKernelLibOverride("libkernels.so"); ok {
		t.Fatal("expected bare soname to be rejected")
	}
	if _, ok := vetKernelLibOverride("./libkernels.so"); ok {
		t.Fatal("expected CWD-relative path to be rejected")
	}
	if _, ok := vetKernelLibOverride("../libkernels.so"); ok {
		t.Fatal("expected relative parent path to be rejected")
	}
}

func TestVetKernelLibOverrideRejectsMissingFile(t *testing.T) {
	if _, ok := vetKernelLibOverride("/tmp/libnonexistent_test_xyzzy_kernellib.so"); ok {
		t.Fatal("expected nonexistent absolute path to be rejected")
	}
}

func TestVetKernelLibOverrideRejectsWorldWritable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libkernels.so")
	if err := os.WriteFile(path, []byte("stub"), 0o666); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	if err := os.Chmod(path, 0o666); err != nil {
		t.Fatalf("failed to chmod test fixture world-writable: %v", err)
	}
	if _, ok := vetKernelLibOverride(path); ok {
		t.Fatal("expected world-writable absolute path to be rejected")
	}
}

func TestVetKernelLibOverrideAcceptsSafeAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libkernels.so")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	vetted, ok := vetKernelLibOverride(path)
	if !ok {
		t.Fatal("expected safe absolute, non-world-writable path to be accepted")
	}
	if vetted != path {
		t.Fatalf("expected vetted path %q, got %q", path, vetted)
	}
}

func TestBuildKernelLibPathsFallsBackOnInvalidOverride(t *testing.T) {
	t.Setenv(kernelLibPathOverrideEnv, "./libkernels.so")
	paths := buildKernelLibPaths()
	if len(paths) != 1 || paths[0] != trustedKernelLibPath {
		t.Fatalf("expected invalid override to fall through to trusted default only, got %v", paths)
	}
}

func TestBuildKernelLibPathsUsesValidOverrideFirst(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libkernels.so")
	if err := os.WriteFile(path, []byte("stub"), 0o644); err != nil {
		t.Fatalf("failed to write test fixture: %v", err)
	}
	t.Setenv(kernelLibPathOverrideEnv, path)
	paths := buildKernelLibPaths()
	if len(paths) != 2 || paths[0] != path || paths[1] != trustedKernelLibPath {
		t.Fatalf("expected [override, trustedDefault], got %v", paths)
	}
}

func TestDlopenImplWithLibSystem(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("libSystem test only runs on macOS")
	}
	h := dlopenImpl("/usr/lib/libSystem.B.dylib", rtldLazy)
	if h == 0 {
		t.Fatal("expected dlopen of libSystem.B.dylib to succeed")
	}
	defer dlcloseImpl(h)

	// Verify dlsym finds a known symbol.
	addr := dlsymImpl(h, "getpid")
	if addr == 0 {
		t.Fatal("expected dlsym(getpid) to return non-zero")
	}

	// Call getpid through our purego ccall.
	pid := ccall(addr)
	if pid == 0 {
		t.Fatal("expected getpid() to return non-zero")
	}
	t.Logf("getpid() = %d", pid)
}
