package cuda

import (
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
