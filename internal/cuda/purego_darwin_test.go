//go:build darwin

package cuda

import "testing"

// These tests guard the darwin dlopen probe path against the regression fixed
// in zerfoo T137.1 / ztensor#171: syscall.syscall6 was handed a func-value PC
// (via funcPC) instead of the raw ABI0 trampoline entry, so probing CUDA at
// package-init time SIGSEGV'd and took down every test binary importing the
// GPU path (e.g. internal/gpuapi.init -> cublas -> cuda.DlopenPath). If the
// mechanism regresses, this test binary crashes at startup and CI on darwin
// goes red.

// TestDarwinTrampolineAddrsResolved verifies the assembly DATA directives
// populated each trampoline address. A zero here means the linker did not wire
// the raw ABI0 entry point, which is the precise defect behind the crash.
func TestDarwinTrampolineAddrsResolved(t *testing.T) {
	cases := []struct {
		name string
		addr uintptr
	}{
		{"dlopen", libc_dlopen_trampoline_addr},
		{"dlsym", libc_dlsym_trampoline_addr},
		{"dlclose", libc_dlclose_trampoline_addr},
		{"dlerror", libc_dlerror_trampoline_addr},
	}
	for _, tc := range cases {
		if tc.addr == 0 {
			t.Errorf("%s trampoline address is 0; DATA directive did not resolve", tc.name)
		}
	}
}

// TestDarwinProbeDoesNotCrash exercises the init-time probe path. On a macOS
// host without CUDA it must return a clean "not available" error rather than
// crash the process.
func TestDarwinProbeDoesNotCrash(t *testing.T) {
	count, err := GetDeviceCount()
	if err == nil {
		t.Fatalf("expected GetDeviceCount to report CUDA unavailable on darwin, got count=%d", count)
	}
	if count != 0 {
		t.Fatalf("expected 0 devices on darwin, got %d", count)
	}
	if Available() {
		t.Fatal("expected Available() = false on darwin without CUDA")
	}
}

// TestDarwinDlopenLibSystem confirms the corrected call mechanism actually
// works: dlopen resolves a real library, dlsym finds a symbol, and the symbol
// is callable through the zero-CGo ccall path.
func TestDarwinDlopenLibSystem(t *testing.T) {
	h := dlopenImpl("/usr/lib/libSystem.B.dylib", rtldLazy)
	if h == 0 {
		t.Fatalf("dlopen(libSystem.B.dylib) failed: %s", dlerrorImpl())
	}
	defer dlcloseImpl(h)

	getpid := dlsymImpl(h, "getpid")
	if getpid == 0 {
		t.Fatalf("dlsym(getpid) failed: %s", dlerrorImpl())
	}
	if pid := ccall(getpid); pid == 0 {
		t.Fatal("expected getpid() to return a non-zero pid")
	}
}
