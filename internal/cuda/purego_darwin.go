//go:build darwin

package cuda

import (
	"unsafe"
	_ "unsafe"
)

// On Darwin, we call C library functions in libSystem via syscall.syscall6
// and syscall.syscall9. These do NOT go through runtime.cgocall, so they are
// true zero-CGo calls.
//
// dlopen/dlsym/dlclose/dlerror are imported dynamically from libSystem.B.dylib
// and reached through assembly JMP trampolines. Crucially, syscall.syscall6
// must be handed the raw ABI0 entry point of each trampoline. We obtain that
// address from an assembly DATA directive (see purego_darwin_{amd64,arm64}.s),
// mirroring the golang.org/x/sys/unix darwin idiom.
//
// The previous implementation derived the trampoline address by taking the
// trampoline as a func() value and double-dereferencing it (funcPC). On
// darwin/amd64 that yields the compiler-generated ABIInternal wrapper, not the
// ABI0 trampoline, so syscall.syscall6 transferred control to a bad PC and the
// process SIGSEGV'd inside dlopen during package init (zerfoo T137.1, #171).

//go:linkname syscall_syscall6 syscall.syscall6
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err uintptr)

//go:linkname syscall_syscall9 syscall.syscall9
func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err uintptr)

// Raw ABI0 entry points of the assembly JMP trampolines. Each is populated by
// a DATA directive in the matching per-arch assembly file. These must be the
// trampoline addresses themselves, not func-value PCs.
var (
	libc_dlopen_trampoline_addr  uintptr
	libc_dlsym_trampoline_addr   uintptr
	libc_dlclose_trampoline_addr uintptr
	libc_dlerror_trampoline_addr uintptr
)

//go:cgo_import_dynamic libc_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlsym dlsym "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlclose dlclose "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlerror dlerror "/usr/lib/libSystem.B.dylib"

func dlopenImpl(path string, mode int) uintptr {
	p := append([]byte(path), 0)
	r1, _, _ := syscall_syscall6(
		libc_dlopen_trampoline_addr,
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(mode), 0, 0, 0, 0,
	)
	return r1
}

func dlsymImpl(handle uintptr, name string) uintptr {
	n := append([]byte(name), 0)
	r1, _, _ := syscall_syscall6(
		libc_dlsym_trampoline_addr,
		handle,
		uintptr(unsafe.Pointer(&n[0])),
		0, 0, 0, 0,
	)
	return r1
}

func dlcloseImpl(handle uintptr) int {
	r1, _, _ := syscall_syscall6(
		libc_dlclose_trampoline_addr,
		handle, 0, 0, 0, 0, 0,
	)
	return int(r1)
}

func dlerrorImpl() string {
	r1, _, _ := syscall_syscall6(
		libc_dlerror_trampoline_addr,
		0, 0, 0, 0, 0, 0,
	)
	if r1 == 0 {
		return ""
	}
	// r1 is a C string pointer.
	return goString(r1)
}

// goString converts a C string (null-terminated) to a Go string.
//
//go:nosplit
//go:nocheckptr
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	// #nosec G103 -- converting C string pointer from dlopen/dlerror
	ptr := (*byte)(ptrFromUintptr(p))
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// ccall calls a C function pointer with up to 9 arguments.
// On Darwin, this uses syscall.syscall9 which does not go through cgocall.
// For functions with more than 9 args (e.g. broadcast kernels with 10),
// a platform-specific extension is needed. Since CUDA is not available
// on macOS, this limit is acceptable for the dev machine.
func ccall(fn uintptr, args ...uintptr) uintptr {
	var a [9]uintptr
	copy(a[:], args)
	r1, _, _ := syscall_syscall9(fn, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
	return r1
}
