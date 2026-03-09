//go:build !cuda && darwin

package cuda

import (
	"unsafe"
	_ "unsafe"
)

// On Darwin, we use syscall.syscall6 and syscall.syscall9 to call C
// library functions via libSystem. These do NOT go through runtime.cgocall
// so they are true zero-CGo calls.
//
// For dlopen/dlsym, we import them dynamically from libSystem.B.dylib
// and call through assembly trampolines (defined in purego_darwin_amd64.s
// or purego_darwin_arm64.s).

//go:linkname syscall_syscall6 syscall.syscall6
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err uintptr)

//go:linkname syscall_syscall9 syscall.syscall9
func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err uintptr)

// Assembly trampolines for dlopen/dlsym/dlclose/dlerror.
// These are JMP stubs to the dynamically imported symbols.
func libc_dlopen_trampoline()
func libc_dlsym_trampoline()
func libc_dlclose_trampoline()
func libc_dlerror_trampoline()

//go:cgo_import_dynamic libc_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlsym dlsym "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlclose dlclose "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_dlerror dlerror "/usr/lib/libSystem.B.dylib"

//go:nosplit
func funcPC(fn func()) uintptr {
	return **(**uintptr)(unsafe.Pointer(&fn))
}

func dlopenImpl(path string, mode int) uintptr {
	p := append([]byte(path), 0)
	r1, _, _ := syscall_syscall6(
		funcPC(libc_dlopen_trampoline),
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(mode), 0, 0, 0, 0,
	)
	return r1
}

func dlsymImpl(handle uintptr, name string) uintptr {
	n := append([]byte(name), 0)
	r1, _, _ := syscall_syscall6(
		funcPC(libc_dlsym_trampoline),
		handle,
		uintptr(unsafe.Pointer(&n[0])),
		0, 0, 0, 0,
	)
	return r1
}

func dlcloseImpl(handle uintptr) int {
	r1, _, _ := syscall_syscall6(
		funcPC(libc_dlclose_trampoline),
		handle, 0, 0, 0, 0, 0,
	)
	return int(r1)
}

func dlerrorImpl() string {
	r1, _, _ := syscall_syscall6(
		funcPC(libc_dlerror_trampoline),
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
	ptr := (*byte)(unsafe.Pointer(p)) //nolint:govet
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
