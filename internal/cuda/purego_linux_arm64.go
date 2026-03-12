//go:build linux && arm64

package cuda

import (
	"unsafe"
	_ "unsafe"
)

// On Linux arm64, we use runtime.asmcgocall to call C functions on the
// system stack. This bypasses runtime.cgocall (the CGo counter is never
// incremented), giving us true zero-CGo function calls.
//
// For dlopen/dlsym, we use //go:cgo_import_dynamic to import them from
// libdl.so.2 (or libc.so.6 on glibc 2.34+). Assembly trampolines
// (in purego_linux_arm64.s) provide call stubs.

// asmcgocall calls fn(arg) on the system stack (g0 stack).
// This is the same mechanism CGo uses internally, but calling it
// directly bypasses the CGo overhead tracking.
//
//go:nosplit
//go:linkname asmcgocall runtime.asmcgocall
func asmcgocall(fn, arg unsafe.Pointer) int32

// Assembly-defined trampoline functions for dlopen/dlsym/dlclose/dlerror.
func libc_dlopen_trampoline()
func libc_dlsym_trampoline()
func libc_dlclose_trampoline()
func libc_dlerror_trampoline()

// Assembly-defined trampoline for calling arbitrary C function pointers
// with up to 12 arguments (8 register + 4 stack on AAPCS64).
func ccallTrampoline()

//go:cgo_import_dynamic libc_dlopen dlopen "libdl.so.2"
//go:cgo_import_dynamic libc_dlsym dlsym "libdl.so.2"
//go:cgo_import_dynamic libc_dlclose dlclose "libdl.so.2"
//go:cgo_import_dynamic libc_dlerror dlerror "libdl.so.2"
//go:cgo_import_dynamic _ _ "libdl.so.2"

//go:nosplit
func funcPC(fn func()) uintptr {
	return **(**uintptr)(unsafe.Pointer(&fn))
}

// ccallArgs is the argument frame passed to the assembly trampoline.
// It must match the layout expected by ccallTrampoline in purego_linux_arm64.s.
type ccallArgs struct {
	fn   uintptr
	args [12]uintptr
	ret  uintptr
}

// runTrampoline calls the ccallTrampoline assembly function via asmcgocall.
// The unsafe.Pointer conversions are required by the purego calling convention:
// asmcgocall expects (fn unsafe.Pointer, arg unsafe.Pointer) where fn is the
// code address of an assembly function and arg is the data pointer passed in R0.
//
//go:nosplit
func runTrampoline(args *ccallArgs) {
	fn := funcPC(ccallTrampoline)
	asmcgocall(
		*(*unsafe.Pointer)(unsafe.Pointer(&fn)),
		unsafe.Pointer(args),
	)
}

func dlopenImpl(path string, mode int) uintptr {
	p := append([]byte(path), 0)
	var args ccallArgs
	args.fn = funcPC(libc_dlopen_trampoline)
	args.args[0] = uintptr(unsafe.Pointer(&p[0]))
	args.args[1] = uintptr(mode)
	runTrampoline(&args)
	return args.ret
}

func dlsymImpl(handle uintptr, name string) uintptr {
	n := append([]byte(name), 0)
	var args ccallArgs
	args.fn = funcPC(libc_dlsym_trampoline)
	args.args[0] = handle
	args.args[1] = uintptr(unsafe.Pointer(&n[0]))
	runTrampoline(&args)
	return args.ret
}

func dlcloseImpl(handle uintptr) int {
	var args ccallArgs
	args.fn = funcPC(libc_dlclose_trampoline)
	args.args[0] = handle
	runTrampoline(&args)
	return int(args.ret)
}

func dlerrorImpl() string {
	var args ccallArgs
	args.fn = funcPC(libc_dlerror_trampoline)
	runTrampoline(&args)
	if args.ret == 0 {
		return ""
	}
	return goString(args.ret)
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

// ccall calls a C function pointer with up to 12 arguments.
// Uses asmcgocall to run on the system stack (zero CGo overhead).
func ccall(fn uintptr, a ...uintptr) uintptr {
	var args ccallArgs
	args.fn = fn
	copy(args.args[:], a)
	runTrampoline(&args)
	return args.ret
}
