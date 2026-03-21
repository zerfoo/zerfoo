//go:build linux && arm64 && !cgo

package cuda

import (
	"unsafe"
	_ "unsafe"
)

// On Linux arm64, the Go linker rejects SDYNIMPORT relocations from
// //go:cgo_import_dynamic when used with assembly JMP trampolines.
// Instead, we resolve dlopen/dlsym/dlclose/dlerror at runtime by
// loading libdl.so.2 through Go's internal dynamic linker support.
//
// This approach uses runtime.asmcgocall to call C functions on the
// system stack without any CGo overhead. The key difference from the
// broken approach is that we resolve dlopen/dlsym function addresses
// at init time via runtime symbols, not via assembly JMP trampolines.

// asmcgocall calls fn(arg) on the system stack (g0 stack).
//
//go:nosplit
//go:linkname asmcgocall runtime.asmcgocall
func asmcgocall(fn, arg unsafe.Pointer) int32

// Runtime symbols for dynamic library loading.
// Go's runtime already resolves these during process startup.
//
//go:linkname runtime_dlopen runtime.dlopen
func runtime_dlopen(path *byte) uintptr

//go:linkname runtime_dlsym runtime.dlsym
func runtime_dlsym(handle uintptr, symbol *byte) uintptr

//go:linkname runtime_dlclose runtime.dlclose
func runtime_dlclose(handle uintptr)

// ccallTrampoline is the assembly-defined trampoline for calling
// arbitrary C function pointers with up to 14 arguments.
func ccallTrampoline()

//go:nosplit
func funcPC(fn func()) uintptr {
	return **(**uintptr)(unsafe.Pointer(&fn))
}

// ccallArgs is the argument frame passed to the assembly trampoline.
type ccallArgs struct {
	fn   uintptr
	args [20]uintptr
	ret  uintptr
}

//go:nosplit
func runTrampoline(args *ccallArgs) {
	fn := funcPC(ccallTrampoline)
	asmcgocall(
		*(*unsafe.Pointer)(unsafe.Pointer(&fn)),
		unsafe.Pointer(args),
	)
}

func cstring(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func dlopenImpl(path string, mode int) uintptr {
	// Use runtime.dlopen which is already resolved by Go's startup.
	// The mode parameter is ignored by runtime.dlopen (it uses RTLD_LAZY).
	return runtime_dlopen(cstring(path))
}

func dlsymImpl(handle uintptr, name string) uintptr {
	return runtime_dlsym(handle, cstring(name))
}

func dlcloseImpl(handle uintptr) int {
	runtime_dlclose(handle)
	return 0
}

func dlerrorImpl() string {
	// runtime.dlopen/dlsym don't expose dlerror.
	// Return empty string; callers check the return value (0 = error).
	return "dlopen/dlsym failed"
}

// goString converts a C string (null-terminated) to a Go string.
//
//go:nosplit
//go:nocheckptr
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	ptr := (*byte)(unsafe.Pointer(p))
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// ccall calls a C function pointer with up to 14 arguments.
func ccall(fn uintptr, a ...uintptr) uintptr {
	var args ccallArgs
	args.fn = fn
	copy(args.args[:], a)
	runTrampoline(&args)
	return args.ret
}
