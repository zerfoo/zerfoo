//go:build linux && arm64 && cuda

package cuda

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>

// ccall_wrapper calls a C function pointer with up to 20 arguments.
static uintptr_t ccall_wrapper(
	uintptr_t fn,
	uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3,
	uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7,
	uintptr_t a8, uintptr_t a9, uintptr_t a10, uintptr_t a11,
	uintptr_t a12, uintptr_t a13, uintptr_t a14, uintptr_t a15,
	uintptr_t a16, uintptr_t a17, uintptr_t a18, uintptr_t a19
) {
	typedef uintptr_t (*fn_t)(
		uintptr_t, uintptr_t, uintptr_t, uintptr_t,
		uintptr_t, uintptr_t, uintptr_t, uintptr_t,
		uintptr_t, uintptr_t, uintptr_t, uintptr_t,
		uintptr_t, uintptr_t, uintptr_t, uintptr_t,
		uintptr_t, uintptr_t, uintptr_t, uintptr_t
	);
	return ((fn_t)fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19);
}
*/
import "C"

import "unsafe"

func dlopenImpl(path string, mode int) uintptr {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	h := C.dlopen(cpath, C.int(mode))
	return uintptr(h)
}

func dlsymImpl(handle uintptr, name string) uintptr {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	p := C.dlsym(unsafe.Pointer(handle), cname) //nolint:govet
	return uintptr(p)
}

func dlcloseImpl(handle uintptr) int {
	return int(C.dlclose(unsafe.Pointer(handle))) //nolint:govet
}

func dlerrorImpl() string {
	p := C.dlerror()
	if p == nil {
		return ""
	}
	return C.GoString(p)
}

// ccall calls a C function pointer with up to 20 arguments via CGo.
func ccall(fn uintptr, a ...uintptr) uintptr {
	var args [20]uintptr
	copy(args[:], a)
	return uintptr(C.ccall_wrapper(
		C.uintptr_t(fn),
		C.uintptr_t(args[0]), C.uintptr_t(args[1]), C.uintptr_t(args[2]), C.uintptr_t(args[3]),
		C.uintptr_t(args[4]), C.uintptr_t(args[5]), C.uintptr_t(args[6]), C.uintptr_t(args[7]),
		C.uintptr_t(args[8]), C.uintptr_t(args[9]), C.uintptr_t(args[10]), C.uintptr_t(args[11]),
		C.uintptr_t(args[12]), C.uintptr_t(args[13]), C.uintptr_t(args[14]), C.uintptr_t(args[15]),
		C.uintptr_t(args[16]), C.uintptr_t(args[17]), C.uintptr_t(args[18]), C.uintptr_t(args[19]),
	))
}
