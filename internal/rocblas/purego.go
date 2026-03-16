package rocblas

import (
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// RocBLASLib holds dlopen handles and resolved function pointers for
// rocBLAS functions. All function pointers are resolved at Open()
// time via dlsym. Calls go through cuda.Ccall which uses the
// platform-specific zero-CGo mechanism.
type RocBLASLib struct {
	handle uintptr // dlopen handle for librocblas

	// rocBLAS function pointers
	rocblasCreateHandle  uintptr
	rocblasDestroyHandle uintptr
	rocblasSetStream     uintptr
	rocblasSgemm         uintptr
}

var (
	globalLib  *RocBLASLib
	globalOnce sync.Once
	errGlobal  error
)

// rocblasPaths lists the shared library names to try, in order.
var rocblasPaths = []string{
	"librocblas.so.4",
	"librocblas.so",
}

// Open loads librocblas via dlopen and resolves all rocBLAS function
// pointers via dlsym. Returns an error if rocBLAS is not available.
func Open() (*RocBLASLib, error) {
	lib := &RocBLASLib{}

	var lastErr error
	for _, path := range rocblasPaths {
		h, err := cuda.DlopenPath(path)
		if err == nil {
			lib.handle = h
			break
		}
		lastErr = err
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("rocblas: dlopen librocblas failed: %w", lastErr)
	}

	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"rocblas_create_handle", &lib.rocblasCreateHandle},
		{"rocblas_destroy_handle", &lib.rocblasDestroyHandle},
		{"rocblas_set_stream", &lib.rocblasSetStream},
		{"rocblas_sgemm", &lib.rocblasSgemm},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("rocblas: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if rocBLAS is loadable on this machine.
// The result is cached after the first call.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global RocBLASLib instance, or nil if rocBLAS is not available.
func Lib() *RocBLASLib {
	if !Available() {
		return nil
	}
	return globalLib
}
