package rocblas

import (
	"fmt"
	"os"
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

// rocblasLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedRocblasLibPaths. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const rocblasLibPathOverrideEnv = "ZERFOO_ROCBLAS_LIB_PATH"

// trustedRocblasLibPaths use /opt/rocm/lib, AMD's standard ROCm install
// prefix (see the identical rationale on trustedHipLibPaths in
// internal/hip/purego.go).
var trustedRocblasLibPaths = []string{
	"/opt/rocm/lib/librocblas.so.4",
	"/opt/rocm/lib/librocblas.so",
}

// rocblasPaths lists the shared library candidates to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a DOCUMENTED residual trust assumption -- see the
// equivalent comment on hipPaths in internal/hip/purego.go for the full
// rationale.
var rocblasPaths = buildRocblasLibPaths()

func buildRocblasLibPaths() []string {
	paths := make([]string, 0, len(trustedRocblasLibPaths)+3)
	if override := os.Getenv(rocblasLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedRocblasLibPaths...)
	paths = append(paths, "librocblas.so.4", "librocblas.so")
	return paths
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
