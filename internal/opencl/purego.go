package opencl

import (
	"fmt"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// OpenCLLib holds dlopen handles and resolved function pointers for
// OpenCL runtime functions. All function pointers are resolved at Open()
// time via dlsym. Calls go through the platform-specific ccall from the
// cuda package (which is a general-purpose zero-CGo C function caller,
// not CUDA-specific).
type OpenCLLib struct {
	handle uintptr

	// Platform/device discovery
	clGetPlatformIDs uintptr
	clGetDeviceIDs   uintptr

	// Context and command queue
	clCreateContext      uintptr
	clReleaseContext     uintptr
	clCreateCommandQueue uintptr
	clReleaseCommandQueue uintptr

	// Memory management
	clCreateBuffer    uintptr
	clReleaseMemObject uintptr

	// Data transfer
	clEnqueueWriteBuffer uintptr
	clEnqueueReadBuffer  uintptr
	clEnqueueCopyBuffer  uintptr

	// Synchronization
	clFinish uintptr
}

var (
	globalLib  *OpenCLLib
	globalOnce sync.Once
	errGlobal  error
)

// openclLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before the bare-soname candidates below.
// See cuda.VetAbsoluteLibPath for the safety checks applied.
const openclLibPathOverrideEnv = "ZERFOO_OPENCL_LIB_PATH"

// openclPaths lists the shared library candidates to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): unlike the
// CUDA/ROCm loaders above, this list has NO trusted absolute default, and
// that is a deliberate, documented decision rather than an oversight.
// OpenCL is architected around the system ICD loader: libOpenCL.so is a
// thin dispatcher that discovers vendor implementations at runtime via
// /etc/OpenCL/vendors/*.icd (an NVIDIA driver package, ocl-icd, an
// Intel/AMD vendor SDK, ...). There is no single "the OpenCL library" path
// that is correct across hosts the way /usr/local/cuda or /opt/rocm are for
// their ecosystems, so forcing an absolute path here would just be wrong on
// most machines rather than more secure. We accept the residual
// LD_LIBRARY_PATH/RPATH hijack exposure of the bare-soname resolution below
// as inherent to the ICD design; an operator who wants to pin a specific,
// vetted absolute build can still do so via openclLibPathOverrideEnv.
var openclPaths = buildOpenclLibPaths()

func buildOpenclLibPaths() []string {
	paths := make([]string, 0, 3)
	if override := os.Getenv(openclLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, "libOpenCL.so.1", "libOpenCL.so")
	return paths
}

// Open loads libOpenCL via dlopen and resolves all OpenCL runtime
// function pointers via dlsym. Returns an error if OpenCL is not
// available (library not found or symbols missing).
func Open() (*OpenCLLib, error) {
	lib := &OpenCLLib{}

	var lastErr string
	for _, path := range openclPaths {
		var err error
		lib.handle, err = cuda.DlopenPath(path)
		if err == nil {
			break
		}
		lastErr = err.Error()
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("opencl: dlopen libOpenCL failed: %s", lastErr)
	}

	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"clGetPlatformIDs", &lib.clGetPlatformIDs},
		{"clGetDeviceIDs", &lib.clGetDeviceIDs},
		{"clCreateContext", &lib.clCreateContext},
		{"clReleaseContext", &lib.clReleaseContext},
		{"clCreateCommandQueue", &lib.clCreateCommandQueue},
		{"clReleaseCommandQueue", &lib.clReleaseCommandQueue},
		{"clCreateBuffer", &lib.clCreateBuffer},
		{"clReleaseMemObject", &lib.clReleaseMemObject},
		{"clEnqueueWriteBuffer", &lib.clEnqueueWriteBuffer},
		{"clEnqueueReadBuffer", &lib.clEnqueueReadBuffer},
		{"clEnqueueCopyBuffer", &lib.clEnqueueCopyBuffer},
		{"clFinish", &lib.clFinish},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("opencl: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if libOpenCL can be loaded on this machine.
// The result is cached after the first call.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global OpenCLLib instance, or nil if OpenCL is not available.
func Lib() *OpenCLLib {
	if !Available() {
		return nil
	}
	return globalLib
}
