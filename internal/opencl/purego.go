package opencl

import (
	"fmt"
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

// openclPaths lists the shared library names to try, in order.
var openclPaths = []string{
	"libOpenCL.so.1",
	"libOpenCL.so",
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
