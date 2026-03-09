//go:build !cuda

package cuda

import (
	"fmt"
	"sync"
)

// CUDALib holds dlopen handles and resolved function pointers for
// CUDA runtime functions. All function pointers are resolved at Open()
// time via dlsym. The actual calls go through platform-specific ccall
// implementations that do NOT use CGo (zero runtime.cgocall overhead).
type CUDALib struct {
	handle uintptr // dlopen handle for libcudart

	// CUDA runtime function pointers
	cudaMalloc             uintptr
	cudaFree               uintptr
	cudaMemcpy             uintptr
	cudaMemcpyAsync        uintptr
	cudaMallocManaged      uintptr
	cudaStreamCreate       uintptr
	cudaStreamSynchronize  uintptr
	cudaStreamDestroy      uintptr
	cudaGetDeviceCount     uintptr
	cudaSetDevice          uintptr
	cudaGetErrorString     uintptr
	cudaGetDeviceProperties uintptr
	cudaMemcpyPeer         uintptr
}

var (
	globalLib  *CUDALib
	globalOnce sync.Once
	errGlobal  error
)

// cudartPaths lists the shared library names to try, in order.
// On Linux, libcudart.so is the standard name. The versioned
// name (libcudart.so.12) is tried first for specificity.
var cudartPaths = []string{
	"libcudart.so.12",
	"libcudart.so",
}

// Open loads libcudart via dlopen and resolves all CUDA runtime
// function pointers via dlsym. Returns an error if CUDA is not
// available (library not found or symbols missing).
func Open() (*CUDALib, error) {
	lib := &CUDALib{}

	// Try each library path until one succeeds.
	var lastErr string
	for _, path := range cudartPaths {
		h := dlopenImpl(path, rtldLazy|rtldGlobal)
		if h != 0 {
			lib.handle = h
			break
		}
		lastErr = dlerrorImpl()
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("cuda: dlopen libcudart failed: %s", lastErr)
	}

	// Resolve all required function pointers.
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cudaMalloc", &lib.cudaMalloc},
		{"cudaFree", &lib.cudaFree},
		{"cudaMemcpy", &lib.cudaMemcpy},
		{"cudaMemcpyAsync", &lib.cudaMemcpyAsync},
		{"cudaMallocManaged", &lib.cudaMallocManaged},
		{"cudaStreamCreate", &lib.cudaStreamCreate},
		{"cudaStreamSynchronize", &lib.cudaStreamSynchronize},
		{"cudaStreamDestroy", &lib.cudaStreamDestroy},
		{"cudaGetDeviceCount", &lib.cudaGetDeviceCount},
		{"cudaSetDevice", &lib.cudaSetDevice},
		{"cudaGetErrorString", &lib.cudaGetErrorString},
		{"cudaGetDeviceProperties", &lib.cudaGetDeviceProperties},
		{"cudaMemcpyPeer", &lib.cudaMemcpyPeer},
	}
	for _, s := range syms {
		addr := dlsymImpl(lib.handle, s.name)
		if addr == 0 {
			_ = lib.Close()
			return nil, fmt.Errorf("cuda: dlsym %s failed: %s", s.name, dlerrorImpl())
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Close releases the dlopen handle.
func (lib *CUDALib) Close() error {
	if lib.handle != 0 {
		dlcloseImpl(lib.handle)
		lib.handle = 0
	}
	return nil
}

// Available returns true if CUDA runtime is loadable on this machine.
// The result is cached after the first call.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global CUDALib instance, or nil if CUDA is not available.
func Lib() *CUDALib {
	if !Available() {
		return nil
	}
	return globalLib
}

// dlopen flags
const (
	rtldLazy   = 0x1
	rtldGlobal = 0x100
)

// kernelLibPaths lists paths to try for the custom kernels shared library.
var kernelLibPaths = []string{
	"libkernels.so",
	"./libkernels.so",
}

// DlopenKernels loads the custom kernels shared library (libkernels.so)
// and returns the dlopen handle. Returns an error if the library cannot
// be found.
func DlopenKernels() (uintptr, error) {
	var lastErr string
	for _, path := range kernelLibPaths {
		h := dlopenImpl(path, rtldLazy|rtldGlobal)
		if h != 0 {
			return h, nil
		}
		lastErr = dlerrorImpl()
	}
	return 0, fmt.Errorf("kernels: dlopen libkernels failed: %s", lastErr)
}

// DlopenPath opens a shared library at the given path via dlopen.
// Returns the handle or an error if the library cannot be loaded.
func DlopenPath(path string) (uintptr, error) {
	h := dlopenImpl(path, rtldLazy|rtldGlobal)
	if h == 0 {
		return 0, fmt.Errorf("dlopen %s: %s", path, dlerrorImpl())
	}
	return h, nil
}

// Dlsym resolves a symbol from a dlopen handle. Returns the function
// pointer address or an error if the symbol is not found.
func Dlsym(handle uintptr, name string) (uintptr, error) {
	addr := dlsymImpl(handle, name)
	if addr == 0 {
		return 0, fmt.Errorf("dlsym %s: %s", name, dlerrorImpl())
	}
	return addr, nil
}

// Ccall calls a C function pointer with up to 12 arguments using the
// platform-specific zero-CGo mechanism. Exported for use by the kernels
// package.
func Ccall(fn uintptr, args ...uintptr) uintptr {
	return ccall(fn, args...)
}
