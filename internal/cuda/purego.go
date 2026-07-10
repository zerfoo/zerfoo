package cuda

import (
	"fmt"
	"os"
	"path/filepath"
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
	cudaGetDeviceProperties  uintptr
	cudaMemcpyPeer          uintptr
	cudaDeviceGetAttribute  uintptr

	// CUDA graph API (optional, resolved separately -- may not exist on older runtimes)
	cudaStreamBeginCapture  uintptr
	cudaStreamEndCapture    uintptr
	cudaGraphInstantiate    uintptr
	cudaGraphLaunch         uintptr
	cudaGraphDestroy        uintptr
	cudaGraphExecDestroy    uintptr
}

var (
	globalLib  *CUDALib
	globalOnce sync.Once
	errGlobal  error
)

// cudartLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedCudartLibPaths. See
// VetAbsoluteLibPath for the safety checks applied.
const cudartLibPathOverrideEnv = "ZERFOO_CUDART_LIB_PATH"

// trustedCudartLibPaths are the CUDA toolkit install locations zerfoo's own
// deployments use: docs/bench/manifests/*.yaml mount /usr/local/cuda
// read-only and set LD_LIBRARY_PATH=/usr/local/cuda/lib64. Trying these
// absolute paths first means dlopen never has to consult
// LD_LIBRARY_PATH/RPATH at all on a host that matches this convention.
var trustedCudartLibPaths = []string{
	"/usr/local/cuda/lib64/libcudart.so.12",
	"/usr/local/cuda/lib64/libcudart.so",
}

// cudartPaths lists the shared library candidates to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a residual, DOCUMENTED trust assumption, not an
// oversight. Unlike libkernels.so (a zerfoo-built artifact with exactly one
// vetted install location -- see kernelLibPaths and CUDA-1), the CUDA
// toolkit is a third-party install whose location genuinely varies across
// hosts (distro package managers, conda/pip environments, non-standard
// prefixes), so there is no single absolute path we can force. We try the
// trusted zerfoo-deployment path (and any vetted operator override) first,
// eliminating the LD_LIBRARY_PATH/RPATH hijack window whenever the trusted
// path resolves, and only fall back to a bare soname -- resolved via the
// default dynamic-linker search path, hijackable via LD_LIBRARY_PATH -- when
// no trusted absolute install is found. TestCudartPathsPreferAbsolute
// enforces that the absolute candidates always precede the bare fallback.
var cudartPaths = buildCudartPaths()

// buildCudartPaths assembles the dlopen candidate list: an optional vetted
// override, then the trusted absolute defaults, then the bare-soname
// fallback documented above.
func buildCudartPaths() []string {
	paths := make([]string, 0, len(trustedCudartLibPaths)+3)
	if override := os.Getenv(cudartLibPathOverrideEnv); override != "" {
		if vetted, ok := VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
		// An invalid override is silently ignored (not fatal): we fall
		// through to the trusted defaults rather than refusing to start.
	}
	paths = append(paths, trustedCudartLibPaths...)
	paths = append(paths, "libcudart.so.12", "libcudart.so")
	return paths
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
		{"cudaDeviceGetAttribute", &lib.cudaDeviceGetAttribute},
	}
	for _, s := range syms {
		addr := dlsymImpl(lib.handle, s.name)
		if addr == 0 {
			_ = lib.Close()
			return nil, fmt.Errorf("cuda: dlsym %s failed: %s", s.name, dlerrorImpl())
		}
		*s.ptr = addr
	}

	// Resolve optional CUDA graph API symbols (available since CUDA 10.0).
	// These are not required for basic operation, so failure is silently ignored.
	optSyms := []sym{
		{"cudaStreamBeginCapture", &lib.cudaStreamBeginCapture},
		{"cudaStreamEndCapture", &lib.cudaStreamEndCapture},
		{"cudaGraphInstantiate", &lib.cudaGraphInstantiate},
		{"cudaGraphLaunch", &lib.cudaGraphLaunch},
		{"cudaGraphDestroy", &lib.cudaGraphDestroy},
		{"cudaGraphExecDestroy", &lib.cudaGraphExecDestroy},
	}
	for _, s := range optSyms {
		addr := dlsymImpl(lib.handle, s.name)
		if addr != 0 {
			*s.ptr = addr
		}
	}

	return lib, nil
}

// GraphAvailable returns true if CUDA graph capture APIs are available.
func (lib *CUDALib) GraphAvailable() bool {
	return lib.cudaStreamBeginCapture != 0 &&
		lib.cudaStreamEndCapture != 0 &&
		lib.cudaGraphInstantiate != 0 &&
		lib.cudaGraphLaunch != 0
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

// trustedKernelLibPath is the vetted, absolute production location for the
// custom kernels shared library. Standard installs and every DGX Spark
// manifest (docs/bench/manifests/*.yaml) mount libkernels.so here.
const trustedKernelLibPath = "/opt/zerfoo/lib/libkernels.so"

// kernelLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedKernelLibPath. It exists so
// dev builds that need a non-standard install location don't have to patch
// this file. See vetKernelLibOverride for the safety checks applied.
const kernelLibPathOverrideEnv = "ZERFOO_KERNEL_LIB_PATH"

// kernelLibPaths lists paths to try for the custom kernels shared library.
//
// SECURITY: every entry here MUST be an absolute path. dlopen executes a
// shared object's ELF constructors at load time, so a bare soname (resolved
// via the default loader search path) or a CWD-relative entry lets an
// attacker who can write into the process's working directory (or influence
// LD_LIBRARY_PATH) achieve code execution the moment CUDA initializes.
// This list previously included "libkernels.so" and "./libkernels.so" --
// see docs/deep-reviews/002-full-codebase.md CUDA-1 and
// docs/adr/094-untrusted-boundary-security-hardening.md. Do not reintroduce
// a relative or bare-soname candidate; TestKernelLibPathsAreAbsolute enforces
// this.
var kernelLibPaths = buildKernelLibPaths()

// buildKernelLibPaths assembles the dlopen candidate list: an optional
// vetted override first, then the trusted absolute default.
func buildKernelLibPaths() []string {
	paths := make([]string, 0, 2)
	if override := os.Getenv(kernelLibPathOverrideEnv); override != "" {
		if vetted, ok := vetKernelLibOverride(override); ok {
			paths = append(paths, vetted)
		}
		// An invalid override is silently ignored (not fatal): we fall
		// through to the trusted default rather than refusing to start.
	}
	paths = append(paths, trustedKernelLibPath)
	return paths
}

// vetKernelLibOverride validates that path is safe to hand to dlopen for the
// kernel library override. Kept as a thin, package-local alias of
// VetAbsoluteLibPath so existing CUDA-1 regression tests keep working
// unchanged.
func vetKernelLibOverride(path string) (string, bool) {
	return VetAbsoluteLibPath(path)
}

// VetAbsoluteLibPath validates that path is safe to hand to dlopen as an
// operator-provided override for any native library candidate list in this
// module (CUDA runtime, cuBLAS, cuDNN, HIP, rocBLAS, MIOpen, OpenCL,
// TensorRT, and the zerfoo-built kernel shims -- see CUDA-2,
// docs/deep-reviews/002-full-codebase.md). The path must be absolute (never
// CWD-relative or a bare soname), it must exist, and it must not be
// world-writable -- a world-writable "trusted" path is just as hijackable
// as a CWD-relative one, since any local user could replace its contents
// with a malicious library whose ELF constructors run on the next dlopen.
// Exported so every internal/*/purego.go loader can share one vetting
// implementation instead of re-deriving these checks.
func VetAbsoluteLibPath(path string) (string, bool) {
	if !filepath.IsAbs(path) {
		return "", false
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	if info.Mode().Perm()&0o002 != 0 {
		return "", false
	}
	return path, true
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
