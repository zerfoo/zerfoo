package hip

import (
	"fmt"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// HIPLib holds dlopen handles and resolved function pointers for
// HIP runtime functions. All function pointers are resolved at Open()
// time via dlsym. Calls go through cuda.Ccall which uses the
// platform-specific zero-CGo mechanism.
type HIPLib struct {
	handle uintptr // dlopen handle for libamdhip64

	// HIP runtime function pointers
	hipMalloc            uintptr
	hipFree              uintptr
	hipMemcpy            uintptr
	hipMemcpyAsync       uintptr
	hipStreamCreate      uintptr
	hipStreamSynchronize uintptr
	hipStreamDestroy     uintptr
	hipGetDeviceCount    uintptr
	hipSetDevice         uintptr
	hipGetErrorString    uintptr
	hipMemcpyPeer        uintptr
}

var (
	globalLib  *HIPLib
	globalOnce sync.Once
	errGlobal  error
)

// hipLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedHipLibPaths. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const hipLibPathOverrideEnv = "ZERFOO_HIP_LIB_PATH"

// trustedHipLibPaths use /opt/rocm/lib, AMD's standard ROCm install prefix
// (documented ROCm convention, not zerfoo-specific -- no zerfoo deployment
// mounts a ROCm host yet, see docs/adr/012-amd-rocm-backend.md: "GPU paths
// untested without hardware"). Trying this absolute path first means dlopen
// never has to consult LD_LIBRARY_PATH/RPATH at all on a standard ROCm
// install.
var trustedHipLibPaths = []string{
	"/opt/rocm/lib/libamdhip64.so.6",
	"/opt/rocm/lib/libamdhip64.so",
}

// hipPaths lists the shared library candidates to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a DOCUMENTED residual trust assumption -- see the
// equivalent comment on cudartPaths in internal/cuda/purego.go for the full
// rationale. ROCm packages sometimes install outside /opt/rocm (distro
// packages, custom prefixes), so we keep the bare-soname fallback rather
// than failing closed on hosts that don't match the standard convention.
var hipPaths = buildHipLibPaths()

func buildHipLibPaths() []string {
	paths := make([]string, 0, len(trustedHipLibPaths)+3)
	if override := os.Getenv(hipLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedHipLibPaths...)
	paths = append(paths, "libamdhip64.so.6", "libamdhip64.so")
	return paths
}

// Open loads libamdhip64 via dlopen and resolves all HIP runtime
// function pointers via dlsym. Returns an error if HIP is not
// available (library not found or symbols missing).
func Open() (*HIPLib, error) {
	lib := &HIPLib{}

	// Try each library path until one succeeds.
	var lastErr error
	for _, path := range hipPaths {
		h, err := cuda.DlopenPath(path)
		if err == nil {
			lib.handle = h
			break
		}
		lastErr = err
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("hip: dlopen libamdhip64 failed: %w", lastErr)
	}

	// Resolve all required function pointers.
	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"hipMalloc", &lib.hipMalloc},
		{"hipFree", &lib.hipFree},
		{"hipMemcpy", &lib.hipMemcpy},
		{"hipMemcpyAsync", &lib.hipMemcpyAsync},
		{"hipStreamCreate", &lib.hipStreamCreate},
		{"hipStreamSynchronize", &lib.hipStreamSynchronize},
		{"hipStreamDestroy", &lib.hipStreamDestroy},
		{"hipGetDeviceCount", &lib.hipGetDeviceCount},
		{"hipSetDevice", &lib.hipSetDevice},
		{"hipGetErrorString", &lib.hipGetErrorString},
		{"hipMemcpyPeer", &lib.hipMemcpyPeer},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("hip: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if HIP runtime is loadable on this machine.
// The result is cached after the first call.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global HIPLib instance, or nil if HIP is not available.
func Lib() *HIPLib {
	if !Available() {
		return nil
	}
	return globalLib
}
