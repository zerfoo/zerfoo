package kernels

import (
	"fmt"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/hip"
)

// KernelLib holds dlopen'd function pointers for custom HIP kernels
// compiled into libhipkernels.so.
type KernelLib struct {
	handle uintptr

	// elementwise binary
	launchAdd, launchSub, launchMul, launchDiv, launchPow uintptr

	// elementwise scalar
	launchAddScalar, launchMulScalar, launchDivScalar uintptr

	// elementwise unary
	launchExp, launchLog, launchSqrt, launchRsqrt, launchTanh uintptr
	launchTanhPrime                                           uintptr

	// elementwise special
	launchFill, launchSumAxis, launchSoftmax uintptr

	// flash attention
	launchFlashAttentionF32 uintptr
}

// trustedHipKernelLibPath is the vetted, absolute production location for
// the custom HIP kernels shared library, mirroring the
// /opt/zerfoo/lib/libkernels.so convention already established for CUDA
// custom kernels (internal/cuda/purego.go, CUDA-1, T141.1) -- both are
// zerfoo-compiled artifacts with one intended install location, not
// third-party vendor libraries.
const trustedHipKernelLibPath = "/opt/zerfoo/lib/libhipkernels.so"

// hipKernelLibPathOverrideEnv names an environment variable that, if set to
// a vetted absolute path, is tried before trustedHipKernelLibPath. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const hipKernelLibPathOverrideEnv = "ZERFOO_HIP_KERNEL_LIB_PATH"

// hipKernelPaths lists paths to try for the custom HIP kernels shared
// library.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): every entry
// here MUST be an absolute path -- see the identical rationale on
// kernelLibPaths in internal/cuda/purego.go (CUDA-1). Do not reintroduce a
// relative or bare-soname candidate; TestHipKernelPathsAreAbsolute enforces
// this.
var hipKernelPaths = buildHipKernelLibPaths()

func buildHipKernelLibPaths() []string {
	paths := make([]string, 0, 2)
	if override := os.Getenv(hipKernelLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedHipKernelLibPath)
	return paths
}

var (
	kernelLib     *KernelLib
	kernelLibOnce sync.Once
	errKernelLib  error
)

// openKernelLib loads libhipkernels.so and resolves all kernel function pointers.
func openKernelLib() (*KernelLib, error) {
	kernelLibOnce.Do(func() {
		if !hip.Available() {
			errKernelLib = fmt.Errorf("hip kernels: hip not available")
			return
		}

		var handle uintptr
		var lastErr error
		for _, path := range hipKernelPaths {
			h, err := cuda.DlopenPath(path)
			if err == nil {
				handle = h
				break
			}
			lastErr = err
		}
		if handle == 0 {
			errKernelLib = fmt.Errorf("hip kernels: dlopen libhipkernels failed: %w", lastErr)
			return
		}

		k := &KernelLib{handle: handle}
		syms := []struct {
			name string
			dest *uintptr
		}{
			// elementwise binary
			{"launch_add", &k.launchAdd},
			{"launch_sub", &k.launchSub},
			{"launch_mul", &k.launchMul},
			{"launch_div", &k.launchDiv},
			{"launch_pow", &k.launchPow},
			// elementwise scalar
			{"launch_add_scalar", &k.launchAddScalar},
			{"launch_mul_scalar", &k.launchMulScalar},
			{"launch_div_scalar", &k.launchDivScalar},
			// elementwise unary
			{"launch_exp", &k.launchExp},
			{"launch_log", &k.launchLog},
			{"launch_sqrt", &k.launchSqrt},
			{"launch_rsqrt", &k.launchRsqrt},
			{"launch_tanh", &k.launchTanh},
			{"launch_tanh_prime", &k.launchTanhPrime},
			// elementwise special
			{"launch_fill", &k.launchFill},
			{"launch_sum_axis", &k.launchSumAxis},
			{"launch_softmax", &k.launchSoftmax},
			// flash attention
			{"flash_attention_forward_f32", &k.launchFlashAttentionF32},
		}
		for _, s := range syms {
			ptr, dlErr := cuda.Dlsym(handle, s.name)
			if dlErr != nil {
				errKernelLib = fmt.Errorf("hip kernels: dlsym %s: %w", s.name, dlErr)
				return
			}
			*s.dest = ptr
		}
		kernelLib = k
	})
	return kernelLib, errKernelLib
}

func klib() *KernelLib {
	k, _ := openKernelLib()
	return k
}

// Available returns true if the HIP kernel library is loadable.
func Available() bool {
	_, err := openKernelLib()
	return err == nil
}
