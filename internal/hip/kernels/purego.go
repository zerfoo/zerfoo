package kernels

import (
	"fmt"
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

// hipKernelPaths lists the shared library names to try, in order.
var hipKernelPaths = []string{
	"libhipkernels.so",
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
