//go:build !cuda

package kernels

import (
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// KernelLib holds dlopen'd function pointers for custom CUDA kernels
// compiled into libkernels.so.
type KernelLib struct {
	handle uintptr

	// elementwise binary
	launchAdd, launchSub, launchMul, launchDiv, launchPow uintptr

	// elementwise scalar
	launchAddScalar, launchMulScalar, launchDivScalar uintptr
	launchSubScalar, launchPowScalar                  uintptr

	// elementwise unary
	launchExp, launchLog, launchSqrt, launchRsqrt, launchTanh uintptr
	launchTanhPrime                                            uintptr

	// elementwise special
	launchFill, launchSumAxis, launchSoftmax uintptr

	// broadcast
	launchAddBroadcast, launchSubBroadcast uintptr
	launchMulBroadcast, launchDivBroadcast uintptr

	// rmsnorm
	launchRMSNorm uintptr

	// gather
	launchGather uintptr

	// transpose
	launchTranspose2D, launchTransposeND uintptr

	// gemm_q4
	launchGemmQ4F32 uintptr
}

var (
	kernelLib     *KernelLib
	kernelLibOnce sync.Once
	errKernelLib  error
)

// openKernelLib loads libkernels.so and resolves all kernel function pointers.
func openKernelLib() (*KernelLib, error) {
	kernelLibOnce.Do(func() {
		if !cuda.Available() {
			errKernelLib = fmt.Errorf("kernels: cuda not available")
			return
		}
		lib, err := cuda.DlopenKernels()
		if err != nil {
			errKernelLib = err
			return
		}
		k := &KernelLib{handle: lib}
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
			{"launch_sub_scalar", &k.launchSubScalar},
			{"launch_pow_scalar", &k.launchPowScalar},
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
			// broadcast
			{"launch_add_broadcast", &k.launchAddBroadcast},
			{"launch_sub_broadcast", &k.launchSubBroadcast},
			{"launch_mul_broadcast", &k.launchMulBroadcast},
			{"launch_div_broadcast", &k.launchDivBroadcast},
			// rmsnorm
			{"launch_rmsnorm", &k.launchRMSNorm},
			// gather
			{"launch_gather", &k.launchGather},
			// transpose
			{"launch_transpose_2d", &k.launchTranspose2D},
			{"launch_transpose_nd", &k.launchTransposeND},
			// gemm_q4
			{"gemm_q4_f32", &k.launchGemmQ4F32},
		}
		for _, s := range syms {
			ptr, dlErr := cuda.Dlsym(lib, s.name)
			if dlErr != nil {
				errKernelLib = fmt.Errorf("kernels: dlsym %s: %w", s.name, dlErr)
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
