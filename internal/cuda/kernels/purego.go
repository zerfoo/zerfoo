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

	// broadcast 4D
	launchAddBroadcast4D, launchSubBroadcast4D uintptr
	launchMulBroadcast4D, launchDivBroadcast4D uintptr

	// rmsnorm
	launchRMSNorm uintptr

	// gather
	launchGather uintptr

	// transpose
	launchTranspose2D, launchTransposeND uintptr

	// repeat
	launchRepeat uintptr

	// gemm_q4
	launchGemmQ4F32 uintptr

	// gemv_q4k (fused dequant+GEMV for Q4_K_M)
	launchGemvQ4KF32 uintptr

	// gemm_q8
	launchGemmQ8F32 uintptr

	// argmax
	launchArgmax uintptr

	// fused_rope
	launchFusedRoPEF32 uintptr

	// fused_swiglu
	launchFusedSwiGLUF32 uintptr

	// fused_add_rmsnorm
	launchFusedAddRMSNormF32 uintptr

	// fused_norm_add
	launchFusedNormAddF32 uintptr

	// fused_qk_norm_rope
	launchFusedQKNormRoPEF32 uintptr

	// scaled_softmax
	launchScaledSoftmaxF32 uintptr

	// flash_attention
	launchFlashAttentionF32 uintptr
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
			// broadcast 4D
			{"launch_add_broadcast4d", &k.launchAddBroadcast4D},
			{"launch_sub_broadcast4d", &k.launchSubBroadcast4D},
			{"launch_mul_broadcast4d", &k.launchMulBroadcast4D},
			{"launch_div_broadcast4d", &k.launchDivBroadcast4D},
			// rmsnorm
			{"launch_rmsnorm", &k.launchRMSNorm},
			// gather
			{"launch_gather", &k.launchGather},
			// transpose
			{"launch_transpose_2d", &k.launchTranspose2D},
			{"launch_transpose_nd", &k.launchTransposeND},
			// repeat
			{"launch_repeat", &k.launchRepeat},
			// gemm_q4
			{"gemm_q4_f32", &k.launchGemmQ4F32},
			// gemv_q4k (fused dequant+GEMV for Q4_K_M)
			{"gemv_q4k_f32", &k.launchGemvQ4KF32},
			// gemm_q8
			{"gemm_q8_f32", &k.launchGemmQ8F32},
			// argmax
			{"launch_argmax", &k.launchArgmax},
			// fused_rope
			{"fused_rope_f32", &k.launchFusedRoPEF32},
			// fused_swiglu
			{"fused_swiglu_f32", &k.launchFusedSwiGLUF32},
		// fused_add_rmsnorm
		{"fused_add_rmsnorm_f32", &k.launchFusedAddRMSNormF32},
		// fused_norm_add
		{"fused_norm_add_f32", &k.launchFusedNormAddF32},
		// fused_qk_norm_rope
		{"fused_qk_norm_rope_f32", &k.launchFusedQKNormRoPEF32},
		// scaled_softmax
		{"scaled_softmax_f32", &k.launchScaledSoftmaxF32},
		// flash_attention
		{"flash_attention_forward_f32", &k.launchFlashAttentionF32},
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
