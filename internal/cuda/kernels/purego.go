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
	launchExp, launchLog, launchSqrt, launchRsqrt, launchSin, launchCos, launchTanh uintptr
	launchTanhPrime                                                                 uintptr

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
	launchGather    uintptr
	launchGatherI32 uintptr

	// transpose
	launchTranspose2D, launchTransposeND uintptr

	// repeat
	launchRepeat uintptr

	// gemm_q4
	launchGemmQ4F32 uintptr

	// gemv_q4k (fused dequant+GEMV for Q4_K_M)
	launchGemvQ4KF32 uintptr

	// dequant_q4k (Q4_K to F32 for non-GEMV cuBLAS path)
	launchDequantQ4KF32 uintptr

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
	launchFlashAttentionF32        uintptr
	launchFlashAttentionDecodeF32  uintptr
	launchFlashDecodeSplitKVF32    uintptr

	// FP16 elementwise
	launchAddFP16, launchSubFP16, launchMulFP16, launchDivFP16 uintptr

	// FP16 rmsnorm
	launchRMSNormFP16 uintptr

	// FP16 scaled_softmax
	launchScaledSoftmaxFP16 uintptr

	// FP16 conversion
	launchF32ToFP16 uintptr
	launchFP16ToF32 uintptr

	// FP8 ops
	launchDequantFP8E4M3ToFP16 uintptr
	launchFP8Add               uintptr
	launchFP8Mul               uintptr
	launchFP8RMSNorm           uintptr

	// counter
	launchIncrementCounter uintptr
	launchResetCounter     uintptr

	// offset_memcpy
	launchOffsetMemcpy     uintptr
	launchOffsetMemcpyFP16 uintptr

	// rope_select
	launchRoPESelect uintptr

	// sgemv_m1
	launchSgemvM1 uintptr
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
			{"launch_sin", &k.launchSin},
			{"launch_cos", &k.launchCos},
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
			{"launch_gather_i32", &k.launchGatherI32},
			// transpose
			{"launch_transpose_2d", &k.launchTranspose2D},
			{"launch_transpose_nd", &k.launchTransposeND},
			// repeat
			{"launch_repeat", &k.launchRepeat},
			// gemm_q4
			{"gemm_q4_f32", &k.launchGemmQ4F32},
			// gemv_q4k (fused dequant+GEMV for Q4_K_M)
			{"gemv_q4k_f32", &k.launchGemvQ4KF32},
			// dequant_q4k (Q4_K to F32 for non-GEMV cuBLAS path)
			{"dequant_q4k_f32", &k.launchDequantQ4KF32},
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
		{"flash_attention_decode_f32", &k.launchFlashAttentionDecodeF32},
		{"flash_decode_splitkv_f32", &k.launchFlashDecodeSplitKVF32},
		// FP16 elementwise
		{"launch_add_fp16", &k.launchAddFP16},
		{"launch_sub_fp16", &k.launchSubFP16},
		{"launch_mul_fp16", &k.launchMulFP16},
		{"launch_div_fp16", &k.launchDivFP16},
		// FP16 rmsnorm
		{"launch_rmsnorm_fp16", &k.launchRMSNormFP16},
		// FP16 scaled_softmax
		{"launch_scaled_softmax_fp16", &k.launchScaledSoftmaxFP16},
		// FP16 conversion
		{"launch_f32_to_fp16", &k.launchF32ToFP16},
		{"launch_fp16_to_f32", &k.launchFP16ToF32},
		// FP8 ops
		{"launch_dequant_fp8e4m3_to_fp16", &k.launchDequantFP8E4M3ToFP16},
		{"launch_fp8_add", &k.launchFP8Add},
		{"launch_fp8_mul", &k.launchFP8Mul},
		{"launch_fp8_rmsnorm", &k.launchFP8RMSNorm},
		// counter
		{"launch_increment_counter", &k.launchIncrementCounter},
		{"launch_reset_counter", &k.launchResetCounter},
		// offset_memcpy
		{"launch_offset_memcpy", &k.launchOffsetMemcpy},
		{"launch_offset_memcpy_fp16", &k.launchOffsetMemcpyFP16},
		// rope_select
		{"launch_rope_select", &k.launchRoPESelect},
		// sgemv_m1
		{"launch_sgemv_m1", &k.launchSgemvM1},
		}
		// Optional symbols: missing is non-fatal (kernel not compiled yet).
		optionalSyms := map[string]bool{
			"flash_attention_decode_f32":      true,
			"flash_decode_splitkv_f32":        true,
			"launch_f32_to_fp16":              true,
			"launch_fp16_to_f32":              true,
			"launch_dequant_fp8e4m3_to_fp16":  true,
			"launch_fp8_add":                  true,
			"launch_fp8_mul":                  true,
			"launch_fp8_rmsnorm":              true,
			"launch_increment_counter":        true,
			"launch_reset_counter":            true,
			"launch_offset_memcpy":            true,
			"launch_offset_memcpy_fp16":       true,
			"launch_rope_select":              true,
			"launch_sgemv_m1":                 true,
		}
		for _, s := range syms {
			ptr, dlErr := cuda.Dlsym(lib, s.name)
			if dlErr != nil {
				if optionalSyms[s.name] {
					continue // leave function pointer as 0; callers check before use
				}
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
