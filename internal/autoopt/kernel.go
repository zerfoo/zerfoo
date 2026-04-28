package autoopt

import "fmt"

// KernelClass identifies a category of computation for which multiple
// implementation strategies exist (e.g. GEMM, attention, normalization).
type KernelClass string

const (
	KernelGEMM           KernelClass = "gemm"            // general matrix multiply
	KernelGEMV           KernelClass = "gemv"            // matrix-vector multiply
	KernelAttention      KernelClass = "attention"        // scaled dot-product attention
	KernelRMSNorm        KernelClass = "rmsnorm"          // RMS normalization
	KernelSoftmax        KernelClass = "softmax"          // softmax
	KernelRoPE           KernelClass = "rope"             // rotary positional embedding
	KernelSiLU           KernelClass = "silu"             // SiLU/SwiGLU activation
	KernelElementwise    KernelClass = "elementwise"      // element-wise add/mul/etc.
	KernelQuantGEMM      KernelClass = "quant_gemm"       // quantized GEMM (Q4/Q8)
	KernelQuantDot       KernelClass = "quant_dot"        // quantized dot product
)

// KernelImpl identifies a specific implementation strategy for a kernel class.
type KernelImpl string

const (
	// Backend implementations
	ImplGenericCPU KernelImpl = "generic_cpu"  // scalar Go fallback
	ImplNEON       KernelImpl = "neon"         // ARM NEON SIMD assembly
	ImplAVX2       KernelImpl = "avx2"         // x86 AVX2 SIMD assembly
	ImplAVX512     KernelImpl = "avx512"       // x86 AVX-512 SIMD assembly
	ImplCUDA       KernelImpl = "cuda"         // CUDA GPU kernel
	ImplCUDAFused  KernelImpl = "cuda_fused"   // CUDA fused kernel (e.g. FlashAttention)
	ImplROCm       KernelImpl = "rocm"         // ROCm/HIP GPU kernel
	ImplROCmFused  KernelImpl = "rocm_fused"   // ROCm fused kernel
	ImplMetal      KernelImpl = "metal"        // Apple Metal GPU kernel
	ImplOpenCL     KernelImpl = "opencl"       // OpenCL GPU kernel
)

// KernelSelection maps each kernel class to the optimal implementation
// for the detected hardware.
type KernelSelection struct {
	// Selections maps each kernel class to its chosen implementation.
	Selections map[KernelClass]KernelImpl

	// Backend is the recommended compute backend ("cuda", "rocm", "metal", "opencl", "cpu").
	Backend string

	// UseFusedOps is true when fused kernel variants should be preferred.
	UseFusedOps bool

	// UseFlashAttention is true when flash attention is available and recommended.
	UseFlashAttention bool

	// MatMulThreads is the recommended number of threads for CPU GEMM.
	// Zero means use all available cores.
	MatMulThreads int

	// Reason is a human-readable summary of why this selection was made.
	Reason string
}

// SelectKernels chooses the optimal kernel implementation for each kernel
// class based on the hardware profile. It returns a KernelSelection that
// maps every kernel class to a concrete implementation.
func SelectKernels(hw *HardwareProfile) *KernelSelection {
	if hw == nil {
		return defaultCPUSelection()
	}

	// GPU path: prefer GPU kernels when a capable GPU is available.
	if hw.GPUAvailable && hw.GPUBackend != "" {
		return selectGPUKernels(hw)
	}

	// CPU-only path.
	return selectCPUKernels(hw)
}

// selectGPUKernels builds a kernel selection for GPU-accelerated workloads.
func selectGPUKernels(hw *HardwareProfile) *KernelSelection {
	sel := &KernelSelection{
		Selections: make(map[KernelClass]KernelImpl),
		Backend:    hw.GPUBackend,
	}

	switch hw.GPUBackend {
	case "cuda":
		selectCUDAKernels(hw, sel)
	case "rocm":
		selectROCmKernels(hw, sel)
	case "metal":
		selectMetalKernels(hw, sel)
	case "opencl":
		selectOpenCLKernels(hw, sel)
	default:
		return selectCPUKernels(hw)
	}

	// CPU SIMD fallback for operations not accelerated on GPU.
	fillCPUSIMDFallbacks(hw, sel)

	return sel
}

// selectCUDAKernels configures kernel selections for CUDA GPUs.
func selectCUDAKernels(hw *HardwareProfile, sel *KernelSelection) {
	// All major ops go to CUDA.
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = ImplCUDA
	}

	// Fused kernels require sufficient compute capability.
	// sm_70+ (Volta) supports basic fused ops.
	// sm_80+ (Ampere) supports flash attention and advanced fused ops.
	computeCap := hw.GPUComputeCap

	if computeCap >= "7.0" {
		sel.UseFusedOps = true
		sel.Selections[KernelRMSNorm] = ImplCUDAFused
		sel.Selections[KernelSiLU] = ImplCUDAFused
		sel.Selections[KernelRoPE] = ImplCUDAFused
		sel.Selections[KernelElementwise] = ImplCUDAFused
		sel.Reason = fmt.Sprintf("CUDA %s (%s): fused kernels enabled", hw.GPUName, computeCap)
	}

	if computeCap >= "8.0" {
		sel.UseFlashAttention = true
		sel.Selections[KernelAttention] = ImplCUDAFused
		sel.Selections[KernelSoftmax] = ImplCUDAFused
		sel.Reason = fmt.Sprintf("CUDA %s (%s): flash attention + fused kernels", hw.GPUName, computeCap)
	}

	// VRAM-constrained: prefer quantized GEMM to save memory.
	if hw.GPUMemory > 0 && hw.GPUMemory < 8*1024*1024*1024 { // < 8 GiB
		sel.Selections[KernelGEMM] = ImplCUDA
		sel.Selections[KernelQuantGEMM] = ImplCUDA
		sel.Reason += "; quantized GEMM preferred (VRAM < 8 GiB)"
	}
}

// selectROCmKernels configures kernel selections for AMD ROCm GPUs.
func selectROCmKernels(hw *HardwareProfile, sel *KernelSelection) {
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = ImplROCm
	}

	// ROCm supports fused flash attention on CDNA2+ (MI200 series) and RDNA3+.
	sel.UseFusedOps = true
	sel.Selections[KernelRMSNorm] = ImplROCmFused
	sel.Selections[KernelSiLU] = ImplROCmFused
	sel.UseFlashAttention = true
	sel.Selections[KernelAttention] = ImplROCmFused
	sel.Reason = fmt.Sprintf("ROCm %s: fused kernels + flash attention", hw.GPUName)
}

// selectMetalKernels configures kernel selections for Apple Metal GPUs.
func selectMetalKernels(hw *HardwareProfile, sel *KernelSelection) {
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = ImplMetal
	}

	// Metal doesn't support fused flash attention yet, but basic fused ops work.
	sel.UseFusedOps = true
	sel.UseFlashAttention = false
	sel.Reason = fmt.Sprintf("Metal %s: basic fused ops, no flash attention", hw.GPUName)
}

// selectOpenCLKernels configures kernel selections for OpenCL GPUs.
func selectOpenCLKernels(hw *HardwareProfile, sel *KernelSelection) {
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = ImplOpenCL
	}

	// OpenCL has limited fused kernel support.
	sel.UseFusedOps = false
	sel.UseFlashAttention = false
	sel.Reason = fmt.Sprintf("OpenCL %s: standard kernels only", hw.GPUName)
}

// selectCPUKernels builds an optimized CPU-only kernel selection.
func selectCPUKernels(hw *HardwareProfile) *KernelSelection {
	sel := &KernelSelection{
		Selections: make(map[KernelClass]KernelImpl),
		Backend:    "cpu",
	}

	impl := cpuSIMDImpl(hw)
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = impl
	}

	// Thread recommendation for CPU GEMM.
	sel.MatMulThreads = recommendThreads(hw)

	// Flash attention is CPU-only when the generic path is used.
	sel.UseFlashAttention = false
	sel.UseFusedOps = false

	sel.Reason = reasonForCPU(hw, impl)
	return sel
}

// cpuSIMDImpl returns the best CPU SIMD implementation for the hardware.
func cpuSIMDImpl(hw *HardwareProfile) KernelImpl {
	if hw == nil {
		return ImplGenericCPU
	}
	switch {
	case hw.HasAVX512:
		return ImplAVX512
	case hw.HasAVX2:
		return ImplAVX2
	case hw.HasNEON:
		return ImplNEON
	default:
		return ImplGenericCPU
	}
}

// fillCPUSIMDFallbacks fills any unset kernel selections with CPU SIMD
// implementations for kernels that may not be fully GPU-accelerated
// (e.g. quantized dot products done on CPU during loading).
func fillCPUSIMDFallbacks(hw *HardwareProfile, sel *KernelSelection) {
	impl := cpuSIMDImpl(hw)
	// Quantized dot products always run on CPU even with GPU acceleration,
	// so override with the best CPU SIMD implementation.
	sel.Selections[KernelQuantDot] = impl
}

// recommendThreads returns the recommended thread count for CPU GEMM.
func recommendThreads(hw *HardwareProfile) int {
	if hw == nil || hw.CPUCores <= 0 {
		return 0 // use all
	}
	cores := hw.CPUCores
	switch {
	case cores >= 32:
		// On high-core-count CPUs, leave some cores for the runtime.
		return cores - 2
	case cores >= 8:
		return cores - 1
	default:
		return cores
	}
}

// defaultCPUSelection returns a safe fallback when no profile is available.
func defaultCPUSelection() *KernelSelection {
	sel := &KernelSelection{
		Selections: make(map[KernelClass]KernelImpl),
		Backend:    "cpu",
		Reason:     "no hardware profile available; using generic CPU kernels",
	}
	for _, kc := range allKernelClasses() {
		sel.Selections[kc] = ImplGenericCPU
	}
	return sel
}

func reasonForCPU(hw *HardwareProfile, impl KernelImpl) string {
	if hw == nil {
		return "no hardware profile; generic CPU"
	}
	return fmt.Sprintf("CPU %d cores, SIMD: %s", hw.CPUCores, impl)
}

// allKernelClasses returns every defined KernelClass.
func allKernelClasses() []KernelClass {
	return []KernelClass{
		KernelGEMM, KernelGEMV, KernelAttention, KernelRMSNorm,
		KernelSoftmax, KernelRoPE, KernelSiLU, KernelElementwise,
		KernelQuantGEMM, KernelQuantDot,
	}
}
