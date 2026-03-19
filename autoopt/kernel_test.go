package autoopt

import "testing"

func TestSelectKernels_NilProfile(t *testing.T) {
	sel := SelectKernels(nil)
	if sel.Backend != "cpu" {
		t.Fatalf("expected cpu backend, got %s", sel.Backend)
	}
	for _, kc := range allKernelClasses() {
		if sel.Selections[kc] != ImplGenericCPU {
			t.Errorf("kernel %s: expected generic_cpu, got %s", kc, sel.Selections[kc])
		}
	}
	if sel.UseFusedOps {
		t.Error("fused ops should be disabled for nil profile")
	}
	if sel.UseFlashAttention {
		t.Error("flash attention should be disabled for nil profile")
	}
}

func TestSelectKernels_CPUOnly_NEON(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores: 8,
		HasNEON:  true,
		TotalRAM: 16 * 1024 * 1024 * 1024, // 16 GiB
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cpu" {
		t.Fatalf("expected cpu backend, got %s", sel.Backend)
	}
	if sel.Selections[KernelGEMM] != ImplNEON {
		t.Errorf("GEMM: expected neon, got %s", sel.Selections[KernelGEMM])
	}
	if sel.Selections[KernelRMSNorm] != ImplNEON {
		t.Errorf("RMSNorm: expected neon, got %s", sel.Selections[KernelRMSNorm])
	}
	if sel.MatMulThreads != 7 {
		t.Errorf("expected 7 threads for 8 cores, got %d", sel.MatMulThreads)
	}
}

func TestSelectKernels_CPUOnly_AVX2(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores: 16,
		HasAVX2:  true,
		TotalRAM: 64 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cpu" {
		t.Fatalf("expected cpu backend, got %s", sel.Backend)
	}
	if sel.Selections[KernelGEMM] != ImplAVX2 {
		t.Errorf("GEMM: expected avx2, got %s", sel.Selections[KernelGEMM])
	}
	if sel.MatMulThreads != 15 {
		t.Errorf("expected 15 threads for 16 cores, got %d", sel.MatMulThreads)
	}
}

func TestSelectKernels_CPUOnly_AVX512(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:  64,
		HasAVX2:   true,
		HasAVX512: true,
		TotalRAM:  256 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cpu" {
		t.Fatalf("expected cpu backend, got %s", sel.Backend)
	}
	if sel.Selections[KernelGEMM] != ImplAVX512 {
		t.Errorf("GEMM: expected avx512, got %s", sel.Selections[KernelGEMM])
	}
	if sel.MatMulThreads != 62 {
		t.Errorf("expected 62 threads for 64 cores, got %d", sel.MatMulThreads)
	}
}

func TestSelectKernels_CPUOnly_Generic(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores: 2,
		TotalRAM: 4 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Selections[KernelGEMM] != ImplGenericCPU {
		t.Errorf("GEMM: expected generic_cpu, got %s", sel.Selections[KernelGEMM])
	}
	if sel.MatMulThreads != 2 {
		t.Errorf("expected 2 threads for 2 cores, got %d", sel.MatMulThreads)
	}
}

func TestSelectKernels_CUDA_Volta(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      16,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "Tesla V100",
		GPUMemory:     16 * 1024 * 1024 * 1024,
		GPUComputeCap: "7.0",
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cuda" {
		t.Fatalf("expected cuda backend, got %s", sel.Backend)
	}
	if !sel.UseFusedOps {
		t.Error("Volta should enable fused ops")
	}
	if sel.UseFlashAttention {
		t.Error("Volta should NOT enable flash attention (requires sm_80+)")
	}
	if sel.Selections[KernelRMSNorm] != ImplCUDAFused {
		t.Errorf("RMSNorm: expected cuda_fused, got %s", sel.Selections[KernelRMSNorm])
	}
	if sel.Selections[KernelAttention] != ImplCUDA {
		t.Errorf("Attention: expected cuda (not fused) on Volta, got %s", sel.Selections[KernelAttention])
	}
}

func TestSelectKernels_CUDA_Ampere(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      32,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "A100",
		GPUMemory:     80 * 1024 * 1024 * 1024,
		GPUComputeCap: "8.0",
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cuda" {
		t.Fatalf("expected cuda backend, got %s", sel.Backend)
	}
	if !sel.UseFusedOps {
		t.Error("Ampere should enable fused ops")
	}
	if !sel.UseFlashAttention {
		t.Error("Ampere should enable flash attention")
	}
	if sel.Selections[KernelAttention] != ImplCUDAFused {
		t.Errorf("Attention: expected cuda_fused, got %s", sel.Selections[KernelAttention])
	}
	if sel.Selections[KernelSoftmax] != ImplCUDAFused {
		t.Errorf("Softmax: expected cuda_fused, got %s", sel.Selections[KernelSoftmax])
	}
}

func TestSelectKernels_CUDA_LowVRAM(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      8,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "RTX 3060",
		GPUMemory:     6 * 1024 * 1024 * 1024, // 6 GiB < 8 GiB threshold
		GPUComputeCap: "8.6",
	}
	sel := SelectKernels(hw)
	if sel.Backend != "cuda" {
		t.Fatalf("expected cuda backend, got %s", sel.Backend)
	}
	// Low VRAM should still use CUDA but with quantized GEMM preference noted in reason.
	if sel.Selections[KernelQuantGEMM] != ImplCUDA {
		t.Errorf("QuantGEMM: expected cuda, got %s", sel.Selections[KernelQuantGEMM])
	}
}

func TestSelectKernels_CUDA_Ada(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      16,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "RTX 4090",
		GPUMemory:     24 * 1024 * 1024 * 1024,
		GPUComputeCap: "8.9",
	}
	sel := SelectKernels(hw)
	if !sel.UseFlashAttention {
		t.Error("Ada should enable flash attention")
	}
	if sel.Selections[KernelSiLU] != ImplCUDAFused {
		t.Errorf("SiLU: expected cuda_fused, got %s", sel.Selections[KernelSiLU])
	}
}

func TestSelectKernels_ROCm(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:     64,
		HasAVX2:      true,
		GPUAvailable: true,
		GPUBackend:   "rocm",
		GPUName:      "MI250X",
		GPUMemory:    128 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Backend != "rocm" {
		t.Fatalf("expected rocm backend, got %s", sel.Backend)
	}
	if !sel.UseFusedOps {
		t.Error("ROCm should enable fused ops")
	}
	if !sel.UseFlashAttention {
		t.Error("ROCm should enable flash attention")
	}
	if sel.Selections[KernelAttention] != ImplROCmFused {
		t.Errorf("Attention: expected rocm_fused, got %s", sel.Selections[KernelAttention])
	}
	if sel.Selections[KernelGEMM] != ImplROCm {
		t.Errorf("GEMM: expected rocm, got %s", sel.Selections[KernelGEMM])
	}
}

func TestSelectKernels_Metal(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:     10,
		HasNEON:      true,
		GPUAvailable: true,
		GPUBackend:   "metal",
		GPUName:      "Apple M2 Max",
		GPUMemory:    32 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Backend != "metal" {
		t.Fatalf("expected metal backend, got %s", sel.Backend)
	}
	if !sel.UseFusedOps {
		t.Error("Metal should enable basic fused ops")
	}
	if sel.UseFlashAttention {
		t.Error("Metal should NOT enable flash attention")
	}
	if sel.Selections[KernelGEMM] != ImplMetal {
		t.Errorf("GEMM: expected metal, got %s", sel.Selections[KernelGEMM])
	}
	// CPU SIMD fallback for quantized dot products.
	if sel.Selections[KernelQuantDot] != ImplNEON {
		t.Errorf("QuantDot: expected neon fallback, got %s", sel.Selections[KernelQuantDot])
	}
}

func TestSelectKernels_OpenCL(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:     8,
		HasAVX2:      true,
		GPUAvailable: true,
		GPUBackend:   "opencl",
		GPUName:      "Intel UHD 770",
		GPUMemory:    2 * 1024 * 1024 * 1024,
	}
	sel := SelectKernels(hw)
	if sel.Backend != "opencl" {
		t.Fatalf("expected opencl backend, got %s", sel.Backend)
	}
	if sel.UseFusedOps {
		t.Error("OpenCL should NOT enable fused ops")
	}
	if sel.UseFlashAttention {
		t.Error("OpenCL should NOT enable flash attention")
	}
	if sel.Selections[KernelGEMM] != ImplOpenCL {
		t.Errorf("GEMM: expected opencl, got %s", sel.Selections[KernelGEMM])
	}
}

func TestSelectKernels_AllClassesCovered(t *testing.T) {
	profiles := []*HardwareProfile{
		nil,
		{CPUCores: 4, HasNEON: true},
		{CPUCores: 8, HasAVX2: true},
		{CPUCores: 8, HasAVX2: true, GPUAvailable: true, GPUBackend: "cuda", GPUComputeCap: "8.0", GPUMemory: 16 * 1024 * 1024 * 1024},
		{CPUCores: 8, HasNEON: true, GPUAvailable: true, GPUBackend: "metal", GPUMemory: 16 * 1024 * 1024 * 1024},
	}

	allClasses := allKernelClasses()
	for i, hw := range profiles {
		sel := SelectKernels(hw)
		for _, kc := range allClasses {
			if _, ok := sel.Selections[kc]; !ok {
				t.Errorf("profile[%d]: kernel class %s has no selection", i, kc)
			}
		}
	}
}

func TestSelectKernels_ReasonNonEmpty(t *testing.T) {
	profiles := []*HardwareProfile{
		nil,
		{CPUCores: 4},
		{CPUCores: 8, HasAVX2: true, GPUAvailable: true, GPUBackend: "cuda", GPUComputeCap: "8.0", GPUMemory: 16 * 1024 * 1024 * 1024},
	}
	for i, hw := range profiles {
		sel := SelectKernels(hw)
		if sel.Reason == "" {
			t.Errorf("profile[%d]: expected non-empty reason", i)
		}
	}
}

func TestRecommendThreads(t *testing.T) {
	tests := []struct {
		cores    int
		expected int
	}{
		{0, 0},
		{1, 1},
		{4, 4},
		{8, 7},
		{16, 15},
		{32, 30},
		{64, 62},
		{128, 126},
	}
	for _, tc := range tests {
		hw := &HardwareProfile{CPUCores: tc.cores}
		got := recommendThreads(hw)
		if got != tc.expected {
			t.Errorf("cores=%d: expected %d threads, got %d", tc.cores, tc.expected, got)
		}
	}
}

func TestCpuSIMDImpl(t *testing.T) {
	tests := []struct {
		name     string
		hw       *HardwareProfile
		expected KernelImpl
	}{
		{"nil", nil, ImplGenericCPU},
		{"no simd", &HardwareProfile{}, ImplGenericCPU},
		{"neon", &HardwareProfile{HasNEON: true}, ImplNEON},
		{"avx2", &HardwareProfile{HasAVX2: true}, ImplAVX2},
		{"avx512", &HardwareProfile{HasAVX512: true}, ImplAVX512},
		{"avx512+avx2", &HardwareProfile{HasAVX2: true, HasAVX512: true}, ImplAVX512},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := cpuSIMDImpl(tc.hw)
			if got != tc.expected {
				t.Errorf("expected %s, got %s", tc.expected, got)
			}
		})
	}
}

func TestDefaultProfile(t *testing.T) {
	p := DefaultProfile()
	if p.CPUCores <= 0 {
		t.Error("expected positive CPU core count")
	}
}
