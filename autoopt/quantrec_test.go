package autoopt

import (
	"testing"
)

const (
	giB = 1024 * 1024 * 1024
)

func TestBitsPerWeight(t *testing.T) {
	tests := []struct {
		format QuantFormat
		want   float64
	}{
		{QuantNVFP4, 4.5},
		{QuantQ4K, 4.5},
		{QuantQ5K, 5.5},
		{QuantQ6K, 6.6},
		{QuantQ8_0, 8.5},
		{QuantFP8, 8.0},
		{QuantBF16, 16.0},
		{QuantFP16, 16.0},
	}
	for _, tt := range tests {
		if got := tt.format.BitsPerWeight(); got != tt.want {
			t.Errorf("BitsPerWeight(%s) = %v, want %v", tt.format, got, tt.want)
		}
	}
}

func TestRecommendQuant_NilHardware(t *testing.T) {
	rec := RecommendQuant(nil, ModelSpec{ParameterCount: 1_000_000_000}, PreferBalanced)
	if rec.Format != QuantQ4K {
		t.Errorf("nil hardware: got %s, want %s", rec.Format, QuantQ4K)
	}
}

func TestRecommendQuant_SmallModelHighVRAM(t *testing.T) {
	// 1B model on 24GB GPU — should fit at FP16 for quality preference.
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "cuda",
		GPUMemory:    24 * giB,
		GPUName:      "RTX 4090",
		TotalRAM:     64 * giB,
	}
	model := ModelSpec{ParameterCount: 1_000_000_000}

	rec := RecommendQuant(hw, model, PreferQuality)
	if rec.Format != QuantFP16 {
		t.Errorf("small model / quality: got %s, want FP16", rec.Format)
	}
	if !rec.FitsInVRAM {
		t.Error("expected FitsInVRAM=true for 1B model on 24GB GPU")
	}
}

func TestRecommendQuant_LargeModelSmallVRAM(t *testing.T) {
	// 70B model on 8GB GPU — should quantize aggressively.
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "cuda",
		GPUMemory:    8 * giB,
		GPUName:      "RTX 4070",
		TotalRAM:     32 * giB,
	}
	model := ModelSpec{ParameterCount: 70_000_000_000}

	rec := RecommendQuant(hw, model, PreferBalanced)
	// 70B at Q4_K_M ≈ 70e9 * 4.5/8 * 1.1 ≈ 43 GB — won't fit in 8GB VRAM.
	if rec.FitsInVRAM {
		t.Error("70B model should not fit in 8GB VRAM")
	}
}

func TestRecommendQuant_BalancedPreference(t *testing.T) {
	// 7B model on 16GB GPU — balanced should pick Q6_K.
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "cuda",
		GPUMemory:    16 * giB,
		GPUName:      "RTX 4080",
		TotalRAM:     64 * giB,
	}
	model := ModelSpec{ParameterCount: 7_000_000_000}

	rec := RecommendQuant(hw, model, PreferBalanced)
	// 7B at Q6_K ≈ 7e9 * 6.6/8 * 1.1 ≈ 6.35 GB — fits in 13.6 GB (85% of 16GB).
	if rec.Format != QuantQ6K {
		t.Errorf("7B balanced: got %s, want Q6_K", rec.Format)
	}
	if !rec.FitsInVRAM {
		t.Error("expected FitsInVRAM=true for 7B Q6_K on 16GB GPU")
	}
}

func TestRecommendQuant_SpeedPreference(t *testing.T) {
	// 7B model on 16GB GPU — speed preference should pick Q4_K_M.
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "cuda",
		GPUMemory:    16 * giB,
		GPUName:      "RTX 4080",
		TotalRAM:     64 * giB,
	}
	model := ModelSpec{ParameterCount: 7_000_000_000}

	rec := RecommendQuant(hw, model, PreferSpeed)
	if rec.Format != QuantQ4K {
		t.Errorf("7B speed: got %s, want Q4_K_M", rec.Format)
	}
}

func TestRecommendQuant_CPUOnly(t *testing.T) {
	// No GPU, 16GB RAM, 3B model — balanced preference.
	hw := &HardwareProfile{
		GPUAvailable: false,
		TotalRAM:     16 * giB,
		CPUCores:     8,
		HasNEON:      true,
	}
	model := ModelSpec{ParameterCount: 3_000_000_000}

	rec := RecommendQuant(hw, model, PreferBalanced)
	// 3B at Q6_K ≈ 3e9 * 6.6/8 * 1.1 ≈ 2.7 GB — fits in 11.2 GB (70% of 16GB).
	if rec.Format != QuantQ6K {
		t.Errorf("CPU-only balanced: got %s, want Q6_K", rec.Format)
	}
	if rec.FitsInVRAM {
		t.Error("expected FitsInVRAM=false when no GPU")
	}
}

func TestRecommendQuant_NVFP4_Ada(t *testing.T) {
	// NVFP4 available on Ada Lovelace (compute 8.9), speed preference.
	hw := &HardwareProfile{
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUMemory:     16 * giB,
		GPUName:       "RTX 4080",
		GPUComputeCap: "8.9",
		TotalRAM:      64 * giB,
	}
	model := ModelSpec{ParameterCount: 13_000_000_000}

	rec := RecommendQuant(hw, model, PreferSpeed)
	// 13B at NVFP4 ≈ 13e9 * 4.5/8 * 1.1 ≈ 8.0 GB — fits in 13.6 GB (85% of 16GB).
	if rec.Format != QuantNVFP4 {
		t.Errorf("Ada speed: got %s, want NVFP4", rec.Format)
	}
}

func TestRecommendQuant_NVFP4_OldGPU(t *testing.T) {
	// NVFP4 should not be recommended on older CUDA GPUs.
	hw := &HardwareProfile{
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUMemory:     16 * giB,
		GPUName:       "RTX 3080",
		GPUComputeCap: "8.6",
		TotalRAM:      64 * giB,
	}
	model := ModelSpec{ParameterCount: 7_000_000_000}

	rec := RecommendQuant(hw, model, PreferSpeed)
	if rec.Format == QuantNVFP4 {
		t.Error("NVFP4 should not be recommended on compute cap 8.6")
	}
}

func TestRecommendQuant_ROCm(t *testing.T) {
	// ROCm GPU — should not recommend NVFP4.
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "rocm",
		GPUMemory:    24 * giB,
		GPUName:      "Radeon RX 7900 XTX",
		TotalRAM:     64 * giB,
	}
	model := ModelSpec{ParameterCount: 7_000_000_000}

	rec := RecommendQuant(hw, model, PreferSpeed)
	if rec.Format == QuantNVFP4 {
		t.Error("NVFP4 should not be recommended on ROCm")
	}
}

func TestRecommendQuant_QualityLargeVRAM(t *testing.T) {
	// 7B model on 80GB A100 — quality should pick FP16.
	hw := &HardwareProfile{
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUMemory:     80 * giB,
		GPUName:       "A100 80GB",
		GPUComputeCap: "8.0",
		TotalRAM:      512 * giB,
	}
	model := ModelSpec{ParameterCount: 7_000_000_000}

	rec := RecommendQuant(hw, model, PreferQuality)
	if rec.Format != QuantFP16 {
		t.Errorf("7B quality on A100: got %s, want FP16", rec.Format)
	}
}

func TestRecommendQuant_MultiGPU(t *testing.T) {
	// Multi-GPU setup with large aggregate VRAM.
	hw := &HardwareProfile{
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUMemory:     80 * giB,
		GPUName:       "A100 80GB",
		GPUComputeCap: "8.0",
		MultiGPU:      true,
		GPUCount:      4,
		TotalRAM:      512 * giB,
	}
	model := ModelSpec{ParameterCount: 70_000_000_000}

	// Single GPU memory is used (80GB). 70B at Q8_0 ≈ 70e9*8.5/8*1.1 ≈ 82 GB.
	// Won't fit in single GPU. Q6_K ≈ 70e9*6.6/8*1.1 ≈ 63 GB — fits.
	rec := RecommendQuant(hw, model, PreferBalanced)
	if rec.Format != QuantQ6K {
		t.Errorf("70B balanced multi-GPU: got %s, want Q6_K", rec.Format)
	}
}

func TestEstimateModelBytes(t *testing.T) {
	// 7B model at FP16: 7e9 * 16/8 * 1.1 = 15.4e9 bytes ≈ 14.3 GiB
	est := estimateModelBytes(7_000_000_000, QuantFP16)
	expectedMin := int64(14 * giB)
	expectedMax := int64(16 * giB)
	if est < expectedMin || est > expectedMax {
		t.Errorf("estimateModelBytes(7B, FP16) = %d, want between %d and %d", est, expectedMin, expectedMax)
	}

	// 7B model at Q4_K_M: 7e9 * 4.5/8 * 1.1 ≈ 4.33e9 bytes ≈ 4.0 GiB
	est = estimateModelBytes(7_000_000_000, QuantQ4K)
	expectedMin = int64(3 * giB)
	expectedMax = int64(5 * giB)
	if est < expectedMin || est > expectedMax {
		t.Errorf("estimateModelBytes(7B, Q4_K_M) = %d, want between %d and %d", est, expectedMin, expectedMax)
	}
}

func TestSupportsNVFP4(t *testing.T) {
	tests := []struct {
		name string
		hw   *HardwareProfile
		want bool
	}{
		{"nil", nil, false},
		{"no GPU", &HardwareProfile{GPUAvailable: false}, false},
		{"rocm", &HardwareProfile{GPUAvailable: true, GPUBackend: "rocm", GPUComputeCap: "9.0"}, false},
		{"cuda 8.6", &HardwareProfile{GPUAvailable: true, GPUBackend: "cuda", GPUComputeCap: "8.6"}, false},
		{"cuda 8.9", &HardwareProfile{GPUAvailable: true, GPUBackend: "cuda", GPUComputeCap: "8.9"}, true},
		{"cuda 9.0", &HardwareProfile{GPUAvailable: true, GPUBackend: "cuda", GPUComputeCap: "9.0"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := supportsNVFP4(tt.hw); got != tt.want {
				t.Errorf("supportsNVFP4(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
