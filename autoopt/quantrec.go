// Package autoopt provides automatic optimization recommendations based on
// hardware profiling. This file implements quantization format recommendation.
package autoopt

// HardwareProfile is defined in profile.go within this package.

// QuantFormat identifies a quantization format for model weights.
type QuantFormat string

// Supported quantization formats ordered roughly from lowest to highest quality.
const (
	QuantNVFP4 QuantFormat = "NVFP4"  // 4-bit NVIDIA FP4 (E2M1)
	QuantQ4K   QuantFormat = "Q4_K_M" // 4-bit K-quant (mixed precision)
	QuantQ5K   QuantFormat = "Q5_K_M" // 5-bit K-quant (mixed precision)
	QuantQ6K   QuantFormat = "Q6_K"   // 6-bit K-quant
	QuantQ8_0  QuantFormat = "Q8_0"   // 8-bit quantization
	QuantFP8   QuantFormat = "FP8"    // 8-bit floating point (E4M3FN)
	QuantBF16  QuantFormat = "BF16"   // Brain floating point 16
	QuantFP16  QuantFormat = "FP16"   // IEEE 754 half precision
)

// BitsPerWeight returns the approximate bits per weight for this format.
func (q QuantFormat) BitsPerWeight() float64 {
	switch q {
	case QuantNVFP4:
		return 4.5 // 4-bit + scale overhead
	case QuantQ4K:
		return 4.5
	case QuantQ5K:
		return 5.5
	case QuantQ6K:
		return 6.6
	case QuantQ8_0:
		return 8.5
	case QuantFP8:
		return 8.0
	case QuantBF16:
		return 16.0
	case QuantFP16:
		return 16.0
	default:
		return 16.0
	}
}

// ModelSpec describes a model's resource requirements for quantization recommendation.
type ModelSpec struct {
	// ParameterCount is the total number of parameters in the model.
	ParameterCount int64

	// OriginalFormat is the model's original weight format (e.g. "FP16", "BF16").
	// Used to avoid recommending a format larger than the original.
	OriginalFormat QuantFormat
}

// Preference expresses the user's priority between inference speed and output quality.
type Preference int

const (
	// PreferQuality favors higher-bit formats that preserve model accuracy.
	PreferQuality Preference = iota

	// PreferBalanced balances throughput and quality (default).
	PreferBalanced

	// PreferSpeed favors lower-bit formats that maximize tokens/second.
	PreferSpeed
)

// Recommendation is the result of RecommendQuant.
type Recommendation struct {
	// Format is the recommended quantization format.
	Format QuantFormat

	// EstimatedVRAM is the estimated GPU memory usage in bytes.
	// Zero when the model fits in system RAM only.
	EstimatedVRAM int64

	// FitsInVRAM is true when the model fits entirely in GPU memory.
	FitsInVRAM bool

	// Reason is a human-readable explanation for the recommendation.
	Reason string
}

// estimateModelBytes returns the estimated memory footprint in bytes for a
// model with the given parameter count at the specified quantization format.
func estimateModelBytes(params int64, format QuantFormat) int64 {
	bits := format.BitsPerWeight()
	// overhead factor ~1.1 for KV cache metadata, graph buffers, etc.
	return int64(float64(params) * bits / 8.0 * 1.1)
}

// RecommendQuant recommends the optimal quantization format for a model
// given the hardware profile and user preference.
func RecommendQuant(hw *HardwareProfile, model ModelSpec, pref Preference) Recommendation {
	if hw == nil {
		return Recommendation{
			Format: QuantQ4K,
			Reason: "no hardware profile available; defaulting to Q4_K_M",
		}
	}

	// Determine available memory. Prefer GPU memory when available;
	// fall back to system RAM with a conservative 70% utilization target.
	availableMem := int64(float64(hw.TotalRAM) * 0.7)
	useGPU := hw.GPUAvailable && hw.GPUMemory > 0
	if useGPU {
		availableMem = int64(float64(hw.GPUMemory) * 0.85) // 85% of VRAM
	}

	// Build candidate list ordered by preference direction.
	candidates := candidatesForPreference(pref, hw)

	// Pick the best candidate that fits in memory.
	for _, fmt := range candidates {
		est := estimateModelBytes(model.ParameterCount, fmt)
		if est <= availableMem {
			rec := Recommendation{
				Format:        fmt,
				EstimatedVRAM: est,
				FitsInVRAM:    useGPU && est <= int64(float64(hw.GPUMemory)*0.85),
				Reason:        reasonString(fmt, est, availableMem, useGPU, pref),
			}
			return rec
		}
	}

	// Nothing fits even at Q4_K_M — recommend the smallest format.
	smallest := QuantQ4K
	if supportsNVFP4(hw) {
		smallest = QuantNVFP4
	}
	est := estimateModelBytes(model.ParameterCount, smallest)
	return Recommendation{
		Format:        smallest,
		EstimatedVRAM: est,
		FitsInVRAM:    false,
		Reason:        "model exceeds available memory at all quantization levels; using smallest available format",
	}
}

// candidatesForPreference returns quantization formats ordered by suitability
// for the given preference. Quality-first returns high-bit formats first;
// speed-first returns low-bit formats first.
func candidatesForPreference(pref Preference, hw *HardwareProfile) []QuantFormat {
	qualityOrder := []QuantFormat{
		QuantFP16, QuantBF16, QuantFP8, QuantQ8_0, QuantQ6K, QuantQ5K, QuantQ4K,
	}
	speedOrder := []QuantFormat{
		QuantQ4K, QuantQ5K, QuantQ6K, QuantQ8_0, QuantFP8, QuantBF16, QuantFP16,
	}
	balancedOrder := []QuantFormat{
		QuantQ6K, QuantQ5K, QuantQ8_0, QuantQ4K, QuantFP8, QuantBF16, QuantFP16,
	}

	// NVFP4 is only useful on CUDA GPUs with compute capability >= 8.9 (Ada/Hopper).
	if supportsNVFP4(hw) {
		switch pref {
		case PreferSpeed:
			speedOrder = append([]QuantFormat{QuantNVFP4}, speedOrder...)
		case PreferBalanced:
			// Insert after Q4_K_M.
			balancedOrder = append(balancedOrder, QuantNVFP4)
		}
		// Quality preference never picks NVFP4 as it's the lowest quality.
	}

	switch pref {
	case PreferQuality:
		return qualityOrder
	case PreferSpeed:
		return speedOrder
	default:
		return balancedOrder
	}
}

// supportsNVFP4 returns true if the hardware supports NVIDIA FP4.
// Requires CUDA backend with compute capability >= 8.9 (Ada Lovelace / Hopper).
func supportsNVFP4(hw *HardwareProfile) bool {
	if hw == nil || !hw.GPUAvailable || hw.GPUBackend != "cuda" {
		return false
	}
	return hw.GPUComputeCap >= "8.9"
}

func reasonString(fmt QuantFormat, est, available int64, useGPU bool, pref Preference) string {
	memType := "system RAM"
	if useGPU {
		memType = "VRAM"
	}
	prefStr := "balanced"
	switch pref {
	case PreferQuality:
		prefStr = "quality"
	case PreferSpeed:
		prefStr = "speed"
	}
	return string(fmt) + " selected: fits in " + memType +
		" with " + prefStr + " preference"
}
