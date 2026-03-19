package autoopt

import (
	"fmt"
)

// GPUGeneration identifies a class of NVIDIA GPU microarchitecture.
type GPUGeneration int

const (
	// GPUGenAmpere represents SM 8.x GPUs (A100, RTX 30xx/40xx Ada).
	GPUGenAmpere GPUGeneration = iota

	// GPUGenHopper represents SM 9.0 GPUs (H100, H200).
	GPUGenHopper

	// GPUGenBlackwell represents SM 10.0+ GPUs (B100, B200, GB200).
	GPUGenBlackwell

	// GPUGenUnknown is returned for GPUs older than Ampere or unrecognized SM versions.
	GPUGenUnknown GPUGeneration = -1
)

// String returns the generation name.
func (g GPUGeneration) String() string {
	switch g {
	case GPUGenAmpere:
		return "Ampere"
	case GPUGenHopper:
		return "Hopper"
	case GPUGenBlackwell:
		return "Blackwell"
	default:
		return "Unknown"
	}
}

// HopperCapabilities describes SM90-class GPU features.
type HopperCapabilities struct {
	// TMA indicates Tensor Memory Accelerator support for async bulk copies.
	TMA bool

	// WGMMA indicates warp-group matrix multiply-accumulate support.
	WGMMA bool

	// FP8Native indicates hardware-native FP8 (E4M3/E5M2) tensor core support.
	FP8Native bool

	// ClusterSize is the maximum thread block cluster size (typically 8 or 16).
	ClusterSize int

	// SharedMemoryBytes is the maximum shared memory per SM in bytes.
	SharedMemoryBytes int64
}

// BlackwellCapabilities describes SM100+-class GPU features.
type BlackwellCapabilities struct {
	// TMA indicates Tensor Memory Accelerator support (inherited from Hopper).
	TMA bool

	// WGMMA indicates warp-group MMA support (inherited from Hopper).
	WGMMA bool

	// FP8Native indicates FP8 tensor core support (inherited from Hopper).
	FP8Native bool

	// FP4Aware indicates the runtime can exploit FP4 quantized weights
	// via native tensor core instructions.
	FP4Aware bool

	// ClusterPrimitives indicates support for cluster-level synchronization
	// and collective operations across thread block clusters.
	ClusterPrimitives bool

	// MaxClusterSize is the maximum cluster size (typically 16 or 32).
	MaxClusterSize int

	// SharedMemoryBytes is the maximum shared memory per SM in bytes.
	SharedMemoryBytes int64
}

// DetectGPUGeneration maps a CUDA compute capability string to a GPUGeneration.
func DetectGPUGeneration(computeCap string) GPUGeneration {
	major, _ := parseComputeCap(computeCap)
	if major == 0 {
		return GPUGenUnknown
	}
	switch {
	case major >= 10:
		return GPUGenBlackwell
	case major >= 9:
		return GPUGenHopper
	case major >= 8:
		return GPUGenAmpere
	default:
		return GPUGenUnknown
	}
}

// parseComputeCap is defined in codegen.go.

// DetectHopperCapabilities returns the Hopper feature set for SM90 GPUs.
// Returns nil if the generation is not Hopper or later.
func DetectHopperCapabilities(gen GPUGeneration) *HopperCapabilities {
	if gen != GPUGenHopper && gen != GPUGenBlackwell {
		return nil
	}
	return &HopperCapabilities{
		TMA:               true,
		WGMMA:             true,
		FP8Native:         true,
		ClusterSize:       8,
		SharedMemoryBytes: 228 * 1024, // 228 KiB per SM on H100
	}
}

// DetectBlackwellCapabilities returns the Blackwell feature set for SM100+ GPUs.
// Returns nil if the generation is not Blackwell.
func DetectBlackwellCapabilities(gen GPUGeneration) *BlackwellCapabilities {
	if gen != GPUGenBlackwell {
		return nil
	}
	return &BlackwellCapabilities{
		TMA:               true,
		WGMMA:             true,
		FP8Native:         true,
		FP4Aware:          true,
		ClusterPrimitives: true,
		MaxClusterSize:    16,
		SharedMemoryBytes: 256 * 1024, // 256 KiB per SM on B200
	}
}

// ExecutionPath identifies which kernel dispatch strategy to use for an operation.
type ExecutionPath string

const (
	// PathStandard uses the baseline CUDA kernel (Ampere or older).
	PathStandard ExecutionPath = "standard"

	// PathTMA uses Tensor Memory Accelerator for async bulk data movement.
	PathTMA ExecutionPath = "tma"

	// PathWGMMA uses warp-group MMA for matrix operations.
	PathWGMMA ExecutionPath = "wgmma"

	// PathTMAWGMMA combines TMA loads with warp-group MMA compute.
	PathTMAWGMMA ExecutionPath = "tma_wgmma"

	// PathFP4Cluster uses FP4-aware cluster-level execution on Blackwell.
	PathFP4Cluster ExecutionPath = "fp4_cluster"
)

// NextGenOptimizer selects optimal execution paths based on GPU generation.
type NextGenOptimizer struct {
	generation GPUGeneration
	hopper     *HopperCapabilities
	blackwell  *BlackwellCapabilities
}

// NewNextGenOptimizer creates an optimizer for the given hardware profile.
// Returns nil if the profile does not describe a CUDA GPU.
func NewNextGenOptimizer(hw *HardwareProfile) *NextGenOptimizer {
	if hw == nil || !hw.GPUAvailable || hw.GPUBackend != "cuda" {
		return nil
	}
	gen := DetectGPUGeneration(hw.GPUComputeCap)
	return &NextGenOptimizer{
		generation: gen,
		hopper:     DetectHopperCapabilities(gen),
		blackwell:  DetectBlackwellCapabilities(gen),
	}
}

// Generation returns the detected GPU generation.
func (o *NextGenOptimizer) Generation() GPUGeneration {
	return o.generation
}

// SelectOptimalPath picks the best execution path for an operation and GPU generation.
func (o *NextGenOptimizer) SelectOptimalPath(op *Op) ExecutionPath {
	if o == nil {
		return PathStandard
	}

	switch o.generation {
	case GPUGenBlackwell:
		return o.selectBlackwellPath(op)
	case GPUGenHopper:
		return o.selectHopperPath(op)
	default:
		return PathStandard
	}
}

// selectHopperPath chooses between Hopper-specific execution strategies.
func (o *NextGenOptimizer) selectHopperPath(op *Op) ExecutionPath {
	switch op.Class {
	case KernelGEMM, KernelQuantGEMM:
		// Large matrix multiplies benefit from TMA + WGMMA.
		if op.M >= 64 && op.N >= 64 && op.K >= 64 {
			return PathTMAWGMMA
		}
		// Smaller GEMMs still benefit from WGMMA alone.
		if op.M >= 16 && op.N >= 16 {
			return PathWGMMA
		}
		return PathStandard

	case KernelAttention:
		// Attention benefits from TMA for loading Q/K/V tiles.
		if op.M >= 32 {
			return PathTMA
		}
		return PathStandard

	default:
		// Elementwise, normalization, etc. — TMA for large tensors.
		if op.MemoryBytes >= 1024*1024 { // >= 1 MiB
			return PathTMA
		}
		return PathStandard
	}
}

// selectBlackwellPath chooses between Blackwell-specific execution strategies.
func (o *NextGenOptimizer) selectBlackwellPath(op *Op) ExecutionPath {
	switch op.Class {
	case KernelGEMM, KernelQuantGEMM:
		// Blackwell can exploit FP4 for quantized GEMM with cluster primitives.
		if op.Class == KernelQuantGEMM && op.M >= 128 && op.N >= 128 {
			return PathFP4Cluster
		}
		// Large GEMMs use TMA + WGMMA (inherited from Hopper).
		if op.M >= 64 && op.N >= 64 && op.K >= 64 {
			return PathTMAWGMMA
		}
		if op.M >= 16 && op.N >= 16 {
			return PathWGMMA
		}
		return PathStandard

	case KernelAttention:
		// Cluster-level attention for large sequence lengths.
		if op.M >= 128 {
			return PathFP4Cluster
		}
		if op.M >= 32 {
			return PathTMA
		}
		return PathStandard

	default:
		if op.MemoryBytes >= 1024*1024 {
			return PathTMA
		}
		return PathStandard
	}
}

// Describe returns a human-readable summary of the optimizer's configuration.
func (o *NextGenOptimizer) Describe() string {
	if o == nil {
		return "NextGenOptimizer: disabled (no CUDA GPU)"
	}
	switch o.generation {
	case GPUGenBlackwell:
		return fmt.Sprintf("NextGenOptimizer: Blackwell (SM100+) — TMA, WGMMA, FP4, cluster primitives (max cluster %d, smem %d KiB)",
			o.blackwell.MaxClusterSize, o.blackwell.SharedMemoryBytes/1024)
	case GPUGenHopper:
		return fmt.Sprintf("NextGenOptimizer: Hopper (SM90) — TMA, WGMMA, FP8 native (cluster %d, smem %d KiB)",
			o.hopper.ClusterSize, o.hopper.SharedMemoryBytes/1024)
	case GPUGenAmpere:
		return "NextGenOptimizer: Ampere (SM80) — standard execution paths"
	default:
		return "NextGenOptimizer: pre-Ampere — standard execution paths"
	}
}
