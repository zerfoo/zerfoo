package autoopt

import (
	"fmt"
	"strconv"
	"strings"
)

// KernelConfig holds hardware-optimized kernel launch parameters and tile
// sizes for a specific kernel class running on specific hardware.
type KernelConfig struct {
	// Tile sizes for tiled algorithms (GEMM, GEMV).
	TileM int
	TileN int
	TileK int

	// UnrollFactor controls loop unrolling depth.
	UnrollFactor int

	// SharedMemBytes is the shared memory allocation per block (GPU).
	SharedMemBytes int

	// RegistersPerThread is the target register usage per thread (GPU).
	RegistersPerThread int

	// GridDim is the compute grid dimensions [x, y, z].
	GridDim [3]int

	// BlockDim is the block/threadgroup dimensions [x, y, z].
	BlockDim [3]int

	// VectorizationWidth is the SIMD vector width in number of float32 elements.
	VectorizationWidth int
}

// KernelTemplate produces an optimal KernelConfig for a given hardware profile.
type KernelTemplate interface {
	// Configure returns hardware-optimized kernel parameters.
	Configure(profile *HardwareProfile) *KernelConfig
}

// GEMMTemplate generates GEMM kernel configurations. Tile sizes and shared
// memory usage scale with GPU compute capability and available resources.
type GEMMTemplate struct {
	// M, N, K are the matrix dimensions for the GEMM operation.
	M, N, K int
}

// Configure produces a GEMM kernel config tuned to the hardware profile.
func (t *GEMMTemplate) Configure(profile *HardwareProfile) *KernelConfig {
	cfg := &KernelConfig{}

	if profile.GPUAvailable && profile.GPUBackend == "cuda" {
		smMajor, smMinor := parseComputeCap(profile.GPUComputeCap)
		configureCUDAGEMM(cfg, smMajor, smMinor)
	} else if profile.GPUAvailable && profile.GPUBackend == "metal" {
		configureMetalGEMM(cfg)
	} else {
		configureCPUGEMM(cfg, profile)
	}

	// Compute grid dimensions to cover the full matrix.
	if cfg.TileM > 0 && cfg.TileN > 0 {
		cfg.GridDim = [3]int{
			ceilDiv(t.M, cfg.TileM),
			ceilDiv(t.N, cfg.TileN),
			1,
		}
	}

	return cfg
}

func configureCUDAGEMM(cfg *KernelConfig, smMajor, smMinor int) {
	sm := smMajor*10 + smMinor

	switch {
	case sm >= 80: // Ampere+: large tiles, lots of shared memory
		cfg.TileM = 128
		cfg.TileN = 128
		cfg.TileK = 32
		cfg.UnrollFactor = 8
		cfg.SharedMemBytes = 48 * 1024 // 48 KiB
		cfg.RegistersPerThread = 128
		cfg.BlockDim = [3]int{256, 1, 1} // 8 warps
	case sm >= 70: // Volta/Turing: medium tiles
		cfg.TileM = 64
		cfg.TileN = 64
		cfg.TileK = 16
		cfg.UnrollFactor = 4
		cfg.SharedMemBytes = 32 * 1024 // 32 KiB
		cfg.RegistersPerThread = 64
		cfg.BlockDim = [3]int{128, 1, 1} // 4 warps
	default: // Pre-Volta: conservative
		cfg.TileM = 32
		cfg.TileN = 32
		cfg.TileK = 8
		cfg.UnrollFactor = 2
		cfg.SharedMemBytes = 16 * 1024 // 16 KiB
		cfg.RegistersPerThread = 32
		cfg.BlockDim = [3]int{64, 1, 1} // 2 warps
	}

	cfg.VectorizationWidth = 4 // CUDA: float4 = 128-bit loads
}

func configureMetalGEMM(cfg *KernelConfig) {
	cfg.TileM = 64
	cfg.TileN = 64
	cfg.TileK = 16
	cfg.UnrollFactor = 4
	cfg.SharedMemBytes = 32 * 1024
	cfg.RegistersPerThread = 64
	cfg.BlockDim = [3]int{64, 1, 1} // Metal threadgroup
	cfg.VectorizationWidth = 4
}

func configureCPUGEMM(cfg *KernelConfig, profile *HardwareProfile) {
	cfg.VectorizationWidth = cpuVectorWidth(profile)
	// CPU tiling optimized for L1/L2 cache.
	cfg.TileM = 64
	cfg.TileN = 64
	cfg.TileK = 16
	cfg.UnrollFactor = cfg.VectorizationWidth
	cfg.BlockDim = [3]int{1, 1, 1}
}

// GEMVTemplate generates matrix-vector multiply kernel configurations.
// Vectorization is chosen based on SIMD width.
type GEMVTemplate struct {
	// Rows and Cols are the matrix dimensions.
	Rows, Cols int
}

// Configure produces a GEMV kernel config tuned to the hardware profile.
func (t *GEMVTemplate) Configure(profile *HardwareProfile) *KernelConfig {
	cfg := &KernelConfig{}

	if profile.GPUAvailable && (profile.GPUBackend == "cuda" || profile.GPUBackend == "rocm") {
		// One thread per row, vectorized loads along columns.
		cfg.VectorizationWidth = 4
		cfg.BlockDim = [3]int{256, 1, 1}
		cfg.GridDim = [3]int{ceilDiv(t.Rows, 256), 1, 1}
		cfg.UnrollFactor = 4
		cfg.SharedMemBytes = 0 // GEMV typically doesn't need shared mem
		cfg.TileM = 1
		cfg.TileN = t.Cols
		cfg.TileK = 1
	} else {
		// CPU path: vectorize based on SIMD capability.
		cfg.VectorizationWidth = cpuVectorWidth(profile)
		cfg.UnrollFactor = cfg.VectorizationWidth
		cfg.TileM = 1
		cfg.TileN = t.Cols
		cfg.TileK = 1
		cfg.BlockDim = [3]int{1, 1, 1}
		cfg.GridDim = [3]int{t.Rows, 1, 1}
	}

	return cfg
}

// ElementwiseTemplate generates element-wise kernel configurations.
// Maximizes occupancy with simple grid/block sizing.
type ElementwiseTemplate struct {
	// NumElements is the total number of elements to process.
	NumElements int
}

// Configure produces an element-wise kernel config tuned to the hardware profile.
func (t *ElementwiseTemplate) Configure(profile *HardwareProfile) *KernelConfig {
	cfg := &KernelConfig{}

	if profile.GPUAvailable && profile.GPUBackend != "" {
		blockSize := 256
		if profile.GPUBackend == "metal" {
			blockSize = 64 // Metal prefers smaller threadgroups
		}
		cfg.BlockDim = [3]int{blockSize, 1, 1}
		cfg.VectorizationWidth = 4
		// Each thread handles VectorizationWidth elements.
		effectiveElements := ceilDiv(t.NumElements, cfg.VectorizationWidth)
		cfg.GridDim = [3]int{ceilDiv(effectiveElements, blockSize), 1, 1}
		cfg.UnrollFactor = 4
	} else {
		cfg.VectorizationWidth = cpuVectorWidth(profile)
		cfg.UnrollFactor = cfg.VectorizationWidth
		cfg.BlockDim = [3]int{1, 1, 1}
		cfg.GridDim = [3]int{ceilDiv(t.NumElements, cfg.VectorizationWidth), 1, 1}
	}

	return cfg
}

// KernelCodegen generates hardware-optimized kernel configurations using
// templates selected by kernel class and tuned to the hardware profile.
type KernelCodegen struct {
	profile *HardwareProfile
}

// NewKernelCodegen creates a KernelCodegen bound to the given hardware profile.
func NewKernelCodegen(profile *HardwareProfile) *KernelCodegen {
	return &KernelCodegen{profile: profile}
}

// GenerateConfig selects a template for the given kernel class, configures it
// for the bound hardware profile, and returns the resulting KernelConfig.
//
// The dims parameter depends on the kernel class:
//   - KernelGEMM: [M, N, K]
//   - KernelGEMV: [rows, cols]
//   - KernelElementwise: [numElements]
//   - Other classes: [numElements] (uses elementwise template)
func (c *KernelCodegen) GenerateConfig(class KernelClass, dims ...int) *KernelConfig {
	var tmpl KernelTemplate

	switch class {
	case KernelGEMM, KernelQuantGEMM:
		m, n, k := extractMNK(dims)
		tmpl = &GEMMTemplate{M: m, N: n, K: k}
	case KernelGEMV, KernelQuantDot:
		rows, cols := extractRowsCols(dims)
		tmpl = &GEMVTemplate{Rows: rows, Cols: cols}
	default:
		// Attention, RMSNorm, Softmax, RoPE, SiLU, Elementwise all use
		// element-wise scheduling.
		n := 1
		if len(dims) > 0 {
			n = dims[0]
		}
		tmpl = &ElementwiseTemplate{NumElements: n}
	}

	return tmpl.Configure(c.profile)
}

// GenerateLaunchParams computes grid and block dimensions for launching a
// kernel of the given class over totalElements work items.
func (c *KernelCodegen) GenerateLaunchParams(class KernelClass, totalElements int) (gridDim, blockDim [3]int) {
	blockSize := 256
	vecWidth := 4

	if c.profile.GPUAvailable && c.profile.GPUBackend != "" {
		if c.profile.GPUBackend == "metal" {
			blockSize = 64
		}
	} else {
		// CPU: single "block"
		blockSize = 1
		vecWidth = cpuVectorWidth(c.profile)
	}

	blockDim = [3]int{blockSize, 1, 1}

	effectiveElements := ceilDiv(totalElements, vecWidth)
	if blockSize > 0 {
		gridDim = [3]int{ceilDiv(effectiveElements, blockSize), 1, 1}
	} else {
		gridDim = [3]int{effectiveElements, 1, 1}
	}

	return gridDim, blockDim
}

// String returns a human-readable summary of the kernel config.
func (cfg *KernelConfig) String() string {
	return fmt.Sprintf("tile=%dx%dx%d unroll=%d shmem=%dB regs=%d grid=[%d,%d,%d] block=[%d,%d,%d] vec=%d",
		cfg.TileM, cfg.TileN, cfg.TileK,
		cfg.UnrollFactor, cfg.SharedMemBytes, cfg.RegistersPerThread,
		cfg.GridDim[0], cfg.GridDim[1], cfg.GridDim[2],
		cfg.BlockDim[0], cfg.BlockDim[1], cfg.BlockDim[2],
		cfg.VectorizationWidth)
}

// --- helpers ---

func parseComputeCap(cap string) (major, minor int) {
	parts := strings.SplitN(cap, ".", 2)
	if len(parts) >= 1 {
		major, _ = strconv.Atoi(parts[0])
	}
	if len(parts) >= 2 {
		minor, _ = strconv.Atoi(parts[1])
	}
	return major, minor
}

func cpuVectorWidth(profile *HardwareProfile) int {
	switch {
	case profile.HasAVX512:
		return 16 // 512-bit / 32-bit = 16 floats
	case profile.HasAVX2:
		return 8 // 256-bit / 32-bit = 8 floats
	case profile.HasNEON:
		return 4 // 128-bit / 32-bit = 4 floats
	default:
		return 1 // scalar
	}
}

func ceilDiv(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func extractMNK(dims []int) (m, n, k int) {
	if len(dims) >= 1 {
		m = dims[0]
	}
	if len(dims) >= 2 {
		n = dims[1]
	}
	if len(dims) >= 3 {
		k = dims[2]
	}
	return m, n, k
}

func extractRowsCols(dims []int) (rows, cols int) {
	if len(dims) >= 1 {
		rows = dims[0]
	}
	if len(dims) >= 2 {
		cols = dims[1]
	}
	return rows, cols
}
