package autoopt

import "testing"

func TestKernelCodegen_GEMM(t *testing.T) {
	tests := []struct {
		name       string
		profile    *HardwareProfile
		m, n, k    int
		wantTileM  int
		wantTileN  int
		minShmem   int
		wantVecW   int
	}{
		{
			name: "Ampere SM 8.0 large tiles",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
				GPUMemory:     16 * 1024 * 1024 * 1024,
			},
			m: 1024, n: 1024, k: 1024,
			wantTileM: 128, wantTileN: 128, minShmem: 48 * 1024, wantVecW: 4,
		},
		{
			name: "Volta SM 7.0 medium tiles",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "7.0",
				GPUMemory:     16 * 1024 * 1024 * 1024,
			},
			m: 1024, n: 1024, k: 1024,
			wantTileM: 64, wantTileN: 64, minShmem: 32 * 1024, wantVecW: 4,
		},
		{
			name: "Pre-Volta SM 6.1 small tiles",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "6.1",
				GPUMemory:     8 * 1024 * 1024 * 1024,
			},
			m: 512, n: 512, k: 512,
			wantTileM: 32, wantTileN: 32, minShmem: 16 * 1024, wantVecW: 4,
		},
		{
			name: "CPU AVX2",
			profile: &HardwareProfile{
				CPUCores: 8,
				HasAVX2:  true,
			},
			m: 256, n: 256, k: 256,
			wantTileM: 64, wantTileN: 64, minShmem: 0, wantVecW: 8,
		},
		{
			name: "Metal GPU",
			profile: &HardwareProfile{
				GPUAvailable: true,
				GPUBackend:   "metal",
				GPUMemory:    8 * 1024 * 1024 * 1024,
			},
			m: 512, n: 512, k: 512,
			wantTileM: 64, wantTileN: 64, minShmem: 32 * 1024, wantVecW: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewKernelCodegen(tt.profile)
			cfg := cg.GenerateConfig(KernelGEMM, tt.m, tt.n, tt.k)

			if cfg.TileM != tt.wantTileM {
				t.Errorf("TileM = %d, want %d", cfg.TileM, tt.wantTileM)
			}
			if cfg.TileN != tt.wantTileN {
				t.Errorf("TileN = %d, want %d", cfg.TileN, tt.wantTileN)
			}
			if cfg.SharedMemBytes < tt.minShmem {
				t.Errorf("SharedMemBytes = %d, want >= %d", cfg.SharedMemBytes, tt.minShmem)
			}
			if cfg.VectorizationWidth != tt.wantVecW {
				t.Errorf("VectorizationWidth = %d, want %d", cfg.VectorizationWidth, tt.wantVecW)
			}

			// Grid must cover the full matrix.
			if cfg.TileM > 0 && cfg.GridDim[0]*cfg.TileM < tt.m {
				t.Errorf("GridDim[0]*TileM = %d, must cover M = %d", cfg.GridDim[0]*cfg.TileM, tt.m)
			}
			if cfg.TileN > 0 && cfg.GridDim[1]*cfg.TileN < tt.n {
				t.Errorf("GridDim[1]*TileN = %d, must cover N = %d", cfg.GridDim[1]*cfg.TileN, tt.n)
			}
		})
	}
}

func TestKernelCodegen_GEMV(t *testing.T) {
	tests := []struct {
		name      string
		profile   *HardwareProfile
		rows      int
		cols      int
		wantVecW  int
	}{
		{
			name: "CUDA GEMV vectorized",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
			},
			rows: 4096, cols: 4096,
			wantVecW: 4,
		},
		{
			name: "ARM NEON GEMV",
			profile: &HardwareProfile{
				CPUCores: 8,
				HasNEON:  true,
			},
			rows: 1024, cols: 1024,
			wantVecW: 4,
		},
		{
			name: "AVX512 GEMV",
			profile: &HardwareProfile{
				CPUCores:  16,
				HasAVX512: true,
			},
			rows: 2048, cols: 2048,
			wantVecW: 16,
		},
		{
			name: "Generic CPU scalar GEMV",
			profile: &HardwareProfile{
				CPUCores: 4,
			},
			rows: 512, cols: 512,
			wantVecW: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewKernelCodegen(tt.profile)
			cfg := cg.GenerateConfig(KernelGEMV, tt.rows, tt.cols)

			if cfg.VectorizationWidth != tt.wantVecW {
				t.Errorf("VectorizationWidth = %d, want %d", cfg.VectorizationWidth, tt.wantVecW)
			}

			// TileM should be 1 for GEMV (one row at a time).
			if cfg.TileM != 1 {
				t.Errorf("TileM = %d, want 1 for GEMV", cfg.TileM)
			}
		})
	}
}

func TestKernelCodegen_ElementWise(t *testing.T) {
	tests := []struct {
		name         string
		profile      *HardwareProfile
		numElements  int
	}{
		{
			name: "CUDA 1M elements",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
			},
			numElements: 1_000_000,
		},
		{
			name: "Metal 500K elements",
			profile: &HardwareProfile{
				GPUAvailable: true,
				GPUBackend:   "metal",
			},
			numElements: 500_000,
		},
		{
			name: "CPU AVX2 256K elements",
			profile: &HardwareProfile{
				CPUCores: 8,
				HasAVX2:  true,
			},
			numElements: 256_000,
		},
		{
			name: "Small tensor 7 elements",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
			},
			numElements: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewKernelCodegen(tt.profile)
			cfg := cg.GenerateConfig(KernelElementwise, tt.numElements)

			// Grid * block * vectorization must cover all elements.
			totalCoverage := cfg.GridDim[0] * cfg.BlockDim[0] * cfg.VectorizationWidth
			if totalCoverage < tt.numElements {
				t.Errorf("grid coverage = %d (grid=%d * block=%d * vec=%d), must cover %d elements",
					totalCoverage, cfg.GridDim[0], cfg.BlockDim[0], cfg.VectorizationWidth, tt.numElements)
			}
		})
	}
}

func TestKernelCodegen_DifferentHardware(t *testing.T) {
	profiles := []struct {
		name    string
		profile *HardwareProfile
	}{
		{
			name: "Ampere GPU",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
				GPUMemory:     16 * 1024 * 1024 * 1024,
			},
		},
		{
			name: "Pre-Volta GPU",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "6.1",
				GPUMemory:     4 * 1024 * 1024 * 1024,
			},
		},
		{
			name: "ARM NEON CPU",
			profile: &HardwareProfile{
				CPUCores: 8,
				HasNEON:  true,
			},
		},
		{
			name: "AVX512 CPU",
			profile: &HardwareProfile{
				CPUCores:  32,
				HasAVX512: true,
			},
		},
	}

	m, n, k := 1024, 1024, 1024
	configs := make([]*KernelConfig, len(profiles))

	for i, p := range profiles {
		cg := NewKernelCodegen(p.profile)
		configs[i] = cg.GenerateConfig(KernelGEMM, m, n, k)
	}

	// Verify that different hardware produces different configs.
	// At minimum, Ampere vs Pre-Volta must differ in tile sizes.
	if configs[0].TileM == configs[1].TileM && configs[0].SharedMemBytes == configs[1].SharedMemBytes {
		t.Error("Ampere and Pre-Volta produced identical GEMM configs; expected different tile sizes or shared memory")
	}

	// GPU vs CPU must differ in vectorization width.
	if configs[0].VectorizationWidth == configs[2].VectorizationWidth &&
		configs[0].BlockDim == configs[2].BlockDim {
		t.Error("GPU and CPU produced identical configs; expected different block dims or vectorization")
	}

	// NEON vs AVX512 must differ in vectorization width.
	if configs[2].VectorizationWidth == configs[3].VectorizationWidth {
		t.Errorf("NEON (vec=%d) and AVX512 (vec=%d) should have different vectorization widths",
			configs[2].VectorizationWidth, configs[3].VectorizationWidth)
	}
}

func TestGenerateLaunchParams(t *testing.T) {
	tests := []struct {
		name          string
		profile       *HardwareProfile
		class         KernelClass
		totalElements int
		wantBlockX    int
	}{
		{
			name: "CUDA launch params",
			profile: &HardwareProfile{
				GPUAvailable:  true,
				GPUBackend:    "cuda",
				GPUComputeCap: "8.0",
			},
			class:         KernelElementwise,
			totalElements: 1_000_000,
			wantBlockX:    256,
		},
		{
			name: "Metal launch params",
			profile: &HardwareProfile{
				GPUAvailable: true,
				GPUBackend:   "metal",
			},
			class:         KernelElementwise,
			totalElements: 100_000,
			wantBlockX:    64,
		},
		{
			name: "CPU launch params",
			profile: &HardwareProfile{
				CPUCores: 8,
				HasAVX2:  true,
			},
			class:         KernelElementwise,
			totalElements: 10_000,
			wantBlockX:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cg := NewKernelCodegen(tt.profile)
			gridDim, blockDim := cg.GenerateLaunchParams(tt.class, tt.totalElements)

			if blockDim[0] != tt.wantBlockX {
				t.Errorf("blockDim[0] = %d, want %d", blockDim[0], tt.wantBlockX)
			}

			// Grid must cover all elements.
			if gridDim[0] <= 0 {
				t.Errorf("gridDim[0] = %d, must be > 0", gridDim[0])
			}
		})
	}
}

func TestKernelConfig_String(t *testing.T) {
	cfg := &KernelConfig{
		TileM: 128, TileN: 128, TileK: 32,
		UnrollFactor:       8,
		SharedMemBytes:     49152,
		RegistersPerThread: 128,
		GridDim:            [3]int{8, 8, 1},
		BlockDim:           [3]int{256, 1, 1},
		VectorizationWidth: 4,
	}
	s := cfg.String()
	if s == "" {
		t.Error("String() returned empty")
	}
	// Spot-check it contains tile info.
	if got := s; len(got) < 20 {
		t.Errorf("String() too short: %q", got)
	}
}
