package autoopt

import "runtime"

// HardwareProfile describes the CPU and GPU capabilities of the current system.
// This mirrors github.com/zerfoo/ztensor/compute.HardwareProfile and will be
// replaced by a direct import once the ztensor dependency is updated.
type HardwareProfile struct {
	// CPU
	CPUCores  int    // logical CPU count (GOMAXPROCS-visible)
	CPUModel  string // human-readable CPU model string
	HasNEON   bool   // ARM SIMD (Neon)
	HasAVX2   bool   // x86 SIMD (AVX2)
	HasAVX512 bool   // x86 advanced SIMD (AVX-512)
	CacheL1   int64  // L1 data cache size in bytes (0 if unknown)
	CacheL2   int64  // L2 cache size in bytes (0 if unknown)
	CacheL3   int64  // L3 cache size in bytes (0 if unknown)
	TotalRAM  int64  // total physical memory in bytes

	// GPU
	GPUAvailable  bool   // true if a usable GPU was detected
	GPUBackend    string // "cuda", "rocm", "metal", "opencl", or ""
	GPUName       string // human-readable GPU name
	GPUMemory     int64  // GPU memory in bytes (0 if unknown)
	GPUComputeCap string // e.g. "8.9" for CUDA compute capability
	MultiGPU      bool   // true if more than one GPU is available
	GPUCount      int    // number of GPUs (0 if none)
}

// DefaultProfile returns a minimal hardware profile based on runtime detection.
func DefaultProfile() *HardwareProfile {
	p := &HardwareProfile{
		CPUCores: runtime.NumCPU(),
	}
	switch runtime.GOARCH {
	case "arm64":
		p.HasNEON = true
	case "amd64":
		p.HasAVX2 = true // conservative default; real detection in ztensor
	}
	return p
}
