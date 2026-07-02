# ADR-012: AMD ROCm Backend

**Status:** Accepted
**Date:** 2026-03-03
**Phase:** 15 (AMD ROCm Backend)

## Context

The framework's GPU support was NVIDIA-only via CUDA. Phase 14 introduced GRAL
(GPU Runtime Abstraction Layer) to decouple compute engines from vendor-specific
APIs. This ADR documents the ROCm backend that implements the GRAL interfaces
for AMD GPUs using HIP, rocBLAS, and MIOpen.

## Decision

### 1. HIP as the Runtime Foundation

HIP provides a near-1:1 mapping to CUDA APIs. The `internal/hip/` package wraps
HIP runtime functions (hipMalloc, hipFree, hipMemcpy, hipStream, hipSetDevice,
hipGetDeviceCount, hipMemcpyPeer) with the same signatures as `internal/cuda/`.

Build tag: `//go:build rocm`
Linker flag: `-lamdhip64`

### 2. rocBLAS for Matrix Operations

The `internal/rocblas/` package wraps `rocblas_sgemm` using the same row-major
to column-major conversion strategy as cuBLAS. The wrapper swaps A and B
operands and transposes dimensions to achieve correct results with row-major
input data.

Linker flag: `-lrocblas`

### 3. MIOpen for DNN Operations

MIOpen replaces cuDNN with notable API differences:

- **Explicit workspace allocation**: `miopenFindConvolutionForwardAlgorithm`
  must be called before forward convolution to determine workspace size.
- **Algorithm search**: MIOpen returns a ranked list of algorithms; the adapter
  selects the fastest one.
- **Bias addition**: Uses `miopenOpTensor` (OpTensorAdd) instead of cuDNN's
  `cudnnAddTensor`.
- **Pooling workspace**: MIOpen requires workspace for pooling index tracking.

Linker flag: `-lMIOpen`

### 4. HIP Kernels Ported from CUDA

All 17 elementwise kernels and the flash attention kernel were ported from CUDA
to HIP. Key changes:

- `<<<blocks, threads, 0, stream>>>` replaced with `hipLaunchKernelGGL`
- `cudaError_t` / `cudaStream_t` replaced with `hipError_t` / `hipStream_t`
- Math functions (`expf`, `sqrtf`, `__shared__`, `__syncthreads__`) unchanged

Build system: `hipcc --offload-arch=gfx906` producing `libhipkernels.a`

### 5. ROCmEngine Delegates to CPU

All 35 Engine[T] methods delegate to CPUEngine for correctness. The GRAL
infrastructure (runtime, blas, dnn, kernels, pool, stream) is wired and ready
for GPU-accelerated paths when AMD hardware is available for testing.

### 6. Integration Points

| Component | File | Purpose |
|-----------|------|---------|
| Device registration | `device/rocm_device.go` | Auto-registers AMD GPUs via `hip.GetDeviceCount()` |
| Device allocator | `device/rocm_allocator.go` | HIP malloc/free for device memory |
| GPU storage | `tensor/gpu_storage_default_rocm.go` | ROCm default runtime for GPUStorage |
| Inference engine | `inference/engine_rocm.go` | Routes "rocm" device strings to ROCmEngine |
| Flash attention | `layers/attention/flash_rocm.go` | Fused attention using HIP kernels |

### 7. Build Tag Strategy

- `//go:build rocm` for ROCm-specific code
- `//go:build cuda || rocm` for shared GPU code (gpu_storage.go, transfer.go)
- `//go:build rocm && !cuda` for ROCm default runtime (avoids duplicate symbols)
- `//go:build !(cuda && cutlass) && !(rocm && cutlass)` for fallback stubs

## Consequences

**Positive:**
- AMD GPU support with zero changes to application code
- Full GRAL interface coverage: all 5 interfaces implemented
- Device auto-registration via init() mirrors CUDA pattern
- Inference pipeline supports "rocm" and "rocm:N" device strings

**Negative:**
- All compute methods currently delegate to CPU (GPU paths untested without hardware)
- MIOpen adapter backward methods are stubbed (training not yet supported on ROCm)
- Dual build (cuda + rocm) is mutually exclusive; cannot combine in one binary

**Risks:**
- HIP API compatibility issues may surface on actual AMD hardware
- MIOpen algorithm search performance characteristics differ from cuDNN
- Flash attention kernel may need tuning for different AMD architectures (gfx900 vs gfx1100)
