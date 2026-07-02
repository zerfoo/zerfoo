# ADR-013: OpenCL Backend

**Status:** Accepted
**Date:** 2026-03-03
**Phase:** 16 (OpenCL Backend)

## Context

The framework supported NVIDIA (CUDA) and AMD (ROCm) GPUs. OpenCL provides
portable GPU compute across Intel, AMD, ARM, and other vendors without
requiring vendor-specific SDKs. Phase 14's GRAL abstraction makes adding an
OpenCL backend straightforward by implementing the same five interfaces.

## Decision

### 1. OpenCL Runtime

The `internal/opencl/` package wraps OpenCL 2.0 APIs using CGo. Key differences
from CUDA/HIP:

- **Memory model**: `cl_mem` buffer objects (not raw device pointers). All
  device memory is wrapped as `cl_mem` cast to `unsafe.Pointer`.
- **Context-bound allocation**: Memory is allocated per-context, not globally.
  `clCreateBuffer` replaces `cudaMalloc`.
- **No peer-to-peer**: Cross-device copies bounce through a host buffer
  (D2H + H2D) since OpenCL has no native peer-to-peer support.

Build tag: `//go:build opencl`
Linker flag: `-lOpenCL`

### 2. CLBlast for BLAS Operations

CLBlast provides Sgemm for OpenCL. The `internal/clblast/` package wraps
`CLBlastSgemm` using the same row-major to column-major conversion strategy
as cuBLAS and rocBLAS.

Linker flag: `-lclblast -lOpenCL`

### 3. No Standard DNN Library

OpenCL has no equivalent to cuDNN or MIOpen. The OpenCLDNN adapter returns
`ErrNotSupported` for all 15 DNN methods (Conv, BatchNorm, Activation,
Pooling, Softmax, AddTensor). The compute engine falls back to CPU for
these operations.

### 4. Runtime Kernel Compilation

Unlike CUDA PTX or HIP fat binaries, OpenCL kernels are compiled from source
at runtime:

- `elementwise.cl` contains 17 kernel functions embedded via `//go:embed`
- `clCreateProgramWithSource` + `clBuildProgram` at initialization
- C dispatch helpers (`run_binary_kernel`, `run_unary_kernel`, etc.) avoid
  repeated kernel creation overhead per call

### 5. Memory Pool

`OpenCLMemPool` implements size-bucketed caching of `cl_mem` buffers, mirroring
the CUDA/ROCm memory pool pattern. This avoids repeated `clCreateBuffer` /
`clReleaseMemObject` overhead for frequently reused tensor sizes.

### 6. OpenCLEngine Delegates to CPU

All 35 Engine[T] methods delegate to CPUEngine for correctness. The GRAL
infrastructure (runtime, CLBlast, kernels, DNN stub, memory pool, stream) is
wired and ready for GPU-accelerated paths when hardware testing is available.

### 7. Integration Points

| Component | File | Purpose |
|-----------|------|---------|
| Device registration | `device/opencl_device.go` | Auto-registers OpenCL GPUs via `opencl.GetDeviceCount()` |
| Device allocator | `device/opencl_allocator.go` | cl_mem buffer allocation |
| GPU storage | `tensor/gpu_storage_default_opencl.go` | OpenCL default runtime for GPUStorage |
| Inference engine | `inference/engine_opencl.go` | Routes "opencl" / "opencl:N" to OpenCLEngine |

### 8. Build Tag Strategy

- `//go:build opencl` for OpenCL-specific code
- `//go:build cuda || rocm || opencl` for shared GPU code (gpu_storage.go, transfer.go)
- `//go:build opencl && !cuda && !rocm` for OpenCL-only defaults and engine dispatch
- Priority: CUDA > ROCm > OpenCL > CPU

## Consequences

**Positive:**
- Portable GPU support across Intel, AMD, ARM, and other OpenCL vendors
- Full GRAL interface coverage: all 5 interfaces implemented
- Device auto-registration via init() mirrors CUDA/ROCm pattern
- Inference pipeline supports "opencl" and "opencl:N" device strings
- Runtime kernel compilation avoids need for offline compilation toolchain

**Negative:**
- All compute methods delegate to CPU (GPU paths untested without hardware)
- No DNN acceleration (no OpenCL equivalent to cuDNN/MIOpen)
- No peer-to-peer device transfers (bounces through host memory)
- Runtime kernel compilation adds startup latency

**Risks:**
- OpenCL driver quality varies significantly across vendors
- cl_mem abstraction may introduce overhead vs raw device pointers
- CLBlast performance may lag behind cuBLAS/rocBLAS on equivalent hardware
