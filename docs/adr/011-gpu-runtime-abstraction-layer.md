# ADR-011: GPU Runtime Abstraction Layer (GRAL)

**Phase:** 14
**Status:** Accepted
**Date:** 2026-03-03

## Context

Before Phase 14, compute/GPUEngine and tensor/GPUStorage imported CUDA-specific
packages (`internal/cuda`, `internal/cublas`, `internal/cudnn`) directly. Every
struct field, method call, and type assertion was tied to CUDA. Adding a second
GPU backend (ROCm, OpenCL) would require duplicating or heavily conditionally
compiling the entire compute/ and tensor/ packages.

The goal was to introduce a thin abstraction layer that decouples GPUEngine and
GPUStorage from any specific GPU vendor API while preserving the existing
Engine[T] interface contract.

## Decision

Introduce `internal/gpuapi/` as the GPU Runtime Abstraction Layer (GRAL). It
defines five Go interfaces:

| Interface | Responsibility | Methods |
|-----------|---------------|---------|
| `Runtime` | Device management, memory allocation, memcpy | SetDevice, Malloc, Free, Memcpy, MemcpyAsync, MemcpyPeer, CreateStream, GetDeviceCount, DeviceType |
| `BLAS` | Matrix multiplication | Sgemm, SetStream, Destroy |
| `DNN` | Convolution, batch norm, activation, pooling, softmax | ConvForward, BatchNormForwardInference, ActivationForward, PoolingForward, SoftmaxForward, AddTensor + backward stubs |
| `KernelRunner` | Element-wise, scalar, and reduction ops | Add, Sub, Mul, Div, Pow, Exp, Log, Sqrt, Rsqrt, Tanh, TanhPrime, AddScalar, MulScalar, DivScalar, Fill, SumAxis, Softmax |
| `MemPool` | Device memory pooling | Alloc, Free, Drain, Stats |

A `Stream` interface abstracts GPU stream handles. Vendor-neutral enum types
(`MemcpyKind`, `ActivationMode`, `PoolingMode`, `BatchNormMode`) replace cuDNN
constants.

CUDA adapters (`CUDARuntime`, `CUDABlas`, `CUDADNN`, `CUDAKernels`,
`CUDAMemPool`) in the same package implement these interfaces by delegating to
the existing `internal/cuda`, `internal/cublas`, and `internal/cudnn` packages.
All adapter files are gated with `//go:build cuda`.

### Abstraction level for DNN

The DNN interface abstracts at the **operation level**, not the descriptor
level. Callers pass shapes as `[4]int` arrays and mode enums. The CUDA adapter
creates and destroys cuDNN descriptors internally per call. This simplifies
callers significantly (gpu_cudnn.go shrank by ~40%) at the cost of descriptor
recreation per call -- acceptable for inference workloads where these operations
are not the bottleneck.

### GPUStorage default runtime

`tensor/GPUStorage` uses a package-level `sync.Once` default runtime for
backward compatibility. Callers that construct GPUStorage via `NewGPUStorage` or
`NewGPUStorageFromSlice` get the default CUDA runtime without API changes. The
runtime is stored per-GPUStorage so future backends can coexist.

## Consequences

**Positive:**
- compute/ and tensor/ have zero direct imports of internal/cuda, internal/cublas, or internal/cudnn
- Adding ROCm (Phase 15) or OpenCL (Phase 16) requires only new adapter implementations
- Engine[T] interface (40 methods) is unchanged -- no downstream breakage
- DNN backward-pass methods are declared in the interface (returning "not yet implemented") for Phase 17

**Negative:**
- One extra level of indirection on every GPU call (negligible vs GPU kernel latency)
- DNN descriptor recreation per call -- may need caching if training backward pass becomes hot path
- BLAS interface simplified to not expose leading dimensions (works for contiguous row-major only)

## Key Files

- `internal/gpuapi/runtime.go` -- Runtime, Stream, MemcpyKind interfaces
- `internal/gpuapi/blas.go` -- BLAS interface
- `internal/gpuapi/dnn.go` -- DNN interface, activation/pooling/batchnorm modes
- `internal/gpuapi/kernels.go` -- KernelRunner interface
- `internal/gpuapi/mempool.go` -- MemPool interface
- `internal/gpuapi/cuda_runtime.go` -- CUDARuntime adapter
- `internal/gpuapi/cuda_blas.go` -- CUDABlas adapter
- `internal/gpuapi/cuda_dnn.go` -- CUDADNN adapter (descriptor management)
- `internal/gpuapi/cuda_kernels.go` -- CUDAKernels adapter
- `internal/gpuapi/cuda_mempool.go` -- CUDAMemPool adapter
- `compute/gpu_engine.go` -- GPUEngine stores GRAL interfaces
- `tensor/gpu_storage.go` -- GPUStorage stores gpuapi.Runtime
- `tensor/transfer.go` -- uses runtime from GPUStorage for transfers
