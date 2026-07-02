# ADR-006: GPU Engine Architecture

**Status:** Accepted
**Phase:** 2-3
**Date:** 2026-03-01

## Context

Zerfoo's compute.Engine[T] interface needed a GPU implementation for hardware
acceleration. The GPU engine must integrate with the existing interface contract
so all layers work transparently on CPU or GPU.

## Decision

### CUDA Float32 Engine

GPUEngine[T] implements Engine[T] with CUDA kernels for float32. 20 operations
have native GPU implementations; remaining operations use CPU fallback by design
(not compute-bound or require Go runtime features).

GPU-accelerated operations:
- Matrix: MatMul (cuBLAS Sgemm, 2D and batched)
- Element-wise: Add, Sub, Mul, Div, Pow (custom CUDA kernels)
- Scalar: AddScalar, MulScalar, DivScalar
- Activation: Tanh, TanhPrime
- Math: Exp, Log, Sqrt, Rsqrt
- Reduction: Sum, ReduceSum, ReduceMean (shared memory)
- Other: Softmax, Fill

CPU fallback operations: UnaryOp, Transpose, Zero/Zeros/Copy, Reshape/Split/
Concat/Repeat, Gather/ScatterAdd, OneHot/RandomUniform.

### Memory Pool

`internal/cuda/mempool.go`: Size-bucketed free-list allocator. Reuses previously
freed device memory, avoiding per-operation cudaMalloc/cudaFree. Mutex-
synchronized. Drained on GPUEngine.Close().

### cuBLAS Row-Major Strategy

cuBLAS operates in column-major order. To compute C = A * B in row-major:
call cublasSgemm with B as first argument, A as second, swapping m and n.
Avoids explicit transposition.

### OOM Fallback

When cudaMalloc fails, GPU operations fall back to CPUEngine transparently.
Atomic counter (OOMFallbackCount()) tracks fallback frequency. Logged at WARN
level via structured logger.

### Device-Resident Pipeline

GPU operations produce tensors with GPUStorage, keeping data on-device between
chained operations. Only first input (if CPU-backed) does H2D copy; only final
result (.Data() call) does D2H copy.

```
CPU Input -> H2D (via pool) -> Kernel -> GPUStorage
                                            |
                               GPUStorage -> Kernel -> GPUStorage
                                                          |
                                             .Data() -> D2H -> CPU
```

### Parity Tolerances

| Operation | Tolerance |
|-----------|-----------|
| MatMul | 1e-5 relative error |
| Element-wise | 1e-6 relative error |
| Reductions | 1e-5 relative error |

### Build Requirements

- CUDA Toolkit 12.x (libcudart, headers)
- cuBLAS library
- NVIDIA GPU with Compute Capability >= 7.0
- All CUDA code gated behind `//go:build cuda`
- Kernels compiled via `internal/cuda/kernels/Makefile` (nvcc)

### Hardware Validation

Blocked on GCP GPU quota. Target: Tesla T4 (sm_75). Also compatible with
V100 (sm_70), L4 (sm_89), A100 (sm_80), DGX Spark GB10 (sm_120).

## Consequences

- Transparent CPU/GPU switching via Engine[T] interface.
- Float32 only on GPU; other types fall back to CPU.
- No broadcasting in GPU kernels (falls back to CPU).
- Single GPU only; no multi-GPU or device selection API.
- Memory pool reduces cudaMalloc overhead for iterative workloads.
- Hardware validation still pending (E29 blocked on quota).

### Key Files

- `compute/gpu_engine.go` -- GPUEngine (pool, stream, cuBLAS)
- `compute/gpu_kernels.go` -- getDevicePtr, makeGPUResult, kernel dispatch
- `tensor/gpu_storage.go` -- GPUStorage[T]
- `internal/cuda/runtime.go` -- CUDA runtime bindings
- `internal/cuda/mempool.go` -- Memory pool
- `internal/cuda/kernels/elementwise.cu` -- 17 CUDA kernels
- `internal/cublas/cublas.go` -- cuBLAS bindings
