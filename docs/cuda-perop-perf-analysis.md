# CUDA Per-Op Path Performance Analysis

**Date:** 2026-03-11
**Observed:** CUDA per-op plan.Run() at 2.22 tok/s vs CPU at 5.71 tok/s on DGX Spark

## Root Cause: MatMul CPU Fallback

The primary cause is that **MatMul falls back to CPU** when built without `-tags cuda`.

### Why MatMul Falls Back

1. `gpuapi.BLASFactory` is registered in `internal/gpuapi/cuda_blas.go` via `init()`,
   but that file has `//go:build cuda` (line 1).
2. Without `-tags cuda`, `BLASFactory` is nil at runtime.
3. `NewGPUEngine` (compute/gpu_engine.go:64) checks `gpuapi.BLASFactory != nil` --
   when nil, `e.blas` stays nil.
4. `GPUEngine.MatMul` (compute/gpu_engine.go:272) checks `e.blas == nil` and falls
   back to `e.cpu.MatMul()`.

This means during inference:
- All elementwise ops (Add, Mul, Exp, Sqrt, etc.) run on GPU via purego kernels
- **All MatMul ops run on CPU** via CPUEngine
- Q4 dequant-GEMM also falls back to CPU (matMulQ4 uses `e.kernels.GemmQ4F32`
  but only when BLAS is non-nil for the float32 path; Q4 has its own kernel path
  but float32 MatMul is the dominant cost)

### Data Flow Problem

Every MatMul triggers expensive round-trip transfers:

```
GPU tensor (GPUStorage) --> Data() calls Slice() --> cudaMemcpy D2H --> CPU MatMul
CPU result (CPUStorage) --> next GPU op calls getDevicePtr() --> cudaMemcpy H2D
```

Each MatMul in a transformer layer causes:
1. **D2H copy** of input A (activation from previous GPU op)
2. **D2H copy** of input B (weight -- though weights are uploaded to GPU, MatMul
   reads them back to CPU when BLAS is nil)
3. CPU computation (slow: no SIMD-optimized BLAS like OpenBLAS or MKL)
4. Result stays on CPU as CPUStorage
5. **H2D copy** when the next GPU op (e.g. Add for residual) needs the result

For a transformer with L layers, each layer has ~4-6 MatMuls (QKV projection,
attention scores, attention output, FFN up, FFN gate, FFN down). That is
~4-6 * L synchronous `cudaMemcpy` round trips per token.

### Secondary: cudaMemcpy is Synchronous

`getDevicePtr` uses `runtime.Memcpy()` which calls `cuda.Memcpy` (not
`cuda.MemcpyAsync`). This is synchronous -- it blocks until the copy completes,
serializing all GPU work. Even if the GPU kernels are fast, the CPU-GPU
synchronization barriers dominate latency.

### Why CPU is Faster

CPU avoids all transfer overhead. Every op runs in-place on host memory with
zero-copy access. The 5.71 vs 2.22 tok/s gap (~2.6x) is entirely explained by
the H2D/D2H transfer overhead eating into what should be GPU compute gains.

## Quantified Impact Estimate

For a Gemma-3 1B model (18 layers, hidden_dim=2048):
- ~108 MatMul ops per token (6 per layer * 18 layers)
- Each MatMul: 2 D2H copies + 1 H2D copy of the result to next GPU op
- At ~2048*2048*4 bytes = 16 MB per weight matrix, synchronous cudaMemcpy of
  ~16 MB takes ~1-2 ms over PCIe (DGX Spark uses NVLink-C2C which is faster,
  but the synchronization overhead per call is still ~0.1-0.5 ms)
- 108 ops * ~0.5 ms overhead = ~54 ms of pure transfer overhead per token
- At 2.22 tok/s, each token takes ~450 ms, so transfers account for ~12% minimum
- But the CPU MatMul itself is slow (no optimized BLAS), adding further cost

## Fixes

### Fix 1: Build with `-tags cuda` (Immediate)

Building with `-tags cuda` registers `BLASFactory` via the init() in
`internal/gpuapi/cuda_blas.go`, enabling cuBLAS for MatMul. This was the
original path that achieved 12.84 tok/s (see plan.md Section 10).

### Fix 2: Convert cuBLAS to purego (Phase 2, Recommended)

Convert `internal/cublas/` from CGo to purego dlopen, matching what was done
for `internal/cuda/`. This would:
- Register BLASFactory without `-tags cuda`
- Enable cuBLAS MatMul in the default (no build tags) binary
- Align with ADR-025's goal of one binary, no build tags

This is listed in plan.md as Phase 2 / out-of-scope for the current ADR-025 work.

### Fix 3: Use GemmQ4F32 kernel for Q4 models (Partial)

The Q4 path in `matMulQ4` uses `e.kernels.GemmQ4F32` which is a purego kernel
and does NOT require cuBLAS. For Q4-quantized models, this path should already
work without `-tags cuda`. However, float32 models still need cuBLAS.

## Recommendations

1. **Short term:** Always build with `-tags cuda` on DGX Spark for per-op inference
2. **Medium term:** Convert cublas to purego (Phase 2 of ADR-025)
3. **Long term:** The megakernel path bypasses this entirely since it runs all ops
   (including MatMul) via generated CUDA code, not cuBLAS
