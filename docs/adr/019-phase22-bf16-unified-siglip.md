# ADR-019: Phase 22 -- BF16 GEMM, Unified Memory, SigLIP Fix

## Status

Accepted (2026-03-05)

## Context

Phase 20-21 validation on DGX Spark GB10 identified three gaps:
1. GPUEngine falls back to CPU for BFloat16 MatMul (no cuBLAS BF16 binding).
2. MemPool only uses cudaMalloc; the GB10 unified memory (128 GB LPDDR5X with
   NVLink-C2C) benefits from cudaMallocManaged for zero-copy access.
3. SigLIP vision model fails at Concat with rank mismatch [1] vs [1 1].

## Decision

### E117: BF16 cuBLAS GEMM

Added `cublasGemmEx` CGo binding with CUDA_R_16BF data type. GPUEngine.MatMul
now dispatches BFloat16 tensors to GPU via cublasGemmEx instead of falling back
to CPU. A `numeric.BFloat16Ops` type implements the full `Arithmetic[float16.BFloat16]`
interface (20+ methods) for host-side operations.

### E118: Unified Memory Allocator

Added `cudaMallocManaged` binding, `AllocManaged`/`FreeManaged` methods on
MemPool (CUDA, ROCm stub, OpenCL stub), and `NewManagedGPUStorage` constructor
in the tensor package. Managed storage skips explicit H2D memcpy since the
hardware provides coherent access.

### E119: SigLIP Concat Shape Mismatch

Root cause: `Squeeze.Forward` produced `[1]` (1D tensor) instead of `[]` (0D
scalar) when squeezing all dimensions. This caused downstream Unsqueeze to
produce `[1,1]` instead of `[1]`, creating rank mismatches at Concat.

Fix: (1) Squeeze produces true 0D scalar via `tensor.New[T](nil, data[:1])`.
(2) Concat adds defensive rank alignment -- prepends size-1 dimensions to
lower-rank inputs before concatenation. PR #27 merged.

## Benchmark Results (DGX Spark GB10, Blackwell sm_121, CUDA 13.0)

### BF16 vs FP32 GEMM Latency

| Size | BF16 (us) | FP32 (us) | Speedup |
|------|-----------|-----------|---------|
| 128  | 123       | 109       | 0.9x    |
| 512  | 212       | 349       | 1.6x    |
| 1024 | 412       | 631       | 1.5x    |
| 2048 | 1262      | --        | --      |

At small sizes (128), kernel launch overhead dominates. At 512+ BF16 is
1.5-1.6x faster than FP32 due to doubled throughput on Blackwell tensor cores.

### Unified Memory Allocation Latency

| Size | cudaMalloc (us) | cudaMallocManaged (ns) | Speedup |
|------|-----------------|------------------------|---------|
| 1 MB | 132             | 600                    | 220x    |
| 16 MB | 702            | 658                    | 1067x   |
| 64 MB | 3370           | 668                    | 5045x   |

cudaMallocManaged returns a pointer without allocating physical pages (demand
paging). On the GB10 unified memory architecture, no H2D copy is needed.

### MatMul with Managed Memory

| Size | Latency (us) |
|------|-------------|
| 512  | 192         |
| 1024 | 603         |

## Parity Test Results

SigLIP ForwardPass: PASS (previously SKIP). Updated in ADR-018.
Total model parity: 18 PASS, 4 SKIP (DeepSeek too large, multi-GPU hardware).

## Consequences

- BFloat16 inference on GPU is now fully functional. Models exported with BF16
  weights run 1.5x faster than FP32 at typical sizes (512+).
- Managed memory eliminates allocation overhead on unified memory hardware.
  Model loading can use NewManagedGPUStorage to avoid explicit H2D transfers.
- All 6 tested model families now pass parity tests (Llama3, Qwen25, Gemma3,
  Mistral, Phi3, SigLIP). Only DeepSeek (too large) and multi-GPU remain SKIP.
- Phase 22 objectives O34-O36 are complete.
