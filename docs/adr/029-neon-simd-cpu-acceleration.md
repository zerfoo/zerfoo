# ADR 029: NEON SIMD CPU Acceleration for Inference Parity with llama.cpp

## Status
Accepted

## Date
2026-03-07

## Context
Zerfoo CPU inference for Gemma 3 2B Q4 achieves 6.86 tok/s on ARM64 (DGX Spark
GB10). Ollama (llama.cpp) on the same hardware achieves ~100 tok/s. While the GPU
megakernel path (Track C) targets 50+ tok/s, many users run CPU-only inference on
macOS ARM64. Closing the CPU performance gap requires SIMD acceleration of the
compute hot path beyond what already exists.

Current SIMD coverage (internal/xblas/):
- Q4 quantized dot product (q4dot_arm64.s): NEON fused dequant + FMLA
- F32 dot product (gemm_simd_arm64.s:vdotf32): NEON FMLA
- F32 GEMM row accumulate (gemm_simd_arm64.s:sgemmAccRowNeon): NEON FMLA

These cover matmul/GEMV, which is the dominant operation. However, profiling shows
significant time in:
- Pow (8.9%): used in RMSNorm for x^2, calls math.Pow via generic ops.Pow
- binaryOp broadcasting overhead (10.4%): coordinate decode for every element even
  when shapes match (no broadcast needed)
- Softmax: per-element exp/div through Go generic interface dispatch
- RMSNorm: per-element sum-of-squares, rsqrt, weight multiply
- SiLU/sigmoid in SwiGLU: per-element math.Exp through Go generics
- RoPE: per-element cos/sin multiply-add
- Scalar ops (MulScalar, AddScalar, DivScalar): per-element through ops.Mul/Add/Div

Three approaches were evaluated:

A) CGo + C SIMD intrinsics: Call out to C functions using NEON intrinsics. Simpler
   to write than raw assembly. But CGo overhead (~100ns per call) dominates for
   small tensors, and the project has been moving away from CGo (ADR 025).

B) Go plan9 assembly: Hand-write ARM64 NEON assembly in Go's plan9 syntax, same
   as existing q4dot_arm64.s and gemm_simd_arm64.s. No CGo overhead, no build
   complexity, portable across Go versions.

C) Go compiler autovectorization: Wait for Go compiler SIMD improvements. Go 1.25
   has limited autovectorization and does not vectorize math.Exp or interface calls.

## Decision
Use approach B: Go plan9 ARM64 NEON assembly in internal/xblas/ with pure-Go
fallbacks for non-ARM64 platforms, following the same pattern as existing q4dot
and gemm_simd code.

Additionally, add same-shape fast paths to CPUEngine binaryOp and scalar ops that
bypass the coordinate-decode broadcasting loop when no broadcasting is needed.

For Pow with constant exponent 2.0, add a specialization that uses x*x instead of
math.Pow. RMSNorm always squares (exponent=2), which is the dominant Pow caller.

Priority order by expected performance impact:
1. Same-shape binaryOp fast path (Go, no assembly needed)
2. Pow x^2 specialization (Go)
3. NEON Softmax (assembly)
4. NEON RMSNorm (assembly)
5. NEON SiLU/sigmoid (assembly)
6. NEON vectorized elementwise (Add, Mul, Sub for same-shape fast path)
7. NEON RoPE (assembly)
8. NEON scalar ops (MulScalar, AddScalar, DivScalar)
9. Tensor arena for buffer reuse (Go)

## Consequences
Positive:
- Estimated 40-50% CPU inference speedup (6.86 -> ~10 tok/s).
- Follows established project pattern (internal/xblas/ with _arm64.s files).
- No CGo dependency. Pure Go binary with embedded assembly.
- Platform-specific files (_arm64.go, _arm64.s) with generic fallbacks.
- AMD64 can get equivalent AVX2 implementations later following the same pattern.

Negative:
- Plan9 ARM64 assembly is difficult to write and debug. Many NEON instructions
  require raw WORD encoding (Go assembler does not support all NEON mnemonics).
- Each assembly function needs a Go declaration, ARM64 implementation, and generic
  fallback -- 3 files per function.
- Testing requires ARM64 hardware or cross-compilation.
- Tensor arena adds complexity to memory management and may interact with GC.
