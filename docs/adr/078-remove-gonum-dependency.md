# ADR 078: Remove gonum.org/v1/gonum Dependency

## Status

Accepted

## Date

2026-04-01

## Context

Zerfoo uses gonum.org/v1/gonum in exactly two contexts:

1. **BLAS GEMM fallback** -- `internal/xblas/gemm.go` uses `blas64.Gemm` for float64
   matrix multiplication. `internal/xblas/gemm_simd_generic.go` uses `blas32.Gemm` as
   a portable SGEMM fallback on architectures without ARM NEON or x86 AVX2 SIMD
   assembly (build constraint: `!arm64 && !amd64`).

2. **DSP/FFT** -- `features/transformers.go` uses `gonum/dsp/fourier` for FFT-based
   time-series feature extraction in the `FFTTransformer`.

Neither use is on the critical inference or training path. The SIMD assembly kernels
(arm64 NEON, amd64 AVX2) and CUDA/ROCm/OpenCL GPU kernels handle all production
workloads. The gonum BLAS path is a fallback for rare architectures; the FFT operates
on small windows (typically 3-128 points) during offline feature engineering.

Gonum pulls in a significant transitive dependency tree. Removing it aligns with
Zerfoo's core principle of minimal external dependencies and zero-CGo builds.

## Decision

Replace gonum with zero-dependency native Go implementations:

- **BLAS GEMM**: Naive triple-loop row-major GEMM for both float32 (generic fallback)
  and float64. No tiling or SIMD optimization needed -- these paths are not
  performance-critical.

- **FFT**: Iterative Cooley-Tukey radix-2 FFT in a new `internal/dsp` package. Input
  lengths that are not powers of 2 are zero-padded. This is sufficient for the small
  window sizes used in time-series feature extraction.

After replacement, remove `gonum.org/v1/gonum` from go.mod entirely.

## Consequences

**Positive:**
- Eliminates the largest external dependency in the compute stack.
- Reduces binary size and compile time.
- Removes a dependency that could introduce breaking changes or CVEs.
- Simplifies the build for all target architectures.

**Negative:**
- Naive GEMM is slower than gonum's optimized BLAS on generic architectures. This is
  acceptable because the generic arch path is already the slow fallback (production
  builds target arm64 or amd64 where SIMD assembly is used).
- Maintaining a custom FFT implementation, though it is small (~50 lines) and unlikely
  to need changes.
