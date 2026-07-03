# Kernel tolerance table (standing gate)

> **Status:** ACTIVE
> **Created:** 2026-07-03 (T135.3, `docs/plan-gpu-training-hardening.md` T3.3)
> **Scope:** `internal/cuda/kernels/` (zerfoo's CUDA kernel fork; the
> purego-loaded `libkernels.so` production path). This is the same directory
> the standing gate (`scripts/dgx-validate.sh`, default `TEST_PKGS` includes
> `./internal/cuda/kernels/...`) exercises on the GB10.

## Why this exists

T3.3 (oracle-gate every remaining kernel; fix divergences) asks for a
committed, per-op tolerance table as the standing gate, not ad-hoc numbers
buried in test files. This document is that table. The Go-level parity tests
in `internal/cuda/kernels/*_test.go` (kernel output vs. a CPU/float64
reference computed in the same test) are this repo's oracle harness for the
kernel fork -- distinct from ztensor's PyTorch-oracle harness (ADR-091),
which gates ztensor's copy of the same kernel family separately. Both must
stay honest; this table only covers the zerfoo-side fork.

## How to read a tolerance

Two shapes appear below:

- **Relative-only** (`|got-want| / |want| <= rtol`, with an absolute floor
  used only to avoid division by ~0): the traditional bound, correct for ops
  whose outputs never pass near zero relative to the compared magnitudes.
- **Combined absolute+relative** (numpy-`allclose` style,
  `|got-want| <= atol + rtol*|want|`): required wherever the *reference*
  value can be small compared to the terms that produced it (reduction /
  dot-product kernels operating on signed data, where catastrophic
  cancellation makes relative error explode on a near-zero row even though
  absolute error stays tiny). Using a purely relative bound loose enough to
  admit the cancellation case would also mask a real bug on any legitimately
  small-magnitude output -- so combined bounds are the honest choice, not a
  relaxation.

## GEMV kernel family (combined tolerance)

`sgemv_m1.cu` (custom M=1 decode GEMV) and `gemv_q4k.cu` / `gemv_q4k_sm121.cu`
(Q4_K dequant-GEMV) share the same reduction shape: each warp lane
sequentially accumulates a strided subset of the row, then a 5-level
warp-shuffle tree combines the 32 lane partials. This is a **fixed,
deterministic** order (bit-reproducible across runs on the same input) but a
**different valid parenthesization** than the naive left-to-right CPU/
float64 reference the tests compare against, so fp32 rounding legitimately
differs between the two.

| Constant | Value | File |
|---|---|---|
| `gemvReductionAbsTol` | `1e-5` | `internal/cuda/kernels/tolerance_test.go` |
| `gemvReductionRelTol` | `1e-4` | `internal/cuda/kernels/tolerance_test.go` |

Bound: `|got[i] - want[i]| <= gemvReductionAbsTol + gemvReductionRelTol*|want[i]|`.

Applies to: `TestSgemvM1_Parity`, `TestSgemvM1_MultipleSizes`,
`TestGemvQ4KF32_Parity`, `TestGemvQ4KF32_LargerMatrix`,
`TestGemvQ4KF32_MultipleSizes` (all in `sgemv_m1_test.go` /
`gemv_q4k_test.go`, via the shared `checkGemvRelError` helper).

**Measured on the GB10 (2026-07-03, T135.3, ref `368d68d1`)** -- true
full-array worst case per test (not the first-failure value; see the
`checkGemvRelError` doc comment for why a first-failure scan under-reports):

| Test | Size | Max rel err | At |diff| |
|---|---|---|---|
| `TestSgemvM1_MultipleSizes/large_4096x4096` | 4096x4096 | 7.32e-3 | ~5e-6 (want=6.30e-4) |
| `TestSgemvM1_MultipleSizes/gemma3_1b_6144x1536` | 6144x1536 | 3.96e-3 | ~6e-6 (want=-1.521e-3) |
| `TestSgemvM1_MultipleSizes/gemma3_1b_1536x1536` | 1536x1536 | 2.39e-3 | ~3e-6 (want=1.253e-3) |
| `TestGemvQ4KF32_MultipleSizes/medium_64x512` | 64x512 | 7.55e-4 | ~6e-7 (want=7.94e-4) |
| `TestGemvQ4KF32_LargerMatrix` | 512x1024 | 4.38e-4 | -- |
| `TestSgemvM1_MultipleSizes/medium_128x512` | 128x512 | 2.36e-4 | -- |

Every observed absolute diff stayed at or below ~6e-6, which is what
`gemvReductionAbsTol=1e-5` is sized against (with margin). At
normal-magnitude references (`|want| ~ 1`), the combined bound is
`~1.1e-4`, essentially unchanged from a flat `1e-4` relative test. A real
kernel bug (wrong index, dropped term, a fast-math-class blowup like the
tanh overflow in ztensor#125) produces absolute errors many orders of
magnitude above `1e-5` and stays caught.

## sgemv_m1.cu alignment fix (T135.3)

`sgemv_m1_kernel` cast `A + row*N` to `float4*` unconditionally for its
vectorized load path. When `N` is not a multiple of 4, most rows are not
16-byte aligned, and the vectorized `__ldg` float4 load faulted with a
misaligned-address error that poisoned the whole CUDA context for the rest
of the process (first observed as
`TestSgemvM1_MultipleSizes/odd_N_127x255` cascading into the next subtest,
#847 tail / #922 class). Fixed by gating the float4 path on the row
pointer's actual 16-byte alignment (`(uintptr_t)row_ptr & 0xF == 0`) instead
of assuming `N % 4 == 0`; misaligned rows fall back to the existing scalar
remainder loop, which is correct for any `N`. Verified on the GB10: the
`odd_N_127x255` and `large_4096x4096` subtests, which previously crashed the
whole package (poisoned-context cascade), now run and pass under the
tolerance above.

## Other kernel families in this package (existing tolerances, unchanged by T135.3)

These were swept through the same oracle-style parity tests during T135.3's
inventory pass; all passed at their existing bars, so no change was needed.
Listed here for completeness of the standing table.

| Kernel(s) | Test(s) | Tolerance | Shape | Notes |
|---|---|---|---|---|
| `elementwise.cu` (add, mul, exp, tanh, sum-axis, softmax, fill, broadcast ops) | `TestKernel*` (`elementwise_test.go`) | `1e-5` / `1e-6` | absolute | Elementwise + reduction ops over small fixed test tensors; no cancellation-prone data. |
| `gemm_q4.cu` (Q4 dequant-GEMM) | `TestGemmQ4F32_Correctness`, `TestGemmQ4F32_LargerMatrix` (`gemm_q4_test.go`) | `0.15` / `0.2` | absolute | Dominated by Q4 quantization noise (dequant error), not fp32 accumulation order -- a coarse bound is correct here, not a red flag. |
| `flash_attention.cu` (purego path) | `TestFlashAttentionPurego*` (`flash_attention_purego_test.go`) | `1e-4` | absolute | Softmax + weighted-sum reduction; passes comfortably at the tight bound. |
| `gather.cu`, `offset_memcpy.cu`, `rope_select.cu` | `TestGather*`, `TestOffsetMemcpy*`, `TestRoPESelect*` | exact match | -- | Pure index/copy operations; no floating-point reduction, so no tolerance needed. |
| `fp8_ops.cu`, `elementwise_fp16.cu` (signature/symbol checks only) | `TestFP8Ops*`, `TestFP16SignaturesCompile` | n/a | -- | These check the purego wrapper signatures and `libkernels.so` symbol presence, not numerics. |

**Coverage gap (noted, not a T135.3 blocker):** `dequant_q4k.cu`,
`gemm_int4.cu`, `gemm_int8.cu`, `gemm_q8.cu`, `gemv_q5k.cu`, `gemv_q6k.cu`,
`transpose.cu`, `rmsnorm.cu`, `argmax.cu`, `scaled_softmax.cu`,
`fused_add_rmsnorm.cu`, `fused_norm_add.cu`, `fused_qk_norm_rope.cu`,
`fused_rope.cu`, `fused_swiglu.cu`, and `megakernel_ops.cu` have no
dedicated numeric-parity test inside `internal/cuda/kernels/`; they are
presumably exercised indirectly through higher-level Go wrapper tests
elsewhere (`layers/`, `tests/parity/`). Extending this table to cover them
directly is follow-up work, not part of T135.3's scope (fix the two named
residuals + sweep what the standing gate's existing test suite covers).

## Build-blocking bug found and fixed during this sweep

Rebuilding `libkernels.so` for this task (see below) failed independently of
both known residuals: `gemv_q4k_sm121.cu` uses `cg::reduce` / `cg::plus`
(Cooperative Groups) but only included the `cooperative_groups.h` umbrella
header, not `cooperative_groups/reduce.h` where those symbols live on
nvcc 13.1 (`nvcr.io/nvidia/pytorch:26.02-py3`, sm_121). Fixed by adding the
explicit include. Without this fix the kernel fork does not compile at all
on a from-scratch nvcc rebuild, independent of any test outcome.

## .so rebuild path (sanctioned, used for this task)

The standing gate (`scripts/dgx-validate.sh` /
`docs/bench/manifests/validate-arm64.yaml`) mounts
`/opt/zerfoo/lib/libkernels.so` **read-only** from the DGX host and never
recompiles `.cu` in-pod (see zerfoo#921 for why `-tags cuda` in-pod is out of
scope). Fixing a `.cu` bug therefore requires a **separate one-shot Spark
build pod**:

1. Push the `.cu` fix to a branch.
2. Submit a `Pod` manifest using a CUDA **devel** image with `nvcc`
   (`nvcr.io/nvidia/pytorch:26.02-py3` -- already used as the oracle image,
   ships nvcc 13.1 with `compute_121`/sm_121 support) that:
   - clones the branch,
   - runs `make -j8 shared CUDA_ARCH=sm_121` in `internal/cuda/kernels/`,
   - mounts `/opt/zerfoo/lib` **read-write** (`readOnly: false`, unlike the
     validation manifest),
   - backs up the existing `.so` (`cp libkernels.so
     libkernels.so.bak-<timestamp>-<sha>`) before an atomic `mv` of the new
     build into place.
3. Delete the build pod; run the standing gate as usual (it will now pick up
   the new `.so` via the same read-only host mount).

No GPU resource claim (`nvidia.com/gpu`) is needed for the build pod --
`nvcc` cross-assembles for `sm_121` without touching the device, so it does
not contend with `SPARK_GPU_MAX=1` validation/training pods.

**Footgun:** keep the build pod's inline command free of embedded
double-quotes / backslash escapes -- Spark's YAML parser does not round-trip
them correctly (observed live during this task: a `\"` inside the command
string was passed through as literal backslash-quote text and broke bash
parsing). Prefer bare words / single quotes only, matching the existing
`validate-arm64.yaml` bootstrap one-liner convention.
