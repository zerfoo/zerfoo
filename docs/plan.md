# Zerfoo Development Plan -- Surpass Ollama Inference Performance (Phase 2)

## 1. Context

### Problem Statement

Zerfoo inference on DGX Spark GB10 achieves 166 tok/s (84.2% of Ollama's
197.21 tok/s). The prior plan (89 tasks across 8 waves) completed all tasks
including D2H elimination, fused kernels, purego conversions, kernel
optimization, and GPU residency improvements.

However, performance regressed from 188.92 tok/s (pre-wave baseline) to
166 tok/s (post-wave). The 31 tok/s gap to Ollama has two components:

1. **Regression (+23 tok/s recoverable):** Something in Waves 1-8 caused a
   12% throughput regression. Bisecting on DGX will identify the culprit.

2. **CUDA graph capture (+10-30 tok/s potential):** The graph infrastructure
   is built (graph/cuda_graph.go) with partial capture support, but 5 D2H
   copy sites in the inference pipeline prevent capture. Eliminating these
   would reduce ~338 kernel launches to a single graph replay (~15us).

3. **Fused dequant+GEMV Q4_K (+5-15 tok/s potential):** The kernel is written
   (gemv_q4k.cu) and wired into the engine, but the GGUF loader was reverted
   to Q4_0 re-quantization because Q4_K preservation caused a regression.
   The dispatch path needs end-to-end validation.

4. **Kernel optimizations (unverified):** Wave 8 added warp shuffle reductions,
   BLOCK_SIZE=64 flash attention, and register tuning (gemm_q4/transpose
   maxrregcount=32 for 100% occupancy). These have not been benchmarked yet.

See docs/design.md for full architecture. See docs/QUALITY.md for benchmark
tables and correctness report.

### Objectives

- O1: Recover the 188.92 tok/s baseline by identifying and fixing the regression.
- O2: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10.
- O3: Enable CUDA graph capture for near-zero kernel launch overhead.
- O4: Validate fused dequant+GEMV Q4_K end-to-end for halved memory bandwidth.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M, measured 2026-03-12).
- Zerfoo current: 166.02 tok/s average (3 runs, 2026-03-13).
- Zerfoo previous best: 188.92 tok/s average (3 runs, 2026-03-12).
- CUDA graph capture disabled by default (ZERFOO_ENABLE_CUDA_GRAPH to opt in).
- Managed memory disabled by default (ZERFOO_ENABLE_MANAGED_MEM to opt in).
- Megakernel abandoned. See docs/adr/032-abandon-megakernel.md.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Regression fixed | >= 188.92 tok/s | bench_tps 3-run avg on DGX Spark |
| Surpass Ollama | > 197.21 tok/s | bench_tps 3-run avg on DGX Spark |
| CUDA graph operational | Graph replay for decode tokens 3+ | Log shows "captured and instantiated successfully" |
| Fused Q4_K validated | Fused kernel used for all Q4_K MatMul | bench_tps log shows fused dispatch |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D30 | Regression bisect and fix | Recover 23 tok/s from 188->166 regression |
| D31 | CUDA graph D2H elimination | Enable graph capture for -2.37ms/token |
| D32 | Fused dequant+GEMV Q4_K validation | Halve Q4 memory bandwidth |
| D33 | Kernel optimization benchmarks | Verify Wave 8 improvements on DGX |
| D34 | BFloat16/Float16 inference path | 2x throughput potential via half-precision compute |
| D35 | FP8 inference path | 4x FP32 throughput on Blackwell via 8-bit compute |

### Out of Scope

- New model architectures.
- Training optimization.
- Multi-GPU / distributed inference.
- Managed memory optimization (known 13% regression, needs cudaMemPrefetchAsync research first).

---

## 3. Checkable Work Breakdown

### E401: Regression Bisect and Fix

- [x] T401.1 Bisect the 188->166 regression on DGX Spark  Owner: TBD  Est: 3h
  - Build bench_tps at the pre-wave baseline commit (388e60d, 2026-03-12).
  - Verify 188+ tok/s is reproducible. If not, the regression is environmental.
  - If reproducible, bisect between 388e60d and current HEAD (~50 commits).
  - Use git bisect with `bench_tps -tokens 64` for faster iteration.
  - File: all files changed in Waves 1-8.
  - Acceptance: Specific commit or set of commits identified as cause.
  - Dependencies: none.

- [x] T401.2 Fix or revert the regression-causing change  Owner: TBD  Est: 2h
  - Based on bisect results, either fix the root cause or revert the change.
  - Likely candidates: int64 gather (T301.1), Q4_K dispatch checks (T304.2),
    SubSlice changes (T301.2), or managed memory detection (T303.1).
  - Acceptance: bench_tps >= 188.92 tok/s (3-run avg) on DGX Spark.
  - Dependencies: T401.1.

- [x] S401.2.1 Regression fix verification  Owner: TBD  Est: 30m
  - Run bench_tps 3 times on DGX after fix. Report tok/s.
  - Acceptance: Average >= 188.92 tok/s. No functional regression (output coherent).
  - Dependencies: T401.2.

### E402: CUDA Graph D2H Elimination

Note: Partial capture infrastructure exists (graph/cuda_graph.go). The
CUDAGraphExecutor splits the plan into capturable and non-capturable regions.
EmbeddingLookup is already excluded. The remaining 5 D2H sites are deep in
the inference pipeline and need GPU-resident alternatives.

- [x] T402.1 Eliminate D2H in MatMul weight pointer caching  Owner: TBD  Est: 1h
  - layers/core/matmul.go:106,117 -- getCachedTranspose calls b.Data()[0]
    to compare weight pointers. Replace with tensor identity comparison
    (pointer equality on the *TensorNumeric, not on the data slice).
  - Acceptance: getCachedTranspose never calls .Data() on GPU tensors.
  - Dependencies: none.

- [x] T402.2 Eliminate D2H in KV cache append CPU fallback  Owner: TBD  Est: 1h
  - generate/tensor_cache.go:110-111 -- copy(lb.kBuf, newK.Data()) is the
    CPU fallback when isGPU=false. During GPU inference isGPU should always
    be true. Add assertion or guard to ensure GPU path is always taken.
    If isGPU is false during decode, diagnose and fix.
  - Acceptance: No .Data() call during KV cache append in GPU inference.
  - Dependencies: none.

- [x] T402.3 Eliminate D2H in FFN split CPU fallback  Owner: TBD  Est: 1h
  - layers/core/ffn.go:321 -- CPU fallback for FFN gate/up split calls
    merged.Data(). The GPU path at line 305 uses NewGPUStorageView.
    Verify the GPU path is always taken during GPU inference. If not,
    fix the condition that causes fallback.
  - Acceptance: No .Data() call in FFN split during GPU inference.
  - Dependencies: none.

- [x] T402.4 Eliminate D2H in GQA CPU fallback paths  Owner: TBD  Est: 2h
  - grouped_query_attention.go:437 -- fused QK norm+RoPE CPU fallback.
  - grouped_query_attention.go:888 -- splitMergedQKV CPU fallback.
  - Both have GPU paths via SubSlice (lines 421-434 and 867-884).
  - Verify GPU path is taken. If CPU fallback is reached, diagnose why
    the GPUStorage type assertion fails.
  - Acceptance: No .Data() call in GQA during GPU decode.
  - Dependencies: none.

- [x] T402.5 Enable CUDA graph capture and verify on DGX  Owner: TBD  Est: 2h
  - Set ZERFOO_ENABLE_CUDA_GRAPH=1 and run bench_tps on DGX Spark.
  - Confirm "captured and instantiated successfully" log appears.
  - If capture still fails, diagnose the remaining D2H site.
  - Acceptance: Graph capture succeeds. Tokens 3+ use graph replay.
  - Dependencies: T402.1, T402.2, T402.3, T402.4.

- [x] T402.6 Benchmark CUDA graph replay vs per-op  Owner: TBD  Est: 30m
  - Run bench_tps 3 times with ZERFOO_ENABLE_CUDA_GRAPH=1, 3 times without.
  - Acceptance: Graph replay faster. Results documented.
  - Dependencies: T402.5.

- [x] S402.6.1 CUDA graph correctness test  Owner: TBD  Est: 30m
  - Compare output with graph enabled vs disabled at temp=0 for 50 tokens.
  - Acceptance: Tokens identical.
  - Dependencies: T402.5.

### E403: Fused Dequant+GEMV Q4_K Validation

- [x] T403.1 Debug Q4_K dispatch regression  Owner: TBD  Est: 2h
  - The GGUF loader was reverted from Q4_K preservation to Q4_0
    re-quantization because preserving Q4_K caused a regression (100 tok/s).
  - Root cause: the Q4_K engine dispatch path may not handle all weight
    shapes, or the CPU dequantize fallback for batch>1 is slow.
  - Read compute/gpu_engine.go (matMulQ4K, matMulQ4KBWeight), model/gguf/loader.go.
  - Add diagnostic logging to the Q4_K dispatch path.
  - Test with Q4_K preservation re-enabled and identify which MatMul calls
    fall back to CPU.
  - Acceptance: Root cause identified. Document which calls fall through.
  - Dependencies: none.

- [x] T403.2 Fix Q4_K dispatch to handle all weight shapes  Owner: TBD  Est: 3h
  - Based on T403.1 findings, fix the dispatch path so all Q4_K weights
    use the fused GEMV kernel for batch=1 and GPU dequantize+cuBLAS for
    batch>1.
  - Re-enable Q4_K preservation in the GGUF loader.
  - Acceptance: bench_tps with Q4_K model produces correct output at
    >= 166 tok/s (no regression from current baseline).
  - Dependencies: T403.1.

- [x] S403.2.1 Q4_K end-to-end benchmark  Owner: TBD  Est: 30m
  - Run bench_tps 3 times with Q4_K preservation enabled.
  - Compare tok/s with Q4_0 re-quantization path.
  - Acceptance: Q4_K path >= Q4_0 path. Results documented.
  - Dependencies: T403.2.

### E404: Kernel Optimization Benchmarks

- [x] T404.1 Rebuild kernels with Wave 8 optimizations on DGX  Owner: TBD  Est: 30m
  - cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
  - The Makefile now includes --maxrregcount=32 for gemm_q4 and transpose,
    FLASH_BLOCK_SIZE=64 for sm_121, and warp shuffle reductions in
    rmsnorm and scaled_softmax.
  - Acceptance: libkernels.so builds without errors.
  - Dependencies: none.

- [x] T404.2 Benchmark kernel optimizations vs pre-Wave-8  Owner: TBD  Est: 1h
  - Build bench_tps with new kernels.
  - Run 3 times and compare with the 166 tok/s baseline.
  - If improvement is measurable, document per-kernel delta.
  - Acceptance: Results documented. Any improvement noted.
  - Dependencies: T404.1.

- [x] S404.2.1 Kernel optimization parity test  Owner: TBD  Est: 30m
  - Run go test on compute/ and internal/cuda/kernels/ on DGX.
  - Verify no correctness regression from register tuning or shared memory changes.
  - Acceptance: All tests pass.
  - Dependencies: T404.1.

### E405: BFloat16/Float16 Inference Path

Note: GB10 Blackwell supports native FP16 at 2x FP32 throughput and BF16
at similar rates. The float16 package (github.com/zerfoo/float16) provides
Go BFloat16 and Float16 types. cublasGemmEx purego wrapper already supports
BF16/FP16 compute types. The goal is to run inference with half-precision
weights and FP32 accumulation to halve memory bandwidth (the primary
bottleneck on LPDDR5x).

- [x] T405.1 Add BFloat16 weight loading to GGUF loader  Owner: TBD  Est: 2h
  - Detect BF16 weight type in GGUF files (GGMLTypeBF16 if it exists, or
    convert F32 weights to BF16 at load time when a --bf16 flag is set).
  - Store weights as float16.BFloat16 tensors using existing BFloat16Storage
    or create BFloat16Storage if it does not exist.
  - File: model/gguf/loader.go, tensor/storage.go.
  - Acceptance: Model loads with BF16 weights. Memory usage halved vs F32.
  - Dependencies: none.

- [x] T405.2 Wire BFloat16 MatMul through cublasGemmEx  Owner: TBD  Est: 3h
  - When weights are BF16 and activations are FP32, use cublasGemmEx with
    CUDA_R_16BF input type and CUDA_R_32F compute/output type (mixed precision).
  - The cublasGemmEx purego wrapper (internal/cublas/cublas_purego.go) already
    supports this via the computeType parameter.
  - Add dispatch logic in GPUEngine.MatMul to detect BF16 storage and route
    to GemmEx instead of Sgemm.
  - File: compute/gpu_engine.go (MatMul path), internal/cublas/cublas_purego.go.
  - Acceptance: BF16 weights x FP32 activations use GemmEx. Output correct
    (max rel error < 1e-3 vs FP32 reference).
  - Dependencies: T405.1.

- [x] T405.3 Add FP16 kernel variants for element-wise ops  Owner: TBD  Est: 3h
  - Add __half variants of elementwise kernels (add, mul, rmsnorm, etc.) in
    internal/cuda/kernels/. Use __half2 for 2-wide SIMD where possible.
  - Add Go dispatch for FP16 via the purego path.
  - Acceptance: FP16 kernels compile for sm_121. Parity with FP32 (rel error < 1e-3).
  - Dependencies: none.

- [x] T405.4 Add full FP16 inference path  Owner: TBD  Est: 4h
  - Enable running the entire forward pass in FP16: weights, activations,
    and intermediates all in FP16. Use FP32 accumulation in reductions
    (RMSNorm, Softmax) to avoid precision loss.
  - Add --dtype=fp16 flag to bench_tps.
  - File: compute/gpu_engine.go, generate/generator.go.
  - Acceptance: bench_tps --dtype=fp16 produces coherent output. Throughput
    measured and documented.
  - Dependencies: T405.2, T405.3.

- [x] S405.4.1 BF16/FP16 parity and benchmark  Owner: TBD  Est: 1h
  - Compare output quality: FP32 vs BF16 vs FP16 at temperature=0 for 50 tokens.
  - Measure tok/s for each precision. Document results.
  - Acceptance: BF16/FP16 output coherent. Throughput improvement documented.
    Expected: ~1.5-2x improvement from halved memory bandwidth.
  - Dependencies: T405.4.

- [x] T405.5 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S405.4.1.

### E406: FP8 Inference Path

Note: GB10 Blackwell (sm_121) supports FP8 at 4x FP32 throughput. Two
FP8 formats exist: E4M3 (4-bit exponent, 3-bit mantissa, better for weights)
and E5M2 (5-bit exponent, 2-bit mantissa, better for activations). The
float8 package (github.com/zerfoo/float8) provides Go FP8 types. FP8
inference requires per-tensor scaling factors to compensate for the very
limited dynamic range (E4M3: [-448, 448], E5M2: [-57344, 57344]).

cuBLAS FP8 support uses cublasLtMatmul (not cublasGemmEx). This requires
wrapping the cuBLASLt (lightweight) API via purego. FP8 MatMul typically
uses FP16 or FP32 accumulation to maintain output precision.

This epic depends on E405 (BF16/FP16) because FP8 compute pipelines use
FP16 accumulation and the FP16 kernel infrastructure must be in place first.

- [x] T406.1 Add FP8 type support to tensor package  Owner: TBD  Est: 2h
  - Add FP8E4M3Storage and FP8E5M2Storage to tensor/storage.go using the
    float8 package types.
  - Support Len, Slice, Set, DeviceType for both formats.
  - Add conversion functions: FP32 to FP8E4M3 (with per-tensor scale),
    FP8E4M3 to FP32 (with per-tensor scale).
  - File: tensor/storage.go or tensor/fp8_storage.go.
  - Acceptance: go test passes. Round-trip conversion FP32->FP8->FP32
    has max rel error < 0.1 (FP8 is very lossy).
  - Dependencies: none.

- [x] T406.2 Add FP8 weight quantization to GGUF loader  Owner: TBD  Est: 2h
  - Add --fp8 flag to convert F32 or BF16 weights to FP8E4M3 at load time.
  - Compute per-tensor absmax scale factor: scale = max(abs(tensor)) / 448.
  - Store scale factors alongside quantized weights.
  - File: model/gguf/loader.go.
  - Acceptance: Model loads with FP8 weights. Memory usage is 1/4 of F32.
  - Dependencies: T406.1.

- [x] T406.3 Create purego wrappers for cublasLt API  Owner: TBD  Est: 4h
  - Wrap via purego dlopen of libcublasLt.so:
    cublasLtCreate, cublasLtDestroy, cublasLtMatmulDescCreate,
    cublasLtMatmulDescSetAttribute, cublasLtMatrixLayoutCreate,
    cublasLtMatmul, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulAlgoGetHeuristic.
  - Add cublasLt.Available() runtime guard.
  - Follow the established purego dlopen pattern from internal/cuda/runtime_purego.go.
  - File: internal/cublas/cublaslt_purego.go.
  - Acceptance: Wrappers compile without CGo. Available() returns true on DGX.
  - Dependencies: none.

- [x] T406.4 Wire FP8 MatMul through cublasLtMatmul  Owner: TBD  Est: 4h
  - When weights are FP8E4M3 and activations are FP16 or FP32:
    use cublasLtMatmul with CUDA_R_8F_E4M3 input type, CUDA_R_16F or
    CUDA_R_32F compute type, and per-tensor scaling via alpha/beta params.
  - Handle the matmul descriptor setup: set scale factors via
    cublasLtMatmulDescSetAttribute (CUBLASLT_MATMUL_DESC_A_SCALE_POINTER).
  - Add dispatch logic in GPUEngine.MatMul to detect FP8 storage.
  - File: compute/gpu_engine.go, internal/cublas/cublaslt_purego.go.
  - Acceptance: FP8 weights x FP16 activations use cublasLtMatmul. Output
    correct (max rel error < 0.05 vs FP32 reference due to FP8 quantization).
  - Dependencies: T406.2, T406.3, E405 (FP16 kernels).

- [x] T406.5 Add FP8 element-wise kernel variants  Owner: TBD  Est: 3h
  - Add __nv_fp8_e4m3 kernel variants for element-wise ops where beneficial.
  - For most ops (RMSNorm, Softmax), keep FP16/FP32 accumulation and only
    use FP8 for weight storage, not compute. The primary gain is from
    reduced memory reads, not faster compute.
  - Add dequantize-on-load kernel: reads FP8 from global memory, converts
    to FP16 in registers, computes in FP16.
  - File: internal/cuda/kernels/fp8_ops.cu.
  - Acceptance: Kernels compile for sm_121. Parity with FP16 path.
  - Dependencies: T406.1, T405.3 (FP16 kernels).

- [x] T406.6 Add full FP8 inference path  Owner: TBD  Est: 4h
  - Enable running inference with FP8 weights: load FP8, dequantize to
    FP16 for element-wise ops, use cublasLtMatmul for MatMul.
  - Add --dtype=fp8 flag to bench_tps.
  - Compute and cache per-tensor scale factors during model loading.
  - File: compute/gpu_engine.go, generate/generator.go.
  - Acceptance: bench_tps --dtype=fp8 produces coherent output. Throughput
    measured and documented.
  - Dependencies: T406.4, T406.5.

- [x] S406.6.1 FP8 parity and benchmark  Owner: TBD  Est: 1h  2026 03 13  NOTE: Benchmarked on DGX. FP16 124.50 tok/s (17% slower than F32 due to conversion overhead). FP8 1.45 tok/s (arena thrashing, degenerate output). Acceptance NOT fully met -- FP8 not coherent, no throughput improvement. See docs/updates.md for full results.
  - Compare output quality: FP32 vs BF16 vs FP16 vs FP8 at temp=0 for 50 tokens.
  - Measure tok/s for each precision. Document results.
  - FP8 output may diverge more from FP32 due to quantization noise.
  - Acceptance: FP8 output coherent (may differ from FP32 in word choice
    but must be grammatically valid and on-topic). Throughput improvement
    documented. Expected: ~2-4x over FP32 from quartered memory bandwidth.
  - Dependencies: T406.6.

- [x] T406.7 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S406.6.1.

---

## 4. Parallel Work

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Regression Bisect | E401 | Critical path. Must complete first. |
| Track B: CUDA Graph D2H | E402 (T402.1-T402.4) | 4 independent D2H elimination tasks. |
| Track C: Q4_K Validation | E403 | Independent. Can run parallel with Track B. |
| Track D: Kernel Benchmarks | E404 | Independent. Can run parallel with all. |
| Track E: BF16/FP16 | E405 | Independent. Can run parallel with Tracks B-D. |
| Track F: FP8 | E406 | Depends on E405 (FP16 kernels). Last in sequence. |

Sync points:
- After Track A (E401): re-baseline. All subsequent work builds on recovered perf.
- After Track B (T402.1-T402.4): T402.5 (enable graph) unblocks.
- After Track E (T405.1-T405.3): T405.4 (full FP16 path) unblocks.
- After Track E complete: Track F (FP8) unblocks.
- After all tracks: final tok/s measurement across all precisions.

Maximum parallelism:
- Wave 1: T401.1 (regression bisect) + T402.1-T402.4 (D2H elimination, 4 parallel)
  + T403.1 (Q4_K debug) + T404.1 (kernel rebuild). T401.1 is highest priority.
- Wave 2: T401.2 (fix regression) + T402.5 (enable graph) + T403.2 (fix Q4_K)
  + T404.2 (benchmark kernels) + T405.1 (BF16 loading) + T405.3 (FP16 kernels).
- Wave 3: T405.2 (BF16 MatMul) + T405.4 (full FP16 path) + T406.1 (FP8 types)
  + T406.3 (cublasLt wrappers) + verification tasks.
- Wave 4: T406.2 (FP8 quantization) + T406.4 (FP8 MatMul) + T406.5 (FP8 kernels)
  + S405.4.1 (BF16/FP16 benchmark).
- Wave 5: T406.6 (full FP8 path) + S406.6.1 (FP8 benchmark) + final verification.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M78: Regression recovered | E401 | bench_tps >= 188.92 tok/s |
| M79: CUDA graph operational | E402 | Graph capture succeeds, replay faster than per-op |
| M80: Q4_K fused GEMV validated | E403 | Q4_K path >= Q4_0 path in tok/s |
| M81: Surpass Ollama | E401-E404 | bench_tps > 197.21 tok/s |
| M82: Half-precision inference | E405 | BF16/FP16 path produces coherent output with measured speedup |
| M83: FP8 inference | E406 | FP8 path produces coherent output with ~2-4x throughput over FP32 |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R401 | Regression is environmental (thermal, background load) not code | Cannot fix | Medium | Run baseline commit 5+ times at different times of day |
| R402 | D2H sites cannot be eliminated without major refactor | CUDA graph stays disabled | Medium | Document remaining sites. Consider graph capture for sub-regions only. |
| R403 | Q4_K fused GEMV slower than Q4_0 dequant+cuBLAS | No gain from fusion | Low | Q4_K halves bandwidth; should be faster for bandwidth-bound workloads |
| R404 | Register tuning causes subtle numerical differences | Incorrect output | Low | Run parity tests before and after on DGX |
| R405 | FP16 precision loss in reductions causes incoherent output | Bad quality | Medium | Use FP32 accumulation in RMSNorm/Softmax. Compare output with FP32 reference. |
| R406 | BF16 GGUF format not standardized | Cannot load BF16 models | Low | Convert F32 to BF16 at load time as fallback. |
| R407 | FP8 quantization noise too high for small models | Incoherent output | Medium | Per-channel scaling instead of per-tensor. Fall back to FP16 if quality degrades. |
| R408 | cublasLt API more complex than cublasGemmEx | Longer implementation | Low | API is well-documented. Follow NVIDIA samples for descriptor setup. |
| R409 | FP8 E4M3 dynamic range insufficient for outlier activations | Numerical overflow | Medium | Use E5M2 for activations, E4M3 for weights. Implement absmax clipping. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without build tags.
3. go test for the modified package passes with -race.
4. Commit passes pre-commit hooks.
5. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits. Do not let changes pile up.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- Benchmark: bench_tps on DGX Spark for performance-related changes.

---

## 8. Progress Log

### Change Summary -- 2026-03-13 (Add FP8 Epic)

Added E406: FP8 inference path (9 tasks: T406.1-T406.7, S406.6.1).
GB10 Blackwell supports FP8 (E4M3/E5M2) at 4x FP32 throughput. Requires
cublasLt purego wrappers (new API), per-tensor scaling factors, and FP16
accumulation. Depends on E405 (BF16/FP16 path must be complete first).
Added Track F, M83 milestone, R407-R409 risks.

### Change Summary -- 2026-03-13 (Add BF16/FP16 Epic)

Added E405: BFloat16/Float16 inference path (6 tasks: T405.1-T405.5, S405.4.1).
GB10 Blackwell supports native FP16 at 2x FP32 throughput. Half-precision
weights halve memory bandwidth (primary bottleneck on LPDDR5x). cublasGemmEx
purego wrapper already supports BF16/FP16 compute types.
Added Track E to parallel work, M82 milestone, R405/R406 risks.

### Change Summary -- 2026-03-13 (New Plan)

Created Phase 2 plan targeting surpassing Ollama (>197.21 tok/s). The prior
plan (89 tasks) is fully complete. This plan focuses on 4 highest-impact
next steps:
1. E401: Bisect and fix the 188->166 tok/s regression.
2. E402: Eliminate remaining D2H copies to enable CUDA graph capture.
3. E403: Validate fused dequant+GEMV Q4_K end-to-end.
4. E404: Benchmark Wave 8 kernel optimizations on DGX.

Completed epics from prior plan trimmed to docs/design.md:
E301-E307, E203-E205, E207-E209, E210-E215, E306.

### Prior Plan Completion -- 2026-03-13

All 89 tasks from the "Surpass Ollama" plan completed across 8 waves:
- Wave 1: D2H elimination (T301.1-3), OpenAI server (T305.1-3,6), transpose (T203.1)
- Wave 2: CUDA graph (T302.1), fused GEMV (T304.1), unified memory (T303.1-2), OpenAPI (T305.4), gather (T204.1)
- Wave 3: Engine wiring (T304.2, T203.2, T204.2), broadcasting (T205.1), OpenAPI endpoint (T305.5)
- Wave 4: Broadcasting wiring (T205.2), fused verification (T306.1), buffer layout (T207.2), cuBLAS GemmEx (T210.1), flash purego (T213.1)
- Wave 5: All purego wrappers (T210.2-3, T211.1, T212.1, T214.1-2, T215.1)
- Wave 6: All build tag removal (T211.2-3, T212.2-3, T213.2, T214.3-4, T215.2-3)
- Wave 7: All parity/verification tests + server integration test
- Wave 8: Kernel optimization (T209.2), megakernel abandonment (T208.1-2), purego parity (S211-215), output quality (T307.2), register tuning (T209.1)

### Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| Ollama GB10 | 197.21 | Measured 2026-03-12 |
| Zerfoo GB10 (pre-wave best) | 188.92 avg | 3-run avg 2026-03-12 |
| Zerfoo GB10 (post-wave) | 166.02 avg | 3-run avg 2026-03-13 |
| Zerfoo GB10 (managed memory) | 145.33 | With cudaMallocManaged |
| Zerfoo GB10 (CUDA graph attempt) | 99.51 | Graph capture fails |
| Zerfoo GB10 (initial GPU Q4) | 8.61 | Pre-optimization |
| Theoretical max (Q4 on GB10) | ~350-400 | 273 GB/s bandwidth ceiling |

---

## 9. Hand-off Notes

- **Prior plan:** All 89 tasks complete. Knowledge preserved in docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** bench_tps -model /home/ndungu/models/gemma3-gguf/model.gguf -tokens 256
  -prompt 'The meaning of life is' -device cuda
- **CUDA graph:** Enable with ZERFOO_ENABLE_CUDA_GRAPH=1. Partial capture in graph/cuda_graph.go.
- **Managed memory:** Enable with ZERFOO_ENABLE_MANAGED_MEM=1. 13% regression on GB10.
- **Megakernel:** Abandoned. See docs/updates.md "T208.1" section.
- **Pre-commit hook:** Rejects multi-directory commits.
- **Pre-wave baseline commit:** 388e60d (for regression bisect).
- **D2H copy sites remaining:**
  (1) layers/core/matmul.go:106,117 -- weight pointer caching
  (2) generate/tensor_cache.go:110-111 -- KV cache CPU fallback
  (3) layers/core/ffn.go:321 -- FFN split CPU fallback
  (4) grouped_query_attention.go:437 -- fused QK norm+RoPE CPU fallback
  (5) grouped_query_attention.go:888 -- splitMergedQKV CPU fallback
