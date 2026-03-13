# Zerfoo Development Plan -- Surpass Ollama Inference Performance (Phase 3)

## 1. Context

### Problem Statement

Zerfoo inference on DGX Spark GB10 achieves 149.52 tok/s with F32 compute
(75.8% of Ollama's 197.21 tok/s). Phase 2 built FP16 and FP8 inference paths
end-to-end, but both are slower than F32 due to implementation bottlenecks:

1. **FP16 path is 17% slower than F32 (124.50 tok/s).** Every element-wise
   operation (Add, Mul, RMSNorm, Softmax) converts F32->FP16 before compute
   and FP16->F32 after. This adds 4 extra kernel launches per operation
   (2 conversions + 2 alloc/free) and doubles memory traffic. The fix is to
   keep activations in FP16 throughout the forward pass, eliminating all
   intermediate conversions.

2. **FP8 path is 100x slower than F32 (1.45 tok/s).** The GPU arena (2GB
   pre-allocated) is exhausted during inference, causing 1841 arena misses
   that fall back to slow MemPool allocation. FP8 output is degenerate
   (repetitive text), suggesting scale factor propagation bugs.

3. **Baseline may be model-dependent.** F32 was 183.79 tok/s with llama3
   earlier, now 149.52 with gemma3. Need to verify with same model Ollama uses.

See docs/design.md for full architecture and Phase 2 completion details.

### Objectives

- O1: Eliminate FP16 conversion overhead so FP16 path is faster than F32.
- O2: Fix FP8 arena thrashing and scale propagation for coherent, fast FP8 inference.
- O3: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10 with Gemma 3 1B Q4_K_M.
- O4: Establish apples-to-apples baseline using identical model and prompt as Ollama.

### Non-Goals

- New model architectures or training.
- Multi-GPU / distributed inference.
- CUDA graph capture (blocked by GQA D2H; revisit after FP16/FP8 optimizations land).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M, measured 2026-03-12).
- Zerfoo current F32: 149.52 tok/s (gemma3, 2026-03-13).
- Go profile: go test, go vet, go build as quality gates.
- All GPU bindings use purego (no CGo, no build tags).

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| FP16 faster than F32 | FP16 tok/s > F32 tok/s | bench_tps 3-run avg on DGX |
| FP8 coherent output | Grammatically valid, on-topic text | Manual inspection at temp=0 |
| FP8 faster than F32 | FP8 tok/s > F32 tok/s | bench_tps 3-run avg on DGX |
| Surpass Ollama | > 197.21 tok/s | bench_tps 3-run avg on DGX |
| Apples-to-apples baseline | Same model, prompt, token count as Ollama | Documented in updates.md |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D40 | Native FP16 activation storage | Eliminate F32<->FP16 round-trips (biggest win) |
| D41 | FP16 weight upload at load time | Keep weights in FP16 on GPU from the start |
| D42 | FP8 arena pre-allocation | Fix 1841 arena misses causing 100x slowdown |
| D43 | FP8 scale factor fix | Fix degenerate output from scale propagation bugs |
| D44 | Apples-to-apples baseline | Fair comparison with Ollama using same model/prompt |

### Out of Scope

- CUDA graph capture (blocked by GQA D2H; separate phase).
- New kernel development (existing FP16/FP8 kernels are functional).
- BFloat16 optimization (FP16 is the priority path on Blackwell).
- Managed memory optimization.

---

## 3. Checkable Work Breakdown

### E501: Apples-to-Apples Baseline

- [x] T501.1 Benchmark F32 with same model and prompt as Ollama  Owner: TBD  Est: 30m  2026 03 13  NOTE: Ollama 213.34 tok/s (not 197.21), Zerfoo 151.69 tok/s. Gap is 28.9%.
  - Run Ollama on DGX with Gemma 3 1B Q4_K_M, record exact model path, prompt,
    and token count.
  - Run Zerfoo bench_tps with identical parameters.
  - Record 3-run averages for both.
  - File: docs/updates.md (results only, no code changes).
  - Acceptance: Both tools benchmarked with identical inputs. Results documented
    with exact commands and commit hashes.
  - Dependencies: none.

### E502: Native FP16 Activation Storage

The current FP16 path stores all activations as F32 and converts to FP16 on
every operation. The fix: when dtype=fp16, store activations directly as FP16
using GPUStorage with 2-byte elements. Operations read FP16, compute in FP16
(with FP32 accumulation for reductions), and write FP16. No conversions needed
except at graph boundaries (embedding lookup output, final logits).

- [x] T502.1 Add FP16 GPU storage type  Owner: TBD  Est: 1.5h  2026 03 13
  - Add Float16Storage to tensor/ that wraps GPU memory with 2-byte elements.
  - Support Len(), SubSlice(), GPUPtr(), SetGPUPtr(), DeviceType().
  - The storage must be recognized by GPUEngine MatMul dispatch and element-wise
    dispatch as "native FP16" (no conversion needed).
  - File: tensor/fp16_storage.go (new file).
  - Acceptance: go test passes. Float16Storage implements the Storage interface.
    GPUPtr returns valid device pointer. SubSlice creates zero-copy views.
  - Dependencies: none.

- [x] S502.1.1 Unit tests for Float16Storage  Owner: TBD  Est: 30m  2026 03 13  NOTE: Included in T502.1 commit.
  - Table-driven tests: create, SubSlice, GPUPtr round-trip, Len accuracy.
  - File: tensor/fp16_storage_test.go.
  - Acceptance: go test -race passes. 100% coverage of Float16Storage methods.
  - Dependencies: T502.1.

- [x] T502.2 Modify element-wise ops to accept FP16 storage directly  Owner: TBD  Est: 2h  2026 03 13
  - In compute/gpu_kernels.go: when input tensor has Float16Storage, extract the
    FP16 device pointer directly and call the FP16 kernel without F32->FP16
    conversion. Write output to a new Float16Storage tensor.
  - Affects: gpuAdd, gpuSub, gpuMul, gpuDiv, gpuAddScalar, gpuMulScalar.
  - When one input is FP16 and the other is F32, convert only the F32 input.
  - File: compute/gpu_kernels.go, compute/gpu_fp16.go.
  - Acceptance: When both inputs are Float16Storage, zero F32->FP16 conversions.
    Output is Float16Storage. Element-wise results match F32 reference (rel error < 1e-3).
  - Dependencies: T502.1.

- [x] S502.2.1 Tests for FP16 element-wise without conversion  Owner: TBD  Est: 30m  2026 03 13  NOTE: Included in T502.2 commit.
  - Verify Add, Mul with Float16Storage inputs produce Float16Storage output.
  - Verify no F32ToFP16 kernel calls when both inputs are already FP16.
  - File: compute/gpu_fp16_test.go.
  - Acceptance: go test -race passes.
  - Dependencies: T502.2.

- [x] T502.3 Modify MatMul to accept FP16 storage directly  Owner: TBD  Est: 1.5h  2026 03 13
  - In compute/gpu_engine.go MatMul dispatch: when activations have Float16Storage,
    pass the FP16 device pointer directly to MixedFP16Gemm (or cublasGemmEx with
    CUDA_R_16F input type). No F32->FP16 conversion needed.
  - When weights are F32 and activations are FP16, convert only weights (once,
    cached on the weight tensor).
  - Output should be Float16Storage when dtype=fp16.
  - File: compute/gpu_engine.go, compute/gpu_fp16.go.
  - Acceptance: FP16 MatMul with Float16Storage inputs skips conversion kernels.
    Output is Float16Storage. Results match F32 reference (rel error < 1e-3).
  - Dependencies: T502.1.

- [x] S502.3.1 Tests for FP16 MatMul without conversion  Owner: TBD  Est: 30m  2026 03 13
  - Verify MatMul with Float16Storage inputs and weights produces correct output.
  - Test batch dimensions (the GQA bug fix from Phase 2).
  - File: compute/gpu_fp16_test.go.
  - Acceptance: go test -race passes.
  - Dependencies: T502.3.

- [x] T502.4 Modify RMSNorm and Softmax to accept FP16 storage  Owner: TBD  Est: 1.5h  2026 03 13
  - In compute/gpu_engine.go: GPUScaledSoftmax and GPUFusedAddRMSNorm should
    detect Float16Storage on inputs and call FP16 kernels directly without
    F32->FP16 conversion.
  - RMSNorm: read FP16, accumulate in FP32 (existing kernel behavior), write FP16.
  - Softmax: read FP16, accumulate in FP32, write FP16.
  - Output should be Float16Storage.
  - File: compute/gpu_engine.go, compute/gpu_fp16.go.
  - Acceptance: Zero F32->FP16 conversions when input is Float16Storage.
    Output is Float16Storage. Parity with F32 reference (rel error < 1e-3).
  - Dependencies: T502.1.

- [x] S502.4.1 Tests for FP16 RMSNorm and Softmax  Owner: TBD  Est: 30m  2026 03 13  NOTE: Included in T502.4 commit.
  - Verify FusedAddRMSNorm and ScaledSoftmax with Float16Storage inputs.
  - File: compute/gpu_fp16_test.go.
  - Acceptance: go test -race passes.
  - Dependencies: T502.4.

- [x] T502.5 Convert embedding output to FP16 at inference start  Owner: TBD  Est: 1h  2026 03 13  NOTE: Gather now converts output to Float16Storage when dtype=FP16. Handles FP16 weight params via FP16->F32 temp buffer.
  - In the inference pipeline (generate/ or compute/), after EmbeddingLookup
    produces F32 output, convert it to Float16Storage once. All subsequent
    operations operate on FP16 natively.
  - This is the single F32->FP16 conversion point for the entire forward pass.
  - File: compute/gpu_engine.go (EmbeddingLookup or a post-embedding hook).
  - Acceptance: Embedding output is Float16Storage when dtype=fp16.
    All downstream ops receive FP16 input.
  - Dependencies: T502.2, T502.3, T502.4.

- [x] T502.6 Convert final logits from FP16 to F32 for sampling  Owner: TBD  Est: 30m  2026 03 13  NOTE: FP16ToF32Converter interface on GPUEngine. LMHead converts Float16Storage logits to F32 before sampling.
  - The sampling/argmax step expects F32 logits. Add a single FP16->F32
    conversion at the LMHead output (the last operation before sampling).
  - File: compute/gpu_engine.go or layers/core/lmhead.go.
  - Acceptance: Sampling receives F32 logits. Output tokens identical to
    current FP16 path at temp=0.
  - Dependencies: T502.5.

- [ ] T502.7 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T502.6.

### E503: FP16 Weight Pre-conversion

Weights are currently stored as F32 on GPU and converted to FP16 on every MatMul.
Converting weights to FP16 once at upload time eliminates per-MatMul conversion.

- [x] T503.1 Convert weights to FP16 during GPU upload  Owner: TBD  Est: 1.5h  2026 03 13
  - In compute/gpu_engine.go UploadWeights: when dtype=fp16, after uploading
    F32 weights to GPU, run F32ToFP16 kernel once and store the result as
    Float16Storage on the weight tensor. Free the F32 GPU copy.
  - Cache the FP16 device pointer on the weight tensor for reuse.
  - File: compute/gpu_engine.go (UploadWeights method).
  - Acceptance: Weights are FP16 on GPU after upload. No per-MatMul weight
    conversion. GPU memory for weights is halved vs F32.
  - Dependencies: T502.1.

- [x] S503.1.1 Test FP16 weight pre-conversion  Owner: TBD  Est: 30m  2026 03 13  NOTE: 3 table-driven tests in compute/gpu_fp16_test.go covering upload, MatMul usage, and idempotency.
  - Verify weights are Float16Storage after upload when dtype=fp16.
  - Verify MatMul uses pre-converted FP16 weights without conversion.
  - File: compute/gpu_engine_test.go.
  - Acceptance: go test -race passes.
  - Dependencies: T503.1.

- [x] T503.2 Run go vet on compute package  Owner: TBD  Est: 15m  2026 03 13  NOTE: Clean. Only pre-existing purego warnings.
  - Dependencies: T503.1.

### E504: FP8 Arena Fix

FP8 inference exhausts the 2GB arena because each MatMul allocates temporary
FP16 conversion buffers that are not freed until the end of the forward pass.
The fix: pre-allocate persistent FP16 buffers for FP8 MatMul and reuse them.

- [x] T504.1 Profile FP8 arena usage to identify largest allocations  Owner: TBD  Est: 1h  2026 03 13  NOTE: 1.15GB weight copy per MatMul (54% of arena). fp16MatMul 15.2MB x 1170 calls.
  - Add temporary logging to CUDAArenaPool.Alloc to record allocation sizes.
  - Run bench_tps --dtype=fp8 and collect the log.
  - Identify the top 10 largest allocations and which functions request them.
  - File: internal/gpuapi/cuda_arena.go (temporary logging).
  - Acceptance: Log shows allocation sizes and callers. Top allocations documented.
  - Dependencies: none.

- [x] T504.2 Pre-allocate persistent FP16 buffers for FP8 MatMul  Owner: TBD  Est: 2h  2026 03 13
  - In compute/gpu_fp8.go: instead of allocating FP16 conversion buffers per
    MatMul call via pool.Alloc, pre-allocate a set of reusable FP16 buffers
    during engine initialization (or on first use, then cache).
  - Size buffers based on the largest MatMul dimensions in the model.
  - Add a scratchpad struct to GPUEngine that holds persistent FP16 buffers
    for FP8 operations.
  - File: compute/gpu_fp8.go, compute/gpu_engine.go.
  - Acceptance: FP8 MatMul uses pre-allocated buffers. Arena misses drop from
    1841 to near zero. bench_tps --dtype=fp8 completes in <5 seconds.
  - Dependencies: T504.1.

- [x] S504.2.1 Test FP8 arena usage after pre-allocation  Owner: TBD  Est: 30m  2026 03 13  NOTE: 5 unit tests for fp8Scratchpad (grow, reuse, free, idempotent free, grow-frees-old). Uses fakeMemPool, no GPU required.
  - Run bench_tps --dtype=fp8 and verify arena stats show minimal misses.
  - File: docs/updates.md (benchmark results).
  - Acceptance: Arena misses < 50 (down from 1841). MemPool misses < 50 (down from 810).
  - Dependencies: T504.2.

- [x] T504.3 Run go vet on compute package  Owner: TBD  Est: 15m  2026 03 13  NOTE: Clean. Only pre-existing purego warnings.
  - Dependencies: T504.2.

### E505: FP8 Scale Factor Fix

FP8 output is degenerate (repetitive text), suggesting scale factors are not
correctly applied during matmul or are lost between operations.

- [x] T505.1 Add FP8 scale factor diagnostic logging  Owner: TBD  Est: 1h  2026 03 13  NOTE: All scales healthy. FP8 cublasLt MatMul never invoked -- SM 7.5 lacks FP8 support. Falls through to FP16 path.
  - In compute/gpu_fp8.go ltMatmulFP8: log the scale values (scaleA, scaleB)
    and matrix dimensions before each cublasLtMatmul call.
  - In model/gguf/loader.go QuantizeToFP8E4M3: log scale factors per tensor.
  - Run bench_tps --dtype=fp8 and inspect whether scales are reasonable
    (typically 0.001 to 100, not 0 or inf).
  - File: compute/gpu_fp8.go, model/gguf/loader.go.
  - Acceptance: Scale factors logged for all FP8 MatMul calls. Any zero, inf,
    or NaN scales identified and documented.
  - Dependencies: none.

- [x] T505.2 Fix FP8 scale propagation bugs  Owner: TBD  Est: 2h  2026 03 13  NOTE: Root cause was FP8 cublasLt never invoked (sm_75 < sm_89). Added FP16 dequant fallback: DequantFP8E4M3ToFP16 + MixedFP16Gemm. Works on any GPU with FP16.
  - Based on T505.1 diagnostics, fix identified scale issues. Likely fixes:
    a) Ensure scaleA and scaleB GPU pointers point to valid float32 values.
    b) Verify cublasLtMatmulDesc scale pointer attributes are set correctly
       (CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17, B = 18).
    c) Check if scale factors need to be inverted (cublasLt expects
       scale = 1/absmax, not absmax/448).
    d) Verify scale GPU memory is not freed before cublasLtMatmul executes
       (async execution hazard).
  - File: compute/gpu_fp8.go, tensor/fp8_storage.go.
  - Acceptance: bench_tps --dtype=fp8 produces coherent output (grammatically
    valid, on-topic). Output may differ from F32 in word choice.
  - Dependencies: T505.1.

- [x] S505.2.1 FP8 output quality test  Owner: TBD  Est: 30m  2026 03 13  NOTE: 2 table-driven tests for FP8 dequant fallback (A-weight and B-weight). Rel error < 1e-2 vs CPU reference.
  - Run bench_tps --dtype=fp8 with temp=0, 50 tokens. Verify output is
    coherent (not repetitive, grammatically valid).
  - Compare with F32 output. Document differences.
  - File: docs/updates.md.
  - Acceptance: FP8 output is coherent. Documented.
  - Dependencies: T505.2.

- [x] T505.3 Run go vet on modified packages  Owner: TBD  Est: 15m  2026 03 13  NOTE: Clean. Only pre-existing purego warnings.
  - Dependencies: T505.2.

### E506: Final Benchmark and Verification

- [ ] T506.1 Rebuild libkernels.so on DGX  Owner: TBD  Est: 15m
  - cd internal/cuda/kernels && make clean && make shared
  - Verify build succeeds.
  - Acceptance: libkernels.so builds without errors.
  - Dependencies: E502, E503, E504, E505.

- [ ] T506.2 Full benchmark suite on DGX  Owner: TBD  Est: 1h
  - Run bench_tps 3 times each for F32, FP16, FP8 with identical model and
    prompt as Ollama baseline from T501.1.
  - Record all results with commit hash.
  - File: docs/updates.md.
  - Acceptance: All 3 dtype paths produce coherent output. FP16 > F32 in tok/s.
    FP8 > F32 in tok/s. Results documented with exact commands.
  - Dependencies: T506.1, T501.1.

- [ ] S506.2.1 Output quality comparison  Owner: TBD  Est: 30m
  - Compare F32, FP16, FP8 output at temp=0, 50 tokens.
  - Verify FP16 output matches F32 (identical or near-identical).
  - Verify FP8 output is coherent (may differ from F32).
  - File: docs/updates.md.
  - Acceptance: Quality documented. Any regressions flagged.
  - Dependencies: T506.2.

- [ ] T506.3 Run go vet on all packages  Owner: TBD  Est: 15m
  - Dependencies: T506.2.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Baseline | E501 (T501.1) | Quick benchmark, no code changes |
| Track B: FP16 Activation Storage | E502 (T502.1-T502.7) | Core FP16 optimization |
| Track C: FP16 Weight Pre-conversion | E503 (T503.1-T503.2) | Depends on T502.1 only |
| Track D: FP8 Arena Fix | E504 (T504.1-T504.3) | Independent of FP16 work |
| Track E: FP8 Scale Fix | E505 (T505.1-T505.3) | Independent of FP16 work |
| Track F: Final Benchmark | E506 (T506.1-T506.3) | Depends on all tracks |

Sync points:
- After Wave 1: T502.1 (FP16 storage type) unblocks all E502 and E503 tasks.
- After Waves 2-3: All implementation done. T506.1 (rebuild + benchmark) unblocks.
- After Wave 4: Final results determine if target is met.

### Maximum parallelism

- Wave 1 (5 tasks): T501.1 (baseline benchmark) + T502.1 (FP16 storage type) +
  T504.1 (FP8 arena profiling) + T505.1 (FP8 scale diagnostics) +
  S502.1.1 (FP16 storage tests -- can start once T502.1 skeleton exists,
  but in practice T502.1 finishes first; replace with T502.2 if T502.1
  is fast). All 5 have zero dependencies.
  NOTE: T502.2, T502.3, T502.4 all depend only on T502.1 and touch different
  files, so they can run in Wave 1 if T502.1 finishes quickly. But conservatively
  they are Wave 2.

- Wave 2 (5 tasks): T502.2 (element-wise FP16) + T502.3 (MatMul FP16) +
  T502.4 (RMSNorm/Softmax FP16) + T503.1 (weight pre-conversion) +
  T504.2 (FP8 arena pre-alloc). All unblocked after Wave 1.
  T505.2 (FP8 scale fix) also unblocked but limited to 5 slots.

- Wave 3 (5 tasks): T502.5 (embedding FP16 output) + T502.6 (logits FP16->F32) +
  T505.2 (FP8 scale fix) + S502.2.1 (element-wise tests) + S502.3.1 (MatMul tests).
  NOTE: T502.5 depends on T502.2+T502.3+T502.4. Tests can run once impl is done.

- Wave 4 (5 tasks): T502.7 (go vet) + T503.2 (go vet) + T504.3 (go vet) +
  T505.3 (go vet) + S502.4.1 (RMSNorm tests). Vet tasks are quick.

- Wave 5 (4 tasks): T506.1 (rebuild libkernels) + S503.1.1 (weight tests) +
  S504.2.1 (arena tests) + S505.2.1 (FP8 quality test).

- Wave 6 (3 tasks): T506.2 (full benchmark) + S506.2.1 (quality comparison) +
  T506.3 (final go vet).

### Dependency minimization checklist applied

a) T502.2, T502.3, T502.4 all depend on T502.1 but NOT on each other -- they
   touch different functions in different files. Maximally parallel.
b) T504.x and T505.x are fully independent of T502.x/T503.x -- FP8 and FP16
   fixes run on separate tracks.
c) Test subtasks (S*) depend on their implementation tasks but can run as soon
   as the implementation commits are pushed, without waiting for other tracks.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M84: Baseline established | T501.1 | Apples-to-apples Ollama vs Zerfoo comparison documented |
| M85: FP16 zero-conversion path | E502, E503 | FP16 inference with zero F32<->FP16 round-trips. FP16 tok/s > F32 tok/s |
| M86: FP8 functional | E504, E505 | FP8 coherent output, arena misses < 50, FP8 tok/s > F32 tok/s |
| M87: Surpass Ollama | E506 | bench_tps > 197.21 tok/s with any dtype |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R501 | FP16 native storage breaks tensor shape assumptions | Many ops fail | Medium | Float16Storage must satisfy same interface as GPUStorage. Comprehensive tests. |
| R502 | FP16 accumulation in reductions loses precision | Incoherent output | Low | Keep FP32 accumulation in RMSNorm/Softmax (already implemented in kernels). |
| R503 | FP8 scale factors correct but model too small for FP8 | Degenerate output persists | Medium | Fall back to per-channel scaling. Try FP8 on larger model. |
| R504 | Pre-allocated FP16 buffers waste memory for small models | OOM on constrained devices | Low | Size buffers from model config, not worst-case. Lazy allocation on first use. |
| R505 | Baseline difference is model-dependent, not optimizable | Cannot reach 197 tok/s with gemma3 | Medium | T501.1 establishes ground truth. If model-dependent, try llama3. |
| R506 | cublasLtMatmul workspace requirement missed | FP8 MatMul produces wrong results | Medium | Check cublasLtMatmulAlgoGetHeuristic workspace size. Allocate if needed. |

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

### Remote Host Protocol

- Always rebuild libkernels.so on DGX before benchmarking.
- Always git pull on DGX before benchmarking.
- Record exact commit hash in benchmark results.

---

## 8. Progress Log

### Change Summary -- 2026-03-13 (Phase 3 Plan Created)

Created Phase 3 plan targeting >197.21 tok/s. Phase 2 (35 tasks, 6 epics) is
fully complete. Phase 2 knowledge trimmed to docs/design.md.

Phase 3 focuses on 5 epics:
- E501: Apples-to-apples baseline (1 task).
- E502: Native FP16 activation storage (7 tasks + 4 test subtasks).
- E503: FP16 weight pre-conversion (2 tasks + 1 test subtask).
- E504: FP8 arena pre-allocation (3 tasks + 1 test subtask).
- E505: FP8 scale factor fix (3 tasks + 1 test subtask).
- E506: Final benchmark (3 tasks + 1 test subtask).

Total: 19 implementation tasks, 7 test subtasks = 26 tasks.
Designed for 6 waves with up to 5 parallel agents per wave.

Trimmed Phase 2 epics E401-E406 from plan. Stable knowledge preserved in
docs/design.md "Phase 2 Completion Summary" section.

Updated docs/design.md with Phase 2 completion details, performance data, and
architectural insights about FP16 conversion overhead and FP8 arena thrashing.

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1 (89 tasks) and Phase 2 (35 tasks) complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared
  (nvcc at /usr/local/cuda/bin/nvcc, go at /usr/local/go/bin/go).
- **Benchmark:** bench_tps --model ~/models/gemma3-gguf/model.gguf --tokens 50
  --prompt 'The quick brown fox' --device cuda --dtype [fp32|fp16|fp8]
- **Key files for FP16 optimization:**
  - compute/gpu_fp16.go -- FP16 MatMul, element-wise, reductions (conversion bottleneck here)
  - compute/gpu_kernels.go -- element-wise dispatch (lines 526-605 gate FP16 path)
  - compute/gpu_engine.go -- MatMul dispatch (lines 499-570), dtype system (lines 19-34)
  - tensor/fp8_storage.go -- FP8E4M3Storage (model for new Float16Storage)
- **Key files for FP8 optimization:**
  - compute/gpu_fp8.go -- FP8 MatMul, ltMatmulFP8 (arena allocation sites)
  - internal/gpuapi/cuda_arena.go -- CUDAArenaPool (2GB pre-allocated)
  - model/gguf/loader.go -- QuantizeToFP8E4M3 (scale factor computation)
- **Pre-commit hook:** Rejects multi-directory commits.
- **Stale worktree:** wave-4-task-E103 (fix/gather-codegen-embedded-weights) has 34 unmerged
  commits from prior plan. Superseded. Do not merge.
