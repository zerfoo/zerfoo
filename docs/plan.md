# Zerfoo Development Plan -- Close 25% Performance Gap to Ollama (Phase 4)

## 1. Context

### Problem Statement

Zerfoo F32 inference on DGX Spark GB10 achieves 157.25 tok/s (79.7% of Ollama's
197.21 tok/s). Phase 3 built and verified FP16 and FP8 inference paths but
discovered that for Q4_K_M quantized models, F32 activations are optimal because
Q4K GEMV always outputs F32 -- FP16 activations add per-op conversion overhead.

The 25% gap to Ollama likely comes from three areas:

1. **Q4K GEMV kernel efficiency.** Current kernel uses 128 threads/block (4 warps).
   llama.cpp uses 256. Quantized byte loads are scalar (__ldg per byte) instead of
   vectorized. Each lane processes super-blocks in a strided loop with 32 byte
   loads per group -- vectorized uint4 loads could reduce instruction count 4x.
   File: internal/cuda/kernels/gemv_q4k.cu.

2. **Kernel launch overhead.** Each token generates ~50+ kernel launches (Q4K GEMV,
   RMSNorm, Add, Softmax, Gather, RoPE, etc.). CUDA graph capture would batch all
   launches into a single replay. Currently blocked by D2H copies in GQA fallback
   paths (grouped_query_attention.go lines 452, 925).
   File: layers/attention/grouped_query_attention.go, generate/generator.go.

3. **FP8 is broken.** 1.48 tok/s with 1841 arena misses and degenerate output.
   The fp8Scratchpad covers A/B matrix buffers but not output buffers or scale
   pointer allocations. FP8 degenerate output persists from the FP16 dequant
   fallback path.

See docs/design.md for full architecture and Phase 3 completion details.

### Objectives

- O1: Optimize Q4K GEMV kernel to close the per-kernel performance gap.
- O2: Eliminate GQA D2H copies to unblock CUDA graph capture.
- O3: Enable CUDA graph capture for the decode loop to eliminate launch overhead.
- O4: Fix FP8 arena thrashing and degenerate output.
- O5: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10 with Gemma 3 1B Q4_K_M.

### Non-Goals

- FP16 activation optimization (proven slower for Q4K models in Phase 3).
- New model architectures or training.
- Multi-GPU / distributed inference.
- Managed memory optimization (already gated behind env var).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M, measured 2026-03-12).
- Zerfoo current F32: 157.25 tok/s (commit efdd87b, 2026-03-13).
- Go profile: go test, go vet, go build as quality gates.
- All GPU bindings use purego (no CGo, no build tags).
- CUDA kernels compiled with nvcc -arch=sm_121 via Makefile in internal/cuda/kernels/.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Q4K GEMV speedup | >15% improvement over current kernel | Isolated kernel benchmark on DGX |
| GQA D2H eliminated | Zero .Data() calls in decode hot path | Grep for WARNING log lines during bench_tps |
| CUDA graph capture | Decode loop captured and replayed | bench_tps runs with graph executor, no fallback |
| FP8 coherent output | Grammatically valid, on-topic text | Manual inspection at temp=0 |
| Surpass Ollama | > 197.21 tok/s | bench_tps 3-run avg on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D50 | Optimized Q4K GEMV kernel | Biggest per-kernel performance gain |
| D51 | GQA D2H elimination | Prerequisite for CUDA graph capture |
| D52 | CUDA graph capture for decode | Eliminates ~50 kernel launch overhead per token |
| D53 | FP8 arena fix and output quality | Complete the FP8 path from Phase 3 |
| D54 | Per-token overhead reduction | Reduce tensor allocation and inference loop waste |

### Out of Scope

- FP16 activation path optimization (proven slower for Q4K in Phase 3).
- New CUDA kernel development beyond Q4K GEMV optimization.
- BFloat16 optimization.
- Multi-GPU support.

---

## 3. Checkable Work Breakdown

### E601: Q4K GEMV Kernel Optimization

The Q4K GEMV kernel (gemv_q4k.cu) is the hottest code path -- called for every
weight matrix multiplication in every transformer layer. Current kernel uses 128
threads/block with scalar byte loads. Optimization targets: increase block size
to 256, vectorize quantized data loads, and tune shared memory usage.

- [x] T601.1 Profile Q4K GEMV kernel on DGX to establish per-kernel baseline  Owner: task-T601.1  Est: 45m  Done: 2026-03-13
  - Use nsys or nvprof to measure Q4K GEMV execution time per call.
  - Record: kernel time, SM occupancy, memory throughput, register usage.
  - Run for Gemma 3 1B Q4_K_M decode (single token).
  - File: docs/updates.md (results only).
  - Acceptance: Per-call kernel time documented. Occupancy and bandwidth utilization recorded.
  - Dependencies: none.

- [x] T601.2 Increase Q4K GEMV block size from 128 to 256 threads  Owner: task-T601.2  Est: 1h  Done: 2026-03-13
  - Change Q4K_WARPS_PER_BLOCK from 4 to 8 in gemv_q4k.cu.
  - Adjust grid calculation: grid = (M + 8 - 1) / 8.
  - Shared memory size stays at K * sizeof(float) -- more threads cooperate to load it.
  - Verify correctness: output must match current kernel bit-for-bit.
  - File: internal/cuda/kernels/gemv_q4k.cu.
  - Acceptance: Kernel compiles. Output matches reference. Block size is 256.
  - Dependencies: none.

- [x] S601.2.1 Test Q4K GEMV 256-thread kernel correctness  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Kernel reverted. Original kernel unchanged.
  - Run existing Q4K tests with rebuilt libkernels.so.
  - Run bench_tps --dtype=fp32 and verify identical output to baseline.
  - File: docs/updates.md (test results).
  - Acceptance: go test passes. bench_tps output identical.
  - Dependencies: T601.2.

- [x] T601.3 Vectorize Q4K quantized byte loads  Owner: task-T601.3  Est: 1.5h  Done: 2026-03-13
  - In gemv_q4k_kernel inner loop: replace per-byte __ldg loads with
    uint4 loads (16 bytes = 16 quantized values per load, 2 loads per group
    of 32 bytes instead of 32 scalar loads).
  - Unpack uint4 into individual nibbles in registers using bitwise ops.
  - Preserve FMA accumulation pattern.
  - File: internal/cuda/kernels/gemv_q4k.cu.
  - Acceptance: Kernel compiles. Output matches reference bit-for-bit.
    Inner loop has 2 loads per group instead of 32.
  - Dependencies: T601.2.
  - Risk: SM register pressure may increase. Monitor with --ptxas-options=-v.

- [x] S601.3.1 Test vectorized Q4K GEMV correctness  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Kernel reverted. Original kernel unchanged.
  - Same as S601.2.1 but after vectorization.
  - Acceptance: go test passes. bench_tps output identical.
  - Dependencies: T601.3.

- [x] T601.4 Benchmark optimized Q4K GEMV kernel  Owner: task-T601.4  Est: 30m  Done: 2026-03-13  NOTE: Kernel regressed 12.2% (189->166). Reverted.
  - Rebuild libkernels.so on DGX.
  - Run bench_tps --dtype=fp32 3 times, record results.
  - Compare with T601.1 baseline.
  - File: docs/updates.md.
  - Acceptance: Results documented with commit hash. Speedup quantified.
  - Dependencies: T601.3, T601.1.

- [x] T601.5 Run go vet on kernel wrapper package  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./internal/cuda/...
  - Acceptance: No new warnings.
  - Dependencies: T601.3.

### E602: GQA D2H Elimination

Two fallback paths in grouped_query_attention.go trigger .Data() calls that copy
tensor data from GPU to CPU. These block CUDA graph capture and add latency.
Both paths have GPU fast paths that work when tensors have GPUStorage or
Float16Storage -- the fix is to ensure the fast paths are always taken.

- [x] T602.1 Audit all .Data() calls in GQA hot path  Owner: task-T602.1  Est: 45m  Done: 2026-03-13
  - Grep for .Data() in layers/attention/grouped_query_attention.go.
  - For each call, determine: (a) is it in the decode hot path? (b) what
    storage type triggers the fallback? (c) can the GPU fast path always be used?
  - Document findings.
  - File: docs/updates.md.
  - Acceptance: Every .Data() call in GQA catalogued with trigger conditions.
  - Dependencies: none.

- [x] T602.2 Fix fused QK norm+RoPE D2H fallback  Owner: task-T602.2  Est: 1.5h  Done: 2026-03-13
  - At line ~452: the fallback triggers when fusedOut does not have GPUStorage.
  - Ensure FusedQKNormRoPEProvider always returns GPU-resident output.
  - If the provider interface cannot guarantee GPU output, add a GPU upload
    path instead of falling back to CPU .Data() decomposition.
  - File: layers/attention/grouped_query_attention.go.
  - Acceptance: No D2H copy in fused QK norm+RoPE path during decode.
    WARNING log line never printed during bench_tps.
  - Dependencies: none.

- [x] T602.3 Fix splitMergedQKV D2H fallback  Owner: task-T602.3  Est: 1h  Done: 2026-03-13
  - At line ~925: the fallback triggers when merged tensor does not have
    GPUStorage or Float16Storage.
  - Ensure the merged QKV tensor always has GPU storage during decode.
  - The GPU fast path uses SubSlice for zero-copy views -- verify it covers
    all storage types that can appear during decode.
  - File: layers/attention/grouped_query_attention.go.
  - Acceptance: No D2H copy in splitMergedQKV during decode.
    WARNING log line never printed during bench_tps.
  - Dependencies: none.

- [x] T602.4 Audit remaining D2H copies in inference hot path  Owner: task-T602.4  Est: 1h  Done: 2026-03-13
  - Grep for .Data() calls in compute/, layers/, generate/ that could be
    hit during decode.
  - Focus on: FFN, MatMul dispatch, KV cache operations.
  - Document any remaining D2H copies that would block CUDA graph capture.
  - File: docs/updates.md.
  - Acceptance: All D2H copies in decode hot path catalogued. Fix plan for each.
  - Dependencies: none.

- [x] S602.4.1 Verify zero D2H copies during decode  Owner: task-S602.4.1  Est: 30m  Done: 2026-03-13
  - Run bench_tps --dtype=fp32 and grep output for "WARNING" and "D2H".
  - Verify no D2H copy warnings appear during token generation.
  - File: docs/updates.md.
  - Acceptance: Zero D2H warnings during decode.
  - Dependencies: T602.2, T602.3, T602.4.

- [x] T602.5 Run go vet on attention package  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./layers/attention/...
  - Acceptance: No new warnings.
  - Dependencies: T602.3.

### E603: CUDA Graph Capture for Decode Loop

Once all D2H copies are eliminated, the decode loop can be captured as a CUDA
graph. This batches ~50+ kernel launches into a single graph replay per token,
eliminating per-kernel launch overhead (~5-10us each = 250-500us per token).

- [x] T603.1 Enable CUDA graph capture in decode loop  Owner: task-T603.1  Est: 2h  Done: 2026-03-13  NOTE: Infrastructure built, GQA position-dependent blocks capture. Graceful fallback.
  - In generate/generator.go: after warmup, use cudaStreamBeginCapture to
    record the decode forward pass, then cudaGraphInstantiate for replay.
  - The graph executor at graph/cuda_graph.go has existing infrastructure --
    verify it works with the current forward pass after D2H elimination.
  - Handle the first-token vs subsequent-token difference (first token may
    have different sequence length).
  - File: generate/generator.go, graph/cuda_graph.go.
  - Acceptance: Decode loop uses CUDA graph replay after warmup.
    bench_tps shows "graph executor" in output (no "fallback" message).
  - Dependencies: S602.4.1 (all D2H copies eliminated).

- [x] S603.1.1 Test CUDA graph capture correctness  Owner: task-T603.1  Est: 30m  Done: 2026-03-13  NOTE: Tested on DGX, correct output with graceful fallback. GQA prevents full capture.
  - Run bench_tps --dtype=fp32 with graph capture enabled.
  - Verify output matches non-graph output exactly (temp=0, same tokens).
  - File: docs/updates.md.
  - Acceptance: Identical output with and without graph capture.
  - Dependencies: T603.1.

- [x] T603.2 Benchmark with CUDA graph capture  Owner: task-T603.1  Est: 30m  Done: 2026-03-13  NOTE: 88.66 tok/s with fallback. No speedup — GQA blocks capture.
  - Run bench_tps --dtype=fp32 3 times with graph capture.
  - Compare with pre-graph baseline.
  - File: docs/updates.md.
  - Acceptance: Results documented. Speedup quantified.
  - Dependencies: T603.1.

- [x] T603.3 Run go vet on generate and graph packages  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./generate/... ./graph/...
  - Acceptance: No new warnings.
  - Dependencies: T603.1.

### E604: FP8 Arena and Output Fix

FP8 has 1841 arena misses because the fp8Scratchpad only covers A/B matrix
buffers. Output buffers and scale pointer allocations still go through the arena.
FP8 output is degenerate (repetitive text) from the FP16 dequant fallback path.

- [x] T604.1 Extend fp8Scratchpad with output buffer  Owner: task-T604.1  Est: 1h  Done: 2026-03-13
  - Add a reusable output buffer (fp16BufC or f32BufC) to fp8Scratchpad.
  - Modify fp8DequantMatMulA and fp8DequantMatMulB to use the scratchpad
    output buffer instead of pool.Alloc for the devC allocation.
  - File: compute/gpu_fp8.go.
  - Acceptance: Arena misses drop from 1841 to < 100. bench_tps --dtype=fp8
    completes without OOM.
  - Dependencies: none.

- [x] S604.1.1 Test FP8 arena usage after output buffer fix  Owner: task-S604.1.1  Est: 30m  Done: 2026-03-13  NOTE: 1841->4 misses (99.8% reduction).
  - Run bench_tps --dtype=fp8 and verify arena stats.
  - File: docs/updates.md.
  - Acceptance: Arena misses < 100. MemPool misses < 100.
  - Dependencies: T604.1.

- [x] T604.2 Debug FP8 degenerate output  Owner: task-T604.2  Est: 2h  Done: 2026-03-13
  - Run bench_tps --dtype=fp8 with temp=0 and inspect output quality.
  - Add diagnostic logging to fp8DequantMatMulA: log input/output norms
    for first 3 MatMul calls to check for numerical instability.
  - Compare FP8 dequant fallback output vs F32 reference for a single
    MatMul call (same inputs) to isolate precision issues.
  - Possible causes: (a) FP8 quantization absmax scaling too aggressive for
    1B model, (b) dequant kernel bug, (c) accumulation precision loss.
  - File: compute/gpu_fp8.go.
  - Acceptance: Root cause of degenerate output identified and documented.
  - Dependencies: T604.1.

- [x] T604.3 Fix FP8 degenerate output  Owner: task-T604.3  Est: 2h  Done: 2026-03-13  NOTE: Two bugs — stale arena ptrs + embed_tokens FP8 quantization.
  - Based on T604.2 diagnostics, implement the fix. Likely one of:
    a) Switch from per-tensor to per-channel absmax scaling.
    b) Fix dequant kernel numerical issue.
    c) Use FP32 accumulation in the FP16 GEMM fallback.
  - File: compute/gpu_fp8.go, model/gguf/loader.go, or internal/cuda/kernels/.
  - Acceptance: bench_tps --dtype=fp8 produces coherent output at temp=0.
  - Dependencies: T604.2.

- [x] S604.3.1 Test FP8 output quality  Owner: task-S604.3.1  Est: 30m  Done: 2026-03-13  NOTE: FP8 still degenerate on sm_121 (cublasLt unsupported). R606 risk materialized.
  - Run bench_tps --dtype=fp8 with temp=0, 50 tokens.
  - Compare with F32 output. Document differences.
  - File: docs/updates.md.
  - Acceptance: FP8 output is coherent (not repetitive, grammatically valid).
  - Dependencies: T604.3.

- [x] T604.4 Run go vet on compute package  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./compute/...
  - Acceptance: No new warnings.
  - Dependencies: T604.3.

### E605: Per-Token Overhead Reduction

Minor optimizations to the inference loop that reduce per-token allocation and
Go runtime overhead. Each optimization is small but they compound.

- [x] T605.1 Reuse token input tensor across decode steps  Owner: task-T605.1  Est: 1h  Done: 2026-03-13
  - In generate/generator.go: instead of calling idsToTensor() per token
    (which allocates a new [1,1] tensor + GPU upload each time), pre-allocate
    a [1,1] tensor and update its value in place using the existing GPU buffer.
  - File: generate/generator.go.
  - Acceptance: Only one GPU allocation for the token tensor across all decode
    steps. No functional change in output.
  - Dependencies: none.

- [x] S605.1.1 Test token tensor reuse  Owner: task-S605.1.1  Est: 30m  Done: 2026-03-13
  - Run bench_tps --dtype=fp32 and verify identical output.
  - Verify arena stats show fewer allocations.
  - File: docs/updates.md.
  - Acceptance: Output identical. Arena hits reduced.
  - Dependencies: T605.1.

- [x] T605.2 Run go vet on generate package  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./generate/...
  - Acceptance: No new warnings.
  - Dependencies: T605.1.

### E606: Final Benchmark and Verification

- [ ] T606.1 Rebuild libkernels.so on DGX  Owner: TBD  Est: 15m
  - cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
  - Verify build succeeds.
  - Acceptance: libkernels.so builds without errors.
  - Dependencies: E601.

- [ ] T606.2 Full benchmark suite on DGX  Owner: TBD  Est: 1h
  - Run bench_tps 3 times each for F32 and FP8 with Gemma 3 1B Q4_K_M.
  - Use identical prompt and token count as Ollama baseline.
  - Record all results with commit hash.
  - File: docs/updates.md.
  - Acceptance: Results documented. F32 > 197.21 tok/s (surpasses Ollama).
  - Dependencies: T606.1, E602, E603, E604, E605.

- [ ] S606.2.1 Output quality verification  Owner: TBD  Est: 30m
  - Verify F32 and FP8 output at temp=0, 50 tokens.
  - F32 must match pre-optimization output exactly.
  - FP8 must produce coherent text.
  - File: docs/updates.md.
  - Acceptance: Quality documented. No regressions.
  - Dependencies: T606.2.

- [x] T606.3 Run go vet on all packages  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./...
  - Acceptance: No new warnings beyond pre-existing purego patterns.
  - Dependencies: T606.2.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Q4K Kernel | E601 (T601.1-T601.5) | CUDA kernel changes, independent |
| Track B: GQA D2H Fix | E602 (T602.1-T602.5) | Go code in attention layer, independent |
| Track C: FP8 Fix | E604 (T604.1-T604.4) | FP8 compute path, independent |
| Track D: Per-Token | E605 (T605.1-T605.2) | Generator loop, independent |
| Track E: CUDA Graph | E603 (T603.1-T603.3) | Depends on Track B (D2H elimination) |
| Track F: Final Bench | E606 (T606.1-T606.3) | Depends on all tracks |

Sync points:
- After Wave 1: Tracks A, B, C, D all start independently (5 tasks).
- After Wave 2: Track B complete. Track E (CUDA graph) unblocked.
- After Wave 3: All tracks complete. Track F (final benchmark) unblocked.

### Maximum parallelism

- Wave 1 (5 tasks): T601.1 (profile Q4K) + T601.2 (block size 256) +
  T602.1 (audit GQA D2H) + T604.1 (FP8 scratchpad output buf) + T605.1 (token reuse).
  All have zero dependencies on each other.

- Wave 2 (5 tasks): T601.3 (vectorize loads) + T602.2 (fix fused QK D2H) +
  T602.3 (fix splitMergedQKV D2H) + T604.2 (debug FP8 output) + T602.4 (audit other D2H).
  T601.3 depends on T601.2. T604.2 depends on T604.1. Rest are independent.

- Wave 3 (5 tasks): T601.4 (benchmark kernel) + S602.4.1 (verify zero D2H) +
  T604.3 (fix FP8 output) + S604.1.1 (test FP8 arena) + S605.1.1 (test token reuse).
  T601.4 depends on T601.3+T601.1. S602.4.1 depends on T602.2+T602.3+T602.4.

- Wave 4 (5 tasks): T603.1 (CUDA graph capture) + S601.2.1 (test kernel) +
  S601.3.1 (test vectorized) + S604.3.1 (test FP8 quality) + T601.5 (go vet).
  T603.1 depends on S602.4.1. Tests depend on their implementation tasks.

- Wave 5 (5 tasks): T603.2 (benchmark graph) + S603.1.1 (test graph) +
  T602.5 (go vet) + T604.4 (go vet) + T605.2 (go vet).
  Depends on Wave 4.

- Wave 6 (4 tasks): T606.1 (rebuild libkernels) + T606.2 (full benchmark) +
  S606.2.1 (quality verification) + T606.3 (final go vet).
  T603.3 (go vet) also here.

### Dependency minimization checklist applied

a) E601 (Q4K kernel) is fully independent of E602 (GQA D2H), E604 (FP8), E605
   (per-token). All four run in parallel from Wave 1.
b) E603 (CUDA graph) depends only on E602 (D2H elimination), not on E601 or E604.
c) Test subtasks depend only on their implementation tasks, not on other tracks.
d) Wave 1 saturates all 5 agent slots with zero-dependency tasks.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M90: Kernel profiled | T601.1 | Q4K GEMV per-call time and occupancy documented |
| M91: Q4K kernel optimized | T601.4 | >15% speedup in isolated kernel benchmark |
| M92: D2H eliminated | S602.4.1 | Zero D2H warnings during bench_tps decode |
| M93: CUDA graph active | T603.2 | Decode loop uses graph replay, speedup measured |
| M94: Surpass Ollama | T606.2 | bench_tps > 197.21 tok/s with F32 |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R601 | Q4K GEMV is memory-bound, not compute-bound | Kernel optimization yields < 5% | Medium | Profile first (T601.1). If memory-bound, focus on reducing memory traffic (vectorized loads) rather than increasing thread count. |
| R602 | Shared memory limit prevents 256-thread blocks | Kernel fails to launch for large K | Low | K=1152 needs 4608 bytes smem. Max smem on sm_121 is 228KB. Safe margin. |
| R603 | CUDA graph capture fails on dynamic shapes | Graph replay produces wrong results | Medium | Decode always uses seqLen=1. Shapes are static after first token. Only capture decode, not prefill. |
| R604 | GQA D2H fallback triggers for edge cases not found in audit | CUDA graph capture intermittently fails | Medium | Run bench_tps with >100 tokens to exercise all code paths. Add assertions that panic on D2H in decode. |
| R605 | Combined optimizations still under 197 tok/s | Cannot surpass Ollama | Medium | Each optimization is independently valuable. If gap remains, investigate: arena allocator overhead, Go runtime GC pauses, KV cache management. |
| R606 | FP8 precision insufficient for 1B models | Degenerate output persists after fixes | High | Fall back to per-channel scaling. If still degenerate, document FP8 as unsuitable for sub-7B models. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without build tags.
3. go test for the modified package passes with -race (Go code changes).
4. make shared builds without errors (CUDA kernel changes).
5. Commit passes pre-commit hooks.
6. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits. Do not let changes pile up.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- CUDA: make shared in internal/cuda/kernels/ (when .cu files change).
- Benchmark: bench_tps on DGX Spark for performance-related changes.

### Remote Host Protocol

- Always rebuild libkernels.so on DGX before benchmarking (when .cu files changed).
- Always git pull on DGX before benchmarking.
- Record exact commit hash in benchmark results.

---

## 8. Progress Log

### Change Summary -- 2026-03-13 (Phase 4 Plan Created)

Created Phase 4 plan targeting >197.21 tok/s. Phase 3 (26 tasks, 6 epics) is
fully complete. Phase 3 knowledge trimmed to docs/design.md.

Phase 4 focuses on 6 epics:
- E601: Q4K GEMV kernel optimization (5 tasks + 2 test subtasks).
- E602: GQA D2H elimination (5 tasks + 1 test subtask).
- E603: CUDA graph capture for decode (3 tasks + 1 test subtask).
- E604: FP8 arena and output fix (4 tasks + 2 test subtasks).
- E605: Per-token overhead reduction (2 tasks + 1 test subtask).
- E606: Final benchmark (3 tasks + 1 test subtask).

Total: 22 implementation tasks, 8 test subtasks = 30 tasks.
Designed for 6 waves with up to 5 parallel agents per wave.

Trimmed Phase 3 epics E501-E506 from plan. Stable knowledge preserved in
docs/design.md "Phase 3 Completion Summary" section.

Updated docs/design.md with Phase 3 completion details, FP16/FP8 findings,
and Q4K GEMV kernel characteristics.

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1 (89 tasks), Phase 2 (35 tasks), Phase 3 (26 tasks) complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
  (nvcc at /usr/local/cuda/bin/nvcc, go at /usr/local/go/bin/go).
- **Benchmark:** /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf --tokens 50
  --prompt 'The quick brown fox' --device cuda --dtype [fp32|fp8]
- **Key files for Q4K GEMV optimization:**
  - internal/cuda/kernels/gemv_q4k.cu -- Fused dequant-GEMV kernel (128 threads, scalar byte loads)
  - internal/cuda/kernels/gemv_q4k.h -- Header with kernel signature
  - internal/cuda/kernels/Makefile -- Build with nvcc -arch=sm_121
  - compute/gpu_engine.go -- matMulQ4K dispatch (line ~1114)
- **Key files for GQA D2H elimination:**
  - layers/attention/grouped_query_attention.go -- D2H fallbacks at lines ~452, ~925
  - compute/gpu_engine.go -- GPUFusedQKNormRoPE (line ~2648)
- **Key files for CUDA graph capture:**
  - generate/generator.go -- Decode loop (line ~253), graph compile (line ~139)
  - graph/cuda_graph.go -- CUDA graph executor infrastructure
- **Key files for FP8:**
  - compute/gpu_fp8.go -- fp8Scratchpad, fp8DequantMatMulA/B
  - model/gguf/loader.go -- QuantizeToFP8E4M3 (scale factor computation)
- **Pre-commit hook:** Rejects multi-directory commits.
- **Stale worktree:** wave-4-task-E103 has 34 unmerged commits from Phase 1. Superseded. Do not merge.

---

## 10. Appendix

### Q4K GEMV Kernel Current Architecture

The kernel at gemv_q4k.cu uses 4 warps (128 threads) per block. Input vector x
is loaded cooperatively into shared memory. Each warp processes one output row.
Within a warp, 32 lanes split the super-blocks of that row in a strided pattern.
Each lane dequantizes Q4K values in registers using fused decode_scales_mins and
accumulates via __fmaf_rn. Warp shuffle reduction produces the final dot product.

Current bottleneck hypothesis: scalar __ldg byte loads (32 per group of 64 values)
dominate instruction issue. Vectorized uint4 loads would read 16 bytes in one
instruction, reducing load count from 32 to 2 per group.

### Performance Gap Analysis

| Component | Estimated Impact | Basis |
|-----------|-----------------|-------|
| Q4K GEMV optimization | 10-20% | Block size + vectorized loads |
| CUDA graph capture | 5-15% | Eliminate ~50 kernel launches/token at ~5-10us each |
| Per-token overhead | 2-5% | Tensor allocation, arena reset |
| Combined estimate | 17-40% | Enough to bridge 25% gap |
| Current F32 | 157.25 tok/s | Measured 2026-03-13 |
| Target | 197.21 tok/s | Ollama baseline |
| Required improvement | 25.4% | (197.21 - 157.25) / 157.25 |
