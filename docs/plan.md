# Zerfoo Development Plan -- Push Beyond 300 tok/s (Phase 7)

## 1. Context

### Problem Statement

Zerfoo F32 inference achieves 234.30 tok/s on DGX Spark GB10 with CUDA graph
capture (Phase 6). This surpasses Ollama (197.21 tok/s) by 18.8%. However,
bandwidth utilization is only ~60% of the 273 GB/s theoretical maximum. At
full utilization, the hardware could sustain ~390 tok/s for this model size.

The remaining 40% gap comes from two sources:
1. **cuBLAS SGEMV overhead for M=1 decode:** cuBLAS is optimized for large
   matrix multiplications. For single-token decode (M=1), the library startup,
   workspace allocation, and heuristic selection overhead is significant
   relative to the actual compute. Each cuBLAS call adds ~3-5us overhead
   beyond the memory transfer time.
2. **F32 KV cache bandwidth:** KV cache reads/writes consume 4 bytes per
   element. FP16 KV would halve this bandwidth and free memory bus for weight
   loading, which is the actual bottleneck.

See docs/design.md for architecture, docs/adr/033-how-we-beat-ollama.md for
the full optimization history.

### Objectives

- O1: Replace cuBLAS SGEMV with custom GEMV kernels optimized for M=1 decode.
- O2: Quantize KV cache to FP16 to halve KV bandwidth.
- O3: Investigate and fix graph/no-graph output divergence from Phase 6.
- O4: Achieve >300 tok/s on DGX Spark GB10 with Gemma 3 1B Q4_K_M.

### Non-Goals

- Speculative decoding (orthogonal, can layer on top later).
- Persistent mega-kernel (high complexity, unclear benefit with graph capture).
- Prefill CUDA graph capture (variable seqLen, different optimization profile).
- Q4K GEMV kernel rewrite (memory-bound, Phase 4 showed regression).
- Multi-GPU / distributed inference.
- New model architectures.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121 (Blackwell), 273 GB/s LPDDR5x, 128GB unified memory.
- Baseline: 234.30 tok/s with CUDA graph (Phase 6, commit 0891914).
- Ollama: 197.21 tok/s (Gemma 3 1B Q4_K_M).
- Go profile: go test, go vet, go build as quality gates.
- CUDA kernels compiled with nvcc -arch=sm_121.
- All GPU bindings use purego (no CGo for production builds).

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Custom GEMV parity | Output matches cuBLAS bit-for-bit at temp=0 | Comparison test on DGX |
| Custom GEMV speedup | >10% vs cuBLAS for M=1 GEMV | Microbenchmark on DGX |
| FP16 KV bandwidth | KV cache at half the memory traffic | nvidia-smi or nsight |
| Output divergence | Graph == no-graph at temp=0 | bench_tps comparison |
| Throughput target | >300 tok/s | bench_tps 3-run avg on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D80 | Custom F32 GEMV kernel for M=1 | Eliminates cuBLAS overhead for decode |
| D81 | FP16 KV cache storage + mixed-precision attention | Halves KV bandwidth |
| D82 | Graph/no-graph output divergence fix | Correctness |
| D83 | Updated CUDA graph with new kernels | All changes must remain graph-capturable |

### Out of Scope

- Speculative decoding.
- Persistent mega-kernel.
- Prefill graph capture.
- Q4K GEMV changes.
- Multi-GPU.

---

## 3. Checkable Work Breakdown

### E901: Custom F32 GEMV Kernel for Decode

cuBLAS SGEMV has ~3-5us overhead per call for M=1. With ~10 SGEMV calls per
layer (QKV projection, output projection, FFN up/gate/down) x 26 layers =
~260 calls/token, replacing cuBLAS with a custom GEMV saves ~780-1300us/token
(18-30% of 4.27ms token time).

The custom GEMV for M=1 is simpler than general GEMM: each thread block
loads a row of the weight matrix and dot-products with the input vector.
Shared memory is used only for the input vector (reused across all rows).

- [ ] T901.1 Profile cuBLAS GEMV overhead on DGX  Owner: TBD  Est: 45m
  - Run bench_tps with nsight-sys or manual timing around cuBLAS calls.
  - Count cuBLAS SGEMV calls per token during decode.
  - Measure per-call overhead (total time minus theoretical memory time).
  - Document: call count, avg latency, overhead fraction.
  - File: docs/updates.md.
  - Acceptance: cuBLAS call count and overhead documented.
  - Dependencies: none.

- [ ] T901.2 Implement custom sgemv_m1 CUDA kernel  Owner: TBD  Est: 1.5h
  - Add kernel: `sgemv_m1(y, A, x, M, N, stream)` where y[M], A[M x N], x[N].
  - Design: each block handles multiple rows. Threads within a block
    cooperatively load x into shared memory, then each thread computes
    partial dot product for its assigned rows.
  - Use vectorized loads (float4) for both A and x to maximize bandwidth.
  - Tune block size for sm_121 (256 threads, target 100% occupancy).
  - File: internal/cuda/kernels/sgemv_m1.cu.
  - Acceptance: Kernel compiles. Output matches cuBLAS SGEMV for random inputs.
  - Dependencies: none.

- [ ] S901.2.1 Test sgemv_m1 correctness  Owner: TBD  Est: 30m
  - Compare output with cuBLAS for matrix sizes matching Gemma 3 1B layers:
    - QKV: [1536, 1536], [256, 1536], [256, 1536]
    - Output: [1536, 1536]
    - FFN up/gate: [6144, 1536]
    - FFN down: [1536, 6144]
  - Acceptance: Max absolute error < 1e-5 for all sizes.
  - Dependencies: T901.2.

- [ ] T901.3 Add Go wrappers for sgemv_m1  Owner: TBD  Est: 45m
  - Add purego wrapper (sgemv_m1_purego.go) and CGo wrapper (sgemv_m1_cgo.go).
  - Register in KernelLib (purego.go) and KernelRunner interface.
  - Update Makefile to include sgemv_m1.cu in SRCS.
  - File: internal/cuda/kernels/, internal/gpuapi/.
  - Acceptance: go build, go vet pass. Kernel callable from Go.
  - Dependencies: T901.2.

- [ ] T901.4 Replace cuBLAS SGEMV with custom kernel in GPUEngine  Owner: TBD  Est: 1.5h
  - In compute/gpu_engine.go (or wherever cuBLAS SGEMV is called for MatMul):
    - Detect M=1 case (single-token decode).
    - Route to custom sgemv_m1 kernel instead of cuBLAS.
    - Keep cuBLAS for M>1 (prefill, batch).
  - File: compute/gpu_engine.go.
  - Acceptance: bench_tps produces correct output. No cuBLAS calls for M=1 decode.
  - Dependencies: T901.3.

- [ ] S901.4.1 Test custom GEMV integration  Owner: TBD  Est: 30m
  - go test ./compute/... -race -timeout 120s.
  - Run bench_tps on DGX with 20 tokens, verify output correctness.
  - Acceptance: All tests pass. Output coherent at temp=0.
  - Dependencies: T901.4.

- [ ] T901.5 Microbenchmark sgemv_m1 vs cuBLAS on DGX  Owner: TBD  Est: 30m
  - Benchmark both kernels for all Gemma 3 1B matrix sizes.
  - Record: throughput (GB/s), latency (us), speedup.
  - File: docs/updates.md.
  - Acceptance: Results documented. Custom kernel faster for M=1.
  - Dependencies: T901.4.

- [ ] T901.6 Run go vet and make shared  Owner: TBD  Est: 15m
  - go vet ./internal/cuda/... ./internal/gpuapi/... ./compute/...
  - make shared CUDA_ARCH=sm_121 on DGX.
  - Acceptance: No new warnings. Build succeeds.
  - Dependencies: T901.4.

### E902: FP16 KV Cache

KV cache stores key/value projections at F32 (4 bytes/element). Converting to
FP16 (2 bytes/element) halves KV read/write bandwidth. For Gemma 3 1B with
26 layers and dim=1536, at 256 tokens the KV cache is 26 x 2 x 256 x 1536 x 4
= 78.6 MB (F32) vs 39.3 MB (FP16). The bandwidth saving grows linearly with
sequence length.

Mixed-precision attention: K and V stored as FP16, but attention computation
remains F32. Convert F32 projections to FP16 before KV append, and FP16 back
to F32 when reading from cache for attention. The F32-to-FP16 and FP16-to-F32
conversion kernels already exist (launch_f32_to_fp16, launch_fp16_to_f32).

- [ ] T902.1 Add FP16 storage mode to TensorCache  Owner: TBD  Est: 1.5h
  - Add a `kvDtype` field to TensorCache (F32 or FP16).
  - When kvDtype=FP16, allocate KV buffers with half the byte size.
  - In Update/AppendGPU: call F32-to-FP16 conversion before writing to cache.
  - In Get: call FP16-to-F32 conversion when reading from cache.
  - Use the existing launch_f32_to_fp16 and launch_fp16_to_f32 kernels.
  - File: generate/tensor_cache.go.
  - Acceptance: TensorCache supports FP16 KV storage. F32 path unchanged.
  - Dependencies: none.

- [ ] S902.1.1 Test FP16 KV cache correctness  Owner: TBD  Est: 30m
  - go test ./generate/... -race -timeout 120s.
  - Compare FP16 KV output with F32 KV output for 10 tokens.
  - Acceptance: Max absolute error < 1e-3 (FP16 precision). Tests pass.
  - Dependencies: T902.1.

- [ ] T902.2 Add FP16 offset_memcpy variant  Owner: TBD  Est: 1h
  - The existing offset_memcpy kernel copies float32. Need a half-precision
    variant: offset_memcpy_fp16(dst_fp16, src_f32, counter, dim, maxSeqLen).
  - This kernel converts F32 src to FP16 during the copy, fusing the
    conversion with the offset computation.
  - File: internal/cuda/kernels/offset_memcpy.cu (add to existing file).
  - Acceptance: Kernel compiles. Fused F32->FP16 copy correct.
  - Dependencies: none.

- [ ] S902.2.1 Test FP16 offset_memcpy  Owner: TBD  Est: 30m
  - Set counter=3, copy F32 src, verify FP16 data at offset 3*dim.
  - Acceptance: Test passes. Values match F32-to-FP16 reference.
  - Dependencies: T902.2.

- [ ] T902.3 Wire FP16 KV into the generator  Owner: TBD  Est: 1h
  - Add a --kv-dtype flag to bench_tps (or auto-detect based on model config).
  - When FP16 KV is enabled, pass kvDtype=FP16 to TensorCache constructor.
  - Ensure CUDA graph capture still works with FP16 KV (offset_memcpy_fp16
    must be graph-capturable).
  - File: generate/generator.go, cmd/bench_tps/main.go.
  - Acceptance: bench_tps supports --kv-dtype=fp16. Graph capture succeeds.
  - Dependencies: T902.1, T902.2.

- [ ] S902.3.1 Test FP16 KV end-to-end on DGX  Owner: TBD  Est: 30m
  - Run bench_tps with --kv-dtype=fp16 on DGX, 20 tokens.
  - Verify output quality (coherent text at temp=0).
  - Acceptance: Output quality acceptable. No crashes.
  - Dependencies: T902.3.

- [ ] T902.4 Run go vet on modified packages  Owner: TBD  Est: 15m
  - go vet ./generate/... ./internal/cuda/... ./cmd/bench_tps/...
  - Acceptance: No new warnings.
  - Dependencies: T902.3.

### E903: Fix Graph/No-Graph Output Divergence

Phase 6 found that CUDA graph and non-graph paths produce different output
at temp=0. Both outputs are coherent and deterministic, but they differ.
This suggests a subtle numerical difference in the captured vs live execution.

- [ ] T903.1 Bisect the divergence source  Owner: TBD  Est: 1.5h
  - Add debug logging to dump intermediate tensor values at key points:
    after embedding, after first GQA, after first FFN, etc.
  - Run both graph and non-graph paths with 5 tokens.
  - Compare dumps to identify where values first diverge.
  - Hypothesis: the offset_memcpy or rope_select kernel produces slightly
    different results when captured vs live (e.g., different thread scheduling
    affecting floating-point accumulation order).
  - File: docs/updates.md.
  - Acceptance: Divergence source identified and documented.
  - Dependencies: none.

- [ ] T903.2 Fix the divergence  Owner: TBD  Est: 1h
  - Based on T903.1 findings, fix the root cause.
  - If it is floating-point ordering in graph replay, ensure kernels use
    deterministic accumulation (e.g., same block/grid dimensions in both paths).
  - If it is a counter sync issue, fix the sync.
  - File: depends on T903.1 findings.
  - Acceptance: Graph and no-graph output identical at temp=0 for 256 tokens.
  - Dependencies: T903.1.

- [ ] S903.2.1 Verify divergence fix on DGX  Owner: TBD  Est: 30m
  - Run bench_tps on DGX in both graph and no-graph modes.
  - Compare output tokens.
  - Acceptance: Identical output at temp=0.
  - Dependencies: T903.2.

### E904: Final Benchmark and Verification

- [ ] T904.1 Full benchmark with all Phase 7 optimizations on DGX  Owner: TBD  Est: 1h
  - Build with custom GEMV + FP16 KV + divergence fix.
  - Run bench_tps 3 times with 256 tokens.
  - Record commit hash, all results.
  - Compare with 234.30 tok/s baseline and Ollama 197.21 tok/s.
  - File: docs/updates.md.
  - Acceptance: Results documented. Target: >300 tok/s.
  - Dependencies: E901, E902, E903.
  - PREFLIGHT: git pull on DGX, rebuild kernels (make clean && make shared).

- [ ] S904.1.1 Output quality verification  Owner: TBD  Est: 15m
  - Verify F32 output at temp=0 matches graph==no-graph.
  - Acceptance: Identical output tokens.
  - Dependencies: T904.1.

- [ ] T904.2 Run go vet on all packages  Owner: TBD  Est: 15m
  - go vet ./...
  - Acceptance: No new warnings beyond pre-existing purego patterns.
  - Dependencies: T904.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Custom GEMV | E901 (T901.1-T901.6) | New CUDA kernel + integration |
| Track B: FP16 KV | E902 (T902.1-T902.4) | TensorCache + new kernel variant |
| Track C: Divergence Fix | E903 (T903.1-T903.2) | Debug + fix |
| Track D: Final Bench | E904 (T904.1-T904.2) | Depends on A, B, C |

### Maximum parallelism

- Wave 1 (5 tasks): T901.1 (profile cuBLAS) + T901.2 (custom GEMV kernel) +
  T902.1 (FP16 TensorCache) + T902.2 (FP16 offset_memcpy) + T903.1 (bisect divergence).
  All 5 are independent. Saturates all agent slots.

- Wave 2 (5 tasks): S901.2.1 (test GEMV) + T901.3 (Go wrappers) +
  S902.1.1 (test FP16 KV) + S902.2.1 (test FP16 memcpy) + T903.2 (fix divergence).
  Each depends only on its Wave 1 predecessor.

- Wave 3 (5 tasks): T901.4 (replace cuBLAS) + T902.3 (wire FP16 KV) +
  S903.2.1 (verify fix) + T901.5 (microbenchmark) + T901.6 (go vet + make).
  T901.4 depends on T901.3. T902.3 depends on T902.1 + T902.2.

- Wave 4 (4 tasks): S901.4.1 (test integration) + S902.3.1 (test FP16 e2e) +
  T902.4 (go vet) + T904.1 (full benchmark).
  T904.1 depends on all tracks but can start once GEMV + FP16 KV are wired.

- Wave 5 (2 tasks): S904.1.1 (quality) + T904.2 (final vet).

### Dependency minimization checklist applied

a) All 5 Wave 1 tasks are fully independent (different packages, different goals).
b) Custom GEMV (Track A) and FP16 KV (Track B) touch different packages
   (internal/cuda/kernels vs generate/) and can run fully in parallel.
c) Divergence fix (Track C) is independent of performance work.
d) T904.1 is the only task requiring all tracks to converge.
e) Each agent works in its own worktree, so file conflicts are not blockers.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M120: Custom GEMV ready | T901.4 | sgemv_m1 integrated, correct output |
| M121: FP16 KV ready | T902.3 | FP16 KV cache wired, graph-capturable |
| M122: Divergence fixed | T903.2 | Graph == no-graph output at temp=0 |
| M123: >300 tok/s | T904.1 | bench_tps 3-run avg >300 tok/s |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R901 | Custom GEMV slower than cuBLAS for some sizes | No speedup for those layers | Medium | Benchmark per-size. Keep cuBLAS fallback for sizes where custom is slower. |
| R902 | FP16 KV precision loss degrades output quality | Unusable output | Low | FP16 has 3 decimal digits of precision. Attention softmax operates on F32. Monitor perplexity. |
| R903 | Graph/no-graph divergence is inherent to CUDA graph replay | Cannot fix | Medium | If inherent, document as known behavior. Both outputs are valid. |
| R904 | Combined optimizations still under 300 tok/s | Goal not met | Medium | 300 is aspirational. Any improvement over 234 is valuable. Re-evaluate and add speculative decoding. |
| R905 | FP16 offset_memcpy not graph-capturable | FP16 KV breaks graph | Low | Same pattern as F32 offset_memcpy which is already captured. |
| R906 | Custom GEMV register pressure on sm_121 | Reduced occupancy | Medium | Profile with --ptxas-options=-v. Limit to 32 registers like gemm_q4. |

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

- Always rebuild libkernels.so on DGX before benchmarking (make clean && make shared).
- Always git pull on DGX before benchmarking.
- Record exact commit hash in benchmark results.

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 7 Plan Created)

Created Phase 7 plan targeting >300 tok/s via custom GEMV + FP16 KV cache.
Phase 6 (20 tasks, 4 epics) complete -- 234.30 tok/s, surpassed Ollama by 18.8%.
Phase 6 knowledge trimmed to docs/design.md.
Updated docs/design.md with Phase 6 completion summary.

Phase 7 focuses on 4 epics:
- E901: Custom F32 GEMV kernel for decode (6 tasks + 2 tests).
- E902: FP16 KV cache (4 tasks + 3 tests).
- E903: Graph/no-graph output divergence fix (2 tasks + 1 test).
- E904: Final benchmark (2 tasks + 1 test).

Total: 14 implementation tasks, 6 test subtasks = 20 tasks.
Designed for 5 waves with up to 5 parallel agents per wave.

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1 (89 tasks), Phase 2 (35 tasks), Phase 3 (26 tasks),
  Phase 4 (30 tasks), Phase 5 (19 tasks), Phase 6 (20 tasks) complete.
  See docs/design.md and docs/adr/033-how-we-beat-ollama.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf
  --tokens 256 --prompt 'The quick brown fox' --device cuda --dtype fp32
- **cuBLAS SGEMV location:** compute/gpu_engine.go calls cublasSgemv_v2 via
  internal/cublas/ purego bindings.
- **FP16 conversion kernels (already exist):**
  - internal/cuda/kernels/elementwise_fp16.cu -- launch_f32_to_fp16, launch_fp16_to_f32
  - These are optional symbols in KernelLib (may need to verify they compile for sm_121).
- **CUDA graph infrastructure:**
  - graph/cuda_graph.go -- CUDAGraphExecutor with capture/replay
  - generate/tensor_cache.go -- TensorCache with GPU counter, AppendGPU
- **GPU counter pattern:**
  - internal/cuda/kernels/counter.cu -- increment_counter, reset_counter
  - internal/cuda/kernels/offset_memcpy.cu -- GPU-indexed memcpy
  - internal/cuda/kernels/rope_select.cu -- GPU-indexed RoPE table lookup
- **Pre-commit hook:** Rejects multi-directory commits.
- **Pre-existing issue:** BatchGenerate race on logitsBuf (unrelated to Phase 7).

---

## 10. Appendix

### Bandwidth Utilization Analysis

| Metric | Value | Source |
|--------|-------|--------|
| Current throughput | 234.30 tok/s | Phase 6 benchmark |
| Token time | 4.27 ms | 1/234.30 |
| Memory bandwidth | 273 GB/s | DGX Spark GB10 spec |
| Model size (Q4_K_M) | ~700 MB | Estimated for Gemma 3 1B |
| Theoretical min token time | ~2.56 ms | 700MB / 273GB/s |
| Theoretical max tok/s | ~390 | 1/0.00256 |
| Current utilization | ~60% | 2.56/4.27 |
| cuBLAS overhead estimate | ~1.0-1.3 ms | 260 calls x 4us overhead |
| FP16 KV savings estimate | ~0.2-0.5 ms | Depends on sequence length |

### cuBLAS Call Pattern (Gemma 3 1B decode, M=1)

Per transformer layer (26 layers):
- QKV projection: 1x cublasSgemv (merged weight, M=1 N=1536 K=1536+256+256)
- Output projection: 1x cublasSgemv (M=1 N=1536 K=1536)
- FFN gate+up: 1x cublasSgemv (merged, M=1 N=12288 K=1536) or 2 separate
- FFN down: 1x cublasSgemv (M=1 N=1536 K=6144)

Total: ~4-5 cuBLAS calls/layer x 26 layers = 104-130 calls/token.
Plus embedding, LM head: ~135-145 total cuBLAS calls per token.
