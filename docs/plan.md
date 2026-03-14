# Zerfoo Development Plan -- Push Beyond 300 tok/s (Phase 7)

## 1. Context

### Problem Statement

Zerfoo F32 inference achieves 234.30 tok/s on DGX Spark GB10 with CUDA graph
capture (Phase 6). This surpasses Ollama (197.21 tok/s) by 18.8%. However,
bandwidth utilization is only ~60% of the 273 GB/s theoretical maximum.

Phase 7 Wave 1 profiling (T901.1) revealed that cuBLAS is only 8% of decode
time because weight matmuls already use the fused Q4K GEMV kernel. The original
plan to replace cuBLAS with a custom SGEMV (T901.4-T901.6) has low ROI.

The real bottleneck is the GQA attention path. The decode fast path
(flash_attention_decode with GPU-resident kv_len) was built in T903.2 but
causes a 93.7% regression for GQA models because engine.Repeat on full
maxSeqLen KV buffer creates ~128 MB temporaries per token. The fast path is
currently disabled for GQA (commit 9803ba1).

The path to >300 tok/s is making flash_attention_decode handle GQA head
replication inside the kernel at register level, eliminating Repeat entirely.
Decision rationale: docs/adr/034-gqa-aware-flash-attention-decode.md.

See docs/design.md for architecture, docs/adr/033-how-we-beat-ollama.md for
the full optimization history.

### Objectives

- O1: Make flash_attention_decode GQA-aware (register-level head replication).
- O2: Re-enable decode fast path for GQA models without Repeat.
- O3: Enable full CUDA graph capture of GQA attention path.
- O4: Fix FP16 KV cache correctness bug (produces pad tokens).
- O5: Achieve >300 tok/s on DGX Spark GB10 with Gemma 3 1B Q4_K_M.

### Non-Goals

- Custom SGEMV replacing cuBLAS (deferred, only 8% of decode time per T901.1).
- Speculative decoding (orthogonal, layer on top later).
- Persistent mega-kernel.
- Prefill CUDA graph capture.
- Q4K GEMV kernel rewrite.
- Multi-GPU / distributed inference.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121 (Blackwell), 273 GB/s LPDDR5x, 128GB unified memory.
- Baseline: 234.30 tok/s with CUDA graph (Phase 6).
- Ollama: 197.21 tok/s (Gemma 3 1B Q4_K_M).
- Gemma 3 1B GQA config: 8 query heads, 4 KV heads, head_dim=256.
- Go profile: go test, go vet, go build as quality gates.
- CUDA kernels compiled with nvcc -arch=sm_121.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| GQA decode kernel | Handles head replication internally | Unit test: output matches SDPA reference |
| Zero Repeat | No engine.Repeat on KV buffer during decode | grep + profiling |
| CUDA graph capture | Full decode including GQA attention captured | bench_tps shows graph executor |
| FP16 KV correctness | Coherent output with --kv-dtype=fp16 | bench_tps output inspection |
| Throughput target | >300 tok/s | bench_tps 3-run avg on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D84 | GQA-aware flash_attention_decode kernel | Eliminates Repeat, enables graph capture |
| D85 | Re-enabled decode fast path for GQA | 15-25% speedup from graph capture |
| D86 | FP16 KV correctness fix | Unblock FP16 KV cache feature |
| D87 | Full CUDA graph decode with GQA | All decode ops captured and replayed |

### Out of Scope

- Custom SGEMV replacing cuBLAS (deferred, low ROI).
- Speculative decoding.
- Prefill graph capture.
- Q4K GEMV changes.
- Multi-GPU.

---

## 3. Checkable Work Breakdown

### E901: Custom F32 GEMV Kernel for Decode (PARTIALLY COMPLETE)

Profiling showed cuBLAS is only 8% of decode (not 30% as estimated).
Kernel implemented and tested. Integration into GPUEngine deferred.

- [x] T901.1 Profile cuBLAS GEMV overhead on DGX  2026 03 14
- [x] T901.2 Implement custom sgemv_m1 CUDA kernel  2026 03 14
- [x] S901.2.1 Test sgemv_m1 correctness  2026 03 14
- [x] T901.3 Add Go wrappers for sgemv_m1 (done as part of T901.2)  2026 03 14

### Archived (deferred -- low ROI per T901.1 profiling)

- T901.4 Replace cuBLAS SGEMV with custom kernel in GPUEngine -- Deferred.
  cuBLAS is only 8% of decode time. Weight matmuls use Q4K GEMV, not cuBLAS.
- T901.5 Microbenchmark sgemv_m1 vs cuBLAS on DGX -- Deferred. Done informally
  during S901.2.1 (152 GFLOPS at 4096x4096).
- T901.6 Run go vet and make shared -- Subsumed by T904.2.

### E902: FP16 KV Cache (PARTIALLY COMPLETE)

Infrastructure built. FP16 produces garbage output (all pad tokens).
Correctness fix needed.

- [x] T902.1 Add FP16 storage mode to TensorCache  2026 03 14
- [x] S902.1.1 Test FP16 KV cache correctness  2026 03 14
- [x] T902.2 Add FP16 offset_memcpy variant  2026 03 14
- [x] S902.2.1 Test FP16 offset_memcpy  2026 03 14
- [x] T902.3 Wire FP16 KV into the generator  2026 03 14

- [ ] T902.5 Fix FP16 KV correctness bug  Owner: TBD  Est: 1.5h
  - FP16 KV produces all pad tokens on DGX. Likely cause: pointer type
    mismatch in the FP16 conversion path, or the FP16-to-F32 read-back
    is not wired correctly in the attention computation.
  - Debug on DGX: add logging to trace the FP16 write/read path.
  - Verify F32-to-FP16 conversion kernel output with a small test.
  - Check if FP16 buffer pointers are passed correctly to flash_attention.
  - File: generate/tensor_cache.go, layers/attention/grouped_query_attention.go.
  - Acceptance: bench_tps --kv-dtype=fp16 produces coherent output at temp=0.
  - Dependencies: none.

- [ ] S902.5.1 Test FP16 KV end-to-end on DGX  Owner: TBD  Est: 30m
  - Run bench_tps with --kv-dtype=fp16 on DGX, 20 tokens.
  - Verify output quality (coherent text at temp=0).
  - Acceptance: Output coherent. No pad tokens.
  - Dependencies: T902.5.

- [ ] T902.6 Run go vet on FP16 changes  Owner: TBD  Est: 15m
  - go vet ./generate/... ./internal/cuda/...
  - Acceptance: No new warnings.
  - Dependencies: T902.5.

### E903: Fix Graph/No-Graph Output Divergence (COMPLETE)

- [x] T903.1 Bisect the divergence source  2026 03 14
- [x] T903.2 Fix the divergence (GPU-resident kv_len)  2026 03 14
- [x] S903.2.1 Verify divergence fix on DGX  2026 03 14

### E905: GQA-Aware Flash Attention Decode Kernel

The flash_attention_decode kernel (T903.2) reads KV length from GPU memory
for CUDA graph capture. But for GQA models (numQueryHeads != numKVHeads),
the decode fast path calls engine.Repeat on the full maxSeqLen KV buffer
to expand KV heads, creating ~128 MB temporaries and regressing 93.7%.

The fix: handle GQA head replication inside the kernel at register level.
Each query head computes kv_head = q_head / (numQ / numKV) and indexes into
the KV buffer directly. Zero extra memory traffic.

Decision rationale: docs/adr/034-gqa-aware-flash-attention-decode.md.

- [ ] T905.1 Add GQA support to flash_attention_decode kernel  Owner: TBD  Est: 2h
  - Modify flash_attention_decode in internal/cuda/kernels/flash_attention.cu:
    - Add numQueryHeads and numKVHeads parameters.
    - In the inner loop, compute kv_head_idx = q_head_idx / (numQueryHeads / numKVHeads).
    - Index K and V using kv_head_idx instead of q_head_idx.
    - When numQueryHeads == numKVHeads, this is a no-op (same index).
  - Update the launcher: launch_flash_attention_decode_gqa(..., numQueryHeads, numKVHeads).
  - Alternatively, modify the existing launcher signature to add the two params.
  - File: internal/cuda/kernels/flash_attention.cu.
  - Acceptance: Kernel compiles. Output matches SDPA reference for GQA config
    (8 Q heads, 4 KV heads, head_dim=256).
  - Dependencies: none.

- [ ] S905.1.1 Test GQA flash_attention_decode correctness  Owner: TBD  Est: 30m
  - Test with Gemma 3 config: 8 Q heads, 4 KV heads, head_dim=256.
  - Compare output with naive attention (QK^T softmax V) using CPU reference.
  - Test with equal heads (numQ == numKV) to verify no regression.
  - File: internal/cuda/kernels/flash_attention_test.go.
  - Acceptance: Max absolute error < 1e-4 for both GQA and non-GQA configs.
  - Dependencies: T905.1.

- [ ] T905.2 Add Go wrappers for GQA flash_attention_decode  Owner: TBD  Est: 45m
  - Update purego wrapper (flash_attention_purego.go) and CGo wrapper.
  - Update KernelRunner interface if needed (add numQueryHeads, numKVHeads params).
  - Register new symbol in KernelLib if a new launcher was added.
  - Update stubs in OpenCL, ROCm, test mock.
  - File: internal/cuda/kernels/, internal/gpuapi/.
  - Acceptance: go build, go vet pass.
  - Dependencies: T905.1.

- [ ] T905.3 Re-enable GQA decode fast path  Owner: TBD  Est: 1.5h
  - In layers/attention/grouped_query_attention.go:
    - Remove the numQueryHeads == numKVHeads guard on the decode fast path.
    - Remove the engine.Repeat expansion code (lines 661-688).
    - Instead, pass numQueryHeads and numKVHeads to the flash_attention_decode
      call and let the kernel handle head replication.
    - The kernel indexes K/V with kv_head, so the KV buffer shape stays
      [batch, maxSeqLen, numKVHeads*headDim] without expansion.
  - File: layers/attention/grouped_query_attention.go.
  - Acceptance: bench_tps produces correct output with GQA. No engine.Repeat
    calls during decode. Decode fast path active for GQA models.
  - Dependencies: T905.2.

- [ ] S905.3.1 Test GQA decode fast path correctness  Owner: TBD  Est: 30m
  - go test ./layers/attention/... -race -timeout 120s.
  - Run bench_tps on DGX with 20 tokens, verify output matches standard path.
  - Acceptance: All tests pass. Output coherent at temp=0.
  - Dependencies: T905.3.

- [ ] T905.4 Verify CUDA graph capture with GQA decode  Owner: TBD  Est: 30m
  - Run bench_tps on DGX and verify graph executor is active (no fallback).
  - Check that GQA attention is included in the captured region.
  - File: docs/updates.md.
  - Acceptance: Graph capture succeeds. No "fallback" in logs.
  - Dependencies: T905.3.

- [ ] T905.5 Run go vet and make shared  Owner: TBD  Est: 15m
  - go vet ./internal/cuda/... ./internal/gpuapi/... ./layers/...
  - make shared CUDA_ARCH=sm_121 on DGX.
  - Acceptance: No new warnings. Build succeeds.
  - Dependencies: T905.3.

### E904: Final Benchmark and Verification

- [ ] T904.1 Full benchmark with all optimizations on DGX  Owner: TBD  Est: 1h
  - Build with GQA decode kernel + FP16 KV fix.
  - Run bench_tps 3 times with 256 tokens (F32 KV).
  - Run bench_tps 3 times with 256 tokens (FP16 KV if fixed).
  - Record commit hash, all results.
  - Compare with 234.30 tok/s baseline and Ollama 197.21 tok/s.
  - File: docs/updates.md.
  - Acceptance: Results documented. Target: >300 tok/s.
  - Dependencies: E905, T902.5.
  - PREFLIGHT: git pull on DGX, rebuild kernels (make clean && make shared).

- [ ] S904.1.1 Output quality verification  Owner: TBD  Est: 15m
  - Verify F32 output at temp=0. Graph and no-graph should match.
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
| Track A: GQA Kernel | E905 (T905.1-T905.5) | New CUDA kernel + GQA fast path |
| Track B: FP16 Fix | T902.5-T902.6 | Debug + fix FP16 correctness |
| Track C: Final Bench | E904 (T904.1-T904.2) | Depends on A, B |

### Maximum parallelism

- Wave 1 (3 tasks): T905.1 (GQA kernel) + T902.5 (FP16 fix) + S905.1.1 (test GQA kernel).
  T905.1 and T902.5 are independent. S905.1.1 can start when T905.1 is done.
  Each agent implements + tests in same worktree.

- Wave 2 (4 tasks): T905.2 (Go wrappers) + S902.5.1 (test FP16 on DGX) +
  T905.3 (re-enable GQA fast path) + T902.6 (go vet FP16).
  T905.2 depends on T905.1. T905.3 depends on T905.2.

- Wave 3 (4 tasks): S905.3.1 (test GQA fast path) + T905.4 (verify graph) +
  T905.5 (go vet + make) + T904.1 (full benchmark).

- Wave 4 (2 tasks): S904.1.1 (quality) + T904.2 (final vet).

### Dependency minimization checklist applied

a) GQA kernel (T905.1) and FP16 fix (T902.5) are fully independent.
b) Go wrappers (T905.2) depend on kernel (T905.1) but can be parallelized
   by having the same agent do both.
c) T904.1 depends on both tracks but can start once GQA fast path works.
d) Each agent works in its own worktree, so file conflicts are not blockers.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M124: GQA kernel ready | T905.2 | GQA flash_attention_decode compiles and passes tests |
| M125: GQA fast path active | T905.4 | Decode fast path enabled for GQA, graph captured |
| M126: FP16 KV fixed | S902.5.1 | FP16 KV produces coherent output |
| M127: >300 tok/s | T904.1 | bench_tps 3-run avg >300 tok/s |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R907 | GQA kernel register pressure from head index computation | Reduced occupancy | Low | Index computation is 1 integer divide + 1 multiply. Negligible vs attention math. |
| R908 | GQA kernel shared memory layout incompatible with variable head counts | Wrong results | Medium | Test with multiple GQA ratios (2x, 4x, 8x). Use head_ratio parameter. |
| R909 | FP16 KV bug is in the conversion kernel itself, not the wiring | Deeper fix needed | Medium | Test F32-to-FP16 kernel independently with known values on DGX. |
| R910 | Graph capture still fails with GQA decode due to other non-capturable ops | Partial capture | Low | The only remaining non-capturable op is EmbeddingLookup (1/185). |
| R911 | Combined speedup still under 300 tok/s | Goal not met | Medium | Recovering 156 attention launches should give ~15-25% improvement. 234 * 1.2 = 281. May need FP16 KV + speculative for >300. |
| R902 | FP16 KV precision loss degrades output quality | Unusable output | Low | FP16 has 3 decimal digits of precision. Attention softmax operates on F32. |

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

### Change Summary -- 2026-03-14 (Phase 7 Plan Updated -- GQA Kernel Pivot)

Major plan revision based on Wave 1 profiling and benchmark findings:

1. T901.1 profiling revealed cuBLAS is only 8% of decode time (weight matmuls
   use Q4K GEMV, not cuBLAS). Deferred T901.4, T901.5, T901.6 (low ROI).

2. T903.2 decode fast path caused 93.7% regression for GQA models due to
   engine.Repeat on full maxSeqLen KV buffer. Fixed by disabling fast path
   for GQA (9803ba1), restoring 234.08 tok/s.

3. FP16 KV produces pad tokens -- correctness bug. Added T902.5 to fix.

4. Created new epic E905: GQA-Aware Flash Attention Decode Kernel. This is
   now the primary optimization path. Register-level head replication inside
   the kernel eliminates Repeat and enables full CUDA graph capture for GQA.

5. Created ADR: docs/adr/034-gqa-aware-flash-attention-decode.md.

Completed tasks: T901.1, T901.2, S901.2.1, T901.3, T902.1, S902.1.1,
T902.2, S902.2.1, T902.3, T903.1, T903.2, S903.2.1.

Remaining: 11 tasks (E905: 7 tasks, E902 fix: 3 tasks, E904: 3 tasks).

### Change Summary -- 2026-03-14 (Phase 7 Plan Created)

Created Phase 7 plan targeting >300 tok/s via custom GEMV + FP16 KV cache.
Phase 6 (20 tasks, 4 epics) complete -- 234.30 tok/s, surpassed Ollama by 18.8%.

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1-6 complete. See docs/design.md and docs/adr/033-how-we-beat-ollama.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf
  --tokens 256 --prompt 'The quick brown fox' --device cuda --dtype fp32
- **GQA decode fast path (target for E905):**
  - layers/attention/grouped_query_attention.go lines 616-710
  - Currently disabled for GQA at line 621 (numQueryHeads == numKVHeads guard)
  - Calls tryFlashDecode at line 701
- **flash_attention_decode kernel (T903.2):**
  - internal/cuda/kernels/flash_attention.cu -- flash_attention_decode_kernel
  - Reads kv_len from GPU-resident pointer
  - Currently does NOT handle GQA (assumes numQ == numKV heads)
- **FP16 KV (broken):**
  - generate/tensor_cache.go -- WithKVDtype("fp16"), appendFP16, Get conversion
  - Produces all pad tokens on DGX. Conversion path needs debugging.
- **GPU counter pattern:**
  - internal/cuda/kernels/counter.cu -- increment_counter, reset_counter
  - internal/cuda/kernels/offset_memcpy.cu -- GPU-indexed memcpy (F32 + FP16)
  - internal/cuda/kernels/rope_select.cu -- GPU-indexed RoPE table lookup
- **sgemv_m1 kernel (built but not wired into GPUEngine):**
  - internal/cuda/kernels/sgemv_m1.cu -- 152 GFLOPS at 4096x4096
  - Deferred integration: cuBLAS is only 8% of decode
- **Pre-commit hook:** Rejects multi-directory commits.

---

## 10. Appendix

### GQA Head Replication in Kernel

```
Gemma 3 1B config:
  numQueryHeads = 8
  numKVHeads = 4
  headRatio = numQueryHeads / numKVHeads = 2

For query head q_idx (0..7):
  kv_head_idx = q_idx / headRatio = q_idx / 2

  q_idx=0 -> kv_head=0
  q_idx=1 -> kv_head=0  (replicates KV head 0)
  q_idx=2 -> kv_head=1
  q_idx=3 -> kv_head=1  (replicates KV head 1)
  q_idx=4 -> kv_head=2
  q_idx=5 -> kv_head=2  (replicates KV head 2)
  q_idx=6 -> kv_head=3
  q_idx=7 -> kv_head=3  (replicates KV head 3)

KV buffer layout: [batch, maxSeqLen, numKVHeads * headDim]
  K for kv_head h at position p: K[b * maxSeqLen * dim + p * dim + h * headDim]

This indexing happens at register level in the kernel inner loop.
Zero extra memory allocation. Zero extra memory traffic.
```

### Performance Projection

| Optimization | Estimated Savings | Estimated tok/s |
|-------------|-------------------|-----------------|
| Baseline (Phase 6) | -- | 234.30 |
| GQA decode kernel (eliminate Repeat) | 0.3-0.5 ms | 250-270 |
| CUDA graph capture of GQA attention | 0.5-0.8 ms | 280-320 |
| FP16 KV cache (if fixed) | 0.1-0.3 ms | +5-10 additional |
| Combined estimate | 0.9-1.6 ms savings | 290-330 |
