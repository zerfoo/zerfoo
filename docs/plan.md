# Zerfoo Development Plan -- Phase 11: ZMF CUDA Graph Capture + README

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 10 fixed the cuBLAS status 7 error (stale tensor caching after arena ResetPool).
ZMF F32 models now complete inference without crashes. However, two issues remain:

1. **CUDA graph capture fails for ZMF models.** During graph capture, the Transpose
   instruction calls `getDevicePtr` on CPUStorage weight tensors, which issues
   `cudaMemcpy` (H2D) on the legacy stream. This conflicts with the capturing
   blocking stream, producing cuda error 901. The graph executor falls back to
   non-graph execution.

2. **Non-graph fallback produces garbage output.** When graph capture fails, the
   fallback path produces `!!!` instead of coherent text. This suggests a
   correctness bug in the non-graph decode path -- possibly related to
   GPUStorage.TrySlice returning zero slices on memcpy failure, or arena pool
   state corruption between tokens.

3. **Pre-existing model-specific bugs.** Mistral 7B panics in Range op (index OOB).
   Phi 4 fails with pow_scalar cuda error 1 (missing kernel).

GGUF Q4K models are unaffected because Q4 storage uses virtual transpose (shape
swap only, no data movement) and the Q4 GEMV kernel reads blocks in native order.
The entire GGUF inference path stays GPU-resident, enabling clean graph capture.

### Root Cause Detail

`compute/gpu_kernels.go:getDevicePtr` (line 60-84) handles CPUStorage by:
1. Calling `t.Data()` to get the CPU slice
2. Allocating GPU memory via `e.pool.Alloc`
3. Calling `e.runtime.Memcpy(H2D)` to upload

Step 3 issues cudaMemcpy on the default/legacy stream. During CUDA graph capture
on a different stream, this causes error 901: "operation would make the legacy
stream depend on a capturing blocking stream."

The fix is to ensure all weight tensors are GPU-resident BEFORE graph capture
begins, eliminating H2D copies during the capture region.

### Objectives

- O1: Make ZMF F32 models work with CUDA graph capture on DGX.
- O2: Fix the non-graph fallback garbage output.
- O3: Fix Mistral Range op panic.
- O4: Fix Phi 4 pow_scalar kernel error.
- O5: Write README with quickstart and benchmark table.

### Non-Goals

- New model architectures.
- Performance optimization beyond fixing correctness.
- FP16/FP8 weight loading for ZMF models (future).
- Multi-GPU / distributed inference.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- GGUF Q4K models work at 232 tok/s with graph capture.
- ZMF F32 models complete inference but with garbage output.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| ZMF graph capture | Llama 3 + Qwen 2.5 use CUDA graph on DGX | bench_tps shows "captured instructions" |
| ZMF output quality | Coherent text at temp=0 for Llama 3 + Qwen 2.5 | bench_tps output inspection |
| Mistral inference | No panic, produces output | bench_tps on DGX |
| Phi 4 inference | No kernel error, produces output | bench_tps on DGX |
| README | Clone to inference in 5 minutes | Manual walkthrough |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D300 | Pre-upload ZMF weights to GPU before graph capture | Fixes CUDA graph capture failure |
| D301 | Fix non-graph fallback garbage output | Correctness for non-graph path |
| D302 | Fix Mistral Range op panic | Model coverage |
| D303 | Fix Phi 4 pow_scalar kernel | Model coverage |
| D304 | README with quickstart | First-time user experience |

### Out of Scope

- FP16 weight conversion for ZMF models.
- Optimizing ZMF model throughput beyond enabling graph capture.
- New model architectures.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E3000: Pre-Upload ZMF Weights to GPU

The fix: upload all frozen weight tensors to GPU during model loading or graph
compilation, so getDevicePtr finds GPUStorage (not CPUStorage) during the capture
region and returns the pointer without any cudaMemcpy.

- [ ] T3000.1 Add weight pre-upload to graph compilation  Owner: TBD  Est: 1.5h
  - In graph/compile.go or graph/cuda_graph.go, after graph compilation and
    before the first warmup run, iterate over all frozen slot tensors.
  - For each frozen tensor with CPUStorage, upload to GPU via pool.Alloc +
    runtime.Memcpy(H2D), wrap in GPUStorage, and replace the slot.
  - This ensures all weights are GPU-resident before graph capture begins.
  - The existing EnsureSlotsGPU in cuda_graph.go (line 163) only handles
    pre-capture instruction outputs. Extend it or add a new PreUploadWeights
    step that runs before warmup.
  - File: graph/cuda_graph.go or graph/compile.go.
  - Acceptance: getDevicePtr never encounters CPUStorage during graph capture
    for ZMF models.
  - Dependencies: none.

- [ ] S3000.1.1 Test weight pre-upload with Llama 3 on DGX  Owner: TBD  Est: 30m
  - Run bench_tps with Llama 3 ZMF: graph capture should succeed.
  - Verify "captured instructions" appears in output.
  - Compare tok/s with and without graph.
  - File: docs/updates.md.
  - Acceptance: CUDA graph captures successfully for Llama 3 ZMF.
  - Dependencies: T3000.1.

- [ ] S3000.1.2 Test weight pre-upload with Qwen 2.5 on DGX  Owner: TBD  Est: 15m
  - Run bench_tps with Qwen 2.5 ZMF.
  - Acceptance: CUDA graph captures successfully for Qwen 2.5 ZMF.
  - Dependencies: T3000.1.

### E3001: Fix Non-Graph Fallback Garbage Output

When CUDA graph capture fails, the fallback produces garbage. This is a separate
correctness bug that must be fixed regardless of graph capture success.

- [ ] T3001.1 Diagnose non-graph fallback garbage output  Owner: TBD  Est: 1h
  - Run bench_tps on DGX with CUDA graph disabled (if flag exists) or with
    a model that forces non-graph execution.
  - Add debug logging to the fallback path in graph/cuda_graph.go Run().
  - Check if arena pool state (reset count, used memory) is correct between
    tokens in non-graph mode.
  - Check if GPUStorage.TrySlice failure (returns zero slice) propagates
    garbage through the generation loop.
  - File: graph/cuda_graph.go, generate/generator.go.
  - Acceptance: Root cause identified.
  - Dependencies: none.

- [ ] T3001.2 Fix the non-graph fallback  Owner: TBD  Est: 1.5h
  - Apply fix based on T3001.1 diagnosis.
  - Likely fixes: ensure arena ResetPool does not corrupt in-flight tensors
    in non-graph mode; handle TrySlice errors by retrying or pre-syncing.
  - File: TBD based on diagnosis.
  - Acceptance: bench_tps with Llama 3 ZMF produces coherent text without
    CUDA graph.
  - Dependencies: T3001.1.

- [ ] S3001.2.1 Verify non-graph fix on DGX  Owner: TBD  Est: 30m
  - Run bench_tps for Llama 3 and Qwen 2.5 without graph.
  - Verify coherent output at temp=0.
  - Dependencies: T3001.2.

### E3002: Fix Mistral Range Op Panic

- [ ] T3002.1 Diagnose Mistral Range op index OOB  Owner: TBD  Est: 1h
  - Read layers/core/range_op.go:29 and trace the input shapes.
  - Run with debug output on DGX: bench_tps with Mistral 7B.
  - Check if the Range op receives empty inputs (length 0) causing index [0]
    with length 0.
  - Likely cause: tokenizer mismatch or graph builder issue for Mistral.
  - File: layers/core/range_op.go, inference/arch_llama.go (Mistral uses
    Llama architecture).
  - Acceptance: Root cause identified.
  - Dependencies: none.

- [ ] T3002.2 Fix Range op for Mistral  Owner: TBD  Est: 1h
  - Apply fix based on T3002.1 diagnosis.
  - Add bounds checking to Range op to prevent panic.
  - File: layers/core/range_op.go or inference/.
  - Acceptance: bench_tps with Mistral 7B produces output without panic.
  - Dependencies: T3002.1.

- [ ] S3002.2.1 Test Mistral fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps with Mistral 7B, 20 tokens.
  - Acceptance: No panic. Output produced.
  - Dependencies: T3002.2.

### E3003: Fix Phi 4 pow_scalar Kernel

- [ ] T3003.1 Diagnose Phi 4 pow_scalar cuda error 1  Owner: TBD  Est: 1h
  - Check if pow_scalar kernel exists in internal/cuda/kernels/.
  - Check if Phi 4 architecture uses Pow nodes (GeLU approximation?).
  - Read compute/gpu_engine.go Pow/PowScalar implementation.
  - File: compute/gpu_engine.go, internal/cuda/kernels/.
  - Acceptance: Root cause identified (missing kernel vs wrong dispatch).
  - Dependencies: none.

- [ ] T3003.2 Fix pow_scalar for Phi 4  Owner: TBD  Est: 1.5h
  - Implement or fix the pow_scalar CUDA kernel.
  - If the kernel is missing, add it to internal/cuda/kernels/ and wire it
    through the purego bindings.
  - File: internal/cuda/kernels/, compute/gpu_engine.go.
  - Acceptance: bench_tps with Phi 4 produces output without kernel error.
  - Dependencies: T3003.1.

- [ ] S3003.2.1 Test Phi 4 fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps with Phi 4, 20 tokens.
  - Acceptance: No kernel error. Output produced.
  - Dependencies: T3003.2.

### E3004: All-Model Verification

- [ ] T3004.1 Run all 5 models on DGX and record results  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3 (GGUF), Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Record tok/s, output quality, CUDA graph status for each.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce coherent output.
  - Dependencies: S3000.1.1, S3001.2.1, S3002.2.1, S3003.2.1.

### E3005: README

- [ ] T3005.1 Write README.md with quickstart  Owner: TBD  Est: 1.5h
  - Sections: What is Zerfoo, Installation, Quickstart (pull + run in 3
    commands), Supported Models (table with tok/s), API Usage (curl examples),
    Performance (vs Ollama chart), Architecture Overview, Contributing.
  - Include benchmark table from T3004.1 results.
  - File: README.md.
  - Acceptance: A new user can go from clone to inference in 5 minutes
    following the README.
  - Dependencies: T3004.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Graph Capture | T3000.1, S3000.1.1, S3000.1.2 | DGX for verification |
| Track B: Fallback Fix | T3001.1, T3001.2, S3001.2.1 | DGX for diagnosis |
| Track C: Mistral | T3002.1, T3002.2, S3002.2.1 | DGX for reproduction |
| Track D: Phi 4 | T3003.1, T3003.2, S3003.2.1 | DGX for reproduction |
| Track E: README | T3005.1 | Blocked on T3004.1 |

### Maximum parallelism

- Wave 1 (4 tasks): T3000.1 (weight pre-upload, local code) + T3001.1 (diagnose
  fallback, DGX) + T3002.1 (diagnose Mistral, DGX) + T3003.1 (diagnose Phi 4, DGX).
  All 4 are independent. T3000.1 is local code; the others need DGX.

- Wave 2 (5 tasks): S3000.1.1 (test Llama 3 graph, DGX) + S3000.1.2 (test Qwen 2.5,
  DGX) + T3001.2 (fix fallback, DGX) + T3002.2 (fix Mistral) + T3003.2 (fix Phi 4).
  S3000.1.x depend on T3000.1. T3001.2 depends on T3001.1. T3002.2 depends on T3002.1.
  T3003.2 depends on T3003.1.

- Wave 3 (3 tasks): S3001.2.1 (verify fallback fix) + S3002.2.1 (verify Mistral) +
  S3003.2.1 (verify Phi 4). All depend on Wave 2.

- Wave 4 (1 task): T3004.1 (all-model verification). Depends on Wave 3.

- Wave 5 (1 task): T3005.1 (README). Depends on T3004.1.

### Dependency minimization checklist applied

a) Diagnosis tasks (T3001.1, T3002.1, T3003.1) are independent of T3000.1.
b) T3000.1 (weight pre-upload) is pure local code, no DGX needed for implementation.
c) All 4 model-specific tracks are independent of each other.
d) Wave 1 has 4 tasks (could be 5 if T3001.1 is split, but diagnosis is atomic).
e) README is the only task with a hard dependency on model verification.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M300: Graph capture works | S3000.1.1 | Llama 3 ZMF uses CUDA graph on DGX |
| M301: All models run | T3004.1 | All 5 models produce coherent output |
| M302: README published | T3005.1 | 5-minute quickstart verified |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3000 | Weight pre-upload increases GPU memory pressure | OOM on smaller GPUs | Low | DGX has 128GB unified. Pre-upload only frozen weights. |
| R3001 | Non-graph garbage output is in generation loop, not graph executor | Harder fix | Medium | T3001.1 diagnosis will narrow. Check generate/generator.go token loop. |
| R3002 | Mistral Range op panic is architectural mismatch | May need Mistral-specific graph builder | Medium | Llama builder may not handle sliding window attention. |
| R3003 | Phi 4 needs a new CUDA kernel (pow_scalar) | Need to add .cu + purego binding | High | Simple kernel. Can use GPU Pow element-wise as fallback. |
| R3004 | make shared link step fails on CUDA 13.0 | Build friction | Medium | Known workaround: pass .pic.o files explicitly to nvcc. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes.
3. go test for the modified package passes with -race.
4. make shared builds without errors (CUDA kernel changes).
5. Commit passes pre-commit hooks.
6. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- CUDA: make shared in internal/cuda/kernels/ (when .cu files change).
- Benchmark: bench_tps on DGX Spark for model verification.

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 11 Plan Created)

Trimmed all 36 completed Phase 10 tasks. Stable knowledge (cuBLAS fix root cause,
decode kernel profiling results, trampoline diagnosis, FP16 KV results) preserved
in docs/design.md and docs/updates.md. Created Phase 11 with 6 epics, 16 tasks.

New issues identified from S2001.2.1 verification:
- CUDA graph capture fails for ZMF models (cuda error 901 from H2D copies during capture)
- Non-graph fallback produces garbage output
- Mistral Range op panic (pre-existing)
- Phi 4 pow_scalar kernel error (pre-existing)

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 232 tok/s Gemma 3 Q4K with CUDA graph (beats Ollama by 18.7%).
- **Branch:** fix/errcheck-issues has all work (~58 commits ahead of main).
- **Key bug (FIXED):** cuBLAS status 7 was stale tensor caching. Fixed by removing
  Transpose/MatMul weight caching and using MatMulTransposeB.
- **Current bugs:** ZMF models: CUDA graph capture fails (H2D during capture),
  non-graph fallback produces garbage. Mistral: Range panic. Phi 4: pow_scalar error.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf (232 tok/s), ~/models/llama3/
  (16.93 tok/s, garbage), ~/models/qwen25/ (14.59 tok/s, garbage),
  ~/models/mistral/ (Range panic), ~/models/phi4/ (pow_scalar error)
- **Pre-commit hook:** Rejects multi-directory commits.
- **Key files for graph capture fix:**
  - compute/gpu_kernels.go:32 -- getDevicePtr (H2D copy that breaks capture)
  - graph/cuda_graph.go:142 -- captureAndRun (capture region)
  - graph/cuda_graph.go:163 -- EnsureSlotsGPU (pre-capture GPU upload)
  - tensor/gpu_storage.go:209 -- TrySlice (D2H copy)
