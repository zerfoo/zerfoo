# Zerfoo Development Plan -- Phase 11: ZMF Graph Capture + D2H Elimination

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 10 fixed cuBLAS status 7 (stale tensor caching). Phase 11 Waves 1-2
diagnosed and partially fixed ZMF model issues. Current state:

**Fixed in Waves 1-2:**
- PreUploadFrozenWeights (69c48af): uploads frozen CPU tensors to GPU before capture.
- KV cache snapshot/restore (425e0c6): prevents double-update on capture failure.
- Range op bounds check (8f3efc6): prevents panic on empty Data().
- libkernels.so rebuilt on DGX with sm_121 (pow_scalar kernel now available).

**Remaining problem: 3 sync-during-capture operations break CUDA graph capture.**
All are D2H round-trips on the legacy/default stream during the capture region:

1. **Transpose CPU fallback** -- GPUEngine.Transpose falls back to CPUEngine for
   some ZMF tensor shapes (rank > 4, or axes/shape mismatch). CPUEngine.Transpose
   calls `a.Data()` which calls `GPUStorage.Slice()` -> `TrySlice()` -> sync D2H
   memcpy on the legacy stream. This conflicts with the capturing blocking stream
   (cuda error 901). Observed at instruction 38 (Llama 3) and 76 (Qwen 2.5).

2. **Pow scalar D2H** -- `gpuPow` in compute/gpu_kernels.go:712 reads a scalar
   exponent from GPU via `MemcpyAsync(D2H)` + `stream.Synchronize()`. The Synchronize
   call conflicts with graph capture. Phi 4 uses `x^3` in GeLU approximation.

3. **Range op Data() on GPU scalars** -- Range.Forward() calls `inputs[i].Data()[0]`
   on scalar constants that were uploaded to GPU by ZMF's UploadWeights. Data() calls
   GPUStorage.Slice() which does sync D2H. Affects Mistral (sliding window positions).

**Non-graph fallback (T3001.2 fix) has NOT been verified on DGX yet.** The KV cache
snapshot/restore was merged into main but not deployed. Verifying this is the
immediate safety net.

### Root Cause Pattern

All three failures share the same root cause: **CPU-side data access during a GPU
capture region.** The GGUF Q4K path avoids this because:
- Q4Storage uses virtual transpose (shape swap, no data movement)
- Q4 GEMV kernel reads blocks in native order
- All scalar constants are embedded in the graph as host-side values
- The entire path stays GPU-resident

ZMF F32 models trigger CPU fallbacks because they use standard Transpose (which
requires data movement for non-trivial permutations), Pow with GPU-resident scalar
exponents, and Range with GPU-resident scalar inputs.

### Objectives

- O1: Verify non-graph fallback produces coherent output with T3001.2 fix (safety net).
- O2: Eliminate all D2H sync operations from the CUDA graph capture region.
- O3: Enable CUDA graph capture for ZMF F32 models (Llama 3, Qwen 2.5).
- O4: Verify all 5 models produce coherent output on DGX.
- O5: Write README with quickstart and benchmark table.

### Non-Goals

- New model architectures.
- FP16/FP8 weight loading for ZMF models (future).
- Multi-GPU / distributed inference.
- Optimizing ZMF throughput beyond enabling graph capture.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- GGUF Q4K models work at 232 tok/s with graph capture.
- Compute engine must remain portable across CUDA/ROCm/OpenCL backends.
  All fixes must go through the Engine[T] or KernelRunner interface, not
  CUDA-specific code paths.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Non-graph coherence | Llama 3 + Qwen 2.5 produce coherent text without graph | bench_tps output at temp=0 |
| ZMF graph capture | Llama 3 + Qwen 2.5 use CUDA graph on DGX | "captured instructions" in bench_tps log |
| All models run | All 5 models produce coherent output | bench_tps on DGX |
| README | Clone to inference in 5 minutes | Manual walkthrough |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D300 | Non-graph fallback verification on DGX | Safety net before further changes |
| D301 | Eliminate Transpose CPU fallback during capture | Fixes cuda 901 at Transpose instructions |
| D302 | Eliminate Pow scalar D2H during capture | Fixes Phi 4 cuda error 1 during capture |
| D303 | Eliminate Range scalar D2H during capture | Fixes Mistral Range failure during capture |
| D304 | All-model verification on DGX | Confirm all 5 models work |
| D305 | README with quickstart | First-time user experience |

### Out of Scope

- FP16 weight conversion for ZMF models.
- New model architectures.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E3100: Verify Non-Graph Fallback (Safety Net)

Deploy T3001.2 (KV cache fix) to DGX and verify fallback produces coherent output.
This is the safety net: even if graph capture fixes fail, ZMF models should work.

- [ ] S3100.1 Verify non-graph fallback on DGX  Owner: TBD  Est: 30m
  - Preflight: ssh to DGX, git pull, rebuild libkernels.so with CUDA_ARCH=sm_121.
  - Run bench_tps for Llama 3 ZMF at temp=0 with 256 tokens.
  - Graph capture will fail (expected). Fallback should produce coherent text
    thanks to T3001.2 KV cache snapshot/restore.
  - Repeat for Qwen 2.5.
  - File: docs/updates.md.
  - Acceptance: Coherent text output for both Llama 3 and Qwen 2.5 in fallback mode.
  - Dependencies: none (T3001.2 already merged to main).

### E3101: Eliminate Transpose D2H During Capture

GPUEngine.Transpose falls back to CPUEngine for some ZMF tensor shapes, triggering
GPUStorage.TrySlice (sync D2H on legacy stream). The fix: ensure the GPU transpose
path handles all cases that occur during ZMF inference, or if CPU fallback is
truly needed, mark the instruction as non-capturable.

- [ ] T3101.1 Diagnose Transpose CPU fallback trigger  Owner: TBD  Est: 45m
  - Add debug logging to GPUEngine.Transpose at each fallback exit point
    (lines 1772, 1782, 1799, 1804, 1858, 1865 in compute/gpu_engine.go).
  - Log the tensor shape, rank, axes, and storage type.
  - Run bench_tps on DGX with ZERFOO_DEBUG_GPU=1 for Llama 3.
  - Determine exactly which condition triggers the CPU fallback.
  - Likely causes: rank > 4, or isTransposeReshape returns false for a case
    that should be a no-op reshape.
  - File: compute/gpu_engine.go.
  - Acceptance: Exact fallback condition identified with tensor shape + axes.
  - Dependencies: none.

- [ ] T3101.2 Fix Transpose to stay GPU-resident during capture  Owner: TBD  Est: 1.5h
  - Apply fix based on T3101.1 diagnosis. Likely options:
    a) Extend the GPU transpose kernel to handle the triggering case.
    b) If the fallback is for a reshape-equivalent case, fix isTransposeReshape.
    c) If the fallback is truly needed (rare shape), mark Transpose as
       conditionally non-capturable for those shapes.
  - The fix must go through the Engine[T] interface -- no CUDA-specific hacks.
  - File: compute/gpu_engine.go.
  - Acceptance: No Transpose CPU fallback during ZMF Llama 3 inference.
  - Dependencies: T3101.1.

- [ ] S3101.2.1 Test Transpose fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Llama 3 with debug logging. Verify zero Transpose CPU
    fallbacks during the capture region.
  - Dependencies: T3101.2.

### E3102: Eliminate Pow Scalar D2H During Capture

gpuPow reads the scalar exponent via async D2H + stream.Synchronize, which
conflicts with graph capture. The exponent is a constant (e.g., 3.0 for GeLU x^3)
that never changes between tokens.

Decision rationale: docs/adr/023-gpu-scalar-ops-d2h-elimination.md

- [ ] T3102.1 Cache Pow scalar exponent during warmup  Owner: TBD  Est: 1h
  - In gpuPow (compute/gpu_kernels.go:701), the scalar exponent path at line 712
    does: detect totalElements(exponent)==1 -> read scalar via MemcpyAsync D2H +
    Synchronize -> call gpuScalarOp with the host scalar.
  - Fix: during warmup runs (before graph capture), the scalar value is read
    normally. Cache it in a map[slotIndex]float32 on the CUDAGraphExecutor.
    During capture and replay, use the cached value instead of reading from GPU.
  - Alternative (simpler): the exponent is always a frozen constant. After
    PreUploadFrozenWeights, check if any frozen tensor has exactly 1 element.
    If so, read the scalar BEFORE the first warmup and store it as a CPU-side
    constant. Replace the GPU tensor with a CPU tensor containing just the scalar.
    This way gpuPow takes the `else` branch at line 730 (`exponent.Data()[0]`)
    and never does D2H during compute.
  - File: compute/gpu_kernels.go or graph/compile.go.
  - Acceptance: gpuPow does not call MemcpyAsync or Synchronize during graph capture.
  - Dependencies: none.

- [ ] S3102.1.1 Test Pow fix with Phi 4 on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Phi 4. Verify no cuda error 1 during capture.
  - Dependencies: T3102.1.

### E3103: Eliminate Range Scalar D2H During Capture

Range.Forward() calls Data()[0] on GPU-resident scalar constants. For ZMF models,
UploadWeights uploads ALL constants to GPU, including 0-element and 1-element
scalars that serve as Range parameters (start, limit, delta).

- [ ] T3103.1 Keep scalar constants CPU-resident during upload  Owner: TBD  Est: 1h
  - In the weight upload path (model loading or graph compilation), skip GPU
    upload for tensors with totalElements <= 1. These are scalar constants
    used by Range, Pow, and other ops that read values on the CPU.
  - Alternatively: in graph/compile.go PreUploadFrozenWeights, add a reverse
    step: for frozen tensors with 1 element that were uploaded to GPU, read
    the value back and replace with CPUStorage. This keeps them CPU-accessible.
  - File: graph/compile.go or model/loader.go.
  - Acceptance: Range.Forward() inputs have CPUStorage for scalar constants.
    Data()[0] works without D2H copy.
  - Dependencies: none.

- [ ] S3103.1.1 Test Range fix with Mistral on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Mistral 7B. Verify no Range errors during capture.
  - Dependencies: T3103.1.

### E3104: Graph Capture Verification

After all D2H eliminations, verify CUDA graph capture works for ZMF models.

- [ ] T3104.1 Verify graph capture for all ZMF models on DGX  Owner: TBD  Est: 1h
  - Run bench_tps for Llama 3, Qwen 2.5, Mistral 7B, Phi 4 on DGX.
  - Verify "captured instructions" appears in log for each.
  - Record tok/s with graph vs fallback.
  - Also run Gemma 3 GGUF to confirm no regression.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce coherent output. At least Llama 3 and
    Qwen 2.5 use CUDA graph capture.
  - Dependencies: S3101.2.1, S3102.1.1, S3103.1.1.

### E3105: README

- [ ] T3105.1 Write README.md with quickstart  Owner: TBD  Est: 1.5h
  - Sections: What is Zerfoo, Installation, Quickstart (pull + run in 3
    commands), Supported Models (table with tok/s from T3104.1), API Usage
    (curl examples), Performance (vs Ollama chart), Architecture Overview,
    Contributing.
  - Include benchmark table from T3104.1 results.
  - File: README.md.
  - Acceptance: A new user can go from clone to inference in 5 minutes.
  - Dependencies: T3104.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Safety Net | S3100.1 | DGX verification, quick |
| Track B: Transpose | T3101.1, T3101.2, S3101.2.1 | Diagnosis + fix + verify |
| Track C: Pow Scalar | T3102.1, S3102.1.1 | Cache scalar, verify |
| Track D: Range Scalar | T3103.1, S3103.1.1 | Keep scalars CPU, verify |
| Track E: README | T3105.1 | Blocked on T3104.1 |

### Maximum parallelism

- Wave 3 (4 tasks): S3100.1 (verify fallback, DGX) + T3101.1 (diagnose Transpose,
  DGX) + T3102.1 (fix Pow scalar, local code) + T3103.1 (fix Range scalars, local
  code). S3100.1 and T3101.1 share DGX but run different commands. T3102.1 and
  T3103.1 are local code, fully independent.

  NOTE: DGX tasks must be sequenced on the shared GPU. Combine S3100.1 and T3101.1
  into one DGX teammate. Local code tasks (T3102.1, T3103.1) run as separate
  teammates. Total: 3 teammates.

- Wave 4 (4 tasks): T3101.2 (fix Transpose, local code, depends T3101.1) +
  S3101.2.1 (test Transpose, DGX) + S3102.1.1 (test Pow, DGX) + S3103.1.1
  (test Range, DGX). T3101.2 is local; the 3 DGX tests can run sequentially
  in one DGX teammate. Total: 2 teammates.

  NOTE: S3101.2.1 depends on T3101.2 completing first. The DGX teammate should
  run S3102.1.1 and S3103.1.1 first (no dependency on T3101.2), then S3101.2.1
  after T3101.2 completes.

- Wave 5 (1 task): T3104.1 (all-model verification, DGX). Depends on all Wave 4
  DGX tests passing. Total: 1 teammate.

- Wave 6 (1 task): T3105.1 (README). Depends on T3104.1. Total: 1 teammate.

### Dependency minimization checklist applied

a) T3102.1 (Pow fix) and T3103.1 (Range fix) have no dependencies -- front-loaded
   in Wave 3 alongside diagnosis work.
b) S3100.1 (fallback verification) has no code dependency -- can run immediately.
c) T3101.1 (Transpose diagnosis) is independent of the Pow and Range fixes.
d) DGX tests are grouped into single teammates to avoid GPU contention.
e) README is the only task with a hard dependency chain to the end.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M310: Fallback works | S3100.1 | Coherent output from Llama 3 + Qwen 2.5 without graph |
| M311: D2H eliminated | S3101.2.1, S3102.1.1, S3103.1.1 | Zero sync D2H ops in capture region |
| M312: Graph capture works | T3104.1 | All 5 models produce coherent output with graph |
| M313: README published | T3105.1 | 5-minute quickstart verified |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3100 | T3001.2 fallback fix does not produce coherent output | No safety net | Low | S3100.1 verifies immediately. If fails, diagnose further before proceeding. |
| R3101 | Transpose CPU fallback has multiple triggers, not just one | More fixes needed | Medium | T3101.1 adds debug logging to ALL exit points. May need multiple fixes. |
| R3102 | Additional D2H ops discovered during capture beyond the 3 known | Whack-a-mole | Medium | After fixing the 3 known ops, T3104.1 runs full verification. If new ops found, add them to E3101-style fixes. |
| R3103 | Keeping scalars CPU-resident breaks some GPU-only code path | Regression | Low | Scalars (1 element) are only read, never computed on. CPU storage is simpler. |
| R3104 | make shared link fails on CUDA 13.0 | Build friction | Medium | Known workaround: pass .pic.o files explicitly to nvcc. |

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

### DGX Preflight (required before any DGX benchmark)

1. ssh ndungu@192.168.86.250
2. cd ~/zerfoo && git pull
3. cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
4. cd ~/zerfoo
5. export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
6. Verify: /usr/local/go/bin/go run ./cmd/bench_tps --help

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 11 Plan Updated for D2H Elimination)

Trimmed completed Waves 1-2 tasks (T3000.1, T3001.1, T3001.2, T3002.1, T3002.2,
T3003.1, T3003.2, S3000.1.1, S3000.1.2, S3002.2.1). Stable knowledge preserved
in this plan's Context section and docs/updates.md.

Wave 2 DGX benchmarks revealed 3 remaining D2H sync operations blocking graph
capture. Restructured remaining work around systematic D2H elimination:
- E3100: Verify fallback safety net (S3100.1)
- E3101: Fix Transpose CPU fallback (T3101.1, T3101.2, S3101.2.1)
- E3102: Fix Pow scalar D2H (T3102.1, S3102.1.1)
- E3103: Fix Range scalar D2H (T3103.1, S3103.1.1)
- E3104: All-model verification (T3104.1)
- E3105: README (T3105.1)

Existing ADR docs/adr/023-gpu-scalar-ops-d2h-elimination.md covers the D2H
elimination strategy. No new ADRs needed.

### Change Summary -- 2026-03-14 (Phase 11 Plan Created)

Created Phase 11 with 6 epics, 16 tasks. Issues identified from S2001.2.1:
CUDA graph capture fails (cuda error 901), non-graph fallback garbage output,
Mistral Range panic, Phi 4 pow_scalar error.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 232 tok/s Gemma 3 Q4K with CUDA graph (beats Ollama by 18.7%).
- **Branch:** main at 1a1cfec. All Wave 1-2 work merged.
- **Key fixes already applied:**
  - PreUploadFrozenWeights (69c48af): frozen CPU tensors uploaded to GPU before capture.
  - KV cache snapshot/restore (425e0c6): prevents double-update on capture failure.
  - Range bounds check (8f3efc6): prevents panic on empty Data().
  - libkernels.so rebuilt on DGX with sm_121 (pow_scalar kernel available).
- **Current bugs:** Graph capture fails for ZMF models (3 D2H sync ops). Fallback
  correctness unverified on DGX (T3001.2 merged but not deployed).
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf (232 tok/s),
  ~/models/llama3/ (16.36 tok/s fallback), ~/models/qwen25/ (14.09 tok/s fallback),
  ~/models/mistral/ (no panic, Range error during capture),
  ~/models/phi4/ (pow_scalar works in warmup, fails during capture)
- **Pre-commit hook:** Rejects multi-directory commits.
- **Key files for D2H elimination:**
  - compute/gpu_engine.go:1770 -- Transpose (CPU fallback paths)
  - compute/gpu_kernels.go:701 -- gpuPow (scalar D2H at line 712-732)
  - layers/core/range_op.go:28 -- Range.Forward (Data() on GPU scalars)
  - graph/compile.go:211 -- PreUploadFrozenWeights (may need scalar exclusion)
  - tensor/gpu_storage.go:212 -- TrySlice (the sync D2H that breaks capture)
  - graph/cuda_graph.go:142 -- captureAndRun (capture region)
