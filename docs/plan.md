# Zerfoo Development Plan -- Phase 16: ONNX Output Quality + CUDA Graph Coverage

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phases 13-15 fixed all crash-causing bugs in the ONNX execution path. All 5
models run on DGX without crashes and zero test failures. Gemma 3 GGUF achieves
232 tok/s with 99.5% CUDA graph capture.

Two categories of issues remain for ONNX models:

1. **Output quality.** All ONNX models produce repetitive text at temp=0. A
   repetition penalty feature was implemented (--repetition-penalty flag) but
   has never been tested end-to-end on DGX. Testing it is a quick win that
   should immediately improve output quality for all ONNX models.

2. **ONNX model performance.** ONNX models run at 4-16 tok/s vs 232 tok/s for
   Gemma 3 GGUF. The primary bottleneck is CUDA graph capture coverage:
   - Gemma 3 GGUF captures 184/185 instructions (99.5%) using fused ops
   - ONNX models capture only 32-66 of 1610-5300 instructions (1-4%)
   The ONNX path decomposes operations like RMSNorm into 6-7 individual ops
   (Pow, ReduceMean, Add, Sqrt, Div, Mul) that each require separate GPU
   kernel launches. Fusing these into single ops would dramatically reduce
   kernel launch overhead and increase CUDA graph capture coverage.

3. **Phi 4 CUDA graph capture.** Still partially fails with TrySlice errors
   during capture. The capture region shifted to [146,164) after Phase 15
   fixes, but GPUStorage.TrySlice triggers cudaMemcpy during capture.

### Objectives

- O1: Test repetition penalty on DGX with all ONNX models (quick win).
- O2: Fuse decomposed ONNX RMSNorm ops into a single fused layer.
- O3: Increase ONNX CUDA graph capture coverage from 1-4% toward 50%+.
- O4: Investigate and fix Phi 4 TrySlice capture failures.
- O5: All 5 models produce non-repetitive output on DGX.

### Non-Goals

- New model architectures.
- Multi-GPU / distributed inference.
- Training, fine-tuning, RLHF.
- Matching ORT output bit-for-bit.
- FP16 mixed precision (deferred to Phase 17).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- DGX requires `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- DGX uses `upstream` HTTPS remote for fetch.
- All 5 models currently run without crashes (Phases 11-15 fixes).
- Main at 4724c47 with all Phase 13-15 work merged.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Repetition penalty | Non-repetitive output for all ONNX models | bench_tps --repetition-penalty 1.2 on DGX |
| ONNX graph capture | 50%+ instructions captured for Llama 3 | bench_tps log output on DGX |
| ONNX throughput | 2x improvement for Llama 3 (target 25+ tok/s) | bench_tps on DGX |
| No regression | Gemma 3 GGUF still 230+ tok/s | bench_tps on DGX |
| Phi 4 graph capture | No TrySlice errors during capture | bench_tps log on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D900 | Repetition penalty DGX verification | Quick win for output quality |
| D901 | ONNX RMSNorm fusion pass | Replace Pow+ReduceMean+Add+Sqrt+Div+Mul with FusedRMSNorm |
| D902 | Phi 4 TrySlice capture fix | Fix remaining CUDA graph capture failure |
| D903 | ONNX CUDA graph capture improvements | More ops in capture region |
| D904 | All-model verification on DGX | Confirm improvements |

### Out of Scope

- FP16 mixed precision (deferred to Phase 17).
- SwiGLU/GQA/FFN fusion (future phase).
- New model architectures.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E3700: Repetition Penalty DGX Verification

The repetition penalty is implemented but never tested end-to-end on DGX.
This is a zero-code-change verification task.

- [ ] T3700.1 Test repetition penalty on DGX with all ONNX models  Owner: TBD  Est: 30m
  - DGX preflight: pull main, rebuild .so and binary.
  - Run bench_tps with --repetition-penalty 1.2 for Llama 3, Qwen 2.5,
    Mistral 7B, Phi 4. Compare output with and without penalty.
  - Run Gemma 3 GGUF without penalty (regression check).
  - File: docs/updates.md.
  - Acceptance: Output is less repetitive with penalty=1.2. Record results.
  - Dependencies: none.

### E3701: ONNX RMSNorm Fusion

ONNX models decompose RMSNorm into 6-7 individual ops per layer. Llama 3 has
32 transformer layers, each with 2 RMSNorm calls = 64 RMSNorm instances =
384-448 individual ops that could be replaced with 64 fused RMSNorm calls.

The fused RMSNorm kernel already exists (internal/cuda/kernels/rmsnorm.cu)
and is used by the ZMF codegen pipeline. The task is to add an ONNX graph
optimization pass that detects the decomposed pattern and replaces it with
the fused op.

- [ ] T3701.1 Identify RMSNorm decomposition pattern in ONNX graph  Owner: TBD  Est: 1h
  - Read graph/compile.go and graph/instruction.go to understand the compiled
    instruction list for ONNX models.
  - Print the instruction list for Llama 3 (first 50 instructions) to identify
    the exact RMSNorm decomposition pattern.
  - Document the pattern: which ops, in what order, with what connectivity.
  - Check if Qwen 2.5, Mistral, and Phi 4 use the same decomposition.
  - File: graph/.
  - Acceptance: Exact op sequence for decomposed RMSNorm documented.
  - Dependencies: none.

- [ ] T3701.2 Implement ONNX graph fusion pass for RMSNorm  Owner: TBD  Est: 2h
  - Add a graph optimization pass that scans the instruction list for the
    RMSNorm decomposition pattern and replaces matching sequences with a
    single FusedRMSNorm instruction.
  - The pass runs after Compile but before CUDA graph capture.
  - Use the existing RMSNorm layer (layers/normalization/rmsnorm.go) and
    its GPU kernel (rmsnorm.cu).
  - The pass must preserve the epsilon parameter from the Add op.
  - File: graph/fusion.go (new file), graph/compile.go.
  - Acceptance: Llama 3 instruction count drops by ~320 (64 fusions x ~5
    ops eliminated per fusion). RMSNorm ops use the fused kernel.
  - Dependencies: T3701.1.

- [ ] S3701.2.1 Test RMSNorm fusion pass  Owner: TBD  Est: 45m
  - Unit test: create a synthetic instruction list with the RMSNorm pattern,
    verify the fusion pass detects and replaces it.
  - Test with epsilon values from real models.
  - go test ./graph/... -race.
  - Dependencies: T3701.2.

- [ ] S3701.2.2 Test RMSNorm fusion on DGX  Owner: TBD  Est: 30m
  - Run Llama 3 with fusion enabled. Verify instruction count dropped.
  - Verify output quality is maintained (no numerical regression).
  - Measure tok/s improvement.
  - Dependencies: T3701.2, S3701.2.1.

### E3702: Phi 4 TrySlice Capture Fix

Phi 4 CUDA graph capture fails because GPUStorage.TrySlice triggers
cudaMemcpy during stream capture. TrySlice reads a small header from GPU
memory to determine slice bounds, which is incompatible with CUDA graph
capture.

- [ ] T3702.1 Diagnose Phi 4 TrySlice capture failure  Owner: TBD  Est: 1h
  - Read compute/gpu_storage.go TrySlice implementation.
  - Identify why TrySlice needs cudaMemcpy (likely reading tensor metadata
    from GPU memory).
  - Check which ops call TrySlice during the capture region.
  - Determine if TrySlice can be made capture-safe by caching metadata
    or using pre-computed offsets.
  - File: compute/gpu_storage.go, graph/cuda_graph.go.
  - Acceptance: Root cause documented. Fix approach identified.
  - Dependencies: none.

- [ ] T3702.2 Fix TrySlice for CUDA graph capture  Owner: TBD  Est: 1.5h
  - Implement the fix based on diagnosis.
  - Options: (a) cache slice metadata before capture, (b) pre-compute
    offsets during warmup, (c) add ops that call TrySlice to nonCapturableOps.
  - File: compute/gpu_storage.go or graph/cuda_graph.go.
  - Acceptance: Phi 4 CUDA graph capture succeeds without TrySlice errors.
  - Dependencies: T3702.1.

- [ ] S3702.2.1 Test Phi 4 capture fix on DGX  Owner: TBD  Est: 15m
  - bench_tps for Phi 4 on DGX. Verify no capture errors in log.
  - Measure tok/s improvement from successful graph capture.
  - Dependencies: T3702.2.

### E3703: ONNX CUDA Graph Capture Improvements

Beyond RMSNorm fusion, additional ops may need to be excluded from capture
or made capture-safe to increase the capturable instruction region.

- [ ] T3703.1 Audit ONNX non-capturable ops  Owner: TBD  Est: 1h
  - For each model (Llama 3, Qwen, Mistral, Phi 4), print the capture
    region and list all ops that fall outside it.
  - Categorize non-capturable ops: (a) inherently non-capturable (KV cache I/O),
    (b) could be made capturable with code changes, (c) already handled.
  - Identify the longest potential capture region if remaining blockers
    were fixed.
  - File: graph/cuda_graph.go.
  - Acceptance: Per-model audit with capture improvement opportunities.
  - Dependencies: none.

- [ ] T3703.2 Expand ONNX capture region  Owner: TBD  Est: 1.5h
  - Based on audit, fix ops that can be made capture-safe.
  - This may include: pre-uploading CPU tensors, caching metadata,
    or splitting the capture into multiple regions.
  - File: graph/cuda_graph.go, affected op files.
  - Acceptance: Llama 3 capture coverage increases from 2% to 10%+.
  - Dependencies: T3703.1.

- [ ] S3703.2.1 Test capture improvements on DGX  Owner: TBD  Est: 15m
  - bench_tps for all models. Verify improved capture coverage.
  - Dependencies: T3703.2.

### E3704: All-Model Verification

- [ ] T3704.1 Run all 5 models on DGX with improvements  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3 (GGUF), Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Use --repetition-penalty 1.2 for ONNX models.
  - Record tok/s, capture coverage, output quality for each.
  - File: docs/updates.md.
  - Acceptance: Gemma 3 >= 230 tok/s. ONNX models show measurable
    improvement in throughput and/or output quality.
  - Dependencies: T3700.1, S3701.2.2, S3702.2.1, S3703.2.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: Repetition Penalty | T3700.1 | DGX verification only |
| Track B: RMSNorm Fusion | T3701.1, T3701.2, S3701.2.1, S3701.2.2 | Graph optimization |
| Track C: Phi 4 TrySlice | T3702.1, T3702.2, S3702.2.1 | CUDA graph fix |
| Track D: Capture Audit | T3703.1, T3703.2, S3703.2.1 | CUDA graph analysis |
| Track E: Final Verify | T3704.1 | DGX, depends on A-D |

### Maximum parallelism

- Wave 1 (4 tasks): T3700.1 (test repetition penalty, DGX) + T3701.1
  (identify RMSNorm pattern) + T3702.1 (diagnose TrySlice) + T3703.1
  (audit non-capturable ops). All independent. 4 teammates.

- Wave 2 (3 tasks): T3701.2 (implement RMSNorm fusion) + T3702.2 (fix
  TrySlice) + T3703.2 (expand capture region). Each depends on its
  diagnosis. 3 teammates.

- Wave 3 (3 tasks): S3701.2.1 (test fusion) + S3701.2.2 (test fusion DGX)
  + S3702.2.1 (test TrySlice DGX) + S3703.2.1 (test capture DGX).
  Combine DGX tests. 2-3 teammates.

- Wave 4 (1 task): T3704.1 (all-model verification, DGX). 1 teammate.

### Dependency minimization checklist applied

a) All 4 Wave 1 tasks are fully independent.
b) RMSNorm pattern identification (T3701.1) is read-only analysis that
   unblocks the implementation task.
c) TrySlice diagnosis (T3702.1) and capture audit (T3703.1) can run in
   parallel as they read different code paths.
d) DGX tasks in Wave 1 (T3700.1) can run on a separate SSH session from
   local code tasks.
e) Wave 1 saturates 4 teammates.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M900: Quick wins | T3700.1 | Repetition penalty tested on DGX |
| M901: Fusion implemented | T3701.2, S3701.2.1 | RMSNorm fusion pass works, tests pass |
| M902: Graph capture improved | T3702.2, T3703.2 | Phi 4 capture fixed, ONNX coverage up |
| M903: All models verified | T3704.1 | 5/5 models improved on DGX |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3700 | RMSNorm decomposition pattern varies between models | Multiple fusion patterns needed | Medium | Start with Llama 3 pattern, extend to others. Document variations. |
| R3701 | RMSNorm fusion changes numerical results | Output quality regression | Low | The fused kernel is already used by GGUF. Compare fused vs decomposed numerically. |
| R3702 | TrySlice fix requires architectural changes to GPU storage | Large scope | Medium | Option (c) -- adding ops to nonCapturableOps -- is always available as fallback. |
| R3703 | ONNX CUDA graph capture has fundamental limitations | Cannot reach 50% coverage | Medium | Even 10-20% coverage with fusion should give measurable speedup. |
| R3704 | Repetition penalty at 1.2 produces incoherent output | No quality improvement | Low | Try different values (1.1, 1.3, 1.5). Penalty is model-dependent. |

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
2. export PATH=/usr/local/cuda/bin:$PATH
3. cd ~/zerfoo && git fetch upstream main && git checkout main
4. cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
5. cd ~/zerfoo
6. export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
7. go build -o bench_tps ./cmd/bench_tps/ (ALWAYS rebuild binary)
8. Verify: ./bench_tps --help

---

## 8. Progress Log

### Change Summary -- 2026-03-15 (Phase 16: ONNX Quality + CUDA Graph)

Trimmed completed Phase 15 work into docs/design.md:
- SentencePiece tokenizer fix (LoadFromJSON decoder parsing)
- CUDA graph nonCapturableOps (ConstantOfShape, Shape)
- TestCPUEngine_Exp tolerance fix
- Gemma 3 measurement artifact confirmed (232 tok/s with 256 tokens)
- Phi 4 stale binary issue (not code regression)
- Phase 15 verification results per model

Phase 16 created to address remaining ONNX quality and performance:
- E3700: Repetition penalty DGX verification (quick win)
- E3701: ONNX RMSNorm fusion (biggest performance lever)
- E3702: Phi 4 TrySlice capture fix
- E3703: ONNX CUDA graph capture improvements
- E3704: All-model final verification

No new ADRs needed.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0.
- **Branch:** main at 4724c47. All Phase 13-15 work merged (PRs #64, #65, #66).
- **Model status (DGX, main branch, Phase 15 verified):**
  - Gemma 3 GGUF: 232.21 tok/s (256 tok), 99.5% graph capture, baseline
  - Llama 3 ONNX: 12.93 tok/s, semi-coherent, 2% graph capture
  - Qwen 2.5 ONNX: 15.79 tok/s, improved (no single-token repetition)
  - Mistral 7B ONNX: 3.94 tok/s, spaces fixed, still repetitive
  - Phi 4 ONNX: 4.14 tok/s, semi-coherent, CUDA graph capture partial
- **Key ONNX bottleneck:** Decomposed ops = many kernel launches + low graph
  capture. RMSNorm fusion is the highest-impact optimization.
- **Repetition penalty:** --repetition-penalty flag implemented but not tested
  on DGX. Quick win to verify.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
  Use `upstream` HTTPS remote for fetch. ALWAYS rebuild binary before benchmarking.
  `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- **Key files:**
  - graph/compile.go -- ONNX instruction compilation
  - graph/cuda_graph.go -- CUDA graph capture, nonCapturableOps
  - layers/normalization/rmsnorm.go -- fused RMSNorm (used by GGUF)
  - internal/cuda/kernels/rmsnorm.cu -- fused RMSNorm kernel
  - compute/gpu_storage.go -- TrySlice implementation
  - cmd/bench_tps/main.go -- --repetition-penalty flag
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf, ~/models/llama3/,
  ~/models/qwen25/, ~/models/mistral/, ~/models/phi4/
- **Pre-commit hook:** Rejects multi-directory commits.
