# Zerfoo Development Plan -- Phase 12: Per-Model Fixes + ONNX Verification

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 11 achieved two major milestones: CUDA graph capture for ZMF codegen models
(232.86 tok/s, 99.5% captured, +26%) and ONNX output quality fix for Llama 3
(coherent text at temp=0). README is published. See docs/design.md items 13-14.

Three ONNX models still have per-model bugs preventing correct inference:

1. **Qwen 2.5: Gather index OOB during decode.** `Gather index 7 out of bounds
   [0,7)` during autoregressive decode. Likely an off-by-one in attention head
   indexing -- the Gather op receives an index equal to the dimension size instead
   of size-1. May be in the ONNX graph's head-splitting logic or in how the
   StatefulInputNode feeds back KV cache tensors with unexpected shapes.

2. **Mistral 7B: Range error.** Range.Forward() receives empty Data() for GPU
   scalar inputs. The bounds-check fix (8f3efc6) prevents the panic but returns
   an error that stops inference. Root cause: scalar constants used by Mistral's
   sliding window position encoding are on GPU; Range needs them on CPU. The
   PreUploadFrozenWeights scalar exclusion (ce1e155) should handle this, but
   Mistral may have additional scalars beyond the frozen set.

3. **Phi 4: pow_scalar error during graph capture.** pow_scalar CUDA kernel
   fails with error 1 during graph capture. The powf NaN fix (elementwise.cu)
   addressed the computation correctness, but the kernel launch itself fails
   during the capture region. The scalar exponent read path (D2H + Synchronize)
   conflicts with stream capture. The scalar CPU-residence fix should handle
   frozen exponents, but Phi 4 may have exponents that are not in the frozen set.

### Objectives

- O1: Fix Qwen 2.5 Gather index OOB to produce coherent output.
- O2: Fix Mistral 7B Range error to produce coherent output.
- O3: Fix Phi 4 pow_scalar capture error to produce coherent output.
- O4: Verify all 5 models produce coherent output on DGX.

### Non-Goals

- New model architectures.
- FP16/FP8 weight loading for ZMF models.
- Multi-GPU / distributed inference.
- CUDA graph capture for ONNX models (path forward is ZMF codegen migration).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- Llama 3 ONNX now works (17.56 tok/s, coherent text). Gemma 3 GGUF works
  (232.86 tok/s, coherent text, 99.5% graph capture).

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Qwen 2.5 output | Coherent text at temp=0 | bench_tps on DGX |
| Mistral 7B output | No error, coherent text | bench_tps on DGX |
| Phi 4 output | No kernel error, coherent text | bench_tps on DGX |
| All models pass | 5/5 produce coherent output | bench_tps on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D500 | Qwen 2.5 Gather index fix | Model coverage |
| D501 | Mistral 7B Range scalar fix | Model coverage |
| D502 | Phi 4 pow_scalar capture fix | Model coverage |
| D503 | All-model verification on DGX | Confirm all 5 models work |

### Out of Scope

- FP16 weight conversion for ZMF models.
- New model architectures.
- Training, fine-tuning, RLHF.
- CUDA graph capture for ONNX models.

---

## 3. Checkable Work Breakdown

### E3300: Fix Qwen 2.5 Gather Index OOB

- [ ] T3300.1 Diagnose Qwen 2.5 Gather index OOB  Owner: TBD  Est: 1h
  - Run bench_tps on DGX with ZERFOO_DEBUG_ONNX=1 for Qwen 2.5.
  - Identify which Gather instruction triggers the OOB error.
  - Trace the index tensor: where does the value 7 come from when the
    dimension size is 7 (valid range [0,6])?
  - Check if this is an off-by-one in the ONNX graph's head-splitting logic
    or in how kvCacheIONode feeds back KV tensors.
  - File: layers/gather/, model/builder.go, graph/graph.go.
  - Acceptance: Root cause identified.
  - Dependencies: none.

- [ ] T3300.2 Fix Qwen 2.5 Gather index OOB  Owner: TBD  Est: 1h
  - Apply fix based on T3300.1 diagnosis.
  - File: TBD.
  - Acceptance: bench_tps with Qwen 2.5 produces coherent text at temp=0.
  - Dependencies: T3300.1.

- [ ] S3300.2.1 Test Qwen 2.5 fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Qwen 2.5 with 256 tokens. Verify coherent output.
  - Also run Llama 3 and Gemma 3 to confirm no regression.
  - Dependencies: T3300.2.

- [ ] S3300.2.2 Run go vet and go test after fix  Owner: TBD  Est: 15m
  - go vet ./..., go build ./..., go test for modified packages with -race.
  - Dependencies: T3300.2.

### E3301: Fix Mistral 7B Range Error

- [ ] T3301.1 Diagnose Mistral Range scalar source  Owner: TBD  Est: 1h
  - Run bench_tps on DGX with ZERFOO_DEBUG_ONNX=1 for Mistral 7B.
  - Identify which Range instruction fails and which scalar input has
    empty Data().
  - Check if the scalar is a frozen constant (should be handled by
    PreUploadFrozenWeights scalar exclusion) or a dynamic intermediate.
  - If dynamic: trace the computation that produces the scalar.
  - File: layers/core/range_op.go, graph/compile.go, model/builder.go.
  - Acceptance: Root cause identified.
  - Dependencies: none.

- [ ] T3301.2 Fix Mistral Range error  Owner: TBD  Est: 1h
  - Apply fix based on T3301.1 diagnosis.
  - If the scalar is dynamic: add CPU readback before Range.Forward().
  - If the scalar is frozen but not in frozenIdx: add it to the frozen set.
  - File: TBD.
  - Acceptance: bench_tps with Mistral 7B produces output without error.
  - Dependencies: T3301.1.

- [ ] S3301.2.1 Test Mistral fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Mistral 7B with 20 tokens. Verify no error and
    coherent output.
  - Dependencies: T3301.2.

- [ ] S3301.2.2 Run go vet and go test after fix  Owner: TBD  Est: 15m
  - go vet ./..., go build ./..., go test for modified packages with -race.
  - Dependencies: T3301.2.

### E3302: Fix Phi 4 pow_scalar Capture Error

- [ ] T3302.1 Diagnose Phi 4 pow_scalar capture failure  Owner: TBD  Est: 1h
  - Run bench_tps on DGX with ZERFOO_DEBUG_ONNX=1 for Phi 4.
  - Identify the instruction that triggers the pow_scalar error.
  - Check if the exponent scalar is in the frozen set (should be kept
    CPU-resident by PreUploadFrozenWeights).
  - If not frozen: the exponent is a dynamic intermediate and needs
    different handling (cache during warmup, or mark Pow as non-capturable).
  - File: compute/gpu_kernels.go, graph/compile.go.
  - Acceptance: Root cause identified.
  - Dependencies: none.

- [ ] T3302.2 Fix Phi 4 pow_scalar error  Owner: TBD  Est: 1h
  - Apply fix based on T3302.1 diagnosis.
  - File: TBD.
  - Acceptance: bench_tps with Phi 4 produces output without kernel error.
  - Dependencies: T3302.1.

- [ ] S3302.2.1 Test Phi 4 fix on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Phi 4 with 20 tokens. Verify no error and output.
  - Dependencies: T3302.2.

- [ ] S3302.2.2 Run go vet and go test after fix  Owner: TBD  Est: 15m
  - go vet ./..., go build ./..., go test for modified packages with -race.
  - Dependencies: T3302.2.

### E3303: All-Model Verification

- [ ] T3303.1 Run all 5 models on DGX and record results  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3 (GGUF), Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Record tok/s, output quality, CUDA graph status for each.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce coherent output.
  - Dependencies: S3300.2.1, S3301.2.1, S3302.2.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: Qwen 2.5 | T3300.1, T3300.2, S3300.2.1, S3300.2.2 | DGX for diagnosis/test |
| Track B: Mistral | T3301.1, T3301.2, S3301.2.1, S3301.2.2 | DGX for diagnosis/test |
| Track C: Phi 4 | T3302.1, T3302.2, S3302.2.1, S3302.2.2 | DGX for diagnosis/test |

### Maximum parallelism

- Wave 1 (3 tasks): T3300.1 (diagnose Qwen, DGX) + T3301.1 (diagnose Mistral,
  DGX) + T3302.1 (diagnose Phi 4, DGX). All independent. DGX tasks should be
  combined into one teammate to avoid GPU contention. Local code analysis can
  run as separate teammates. Total: 1-3 teammates depending on DGX usage.

  NOTE: All 3 diagnosis tasks need DGX. Combine into 1 DGX teammate running
  sequentially, plus up to 2 local code analysis teammates. Total: up to 3.

- Wave 2 (3 tasks): T3300.2 (fix Qwen) + T3301.2 (fix Mistral) + T3302.2 (fix
  Phi 4). Each depends on its diagnosis. Local code, can run in parallel.
  Total: 3 teammates.

- Wave 3 (4 tasks): S3300.2.1 (test Qwen, DGX) + S3301.2.1 (test Mistral, DGX)
  + S3302.2.1 (test Phi 4, DGX) + lint tasks. Combine DGX tests into 1 teammate.
  Total: 1-2 teammates.

- Wave 4 (1 task): T3303.1 (all-model verification, DGX). Total: 1 teammate.

### Dependency minimization checklist applied

a) All 3 diagnosis tasks are independent and front-loaded in Wave 1.
b) Fix tasks genuinely depend on diagnosis (cannot fix without root cause).
c) DGX tasks combined to avoid GPU contention.
d) Wave 1 has 3 parallelizable tasks (code analysis portions are local).

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M500: All bugs diagnosed | T3300.1, T3301.1, T3302.1 | Root cause identified for each |
| M501: All bugs fixed | T3300.2, T3301.2, T3302.2 | Each model runs without error |
| M502: All models coherent | T3303.1 | 5/5 models produce coherent output on DGX |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3300 | Qwen 2.5 Gather OOB is in ONNX graph structure, not zerfoo code | Need to re-export model | Low | Check if head count differs from Llama 3. Qwen may use different GQA config. |
| R3301 | Mistral sliding window attention is architecturally unsupported | Major feature gap | Medium | Mistral uses Llama architecture but with sliding window. May need Mistral-specific builder. |
| R3302 | Phi 4 GeLU uses ops not in the capture-safe set | More nonCapturableOps | Medium | If pow_scalar is needed during capture for Phi 4, mark it non-capturable or cache the value. |
| R3303 | Fixing one model breaks another | Regression | Low | Always test all 5 models after each fix (DGX verification). |
| R3304 | make shared link fails on CUDA 13.0 | Build friction | Medium | Known workaround: pass .pic.o files explicitly to nvcc. |

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

### Change Summary -- 2026-03-14 (Phase 12: Per-Model Fixes)

Trimmed completed Phase 11b tasks (T3105.1 README, T3200.1 ONNX diagnosis,
T3200.2 ONNX fix for Llama 3). Stable knowledge preserved in docs/design.md:
- ONNX garbage output root causes (powf NaN, KV cache, position IDs, mask)
- Llama 3 ONNX now coherent

Created Phase 12 with 3 epics for remaining per-model bugs:
- E3300: Qwen 2.5 Gather index OOB (T3300.1, T3300.2)
- E3301: Mistral 7B Range error (T3301.1, T3301.2)
- E3302: Phi 4 pow_scalar capture error (T3302.1, T3302.2)
- E3303: All-model verification (T3303.1)

No new ADRs needed.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 232.86 tok/s Gemma 3 Q4K with CUDA graph (+26% vs no-graph).
- **Branch:** main at 48059df. All Phase 11 work merged.
- **Working models:** Gemma 3 GGUF (232.86 tok/s, coherent, 99.5% graph capture),
  Llama 3 ONNX (17.56 tok/s, coherent, 2% graph capture).
- **Broken models:**
  - Qwen 2.5: Gather index 7 OOB [0,7) during decode.
  - Mistral 7B: Range error (empty Data() on GPU scalar).
  - Phi 4: pow_scalar cuda error 1 during graph capture.
- **Key fixes applied in Phase 11:**
  - CUDA graph capture: 7 fixes (see docs/design.md item 13).
  - ONNX correctness: powf NaN, StatefulInputNode KV feedback, position IDs,
    attention mask, Greater/Where N-D broadcasting (see docs/design.md item 14).
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf, ~/models/llama3/,
  ~/models/qwen25/, ~/models/mistral/, ~/models/phi4/
- **Pre-commit hook:** Rejects multi-directory commits.
