# Zerfoo Development Plan -- Phase 13: ONNX Execution Path Correctness

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

All ONNX models (Llama 3, Qwen 2.5, Mistral 7B, Phi 4) fail during inference
on DGX with a properly rebuilt libkernels.so. The failures all involve shape
mismatches in GPU compute operations during the ONNX execution path.

**Observed errors (all on DGX with sm_121 .so rebuilt):**

| Model | Error | Node | Details |
|-------|-------|------|---------|
| Llama 3 | MatMul 1D vs 2D | node[106] | Input shapes [[2048] [2048 2048]], dep ops [Mul Parameter] |
| Qwen 2.5 | Runs but poor output | -- | 5.26 tok/s, repetitive/garbage text |
| Mistral 7B | Or shape mismatch | node[98] | Input sizes differ (4 vs 2), boolean broadcasting |
| Phi 4 | Add size mismatch | node[125] | Storage/tensor size mismatch |

**Common root cause pattern:** GPU compute operations produce tensors with
incorrect shapes. Specifically, broadcasting operations like `gpuBroadcastOp`
and `gpuBroadcast4DOp` in compute/gpu_kernels.go may incorrectly collapse
leading unit dimensions. When `[1,1,2048] * [2048]` is broadcast, the output
shape should be `[1,1,2048]` (3D) but the GPU path may return `[2048]` (1D).
Downstream MatMul then fails because it requires 2D+ inputs.

The ZMF codegen pipeline (Gemma 3 GGUF) avoids these issues because it uses
fused ops (GroupedQueryAttention, FusedAddRMSNorm, FFN) that handle shapes
internally. The ONNX path decomposes these into individual ops (Mul, Add, Pow,
ReduceMean, Sqrt, Div, Reshape, Transpose) which expose the broadcasting bugs.

**Prior fixes already applied:**
- Phase 11: CUDA graph capture (99.5% for ZMF codegen, 232.86 tok/s)
- Phase 11: ONNX powf NaN, KV cache feedback, position IDs, attention mask
- Phase 12: Cast aliasing fix, Gather index clamping
See docs/design.md items 13-14 for full history.

### Objectives

- O1: Fix GPU broadcast output shapes to preserve leading unit dimensions.
- O2: Fix Or/boolean operation broadcasting for Mistral attention mask.
- O3: Fix Add/storage size mismatch for Phi 4.
- O4: Verify all 4 ONNX models produce coherent output on DGX.

### Non-Goals

- New model architectures.
- FP16/FP8 weight loading for ZMF models.
- Multi-GPU / distributed inference.
- CUDA graph capture for ONNX models.
- Performance optimization (correctness first).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- Gemma 3 GGUF works correctly (232.86 tok/s, 99.5% graph capture).
- DGX requires `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Llama 3 output | Coherent text at temp=0 | bench_tps on DGX |
| Qwen 2.5 output | Coherent text at temp=0 | bench_tps on DGX |
| Mistral 7B output | No crash, coherent text | bench_tps on DGX |
| Phi 4 output | No crash, coherent text | bench_tps on DGX |
| No regression | Gemma 3 GGUF still 230+ tok/s | bench_tps on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D600 | Fix broadcastShape to preserve leading unit dims | Fixes Llama 3 MatMul 1D error |
| D601 | Fix Or/boolean broadcasting for Mistral | Fixes Mistral Or shape mismatch |
| D602 | Fix Add storage size mismatch for Phi 4 | Fixes Phi 4 Add crash |
| D603 | All-model verification on DGX | Confirm all 5 models work |

### Out of Scope

- FP16 weight conversion for ZMF models.
- New model architectures.
- CUDA graph capture for ONNX models.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E3400: Fix GPU Broadcast Output Shape

The broadcastShape function in compute/gpu_kernels.go computes the NumPy-style
broadcast output shape. It may strip leading unit dimensions (e.g., returning
[2048] instead of [1,1,2048]) which causes downstream MatMul to fail.

- [ ] T3400.1 Audit broadcastShape and GPU broadcast ops  Owner: TBD  Est: 1.5h
  - Read compute/gpu_kernels.go: broadcastShape, gpuBroadcastOp, gpuBroadcast4DOp.
  - Write a unit test: broadcastShape([1,1,2048], [2048]) must return [1,1,2048].
  - Write a unit test: gpuBroadcastOp (Mul) with [1,1,2048] * [2048] must
    return a tensor with shape [1,1,2048], not [2048].
  - Check if makeGPUResult uses the correct output shape from broadcastShape.
  - Check if gpuBroadcastOp's leading-dimension broadcast path (trailingDimsMatch)
    computes outShape correctly.
  - File: compute/gpu_kernels.go, compute/gpu_kernels_test.go.
  - Acceptance: Unit tests written that expose the bug. Tests fail before fix.
  - Dependencies: none.

- [ ] T3400.2 Fix broadcastShape output shape  Owner: TBD  Est: 1h
  - Fix broadcastShape to preserve leading unit dimensions in the output shape.
  - Fix gpuBroadcastOp to pass the correct N-D output shape to makeGPUResult
    (currently may flatten to 2D [M,D] and lose the original shape).
  - Verify the gpuBroadcast4DOp path also preserves shapes.
  - File: compute/gpu_kernels.go.
  - Acceptance: T3400.1 unit tests pass. broadcastShape([1,1,2048], [2048])
    returns [1,1,2048].
  - Dependencies: T3400.1.

- [ ] S3400.2.1 Run go vet and full test suite  Owner: TBD  Est: 15m
  - go vet ./..., go build ./..., go test ./compute/... -race -timeout 120s.
  - Dependencies: T3400.2.

- [ ] S3400.2.2 Test Llama 3 on DGX  Owner: TBD  Est: 15m
  - Preflight: ssh to DGX, git pull, rebuild .so.
  - Run bench_tps for Llama 3 with 20 tokens.
  - Acceptance: No MatMul 1D error. Output produced.
  - Dependencies: T3400.2.

### E3401: Fix Boolean Op Broadcasting (Mistral)

Mistral 7B fails at node[98] (Or) with "input sizes differ (4 vs 2)". The Or
op is used for attention mask computation and needs to support N-D broadcasting
for boolean tensors.

- [ ] T3401.1 Audit Or/And/boolean ops for broadcasting  Owner: TBD  Est: 1h
  - Read layers/core/ for Or, And, Greater, Less, Equal implementations.
  - Check if they support N-D broadcasting or require same-shape inputs.
  - The error "input sizes differ" likely means the op checks storage lengths
    instead of broadcast-compatible shapes.
  - Write a unit test: Or with shapes [1,5,1] and [1,1,13] must broadcast
    to [1,5,13].
  - File: layers/core/.
  - Acceptance: Root cause identified. Unit test written that fails.
  - Dependencies: none.

- [ ] T3401.2 Add broadcasting to Or/boolean ops  Owner: TBD  Est: 1.5h
  - Extend Or (and other boolean ops if affected) to support N-D broadcasting.
  - Use the same broadcast pattern as Greater/Where (which were already fixed
    in Phase 11 with validatedBroadcast helper).
  - File: layers/core/.
  - Acceptance: T3401.1 unit tests pass. Or([1,5,1], [1,1,13]) returns [1,5,13].
  - Dependencies: T3401.1.

- [ ] S3401.2.1 Run go vet and test  Owner: TBD  Est: 15m
  - go vet ./..., go build ./..., go test ./layers/core/... -race.
  - Dependencies: T3401.2.

- [ ] S3401.2.2 Test Mistral on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Mistral 7B with 20 tokens.
  - Acceptance: No Or shape error. Output produced.
  - Dependencies: T3401.2, S3400.2.2 (broadcast fix may also be needed).

### E3402: Fix Add Storage Size Mismatch (Phi 4)

Phi 4 fails at node[125] (Add) with "storage/tensor size mismatch". This is
likely the same broadcast shape bug as E3400 but manifesting in Add instead
of Mul/MatMul.

- [ ] T3402.1 Diagnose Phi 4 Add mismatch  Owner: TBD  Est: 45m
  - Add debug logging to Add at node[125]: print input shapes and storage lengths.
  - Run on DGX or trace locally from the ONNX graph structure.
  - Check if this is the same broadcastShape bug as E3400 or a different issue.
  - File: compute/gpu_kernels.go.
  - Acceptance: Root cause identified. Likely same fix as T3400.2.
  - Dependencies: none.

- [ ] T3402.2 Fix Phi 4 Add error  Owner: TBD  Est: 30m
  - If same root cause as E3400: verify T3400.2 fix resolves it.
  - If different: apply targeted fix.
  - File: compute/gpu_kernels.go.
  - Acceptance: bench_tps with Phi 4 produces output without error.
  - Dependencies: T3402.1, T3400.2.

- [ ] S3402.2.1 Test Phi 4 on DGX  Owner: TBD  Est: 15m
  - Run bench_tps for Phi 4 with 20 tokens.
  - Dependencies: T3402.2.

### E3403: All-Model Verification

- [ ] T3403.1 Run all 5 models on DGX and record results  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3 (GGUF), Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Record tok/s, output quality for each.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce coherent output at temp=0.
  - Dependencies: S3400.2.2, S3401.2.2, S3402.2.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: Broadcast Shape | T3400.1, T3400.2, S3400.2.1 | Core fix, local code |
| Track B: Boolean Ops | T3401.1, T3401.2, S3401.2.1 | Mistral-specific, local code |
| Track C: Phi 4 Add | T3402.1, T3402.2, S3402.2.1 | May merge with Track A |
| Track D: DGX Verify | S3400.2.2, S3401.2.2, T3403.1 | DGX, sequential |

### Maximum parallelism

- Wave 1 (3 tasks): T3400.1 (audit broadcastShape) + T3401.1 (audit boolean ops)
  + T3402.1 (diagnose Phi 4 Add). All independent, all local code. 3 teammates.

- Wave 2 (3 tasks): T3400.2 (fix broadcast) + T3401.2 (fix boolean ops) +
  T3402.2 (fix Phi 4 Add). Each depends on its diagnosis. 3 teammates.

- Wave 3 (3 tasks): S3400.2.1 (lint) + S3401.2.1 (lint) + DGX tests
  (S3400.2.2 + S3401.2.2 + S3402.2.1 combined). 2-3 teammates.

- Wave 4 (1 task): T3403.1 (all-model verification, DGX). 1 teammate.

### Dependency minimization checklist applied

a) All 3 audit/diagnosis tasks are independent and front-loaded in Wave 1.
b) T3402.1 may conclude the root cause is same as T3400 -- if so, T3402.2
   is a no-op verification, not a separate fix.
c) DGX tests combined into single teammate to avoid GPU contention.
d) Wave 1 saturates 3 teammates with independent local code analysis.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M600: Root causes identified | T3400.1, T3401.1, T3402.1 | All 3 shape bugs have root cause + failing tests |
| M601: Shape bugs fixed | T3400.2, T3401.2, T3402.2 | Unit tests pass, no shape errors |
| M602: All models coherent | T3403.1 | 5/5 models produce coherent output on DGX |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3400 | broadcastShape fix breaks existing GPU ops for GGUF models | Regression | Medium | Run Gemma 3 GGUF after every fix. The broadcast change must be backward-compatible. |
| R3401 | Boolean broadcasting needs GPU kernel (not just CPU) | More work | Low | GPU Or/And kernels likely not needed; CPU fallback is acceptable for mask ops. |
| R3402 | Fixing shape bugs reveals more downstream errors | Whack-a-mole | High | Each fix may uncover the next error. Budget for 2-3 rounds of fix-test-fix. |
| R3403 | ONNX models have fundamentally different execution expectations | Architectural mismatch | Low | The ONNX standard specifies shape semantics. Fixing the compute ops to match ONNX spec is the right approach. |

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
3. cd ~/zerfoo && git pull
4. cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
5. cd ~/zerfoo
6. export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
7. Verify: /usr/local/go/bin/go run ./cmd/bench_tps --help

---

## 8. Progress Log

### Change Summary -- 2026-03-15 (Phase 13: ONNX Execution Path Correctness)

Trimmed completed Phase 12 tasks (T3300.1 diagnosis, T3300.2 Gather fix, T3301.1
diagnosis, T3301.2 Cast fix, T3302.1 diagnosis, T3302.2 now same-fix-as-T3301.2).
Stable knowledge preserved in docs/design.md: Cast aliasing fix, Gather index clamping,
remaining shape errors per model.

DGX testing with properly rebuilt .so revealed that ALL ONNX models have shape-related
errors in the GPU compute path. The common root cause is broadcasting operations that
collapse leading unit dimensions. Restructured into Phase 13 with 3 targeted epics:
- E3400: Fix broadcastShape output shape (Llama 3 MatMul 1D error)
- E3401: Fix boolean op broadcasting (Mistral Or shape mismatch)
- E3402: Fix Add storage mismatch (Phi 4, likely same root cause as E3400)
- E3403: All-model verification

No new ADRs needed.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0.
- **Performance:** 232.86 tok/s Gemma 3 Q4K with CUDA graph (+26%).
- **Branch:** main at aecf437. All Phase 11-12 work merged.
- **Working models:** Gemma 3 GGUF only (232.86 tok/s, coherent, 99.5% graph).
- **Broken ONNX models (all with rebuilt .so):**
  - Llama 3: MatMul 1D vs 2D at node[106] (Mul produces [2048] instead of [1,1,2048])
  - Qwen 2.5: Runs at 5.26 tok/s, poor output quality
  - Mistral 7B: Or shape mismatch at node[98] (boolean broadcasting)
  - Phi 4: Add storage size mismatch at node[125]
- **Common root cause:** GPU broadcast ops collapse leading unit dimensions.
- **Key files to fix:**
  - compute/gpu_kernels.go -- broadcastShape, gpuBroadcastOp, gpuBroadcast4DOp
  - layers/core/ -- Or, And, boolean op broadcasting
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
  IMPORTANT: `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf, ~/models/llama3/,
  ~/models/qwen25/, ~/models/mistral/, ~/models/phi4/
- **Pre-commit hook:** Rejects multi-directory commits.
