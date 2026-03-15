# Zerfoo Development Plan -- Phase 14: ONNX Precision + GPU-Native Ops + Sampling

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 13 confirmed all ONNX models run without crashes on DGX. The compute
engine is correct -- broadcastShape, Or broadcasting, and flattenTo2D collapse
are all fixed. 42+ coverage tests verify broadcast correctness.

The remaining issue is **float32 precision accumulation**. Comparison with ONNX
Runtime shows the first 3 generated tokens match exactly; divergence begins at
token 4 due to ~0.001/layer attention score drift compounding through 16
transformer layers. This is not a bug -- it is inherent to float32 with
different GEMM accumulation orders.

Four improvements will reduce precision drift and improve output quality:

1. **GPU-native Cos/Sin/Expand/ScatterND ops.** These ops currently force D2H
   copies (`.Data()` on GPU tensors), creating CPU/GPU execution boundaries
   that change float32 accumulation patterns. Keeping all computation on GPU
   eliminates unnecessary precision-boundary crossings and improves performance.

2. **Qwen 2.5 / Mistral model-specific investigation.** Qwen produces
   single-token repetition ("fox fox fox"), Mistral produces garbled tokens.
   These patterns suggest model-specific bugs beyond precision drift (e.g.,
   attention mask construction, tokenizer integration, head count mismatch).

3. **Repetition penalty in sampling.** Small models at temp=0 without
   repetition penalty naturally produce repetitive output. Adding repetition
   penalty to the sampling path will improve output quality for all models.

4. **FP16 mixed precision.** Running with fp16 weights and fp32 compute
   constrains the precision space and may improve alignment with ORT (which
   uses fp16 by default for transformer models).

### Objectives

- O1: Implement GPU-native Cos, Sin, Expand, ScatterND ops.
- O2: Investigate and fix Qwen 2.5 and Mistral model-specific output issues.
- O3: Add repetition penalty to the generation sampling loop.
- O4: All 5 models produce usable output on DGX.

### Non-Goals

- New model architectures.
- Multi-GPU / distributed inference.
- Training, fine-tuning, RLHF.
- Matching ORT output bit-for-bit (float32 precision limits make this impossible).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- DGX requires `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- All 5 models currently run without crashes (Phases 11-13 fixes).
- Llama 3 produces semi-coherent output (first 3 tokens correct).

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| GPU-native ops | Cos/Sin/Expand/ScatterND run on GPU without D2H | bench_tps with ZERFOO_DEBUG_GPU=1 |
| Repetition penalty | Less repetitive output at temp=0 | bench_tps output inspection |
| Qwen 2.5 | Coherent text (not single-token repetition) | bench_tps on DGX |
| Mistral 7B | Coherent text (not garbled) | bench_tps on DGX |
| No regression | Gemma 3 GGUF still 230+ tok/s | bench_tps on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D700 | GPU Cos/Sin kernels | Eliminate D2H for rotary embedding ops |
| D701 | GPU Expand op | Eliminate D2H for attention mask expansion |
| D702 | GPU ScatterND op | Eliminate D2H for KV cache scatter |
| D703 | Repetition penalty in sampling | Better output quality for all models |
| D704 | Qwen 2.5 model-specific fix | Model coverage |
| D705 | Mistral 7B model-specific fix | Model coverage |
| D706 | All-model verification on DGX | Confirm improvements |

### Out of Scope

- FP16 mixed precision (deferred to Phase 15).
- New model architectures.
- Training, fine-tuning, RLHF.
- Bit-for-bit ORT matching.

---

## 3. Checkable Work Breakdown

### E3500: GPU-Native Cos/Sin Kernels

Cos and Sin ops in layers/core/cos.go and layers/core/sin.go call .Data() on
GPU tensors, forcing D2H copies. These are used in rotary position embedding
for ONNX models.

- [x] T3500.1 Add GPU Cos kernel  Owner: agent  Done: 2026-03-15
  - Add a CUDA cos kernel to internal/cuda/kernels/elementwise.cu:
    `kernel_cos(float* a, float* c, int n)` using `cosf()`.
  - Add launcher and purego binding in elementwise_purego.go.
  - Wire through KernelRunner interface and GPUEngine.
  - Add gpuCos method to GPUEngine (same pattern as gpuExp, gpuSqrt).
  - File: internal/cuda/kernels/elementwise.cu, compute/gpu_engine.go.
  - Acceptance: Cos on GPU tensors runs on GPU without D2H copy.
  - Dependencies: none.

- [x] S3500.1.1 Test GPU Cos kernel  Owner: agent  Done: 2026-03-15
  - Write CPU vs GPU parity test for Cos with various shapes.
  - go test ./compute/... -race.
  - Dependencies: T3500.1.

- [x] T3500.2 Add GPU Sin kernel  Owner: agent  Done: 2026-03-15
  - Same pattern as T3500.1 but using `sinf()`.
  - File: internal/cuda/kernels/elementwise.cu, compute/gpu_engine.go.
  - Acceptance: Sin on GPU tensors runs on GPU without D2H copy.
  - Dependencies: none.

- [x] S3500.2.1 Test GPU Sin kernel  Owner: agent  Done: 2026-03-15
  - Write CPU vs GPU parity test for Sin.
  - Dependencies: T3500.2.

- [ ] S3500.2.2 Run make shared on DGX after kernel changes  Owner: TBD  Est: 15m
  - Rebuild libkernels.so with CUDA_ARCH=sm_121.
  - Dependencies: T3500.1, T3500.2.

### E3501: GPU-Native Expand Op

Expand op in layers/core/expand.go operates entirely on CPU data. For GPU
tensors, it forces D2H copies. Used in attention mask construction.

- [x] T3501.1 Add GPU Expand using broadcast kernel  Owner: agent  Done: 2026-03-15
  - Extend GPUEngine to handle Expand via the existing broadcast4D kernel.
  - Expand(x, target_shape) is equivalent to broadcasting x against a tensor
    of ones with target_shape. Reuse gpuBroadcast4DOp with a fill-ones buffer.
  - Alternative: implement Expand as a stride-based view (zero-copy for
    contiguous expansions).
  - File: compute/gpu_engine.go, layers/core/expand.go.
  - Acceptance: Expand on GPU tensors stays GPU-resident.
  - Dependencies: none.

- [x] S3501.1.1 Test GPU Expand  Owner: agent  Done: 2026-03-15
  - CPU vs GPU parity test for Expand with shapes [1,5,1] -> [3,5,7].
  - Dependencies: T3501.1.

### E3502: GPU-Native ScatterND Op

ScatterND in layers/core/scatternd.go operates on CPU data. Used by ONNX
models for KV cache updates.

- [ ] T3502.1 Add GPU ScatterND  Owner: TBD  Est: 2h
  - Implement ScatterND on GPU. This is more complex than Cos/Sin because
    it requires reading index tensors and performing scattered writes.
  - Option A: GPU kernel with atomicAdd for scattered updates.
  - Option B: Keep indices on CPU, compute offsets, then use D2D memcpy
    for the actual data scatter (avoids atomic contention).
  - File: internal/cuda/kernels/, compute/gpu_engine.go, layers/core/scatternd.go.
  - Acceptance: ScatterND with GPU data tensor stays GPU-resident for data.
  - Dependencies: none.

- [ ] S3502.1.1 Test GPU ScatterND  Owner: TBD  Est: 30m
  - CPU vs GPU parity test for ScatterND.
  - Dependencies: T3502.1.

### E3503: Repetition Penalty in Sampling

Small models at temp=0 produce repetitive output without repetition penalty.
Adding this to the sampling path improves quality for all models.

- [x] T3503.1 Add repetition penalty to sampling  Owner: agent  Done: 2026-03-15
  - In generate/sampling.go (or wherever top-k/top-p sampling is implemented),
    add a repetition_penalty parameter (default 1.0 = no penalty).
  - For each generated token, check if it appeared in the last N tokens.
  - If so, divide its logit by repetition_penalty (for values > 0) or
    multiply by repetition_penalty (for values < 0).
  - Wire the parameter through the generation API and CLI flags.
  - File: generate/sampling.go, generate/generator.go, cmd/cli/.
  - Acceptance: bench_tps with --repetition-penalty 1.2 produces less
    repetitive output.
  - Dependencies: none.

- [x] S3503.1.1 Test repetition penalty  Owner: agent  Done: 2026-03-15
  - Unit test: verify logit modification for repeated tokens.
  - Test with penalty=1.0 (no change) and penalty=1.5 (reduced logits).
  - Dependencies: T3503.1.

### E3504: Qwen 2.5 Model-Specific Investigation

Qwen produces "fox fox fox..." (single-token repetition). This pattern
suggests a bug beyond precision drift -- possibly in attention mask
construction or head count handling.

- [x] T3504.1 Diagnose Qwen 2.5 repetition  Owner: agent  Done: 2026-03-15
  - Run Qwen on DGX with debug logging at attention mask, KV cache, and
    logit stages.
  - Compare attention mask shape vs expected (Qwen uses 7 KV heads, not 8).
  - Check if the Gather index clamping fix (f0e4897) is masking a deeper
    issue with head count.
  - Acceptance: Root cause of single-token repetition identified.
  - Dependencies: none.

- [ ] T3504.2 Fix Qwen 2.5 output  Owner: TBD  Est: 1h
  - Apply fix based on diagnosis.
  - Dependencies: T3504.1.

- [ ] S3504.2.1 Test Qwen fix on DGX  Owner: TBD  Est: 15m
  - bench_tps for Qwen 2.5 with 20 tokens.
  - Acceptance: Not single-token repetition.
  - Dependencies: T3504.2.

### E3505: Mistral 7B Model-Specific Investigation

Mistral produces garbled tokens without spaces. May be a tokenizer issue
(SentencePiece vs BPE) or sliding window attention handling.

- [ ] T3505.1 Diagnose Mistral garbled output  Owner: TBD  Est: 1.5h
  - Run Mistral on DGX with debug logging.
  - Check tokenizer: is the BPE tokenizer handling Mistral's vocabulary
    correctly? Compare token IDs with Python tokenizer.
  - Check if sliding window attention parameters are applied correctly.
  - Acceptance: Root cause of garbled output identified.
  - Dependencies: none.

- [ ] T3505.2 Fix Mistral output  Owner: TBD  Est: 1h
  - Apply fix based on diagnosis.
  - Dependencies: T3505.1.

- [ ] S3505.2.1 Test Mistral fix on DGX  Owner: TBD  Est: 15m
  - bench_tps for Mistral 7B with 20 tokens.
  - Dependencies: T3505.2.

### E3506: All-Model Verification

- [ ] T3506.1 Run all 5 models on DGX with improvements  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3, Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Use --repetition-penalty 1.2 for ONNX models.
  - Record tok/s, output quality, D2H copy count.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce usable output. Zero D2H copies for
    Cos/Sin/Expand during inference.
  - Dependencies: S3500.2.2, S3501.1.1, T3503.1, S3504.2.1, S3505.2.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: GPU Cos/Sin | T3500.1, T3500.2, S3500.1.1, S3500.2.1 | CUDA kernels |
| Track B: GPU Expand | T3501.1, S3501.1.1 | Compute engine |
| Track C: GPU ScatterND | T3502.1, S3502.1.1 | CUDA kernel |
| Track D: Sampling | T3503.1, S3503.1.1 | Generation loop |
| Track E: Qwen | T3504.1, T3504.2, S3504.2.1 | DGX diagnosis |
| Track F: Mistral | T3505.1, T3505.2, S3505.2.1 | DGX diagnosis |

### Maximum parallelism

- Wave 1 (5 tasks): T3500.1 (GPU Cos) + T3500.2 (GPU Sin) + T3501.1 (GPU Expand)
  + T3503.1 (repetition penalty) + T3504.1 (diagnose Qwen, DGX). All independent.
  Total: 5 teammates. DGX tasks combined if needed.

- Wave 2 (4 tasks): T3502.1 (GPU ScatterND) + T3505.1 (diagnose Mistral, DGX)
  + S3500.1.1 (test Cos) + S3500.2.1 (test Sin). Total: up to 4 teammates.

- Wave 3 (4 tasks): T3504.2 (fix Qwen) + T3505.2 (fix Mistral) + S3501.1.1
  (test Expand) + S3503.1.1 (test repetition). Total: up to 4 teammates.

- Wave 4 (3 tasks): S3504.2.1 (test Qwen, DGX) + S3505.2.1 (test Mistral, DGX)
  + S3500.2.2 (rebuild .so, DGX). Combine DGX tasks. Total: 1-2 teammates.

- Wave 5 (1 task): T3506.1 (all-model verification, DGX). Total: 1 teammate.

### Dependency minimization checklist applied

a) GPU kernel tasks (Cos, Sin, Expand, ScatterND) are all independent.
b) Repetition penalty has no code dependency on GPU ops.
c) Qwen/Mistral diagnosis can start immediately on DGX.
d) Wave 1 saturates 5 teammates with independent work.
e) ScatterND deferred to Wave 2 to prioritize simpler ops.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M700: GPU ops implemented | T3500.1, T3500.2, T3501.1, T3502.1 | Cos/Sin/Expand/ScatterND run on GPU |
| M701: Sampling improved | T3503.1 | Repetition penalty available via CLI flag |
| M702: Model bugs fixed | T3504.2, T3505.2 | Qwen/Mistral produce coherent output |
| M703: All models verified | T3506.1 | 5/5 models produce usable output on DGX |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3500 | GPU Cos/Sin kernels have different precision than CPU math.Cos/Sin | Minor numerical drift | Low | Use cosf/sinf which match Go's float32 math. Add parity tests. |
| R3501 | GPU ScatterND with atomics is slower than CPU for small scatter counts | No perf gain | Medium | Use option B (CPU indices + D2D memcpy) if atomic approach is slow. |
| R3502 | Qwen/Mistral issues are in the zonnx converter, not zerfoo | Different repo to fix | Medium | Read the ONNX model files directly to check if the graph structure is correct. |
| R3503 | Repetition penalty changes expected output for existing tests | Test breakage | Low | Default penalty=1.0 means no change. Only active when explicitly set. |
| R3504 | make shared link fails on CUDA 13.0 | Build friction | Medium | Known workaround: pass .pic.o files explicitly. |

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

### Change Summary -- 2026-03-15 (Phase 14: Precision + GPU Ops + Sampling)

Trimmed completed Phase 13 work (broadcastShape audit, Or broadcasting fix,
flattenTo2D collapse fix, 42+ broadcast coverage tests). Stable knowledge
preserved in docs/design.md: broadcastShape is correct, Or uses validatedBroadcast,
flattenTo2D element-count guard, float32 precision accumulation analysis.

Phase 13 investigation confirmed ONNX output quality is a float32 precision
issue (not a bug). First 3 tokens match ORT exactly; divergence at token 4
from ~0.001/layer drift. Contributing factors: CPU/GPU bouncing in Cos/Sin/
Expand/ScatterND ops, decomposed RMSNorm accumulation order.

Phase 14 targets 4 improvement areas:
- E3500: GPU-native Cos/Sin kernels (eliminate D2H in rotary embedding)
- E3501: GPU-native Expand (eliminate D2H in attention mask)
- E3502: GPU-native ScatterND (eliminate D2H in KV cache scatter)
- E3503: Repetition penalty in sampling (improve output quality)
- E3504: Qwen 2.5 model-specific fix
- E3505: Mistral 7B model-specific fix
- E3506: All-model verification

No new ADRs needed.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0.
- **Performance:** 232.86 tok/s Gemma 3 Q4K with CUDA graph (+26%).
- **Branch:** main at 400fad8. All Phase 11-13 work merged.
- **Model status (all run without crashes):**
  - Gemma 3 GGUF: 232 tok/s, coherent, 99.5% graph capture
  - Llama 3 ONNX: 12.68 tok/s, semi-coherent (first 3 tokens correct, then repetitive)
  - Qwen 2.5 ONNX: 15.40 tok/s, single-token repetition ("fox fox fox")
  - Mistral 7B ONNX: 3.94 tok/s, garbled tokens
  - Phi 4 ONNX: 4.25 tok/s, semi-coherent but noisy
- **Root cause of ONNX output quality:** Float32 precision accumulation (~0.001/layer
  drift). NOT a bug. Cos/Sin/Expand/ScatterND D2H bouncing contributes.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
  IMPORTANT: `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- **Key files:**
  - layers/core/cos.go, layers/core/sin.go -- D2H bouncing ops to fix
  - layers/core/expand.go -- D2H bouncing op to fix
  - layers/core/scatternd.go -- D2H bouncing op to fix
  - generate/sampling.go -- add repetition penalty
  - internal/cuda/kernels/elementwise.cu -- add cos/sin kernels
- **Pre-commit hook:** Rejects multi-directory commits.
