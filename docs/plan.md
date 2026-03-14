# Zerfoo Development Plan -- Phase 11b: README + ONNX Output Quality

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 11 (Waves 1-4) fixed CUDA graph capture for ZMF codegen models. The ZMF
codegen pipeline (fused ops: GroupedQueryAttention, FusedAddRMSNorm, FFN) captures
99.5% of instructions, achieving 232.86 tok/s (+26% vs no-graph). All graph
capture work is DONE. See docs/design.md item 13 for the full list of fixes.

Two tasks remain:

1. **README.** The project has no README. A quickstart guide with benchmark
   table is needed for first-time users.

2. **ONNX model output quality.** Llama 3, Qwen 2.5, Mistral 7B, and Phi 4
   loaded via ONNX (not ZMF codegen) produce garbage output ("!!!" characters)
   at temp=0. This is a pre-existing correctness bug unrelated to graph capture
   -- it occurs with and without CUDA graph. The GGUF/ZMF codegen pipeline
   (Gemma 3) produces correct output. The bug is in the ONNX model loading
   or execution path, not in the compute engine.

### Objectives

- O1: Write README with quickstart and benchmark table.
- O2: Diagnose and fix ONNX model garbage output on DGX.

### Non-Goals

- New model architectures.
- FP16/FP8 weight loading for ZMF models (future).
- Multi-GPU / distributed inference.
- Further graph capture optimization (already at 99.5% for ZMF codegen).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- ZMF codegen models work correctly (Gemma 3 GGUF: 232.86 tok/s, coherent).
- ONNX models run without crashes but produce garbage output.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| README | Clone to inference in 5 minutes | Manual walkthrough |
| ONNX output quality | Coherent text at temp=0 for Llama 3 + Qwen 2.5 | bench_tps output inspection |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D400 | README with quickstart and benchmark table | First-time user experience |
| D401 | ONNX output quality diagnosis | Identify root cause of garbage output |
| D402 | ONNX output quality fix | Coherent text from ONNX models |

### Out of Scope

- FP16 weight conversion for ZMF models.
- New model architectures.
- Training, fine-tuning, RLHF.
- CUDA graph capture for ONNX models (1-2% capture is acceptable; path
  forward is migrating to ZMF codegen pipeline).

---

## 3. Checkable Work Breakdown

### E3105: README

- [ ] T3105.1 Write README.md with quickstart  Owner: TBD  Est: 1.5h
  - Sections: What is Zerfoo, Installation, Quickstart (pull + run in 3
    commands), Supported Models (table with tok/s), API Usage (curl examples),
    Performance (vs Ollama), Architecture Overview, Contributing.
  - Benchmark table:
    - Gemma 3 1B GGUF Q4K: 232.86 tok/s (CUDA graph, +26%)
    - Llama 3.2 1B ZMF: 17.56 tok/s (ONNX, no graph -- output quality pending)
    - Qwen 2.5 ZMF: 7.87 tok/s (ONNX, no graph -- output quality pending)
  - File: README.md.
  - Acceptance: A new user can go from clone to inference in 5 minutes.
  - Dependencies: none.

### E3200: ONNX Model Output Quality

ONNX-loaded models (Llama 3, Qwen 2.5, Mistral 7B, Phi 4) produce garbage
output ("!!!" characters) at temp=0. The ZMF codegen pipeline (Gemma 3) works
correctly. This suggests the bug is in the ONNX graph execution path, not in
the compute engine itself.

Likely areas to investigate:
- Weight loading: incorrect tensor shapes, transposition, or data type conversion
  during ONNX-to-ZMF conversion (zonnx/ package).
- Graph execution order: topological sort may differ from expected ONNX order.
- Attention mask or position encoding: ONNX models may need different mask/position
  handling than the ZMF codegen path.
- Normalization: ONNX decomposes RMSNorm into Pow+ReduceMean+Sqrt+Div+Mul; numerical
  differences from the fused RMSNorm kernel could accumulate.
- Token sampling: the generation loop may handle ONNX logits differently.

- [ ] T3200.1 Diagnose ONNX output quality on DGX  Owner: TBD  Est: 2h
  - Run Llama 3 on DGX with debug logging at key points:
    a) After weight loading: verify weight tensor shapes and values match expected
       (compare first 10 values of embedding table, first attention Q/K/V weight).
    b) After embedding lookup: print first 5 values of embedding output.
    c) After first transformer layer: print first 5 logit values.
    d) After final layer: print top-5 token predictions and their logits.
  - Compare against a reference implementation (e.g., run same model in Python
    with transformers library, or compare with ONNX Runtime output).
  - Check if the first token prediction is correct (model produces a valid
    continuation) or garbage from the start.
  - If first token is correct but subsequent tokens degrade: the issue is in
    the autoregressive loop (KV cache, position encoding).
  - If first token is already garbage: the issue is in weight loading or
    the forward pass itself.
  - File: compute/gpu_engine.go, generate/generator.go, model/loader.go.
  - Acceptance: Root cause identified with specific layer, tensor, or code path.
  - Dependencies: none.

- [ ] S3200.1.1 Run go vet and go build after diagnostic changes  Owner: TBD  Est: 10m
  - go vet ./... and go build ./... must pass.
  - Dependencies: T3200.1.

- [ ] T3200.2 Fix ONNX output quality  Owner: TBD  Est: 2h
  - Apply fix based on T3200.1 diagnosis.
  - The fix must not regress the ZMF codegen path (Gemma 3 GGUF).
  - File: TBD based on diagnosis.
  - Acceptance: bench_tps with Llama 3 ZMF produces coherent text at temp=0.
  - Dependencies: T3200.1.

- [ ] S3200.2.1 Test ONNX fix on DGX with all ONNX models  Owner: TBD  Est: 30m
  - Run bench_tps for Llama 3, Qwen 2.5, Mistral 7B, Phi 4 on DGX.
  - Verify coherent output at temp=0 for each.
  - Also run Gemma 3 GGUF to confirm no regression.
  - File: docs/updates.md.
  - Acceptance: All 5 models produce coherent output.
  - Dependencies: T3200.2.

- [ ] S3200.2.2 Run go test and go vet after fix  Owner: TBD  Est: 15m
  - go test ./... -race -timeout 120s for modified packages.
  - go vet ./...
  - go build ./...
  - Dependencies: T3200.2.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: README | T3105.1 | Local, no dependencies |
| Track B: ONNX Diagnosis | T3200.1, S3200.1.1 | DGX for reproduction |

### Maximum parallelism

- Wave 5 (2 tasks): T3105.1 (README, local) + T3200.1 (ONNX diagnosis, DGX).
  Fully independent. Total: 2 teammates.

- Wave 6 (2 tasks): T3200.2 (fix, depends T3200.1) + S3200.2.2 (lint after fix).
  Sequential within track B. Total: 1 teammate.

- Wave 7 (1 task): S3200.2.1 (verify all models on DGX). Depends on T3200.2.
  Total: 1 teammate.

### Dependency minimization checklist applied

a) T3105.1 (README) has zero dependencies -- runs in Wave 5 alongside diagnosis.
b) T3200.1 (ONNX diagnosis) has zero dependencies -- runs immediately on DGX.
c) T3200.2 genuinely depends on T3200.1 (cannot fix without diagnosis).
d) Wave 5 has 2 tasks (README is a large task, ONNX diagnosis needs DGX time).

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M400: README published | T3105.1 | 5-minute quickstart verified |
| M401: ONNX diagnosed | T3200.1 | Root cause identified |
| M402: All models coherent | S3200.2.1 | All 5 models produce coherent output on DGX |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3200 | ONNX garbage output is in weight conversion (zonnx package) | Harder fix, different repo | Medium | Compare weight values against Python reference. If zonnx issue, fix there and re-export. |
| R3201 | ONNX garbage output has multiple independent causes per model | Per-model fixes needed | Medium | Start with Llama 3 (simplest). If fix generalizes, test on others. If not, diagnose each. |
| R3202 | Numerical accumulation from decomposed RMSNorm | Subtle precision issue | Low | Compare fused vs decomposed RMSNorm output for same input. If drift > 1e-3, use fused path. |
| R3203 | make shared link fails on CUDA 13.0 | Build friction | Medium | Known workaround: pass .pic.o files explicitly to nvcc. |

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

### Change Summary -- 2026-03-14 (Phase 11b: README + ONNX Quality)

Trimmed all completed Phase 11 graph capture epics (E3100-E3104) and their tasks.
Stable knowledge (graph capture fixes, nonCapturableOps, longest-contiguous-region
scan, benchmark results) preserved in docs/design.md item 13.

Phase 11 graph capture results:
- ZMF codegen (Gemma 3 GGUF): 232.86 tok/s, 99.5% graph capture, +26% speedup.
- ONNX models: 1-2% graph capture, garbage output (pre-existing correctness bug).

Remaining work restructured into Phase 11b:
- E3105: README with quickstart (T3105.1, no dependencies).
- E3200: ONNX output quality diagnosis + fix (T3200.1, T3200.2, S3200.2.1).

No new ADRs needed. Existing ADR 023 covers D2H elimination strategy.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 232.86 tok/s Gemma 3 Q4K with CUDA graph (+26% vs no-graph,
  beats Ollama 197.21 tok/s by 18.1%).
- **Branch:** main at 1391219. All Phase 11 graph capture work merged.
- **Graph capture status:** DONE for ZMF codegen (99.5% captured). ONNX models
  capture 1-2% (acceptable; path forward is ZMF codegen migration).
- **Current bug:** ONNX models (Llama 3, Qwen 2.5, Mistral, Phi 4) produce
  garbage output ("!!!") at temp=0. Pre-existing, unrelated to graph capture.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf (232.86 tok/s, coherent),
  ~/models/llama3/ (17.56 tok/s, garbage), ~/models/qwen25/ (7.87 tok/s, garbage),
  ~/models/mistral/ (Range error), ~/models/phi4/ (pow_scalar error during capture)
- **Pre-commit hook:** Rejects multi-directory commits.
- **Key files for ONNX diagnosis:**
  - generate/generator.go -- autoregressive token loop
  - generate/tensor_cache.go -- KV cache management
  - model/loader.go -- ZMF model loading
  - inference/ -- model architecture builders (arch_llama.go, etc.)
  - graph/compile.go -- graph compilation and execution
