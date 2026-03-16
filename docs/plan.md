# Zerfoo Development Plan

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

All Phase 13-16 work is complete and merged. No incomplete tasks remain.

### Current State

- 5 models run without crashes on DGX Spark.
- Gemma 3 GGUF: 232 tok/s, 99.5% CUDA graph capture.
- ONNX models: 4-16 tok/s, limited CUDA graph capture (1-4%).
- Repetition penalty (--repetition-penalty 1.2) improves ONNX output quality.
- RMSNorm fusion pattern matching works (1610 -> 1445 instructions) but fused
  Forward produces wrong numerical output. Runtime slot resolution needs fixing.
- Static Reshape is now capturable (isNonCapturable function).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- DGX requires `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- DGX uses `upstream` HTTPS remote for fetch.
- ALWAYS rebuild binary on DGX before benchmarking.

---

## 2. Scope and Deliverables

No active deliverables. Plan a new phase with `/plan`.

---

## 3. Checkable Work Breakdown

No active tasks.

---

## 4. Progress Log

### Change Summary -- 2026-03-15

Trimmed plan. Stable knowledge preserved in docs/design.md, docs/adr/, and
docs/devlog.md. Removed completed epics: E3700, E3701, E3702, E3703, E3704.

Phase 16 completed:
- E3700: Repetition penalty verified on DGX for all ONNX models.
- E3701: RMSNorm fusion pass implemented (pattern matching works, runtime blocked).
- E3702: Phi 4 TrySlice capture fix diagnosed and addressed.
- E3703: Static Reshape made capturable, ONNX capture region expanded.
- E3704: All-model verification passed on DGX.

---

## 5. Hand-off Notes

- **Current version:** v1.2.0.
- **Performance:** See docs/benchmarks.md for current baselines.
- **RMSNorm fusion status:** Pattern matching correct. Runtime slot resolution
  for fused Forward function needs investigation. See docs/devlog.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
  Use `upstream` HTTPS remote. ALWAYS rebuild binary before benchmarking.
  `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- **Key files:**
  - graph/fusion.go -- RMSNorm fusion pass
  - graph/cuda_graph.go -- CUDA graph capture, isNonCapturable()
  - compute/gpu_storage.go -- TrySlice implementation
  - cmd/bench_tps/main.go -- --repetition-penalty flag
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf, ~/models/llama3/,
  ~/models/qwen25/, ~/models/mistral/, ~/models/phi4/
- **Pre-commit hook:** Rejects multi-directory commits.
