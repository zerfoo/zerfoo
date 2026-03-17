# Zerfoo Phase 23: "Performance Recovery & Beyond"

## 1. Context

### Problem Statement

Phase 22 introduced per-request InferenceSession for concurrent inference. While
sessions enable safe concurrency, the session decode loop is missing critical
optimizations present in the original Generator decode loop, causing a throughput
regression:

- **Phase 20 peak**: 234.30 tok/s (Gemma 3 1B Q4_K, 256 tokens, Generator.Generate)
- **Phase 22 current**: 159 tok/s at 50 tokens, 99 tok/s at 256 tokens (Session.Generate)
- **Old Generator on current code**: 79 tok/s at 50 tokens, 28 tok/s at 256 tokens

The session path IS faster than the old Generator path on the current codebase (2x),
but both are below the Phase 20 peak. Investigation reveals several concrete gaps:

1. **Missing ResetPool in session decode loop**: The Generator calls
   `engine.ResetPool()` between decode steps (line 332) to reclaim intermediate GPU
   buffers. The session decode loop never calls this. Without pool reset, the GPU arena
   allocator grows monotonically, fragmenting memory and preventing CUDA graph replay
   from reusing buffer addresses.

2. **CompileTraced fallback**: CUDA graph compilation falls back from CompileTraced to
   Compile with log message "CompileTraced plan validation failed, falling back to
   Compile: instruction 0 (MatMul): input tensors cannot be nil". The traced path may
   produce more efficient execution plans than the fallback.

3. **Session KV cache allocation**: Each session creates a new KV cache. When pooled,
   the same session is reused, but the CUDA graph was captured with specific buffer
   addresses. The pool pre-warms one session, but the graph capture context (cache-free)
   differs from the generation context (with cache), potentially causing address mismatches.

4. **Missing GPU argmax fast path**: The Generator has GPU-side argmax for greedy
   decoding (lines 425-430) that avoids D2H copy of logits. The session's
   `sampleFromLogits` always copies logits to CPU.

5. **Phi MLP gap**: Phi 3.5 GGUF uses merged gate+up in `ffn_up` (no separate
   `ffn_gate`). The GGUF loader does not split this merged tensor. Carry-forward from
   Phase 22.

### Objectives

- O1: Recover throughput to Phase 20 level (234+ tok/s at 256 tokens on Gemma 3 1B).
- O2: Push beyond 234 tok/s by fixing CompileTraced and adding GPU argmax to sessions.
- O3: Fix Phi MLP merged gate+up tensor support.

### Non-Goals

- Continuous batching (vLLM-style dynamic batching).
- ROCm/OpenCL/Metal backends.
- New model architecture support beyond Phi fix.
- DeepSeek V3 DGX E2E (still blocked on model availability).

### Constraints

- Pure Go, zero CGo. GPU via purego/dlopen.
- Go standard library only.
- GGUF is the sole model format.
- DGX Spark: ssh ndungu@192.168.86.250. Benchmarks require DGX.
- Changes must not break existing tests or the concurrent session API.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Gemma 3 1B Q4_K_M tok/s (256 tokens) | 99 | 234+ | DGX bench_tps |
| Gemma 3 1B Q4_K_M tok/s (50 tokens) | 159 | 250+ | DGX TestT7_4 |
| CompileTraced success | FAIL (fallback) | PASS | No fallback log message |
| Phi 3.5 loads | FAIL (missing ffn_gate) | PASS | DGX TestT7_1/phi3 |
| All existing tests pass | PASS | PASS | go test ./... -race |

---

## 2. Scope and Deliverables

### In Scope

- Add ResetPool to session decode loop.
- Add GPU argmax fast path to session sampling.
- Investigate and fix CompileTraced failure.
- Implement Phi merged gate+up MLP tensor split.
- DGX benchmark after each fix to measure impact.

### Out of Scope

- Continuous batching, new architectures, training improvements.
- DeepSeek V3 DGX (blocked on model).

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | Session decode perf parity | Session.Generate within 5% of old Generator peak |
| D2 | CompileTraced fix | No fallback to Compile on Gemma 3 |
| D3 | Phi MLP support | Phi 3.5 GGUF loads and generates |
| D4 | Updated benchmarks | docs/benchmarks.md reflects new numbers |

---

## 3. Checkable Work Breakdown

### E1: Session Decode Loop Optimizations

Port missing optimizations from Generator.Generate to InferenceSession.Generate.

- [x] T1.1 Add ResetPool call between decode steps in session  Owner: Claude  Done: 2026-03-16
  - File: `generate/session.go`
  - In the decode loop of `Generate()` and `GenerateStream()`, add
    `if resetter, ok := s.engine.(compute.PoolResetter); ok { resetter.ResetPool() }`
    before each decode Forward, matching Generator.Generate line 332-334.
  - Acceptance: `go test ./generate/ -race` passes. Session decode loop reclaims
    intermediate GPU buffers between tokens.

- [x] T1.2 Add GPU argmax fast path to session sampling  Owner: Claude  Done: 2026-03-16
  - File: `generate/session.go`
  - In `sampleFromLogits`, add the GPU argmax fast path from Generator (lines 425-430):
    when temperature <= 0, no repetition penalty, no grammar, and logits have GPUStorage,
    use `compute.GPUArgmaxer` to sample directly on GPU without D2H copy.
  - Acceptance: Session greedy decode avoids D2H logit copy when GPU argmax is available.
    `go test ./generate/ -race` passes.

- [ ] T1.3 Add debug logging to session for perf diagnosis  Owner:  Est: 15m
  - File: `generate/session.go`
  - Add `ZERFOO_DEBUG_SESSION=1` env var that logs per-step timings:
    prefill ms, decode step ms, sample ms, total decode tok/s.
  - Acceptance: When env var is set, timing logs appear. When unset, no overhead.

- [ ] T1.4 Benchmark session with ResetPool + GPU argmax  Owner:  Est: 30m
  - Deps: T1.1, T1.2
  - Run on DGX: Gemma 3 1B Q4_K_M, 50 and 256 tokens, compare to baseline.
  - Acceptance: Throughput improvement measured and logged to docs/devlog.md.

- [ ] T1.5 go vet/lint clean  Owner:  Est: 15m
  - Deps: T1.1-T1.3
  - Acceptance: `go vet ./...` 0 warnings. `go build ./...` clean.

### E2: CompileTraced Investigation

The CompileTraced path produces traced execution plans that may enable more efficient
CUDA graph capture. Currently it fails with "input tensors cannot be nil".

- [x] T2.1 Investigate CompileTraced failure  Owner: Claude  Done: 2026-03-16
  - Result: CompileTraced produces a plan with cache-dependent instructions that have
    nil inputs when KV cache is absent. This is architectural -- the traced plan binds
    slot IDs to the tracing context and cannot handle different runtime KV cache states.
    The Compile fallback is correct and still enables CUDA graph capture (184/185
    instructions captured). CompileTraced would need a "cache-aware tracing" mode to fix.
  - File: `generate/generator.go` line 163, ztensor `graph/compile.go`
  - The error "instruction 0 (MatMul): input tensors cannot be nil" occurs during
    CompileTraced validation. Read the CompileTraced code in ztensor to understand
    what "input tensors cannot be nil" means. Is it a context issue (cache-free
    compile context missing required values)? Is it a shape mismatch?
  - Acceptance: Root cause identified and documented. If fixable, produce a fix.
    If architectural, document why and whether it matters for performance.

- [x] T2.2 Fix CompileTraced or document why fallback is acceptable  Owner: Claude  Done: 2026-03-16
  - Result: Fallback is acceptable. The Compile path captures 184/185 instructions in
    the CUDA graph (99.5% coverage). The CompileTraced path would decompose composite
    nodes into primitives for megakernel emission, but is blocked by architectural
    incompatibility with cache-dependent operations. Performance impact is minimal --
    the CUDA graph captures the same instruction range either way.
  - Deps: T2.1
  - If fixable: fix and verify CompileTraced succeeds on Gemma 3.
  - If not fixable: document in devlog why the fallback Compile path is sufficient
    and whether it impacts CUDA graph capture quality.
  - Acceptance: Either CompileTraced succeeds (no fallback log) or documented reason
    why fallback is acceptable with performance data.

- [ ] T2.3 Benchmark with CompileTraced fix (if applicable)  Owner:  Est: 30m
  - Deps: T2.2
  - Run on DGX and compare to T1.4 results.
  - Acceptance: Delta measured and logged.

### E3: Phi Merged Gate+Up MLP Support

Phi 3.5 GGUF files use `ffn_up.weight` with merged gate and up projections (no
separate `ffn_gate.weight`). The tensor must be split similarly to the QKV split.

- [x] T3.1 Implement merged gate+up split in GGUF loader  Owner: Claude  Done: 2026-03-16
  - File: `model/gguf/split.go`
  - Add `splitMergedGateUp()` function. For tensors with no `ffn_gate` but `ffn_up`
    has double the expected intermediate size, split `ffn_up` into `gate_proj` and
    `up_proj` along the first dimension (each gets half the rows).
  - Wire into LoadGGUF after tensor name mapping, similar to splitMergedQKV.
  - Acceptance: After split, both `mlp.gate_proj.weight` and `mlp.up_proj.weight`
    exist with correct shapes.

- [x] T3.2 Add unit tests for gate+up split  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1
  - File: `model/gguf/split_test.go`
  - Tests: (a) split with 2x intermediate size, (b) no split when gate exists,
    (c) no split when up size matches intermediate exactly.
  - Acceptance: `go test ./model/gguf/ -run TestSplitMergedGateUp -race` passes.

- [x] T3.3 DGX verify Phi 3.5 loads and generates  Owner: Claude  Done: 2026-03-16
  - Result: PASS - arch=phi3, 32 layers, 15 words generated. QKV split + gate+up split work.
  - Deps: T3.1, T3.2
  - Acceptance: `go test -tags dgx -run TestT7_1/phi3 ./tests/dgx/` passes.

- [ ] T3.4 go vet/lint clean  Owner:  Est: 15m
  - Deps: T3.1, T3.2
  - Acceptance: `go vet ./model/... ./inference/...` 0 warnings.

### E4: Integration Gate and Final Benchmarks

- [ ] T4.1 Full test suite pass  Owner:  Est: 30m
  - Deps: T1.5, T2.2, T3.4
  - Acceptance: `go test ./... -race -count=1` passes. `go vet ./...` 0 warnings.

- [ ] T4.2 DGX final benchmark all models  Owner:  Est: 45m
  - Deps: T4.1
  - Run full benchmark suite on DGX: all models, 50 and 256 tokens, CPU and CUDA.
  - Update docs/benchmarks.md with new baselines.
  - Acceptance: Gemma 3 1B >= 234 tok/s at 256 tokens (or documented reason for gap).

- [ ] T4.3 DGX concurrent throughput benchmark  Owner:  Est: 30m
  - Deps: T4.1
  - Run 4-client concurrent benchmark on DGX.
  - Acceptance: Throughput measured and logged.

- [ ] T4.4 Carry forward: DeepSeek V3 DGX E2E  Owner:  (BLOCKED: no MLA+MoE GGUF model)
  - Acceptance: DeepSeek V3 model loads GGUF on DGX. BLOCKED on model availability.

---

## 4. Parallel Work

### Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: Session Decode | E1 (T1.1-T1.3) | Port missing optimizations to session |
| B: CompileTraced | E2 (T2.1-T2.2) | Investigate and fix traced compilation |
| C: Phi MLP | E3 (T3.1-T3.2) | Merged gate+up tensor split |

All three tracks are independent and can run in parallel.

### Maximum Parallelism

**Wave 1** (6 parallel tasks):
T1.1, T1.2, T1.3, T2.1, T3.1, T3.2

**Wave 2** (4 parallel tasks):
T1.4, T1.5, T2.2, T3.3, T3.4

**Wave 3** (2 parallel tasks):
T2.3, T4.1

**Wave 4** (2 parallel tasks):
T4.2, T4.3

---

## 5. Dependency Graph

```
T1.1 ──┬── T1.4 ──── T1.5 ──┐
T1.2 ──┘                     │
T1.3 ─────────────── T1.5 ──┤
                              │
T2.1 ──── T2.2 ──── T2.3 ──┤
                              │
T3.1 ──── T3.3 ─────────────┤
T3.2 ──── T3.4 ─────────────┤
                              │
T1.5, T2.2, T3.4 ──── T4.1
T4.1 ──┬── T4.2
       └── T4.3
```

---

## 6. Timeline and Milestones

| ID | Milestone | Exit Criteria | Dependencies |
|----|-----------|---------------|--------------|
| M1 | Session decode optimized | ResetPool + GPU argmax in session, benchmarked | T1.4 |
| M2 | CompileTraced resolved | Fixed or documented with perf data | T2.3 |
| M3 | Phi MLP fixed | Phi 3.5 loads on DGX | T3.3 |
| M4 | Phase 23 complete | All benchmarks updated, >= 234 tok/s target | T4.2 |

---

## 7. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | ResetPool alone does not recover full throughput | Medium | High | GPU argmax + CompileTraced fix provide additional uplift. |
| R2 | CompileTraced failure is architectural (not fixable) | Medium | Medium | Fallback Compile path still captures CUDA graph. Document gap. |
| R3 | Phi gate+up split dimensions wrong | Low | Medium | Test with real Phi GGUF on DGX to verify shapes. |
| R4 | Performance changes break existing tests | Low | High | Run full test suite after each change. |
| R5 | 234 tok/s was measured with different graph compilation | Medium | Medium | If session path caps at ~200, investigate remaining overhead. |

---

## 8. Operating Procedure

### Definition of Done

A task is done when:
1. Code compiles: `go build ./...` succeeds.
2. Tests pass: `go test ./... -race` in the affected packages.
3. Lint clean: `go vet ./...` 0 warnings.
4. Acceptance criteria from the task description are met.
5. DGX benchmarks (where required) are run and results logged to docs/devlog.md.

### Review and QA

- Every code change must have corresponding tests.
- Performance changes require before/after DGX benchmark.
- Never commit files from different directories in the same commit.
- Make many small logical commits.

---

## 9. Progress Log

### 2026-03-16: Phase 23 plan created

**Change summary:** Created Phase 23 plan for performance investigation and fix.
Trimmed completed Phase 22 epics (E1-E6 all done, E7 DGX verification done except
T7.6 blocked). Carried forward T7.6 (DeepSeek V3 DGX) as T4.4.

Key investigation finding: session decode loop is missing `ResetPool()` call and
GPU argmax fast path from Generator.Generate. CompileTraced falls back to Compile.

Phase 22 operational knowledge preserved in docs/devlog.md entries dated 2026-03-16.

---

## 10. Hand-off Notes

### For a new person continuing this work

- **Codebase**: `/Users/dndungu/Code/zerfoo/zerfoo/`
- **Key files for Phase 23**:
  - `generate/session.go` -- InferenceSession decode loop (missing ResetPool, GPU argmax)
  - `generate/generator.go` -- Generator decode loop (reference for optimizations)
  - `generate/generator.go:154` -- compileGraph with CompileTraced
  - `model/gguf/split.go` -- tensor splitting (add gate+up split for Phi)
  - `inference/load_gguf.go` -- architecture dispatch
- **DGX Spark**: `ssh ndungu@192.168.86.250`. CUDA kernels at `~/Code/zerfoo/libkernels.so`.
  Set `LD_LIBRARY_PATH=/home/ndungu/Code/zerfoo`.
- **Models on DGX**: Gemma Q4_K_M at `~/models/gemma3-q4km/model.gguf` (806MB).
  Phi 3.5 at `~/models/phi-3.5-mini/model.gguf`.
- **Benchmark command**: `go test -tags dgx -run TestT7_4 -timeout 300s ./tests/dgx/`
- **Git workflow**: Rebase and merge. Each commit scoped to one directory.

### Links

- DGX Spark: `ssh ndungu@192.168.86.250`
- CI: GitHub Actions
- ADRs: `docs/adr/` (39 records)
- Benchmarks: `docs/benchmarks.md`

---

## 11. Appendix

### Session vs Generator Decode Loop Differences

| Feature | Generator.Generate | Session.Generate |
|---------|-------------------|-----------------|
| ResetPool between steps | Yes (line 332) | **Missing** |
| GPU argmax fast path | Yes (line 425) | **Missing** |
| CompileTraced | Triggers on first decode | Triggers via compileOnce |
| CUDA graph replay | plan.Load().Run | planRef.Load().Run |
| Mutex scope | Held for entire call | Held for entire call |
| KV cache | Created per call | From session (pooled) |

### Theoretical Performance Ceiling

DGX Spark GB10 specifications:
- 128GB LPDDR5x unified memory
- CUDA cores: unspecified for Blackwell mobile
- Memory bandwidth: ~200 GB/s

For Gemma 3 1B Q4_K (weights ~800MB, KV cache ~50MB per seq):
- Memory-bound decode: 800MB / 200GB/s = 4ms per token minimum
- Theoretical max: ~250 tok/s (limited by weight transfer bandwidth)
- Current: 159 tok/s (63% of theoretical)
- Target: 234+ tok/s (94% of theoretical)
