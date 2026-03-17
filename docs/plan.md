# Phase 27: Exceed Ollama via llama.cpp Optimizations + RMSNorm Backward Fix

## 1. Context

See docs/design.md for full architecture. See docs/devlog.md for Phase 27
investigation history (diffs, bisect, root cause analysis) and for the
2026-03-17 RMSNorm backward pass nil pointer dereference entry.

Phase 27 restored inference throughput from 186 to 245 tok/s on Gemma 3 1B
Q4_K_M (DGX Spark GB10). Two root causes were found and fixed:
1. FlashAttentionDecode in GQA (disabled -- cuBLAS SDPA is faster).
2. Q5_K/Q6_K weights dequantized to float32 instead of re-quantized to Q4_0
   (restored Q4_0 re-quant in GGUF loader).

Current status: 245 tok/s with CUDA graphs, +20% vs Ollama (204 tok/s).

A confirmed bug was also found during Phase 27: RMSNorm.Backward() panics if
called before Forward, or if Forward fails partway through. This blocks all
training workloads that use RMSNorm (all modern transformer architectures).

llama.cpp study (T2.1-T2.4) identified three optimization opportunities:
- dp4a Q4 GEMV kernel (2-4x compute advantage over scalar FMA).
- Warp-level flash attention decode (128 threads, Q in registers, warp shuffles).
- Tensor lifetime analysis for 30-50% GPU memory reduction.

### Objectives

- Fix RMSNorm.Backward nil pointer dereference (critical, blocks training).
- Update CLAUDE.md benchmark claim to reflect verified 245 tok/s.
- Apply llama.cpp optimization ideas to push throughput beyond 245 tok/s (stretch).

### Non-Goals

- Rewrite RMSNorm from scratch -- minimal surgical fix only.
- Change the public RMSNorm API.
- Fix the no-graph baseline gap (174 vs 186 tok/s) -- carry to Phase 28.
- Fix FlashAttentionDecode kernel -- needs full kernel rewrite (carry to Phase 28).

---

## 2. Checkable Work Breakdown

### E3: Finalize Benchmark Documentation

- [x] T3.4 Update CLAUDE.md benchmark claim  Owner: Claude  Est: 10m  Done: 2026-03-17
  - Update Performance Benchmarks section in the root CLAUDE.md with the
    verified 245 tok/s number and corrected Ollama baseline of 204 tok/s.
  - Current stale claim: "234.30 tok/s decode (18.8% faster than Ollama ~197 tok/s)".
  - New claim: "245 tok/s decode (20% faster than Ollama 204 tok/s)".
  - Acceptance: CLAUDE.md Performance Benchmarks section reflects 245 tok/s and
    204 tok/s Ollama baseline.

### E5: Fix RMSNorm Backward Nil Pointer Dereference

**Background (from devlog 2026-03-17):** `RMSNorm.Backward()` in
`layers/normalization/rmsnorm.go` dereferences `r.rms` at lines 203, 240, 245,
and 250, and `r.inputTensor` at line 200, without nil checks. Forward caches
`r.rms` via three code paths (lines 131, 147, 178). If none executes (e.g.
Backward called before Forward, or Forward returns early on error), both fields
remain nil and Backward panics. The sibling `SimplifiedLayerNormalization` has
the correct guard pattern at lines 152-154. Blocks all training workloads.

- [x] T5.1 Add nil guard at top of RMSNorm.Backward  Owner: Claude  Est: 15m  Done: 2026-03-17  Commit: f956329
  - File: `layers/normalization/rmsnorm.go`, insert after line 198 (the
    `len(inputs) != 1` check, before `input := r.inputTensor`):
    ```
    if r.rms == nil || r.inputTensor == nil {
        return nil, fmt.Errorf("RMSNorm: backward called before forward: missing cached tensors")
    }
    ```
  - Reference pattern: `simplified_layer_normalization.go` lines 152-154.
  - This guard covers all 4 nil dereference sites (lines 203, 240, 245, 250)
    and the inputTensor dereference (line 200).
  - Acceptance: `go build ./layers/normalization/` passes with zero errors.

- [x] T5.2 Add regression tests for Backward-before-Forward  Owner: Claude  Est: 30m  Done: 2026-03-17  Commit: 7ea8be3
  - Deps: T5.1
  - Table-driven test in `layers/normalization/rmsnorm_test.go` (or existing
    test file for RMSNorm):
    * Case 1: construct RMSNorm, call Backward without calling Forward.
      Assert error returned (not panic), error message contains "backward called
      before forward".
    * Case 2: construct RMSNorm, call Forward, then call Backward.
      Assert no error, gradients are non-nil, shapes match input.
    * Case 3: call Backward twice (second call without re-running Forward).
      Assert second Backward returns error (not panic) -- r.rms is cleared or
      reused, verify behavior matches intent.
  - Acceptance: `go test -run TestRMSNormBackward -v ./layers/normalization/`
    passes with all cases.

- [x] T5.3 Run `go build ./...` and `go vet ./...`  Owner: Claude  Est: 10m  Done: 2026-03-17
  - Deps: T5.1
  - Run from the zerfoo/ repo root.
  - Acceptance: zero build errors, zero vet warnings beyond pre-existing baseline.

- [x] T5.4 Run full normalization test suite with race detector  Owner: Claude  Est: 10m  Done: 2026-03-17
  - Deps: T5.1, T5.2
  - Command: `go test ./layers/normalization/ -race -count=1 -timeout 120s`
  - Acceptance: all tests pass, no race conditions detected, no regressions
    relative to pre-fix baseline.

### E4: Apply llama.cpp Optimizations (Stretch)

- [ ] T4.1 Apply GEMV optimization ideas from T2.2  Owner: TBD  Est: 90m
  - Implement dp4a with Q8_1 pre-quantized input for the Q4 GEMV kernel.
  - Go/purego only -- no CGo, no C++ files.
  - Acceptance: measurable throughput improvement beyond 245 tok/s on DGX.

- [ ] T4.2 Apply memory management ideas from T2.4  Owner: TBD  Est: 60m
  - Implement buffer reuse or arena improvements from llama.cpp tensor lifetime
    analysis.
  - Go/purego only -- no CGo, no C++ files.
  - Acceptance: no throughput regression vs 245 tok/s; reduced GPU memory
    footprint observable via nvidia-smi.

- [ ] T4.3 Final benchmark after optimizations  Owner: TBD  Est: 15m
  - Deps: T4.1, T4.2
  - Full benchmark suite on DGX at 50, 256, and 512 tokens.
  - Command: `LD_LIBRARY_PATH=. ./bench_tps -device cuda -model ~/models/gemma3-q4km -tokens 256 -prompt Hi`
  - Acceptance: results documented in docs/benchmarks.md.

---

## 3. Parallel Work

**Wave 1 (all independent):** T3.4, T5.1, T4.1, T4.2
**Wave 2 (after prerequisites):** T5.2 (after T5.1), T5.3 (after T5.1), T4.3 (after T4.1 + T4.2)
**Wave 3:** T5.4 (after T5.1 + T5.2)

T4.1 and T4.2 require DGX access for validation. T3.4, T5.1-T5.4 are fully
local and can be completed without GPU access.

---

## 4. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | RMSNorm guard breaks existing Forward-then-Backward callers | Low | High | Case 2 in T5.2 tests the happy path end-to-end |
| R2 | Case 3 (double Backward) behavior undefined -- guard may mask legitimate reuse | Medium | Low | Verify intent with existing call sites before merging |
| R3 | llama.cpp optimizations require CGo or C++ code | Medium | Low | Apply concepts in Go/purego only. Do not introduce CGo. |

---

## 5. Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the zerfoo repo.
2. Tests pass: `go test ./... -race -timeout 120s`.
3. No panics on Backward-before-Forward (verified by T5.2 regression test).
4. For stretch tasks: DGX throughput verified at 50, 256, 512 tokens.

### Quality Gates

- Zero build errors, zero vet warnings.
- New code must have paired tests (T5.1 paired with T5.2).
- Run `go vet ./...` after every code change before committing.
- Commit each task as its own commit (one logical change per commit).
- Do not commit files from different directories in the same commit.

---

## 6. Progress Log

### 2026-03-17: Incorporated devlog RMSNorm findings into plan

**Change summary:** Updated E5 tasks with full context from devlog entry
"2026-03-17: RMSNorm backward pass nil pointer dereference (confirmed bug)".
Added all 4 nil dereference line numbers (203, 240, 245, 250) and inputTensor
site (200). Expanded T5.2 to three test cases (before-forward, happy path,
double-backward). Added R1 and R2 to risk register. Renamed plan title to
reflect dual focus. No ADRs created (bug fix, not architectural decision).

### 2026-03-17: Trimmed plan after Phase 27 regression fix

**Change summary:** Removed completed epics E1, E1c, E1d, E2, E3 (T3.1-T3.3).
Stable knowledge preserved in docs/devlog.md (investigation entries already present).
No new ADRs needed (bug fix, not architectural decision). No design.md updates
needed (no new architectural knowledge).

Remaining work: T3.4 (update CLAUDE.md) and E4 (llama.cpp optimizations).

---

## 7. Hand-off Notes

- **DGX**: `ssh ndungu@192.168.86.250`, `LD_LIBRARY_PATH=~/Code/zerfoo`
- **RMSNorm bug**: `layers/normalization/rmsnorm.go` -- nil dereferences at
  lines 200, 203, 240, 245, 250. Fix: nil guard after line 198.
- **Reference pattern**: `layers/normalization/simplified_layer_normalization.go`
  lines 152-154 (identical guard for SimplifiedLayerNormalization).
- **Current throughput**: 245 tok/s at 256t with CUDA graphs (commit 8717a12)
- **Ollama baseline**: 204 tok/s (gemma3:1b)
- **Benchmark command**: `LD_LIBRARY_PATH=. ./bench_tps -device cuda -model ~/models/gemma3-q4km -tokens 256 -prompt Hi`
- **llama.cpp study findings**: docs/devlog.md entry "Phase 27 Wave 1"
