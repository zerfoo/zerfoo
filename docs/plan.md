# Phase 27: Exceed Ollama via llama.cpp Optimizations

## 1. Context

See docs/design.md for full architecture. See docs/devlog.md for Phase 27
investigation history (diffs, bisect, root cause analysis).

Phase 27 restored inference throughput from 186 to 245 tok/s on Gemma 3 1B
Q4_K_M (DGX Spark GB10). Two root causes were found and fixed:
1. FlashAttentionDecode in GQA (disabled -- cuBLAS SDPA is faster).
2. Q5_K/Q6_K weights dequantized to float32 instead of re-quantized to Q4_0
   (restored Q4_0 re-quant in GGUF loader).

Current status: 245 tok/s with CUDA graphs, +20% vs Ollama (204 tok/s).

llama.cpp study (T2.1-T2.4) identified three optimization opportunities:
- dp4a Q4 GEMV kernel (2-4x compute advantage over scalar FMA).
- Warp-level flash attention decode (128 threads, Q in registers, warp shuffles).
- Tensor lifetime analysis for 30-50% GPU memory reduction.

---

## 2. Checkable Work Breakdown

### E3: Finalize Benchmark Documentation

- [ ] T3.4 Update CLAUDE.md benchmark claim  Owner: TBD  Est: 10m
  - Update Performance Benchmarks section with verified 245 tok/s number.
  - Acceptance: CLAUDE.md reflects actual measured throughput.

### E4: Apply llama.cpp Optimizations (Stretch)

- [ ] T4.1 Apply GEMV optimization ideas from T2.2  Owner: TBD  Est: 90m
  - Implement dp4a with Q8_1 pre-quantized input for Q4 GEMV kernel.
  - Acceptance: measurable throughput improvement beyond 245 tok/s.

- [ ] T4.2 Apply memory management ideas from T2.4  Owner: TBD  Est: 60m
  - Implement buffer reuse or arena improvements from llama.cpp study.
  - Acceptance: no throughput regression, reduced memory footprint.

- [ ] T4.3 Final benchmark after optimizations  Owner: TBD  Est: 15m
  - Deps: T4.1, T4.2
  - Full benchmark suite on DGX at 50, 256, 512 tokens.
  - Acceptance: documented results in docs/benchmarks.md.

---

## 3. Parallel Work

**Wave 1:** T3.4, T4.1, T4.2 (all independent)
**Wave 2:** T4.3 (after T4.1 and T4.2)

---

## 4. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R3 | llama.cpp optimizations require CGo or C++ code | Medium | Low | Apply concepts in Go/purego only. Do not introduce CGo. |

---

## 5. Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in both ztensor and zerfoo repos.
2. Tests pass: `go test ./... -race -timeout 120s`.
3. DGX throughput verified at 50, 256, 512 tokens.

---

## 6. Progress Log

### 2026-03-17: Trimmed plan after Phase 27 regression fix

**Change summary:** Removed completed epics E1, E1c, E1d, E2, E3 (T3.1-T3.3).
Stable knowledge preserved in docs/devlog.md (investigation entries already present).
No new ADRs needed (bug fix, not architectural decision). No design.md updates
needed (no new architectural knowledge).

Remaining work: T3.4 (update CLAUDE.md) and E4 (llama.cpp optimizations).

---

## 7. Hand-off Notes

- **DGX**: `ssh ndungu@192.168.86.250`, `LD_LIBRARY_PATH=~/Code/zerfoo`
- **Current throughput**: 245 tok/s at 256t with CUDA graphs (commit 8717a12)
- **Ollama baseline**: 204 tok/s (gemma3:1b)
- **Benchmark command**: `LD_LIBRARY_PATH=. ./bench_tps -device cuda -model ~/models/gemma3-q4km -tokens 256 -prompt Hi`
- **llama.cpp study findings**: docs/devlog.md entry "Phase 27 Wave 1"
