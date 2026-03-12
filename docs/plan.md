# Zerfoo Development Plan -- Phase 34 Remaining Work

## 1. Context

Cleanup work (E1-E6) complete. PR #45 merged feat/neon-softmax to main.
Track 0 complete (see design.md 15.16). Waves C1-C5 complete.

Active branch: `feat/purego-runtime-unify` (T87.3, T100.2 code landed).

---

## 2. Checkable Work Breakdown

### E1-E6: Documentation and Code Cleanup — COMPLETE

All 21 tasks done. PR #45 merged (commit 765108e). See git log for details.

### Track A: Remaining purego Cleanup

- [x] T87.3 Replace runtime.go CGo functions with dlopen calls — commit 8286656
  - CGo runtime.go removed. Purego dlopen is sole implementation.
  - Branch: feat/purego-runtime-unify.
- [x] S87.3.1 Runtime function parity test — commit 8286656
  - runtime_parity_test.go verifies signatures, constants, graceful failure.
- [ ] S88.2.1 Elementwise kernel parity test  Owner: TBD  Est: 1.5h
  - Dependencies: T87.3 (done).
- [ ] S88.3.1 Full kernel test suite  Owner: TBD  Est: 2h
  - Dependencies: S88.2.1.
- [ ] T89.2 Remove build tags from compute/ GPU files  Owner: TBD  Est: 2h
  - Dependencies: S88.3.1 (kernel parity verified).
- [ ] S89.3.1 Cross-platform build verification  Owner: TBD  Est: 1h
  - Dependencies: T89.2.

### Track C: Tracing Compiler Integration (remaining)

Waves C1-C4 complete. See design.md section 15.16 and docs/adr/028-tracing-compiler.md.

- [x] S100.1.1 Integration test on DGX Spark — 2026-03-11
  - Megakernel did NOT fire. codegen.CheckSupport rejects unsupported ops.
  - GPU throughput: F32 12.84 tok/s, Q4 8.61 tok/s (non-megakernel path).
  - Recommendation: add diagnostic logging to tryCompileMegakernel failure points.

- [x] T100.2 Update tryCompileMegakernel for GPU KV cache — commits 0b3ab3b, 6116588, 9ffa7bc
  - detectKVCacheOps scans instruction tape.
  - extractKVCacheDims gets numLayers, numHeads, headDim from tensor shapes.
  - GPUKVCache allocated when KV ops present.
  - Device pointers passed to runner.Launch() via SetKVCache().
  - seq_pos wired from kvCache.SeqLen() (was hardcoded 0).
  - Branch: feat/purego-runtime-unify.

- [ ] S100.2.1 KV cache integration test  Owner: TBD  Est: 1.5h
  - Generate 50 tokens with megakernel. Compare with plan.Run() output.
  - Dependencies: T100.2 (done), DGX Spark.

- [ ] T100.3 End-to-end megakernel correctness test  Owner: TBD  Est: 2h
  - Load Gemma 3 2B Q4, generate 50 tokens, compare megakernel vs plan.Run().
  - Dependencies: S100.2.1.

- [ ] T100.4 Run golangci-lint on generate/, graph/  Owner: TBD  Est: 15m
  - Dependencies: T100.3.

### Track B: Megakernel Performance Tuning

- [ ] T94.1 Profile megakernel with nsys  Owner: TBD  Est: 2h
  - Dependencies: T100.3 (correctness verified).
- [ ] T94.2 Optimize memory access patterns  Owner: TBD  Est: 3h
- [ ] T94.3 Tune thread block and grid dimensions  Owner: TBD  Est: 2h
- [ ] T94.4 Run golangci-lint  Owner: TBD  Est: 15m
- [ ] T95.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
- [ ] S95.1.1 GPU profile report  Owner: TBD  Est: 30m
- [ ] T95.2 Compare all configurations  Owner: TBD  Est: 1.5h
- [ ] S95.2.1 Benchmark comparison report  Owner: TBD  Est: 30m
- [ ] T95.3 Verify output correctness across all paths  Owner: TBD  Est: 1h
- [ ] S95.3.1 Output correctness report  Owner: TBD  Est: 30m
- [ ] T95.4 Run golangci-lint  Owner: TBD  Est: 15m

---

## 3. Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| GPU F32 (non-megakernel) | 12.84 | S100.1.1 DGX Spark test (2026-03-11) |
| GPU Q4 (non-megakernel) | 8.61 | S100.1.1 DGX Spark test (2026-03-11) |
| CPU ARM64 (post Track D) | 8.15 median | Phase 34 Track D |
| CPU ARM64 (pre Track D) | 6.86 | Phase 30 |
| GPU (cuda, old baseline) | 10.32 peak / 7.78 median | Phase 33 |
| Ollama GB10 | ~100 (est.) | Interpolated |

---

## 4. Key Milestones

| Milestone | Status |
|-----------|--------|
| M56: Tracing compiler works | DONE (CompileTraced produces primitive-op tape) |
| M57: GPU KV cache works | DONE (T100.2 code complete, integration pending) |
| M58: Megakernel fires | PENDING (S100.2.1, T100.3 on DGX Spark) |
| M59: 50 tok/s GPU | PENDING (Track B after M58) |
| M60: 10 tok/s CPU ARM64 | PARTIAL (8.15 tok/s, needs GEMM cache tiling) |

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R92 | Register pressure: hidden_dim=2048 | Must tile, slower | High | Profile with nvcc --ptxas-options=-v |
| R95 | KV cache reads limit bandwidth | Cannot reach max | High | Focus on short contexts (<512) |
| R98 | 50 tok/s not achieved | Unknown bottleneck | Medium | If 30+, profile; if <30, fused kernels |
| R100 | Tracing captures wrong path | Wrong megakernel output | Medium | Only use for decode (seqLen=1) |
| R101 | Tensor identity via pointer fragile | Wrong slot wiring | Medium | Disable pooling during tracing |
| R102 | GPU KV cache memory budget | OOM for long contexts | Low | Default 512 tokens (~104MB) |
| R103 | EngineProxy dispatch overhead | Small regression | Low | ~1us total, negligible |

---

## 6. Progress Log

### 2026-03-11

Wave 1 executed (3 agents, parallel worktrees):
- T87.3: CGo runtime replaced with dlopen. Purego is sole implementation. (commit 8286656)
- S87.3.1: Parity test written. (commit 8286656)
- S100.1.1: DGX Spark test. Megakernel did not fire (expected). GPU F32 12.84 tok/s.
- T100.2: GPU KV cache wired into megakernel. (commits 0b3ab3b, 6116588, 9ffa7bc)
Newly unblocked: S88.2.1, S100.2.1, T100.3.

### 2026-03-09

Plan created. E1-E6 cleanup tasks identified and subsequently completed. PR #45 merged.

---

## 7. Hand-off Notes

- **PR workflow:** PRs go to zerfoo/zerfoo (upstream), not dndungu/zerfoo.
  Use `gh pr create --repo zerfoo/zerfoo --head dndungu:<branch>`.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Pre-commit hook:** rejects multi-directory commits.
