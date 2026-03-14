# Zerfoo Development Plan -- Phase 10 Remaining Work

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 10 Wave 1 completed 5 test coverage and lint tasks. The remaining 14 tasks
span four workstreams: (A) two more test coverage tasks, (B) the cuBLAS status 7
graph execution bug fix, (C) Phase 8 technical debt (decode kernel + trampoline),
and (D) DGX verification + README.

The critical blocker is the cuBLAS status 7 error: all non-GGUF (ZMF F32) models
fail on GPU at the LM head projection. cuBLAS works in isolation with large
dimensions -- the failure is in how getDevicePtr handles H2D copies for 1GB+
weight matrices during graph forward. See docs/updates.md for the deep dive.

### Objectives

- O1: Fix the graph execution memory lifecycle bug (cuBLAS status 7).
- O2: Complete test coverage (getDevicePtr lifecycle, CLI pull integration).
- O3: Complete Phase 8 debt (decode kernel, trampoline).
- O4: Verify FP16 KV cache and GQA decode fast path on DGX.
- O5: Write README with quickstart once models are verified.

### Non-Goals

- New model architectures.
- Performance optimization beyond Phase 7 work.
- Training or fine-tuning.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- GGUF Q4K models work. ZMF F32 models fail on GPU.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| ZMF model inference | All 4 ZMF models produce coherent output on GPU | bench_tps on DGX |
| Test coverage | getDevicePtr lifecycle + CLI pull integration tests | go test ./... |
| Phase 8 debt | Decode kernel profiled/resolved, trampoline fixed | DGX profiling + tests |
| README | Users can go from clone to inference in 5 minutes | Manual walkthrough |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D200 | Fix cuBLAS status 7 in graph execution | Unblocks all non-GGUF models |
| D201 | Remaining test coverage (getDevicePtr, CLI pull integration) | Prevents regressions |
| D202 | Phase 8 decode kernel + trampoline resolution | Technical debt cleanup |
| D203 | FP16 KV + GQA fast path verification on DGX | Phase 7 completion |
| D204 | README with quickstart | First-time user experience |

### Out of Scope

- New model architectures not already in the codebase.
- Performance tuning beyond Phase 7 work.
- Multi-GPU / distributed inference.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E2000: Test Coverage Gaps (2 remaining)

- [x] T2000.5 Add getDevicePtr memory lifecycle test  Owner: task-T2000.5  Completed: 2026-03-14
  - Test that getDevicePtr for large CPUStorage tensors returns valid GPU pointers.
  - Test that sequential getDevicePtr calls (simulating graph forward) do not
    return overlapping or freed memory.
  - Test that cleanup functions properly free GPU memory.
  - File: compute/gpu_kernels_test.go.
  - Acceptance: No double-free or use-after-free detected. CUDA memcheck clean.
  - Dependencies: none.

- [x] T2000.6 Add zerfoo pull CLI integration test  Owner: task-T2000.6  Completed: 2026-03-14
  - Wire NewHFPullFunc into the CLI pull command (fix the broken pull path).
  - Test the full CLI flow: NewPullCommand -> Run -> pulls from mock HF server.
  - File: cmd/cli/pull_test.go.
  - Acceptance: `go test ./cmd/cli/... -run TestPull` passes.
  - Dependencies: none (T2000.2 completed).

### E2001: Fix Graph Execution Memory Bug

Root cause investigation from Phase 9 Wave 2: cuBLAS Sgemm works in isolation with
large dimensions, but fails during graph forward pass. The issue is in how
getDevicePtr handles H2D copies for 1GB+ weight matrices during the graph
forward pass.

- [x] T2001.1 Add debug logging to getDevicePtr for large allocations  Owner: task-T2001.1  Completed: 2026-03-14
  - Log allocation size, pointer address, and memcpy result for allocations > 100MB.
  - Log cuBLAS Sgemm arguments (m, n, k, pointers) before the call.
  - Instrument compute/gpu_kernels.go and compute/gpu_engine.go.
  - File: compute/gpu_kernels.go, compute/gpu_engine.go.
  - Acceptance: Debug output shows the exact failure point.
  - Dependencies: none.

- [ ] T2001.2 Diagnose and fix the cuBLAS status 7 root cause  Owner: TBD  Est: 2h
  - Run instrumented bench_tps on DGX with Llama 3 ZMF model.
  - Analyze debug output: verify pointer validity, memcpy return codes, buffer sizes.
  - Likely fixes: synchronize stream before cuBLAS call, check memcpy error return,
    pre-allocate weight buffers during model load instead of on-demand H2D.
  - File: compute/gpu_kernels.go or compute/gpu_engine.go.
  - Acceptance: bench_tps with Llama 3 ZMF produces output without cuBLAS error.
  - Dependencies: T2001.1.

- [ ] S2001.2.1 Test fix with all 4 models on DGX  Owner: TBD  Est: 1h
  - Re-run bench_tps for Llama 3, Qwen 2.5, Mistral 7B, Phi 4 on DGX.
  - Record tok/s and output quality for each.
  - File: docs/updates.md.
  - Acceptance: All 4 models produce coherent output at temp=0.
  - Dependencies: T2001.2.

- [ ] S2001.2.2 Test fix locally with unit tests  Owner: TBD  Est: 30m
  - go test ./compute/... ./graph/... -race -timeout 120s.
  - Verify T2000.1 (large MatMul) and T2000.5 (getDevicePtr lifecycle) pass.
  - Acceptance: All tests pass with -race.
  - Dependencies: T2001.2.

### E1001: Decode Kernel (Phase 8 retained)

- [x] T1001.1 Profile flash_attention_decode on DGX  Owner: task-T1001.1  Completed: 2026-03-14
  - Run nsys profile on bench_tps with Gemma 3 1B.
  - Measure decode kernel time vs total decode time.
  - File: docs/updates.md.
  - Acceptance: Profile data shows kernel time breakdown.
  - Dependencies: none.

- [ ] T1001.2 Optimize or revert the decode kernel  Owner: TBD  Est: 2h
  - Based on T1001.1 profiling, either optimize or revert to cuBLAS attention.
  - File: internal/cuda/kernels/flash_attention.cu.
  - Acceptance: Decode tok/s equal to or better than before.
  - Dependencies: T1001.1.

- [ ] S1001.2.1 Test decode kernel changes  Owner: TBD  Est: 30m
  - go test ./internal/cuda/kernels/... -race -timeout 120s.
  - Run bench_tps on DGX to verify no regression.
  - Acceptance: All kernel tests pass. Throughput >= 230 tok/s.
  - Dependencies: T1001.2.

- [ ] T1001.3 Run go vet and make shared  Owner: TBD  Est: 15m
  - go vet ./internal/cuda/...
  - make shared CUDA_ARCH=sm_121 on DGX.
  - Acceptance: No new warnings. Build succeeds.
  - Dependencies: T1001.2.

### E1002: Purego Trampoline (Phase 8 retained)

- [x] T1002.1 Diagnose purego trampoline segfault  Owner: task-T1002.1  Completed: 2026-03-14
  - Reproduce segfault on DGX with a minimal test case.
  - Check ccallTrampoline assembly for ARM64 AAPCS64 compliance.
  - Verify stack alignment for 14+ argument C functions.
  - File: internal/cuda/purego_linux_arm64.s.
  - Acceptance: Root cause identified and documented.
  - Dependencies: none.

- [ ] T1002.2 Fix the trampoline  Owner: TBD  Est: 1.5h
  - Apply fix based on T1002.1 diagnosis.
  - File: internal/cuda/purego_linux_arm64.s or purego_linux_arm64.go.
  - Acceptance: Segfault no longer reproduces.
  - Dependencies: T1002.1.

- [ ] S1002.2.1 Verify trampoline fix on DGX  Owner: TBD  Est: 30m
  - Run full test suite on DGX: go test ./internal/cuda/... -race.
  - Run bench_tps to verify inference still works.
  - Acceptance: No segfaults. All tests pass.
  - Dependencies: T1002.2.

### E2002: DGX Verification (Phase 7 leftover)

- [ ] S902.5.1 Test FP16 KV end-to-end on DGX  Owner: TBD  Est: 30m
  - Run bench_tps with --kv-dtype=fp16 on DGX, 20 tokens.
  - Verify output quality (coherent text at temp=0).
  - Acceptance: Output coherent. No pad tokens.
  - Dependencies: none.

- [ ] S905.3.1 Test GQA decode fast path correctness  Owner: TBD  Est: 30m
  - go test ./layers/attention/... -race -timeout 120s.
  - Run bench_tps on DGX with 20 tokens, verify output matches standard path.
  - Acceptance: All tests pass. Output coherent at temp=0.
  - Dependencies: none.

### E1103: README (blocked on model fix)

- [ ] T1103.1 Write README.md with quickstart  Owner: TBD  Est: 1.5h
  - Sections: What is Zerfoo, Installation, Quickstart (pull + run in 3 commands),
    Supported Models (table with tok/s), API Usage (curl examples),
    Performance (vs Ollama chart), Architecture Overview, Contributing.
  - Include benchmark table from DGX results.
  - File: README.md.
  - Acceptance: A new user can go from clone to inference in 5 minutes
    following the README.
  - Dependencies: S2001.2.1 (need verified model benchmarks).

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Test Coverage | T2000.5, T2000.6 | Local tests, no DGX |
| Track B: Graph Fix | T2001.1-S2001.2.2 | DGX for T2001.2+ |
| Track C: Decode Kernel | T1001.1-T1001.3 | DGX-only |
| Track D: Trampoline | T1002.1-S1002.2.1 | DGX-only |
| Track E: Verification | S902.5.1, S905.3.1 | DGX-only |

### Maximum parallelism

- Wave 2 (5 tasks): T2000.5 (getDevicePtr lifecycle test) + T2000.6 (CLI pull integration) +
  T2001.1 (debug logging) + T1001.1 (decode kernel profile, DGX) + T1002.1 (trampoline diagnosis, DGX).
  All 5 are independent.

- Wave 3 (5 tasks): T2001.2 (fix cuBLAS root cause, DGX) + T1001.2 (optimize decode kernel, DGX) +
  T1002.2 (fix trampoline, DGX) + S902.5.1 (FP16 KV test, DGX) + S905.3.1 (GQA fast path, DGX).
  T2001.2 depends on T2001.1. T1001.2 depends on T1001.1. T1002.2 depends on T1002.1.
  S902.5.1 and S905.3.1 are independent.

- Wave 4 (5 tasks): S2001.2.1 (verify 4 models, DGX) + S2001.2.2 (local unit tests) +
  S1001.2.1 (decode kernel tests) + T1001.3 (go vet + make shared) + S1002.2.1 (trampoline verify).
  All depend on Wave 3 outputs.

- Wave 5 (1 task): T1103.1 (README -- depends on model verification results from Wave 4).

### Dependency minimization checklist applied

a) T2000.5 and T2000.6 are independent of all DGX work.
b) Graph fix debug logging (T2001.1) is independent of test coverage tasks.
c) Decode kernel (E1001) and trampoline (E1002) are fully independent of each other.
d) FP16 KV and GQA tests (E2002) are independent of everything else.
e) README is the only task with a hard dependency on model fix results.
f) Wave 2 saturates all 5 slots with zero inter-dependencies.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M200: Test coverage complete | T2000.6 | getDevicePtr + CLI pull integration tests pass |
| M201: Models fixed | S2001.2.1 | All 4 ZMF models produce coherent GPU output |
| M202: Phase 8 done | S1002.2.1 | Decode kernel resolved, trampoline fixed |
| M203: Phase 7 verified | S905.3.1 | FP16 KV and GQA fast path verified on DGX |
| M204: README published | T1103.1 | 5-minute quickstart verified |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R2000 | cuBLAS status 7 root cause is deeper than memory lifecycle | Extended debugging | Medium | T2001.1 debug logging will narrow the search. Fallback: pre-allocate all weight buffers at model load time. |
| R2001 | Trampoline segfault is a Go runtime/assembly interaction | Hard to fix | Medium | CGo fallback path exists (purego_linux_arm64_cgo.go) but needs CUDA headers. |
| R2002 | FP16 KV still produces garbage | Feature unusable | Medium | FP16 KV is optional. F32 KV works at 234 tok/s. |
| R2003 | Decode kernel optimization yields no speedup | Wasted effort | Low | T1001.1 profiling first. Can revert to cuBLAS attention. |

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

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 10 Wave 1 Complete, Plan Trimmed)

Trimmed 5 completed tasks from Wave 1. Stable knowledge preserved in
docs/design.md (updated known limitation 16, test coverage section).

Completed and removed:
- E2000 tasks T2000.1-T2000.4: large MatMul GPU tests, CLI pull tests, Range op
  edge cases, graph forward tests (40+ new tests across 4 packages).
- E2002 task S1003.4.1: CI strict lint verified, 3 lint fixes applied
  (duplicate constantNode, errcheck in serve/metrics.go, dupl in serve/server_test.go).

Remaining: 14 tasks across 6 epics. Designed for 4 waves (Wave 2-5) with up to
5 parallel agents per wave. Wave 2 saturates all 5 slots.

### Change Summary -- 2026-03-14 (Phase 10 Plan Created)

Trimmed 16 completed tasks from Phase 9 plan. Created new epic E2000 (Test
Coverage Gaps) with 6 tasks. Restructured E2001 with debug instrumentation
approach. Retained Phase 8 E1001/E1002 and Phase 7 verification tasks.
Total: 24 tasks across 6 epics.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 234 tok/s F32 with CUDA graph (beats Ollama 197.21 by 18.7%).
- **Branches:** fix/errcheck-issues has all work (~42 commits ahead of main).
- **Key bug:** Non-GGUF models fail on GPU with cuBLAS status 7 at LM head.
  cuBLAS works in isolation. See docs/updates.md for deep dive.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf (works), ~/models/llama3/
  (ZMF, fails), ~/models/qwen25/ (ZMF, fails), ~/models/mistral/ (ZMF, fails)
- **Pre-commit hook:** Rejects multi-directory commits.
- **OpenAI API endpoints:** /v1/chat/completions, /v1/completions, /v1/models, /metrics
