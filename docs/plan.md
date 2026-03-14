# Zerfoo Development Plan -- Technical Debt Cleanup (Phase 8)

## 1. Context

### Problem Statement

Zerfoo v1.1.0 shipped at 234 tok/s (18.7% faster than Ollama). Phase 7 is
complete. Three technical debt items remain that affect developer experience,
CI reliability, and code quality but do not block users.

See docs/design.md for full architecture and Phase 1-7 completion summaries.
See docs/adr/033-how-we-beat-ollama.md for the performance optimization history.

### Objectives

- O1: Optimize the GQA-aware flash_attention_decode kernel to match or beat
  the cuBLAS SDPA path (currently 114 vs 234 tok/s), re-enabling the decode
  fast path for GQA models.
- O2: Fix the purego assembly trampoline segfault on Go 1.25/arm64 without -race.
- O3: Fix the 276 pre-existing golangci-lint issues so CI lint is strict.

### Non-Goals

- New features or model architectures.
- Speculative decoding.
- Multi-GPU.
- Performance beyond 234 tok/s (unless O1 succeeds, then re-benchmark).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121 (Blackwell), 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego arm64 assembly trampoline.
- CUDA kernels compiled with nvcc -arch=sm_121.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| GQA decode kernel | >= 234 tok/s (match SDPA baseline) | bench_tps on DGX |
| purego segfault | Zero segfaults without -race on arm64 | go test without -race on DGX |
| Lint issues | Zero issues with strict golangci-lint | CI lint step passes without || true |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D90 | Optimized GQA decode kernel or revert | Unblock >234 tok/s or clean up dead code |
| D91 | purego trampoline fix | Reliable tests on arm64 without -race |
| D92 | Clean lint baseline | Strict CI lint enforcement |

### Out of Scope

- New features, new models, speculative decoding, multi-GPU.
- Performance optimization beyond re-enabling the decode fast path.

---

## 3. Checkable Work Breakdown

### E1001: GQA Flash Attention Decode Kernel Optimization

The current flash_attention_decode kernel (E905) is 2x slower than cuBLAS SDPA
for GQA models (114 vs 234 tok/s). The decode fast path is disabled for GQA.

Decision: either optimize the kernel to match cuBLAS, or revert the dead code
and document the lesson. See docs/adr/034-gqa-aware-flash-attention-decode.md.

- [ ] T1001.1 Profile flash_attention_decode on DGX  Owner: TBD  Est: 1h
  - Use nsight-sys or nvprof to identify the bottleneck in the decode kernel.
  - Compare occupancy, memory throughput, and compute utilization vs cuBLAS.
  - Hypotheses: (a) shared memory bank conflicts, (b) low occupancy from
    register pressure, (c) suboptimal tiling for sm_121.
  - File: docs/updates.md.
  - Acceptance: Bottleneck identified with nsight data.
  - Dependencies: none.

- [ ] T1001.2 Optimize or revert the decode kernel  Owner: TBD  Est: 2h
  - If T1001.1 identifies a fixable bottleneck:
    - Apply the optimization (e.g., reduce shared memory, increase occupancy,
      use warp-level primitives, tune tile sizes for Blackwell).
    - Re-benchmark on DGX.
    - If >= 234 tok/s: re-enable decode fast path for GQA, remove the guard.
    - If still slower: revert to disabled state.
  - If T1001.1 shows the kernel is fundamentally limited:
    - Remove the decode fast path code and flash_attention_decode kernel.
    - Keep the GPU-resident kv_len counter (useful for future work).
    - Document the lesson in docs/adr/034.
  - File: internal/cuda/kernels/flash_attention.cu, layers/attention/.
  - Acceptance: Either kernel matches SDPA throughput or dead code removed.
  - Dependencies: T1001.1.

- [ ] S1001.2.1 Test decode kernel changes  Owner: TBD  Est: 30m
  - go test ./internal/cuda/kernels/... ./layers/attention/... -race -timeout 120s.
  - If kernel optimized: bench_tps on DGX, verify >= 234 tok/s.
  - If kernel reverted: verify 234 tok/s baseline maintained.
  - Acceptance: Tests pass. Performance at or above 234 tok/s.
  - Dependencies: T1001.2.

- [ ] T1001.3 Run go vet and make shared  Owner: TBD  Est: 15m
  - go vet ./internal/cuda/... ./layers/...
  - make shared CUDA_ARCH=sm_121 on DGX.
  - Acceptance: No new warnings. Build succeeds.
  - Dependencies: T1001.2.

### E1002: Fix purego Trampoline Segfault on arm64

All CUDA kernel tests segfault without -race on Go 1.25/arm64 (DGX Spark).
Tests pass with -race. The issue is in the purego assembly trampoline that
bridges Go to C function calls (internal/cuda/purego_linux_arm64.go/.s).

The -race flag changes Go's memory layout and stack behavior, masking the
underlying alignment or stack size issue.

- [ ] T1002.1 Diagnose the segfault root cause  Owner: TBD  Est: 1.5h
  - On DGX, run kernel tests without -race under GDB or with GOTRACEBACK=crash:
    GOTRACEBACK=crash go test ./internal/cuda/kernels/... -timeout 30s 2>&1
  - Analyze the crash: stack trace, faulting instruction, register state.
  - Read internal/cuda/purego_linux_arm64.go and purego_linux_arm64.s.
  - Hypotheses: (a) stack alignment violation (arm64 requires 16-byte aligned SP),
    (b) stack size too small for ccall trampoline, (c) signal handler conflict.
  - File: docs/updates.md.
  - Acceptance: Root cause identified with crash analysis.
  - Dependencies: none.

- [ ] T1002.2 Fix the trampoline  Owner: TBD  Est: 1.5h
  - Based on T1002.1 findings, fix the assembly trampoline or Go wrapper.
  - Common fixes: ensure SP alignment before ccall, increase goroutine stack,
    use runtime.LockOSThread in the trampoline entry, fix signal mask.
  - File: internal/cuda/purego_linux_arm64.go, internal/cuda/purego_linux_arm64.s.
  - Acceptance: go test ./internal/cuda/kernels/... passes without -race on DGX.
  - Dependencies: T1002.1.

- [ ] S1002.2.1 Verify trampoline fix on DGX  Owner: TBD  Est: 30m
  - Run full kernel test suite without -race on DGX.
  - Run with -race to verify no regression.
  - Acceptance: All tests pass both with and without -race.
  - Dependencies: T1002.2.

### E1003: Fix 276 Pre-existing golangci-lint Issues

CI currently runs golangci-lint with || true to avoid blocking on 276
pre-existing issues (164 errcheck, 27 dupl, 26 gocritic, etc.).

- [ ] T1003.1 Categorize and triage lint issues  Owner: TBD  Est: 45m
  - Run golangci-lint run locally, capture full output.
  - Group by linter and severity.
  - Decide per-category: fix, suppress with nolint, or disable the linter.
  - File: docs/updates.md.
  - Acceptance: Triage documented with action per category.
  - Dependencies: none.

- [ ] T1003.2 Fix errcheck issues (164 issues)  Owner: TBD  Est: 2h
  - The largest category. Most are unchecked error returns from Close(),
    Write(), or similar functions.
  - Fix by either handling the error or adding _ = assignment with comment.
  - Split into sub-PRs by package to keep commits focused.
  - File: across all packages.
  - Acceptance: golangci-lint errcheck reports zero issues.
  - Dependencies: T1003.1.

- [ ] T1003.3 Fix remaining lint issues  Owner: TBD  Est: 1.5h
  - Fix or suppress: dupl (27), gocritic (26), noctx (12), errorlint (9),
    nolintlint (9), gosec (7), staticcheck (7), unused (7), govet (2),
    ineffassign (2), misspell (1), errname (3).
  - For false positives: add targeted nolint comments with justification.
  - For valid issues: fix them.
  - File: across all packages.
  - Acceptance: golangci-lint reports zero issues.
  - Dependencies: T1003.1.

- [ ] T1003.4 Remove || true from CI lint step  Owner: TBD  Est: 15m
  - In .github/workflows/ci.yml, change:
    golangci-lint run --new-from-rev=origin/main || true
    to:
    golangci-lint run
  - This makes lint failures block PRs.
  - File: .github/workflows/ci.yml.
  - Acceptance: CI lint step runs without || true and passes.
  - Dependencies: T1003.2, T1003.3.

- [ ] S1003.4.1 Verify CI passes with strict lint  Owner: TBD  Est: 15m
  - Push a test commit and verify CI passes.
  - Acceptance: CI green with strict lint.
  - Dependencies: T1003.4.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Decode Kernel | E1001 (T1001.1-T1001.3) | Profile + optimize/revert |
| Track B: Trampoline | E1002 (T1002.1-S1002.2.1) | DGX debugging |
| Track C: Lint | E1003 (T1003.1-S1003.4.1) | Bulk code fixes |

### Maximum parallelism

- Wave 1 (3 tasks): T1001.1 (profile kernel) + T1002.1 (diagnose segfault) +
  T1003.1 (triage lint). All independent. All can start immediately.

- Wave 2 (3 tasks): T1001.2 (optimize/revert kernel) + T1002.2 (fix trampoline) +
  T1003.2 (fix errcheck). Each depends on its Wave 1 task.

- Wave 3 (4 tasks): S1001.2.1 (test kernel) + S1002.2.1 (test trampoline) +
  T1003.3 (fix remaining lint) + T1001.3 (go vet).

- Wave 4 (2 tasks): T1003.4 (strict CI lint) + S1003.4.1 (verify CI).

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M130: Kernel decision | T1001.2 | Decode kernel optimized or reverted |
| M131: Trampoline fixed | S1002.2.1 | Tests pass without -race on DGX |
| M132: Lint clean | S1003.4.1 | CI lint strict and passing |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1001 | Decode kernel cannot be optimized to match cuBLAS | Revert needed | High | Profiling first. If unfixable, clean revert with lesson documented. |
| R1002 | Trampoline segfault is a Go runtime bug, not fixable in user code | Stuck with -race workaround | Medium | Report upstream if confirmed. Use -race in CI as permanent workaround. |
| R1003 | Fixing 276 lint issues introduces regressions | Broken code | Low | Fix in small batches with tests. Run full test suite after each batch. |

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
- Lint: golangci-lint run (after E1003 complete).

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 8 Plan Created)

Phase 7 complete. v1.1.0 released. Trimmed all completed Phase 7 tasks to
docs/design.md. Created Phase 8 plan for 3 open technical debt items:

- E1001: GQA decode kernel optimization or revert (4 tasks).
- E1002: purego trampoline segfault fix (3 tasks).
- E1003: Fix 276 golangci-lint issues (5 tasks).

Total: 12 tasks. Designed for 4 waves with 3-4 parallel agents per wave.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 234 tok/s F32 with CUDA graph (beats Ollama 197.21 by 18.7%).
- **Prior plans:** Phase 1-7 complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf
  --tokens 256 --prompt 'The quick brown fox' --device cuda --dtype fp32
- **GQA decode fast path:** Disabled at line 624 of grouped_query_attention.go.
  Guard: seqLen == 1 && gqa.numQueryHeads == gqa.numKeyValueHeads.
- **purego trampoline:** internal/cuda/purego_linux_arm64.go/.s.
  Segfaults without -race on Go 1.25/arm64. Works with -race.
- **Lint status:** 276 issues. CI uses || true. Categories: errcheck (164),
  dupl (27), gocritic (26), noctx (12), errorlint (9), others.
- **Pre-commit hook:** Rejects multi-directory commits.
