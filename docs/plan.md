# Zerfoo Development Plan -- Close Final 3% Gap to Ollama (Phase 5)

## 1. Context

### Problem Statement

Zerfoo F32 inference on DGX Spark GB10 achieves 191.28 tok/s (97.0% of Ollama's
197.21 tok/s). Phase 4 closed the gap from 25% to 3% via FP16/FP8 dispatch
elimination, D2H copy removal, token tensor reuse, and async memcpy. The
remaining 5.93 tok/s gap (3%) requires Go runtime and compiler optimizations
rather than GPU kernel changes.

Phase 4 key findings:
- Q4K GEMV kernel is memory-bound with 43 registers/thread. Vectorization and
  tiling increased register pressure (43->54) causing 12.2% regression. Reverted.
- CUDA graph capture blocked by GQA position-dependent RoPE/KV cache operations.
  Infrastructure built with graceful fallback.
- FP8 fixed (53.70 tok/s) but quality degenerate on sm_121 (no native cublasLt).
- The biggest win was eliminating FP16/FP8 type dispatch overhead on F32 path
  (+32 tok/s), which was not predicted by the original gap analysis.

See docs/design.md for full architecture and Phase 4 completion details.

### Objectives

- O1: Apply Profile-Guided Optimization (PGO) for compiler-level improvements.
- O2: Eliminate Go GC pauses during inference.
- O3: Eliminate bounds checks in CPU-side hot loops.
- O4: Reduce purego FFI call overhead for GPU kernel launches.
- O5: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10 with Gemma 3 1B Q4_K_M.

### Non-Goals

- Q4K GEMV kernel changes (memory-bound, optimization regressed in Phase 4).
- CUDA graph capture for GQA (requires pre-computed RoPE table, deferred).
- FP8 quality on sm_121 (no native cublasLt support).
- New model architectures or training.
- Multi-GPU / distributed inference.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M, measured 2026-03-12).
- Zerfoo current F32: 191.28 tok/s (commit f5fada5, 2026-03-13).
- Go profile: go test, go vet, go build as quality gates.
- All GPU bindings use purego (no CGo, no build tags).
- Go 1.25 with PGO support (default.pgo in main package).

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| PGO speedup | >2% improvement | bench_tps 3-run avg with vs without PGO |
| GC elimination | Zero GC pauses during decode | GODEBUG=gctrace=1 output |
| Bounds checks removed | All BCE in hot loops eliminated | go build -gcflags='-d=ssa/check_bce/debug=1' |
| Surpass Ollama | >197.21 tok/s | bench_tps 3-run avg on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D60 | PGO-optimized binary | 2-7% free throughput from compiler optimizations |
| D61 | GC-free inference path | Eliminate GC pauses during decode |
| D62 | BCE-clean hot loops | Remove bounds check overhead in CPU-side code |
| D63 | Reduced purego overhead | Fewer FFI calls per token |
| D64 | runtime.LockOSThread for CUDA | Prevent CUDA context switches |

### Out of Scope

- Q4K GEMV kernel rewrite (memory-bound, Phase 4 showed regression).
- CUDA graph GQA compatibility (significant refactor).
- FP8 quality improvements (hardware limitation).
- New CUDA kernels.

---

## 3. Checkable Work Breakdown

### E701: Profile-Guided Optimization (PGO)

PGO uses a CPU profile collected during inference to guide compiler inlining
and branch prediction. Go 1.21+ automatically uses default.pgo in the main
package directory. Expected: 2-7% throughput improvement for zero code changes.

- [x] T701.1 Collect PGO profile on DGX  Owner: task-T701.1  Est: 30m  Done: 2026-03-13
  - Build and run bench_tps on DGX with -cpuprofile flag.
  - Generate profile: go test -bench=BenchmarkGenerate -cpuprofile=cpu.pprof
    or add pprof collection to bench_tps main.
  - If bench_tps does not support -cpuprofile, add runtime/pprof Start/StopCPUProfile
    around the generation loop in cmd/bench_tps/main.go.
  - Copy the profile to cmd/bench_tps/default.pgo.
  - File: cmd/bench_tps/main.go, cmd/bench_tps/default.pgo.
  - Acceptance: default.pgo exists and is >100KB. go build uses it (prints "PGO" in verbose).
  - Dependencies: none.

- [x] T701.2 Benchmark with PGO-optimized binary on DGX  Owner: lead  Est: 30m  Done: 2026-03-13  NOTE: ~190 tok/s, no measurable PGO improvement (GPU-bound).
  - Rebuild bench_tps with default.pgo in place.
  - Run bench_tps 3 times on DGX, record results.
  - Compare with 191.28 tok/s baseline (without PGO).
  - File: docs/updates.md.
  - Acceptance: Results documented. Speedup quantified. If <1% improvement, note for analysis.
  - Dependencies: T701.1.

- [x] S701.2.1 Verify PGO output correctness  Owner: lead  Est: 15m  Done: 2026-03-13  NOTE: Identical output.
  - Run bench_tps with PGO binary at temp=0 and compare output with non-PGO.
  - Acceptance: Identical output tokens.
  - Dependencies: T701.2.

### E702: GC Elimination During Inference

Go GC can pause all goroutines during collection. For latency-sensitive inference,
disabling GC during the decode loop eliminates unpredictable pauses. The arena
allocator already handles GPU memory; CPU allocations in the hot path should be
minimal.

- [x] T702.1 Measure GC impact during inference  Owner: task-T702.1  Est: 30m  Done: 2026-03-13  NOTE: Zero GC pauses during decode. Skip E702.
  - Run bench_tps on DGX with GODEBUG=gctrace=1.
  - Count GC pauses during the generation phase (after model load).
  - Record: number of pauses, total pause time, heap size.
  - File: docs/updates.md.
  - Acceptance: GC metrics documented.
  - Dependencies: none.

- [x] T702.2 Add GOGC=off for decode loop  Owner: N/A  Est: 45m  Done: 2026-03-13  NOTE: Skipped -- zero GC pauses, no benefit.
  - In generate/generator.go: call debug.SetGCPercent(-1) before the decode
    loop starts, restore original value after generation completes.
  - Set debug.SetMemoryLimit(4 << 30) as safety net (4GB soft limit).
  - Ensure GC is re-enabled after generation so long-running servers do not leak.
  - File: generate/generator.go.
  - Acceptance: GODEBUG=gctrace=1 shows zero GC pauses during decode.
    Memory does not grow unbounded during multi-request serving.
  - Dependencies: T702.1.

- [x] S702.2.1 Test GC-free decode correctness and memory  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Skipped.
  - Run bench_tps on DGX with GC disabled.
  - Verify identical output.
  - Run 3 consecutive generations to verify memory does not grow.
  - File: docs/updates.md.
  - Acceptance: Output identical. Memory stable across runs.
  - Dependencies: T702.2.

- [x] T702.3 Benchmark with GC disabled on DGX  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Skipped.
  - Run bench_tps 3 times with GC disabled during decode.
  - Compare with baseline.
  - File: docs/updates.md.
  - Acceptance: Results documented. Speedup quantified.
  - Dependencies: T702.2.

### E703: Bounds Check Elimination (BCE)

Go inserts bounds checks on every slice access. In tight loops (sampling,
tokenization, pre/post-processing), these add overhead. The compiler can
eliminate bounds checks when it can prove the index is in range.

- [x] T703.1 Audit bounds checks in hot paths  Owner: task-T703.1  Est: 45m  Done: 2026-03-13  NOTE: 928 BCE, only 8 hot-path (<0.1%). Skip E703.
  - Build with: go build -gcflags='-d=ssa/check_bce/debug=1' ./generate/... ./compute/... ./layers/...
  - Parse output to find remaining bounds checks in hot-path files.
  - Focus on: generate/generator.go, generate/sampling.go, compute/gpu_kernels.go,
    compute/cpu_engine.go, numeric/ arithmetic loops.
  - Document each bounds check with file:line and whether it is in the hot path.
  - File: docs/updates.md.
  - Acceptance: All hot-path bounds checks catalogued.
  - Dependencies: none.

- [x] T703.2 Eliminate hot-path bounds checks  Owner: N/A  Est: 1h  Done: 2026-03-13  NOTE: Skipped -- <0.1% overhead.
  - For each hot-path bounds check from T703.1, add the appropriate hint:
    a) Assert bounds once before the loop: _ = slice[n-1]
    b) Use range loops instead of index loops where possible.
    c) Use sub-slicing to prove bounds: s := slice[:n]; for i := range s { ... }
  - Do NOT use unsafe to bypass bounds checks -- use compiler hints only.
  - File: generate/, compute/, numeric/ as needed.
  - Acceptance: go build -gcflags='-d=ssa/check_bce/debug=1' shows zero bounds
    checks in hot-path functions identified in T703.1.
  - Dependencies: T703.1.

- [x] S703.2.1 Test BCE changes correctness  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Skipped.
  - go test ./generate/... ./compute/... ./numeric/... -race -timeout 120s.
  - Acceptance: All tests pass.
  - Dependencies: T703.2.

- [x] T703.3 Benchmark with BCE eliminated on DGX  Owner: N/A  Est: 30m  Done: 2026-03-13  NOTE: Skipped.
  - Run bench_tps 3 times on DGX after BCE changes.
  - Compare with baseline.
  - File: docs/updates.md.
  - Acceptance: Results documented. Speedup quantified.
  - Dependencies: T703.2.

### E704: Purego FFI Overhead Reduction

Each CUDA kernel launch goes through purego function pointers. With ~338 kernel
launches per token, reducing per-call overhead compounds. The key optimization
is ensuring function pointers are resolved once at init time, not per call.

- [x] T704.1 Audit purego call frequency during decode  Owner: task-T704.1  Est: 30m  Done: 2026-03-13  NOTE: ~395 calls/token, ~20us total (<0.4%). Already optimal.
  - Count purego.SyscallN calls per token during decode.
  - Identify the top 10 most-called GPU functions.
  - Check if any function pointer resolution happens per-call vs per-init.
  - File: docs/updates.md.
  - Acceptance: Per-token purego call count documented. Top 10 functions listed.
  - Dependencies: none.

- [x] T704.2 Cache function pointers and reduce call count  Owner: N/A  Est: 1h  Done: 2026-03-13  NOTE: Skipped -- already cached at init.
  - Verify all RegisterLibFunc calls happen at GPUEngine init, not per-op.
  - If any kernel dispatch goes through reflection per-call, cache the result.
  - Identify adjacent kernel launches that can be batched (e.g., two back-to-back
    element-wise ops on the same data).
  - File: internal/cuda/, compute/gpu_kernels.go.
  - Acceptance: Zero per-call function pointer resolution. Call count reduced
    or documented as already optimal.
  - Dependencies: T704.1.

- [x] S704.2.1 Test purego changes correctness  Owner: N/A  Est: 15m  Done: 2026-03-13  NOTE: Skipped.
  - go test ./internal/cuda/... ./compute/... -race -timeout 120s.
  - Acceptance: All tests pass.
  - Dependencies: T704.2.

### E705: Runtime Thread Pinning

CUDA contexts are thread-local. If the Go scheduler migrates a goroutine to a
different OS thread mid-inference, CUDA context switches add latency.
runtime.LockOSThread prevents this.

- [x] T705.1 Add runtime.LockOSThread to GPU inference path  Owner: task-T705.1  Est: 30m  Done: 2026-03-13  NOTE: Caused 2.6% regression. Reverted.
  - In generate/generator.go: call runtime.LockOSThread() at the start of
    Generate/GenerateStream, defer runtime.UnlockOSThread().
  - Ensure this does not interfere with goroutine-based streaming.
  - File: generate/generator.go.
  - Acceptance: LockOSThread called before first GPU operation. UnlockOSThread
    called after generation completes.
  - Dependencies: none.

- [x] S705.1.1 Test thread pinning correctness  Owner: task-T705.1  Est: 15m  Done: 2026-03-13  NOTE: Tests passed, but perf regressed. Reverted.
  - go test ./generate/... -race -timeout 120s.
  - Acceptance: All tests pass. No deadlocks.
  - Dependencies: T705.1.

### E706: Final Benchmark and Verification

- [x] T706.1 Full benchmark with all optimizations on DGX  Owner: lead  Est: 1h  Done: 2026-03-13  Result: 189.95 tok/s (PGO only, LockOSThread reverted).
  - Build with PGO. Enable GC-off. BCE eliminated. Thread pinning.
  - Run bench_tps 3 times on DGX with Gemma 3 1B Q4_K_M.
  - Record commit hash, all results.
  - File: docs/updates.md.
  - Acceptance: Results documented. F32 > 197.21 tok/s (surpasses Ollama).
  - Dependencies: E701, E702, E703, E704, E705.

- [x] S706.1.1 Output quality verification  Owner: lead  Est: 15m  Done: 2026-03-13  NOTE: Identical output.
  - Verify F32 output at temp=0 matches baseline exactly.
  - Acceptance: Identical output tokens.
  - Dependencies: T706.1.

- [x] T706.2 Run go vet on all packages  Owner: lead  Est: 15m  Done: 2026-03-13
  - go vet ./...
  - Acceptance: No new warnings beyond pre-existing purego patterns.
  - Dependencies: T706.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: PGO | E701 (T701.1-T701.2, S701.2.1) | DGX profiling + rebuild |
| Track B: GC-free | E702 (T702.1-T702.3, S702.2.1) | Go runtime tuning |
| Track C: BCE | E703 (T703.1-T703.3, S703.2.1) | Compiler hints |
| Track D: Purego | E704 (T704.1-T704.2, S704.2.1) | FFI optimization |
| Track E: Thread Pin | E705 (T705.1, S705.1.1) | runtime.LockOSThread |
| Track F: Final Bench | E706 (T706.1-T706.2, S706.1.1) | Depends on all tracks |

Sync points:
- Wave 1: Tracks A-E all start independently (5 tasks, one per track).
- Wave 2: Track-internal dependencies (benchmarks depend on implementations).
- Wave 3: E706 final benchmark after all tracks complete.

### Maximum parallelism

- Wave 1 (5 tasks): T701.1 (PGO profile) + T702.1 (GC measurement) +
  T703.1 (BCE audit) + T704.1 (purego audit) + T705.1 (thread pinning).
  All have zero dependencies. Saturates 5 agent slots.

- Wave 2 (5 tasks): T701.2 (PGO benchmark) + T702.2 (GC disable) +
  T703.2 (BCE fixes) + T704.2 (purego cache) + S705.1.1 (test thread pin).
  Each depends only on its Wave 1 predecessor.

- Wave 3 (5 tasks): S701.2.1 (test PGO) + T702.3 (GC benchmark) +
  S703.2.1 (test BCE) + S704.2.1 (test purego) + S702.2.1 (test GC memory).
  Each depends only on its Wave 2 predecessor.

- Wave 4 (3 tasks): T703.3 (BCE benchmark) + T706.1 (full benchmark) +
  S706.1.1 (quality check).
  T706.1 depends on all tracks. T703.3 depends on T703.2.

- Wave 5 (1 task): T706.2 (final go vet).

### Dependency minimization checklist applied

a) All 5 tracks are fully independent of each other. Wave 1 saturates 5 slots.
b) Audit tasks (T701.1, T702.1, T703.1, T704.1) produce documentation only,
   not code changes, so they never conflict with each other.
c) Implementation tasks (T702.2, T703.2, T704.2, T705.1) touch different
   packages (generate/, compute/, internal/cuda/), minimizing merge conflicts.
d) T706.1 is the only task that depends on all 5 tracks completing.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M100: Audits complete | T701.1, T702.1, T703.1, T704.1 | All measurement baselines documented |
| M101: Optimizations applied | T702.2, T703.2, T704.2, T705.1 | All code changes committed and tested |
| M102: PGO validated | T701.2 | PGO speedup measured and documented |
| M103: Surpass Ollama | T706.1 | bench_tps > 197.21 tok/s with F32 |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R701 | PGO yields <1% improvement | Gap not closed by PGO alone | Medium | PGO is one of 5 optimizations. Even small gains compound. |
| R702 | GC is already negligible during decode | No speedup from GOGC=off | Medium | Measure first (T702.1). If GC pauses < 0.1ms total, skip E702. |
| R703 | Bounds checks are not in hot path | No speedup from BCE | Medium | Audit first (T703.1). If no hot-path BCE, skip E703. |
| R704 | Purego overhead is already minimal | No speedup from FFI tuning | Medium | Audit first (T704.1). If function pointers already cached, skip E704. |
| R705 | Combined optimizations still under 197 tok/s | Cannot surpass Ollama | Low | Gap is only 3%. Any 2 of the 5 optimizations yielding 1.5% each is sufficient. |
| R706 | LockOSThread causes deadlock with streaming | Inference hangs | Low | Test with GenerateStream. If blocked, restrict to Generate only. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without build tags.
3. go test for the modified package passes with -race (Go code changes).
4. Commit passes pre-commit hooks.
5. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits. Do not let changes pile up.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- Benchmark: bench_tps on DGX Spark for performance-related changes.

### Remote Host Protocol

- Always git pull on DGX before benchmarking.
- Always rebuild Go binary on DGX (kernels unchanged from Phase 4).
- Record exact commit hash in benchmark results.

---

## 8. Progress Log

### Change Summary -- 2026-03-13 (Phase 5 Plan Created)

Created Phase 5 plan targeting >197.21 tok/s. Phase 4 (30 tasks, 6 epics) is
fully complete. Phase 4 knowledge trimmed to docs/design.md.

Phase 5 focuses on 6 epics:
- E701: Profile-Guided Optimization (2 tasks + 1 test subtask).
- E702: GC elimination during inference (3 tasks + 1 test subtask).
- E703: Bounds check elimination (3 tasks + 1 test subtask).
- E704: Purego FFI overhead reduction (2 tasks + 1 test subtask).
- E705: Runtime thread pinning (1 task + 1 test subtask).
- E706: Final benchmark (2 tasks + 1 test subtask).

Total: 13 implementation tasks, 6 test subtasks = 19 tasks.
Designed for 5 waves with up to 5 parallel agents per wave.

Trimmed Phase 4 epics E601-E606 from plan. Stable knowledge preserved in
docs/design.md "Phase 4 Completion Summary" section.

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1 (89 tasks), Phase 2 (35 tasks), Phase 3 (26 tasks),
  Phase 4 (30 tasks) complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./... (kernels unchanged -- no make needed).
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf
  --tokens 256 --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Key files for PGO:**
  - cmd/bench_tps/main.go -- Add pprof collection.
  - cmd/bench_tps/default.pgo -- PGO profile (auto-detected by go build).
- **Key files for GC elimination:**
  - generate/generator.go -- Decode loop. Add debug.SetGCPercent(-1).
- **Key files for BCE:**
  - generate/sampling.go -- Token sampling hot path.
  - compute/gpu_kernels.go -- GPU kernel dispatch (getDevicePtr, gpuBinaryOp).
  - numeric/ -- Arithmetic loops.
- **Key files for purego:**
  - internal/cuda/purego.go -- RegisterLibFunc calls.
  - compute/gpu_kernels.go -- Kernel launch dispatch.
- **Pre-commit hook:** Rejects multi-directory commits.
- **Stale worktree:** wave-4-task-E103 has 34 unmerged commits from Phase 1.
  Superseded. Do not merge.

---

## 10. Appendix

### Performance Gap Analysis

| Component | Estimated Impact | Basis |
|-----------|-----------------|-------|
| PGO | 2-7% | Go compiler documentation, community benchmarks |
| GC elimination | 0-3% | Depends on current GC frequency during decode |
| Bounds check elimination | 0-2% | Depends on BCE count in hot loops |
| Purego FFI reduction | 0-1% | Already optimized in Phase 4 |
| Thread pinning | 0-1% | Prevents CUDA context switches |
| Combined estimate | 2-14% | Only need 3.1% to surpass Ollama |
| Current F32 | 191.28 tok/s | Measured 2026-03-13, commit f5fada5 |
| Target | 197.21 tok/s | Ollama baseline |
| Required improvement | 3.1% | (197.21 - 191.28) / 191.28 |
