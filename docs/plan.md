# Zerfoo Phase 23: "95% of Theoretical Ceiling"

## 1. Context

### Problem Statement

Zerfoo achieves 167 tok/s (Gemma 3 1B Q4_K_M, 50 tokens, CUDA graph) on DGX Spark
GB10. Ollama achieves 209 tok/s on the same hardware and model. The theoretical ceiling
for this hardware is ~250 tok/s (memory-bandwidth bound: 800MB weights / 200 GB/s
bandwidth = ~4ms per token). The target is 95% of ceiling = 237+ tok/s.

### Per-Step Overhead Analysis

Each decode step in the session does:

1. `PoolResetter` type assertion — `any(s.engine).(compute.PoolResetter)` allocates an
   interface value per step. Fix: cache the resetter once at session creation.
2. `PrepareSlots` — copies 185 slot pointers every step (`copy(slots, p.slots)`).
   Fix: skip copy on replay; only set the input slot.
3. `EnsureSlotsGPU` — iterates all 185 slots checking GPU residency every step.
   Fix: after first replay, all slots are GPU-resident; skip subsequent checks.
4. `capturedSlots` restore — iterates map every step to restore captured output tensors.
   Fix: store as a flat slice instead of map.
5. `stream.Synchronize()` — blocks Go goroutine until GPU finishes. This is necessary
   but costs ~50us of Go runtime overhead per call (goroutine park/wake). Ollama (C++)
   uses a raw cudaStreamSynchronize with zero Go overhead.

### Root Causes of 167 vs 209 Gap

1. **Go runtime overhead per GPU sync**: Each `stream.Synchronize()` goes through
   purego (dlopen/dlsym), which adds function call overhead vs C++ direct calls.
   Ollama's llama.cpp kernel dispatch is entirely in C++.

2. **Redundant work per step**: PrepareSlots, EnsureSlotsGPU, and capturedSlots restore
   do O(N) work every step where N=185 instructions. The actual GPU work is ~1ms;
   adding 200us of Go overhead per step = ~17% waste.

3. **No input tensor reuse**: The session creates a `[1,1]` token tensor once and
   reuses the backing slice, but `PrepareSlots` copies all 185 slots anyway. The
   replay path should only update the single input slot.

### Objectives

- O1: Reach 237+ tok/s on Gemma 3 1B Q4_K_M (256 tokens) on DGX Spark GB10.
- O2: Beat Ollama on the same hardware (>209 tok/s).
- O3: Eliminate all redundant per-step overhead in the replay path.

### Non-Goals

- Continuous batching, new model architectures, training.
- Changes to ztensor CUDA kernels (kernel performance is not the bottleneck).
- CompileTraced (architectural limitation, documented in Phase 23 T2.1-T2.2).

### Constraints

- Pure Go, zero CGo.
- Changes to ztensor (separate repo) must be committed there first.
- All existing tests must continue to pass.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Gemma 3 1B 50 tokens | 167 tok/s | 237+ | DGX bench |
| Gemma 3 1B 256 tokens | 105 tok/s | 237+ | DGX bench |
| vs Ollama | -20% | +10% | Side-by-side DGX bench |

---

## 2. Scope and Deliverables

### In Scope

- Eliminate per-step overhead in session decode loop and CUDA graph replay.
- Optimize the CUDAGraphExecutor replay hot path in ztensor.
- DGX benchmarks after each optimization.

### Out of Scope

- CUDA kernel changes, CompileTraced, new model support.
- Continuous batching, distributed inference.

---

## 3. Checkable Work Breakdown

### E1: Session Decode Loop Optimizations

Eliminate per-step overhead in the Go-side decode loop.

- [x] T1.1 Add ResetPool call between decode steps in session  Owner: Claude  Done: 2026-03-16
- [x] T1.2 Add GPU argmax fast path to session sampling  Owner: Claude  Done: 2026-03-16

- [ ] T1.3 Cache PoolResetter interface at session creation  Owner:  Est: 20m
  - File: `generate/session.go`
  - Add a `poolResetter compute.PoolResetter` field to InferenceSession. Set it in
    NewSession with a type assertion. In the decode loop, replace the per-step
    `any(s.engine).(compute.PoolResetter)` with `if s.poolResetter != nil`.
  - Acceptance: No per-step type assertion. `go test ./generate/ -race` passes.

- [ ] T1.4 Cache stopSet and pre-allocate generatedIDs in session  Owner:  Est: 15m
  - File: `generate/session.go`
  - The stopSet map is rebuilt every Generate call. Pre-allocate it once. Similarly,
    `generatedIDs := make([]int, 0, sc.MaxNewTokens)` allocates every call.
    Pre-allocate a reusable slice on the session and reset it.
  - Acceptance: Zero allocations in the decode hot loop (except GPU ops).
    `go test ./generate/ -race` passes.

### E2: CUDA Graph Replay Hot Path (ztensor)

Optimize CUDAGraphExecutor.replay() to minimize Go overhead between GPU graph launches.

- [ ] T2.1 Add fast replay path that skips PrepareSlots/EnsureSlotsGPU  Owner:  Est: 60m
  - File: `ztensor/graph/cuda_graph.go` method `replay()`
  - After the first successful replay, all slots are GPU-resident and capturedSlots
    are set. Add a `replayReady bool` flag. When true, skip:
    - `PrepareSlots` (just set the input slot directly)
    - `EnsureSlotsGPU` (all slots already GPU-resident after first replay)
    - `capturedSlots` map iteration (slots already restored)
  - Only need: set input slot, GraphLaunch, Synchronize, return OutputTensor.
  - Acceptance: replay() does O(1) Go work between graph launches.
    `go test ./graph/ -race` passes.

- [ ] T2.2 Convert capturedSlots from map to slice  Owner:  Est: 20m
  - File: `ztensor/graph/cuda_graph.go`
  - Replace `capturedSlots map[int]*tensor.TensorNumeric[T]` with a flat slice.
    Map iteration on every replay adds GC pressure and is slower than slice indexing.
  - Acceptance: No map in the replay hot path. `go test ./graph/ -race` passes.

- [ ] T2.3 Benchmark after ztensor replay optimization  Owner:  Est: 30m
  - Deps: T2.1, T2.2
  - Requires: push ztensor changes, update go.mod in zerfoo.
  - Run DGX benchmark: Gemma 3 1B Q4_K_M, 50 and 256 tokens.
  - Acceptance: Measurable improvement logged to docs/devlog.md.

### E3: Session-Level Optimizations

- [ ] T3.1 Skip EmbeddingLookup on replay when input unchanged  Owner:  Est: 45m
  - File: `ztensor/graph/cuda_graph.go`
  - EmbeddingLookup is the pre-capture instruction (instruction 0). It runs on every
    replay step, doing a CPU table lookup and H2D copy. For the decode loop, the
    embedding table doesn't change -- only the input token ID changes. Cache the
    previous embedding result and update via `cudaMemcpy` of just the token's
    embedding vector (hidden_size * 4 bytes) instead of re-running the full op.
  - Acceptance: Pre-capture region does minimal work on replay. DGX benchmark shows
    improvement.

- [ ] T3.2 Eliminate redundant context.Value lookups  Owner:  Est: 30m
  - File: `generate/session.go`, `generate/context.go`
  - `WithCache(ctx, s.cache)` and `GetCache(ctx)` do `context.WithValue` and
    `ctx.Value` on every forward call. Pass the cache directly via the session
    struct instead of through context. The graph Forward/Plan.Run functions can
    accept a CacheProvider as an explicit parameter.
  - Risk: This touches the graph/cache interface boundary. May require changes to
    ztensor's Forward signature. Evaluate whether the context overhead is significant
    before implementing.
  - Acceptance: No context.Value lookups in the decode hot path, or documented
    evidence that the overhead is negligible.

### E4: Final Benchmarks and Gate

- [ ] T4.1 Full test suite pass  Owner:  Est: 30m
  - Deps: all above
  - Acceptance: `go test ./... -race -count=1` passes in both zerfoo and ztensor.

- [ ] T4.2 DGX final benchmark vs Ollama  Owner:  Est: 45m
  - Deps: T4.1
  - Run side-by-side: Zerfoo vs Ollama, Gemma 3 1B Q4_K_M, 50 and 256 tokens.
  - Update docs/benchmarks.md.
  - Acceptance: Zerfoo >= 237 tok/s (95% of theoretical) or documented analysis of
    remaining gap with specific attribution (GPU kernel vs Go overhead vs memory).

- [ ] T4.3 Carry forward: DeepSeek V3 DGX E2E  Owner:  (BLOCKED: no MLA+MoE GGUF model)

---

## 4. Parallel Work

### Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: Session loop | T1.3, T1.4 | Go-side decode loop |
| B: ztensor replay | T2.1, T2.2 | CUDA graph replay hot path |
| C: Advanced opts | T3.1, T3.2 | Embedding cache, context elimination |

All tracks are independent.

### Maximum Parallelism

**Wave 1** (6 tasks, all independent):
T1.3, T1.4, T2.1, T2.2, T3.1, T3.2

**Wave 2** (2 tasks):
T2.3, T4.1

**Wave 3** (1 task):
T4.2

---

## 5. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | ztensor changes break zerfoo | Low | High | Run both test suites before and after. |
| R2 | Go runtime overhead is irreducible | Medium | High | Profile with pprof to identify. Worst case, the overhead is <5% and we approach but don't reach Ollama. |
| R3 | Replay optimizations break CUDA graph correctness | Medium | High | Each optimization must pass the full test suite with -race. |
| R4 | 237 tok/s is unreachable on GB10 | Medium | High | If gap remains after all optimizations, profile GPU vs CPU time to attribute. |

---

## 6. Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...`
2. Tests pass: `go test ./... -race`
3. Lint clean: `go vet ./...`
4. DGX benchmark for performance tasks.

---

## 7. Progress Log

### 2026-03-16: Phase 23 plan created

**Change summary:** Created Phase 23 plan targeting 95% of theoretical ceiling
(237+ tok/s). Trimmed completed Phase 22 tasks and earlier Phase 23 investigation tasks.
Key insight: per-step Go overhead in CUDA graph replay (PrepareSlots, EnsureSlotsGPU,
map iteration) adds ~200us per step, explaining the 167 vs 209 gap with Ollama.

Previously completed:
- T1.1 ResetPool in session decode
- T1.2 GPU argmax in session decode
- T2.1-T2.2 CompileTraced investigation (architectural limitation, fallback acceptable)
- T3.1-T3.3 Phi merged gate+up MLP split (DGX verified)

---

## 8. Hand-off Notes

- **Codebase**: zerfoo at `/Users/dndungu/Code/zerfoo/zerfoo/`, ztensor at `../ztensor/`
- **Key files**:
  - `generate/session.go` — session decode loop
  - `ztensor/graph/cuda_graph.go` — CUDAGraphExecutor.replay() hot path
  - `ztensor/graph/compile.go` — ExecutionPlan.PrepareSlots, RunInstructionRange
- **DGX**: `ssh ndungu@192.168.86.250`, `LD_LIBRARY_PATH=~/Code/zerfoo`
- **Models**: Gemma Q4_K_M at `~/models/gemma3-q4km/model.gguf`
- **Ollama**: installed at `/usr/local/bin/ollama`, model `gemma3:1b-it-q4_K_M`

---

## 9. Appendix

### Decode Step Overhead Breakdown

Per step in CUDAGraphExecutor.replay():
```
PrepareSlots:     ~10us (copy 185 pointers + set input)
Pre-capture run:  ~50us (EmbeddingLookup: CPU table lookup + H2D copy)
EnsureSlotsGPU:   ~20us (iterate 185 slots, check type)
CapturedSlots:    ~5us  (map iteration, set pointers)
GraphLaunch:      ~5us  (CUDA API call)
Synchronize:      ~1ms  (GPU compute) + ~50us (Go goroutine park/wake)
Post-capture:     ~0us  (no post-capture instructions)
OutputTensor:     ~1us
---
Total Go overhead: ~140us per step
GPU compute:       ~1000us per step
Overhead ratio:    ~12% (explains most of the gap)
```

### Optimal Replay Path (Target)

```
Set input slot:   ~1us  (single pointer write)
GraphLaunch:      ~5us
Synchronize:      ~1ms + ~50us
OutputTensor:     ~1us
---
Total Go overhead: ~57us per step
Overhead ratio:    ~5%
```
