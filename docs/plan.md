# Zerfoo Development Plan -- Surpass Ollama via CUDA Graph Capture (Phase 6)

## 1. Context

### Problem Statement

Zerfoo F32 inference achieves ~190 tok/s on DGX Spark GB10 (96.4% of Ollama's
197.21 tok/s). Phase 5 proved the remaining 3.6% gap is entirely in CUDA kernel
launch overhead -- not Go runtime. The decode loop issues ~338 kernel launches
per token at ~5us each = ~1.7ms of the 5.26ms/token budget. CUDA graph replay
would eliminate this overhead entirely.

CUDA graph infrastructure (capture/instantiate/replay, arena reset floor,
captured slot restore) was built in Phase 4. It cannot be used because
GroupedQueryAttention (GQA) reads three CPU-side values per token:

1. `cache.SeqLen()` -- host cursor position (kvcache.go line 153).
2. RoPE angle offset -- derived from cache.SeqLen() (grouped_query_attention.go line 395).
3. KV cache append offset -- `lb.cursor * dim` (kvcache.go line 138).

These values are baked into the captured graph as kernel arguments. On replay,
stale values produce wrong output. GQA appears at instruction 2 in every
transformer layer, so no contiguous capturable region exists.

See docs/design.md for full architecture, Phase 4/5 completion details.
Decision rationale: docs/adr/032-gpu-resident-position-counter.md.

### Objectives

- O1: Store decode position counter as GPU-resident scalar.
- O2: Make RoPE angle selection use GPU counter instead of CPU offset.
- O3: Make KV cache append use GPU counter for offset computation.
- O4: Enable full CUDA graph capture for the decode loop.
- O5: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10 with Gemma 3 1B Q4_K_M.

### Non-Goals

- Prefill (variable seqLen) CUDA graph capture.
- Q4K GEMV kernel changes (memory-bound, optimization regressed in Phase 4).
- Go-side optimizations (exhausted in Phase 5).
- FP8 quality on sm_121.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M).
- Zerfoo current F32: ~190 tok/s.
- CUDA graph infrastructure already built (graph/cuda_graph.go).
- Go profile: go test, go vet, go build as quality gates.
- CUDA kernels compiled with nvcc -arch=sm_121.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| GPU position counter | Counter incremented by GPU kernel per token | Unit test: counter value matches token count |
| RoPE GPU-only | Zero CPU-side SeqLen reads during decode | Grep for WARNING during bench_tps |
| KV cache GPU-only | Append offset from GPU counter | Zero D2H in KV path during decode |
| CUDA graph active | Decode loop fully captured and replayed | bench_tps shows "graph executor", no "fallback" |
| Surpass Ollama | >197.21 tok/s | bench_tps 3-run avg on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D70 | GPU-resident position counter | Eliminates CPU-side SeqLen dependency |
| D71 | GPU-driven RoPE angle selection | Eliminates CPU RoPE offset computation |
| D72 | GPU-driven KV cache append | Eliminates CPU cursor arithmetic |
| D73 | Full CUDA graph decode | Eliminates ~1.7ms/token launch overhead |

### Out of Scope

- Prefill graph capture (variable seqLen).
- Q4K GEMV kernel changes.
- Go runtime optimizations (exhausted).
- FP8 quality.
- Multi-GPU.

---

## 3. Checkable Work Breakdown

### E801: GPU Position Counter CUDA Kernel

A trivial CUDA kernel that atomically increments a GPU-resident int32.
This replaces the host-side `lb.cursor` increment in kvcache.go.

- [ ] T801.1 Add increment_counter CUDA kernel  Owner: TBD  Est: 45m
  - Add `__global__ void increment_counter(int* counter, int delta)` to a new
    kernel file internal/cuda/kernels/counter.cu or to an existing file.
  - Add Go wrapper via purego RegisterLibFunc.
  - Add to KernelRunner interface in internal/gpuapi/.
  - File: internal/cuda/kernels/, internal/gpuapi/, internal/cuda/.
  - Acceptance: Kernel compiles. Go wrapper callable. Unit test passes.
  - Dependencies: none.

- [ ] S801.1.1 Test GPU counter kernel  Owner: TBD  Est: 30m
  - Allocate GPU int, call increment_counter 100 times, D2H copy, verify == 100.
  - File: internal/cuda/ test file.
  - Acceptance: Test passes on DGX.
  - Dependencies: T801.1.

- [ ] T801.2 Add offset_memcpy CUDA kernel  Owner: TBD  Est: 45m
  - Add kernel: `offset_memcpy(dst, src, counter, dim, maxSeqLen)` that reads
    the counter to compute `dstOff = counter * dim` and copies `dim` floats
    from src to dst+dstOff. This replaces the CPU-computed KV append offset.
  - File: internal/cuda/kernels/.
  - Acceptance: Kernel compiles. Go wrapper callable.
  - Dependencies: none.

- [ ] S801.2.1 Test offset_memcpy kernel  Owner: TBD  Est: 30m
  - Pre-set counter to 5, call offset_memcpy, verify data at offset 5*dim.
  - Acceptance: Test passes on DGX.
  - Dependencies: T801.2.

- [ ] T801.3 Add GPU-indexed RoPE selection kernel  Owner: TBD  Est: 1h
  - Add kernel: `rope_select(cos_table, sin_table, cos_out, sin_out, counter,
    halfRotary)` that reads counter to compute offset and copies the correct
    cos/sin slice from the precomputed table.
  - The RoPE table is already GPU-resident (RotaryPositionalEmbedding stores
    cos/sin as GPU tensors). This kernel replaces the CPU `GetAngles` offset.
  - File: internal/cuda/kernels/.
  - Acceptance: Kernel compiles. Go wrapper callable.
  - Dependencies: none.

- [ ] S801.3.1 Test RoPE selection kernel  Owner: TBD  Est: 30m
  - Pre-compute RoPE table, set counter to 7, call rope_select, verify output
    matches table[7*halfRotary : 8*halfRotary].
  - Acceptance: Test passes on DGX.
  - Dependencies: T801.3.

- [ ] T801.4 Run go vet and make shared  Owner: TBD  Est: 15m
  - go vet ./internal/cuda/... ./internal/gpuapi/...
  - make shared CUDA_ARCH=sm_121 in internal/cuda/kernels/.
  - Acceptance: No new warnings. Build succeeds.
  - Dependencies: T801.1, T801.2, T801.3.

### E802: Wire GPU Counter into KV Cache

Replace the host-resident `lb.cursor` in kvcache.go with a GPU-resident counter.
The GPU counter is incremented by the increment_counter kernel instead of CPU `+=`.

- [ ] T802.1 Add GPU counter to KVCache struct  Owner: TBD  Est: 1h
  - In generate/kvcache.go (or wherever KVCache is defined):
    - Add `gpuCounter unsafe.Pointer` field (GPU-allocated int32).
    - Allocate at KVCache init, free at Close.
    - Initialize to 0.
    - Add method `GPUCounterPtr() unsafe.Pointer` for use by kernels.
  - File: generate/kvcache.go (or kv_cache.go).
  - Acceptance: GPU counter allocated and accessible. CPU SeqLen still works.
  - Dependencies: T801.1.

- [ ] T802.2 Use offset_memcpy for KV append  Owner: TBD  Est: 1.5h
  - In KVCache.Update(): replace CPU offset computation + cudaMemcpy with
    the offset_memcpy kernel that reads the GPU counter.
  - After the offset_memcpy, call increment_counter to advance the position.
  - Keep CPU cursor in sync by D2H copying the counter AFTER generation
    (not per token) for debugging/logging.
  - File: generate/kvcache.go, compute/gpu_engine.go.
  - Acceptance: KV cache append uses GPU counter. No CPU cursor read during decode.
  - Dependencies: T802.1, T801.2.

- [ ] S802.2.1 Test KV cache GPU append  Owner: TBD  Est: 30m
  - Generate 10 tokens, verify KV cache content matches CPU reference.
  - File: generate/ test file.
  - Acceptance: KV cache data identical to CPU path.
  - Dependencies: T802.2.

- [ ] T802.3 Run go vet on generate package  Owner: TBD  Est: 15m
  - go vet ./generate/...
  - Acceptance: No new warnings.
  - Dependencies: T802.2.

### E803: Wire GPU Counter into RoPE

Replace CPU-side `posOffset` in GQA with GPU-driven RoPE angle selection.

- [ ] T803.1 Add GPU RoPE selection to RotaryPositionalEmbedding  Owner: TBD  Est: 1.5h
  - Add method `GetAnglesGPU(counterPtr unsafe.Pointer, seqLen int)` to
    RotaryPositionalEmbedding that calls the rope_select kernel instead
    of CPU offset arithmetic.
  - The cos/sin table GPU pointers are already stored in the struct.
  - Output: GPU tensor with correct cos/sin values for the current position.
  - File: layers/embeddings/rotary_positional_embedding.go.
  - Acceptance: GetAnglesGPU returns correct angles indexed by GPU counter.
  - Dependencies: T801.3, T802.1.

- [ ] T803.2 Update GQA to use GPU RoPE selection  Owner: TBD  Est: 1.5h
  - In layers/attention/grouped_query_attention.go:
    - At line ~393: replace `posOffset = cache.SeqLen()` with
      `counterPtr = cache.GPUCounterPtr()`.
    - At line ~395: replace `gqa.rope.GetAngles(posOffset, 1)` with
      `gqa.rope.GetAnglesGPU(counterPtr, 1)`.
    - Remove all CPU-side posOffset computation from the decode path.
  - File: layers/attention/grouped_query_attention.go.
  - Acceptance: Zero CPU SeqLen reads during decode. Output identical.
  - Dependencies: T803.1.

- [ ] S803.2.1 Test GQA GPU RoPE correctness  Owner: TBD  Est: 30m
  - Run bench_tps on DGX, verify output matches non-GPU-RoPE output exactly.
  - File: docs/updates.md.
  - Acceptance: Identical output at temp=0.
  - Dependencies: T803.2.

- [ ] T803.3 Run go vet on layers packages  Owner: TBD  Est: 15m
  - go vet ./layers/...
  - Acceptance: No new warnings.
  - Dependencies: T803.2.

### E804: Enable Full CUDA Graph Capture

With all position-dependent operations using GPU counters, the decode loop
should be fully capturable. Use the existing CUDA graph infrastructure.

- [ ] T804.1 Remove GQA from non-capturable list  Owner: TBD  Est: 45m
  - In the CUDA graph executor (graph/cuda_graph.go or generate/generator.go):
    - Remove GQA from the non-capturable op list.
    - Ensure the graph capture includes all ops including GQA.
  - Verify capture succeeds (no D2H copies trigger capture failure).
  - File: graph/cuda_graph.go, generate/generator.go.
  - Acceptance: CUDA graph capture succeeds without fallback.
  - Dependencies: T802.2, T803.2.

- [ ] S804.1.1 Test CUDA graph correctness on DGX  Owner: TBD  Est: 30m
  - Run bench_tps with graph capture on DGX.
  - Compare output with non-graph run at temp=0.
  - File: docs/updates.md.
  - Acceptance: Identical output. No "fallback" message in logs.
  - Dependencies: T804.1.

- [ ] T804.2 Benchmark with CUDA graph on DGX  Owner: TBD  Est: 30m
  - Run bench_tps 3 times with graph capture on DGX.
  - Record commit hash and results.
  - Compare with ~190 tok/s baseline and Ollama 197.21 tok/s.
  - File: docs/updates.md.
  - Acceptance: Results documented. Speedup quantified. Target: >197.21 tok/s.
  - Dependencies: T804.1.
  - PREFLIGHT: git pull on DGX, rebuild kernels (make clean && make shared).

- [ ] S804.2.1 Output quality verification  Owner: TBD  Est: 15m
  - Verify F32 output at temp=0, 256 tokens matches baseline.
  - Acceptance: Identical tokens.
  - Dependencies: T804.2.

- [ ] T804.3 Run go vet on all packages  Owner: TBD  Est: 15m
  - go vet ./...
  - Acceptance: No new warnings beyond pre-existing purego patterns.
  - Dependencies: T804.2.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Counter Kernel | T801.1, S801.1.1 | New CUDA kernel |
| Track B: Offset Memcpy | T801.2, S801.2.1 | New CUDA kernel |
| Track C: RoPE Select | T801.3, S801.3.1 | New CUDA kernel |
| Track D: KV Wire | E802 (T802.1-T802.3) | Depends on T801.1, T801.2 |
| Track E: RoPE Wire | E803 (T803.1-T803.3) | Depends on T801.3, T802.1 |
| Track F: Graph Capture | E804 (T804.1-T804.3) | Depends on D, E |

### Maximum parallelism

- Wave 1 (5 tasks): T801.1 (counter kernel) + T801.2 (offset memcpy kernel) +
  T801.3 (RoPE select kernel) + S801.1.1 (test counter) + S801.2.1 (test offset).
  T801.1/T801.2/T801.3 are independent kernel implementations.
  S801.1.1 depends on T801.1 but can run in same worktree.
  Each agent implements + tests their kernel.

- Wave 2 (5 tasks): S801.3.1 (test RoPE) + T801.4 (go vet + make) +
  T802.1 (GPU counter in KVCache) + T803.1 (GPU RoPE in embedding) +
  T802.2 (KV offset wire).
  T802.1 and T803.1 are independent (different packages).
  T802.2 depends on T802.1 + T801.2 but can start after Wave 1.

- Wave 3 (5 tasks): S802.2.1 (test KV) + T803.2 (GQA RoPE wire) +
  T802.3 (go vet generate) + T803.3 (go vet layers) + S803.2.1 (test GQA).
  T803.2 depends on T803.1. Others depend on Wave 2.

- Wave 4 (5 tasks): T804.1 (enable graph capture) + S804.1.1 (test graph) +
  T804.2 (benchmark) + S804.2.1 (quality) + T804.3 (final vet).
  T804.1 depends on T802.2 + T803.2.

### Dependency minimization checklist applied

a) All 3 kernel tasks (T801.1, T801.2, T801.3) are fully independent.
b) KV wire (E802) and RoPE wire (E803) touch different packages and can
   run in parallel after their kernel dependencies are met.
c) Wave 1 saturates 5 slots with zero-dependency tasks.
d) T804.1 is the only task requiring all tracks to converge.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M110: CUDA kernels ready | T801.4 | All 3 kernels compile and pass unit tests |
| M111: GPU counter wired | T802.2, T803.2 | KV cache and RoPE use GPU counter, no CPU SeqLen |
| M112: CUDA graph active | T804.1 | Decode loop captured, no fallback |
| M113: Surpass Ollama | T804.2 | bench_tps > 197.21 tok/s |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R801 | GPU counter adds kernel launch overhead | Net zero improvement | Low | Counter increment is 1 thread, <1us. Net savings from graph = ~1.7ms. |
| R802 | CUDA graph capture fails on other ops | Partial capture only | Medium | Test incrementally: capture KV+RoPE first, add remaining ops. |
| R803 | Graph replay produces wrong output | Regression | Medium | Bit-exact comparison with non-graph baseline at temp=0. |
| R804 | KV cache GPU offset race condition | Data corruption | Low | Counter increment is atomic. Each decode step is sequential. |
| R805 | Combined speedup still under 197 tok/s | Cannot surpass Ollama | Low | 1.7ms launch overhead = 32% of token time. Even 50% elimination = 15% speedup. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without build tags.
3. go test for the modified package passes with -race (Go code changes).
4. make shared builds without errors (CUDA kernel changes).
5. Commit passes pre-commit hooks.
6. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits. Do not let changes pile up.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- CUDA: make shared in internal/cuda/kernels/ (when .cu files change).
- Benchmark: bench_tps on DGX Spark for performance-related changes.

### Remote Host Protocol

- Always rebuild libkernels.so on DGX before benchmarking (make clean && make shared).
- Always git pull on DGX before benchmarking.
- Record exact commit hash in benchmark results.

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 6 Plan Created)

Created Phase 6 plan targeting >197.21 tok/s via CUDA graph capture.
Phase 5 (19 tasks) complete -- proved Go-side optimizations exhausted.
Phase 5 knowledge trimmed to docs/design.md.

Phase 6 focuses on 4 epics:
- E801: GPU position counter CUDA kernels (3 kernels + 3 tests + 1 build).
- E802: Wire GPU counter into KV cache (3 tasks + 1 test).
- E803: Wire GPU counter into RoPE (3 tasks + 1 test).
- E804: Enable full CUDA graph capture (3 tasks + 2 tests).

Total: 13 implementation tasks, 7 test subtasks = 20 tasks.
Designed for 4 waves with up to 5 parallel agents per wave.

Created ADR: docs/adr/032-gpu-resident-position-counter.md

---

## 9. Hand-off Notes

- **Prior plans:** Phase 1 (89 tasks), Phase 2 (35 tasks), Phase 3 (26 tasks),
  Phase 4 (30 tasks), Phase 5 (19 tasks) complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build kernels:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model ~/models/gemma3-gguf/model.gguf
  --tokens 256 --prompt 'The quick brown fox' --device cuda --dtype fp32
- **CUDA graph infrastructure (Phase 4):**
  - graph/cuda_graph.go -- CUDAGraphExecutor with capture/replay
  - graph/compile.go -- ScratchSlot, SetScratchSlot, EnsureSlotsGPU
  - internal/cuda/arena.go -- SetResetFloor for captured buffers
  - compute/gpu_engine.go -- ArenaUsedBytes, SetArenaResetFloor
- **GQA position-dependent code (targets for this phase):**
  - layers/attention/grouped_query_attention.go -- lines 393, 395 (SeqLen + GetAngles)
  - generate/kvcache.go -- line 138 (lb.cursor offset), line 153 (SeqLen method)
  - layers/embeddings/rotary_positional_embedding.go -- GetAngles method
- **Pre-commit hook:** Rejects multi-directory commits.

---

## 10. Appendix

### Kernel Launch Overhead Analysis

| Metric | Value | Source |
|--------|-------|--------|
| Kernel launches per token | ~338 | 13/layer x 26 layers + overhead |
| Launch overhead per kernel | ~5 us | CUDA runtime on sm_121 |
| Total launch overhead/token | ~1.7 ms | 338 x 5us |
| Token time at 190 tok/s | 5.26 ms | 1/190 |
| Launch overhead % of token | 32% | 1.7/5.26 |
| Expected speedup from graph | 15-32% | Eliminating 50-100% of launch overhead |
| Expected tok/s with graph | 219-250 | 190 / (1 - 0.15 to 0.32) |
| Ollama target | 197.21 | Measured 2026-03-12 |

### GPU Counter Design

```
GPU memory: [int32 counter] -- initialized to 0

Per decode token:
1. rope_select(cos_table, sin_table, cos_out, sin_out, &counter, halfRotary)
   -- reads counter, copies cos[counter*halfRotary : (counter+1)*halfRotary]
2. offset_memcpy(kv_dst, k_src, &counter, dim, maxSeqLen)
   -- reads counter, copies k to kv_dst[counter*dim]
3. offset_memcpy(kv_dst, v_src, &counter, dim, maxSeqLen)
   -- same for V cache
4. increment_counter(&counter, 1)
   -- counter++ for next token

All 4 operations are GPU kernels -- no CPU involvement.
CUDA graph captures all of them as part of the decode graph.
On replay, counter is live GPU memory, so each replay reads the current value.
```
