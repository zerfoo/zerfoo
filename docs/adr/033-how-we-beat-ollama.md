# ADR 033: How Zerfoo Surpassed Ollama — 241 tok/s on DGX Spark GB10

## Status
Accepted

## Date
2026-03-14

## Context
On 2026-03-11, Zerfoo achieved 0.44-12.84 tok/s on DGX Spark GB10 with Gemma 3
1B Q4_K_M. Ollama on the same hardware achieved ~197 tok/s. The gap was 15-450x.
Over four phases of optimization (Phases 3-6), Zerfoo reached 234.30 tok/s —
18.8% faster than Ollama.

This ADR documents the key decisions, optimizations, and dead ends across all
phases so the knowledge is preserved.

## Hardware
- NVIDIA DGX Spark GB10 (Blackwell, sm_121)
- 273 GB/s LPDDR5x unified memory
- 128 GB total memory
- NVLink-C2C between CPU and GPU (managed memory capable)

## The Journey: 12.84 → 234.30 tok/s

### Phase 3: Correctness + GPU Residency (12.84 → ~140 tok/s)

**Problem:** Degenerate output (model producing garbage) and massive CPU-GPU
round-trips. 43% of inference time was cgocall overhead from D2H/H2D copies.

**Key optimizations:**
1. **Fixed output correctness** — Pre-existing bugs in tensor operations.
2. **GPU Transpose and GPU Gather** — Eliminated CPU fallbacks that caused
   D2H→compute→H2D round-trips for every transpose and gather operation.
3. **GPU broadcasting** — Moved broadcast operations to GPU, eliminating
   another class of CPU round-trips.
4. **purego FFI** — Replaced CGo calls with zero-overhead purego trampolines
   (`runtime.asmcgocall` + assembly). Eliminated CGo stack switching overhead.

**Lesson:** GPU residency matters more than kernel optimization. Keeping tensors
on GPU and eliminating CPU round-trips was worth 10x.

### Phase 4: Kernel Fusion + CUDA Graph Infrastructure (140 → 191 tok/s)

**Problem:** Per-kernel launch overhead (~338 launches/token) and redundant
memory traffic from unfused kernels.

**Key optimizations:**
1. **FP16/FP8 dispatch elimination** — The F32 inference path was checking
   FP16/FP8 type switches on every operation. Eliminating this dispatch gave
   +32 tok/s. This was the largest single win and was not predicted by
   profiling — it was discovered by code review.
2. **Fused kernels** — SwiGLU, Scale+Softmax, Add+RMSNorm, QK-Norm+RoPE.
   Each fusion reduced kernel launches and eliminated intermediate memory
   writes. The fused QK-Norm+RoPE was particularly effective (4 launches → 1).
3. **Token tensor reuse** — Reused the same GPU buffer for the single-token
   decode input instead of allocating per step.
4. **D2H copy removal** — Eliminated unnecessary device-to-host copies that
   were left over from debugging.
5. **Async memcpy** — Used cudaMemcpyAsync for KV cache D2D copies during
   CUDA graph capture attempts.
6. **CUDA graph infrastructure** — Built the full capture/instantiate/replay
   framework, arena reset floor, and captured slot restore. Could not activate
   because GQA reads CPU-side position values (SeqLen, RoPE offset, KV offset).

**Dead ends:**
- Q4K GEMV vectorization: Increased register pressure (43→54 regs/thread),
  caused 12.2% throughput regression. Reverted. The kernel is memory-bound
  and occupancy matters more than instruction throughput.
- CUDA graph for GQA: Blocked by position-dependent CPU reads. Deferred
  to Phase 6.
- FP8 inference: Worked (53.70 tok/s) but produced degenerate quality on
  sm_121 because there is no native cublasLt FP8 GEMM support.

**Lesson:** The biggest win (+32 tok/s) came from eliminating type dispatch
overhead, not from GPU kernel optimization. Profile the host code, not just
the device code.

### Phase 5: Go Runtime Optimizations — All Dead Ends (191 → 190 tok/s)

**Problem:** 3% gap to Ollama. Hypothesis: Go runtime overhead (GC, bounds
checks, purego FFI, PGO, thread scheduling).

**Key findings — all negative:**
1. **PGO (Profile-Guided Optimization)** — No measurable improvement. The hot
   path is CUDA kernel execution, not Go code. PGO optimizes Go instructions
   that are <1% of token time.
2. **GC elimination** — Zero GC pauses during decode. The arena allocator
   already prevented allocations in the hot path.
3. **Bounds check elimination** — 928 total BCE in the codebase, only 8 in
   hot-path code (<0.1% of runtime). Not worth the code complexity.
4. **purego FFI overhead** — ~395 calls/token at ~50ns each = ~20μs total
   (<0.4% of 5.26ms token time). Already optimized.
5. **runtime.LockOSThread** — Caused a 2.6% regression (190→185 tok/s).
   The Go scheduler's thread migration cost was less than the overhead of
   pinning. Reverted.

**Lesson:** When the hot path is GPU kernel execution, Go runtime overhead
is negligible. Every audit confirmed the bottleneck was CUDA kernel launch
overhead, not Go. This phase was valuable for proving where the problem
was NOT, which directed Phase 6.

### Phase 6: CUDA Graph Capture (190 → 241 tok/s)

**Problem:** ~338 kernel launches/token at ~5μs each = ~1.7ms of the 5.26ms
token budget (32%). CUDA graph replay eliminates launch overhead by replaying
a recorded sequence of kernel launches as a single GPU operation.

**Blocker:** GroupedQueryAttention (GQA) reads three CPU-side values per token:
1. `cache.SeqLen()` — host cursor position
2. RoPE angle offset — derived from SeqLen
3. KV cache append offset — `cursor * dim`

These values are baked into captured kernel arguments. On graph replay, stale
values produce wrong output.

**Solution: GPU-resident position counter (ADR-032)**
1. Added 3 CUDA kernels: `increment_counter`, `offset_memcpy`, `rope_select`.
2. Stored decode position as GPU-resident int32, incremented by a trivial
   CUDA kernel instead of CPU `cursor++`.
3. `rope_select` reads the GPU counter to index into the precomputed cos/sin
   table — no CPU offset computation needed.
4. `offset_memcpy` reads the GPU counter to compute KV cache append offset —
   no CPU arithmetic needed.
5. Removed GQA from the non-capturable ops list.
6. Added GPU counter to TensorCache (the production KV cache implementation).
7. Result: 184/185 instructions captured (only EmbeddingLookup excluded).

**Bug found:** GPU counter was not synced after prefill. The decode graph
replayed with counter=0, causing offset_memcpy to overwrite position 0 and
rope_select to use wrong angles. Fixed by H2D copy of the prefill token
count into the GPU counter before decode starts.

**Result:** 234.30 tok/s (3-run avg: 235.09, 234.42, 233.39).
25.9% speedup from CUDA graph alone.

**Open issue:** Graph and no-graph paths produce different (but both coherent
and deterministic) output at temp=0. Likely floating-point ordering difference
from captured vs individual kernel launches. Does not affect throughput.

## Summary of What Mattered

| Optimization | Phase | Impact | Category |
|-------------|-------|--------|----------|
| GPU residency (no D2H/H2D) | 3 | ~10x | Architecture |
| purego FFI (no CGo overhead) | 3 | ~2x | Architecture |
| FP16/FP8 dispatch elimination | 4 | +32 tok/s | Host code |
| Fused kernels (SwiGLU, RoPE, etc.) | 4 | +15 tok/s | GPU kernels |
| Token tensor reuse | 4 | +5 tok/s | Memory |
| CUDA graph capture | 6 | +44 tok/s | Architecture |

## What Did NOT Matter

| Optimization | Phase | Impact | Why |
|-------------|-------|--------|-----|
| PGO | 5 | 0% | Hot path is CUDA, not Go |
| GC elimination | 5 | 0% | Zero GC pauses already |
| Bounds check elimination | 5 | 0% | <0.1% of runtime |
| purego call reduction | 5 | 0% | <0.4% of runtime |
| runtime.LockOSThread | 5 | -2.6% | Worse than scheduler default |
| Q4K GEMV vectorization | 4 | -12.2% | Register pressure > instruction throughput |

## Key Principles Learned

1. **GPU residency is everything.** A single D2H copy costs more than hundreds
   of GPU kernel launches. Keep tensors on GPU at all costs.

2. **Profile the host, not just the device.** The +32 tok/s type dispatch win
   was invisible in GPU profiling. It only showed up in CPU flame graphs.

3. **Eliminate launch overhead before optimizing kernels.** CUDA graph capture
   gave 25.9% speedup with zero kernel changes. Optimizing the Q4K kernel
   regressed performance.

4. **Prove where the bottleneck is NOT.** Phase 5's negative results (all five
   Go runtime hypotheses disproven) directed Phase 6 to the right target.

5. **Memory-bound kernels care about occupancy, not instruction throughput.**
   The Q4K GEMV vectorization increased register pressure, reduced occupancy,
   and regressed 12.2%. For memory-bound kernels, more warps > faster warps.

6. **Make GPU state self-contained for graph capture.** Any CPU-read dependency
   in the captured region breaks replay. The GPU counter pattern (ADR-032) is
   generalizable: replace any CPU-side counter/offset with a GPU-resident scalar
   incremented by a trivial kernel.

## Consequences
- Zerfoo F32 inference on Gemma 3 1B Q4_K_M surpasses Ollama by 18.8%.
- The CUDA graph architecture is extensible: adding new capturable ops is
  straightforward now that the position counter pattern is established.
- Remaining optimization headroom: ~40% to theoretical bandwidth limit (~390
  tok/s). Primary targets: custom GEMV replacing cuBLAS, FP16 KV cache.
- The GPU counter adds one D2H sync after generation completes. This is
  negligible for throughput but means `SeqLen()` is stale during decode.
  Code that needs mid-decode position must use `GPUCounterPtr()`.
