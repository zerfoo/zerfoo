# ADR-088: CUDA Graph Capture Compatibility for Gemma 4 Edge PLE Combiner

**Status:** Accepted
**Date:** 2026-04-15
**Epic:** E99 (CUDA graph capture + pleCombinedProducer H2D incompatibility)

## Context

When `gemma4e` runs on CUDA with CUDA graph capture enabled, the capture
stream fails with:

```
cudaMemcpy failed: operation would make the legacy stream depend on
a capturing blocking stream
```

The failure is triggered inside `Gemma4PLECombinedProducer.Forward`
(`inference/gemma4_edge_ple_nodes.go`), which:

1. Reads token ids from GPU via `ids.Data()` (D2H copy).
2. Gathers per-token PLE rows into a CPU buffer (`tokenFlat`, ~35*256
   floats per token).
3. Wraps the buffer as a `tensor.TensorNumeric[T]` via `tensor.New` —
   this is a CPU-resident (`CPUStorage`) tensor.
4. Calls `engine.MulScalar(tokenPLE, sqrt(pleDim))`. The GPU engine's
   `getDevicePtr` sees a CPUStorage input and issues a synchronous H2D
   cudaMemcpy on the capturing stream, which CUDA rejects.

Downstream of the producer, each of 35 `pleSliceNode` instances extracts
a `[B, S, 256]` slice from the producer's full-width `tokenPLE` and
`modelProj` tensors. The slicing is done by `sliceLastDim` which calls
`.Data()` on the source and allocates a new CPU tensor — also a
capture-breaker, even if the producer is handled.

The workaround currently in production is
`ZERFOO_DISABLE_CUDA_GRAPH=1` (applied in
`docs/bench/manifests/gemma4-e2e.yaml`). This disables graph capture
entirely for gemma4e, costing roughly 10-30% decode throughput versus
captured execution (based on the 184/185 instruction capture rate seen
on other GGUF models).

Additional context:

- `pleCombinedProducer` runs once per forward pass (pre-layer-loop).
- `pleSliceNode` runs once per transformer layer, interleaved with
  attention + FFN. If these stay non-capturable, the "longest
  contiguous capturable run" in `ztensor/graph/cuda_graph.go` collapses
  to a single layer, making capture practically worthless.
- Full investigation: `docs/devlog.md` entries 2026-04-14/15 under E98,
  and the E99 section in `docs/plan.md`.

## Options Considered

### A. Exclude `Gemma4PLECombinedProducer` from capture (partial capture)

**What it means.** Add `Gemma4PLECombinedProducer` to the
`nonCapturableOps` map in `ztensor/graph/cuda_graph.go`. The producer
naturally belongs in pre-capture (it runs once, before the layer loop).

**Why it is insufficient on its own.** Does not address the 35 per-layer
`Gemma4PLESlice` instances, which also do CPU work (`sliceLastDim`
calls `.Data()`). If we also mark `Gemma4PLESlice` as non-capturable,
the capture region fragments into 35 tiny runs (one layer's
attention+FFN each), destroying the benefit of graph replay.

### B. Move host-side work to GPU kernels

**What it means.** Write a custom CUDA kernel that does the PLE
embedding lookup + per-layer slicing entirely on GPU.

**Why rejected.** Requires new CUDA kernel code in ztensor, purego
bindings, tests, and cross-repo coordination. The embedding lookup is
inherently a gather, which is already classified as non-capturable for
every other model (the main EmbeddingLookup node). A GPU-kernel rewrite
would not change that classification for the producer — it would still
need to run in pre-capture. Large effort for marginal gain.

### C. Hybrid: producer in pre-capture, slices pre-computed with stable GPU addresses

**What it means.**

1. Mark `Gemma4PLECombinedProducer` as non-capturable (ztensor change,
   additive to the `nonCapturableOps` map). The producer runs in
   pre-capture like `EmbeddingLookup`.
2. In the producer, after computing the full-width `tokenPLE` and
   `modelProj` CPU tensors, pre-slice them into 35 per-layer tensors.
3. On the first forward pass, allocate stable GPU buffers for each
   per-layer slice (via `engine.MulScalar(slice, 1.0)` — a no-op
   multiply that returns a GPU tensor). Cache the buffers in the
   producer struct.
4. On every subsequent pass, reuse the cached GPU buffers by calling
   `GPUStorage.CopyFromHost(newSliceData, 0)`. GPU addresses are stable
   across decode steps — exactly what CUDA graph replay requires.
5. Refactor `pleSliceNode.Forward` to read the producer's pre-computed
   GPU slices directly (via Go struct field), skipping `sliceLastDim`
   and all `.Data()` calls. Downstream `RMSNorm`, `Add`, and
   `MulScalar` then execute on GPU-resident inputs — fully capturable.

**Why chosen.**

- **Addresses both host-side blockers.** Producer goes to pre-capture
  (one instruction out of the region). Slicers become GPU-only, so they
  stay inside the capture region.
- **Preserves the full-layer-stack capture.** The longest capturable
  run becomes the entire transformer body (~385 instructions at 35
  layers), mirroring the capture pattern on other GGUF architectures.
- **Mimics an existing pattern.** The CUDA graph executor already
  caches the `EmbeddingLookup` output GPU buffer and refreshes it via
  `CopyFromHost` on each replay (see
  `ztensor/graph/cuda_graph.go:509-528`). Applying the same pattern to
  PLE slices is idiomatic.
- **No new CUDA kernels.** All GPU operations used (MulScalar, MatMul,
  CopyFromHost) already exist in ztensor.
- **Cross-repo coordination is small.** One additive entry in ztensor;
  the rest is local to `zerfoo/inference/gemma4_edge_ple_nodes.go`.

### D. Flow PLE data through the graph slot system

**What it means.** Restructure `pleCombinedProducer` to emit `tokenPLE`
and `modelProj` as proper graph outputs, and make each `pleSliceNode`
receive them as graph inputs. The slot system then tracks them, and
`EnsureSlotsGPU` handles upload.

**Why rejected.** The `graph.Node[T]` interface currently returns a
single tensor from `Forward`. Supporting multi-output nodes is a
cross-cutting change across ztensor's graph compiler, plan executor,
and every existing builder. Too invasive for a gemma4e-specific
performance fix.

## Decision

**Adopt Option C.** Pre-slice in the producer, maintain stable GPU
buffers via `CopyFromHost`, read pre-computed slices in
`pleSliceNode`.

## Consequences

### Positive

- gemma4e decode on CUDA can drop `ZERFOO_DISABLE_CUDA_GRAPH=1` and
  recover the 10-30% graph-replay speedup other architectures get.
- Pattern is reusable for any future node that does CPU work but must
  feed capturable downstream ops (e.g., future audio/vision PLE
  variants).

### Negative / cost

- `pleCombinedProducer` becomes stateful across calls (it caches the
  GPU slice tensors). Currently it rebuilds `tokenPLE` and `modelProj`
  from scratch each call; the cache is purely additive.
- Per-step work increases slightly in pre-capture: one extra CPU slice
  loop (35 * B*S*256 elements) and 35 `CopyFromHost` calls
  (~35 KB per decode step). Negligible next to the 35-layer
  transformer.
- First-call path allocates 35 * 2 = 70 GPU buffers (each small). The
  pool handles this; allocations are amortized.

### Follow-ups

- T99.1.2: implement the change in ztensor (add to `nonCapturableOps`)
  and zerfoo (pre-slice producer + simplified slicer).
- T99.1.3: rerun `scripts/gemma4-spark.sh -mode generate -device cuda
  -steps 20` on DGX without `ZERFOO_DISABLE_CUDA_GRAPH=1`. Verify the
  capture message appears, output is non-degenerate, and tok/s meets
  or beats the uncaptured baseline (3.85 tok/s at commit `72828131`).
- If throughput does not improve, that is a separate investigation —
  correctness first; drop the env var from the manifest only when both
  correctness and throughput are confirmed.
