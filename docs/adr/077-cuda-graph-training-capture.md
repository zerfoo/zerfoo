# ADR 077: CUDA Graph Capture for Time Series Training

## Status
Accepted

## Date
2026-03-30

## Context

PatchTST GPU training takes 63.7s/epoch on DGX Spark (28K samples x 20
channels x batch=64) despite channel batching and batched attention. Profiling
shows the bottleneck is Go-to-GPU synchronization overhead: ~500 engine op
calls per batch iteration, each requiring a Go round-trip to the GPU. The
tensors are small ([7680, 64]) so individual op execution is fast, but the
cumulative sync overhead dominates.

CUDA graph capture records a sequence of GPU operations into a replayable
graph that executes in a single GPU submission with zero intermediate
synchronization. ztensor already has CUDA graph capture infrastructure for
inference (graph/cuda_graph.go) with StreamBeginCapture, StreamEndCapture,
GraphInstantiate, and GraphLaunch bindings.

The training loop has a partial batch problem: the last batch in each epoch
may have fewer samples than batch_size (e.g., 28000 % 64 = 0, but for other
dataset sizes this is common). CUDA graphs require fixed tensor shapes between
capture and replay. Two options: (A) pad the partial batch to full size and
mask out padded samples, or (B) drop the partial batch entirely.

## Decision

1. **Drop partial batches** (`drop_last=True` semantics). When
   `nSamples % batchSize != 0`, the final partial batch is skipped. At most
   `batchSize - 1` samples are excluded per epoch (0.2% for batch=64,
   n=28000). This is standard practice (PyTorch DataLoader default for
   training) and eliminates the need for shape-conditional graph replay or
   loss masking logic.

2. **Capture one combined forward+backward graph** on the first full batch,
   then replay it for all subsequent batches within the epoch. The graph
   includes all engine ops from gradient zeroing through backward pass
   completion. Layer norm and GELU remain as engine ops (Sum, Mul, Sub, etc.)
   -- they are capturable because they only use GPU tensors. AdamW and loss
   computation stay outside the graph (they read .Data() which triggers D2H).

3. **Implement capture in the training loop** (zerfoo repo), not in ztensor's
   graph compiler. The training loop already calls engine ops directly; we add
   StreamBeginCapture/EndCapture around the forward+backward block and
   GraphLaunch for replays. This avoids coupling training to ztensor's
   inference-focused graph compilation pipeline.

4. **Pre-allocate all tensors** before capture. CUDA graph capture fails if
   new GPU memory is allocated during capture (cudaMalloc inside capture
   returns error 901). All layer caches, gradient tensors, and intermediate
   buffers must be allocated before StreamBeginCapture. The warmup batch
   (first batch) runs without capture to establish buffer sizes, then capture
   happens on the second batch.

## Consequences

**Positive:**
- Eliminates ~500 Go-to-GPU sync points per batch (5-20x expected speedup)
- No changes needed to ztensor's engine interface or CUDA bindings
- Drop-last is simpler and numerically correct (no gradient pollution from padding)
- Graph replay is deterministic and reproducible

**Negative:**
- Up to batchSize-1 samples excluded per epoch (negligible for large datasets)
- Tensor shapes are fixed after capture -- cannot change batch size mid-epoch
- Layer cache tensors must be pre-allocated (slightly higher peak memory)
- First two batches of each epoch are slower (warmup + capture overhead)
- AdamW optimizer step remains outside graph (minor, runs once per batch)
