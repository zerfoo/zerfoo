# ADR 075: Batched Training with Kernel Fusion for Time Series Backends

## Status
Accepted

## Date
2026-03-29

## Context
Training any time series backend (DLinear, PatchTST, iTransformer, Mamba, CfC, N-HiTS,
FreTS, TTM) on real-world datasets (28K+ rows) is impractical. The root cause is per-sample
Go dispatch overhead: the training loop calls GPU for individual tensor operations per sample,
with Go-to-GPU-to-Go round-trip overhead dominating. GPU utilization is 0% during TrainWindowed.

Prior fixes (#157, #169, #172, #207) improved batching within individual operations but did
not address the fundamental issue: the forward pass must be batched across samples so one GPU
kernel launch handles an entire mini-batch, not one sample at a time.

## Decision
Refactor the training forward path to operate on batched tensors:

1. **Batched tensor layout**: Change TrainWindowed input from `[nSamples][channels][inputLen]`
   (slice-of-slice, one GPU call per sample) to a single `tensor.TensorNumeric[float32]` of
   shape `[batchSize, channels, inputLen]`. One GPU forward call per mini-batch.

2. **Engine-batched forward**: Each backend's `_engine.go` file gets a `forwardBatch()` method
   that accepts `[batch, channels, inputLen]` and returns `[batch, outputDim]` in one pass.
   All MatMul, LayerNorm, and attention operations operate on the full batch tensor.

3. **Backward through batch**: Each backend's `_backward.go` file computes gradients on the
   full batch. The loss is averaged across the batch (mean reduction).

4. **DataLoader abstraction**: A new `timeseries.DataLoader` handles windowing, shuffling,
   and mini-batch assembly from raw float64 data into engine-ready float32 tensors.

## Consequences
Positive:
- 100-1000x speedup expected (matches LightGBM training times on same data).
- GPU utilization rises from 0% to practical levels.
- DataLoader is reusable across all 9+ backends.

Negative:
- Backward pass must be rewritten for all backends with engine support.
- Memory footprint increases (full batch in GPU memory vs one sample).
- Breaking change to internal training API (TrainWindowed signature changes).
