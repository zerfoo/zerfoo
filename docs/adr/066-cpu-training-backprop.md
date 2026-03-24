# ADR 066: Replace finite-difference gradients with analytical backprop in timeseries CPU training

## Status
Accepted

## Date
2026-03-24

## Context

GitHub issue #157 reports that ALL timeseries backends time out (>300s) on trivially
small datasets (1K rows, 5 features, 5 epochs) when using the CPU (pure-Go) training
path. Investigation revealed three distinct gradient computation strategies across
the 7 backends:

1. **Analytical backprop** (DLinear, NHiTS, FreTS): O(P) per sample. These backends
   derive gradients by hand and accumulate them in a flat gradient vector. They are
   already fast enough.

2. **Finite differences** (ITransformer, PatchTST CPU path): O(P * forward_cost)
   per sample. These backends perturb each parameter and re-run the full forward
   pass to estimate the gradient numerically. For a model with P=5000 parameters,
   this means 5000-10000 forward passes per sample per epoch.

3. **Full Jacobian** (CfC): O(outDim * P) per sample. The backwardSample method
   returns a [outDim][nParams] matrix instead of a single gradient vector. This
   requires computing and storing outDim separate gradient vectors.

The engine-based path (using ztensor compute.Engine with graph.Backward) provides
proper autograd for all backends but requires explicit engine construction.

## Decision

Replace finite-difference and full-Jacobian backward passes with analytical
backpropagation in the CPU training path. Specifically:

- **ITransformer**: Replace forward finite differences with chain-rule backprop
  through the transformer layers (embedding, attention, FFN, projection).
- **PatchTST CPU**: Replace central finite differences with chain-rule backprop
  through patch embedding, transformer encoder, and output head.
- **CfC**: Refactor backwardSample to return a single gradient vector (not Jacobian)
  by passing the upstream dLoss/dOutput through the backward pass directly.

For Mamba, which already uses the engine path, add a CPU fallback that constructs
a CPUEngine automatically when no engine is provided.

## Consequences

**Positive:**
- CPU training becomes O(E * N * P) instead of O(E * N * P^2) or O(E * N * P * forward).
- Expected speedup: 1000x-10000x for ITransformer/PatchTST, 10x-100x for CfC.
- 1K rows x 5 features x 5 epochs should complete in <1s.
- No new dependencies required.

**Negative:**
- Manual backprop code is error-prone. Each backend needs finite-difference gradient
  tests to verify correctness.
- More code to maintain per backend.
- Alternative (routing through CPU engine) would have been simpler but adds a hard
  dependency on ztensor engine for CPU-only users.
