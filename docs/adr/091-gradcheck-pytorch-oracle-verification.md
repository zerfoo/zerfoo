# ADR 091: Per-op verification harnesses -- gradcheck, engine parity under arena stress, and PyTorch as oracle

## Status
Accepted

## Date
2026-06-10

## Context
Eight distinct GPU-training bugs were found and fixed in one investigation
(arm64 NEON softmax NaN, fast-math tanh overflow, LayerNorm cached-variance
corruption, AdamW f32 second moment, gradient-zeroing arena corruption, plus
model-side issues), and GPU f32 training of the Wolf CrossAsset model still
diverges while CPU trains cleanly. Every one of those bugs was discovered as a
production NaN on the DGX -- hours of live-cluster bisection each -- because
Zerfoo/ztensor has no systematic per-op verification. PyTorch prevents these
classes with torch.autograd.gradcheck (finite-difference validation of every
op's backward, OpInfo-driven) and a decade of kernel validation. Zerfoo has
ad-hoc finite-difference checks in a few timeseries tests but no general
harness, no GPU-vs-CPU parity testing under allocator pressure, and no external
numerical reference.

We also have an asset PyTorch never had: PyTorch itself, running on the same
GB10 (NGC container nvcr.io/nvidia/pytorch:26.02-py3), as a ground-truth oracle
for both forward and backward numerics.

## Decision
Build three complementary harnesses, and require them for all new ops:

1. **gradcheck (math correctness).** A general finite-difference gradient
   checker for graph.Node implementations: central differences at float64 on
   CPU vs the node's analytic Backward, driven by an OpInfo-style registry of
   ops with representative shapes/domains. Catches wrong Jacobians.
2. **Engine parity under arena stress (implementation correctness).** The same
   op sequence run CPU-f32 vs GPU-f32, forward AND backward, in interleaved
   schedules (A.fwd, B.fwd, ..., A.bwd) with a small arena to force buffer
   reuse. Catches kernel bugs and the cached-intermediate corruption class that
   single-op tests cannot see (pairs with ztensor ADR 006 poison mode).
3. **PyTorch oracle (convention correctness).** A tensor-exchange format plus a
   Python runner in the NGC PyTorch container on the GB10: the same op, same
   inputs, through ztensor and torch; diff forward and backward outputs within
   per-op tolerances. Catches numerics-convention divergence (fast-math,
   reduction ordering, eps placement) that both Go engines could share.

Each harness must encode at least one historically-fixed bug as a regression
fixture proving it would have caught that bug class. GPU-dependent harness runs
execute as Spark pods on the GB10 (CI has no GPU); the gradcheck harness runs
in ordinary CI.

## Consequences
Positive: converts "mystery NaN on the DGX" into a failing test that names the
op; gives every future kernel/op a mechanical acceptance gate; the oracle
harness lets a small team borrow PyTorch's decade of numerical validation
instead of re-deriving it.

Negative: an OpInfo registry to maintain; GPU harness runs consume DGX time and
must serialize on the single GB10; the oracle adds a Python container
dependency (pinned NGC tag) for test infrastructure only -- the production
stack remains pure Go.

References: ztensor docs/adr/006 (save-for-backward + poison mode); Wolf
docs/adr/072 and devlog 2026-06-10 (the failure history);
docs/plan-gpu-training-hardening.md (the umbrella plan).
