# ADR 042: dp4a INT8 Q4_K GEMV with FP32 FMA Fallback

## Status
Accepted

## Date
2026-03-17

## Context

Q4_K GEMV at batch=1 autoregressive decode is memory-bandwidth-bound on GB10
(128GB LPDDR5x). The existing scalar FP32 FMA Q4_K GEMV kernel is correct but
leaves compute throughput on the table.

llama.cpp analysis showed that dp4a INT8 dot-product intrinsics achieve 4 MACs
per instruction vs scalar FP32 FMA (1 MAC per instruction), a 2-4x compute
throughput gap. At batch>1, when multiple sequences decode in parallel, the
operation shifts from memory-bound to compute-bound and dp4a becomes decisive.

The new kernel must not regress single-batch decode performance since the
existing 245 tok/s baseline is memory-bound and cannot benefit from faster
compute alone.

## Decision

Add a dp4a INT8 Q4_K GEMV kernel alongside the FP32 FMA path in gemv_q4k.cu.
The purego wrapper (gemv_q4k_purego.go) registers the dp4a symbol as optional
(soft symbol via purego dlsym). At runtime, GPUEngine selects dp4a when the
symbol is available on the device; otherwise falls back to FP32 FMA
transparently. No user-facing API changes, no build tags, no CGo.

## Consequences

+ At batch>1, dp4a provides up to 4x compute throughput improvement over
  scalar FP32 FMA, enabling competitive throughput at larger batch sizes.
+ Zero performance regression at batch=1 on memory-bandwidth-bound hardware
  (results within measurement noise of the 245 tok/s baseline).
+ No build complexity: the fallback path keeps CPU-only builds working without
  modification.
+ Purego soft-symbol pattern generalizes to future optional kernel variants.
- dp4a benefit is not visible in single-batch decode benchmarks on GB10 where
  memory bandwidth is always the bottleneck.
