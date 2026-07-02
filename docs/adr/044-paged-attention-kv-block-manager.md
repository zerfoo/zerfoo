# ADR 044: PagedAttention KV Block Manager

## Status
Accepted

## Date
2026-03-17

## Context
Zerfoo's current KV cache allocates a contiguous tensor per sequence at session
creation. This causes three problems: (1) internal fragmentation when sequences
finish early and free irregular chunks, (2) inability to share prefill KV pages
across sequences with identical prompt prefixes, and (3) hard upper bound on
concurrent sessions tied to peak allocated memory rather than actual KV usage.
vLLM demonstrated that PagedAttention eliminates KV cache memory waste from
60-80% down to under 4% and enables 2-4x throughput at identical memory budgets.

## Decision
Implement a PagedAttention block manager in ztensor's graph/kv/ package. KV cache
is divided into fixed-size blocks of 16 tokens each. A block table per sequence
maps logical block indices to physical GPU memory blocks. The block manager
maintains a free-block pool and allocates/frees blocks dynamically as sequences
grow and complete. Prefix sharing reuses physical blocks for identical prompt
prefixes (RadixAttention extension in ADR-046).

Block size 16 tokens is chosen to balance: (a) fine-grained allocation granularity
vs. (b) GPU memory coalescing efficiency. Blocks are typed by KV dtype (FP16 default,
FP32 fallback) and head dimension.

The block manager is implemented in Go and GPU block allocation uses the existing
arena allocator in ztensor/internal/cuda/arena.go extended with a block registry.

## Consequences
Positive:
- Removes hard concurrency ceiling imposed by pre-allocated KV tensors
- Enables prefix caching (ADR-046) as a zero-overhead extension
- Enables speculative decoding (ADR-045) by allowing draft/target to share KV pages

Negative:
- Attention kernel must accept block tables (pointer-of-pointers indirection)
  instead of contiguous KV buffers; custom CUDA kernel required
- Block table management adds per-step Go overhead (~5-10 us); acceptable given
  GPU compute dominates at 4ms+ per step
- Initial implementation targets single-GPU; multi-GPU requires distributed block
  manager (deferred to Year 2)
