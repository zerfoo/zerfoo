# ADR 032: GPU-Resident Position Counter for CUDA Graph Capture

## Status
Accepted

## Date
2026-03-14

## Context
CUDA graph capture records GPU kernel launches and replays them without CPU
intervention. Zerfoo's decode loop cannot be captured because GroupedQueryAttention
(GQA) reads three CPU-side values per token:

1. `cache.SeqLen()` -- host-resident cursor position (kvcache.go line 153)
2. RoPE angle offset -- derived from cache.SeqLen() (grouped_query_attention.go line 395)
3. KV cache append offset -- `lb.cursor * dim` (kvcache.go line 138)

These values are passed as kernel arguments during capture, then baked into the
graph. On replay, the same (stale) values are used, producing wrong output.

Phase 4 built the full CUDA graph infrastructure (capture, instantiate, replay,
arena reset floor, captured slot restore) but cannot use it because GQA appears
at instruction 2, interleaved in every transformer layer.

Phase 5 proved the remaining 3.6% gap to Ollama is entirely in CUDA kernel
launch overhead (~338 launches/token at ~5us each = ~1.7ms of 5.26ms/token).
CUDA graph replay would eliminate this overhead.

## Decision
Store the decode position counter as a GPU-resident scalar. At each decode step,
increment the counter via a trivial CUDA kernel (1 thread, 1 add). RoPE angle
selection and KV cache append offset computation will use this GPU counter
instead of CPU-side `cache.SeqLen()`.

Specifically:
- Allocate a single int32 on GPU during KV cache init.
- Add a CUDA kernel `increment_counter(int* counter)` to atomically increment it.
- Modify `RotaryPositionalEmbedding.GetAngles()` to accept a GPU pointer to the
  position offset instead of a CPU int.
- Modify `KVCache.Update()` to compute the append offset from the GPU counter
  using a CUDA kernel instead of CPU arithmetic.

This makes all three blockers GPU-resident, enabling full CUDA graph capture.

## Consequences
- Positive: Enables CUDA graph capture for the full decode loop. Expected
  throughput gain: 10-30% (1.7ms launch overhead eliminated from 5.26ms/token).
  Would close the 3.6% Ollama gap and likely surpass it significantly.
- Positive: GPU counter is a single int32 -- negligible memory cost.
- Negative: Requires new CUDA kernels for counter increment and offset-based
  memcpy. Estimated 3-4 new kernels.
- Negative: CPU can no longer read the current position without a D2H copy.
  Add a sync method for debugging/logging only.
- Negative: Prefill (variable seqLen) still needs CPU-side orchestration.
  CUDA graph capture applies only to decode (seqLen=1).
