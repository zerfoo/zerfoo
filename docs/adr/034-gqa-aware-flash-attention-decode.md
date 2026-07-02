# ADR 034: GQA-Aware Flash Attention Decode Kernel

## Status
Accepted

## Date
2026-03-14

## Context
Phase 7 added a decode-specific flash_attention_decode kernel (T903.2) that
reads KV sequence length from GPU memory, enabling CUDA graph capture. However,
for GQA models (Gemma 3: 8 query heads, 4 KV heads), the decode fast path
must expand KV heads to match query heads via engine.Repeat. This Repeat
operates on the full maxSeqLen (8192) KV buffer, creating ~128 MB temporaries
per token and causing a 93.7% throughput regression (234 -> 14 tok/s).

The decode fast path was disabled for GQA models (commit 9803ba1), restoring
241 tok/s. But this means the GQA attention path is NOT CUDA-graph-captured,
losing the launch overhead savings for the ~156 attention kernel launches per
token.

T901.1 profiling showed cuBLAS is only 8% of decode time (not 30% as planned).
Weight matmuls already use fused Q4K GEMV. The original plan to replace cuBLAS
with a custom SGEMV has low ROI. The real optimization opportunity is making
the attention path GQA-aware and graph-capturable.

## Decision
Implement GQA head replication inside the flash_attention_decode CUDA kernel
rather than using engine.Repeat on the host side. The kernel will:

1. Accept separate numQueryHeads and numKVHeads parameters.
2. For each query head, compute which KV head to use: kv_head = q_head / (numQueryHeads / numKVHeads).
3. Index into the KV buffer using the computed kv_head, performing the
   replication at register level with zero extra memory traffic.
4. Read KV sequence length from GPU-resident pointer (existing pattern).
5. Remain fully CUDA-graph-capturable.

This approach extends the existing flash_attention_decode kernel rather than
creating a separate kernel, keeping the codebase clean.

Custom SGEMV integration (T901.4, T901.5, T901.6) is deferred as low ROI
per profiling results.

## Consequences
Positive:
- Eliminates 156 attention kernel launches per token for GQA models.
- Eliminates ~128 MB/token temporary allocations from Repeat.
- Enables full CUDA graph capture for GQA decode path.
- Expected speedup: 15-25% (recovering launch overhead + eliminating Repeat).
- Clean architecture: single kernel handles both GQA and non-GQA.

Negative:
- Kernel complexity increases (GQA indexing logic in the inner loop).
- Register pressure may increase slightly from the extra index computation.
- Must handle variable head ratios (2x for Gemma 3, but could be 4x, 8x, etc.).
