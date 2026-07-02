# ADR 070: Make RoPE Optional in GQA for Non-Rotary Architectures

## Status
Accepted

## Date
2026-03-28

## Context
GPT-2 uses learned absolute position embeddings instead of RoPE. To reuse
the existing GroupedQueryAttention layer (with KV cache, causal masking,
and GQA head splitting), RoPE needed to be optional rather than required.

## Decision
Added WithNoRoPE[T]() option to GQA. When enabled, RoPE is nil and Q/K
pass through without rotation. All RoPE call sites guarded with nil checks.

## Consequences
- GPT-2 and any future non-rotary architecture can reuse GQA infrastructure.
- Zero performance impact on existing RoPE architectures (nil check is branch-predicted).
- Backward compatible -- existing builders unaffected.
