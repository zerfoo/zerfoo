# ADR 072: Kimi Linear Attention Layer

## Status
Accepted

## Date
2026-03-28

## Context
Kimi K2/K2.5 (Moonshot AI) uses linear attention instead of softmax attention.
Standard attention is O(n^2*d); linear attention achieves O(n*d^2) by avoiding
the softmax normalization step.

## Decision
Implemented kimiLinearAttentionNode with feature map phi(x) = ELU(x) + 1.
Computes O = phi(Q) * (phi(K)^T * V) with causal accumulation of KV state
and per-position normalization. Registered as "kimi-linear" architecture.

## Consequences
- Enables loading Kimi K2.5 GGUF models (MoE with linear attention).
- Linear attention is a new attention primitive alongside softmax GQA -- could be
  reused by future linear attention models (RWKV variants, RetNet, etc.).
- No impact on existing softmax attention architectures.
