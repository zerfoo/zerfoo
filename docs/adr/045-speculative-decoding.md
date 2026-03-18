# ADR 045: Speculative Decoding Implementation Strategy

## Status
Accepted

## Date
2026-03-17

## Context
Autoregressive decoding is memory-bandwidth-bound at batch=1. Each step loads all
model weights from HBM for a single token output. Speculative decoding addresses
this by using a small draft model to generate K candidate tokens in one pass, then
verifying all K tokens with the target model in a single forward pass. When the
acceptance rate (alpha) is high enough, this achieves 2-3x throughput improvement
without changing output distribution.

Key design choices: draft model selection strategy, rejection sampling algorithm,
and fallback for low-alpha workloads.

## Decision
Implement speculative decoding in generate/speculative.go with two modes:

1. Independent draft model (ExternalDraft): a separate smaller GGUF model (e.g.,
   Gemma 3 1B drafts for Gemma 3 27B target). Draft and target share the same
   Engine[T] and KV block manager but use separate KV caches.

2. Self-speculative decoding (SelfDraft): the same model generates drafts by
   running early-exit inference through a subset of layers (layers 0..N/2), then
   verifies with the full model. Eliminates the need for a separate model file.

Draft length K=4 tokens by default, configurable via GenerateOptions.DraftTokens.
Acceptance uses standard speculative sampling (Leviathan et al. 2023): accept token
k if rand() <= min(1, p_target(k) / p_draft(k)), then resample a correction token.
This guarantees identical output distribution to greedy/temperature sampling.

Alpha (acceptance rate) is tracked as an exponential moving average and logged as
a Prometheus metric (zerfoo_speculative_acceptance_rate). If alpha < 0.4 for 100
consecutive steps, speculative decoding auto-disables to avoid overhead.

## Consequences
Positive:
- 2-3x throughput for single-sequence decode at batch=1 when alpha > 0.6
- Output distribution is mathematically identical to non-speculative decoding
- Self-speculative mode works with any model; no separate draft download required

Negative:
- Memory cost: ExternalDraft mode loads two models; increases GPU memory usage
- Complexity: Speculative sampling requires careful token-by-token probability
  comparison; numerical precision issues with FP16 probabilities require FP32
  accumulation
- Low alpha workloads (code generation, structured output) may see neutral or
  negative throughput impact
- KV cache synchronization between draft and target adds complexity
