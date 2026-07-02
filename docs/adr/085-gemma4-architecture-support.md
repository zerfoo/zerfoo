# ADR 085: Gemma 4 Architecture Support

## Status

Accepted

## Date

2026-04-13

## Context

Gemma 4 is Google's latest open model family (Apache-2.0, released 2026-03-02).
It introduces several new features over Gemma 3:

1. **MoE variant** (26B-A4B): 128 experts, top-8 routing, 3.8B active params.
2. **Hybrid attention**: Interleaved sliding-window + global attention with
   different KV head counts and head dimensions per layer type.
3. **Dual RoPE**: Standard RoPE (theta=10K) for sliding layers, proportional
   RoPE (theta=1M, partial_rotary_factor=0.25) for global layers.
4. **Unified K=V**: Global attention layers share K and V projections (31B, 26B).
5. **Per-Layer Embeddings (PLE)**: Edge variants (E2B, E4B) use per-layer input
   embeddings for parameter-efficient layer specialization.
6. **KV-shared layers**: Edge variants share KV projections across groups of layers.
7. **GELU activation** (gelu_pytorch_tanh) instead of Gemma 3's SwiGLU.
8. **256K context** (31B/26B), 128K (E variants).
9. **262K vocabulary** (up from 256K in Gemma 3).
10. **Vision encoder** (SigLIP-style) on all variants; audio encoder on E variants.

Four model variants: 31B (dense), 26B-A4B (MoE), E4B (edge+audio), E2B (edge+audio).

GGUF conversions are widely available from Unsloth, LM Studio, and bartowski.

## Decision

Implement Gemma 4 support in three phases:

**Phase 1 (text-only, dense):** Build `arch_gemma4.go` for the 31B dense variant.
This covers: hybrid attention with per-layer KV head counts and head dims, dual
RoPE configuration, K=V in global layers, GELU activation, 4 norms per layer,
logit softcapping, and tied embeddings. Extend `transformerGraphOpts` with fields
for per-layer attention configuration rather than single global values. Register
as "gemma4" in the architecture registry.

**Phase 2 (MoE):** Add MoE FFN routing for the 26B-A4B variant. Reuse the
existing `core.MixtureOfExperts` and `core.MoEGate` (proven in DeepSeek, Mixtral,
DBRX, Kimi, LFM2). The per-layer FFN becomes conditional: dense SwiGLU for layers
that are not MoE, `MixtureOfExperts` for MoE layers. Register as "gemma4moe".

**Phase 3 (edge):** Add PLE and KV-shared layer support for E4B/E2B. Register
as "gemma4e". Vision and audio encoders are deferred (multimodal is out of scope
for this epic).

## Consequences

**Positive:**
- Gemma 4 is the most-downloaded open model family on HuggingFace; supporting it
  keeps Zerfoo competitive with llama.cpp and Ollama.
- The dual RoPE and hybrid attention patterns are reusable for future architectures
  (Llama 4, Mistral variants).
- MoE reuse validates the existing `MixtureOfExperts` layer at 128 experts (largest
  expert count in any supported model).

**Negative:**
- Per-layer varying KV heads and head dims requires breaking the assumption in
  `buildTransformerGraph` that all layers share the same `cfg.NumKVHeads` and
  `headDim`. This will be handled in the Gemma 4 builder rather than modifying
  the shared function, keeping other architectures unchanged.
- The K=V optimization (shared K/V projection) is a new attention variant that
  may need a GQA extension or a thin wrapper.
- Edge variants (PLE, KV-shared layers) are architecturally novel and may need
  new abstractions.

**Deferred:**
- Vision encoder (SigLIP): deferred to a future multimodal epic.
- Audio encoder: deferred to a future multimodal epic.
- Video frame extraction: deferred.
