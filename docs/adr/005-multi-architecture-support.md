# ADR-005: Multi-Architecture Support

**Status:** Accepted
**Phase:** 9
**Date:** 2026-03-02

## Context

After Phase 8, only Gemma 3 was validated. The open-weights ecosystem uses a
small set of architectural building blocks in different combinations per model
family. Gap analysis of major families identified a tier system based on
implementation effort.

## Decision

### Tier System

| Tier | Models | Gap | Status |
|------|--------|-----|--------|
| 1 | Llama 3/4, Mistral, Mixtral | Config mapping only | Complete |
| 2 | Qwen 2.5/3 | QKV bias, YaRN RoPE | Complete |
| 3 | Phi-4 | Partial RoPE, tied embeddings | Complete |
| 4 | DeepSeek V3/R1 | MLA, shared expert MoE | Complete |
| 5 | Falcon Mamba, RWKV, DeltaNet | Different paradigm (SSM/linear attn) | Out of scope |

### Config Registry

`inference.ConfigRegistry` maps model_type strings (from HuggingFace config.json)
to parser functions. Each normalizes model-specific JSON fields into unified
ModelMetadata. Registered parsers: gemma2, gemma3, llama, mistral, qwen2, phi,
deepseek. Fallback parser for unknown model_type attempts direct unmarshal.

### Parameter Name Resolver

`model.ParamResolver` maps architecture-specific weight names to canonical names:
- Llama/Gemma: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
- Qwen: same + .bias suffix
- Phi: self_attn.{q,k,v,dense}_proj.weight
- DeepSeek: self_attn.{kv_a_proj,kv_b_proj,q_a_proj,q_b_proj,o_proj}.weight
- FFN: model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.weight (all)

Called during model building as fallback when exact name not found.

### QKV Bias (Tier 2 -- Qwen)

Optional bias in GQA handled by existing Dense layer (core.NewDenseFromParams
accepts *Bias). Registry builder uses optionalBias helper to look up bias params.
Backwards compatible (nil bias = no change).

### YaRN RoPE Scaling (Tier 2 -- Qwen)

`embeddings.WithYaRNScaling(factor, origMaxLen)` modifies inverse frequencies:
- Low-frequency components scaled by 1/factor
- High-frequency components kept unchanged
- Intermediate linearly interpolated
- Attention scaling factor: sqrt(1 + ln(factor) / ln(origMaxLen))

Construction-time change; no forward-pass performance impact.

### Partial RoPE (Tier 3 -- Phi-4)

`embeddings.WithRotaryDimFraction(fraction)` rotates only a fraction of head
dimensions. Default 1.0 (all rotated). Phi-4 uses 0.75. Forward splits input
into rotated/non-rotated portions along last dimension, applies RoPE to rotated
part, concatenates.

### Tied Embeddings (Tier 3 -- Phi-4)

`core.NewTiedLMHead(engine, tiedWeight)` reuses token embedding weight
(transposed) as output projection. Builder checks tie_word_embeddings config
flag. Existing untied LMHead unchanged.

### Multi-head Latent Attention (Tier 4 -- DeepSeek)

New `attention.MultiHeadLatentAttention[T]` layer (not a GQA modification):
1. Down-project: c_kv = x * W_DKV (compress to kv_lora_dim, e.g., 512)
2. Up-project: K = c_kv * W_UK, V = c_kv * W_UV
3. Q = x * W_Q
4. Decoupled RoPE on Q and K subvector
5. Standard scaled dot-product attention
6. Output projection

KV cache stores full tensors (compressed latent caching deferred).

### Shared Expert MoE (Tier 4 -- DeepSeek)

MixtureOfExperts.SharedExpert field. When non-nil, Forward() runs shared expert
on every token and adds output to weighted routed sum. Builder loading of shared
expert weights deferred (requires ZMF sub-graph support).

### Global Attributes

Model-level attributes (rope_scaling, partial_rotary_factor, tie_word_embeddings)
injected via `model.WithGlobalAttributes` during graph construction so individual
layer builders can access them.

## Consequences

- 6 model families supported, covering >90% of HuggingFace open-weight downloads.
- Each tier is independently shippable.
- Parity tests for all 6 families (env-var gated).
- MLA KV cache compression and shared expert weight loading are documented tech debt.
- SSM/linear attention architectures (Tier 5) deferred to future phase.

### Key Files

- `inference/arch_config.go` -- Config registry and parsers
- `model/param_resolver.go` -- Architecture-specific parameter resolution
- `layers/embeddings/rotary_positional_embedding.go` -- YaRN + partial RoPE
- `layers/attention/multi_head_latent_attention.go` -- MLA
- `layers/attention/mla_registry.go` -- MLA builder
- `layers/core/lm_head.go` -- Tied embeddings
- `layers/core/moe.go` -- Shared expert support
- `tests/parity/{llama3,mistral,qwen,phi4,deepseek}_test.go` -- Parity tests
