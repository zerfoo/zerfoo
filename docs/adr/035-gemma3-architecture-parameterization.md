# ADR 035: Gemma 3 Architecture Parameterization

## Status
Accepted

## Date
2026-03-11

## Context

The `buildTransformerGraph` function in `inference/arch_common.go` hardcodes
Llama-derived defaults for four parameters that differ in Gemma 3:

1. **head_dim**: Computed as `cfg.HiddenSize / cfg.NumHeads` (1152/4 = 288),
   but Gemma 3's config.json specifies `head_dim: 256` explicitly.
2. **RMSNorm epsilon**: Hardcoded as `1e-5`, but Gemma 3 uses `1e-6`.
3. **FFN activation**: Always uses `core.WithSwiGLU` (SiLU gating), but
   Gemma 3 uses `gelu_pytorch_tanh` (tanh-approximated GELU gating).
4. **Attention scaling**: Uses default `1/sqrt(head_dim)`, but Gemma 3
   specifies `query_pre_attn_scalar: 256`.

These mismatches cause degenerate (repetitive, nonsensical) output from both
CPU and CUDA inference paths. The model loads successfully but computes
incorrect hidden states at every layer.

## Decision

Extend `transformerGraphOpts` and `gguf.ModelConfig` to carry these four
parameters explicitly, with backward-compatible defaults that preserve
existing Llama behavior:

1. **`gguf.ModelConfig`**: Add `HeadDim int`, `RMSNormEps float64`,
   `HiddenActivation string`, `QueryPreAttnScalar float64`. Parse from GGUF
   metadata keys `attention.key_length`, `attention.layer_norm_rms_epsilon`,
   `hidden_activation`, `attention.query_pre_attn_scalar`. Zero values fall
   back to current defaults.

2. **`transformerGraphOpts`**: Add `headDim int`, `rmsNormEps float64`,
   `activation string`, `attnScale float64`. These override ModelConfig values
   when set, allowing architecture builders to override GGUF metadata.

3. **`buildTransformerGraph`**: Resolve each parameter from opts, then cfg,
   then hardcoded default. Use resolved headDim when constructing GQA and
   RoPE. Switch on activation to choose between SwiGLU and a new GeLU-gate
   FFN variant (using the existing `FastGelu` layer).

4. **`NewGroupedQueryAttentionFromParams`**: Accept explicit headDim instead
   of always computing `modelDim / numQueryHeads`.

5. **`buildGemmaGraph`**: Populate opts from cfg fields.

No new packages, interfaces, or dependencies. The existing `FastGelu` layer
(`layers/activations/fast_gelu.go`) already implements the exact
`gelu_pytorch_tanh` formula: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

## Consequences

**Positive:**
- Fixes degenerate inference output for Gemma 3 models loaded from GGUF.
- Backward compatible: Llama and other architectures unaffected (zero-value
  defaults preserve current behavior).
- Minimal surface area: 4 files, approximately 60 lines of net new code.
- No new dependencies or interfaces.

**Negative:**
- Does not address sliding window attention (`sliding_window_pattern: 6`).
  Output will be correct for short contexts but suboptimal for sequences
  exceeding the sliding window size. This is a separate feature.
- Does not address array-valued `eos_token_id` (generator concern, not
  architecture).
- The GeLU-gate FFN variant requires a new FFN option or a separate code
  path in the FFN Forward method.
