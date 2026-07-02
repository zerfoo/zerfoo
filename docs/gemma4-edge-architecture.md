# Gemma 4 Edge (E2B/E4B) Architecture Reference

Source of truth: tensor/metadata layout of `unsloth/gemma-4-E2B-it-GGUF`
(Q4_K_M, 35 layers, hidden=1536). Shapes and names verified by direct
inspection of the GGUF on 2026-04-13.

This document catalogues every tensor observed in a real Gemma 4 edge GGUF
and its inferred role in the forward pass. It exists to guide the rewrite
of `inference/arch_gemma4_edge.go` (task T93.3.1) so that zerfoo can load
and run canonical Gemma 4 edge models produced by llama.cpp's converter.

## Model scale (observed for E2B)

- Hidden size: 1536
- Num layers: 35
- Num query heads: 8 (`attn_q` = [1536, 2048] = [hidden, 8 * 256])
- Num KV heads: 1 (`attn_k` = [1536, 256])
- Head dim: 256 (K = V, verified by equal shapes)
- Intermediate (FFN) size: 6144
- Vocabulary: 262144
- PLE slice dim: 256 (per-layer embedding width)
- PLE table width: 8960 = 35 * 256 (one slice per layer, packed contiguously)

## Metadata keys (canonical, llama.cpp-style)

All keys observed on unsloth/gemma-4-E2B-it-Q4_K_M.gguf. These are the keys
zerfoo's extractor must consume:

```
gemma4.attention.head_count
gemma4.attention.head_count_kv
gemma4.attention.key_length
gemma4.attention.key_length_swa
gemma4.attention.layer_norm_rms_epsilon
gemma4.attention.shared_kv_layers
gemma4.attention.sliding_window
gemma4.attention.sliding_window_pattern
gemma4.attention.value_length
gemma4.attention.value_length_swa
gemma4.block_count
gemma4.context_length
gemma4.embedding_length
gemma4.embedding_length_per_layer_input
gemma4.feed_forward_length
gemma4.final_logit_softcapping
gemma4.rope.dimension_count
gemma4.rope.dimension_count_swa
gemma4.rope.freq_base
gemma4.rope.freq_base_swa
```

Extractor handling of these keys landed in commit `8213a7e6` (2026-04-13).

## Tensor inventory (canonical GGUF names and shapes)

All `[rows, cols]` shapes are as reported by the GGUF tensor header.
Recall that GGUF stores matrices transposed relative to HuggingFace
convention: for a weight W that computes `out = x @ W.T` in HF, GGUF stores
shape `[in_features, out_features]`. For embedding tables, GGUF shape is
`[hidden, vocab]`.

### Global tensors

| Tensor | Shape | Role |
|--------|-------|------|
| `token_embd.weight` | `[1536, 262144]` | Main token embedding. Gemma 3/4 tie this with the output head. |
| `output_norm.weight` | `[1536]` | Final RMSNorm before the LM head. |
| `rope_freqs.weight` | `[256]` | Precomputed RoPE frequencies (head_dim = 256). |
| `per_layer_token_embd.weight` | `[8960, 262144]` | **PLE**: shared per-layer embedding table. 262144 vocab entries, each a concatenation of 35 * 256 = 8960 per-layer slices. |
| `per_layer_model_proj.weight` | `[1536, 8960]` | Projects the model hidden state (1536) to the full PLE width (8960 = num_layers * ple_dim). Used to produce per-layer features from the current hidden. |
| `per_layer_proj_norm.weight` | `[256]` | RMSNorm applied to a single per-layer PLE slice (dim 256). |

### Per-block tensors (observed once per layer; N = 0..34 for E2B)

| Tensor | Shape | Role |
|--------|-------|------|
| `blk.N.attn_norm.weight` | `[1536]` | Input RMSNorm (pre-attention). |
| `blk.N.attn_q.weight` | `[1536, 2048]` | Query projection (8 heads * 256). |
| `blk.N.attn_k.weight` | `[1536, 256]` | Key projection (1 KV head * 256). K = V shape. |
| `blk.N.attn_v.weight` | `[1536, 256]` | Value projection. |
| `blk.N.attn_q_norm.weight` | `[256]` | QK-norm on queries. |
| `blk.N.attn_k_norm.weight` | `[256]` | QK-norm on keys. |
| `blk.N.attn_output.weight` | `[2048, 1536]` | Output projection (fuses 8 heads back to hidden). |
| `blk.N.post_attention_norm.weight` | `[1536]` | Post-attention RMSNorm applied to the attention output before residual add. |
| `blk.N.ffn_norm.weight` | `[1536]` | Pre-FFN RMSNorm. |
| `blk.N.ffn_gate.weight` | `[1536, 6144]` | SwiGLU/GELU gate projection. |
| `blk.N.ffn_up.weight` | `[1536, 6144]` | SwiGLU/GELU up projection. |
| `blk.N.ffn_down.weight` | `[6144, 1536]` | FFN down projection. |
| `blk.N.post_ffw_norm.weight` | `[1536]` | Post-FFN RMSNorm applied to the FFN output before residual add. |
| `blk.N.post_norm.weight` | `[1536]` | Additional norm applied to the block residual (position within the block inferred; confirm against llama.cpp). |
| `blk.N.inp_gate.weight` | `[1536, 256]` | Gating projection from current hidden (1536) to PLE slice dim (256). Modulates the per-layer PLE contribution before injection into the residual stream. |
| `blk.N.proj.weight` | `[256, 1536]` | Per-layer PLE projection. Takes the gated 256-dim per-layer PLE vector and projects to hidden (1536) for residual injection. |
| `blk.N.layer_output_scale.weight` | `[1]` | Per-layer scalar scale applied to the block output before feeding the next block (or before the final norm). |

## Inferred per-block forward pass

The block wiring below is reconstructed from tensor names, shapes, and the
Gemma 3/Gemma 4 design notes in the Google tech reports. The final order
MUST be confirmed against llama.cpp's `build_gemma4` (or equivalent) graph
builder during T93.3.1. Flagged points need verification (V).

```
Inputs per token t at layer N:
  h      : [seq, 1536]    # running residual
  tok_id : [seq]           # integer token id

1) Pre-attention norm:
     a0 = attn_norm(h)

2) QKV projections with QK-norm:
     q = q_norm(attn_q(a0))     # [seq, heads=8, 256]
     k = k_norm(attn_k(a0))     # [seq, 1, 256]   (K = V in shape)
     v =       attn_v(a0)       # [seq, 1, 256]

3) Hybrid attention (alternating global / sliding; pattern from
   attention.sliding_window_pattern = 5 for E2B):
     attn_out = attention(q, k, v, rope_global or rope_swa per layer)
     o = attn_output(attn_out)  # [seq, 1536]

4) Post-attention norm + residual:
     h = h + post_attention_norm(o)

5) Per-layer embedding injection (PLE):
     ple_all   = per_layer_token_embd(tok_id)   # [seq, 8960]
     ple_slice = ple_all[:, N*256 : (N+1)*256]  # slice this layer
     ple_n     = per_layer_proj_norm(ple_slice) # RMSNorm dim=256
     gate      = inp_gate(h)                    # [seq, 256]
     injected  = proj(ple_n * gate)             # [seq, 1536]    # (V) exact combiner
     h         = h + injected

   NOTE: the relationship between `per_layer_model_proj` (hidden -> 8960)
   and the packed PLE table is not fully determined from shapes alone.
   Three plausible uses to verify against llama.cpp:
     (a) per_layer_model_proj replaces or augments the raw PLE slice at
         each layer (a per-layer computed PLE, not just a lookup).
     (b) per_layer_model_proj produces a residual that is added to the
         PLE slice before gating.
     (c) per_layer_model_proj is used only during the *first* layer or as
         a shared initial projection into the PLE space.

6) Pre-FFN norm:
     f0 = ffn_norm(h)

7) FFN (Gemma 4 uses GELU, not SwiGLU silu -- confirm T92.2.2 field):
     ff = ffn_down(gelu(ffn_gate(f0)) * ffn_up(f0))
     h  = h + post_ffw_norm(ff)

8) Block output shaping:
     h = post_norm(h) * layer_output_scale   # (V) order

Output (after final block):
     logits = lm_head(output_norm(h))        # lm_head shares token_embd
     logits = softcap(logits, final_logit_softcapping)
```

## KV cache sharing (edge only)

`gemma4.attention.shared_kv_layers` = 10 for E2B indicates that the last N
layers reuse KV projections from earlier layers (details to be confirmed
against llama.cpp). The existing `KVSharedLayers` ModelConfig field carries
this value; the builder must skip recomputing K/V for shared-block layers
and read from the shared cache instead.

## Zerfoo-canonical tensor names (target of the rename mapper, T93.2.1)

GGUF `blk.N.*` tensors are already mapped to `model.layers.N.*` by the
existing name mapper. The Gemma 4 edge additions are:

| GGUF name | Zerfoo-canonical name |
|-----------|------------------------|
| `per_layer_token_embd.weight` | `model.ple_embed_tokens.weight` |
| `per_layer_model_proj.weight` | `model.ple_model_proj.weight` |
| `per_layer_proj_norm.weight` | `model.ple_proj_norm.weight` |
| `blk.N.proj.weight` | `model.layers.N.ple_layer_proj.weight` |
| `blk.N.inp_gate.weight` | `model.layers.N.input_gate.weight` |
| `blk.N.layer_output_scale.weight` | `model.layers.N.layer_output_scale.weight` |
| `blk.N.post_attention_norm.weight` | `model.layers.N.post_attention_layernorm.weight` |
| `blk.N.post_ffw_norm.weight` | `model.layers.N.post_ffw_layernorm.weight` |
| `blk.N.post_norm.weight` | `model.layers.N.post_layernorm.weight` |

## Open items to verify against llama.cpp during T93.3.1

1. Exact combiner for `ple_slice * gate` vs. the output of
   `per_layer_model_proj`. See PLE step 5 note above.
2. Position of `post_norm` and `layer_output_scale` in the block: is the
   scale applied pre-norm, post-norm, or fused into the residual? Affects
   numerical output even if shapes align.
3. Whether `rope_freqs.weight` is consumed at runtime or is metadata-only
   (the existing RoPE implementation derives frequencies from freq_base).
4. Exact semantics of `shared_kv_layers` -- which layers share and how
   the cache indices map.
5. FFN activation: Gemma 4 dense uses GELU. Confirm the edge variant does
   too (the tech report implies yes; the existing `UseGELUFFN` config field
   from T92.2.2 should cover this).

These items are captured as risks in the E93 plan and are the reason
T93.1.2 (ADR-086) locks the sharing model first, before T93.3.1 attempts
the builder rewrite.
