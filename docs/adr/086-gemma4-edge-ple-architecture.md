# ADR 086: Adopt the Canonical Shared-PLE-Plus-Per-Layer-Proj Layout for Gemma 4 Edge

## Status
Accepted

## Date
2026-04-13

## Context

Zerfoo's initial Gemma 4 edge builder (`inference/arch_gemma4_edge.go`,
landed via Wave E92-3) was written against an invented tensor layout that
assumed each transformer block owned its own PLE embedding table
(`model.layers.N.ple_embedding.weight`). The canonical Gemma 4 edge GGUF
produced by llama.cpp's converter (and shipped by Google/unsloth) uses a
fundamentally different layout:

- A single shared `per_layer_token_embd.weight` of shape `[8960, 262144]`
  that packs 35 per-layer 256-dim slices into one vocab-sized lookup table.
- A global `per_layer_model_proj.weight` `[1536, 8960]` that projects the
  current model hidden state into the full per-layer feature space.
- A shared `per_layer_proj_norm.weight` `[256]` RMSNorm applied to the
  per-layer slice.
- Per-block `proj.weight` `[256, 1536]` that projects the (gated)
  per-layer slice back into the residual stream.
- Per-block `inp_gate.weight` `[1536, 256]` that gates the per-layer
  injection based on the current hidden state.
- Per-block `layer_output_scale.weight` `[1]` applied to the block output.
- Three additional per-block norms beyond what Gemma 3 required:
  `post_attention_norm.weight`, `post_ffw_norm.weight`, `post_norm.weight`.

See `docs/gemma4-edge-architecture.md` for the full tensor inventory and
inferred forward-pass wiring.

Integration test `TestGemma4E2B_EndToEnd` against a real unsloth GGUF
fails at graph build with "missing tensor: model.layers.0.ple_embedding.weight"
because the current builder looks for tensors that do not exist in the
canonical format (see `docs/devlog.md` 2026-04-13).

We had three viable options:

1. **Adopt the canonical layout.** Rewrite the builder to consume the
   shared PLE table, per-block projection, gating, and additional norms
   exactly as llama.cpp does.
2. **Post-process the GGUF at load time.** Split the shared PLE table into
   per-layer tables inside `LoadGGUF` so the existing builder keeps
   working unchanged.
3. **Require users to re-quantize with a zerfoo-specific converter.**
   Produce a zerfoo-flavored GGUF that matches the existing builder.

## Decision

Adopt option 1: rewrite `arch_gemma4_edge.go` to match the canonical
Gemma 4 edge architecture as it exists in llama.cpp and public Google
releases. The zerfoo-canonical tensor name mapping in
`inference/load_gguf.go` will translate the GGUF names to zerfoo-style
names (for example `per_layer_token_embd.weight` ->
`model.ple_embed_tokens.weight`), but the builder will consume one shared
PLE embedding plus per-layer projection, not per-layer embedding tables.

## Consequences

### Positive

- **Drop-in compatibility with public Gemma 4 GGUFs.** No user-facing
  conversion step, no zerfoo-specific variant to maintain.
- **Smaller memory footprint.** One 262144 * 8960 PLE table instead of 35
  separate 262144 * 256 tables. The shared layout is strictly smaller than
  the implied per-layer duplication and matches how Google trained the
  model.
- **Future Gemma 4 releases work automatically.** llama.cpp sets the
  canonical naming for the whole ecosystem; zerfoo inherits that
  compatibility for E4B and any future edge variants without per-release
  work.
- **Correct numerics.** The canonical layout is what the weights were
  trained against; using an invented layout risks subtle forward-pass
  divergence even if shapes align.

### Negative

- **Builder rewrite cost.** The existing `arch_gemma4_edge.go` is
  effectively scrapped. Synthetic fixtures in `inference/arch_gemma4_test.go`
  must also be rewritten. Budgeted in tasks T93.3.1 and T93.3.2.
- **Open ambiguities RESOLVED (2026-04-13) against HuggingFace
  `transformers/src/transformers/models/gemma4/modeling_gemma4.py`**
  (llama.cpp has no Gemma 4 builder yet — confirmed no `LLM_ARCH_GEMMA4`
  symbol, no gemma4 commits. HF transformers is the canonical reference.)

  1. **PLE combiner semantics** — `Gemma4TextDecoderLayer.forward` lines
     1401-1408: the PLE sub-block runs AFTER the attention+FFN residual as
     a third sub-block. Sequence: save residual, `per_layer_input_gate`
     (Linear 1536->256, GGUF `blk.N.inp_gate`), GELU, elementwise multiply
     with `per_layer_input` (dim 256), `per_layer_projection` (Linear
     256->1536, GGUF `blk.N.proj`), `post_per_layer_input_norm` (RMSNorm
     hidden-dim, GGUF `blk.N.post_norm`), add residual. The
     `per_layer_input` for layer i comes from
     `Gemma4TextModel.project_per_layer_inputs` (lines 1674-1696):
     `per_layer_projection = RMSNorm(reshape(per_layer_model_proj(embeds)
     * hidden_size**-0.5))`, combined with a token-identity PLE slice
     scaled by `sqrt(256)=16`, then multiplied by
     `per_layer_input_scale = 1/sqrt(2)`.

  2. **`post_norm` and `layer_output_scale` positions** — `post_norm`
     (hidden-dim 1536) is `post_per_layer_input_norm` and sits inside the
     PLE sub-block immediately before its residual add (line 1407). It is
     NOT a post-attention or post-FFN norm — those exist separately
     (`post_attention_layernorm` line 1377, `post_feedforward_layernorm`
     line 1398). `layer_output_scale` (`[1]` scalar) is `self.layer_scalar`
     (line 1337), a learned per-layer output multiplier applied at the
     very end of the decoder layer (line 1410: `hidden_states *=
     self.layer_scalar`) after all sub-blocks.

  3. **`shared_kv_layers=20` semantics** — lines 1149-1226: layers with
     `N >= first_kv_shared_layer_idx` (=20) are `is_kv_shared_layer=True`.
     HF does NOT instantiate `k_proj`, `v_proj`, `k_norm`, `v_norm` on
     these layers (line 1167 comment: "Layers sharing kv states don't
     need any weight matrices"). The forward path skips K/V projection
     and pulls from `shared_kv_states[self.kv_shared_layer_index]`, where
     `kv_shared_layer_index` is the last non-shared layer of the SAME
     `layer_type` (sliding vs global). `store_full_length_kv=True` on the
     donor layer stashes full-length K/V before sliding-window Cache
     truncates. Unsloth's per-layer K/V weights on layers 20-34 in the
     GGUF are therefore unused by the reference; the zerfoo builder
     should ignore them and route to the donor layer.

  4. **Double-wide MLP gating** (bonus finding, line 1021):
     `use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer`
     — MLP intermediate doubles (6144 -> 12288) on shared-KV layers only.
     The unsloth GGUF's `feed_forward_length` per-layer array shows 6144
     for 0-14 and 12288 for 15-34 (boundary at layer 15, not 20).
     Boundary mismatch between HF (20) and unsloth (15) must be verified
     against the actual per-layer tensor shapes during builder
     implementation; use the GGUF's explicit per-layer `ffn_up`/`ffn_gate`
     shapes as ground truth rather than the scalar config.
- **Divergence risk.** If llama.cpp changes its Gemma 4 layout in a future
  version, zerfoo must track it. Mitigation: integration test
  `TestGemma4E2B_EndToEnd` runs against a real GGUF and catches layout
  drift at test time rather than in user reports.

### Rejected alternatives

Option 2 (split at load time) was rejected because it duplicates the PLE
table 35x in memory and obscures the true architecture from every reader
of the codebase. Option 3 (zerfoo-specific GGUF variant) was rejected
because it creates a maintenance burden without technical justification
and breaks the drop-in model story that is central to zerfoo's value
proposition.
