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
- **Open ambiguities.** Three wiring questions (PLE combiner with
  `per_layer_model_proj`, position of `post_norm` and `layer_output_scale`,
  exact `shared_kv_layers` semantics) cannot be resolved from tensor shapes
  alone and require reading llama.cpp's Gemma 4 graph builder during
  T93.3.1.
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
