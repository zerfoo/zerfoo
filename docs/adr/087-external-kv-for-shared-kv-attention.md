# ADR 087: External K/V Input Path for GroupedQueryAttention

## Status
Accepted

## Date
2026-04-13

## Context

Several transformer architectures share K/V tensors across layers to reduce
memory and compute. Gemma 4 edge (E2B) is the first such architecture we need
to support: HuggingFace transformers `modeling_gemma4.py` lines 1148-1226
specify that layers with `layer_idx >= first_kv_shared_layer_idx` skip their
K/V projection entirely and read K/V from a donor layer (the last non-shared
layer of the same attention type, sliding vs global).

The current `layers/attention/grouped_query_attention.go` has a hard-coded
forward path that always runs `wk.Forward(input)` and `wv.Forward(input)`
from the layer's own weights on the layer's own hidden state. It has no API
to:

- Skip K/V projection.
- Accept K/V from an external graph node.
- Expose a donor layer's K/V output so a downstream shared layer can read it.

We evaluated three options during E93-3 planning:

1. **Option A** -- donor weight reuse only. Swap `wk`/`wv` weights for the
   shared layer to point to the donor's weights, but still run projection
   on the shared layer's hidden state. Numerically incorrect vs HF
   reference (different hidden state, same weights), and special-cases GQA
   construction with no reusable primitive.
2. **Option B** -- external K/V input path (this ADR). Add a first-class
   API on GQA: `WithExternalKV(bool)` sets the mode, `Forward` accepts
   pre-computed K/V via extra input slots, donor layer exposes a K/V
   port node consumed by the shared layer.
3. **Option C** -- ignore sharing. Each shared layer uses its own shipped
   K/V weights on its own hidden state. Simple but hides the architecture
   and deviates from HF numerics. Any future architecture with shared KV
   has to re-decide this question.

## Decision

Adopt Option B. K/V sharing becomes a first-class graph concept:

- `GroupedQueryAttention` gains an "external K/V" mode. When enabled, the
  layer skips its own K/V projection (and the matching Q/K norms that
  apply to K) and reads K, V from additional graph inputs.
- `GroupedQueryAttention` always exposes its computed K/V as output ports
  that downstream nodes can read. A donor layer is any layer whose K/V
  is consumed by another layer; no special construction is required.
- A new graph node in `inference/` wires a donor layer's K/V output into
  a shared layer's external-K/V input slots. The builder resolves donor
  indices (last non-shared layer of the same attention type) at build
  time.
- The shared layer does not instantiate `wk`, `wv`, `k_norm` in
  external-K/V mode, matching HF's "don't need any weight matrices"
  invariant. Q/Q-norm/W_out remain layer-local.

Semantics of the `KVSharedLayers` config field: the value is a COUNT of
shared layers, not an index. `firstKVSharedIdx = NumLayers - KVSharedLayers`.
This matches Google's `config.json` and is consistent with the double-wide
MLP boundary observed in the real Gemma 4 E2B GGUF.

## Consequences

### Positive

- **First-class K/V sharing as a graph concept.** Any future architecture
  with shared K/V (Gemma 4 E4B, MLA variants, Mamba-hybrid transformers)
  uses the same API with no rework.
- **Numerically correct against HF transformers reference.** Enables the
  T93.4.3 parity test to meaningfully gate correctness.
- **No construction-time weight swapping.** The donor's K/V port is a
  normal graph edge; no special factory paths per layer.
- **Zero-stub builder.** The Gemma 4 edge builder can be a clean
  transcription of HF semantics without hiding the sharing behavior.

### Negative

- **Blast radius.** `layers/attention/grouped_query_attention.go` is used
  by Llama, Gemma 3, Mistral, Qwen 2, Phi, and DeepSeek builders. The
  external-K/V code path must be opt-in and default-off so none of those
  change behavior. Requires a non-regression test pass across the seven
  architectures.
- **Engineering cost.** Larger than Options A or C. Bounded: one file in
  `layers/attention`, one new graph node in `inference/`, new tests.
- **Graph complexity.** The donor→shared edge is a cross-layer
  dependency that compile passes and CUDA graph capture must handle.
  Existing compile infrastructure handles cross-layer edges
  (residuals already do), so expected to be small.

### Rejected alternatives

Option A mutates weights at construction time, producing a numerically
wrong layer and no reusable primitive. Option C is simpler but encodes a
numerical lie that later has to be unwound; it also forecloses correct
parity testing.

## Implementation notes

- API sketch (final API in code review):
  ```go
  type GroupedQueryAttentionOption func(*groupedQueryAttention)
  func WithExternalKV() GroupedQueryAttentionOption
  // Forward signature: inputs[0] = hidden, inputs[1] = k_external,
  // inputs[2] = v_external when WithExternalKV is set.
  ```
- Donor resolution is a builder-side concern. The GQA layer does not
  know about layer indices; it just accepts external K/V when told to.
- Existing builders remain unchanged (WithExternalKV defaults off).
  Non-regression tests pin their behavior.

## References

- HF transformers `src/transformers/models/gemma4/modeling_gemma4.py`
  lines 1148-1226 for is_kv_shared_layer logic and donor index
  resolution (same attention type, last non-shared).
- ADR-086 for the PLE sub-block wiring and related Gemma 4 edge
  decisions.
