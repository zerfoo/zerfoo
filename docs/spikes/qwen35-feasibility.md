# Spike: `qwen35` architecture support in Zerfoo

**Status:** analysis only — no implementation, no GPU time consumed.
**Date:** 2026-08-20
**Question:** what would it take for Zerfoo to run the GGUF architecture
string `qwen35` (Qwen3.5 / 3.6 / 3.8 family), specifically
**`Qwen/Qwen3.8-27B`**?
**Verdict up front:** a **multi-week epic**, not a weekend. Text-only decode
of Qwen3.8-27B is **14-22 engineer-days**; vision is a separate epic of
**15-25 days** on top. The cheap-looking part (the builder) really is cheap;
the expensive part is that Zerfoo has never once run a stateful recurrent
decode, and `qwen35` is 75% recurrent layers.

This document is CURRENT-STATE analysis. It does not amend `docs/plan.md`
and proposes no work item; it exists so the decision to fund (or not fund)
the epic can be made against evidence.

---

## 1. Ground truth: what `qwen35` actually is

Read directly from the GGUF headers of two published files (metadata + full
tensor list parsed from HTTP range requests; not from a blog post, not from
memory).

### `ggml-org/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-Q4_K_M.gguf`

```
general.architecture              = qwen35
general.size_label                = 27B          general.license = apache-2.0
qwen35.block_count                = 64
qwen35.embedding_length           = 5120
qwen35.feed_forward_length        = 17408
qwen35.attention.head_count       = 24
qwen35.attention.head_count_kv    = 4
qwen35.attention.key_length       = 256          qwen35.attention.value_length = 256
qwen35.attention.layer_norm_rms_epsilon = 1e-6
qwen35.rope.freq_base             = 1e7
qwen35.rope.dimension_count       = 64           -> partial rotary factor 64/256 = 0.25
qwen35.rope.dimension_sections    = [11, 11, 10, 0]   (mRoPE t/h/w sections)
qwen35.full_attention_interval    = 4
qwen35.ssm.conv_kernel            = 4
qwen35.ssm.state_size             = 128
qwen35.ssm.group_count            = 16
qwen35.ssm.time_step_rank         = 48
qwen35.ssm.inner_size             = 6144
qwen35.context_length             = 262144
tokenizer.ggml.model = gpt2   tokenizer.ggml.pre = qwen35   vocab 248320
general.file_type = 15 (Q4_K_M)   851 tensors
```

### The hybrid layout is a single metadata integer

`full_attention_interval = 4`. Observed: `blk.3` carries `attn_q/attn_k/attn_v`;
`blk.0/1/2/4` carry `ssm_*`. So:

```
isFullAttention(i)  ==  (i+1) % full_attention_interval == 0
```

64 layers = **48 Gated DeltaNet layers + 16 full-attention layers**. Tensor
count confirms the split exactly: `48*14 + 16*11 + 3 = 851`. There is no
per-layer list to parse and no tensor probing required — though probing
(the `arch_nemotron_h.go` pattern) is the more robust implementation.

### Tensor inventory (the whole architecture, in 23 names)

**Gated DeltaNet layer (48 of them):**

| tensor | 27B shape | 0.8B shape | role |
|---|---|---|---|
| `blk.N.attn_norm.weight` | [5120] | [1024] | pre-block RMSNorm |
| `blk.N.attn_qkv.weight` | [5120, 10240] | [1024, 6144] | fused q/k/v for the delta state |
| `blk.N.ssm_conv1d.weight` | [4, 10240] | [4, 6144] | depthwise causal conv over q\|k\|v |
| `blk.N.ssm_alpha.weight` | [5120, 48] | [1024, 16] | per-head decay (gate) projection |
| `blk.N.ssm_beta.weight` | [5120, 48] | [1024, 16] | per-head delta-rule write strength |
| `blk.N.ssm_a` | [48] | [16] | per-head A (decay base) |
| `blk.N.ssm_dt.bias` | [48] | [16] | per-head dt bias |
| `blk.N.ssm_norm.weight` | [128] | [128] | per-head RMSNorm over head_v_dim |
| `blk.N.attn_gate.weight` | [5120, 6144] | [1024, 2048] | output gate |
| `blk.N.ssm_out.weight` | [6144, 5120] | [2048, 1024] | output projection |
| `blk.N.post_attention_norm.weight` | [5120] | [1024] | pre-FFN RMSNorm |
| `blk.N.ffn_{gate,up,down}.weight` | 17408 | 3584 | SwiGLU |

The fused `attn_qkv` decomposes arithmetically and consistently across both
model sizes:

```
head_dim   = ssm.state_size   = 128
num_k_head = ssm.group_count  = 16
num_v_head = ssm.time_step_rank
27B:  q 16*128=2048 | k 16*128=2048 | v 48*128=6144  -> 10240  (= ssm.inner_size 6144 for v)
0.8B: q 16*128=2048 | k 16*128=2048 | v 16*128=2048  ->  6144  (= ssm.inner_size 2048 for v)
```

So the delta state is GQA-shaped: 16 key heads broadcast over 48 value heads
on the 27B.

**Full-attention layer (16 of them):** `attn_norm`, `attn_q [5120,12288]`,
`attn_k [5120,1024]`, `attn_v [5120,1024]`, `attn_q_norm [256]`,
`attn_k_norm [256]`, `attn_output [6144,5120]`, `post_attention_norm`, SwiGLU FFN.

Note `attn_q` is **12288 = 2 x (24 heads x 256)**. That is the
`attn_output_gate`: `q_proj` emits `[q | gate]` and the attention output is
gated by the second half before `o_proj`. `attn_k`/`attn_v` are 4 x 256 (GQA
24:4), and QK-RMSNorm is over the full head dim 256.

**Global:** `token_embd.weight [5120, 248320]`, `output_norm.weight`,
`output.weight [5120, 248320]`. The 27B is **untied**; the 0.8B has no
`output.weight` and is **tied**. Both cases must work.

**No vision tensors are in the main GGUF.** The tower ships entirely in
`mmproj-Qwen3.8-27B-*.gguf`, and the multi-token-prediction head entirely in
`mtp-Qwen3.8-27B-*.gguf`. This is load-bearing for the plan: the main file
*is* a pure text model.

---

## 2. Inventory: what Zerfoo already has that helps

### 2.1 The loader and registry cost is near zero

- **Metadata extraction is generic over the architecture prefix.**
  `model/gguf/arch.go:132-144` builds `prefix := arch + "."` and every
  extraction is `f.GetUint32(prefix + suffix)`. `qwen35.block_count`,
  `embedding_length`, `feed_forward_length`, `attention.head_count`,
  `head_count_kv`, `attention.key_length` (→ `cfg.HeadDim`),
  `attention.layer_norm_rms_epsilon`, `rope.freq_base`, `ssm.state_size`,
  `ssm.conv_kernel` all land in `ModelConfig` **with zero new code**.
- **`rope.dimension_count` already becomes `PartialRotaryFactor`**
  (`arch.go:208-216`): `64/256 = 0.25`, exactly the value `qwen35` needs, and
  `arch_common.go:183` already threads it into
  `embeddings.WithRotaryDimFraction`. Partial RoPE is **free**.
- **Registration is one line.** `RegisterArchitecture("qwen35", buildQwen35Graph)`
  in `inference/registry_init.go`, plus a `chatTemplateForArch` case in
  `inference/gguf.go:330-348` (skipping this is a silent footgun — an unknown
  template defaults to Gemma turn markers at `inference.go:631-634`).
- **Keeping GGUF tensor names is an established pattern.** `nemotronHTensorNameMap`
  (`model/gguf/arch.go:560-584`) is an identity map that stops `MapTensorName`
  from rewriting `blk.N.*` into HuggingFace names. A `qwen35TensorNameMap` is
  ~25 lines of the same.
- **Q4_K_M loads.** `Q4_K` (12), `Q6_K` (14), `Q8_0` (8) and `F32` (0) — the
  only four types in the 27B file — all have decode paths
  (`model/gguf/loader.go:224-259`). See the caveat in §5.

### 2.2 The hybrid-loop pattern exists twice, and is trivially reusable

- `inference/arch_jamba.go:68-75` dispatches on index arithmetic
  (`i % AttentionLayerOffset == 0`) inside its own layer loop at `:193-196`.
  Retargeting that to `(i+1) % 4 == 0` is a one-line change.
- `inference/arch_nemotron_h.go:100-121` dispatches on *tensor presence*
  (`tl.Has(prefix + "ssm_in.weight")`), driving a 4-way switch at `:260`. This
  is the better pattern for `qwen35` — it needs no metadata and cannot drift.

**Caveat:** neither reuses `arch_common.go`. `buildTransformerGraph`
(`arch_common.go:104`) has exactly one homogeneous loop (`:150`) and no
layer-type dispatch, so a hybrid builder re-implements its own tensor lookups,
transposes, GQA/RoPE/FFN construction. `arch_jamba.go` is 454 lines and
`arch_nemotron_h.go` is 849 lines for this reason. That is the real
boilerplate floor for `qwen35`, versus the **12 lines of body** that
`inference/arch_qwen3.go` needed (PR #989: 730 additions across 16 files, and
almost all of that is docs and tests).

### 2.3 SSM primitives that are structurally close

`layers/ssm/mimo_ssm.go` — `MIMOMambaBlock.headSelectiveScan` (`:451-513`) is
the closest existing analogue. Per head, per batch it carries a **matrix
state** `h[headDim * dState]` across timesteps and updates it as

```go
h[d*dState+n] = dA*h[d*dState+n] + dB*xVal      // decay + rank-1 write
yVal += cVal * h[d*dState+n]                    // read out
```

That is the same shape of loop Gated DeltaNet needs (decay + write + read on a
`d_k x d_v` state). Also present and reusable-in-spirit: the depthwise causal
conv1d and its backward (`layers/ssm/mamba_block.go:894`), SiLU/softplus
helpers, and `layers/ssm/bc_norm.go`.

`inference/arch_kimi.go:271-390` (`kimiLinearAttentionNode`) is a working
template for "bespoke linear-attention node written directly as a graph node":
`phi(Q)(phi(K)^T V)` with `phi = ELU+1`. It is forward-only (`Backward` at
`:391` is a stub) and prefill-only, but it proves the node shape.

### 2.4 The attention half is nearly free

The 16 full-attention layers are Qwen3 plus one thing. QK-RMSNorm over
`[headDim]`, GQA 24:4, decoupled head dim from `attention.key_length`,
no attention bias, SwiGLU, RMSNorm eps from metadata, tied-or-untied LM head —
all of that is exactly what PR #989 shipped and what `arch_common.go` already
does. The only genuinely new bit is the `attn_output_gate`.

### 2.5 A state container was already sketched

`generate/ssm_state.go` (72 lines) defines `SSMState[T]` with per-layer state,
`Reset`, `GetLayer`/`SetLayer`, `MemoryBytes`. **It has zero callers anywhere
in the repo** and is sized for Mamba-1 (`[1, d_inner, d_state]`), not a
per-head matrix state. It is a sketch, not a mechanism — but it means the
design intent is on record.

---

## 3. The gap, itemised

Each item marked **reuses-existing** / **needs-extension** / **genuinely-new**.

| # | Piece | Class | Notes |
|---|---|---|---|
| 1 | GGUF parse, mmap, gpt2 BPE tokenizer, Q4_K/Q6_K/Q8_0/F32 decode | reuses-existing | zero work |
| 2 | `qwen35.*` scalar metadata extraction | reuses-existing | generic prefix; free |
| 3 | `ssm.inner_size`, `ssm.group_count`, `ssm.time_step_rank`, `full_attention_interval` | needs-extension | 4 new `ModelConfig` fields + 4 extraction lines (`arch.go`). Trivial but not free — unknown keys are silently dropped today |
| 4 | `rope.dimension_sections` (array-typed metadata) | needs-extension | only needed for vision; scalar getters can't read a `uint32[4]` |
| 5 | `qwen35TensorNameMap` identity map + registry + chat template | reuses-existing (pattern) | ~40 lines total |
| 6 | Hybrid layer loop, 48 DeltaNet / 16 attention | reuses-existing (pattern) | copy `arch_nemotron_h.go`'s probe-and-switch; ~150 lines |
| 7 | Partial RoPE 0.25 on the attention layers | reuses-existing | `WithRotaryDimFraction`, already wired from `rope.dimension_count` |
| 8 | GQA + QK-RMSNorm(256) + SwiGLU + untied/tied LM head | reuses-existing | as PR #989 |
| 9 | **Gated attention output** (`attn_q` = `[q\|gate]`, gate applied pre-`o_proj`) | needs-extension | `buildTransformerGraph` has no hook for it; the hybrid loop hand-builds the attention block anyway, so it is ~1 day of glue, not a new op |
| 10 | Fused `attn_qkv` split into q(16x128) / k(16x128) / v(48x128) | needs-extension | slice + reshape glue |
| 11 | Depthwise causal conv1d over the concatenated q\|k\|v | needs-extension | logic exists inside `MambaBlock`; must be lifted out or re-written as a standalone node |
| 12 | **Gated DeltaNet recurrence (the delta rule)** | **genuinely-new** | see below — the core of the spike |
| 13 | **Per-head gated RMSNorm** (`ssm_norm [128]` x `silu(attn_gate)`) | needs-extension | RMSNorm exists; per-head application + gating is new glue |
| 14 | **Recurrent state cache across decode steps** (delta state + conv ring buffer) | **genuinely-new** | see §5 — the schedule risk |
| 15 | Interleaved mRoPE with sections `[11,11,10,0]` | **genuinely-new** — but **not needed for text-only** | see below |
| 16 | MTP head | **can be ignored entirely** | see below |
| 17 | mmproj vision tower, image injection, preprocessing | **genuinely-new, epic-sized** | see §6 |

### Item 12 — the delta rule, in detail

Gated DeltaNet keeps a matrix state per value head, `S ∈ R^{d_k x d_v}` =
`R^{128x128}`, and updates it with a *gated delta rule* rather than a pure
decay-and-add:

```
S_t = alpha_t * ( I - beta_t k_t k_t^T ) S_{t-1}  +  beta_t k_t v_t^T
o_t = S_t^T q_t
```

The `- beta_t k_t (k_t^T S_{t-1})` term is what makes this a *delta* rule
rather than the Mamba-style linear recurrence Zerfoo already has: the state's
existing content is read back and partially erased before the write. Concretely
that means the inner loop cannot stay a single fused pass the way
`headSelectiveScan` does — each timestep needs `k^T S` computed first
(a `d_k`-length matvec against the whole `128x128` state), then the rank-1
update, then the readout. `alpha_t` is derived from `ssm_alpha`, `ssm_a` and
`ssm_dt.bias`; `beta_t` from `ssm_beta`.

Estimated 250-400 lines with tests, modelled on `headSelectiveScan`. It is
tractable — but it is a new numerical kernel, and see §5 for why the *formula*
is the biggest unknown rather than the code.

### Item 15 — mRoPE is not on the critical path

`rope.dimension_sections = [11,11,10,0]` splits the 32 rotated dim-pairs
(64 dims of a 256-dim head) into temporal / height / width sections, applied
interleaved. Zerfoo has **no mRoPE at all** — a repo-wide search for
`mrope|rope_sections|3d rope` returns zero real hits.

**However:** with no image or video tokens in the sequence, all three position
components equal the sequence index, so mRoPE degenerates *exactly* to
ordinary 1D RoPE over the same 64 rotated dims. Partial RoPE at 0.25 is
already supported. **A text-only `qwen35` needs no mRoPE work.** This should
still be validated by first-token logit parity against llama.cpp rather than
assumed.

### Item 16 — MTP can be ignored, and this removes a whole workstream

Multi-token prediction is a *speculative decoding accelerator*. The MTP head
ships as a **separate file** (`mtp-Qwen3.8-27B-Q4_0.gguf`, 1.68 GB) and is
never referenced by plain autoregressive decode: you sample `argmax` over the
main model's logits and the MTP weights are simply not loaded. **Confirmed
ignorable.** If someone later wants the speedup, Zerfoo already has EAGLE
speculative infrastructure (`inference/eagle.go`, `generate/eagle_speculative.go`)
that the MTP head could be adapted into — a future optimisation, not a
prerequisite.

---

## 4. Phased plan

### M0 — Recon and golden reference — **1 day**

Pin `unsloth/Qwen3.5-0.8B-GGUF / Qwen3.5-0.8B-Q8_0.gguf` (812 MB, same
`qwen35` architecture, 24 layers = 18 DeltaNet + 6 attention) as the
development fixture. Capture per-layer activations and logits from llama.cpp
for a fixed prompt **before writing any builder code**.

*Delivers:* a golden reference and a size/hash-pinned fixture.
*Does not deliver:* any Zerfoo code.

### M1 — Text-only `qwen35` forward pass on the 0.8B — **6-9 days**

- `qwen35TensorNameMap`, `ModelConfig` fields, registry, chat template — 0.5 d
- Hybrid builder loop + gated-output attention layer — 2 d
- Gated DeltaNet node, **prefill only** (full sequence per forward, no state
  cache) — 3-4 d
- conv1d node, qkv split, per-head gated RMSNorm glue — 1 d
- Tests + layer-by-layer numeric comparison against the M0 reference — 1.5 d

*Delivers:* `inference.LoadFile("Qwen3.5-0.8B-Q8_0.gguf")` builds a graph and
produces logits that match llama.cpp layer-by-layer. Greedy decode works by
re-forwarding the whole prefix each step — correct, `O(n^2)`, unusably slow,
but provable.
*Does not deliver:* usable decode speed, vision, MTP, GPU, long context, the 27B.

### M2 — Recurrent state cache → linear-time decode — **5-8 days**

- State container (48 heads x 128 x 128 delta state + conv ring buffer,
  per layer, per sequence), lifecycle, reset — 3-5 d
- Single-token step path on the DeltaNet node + an equivalence test proving
  stepwise decode == full re-forward — 2-3 d

*Delivers:* linear-time decode at CPU speeds comparable to other Zerfoo
architectures. Replaces the dead `generate/ssm_state.go` with something real.
*Does not deliver:* batched/paged recurrent state, GPU kernels.

### M3 — Run Qwen3.8-27B Q4_K_M end to end — **2-4 days**

Download and provision (19 GB), exercise the untied-`output.weight` path,
memory and throughput shakeout, investigate the Q4_K→Q4_0 requantization
effect on 27B logits, add the `verified-models.md` row with a real parity test
and a benchmark manifest.

*Delivers:* **David's actual ask — Qwen3.8-27B running in Zerfoo as a text
model**, with evidence that meets the ADR-093 gate.
*Does not deliver:* image input.

**Subtotal M0-M3: 14-22 engineer-days (~3-4.5 weeks for one engineer).**

### M4 — Make it fast — **5-10 days, genuinely open-ended**

The DeltaNet scan as written in M1/M2 is a scalar Go loop. At 48 layers x 48
heads x 128 x 128 on a 27B model this will be slow enough that "it runs" and
"it is usable" are different claims. Reaching usable throughput needs an engine
op or a CUDA kernel, plus chunked (rather than strictly sequential) prefill.
Budget this separately and do not fold it into the M3 estimate.

### M5 — Vision — **15-25 days. Out of scope; its own epic.**

See §6. Do not attach this to the `qwen35` text milestone.

---

## 5. Risks and unknowns

### Biggest technical risk: the recurrent state cache (M2)

Zerfoo's entire inference stack is KV-cache-shaped. `generate/` contains
sixteen files of KV cache machinery (paged, quantized, tiered, radix, prefix,
GPU) and **one dead 72-line file** for SSM state. Nothing threads mutable
per-sequence recurrent state through `graph.Forward`.

The honest reading of the "we already have `mamba`/`mamba3`/`jamba`" lead:
those builders are **structurally real and empirically unproven**.
`docs/verified-models.md` — this repo's own honesty gate — has **no row at
all** for `mamba`, `mamba3`, `jamba`, `nemotron_h` or `rwkv`. Stronger
evidence still: `arch_jamba.go:421-428` looks up `ssm_in_proj.weight`,
`ssm_x_proj.weight`, `ssm_A_log`, `ssm_dt_proj.weight` — those are
HuggingFace/safetensors names. No llama.cpp GGUF emits them (llama.cpp uses
`ssm_in.weight`, `ssm_dt.weight`, `ssm_a`, `ssm_d`, `ssm_out.weight`, as
`arch_nemotron_h.go` correctly expects). **The Jamba builder has almost
certainly never been run against a real GGUF file.** Treat it as a design
template, not as working infrastructure.

M2 is therefore first-of-its-kind work in this codebase, and it is where the
schedule can double.

### Biggest unknown: the exact Gated DeltaNet numerics

The GGUF metadata pins every *shape* but none of the *formulas*. Unresolved
from in-repo evidence alone:

- the exact `alpha_t` construction from `ssm_alpha` / `ssm_a` / `ssm_dt.bias`
  (softplus? sigmoid? `exp(-softplus(x) * exp(A))`?)
- whether `q`/`k` are L2-normalised before the delta update (Gated DeltaNet
  variants differ)
- the ordering of conv1d, normalization and gating within the block
- how the 16 key heads broadcast over the 48 value heads on the 27B

These must be read off llama.cpp's `qwen35` graph builder or the HF modeling
code. **Getting them subtly wrong produces finite, plausible-looking logits
that are simply wrong** — the hardest failure mode to detect. This is exactly
why M0 (capture a golden per-layer reference first) is a separate milestone
and not an afterthought.

### Secondary risks

- **Q4_K is re-quantized to Q4_0 at load.** `model/gguf/loader.go:285-305`
  round-trips every `Q4_K`/`Q5_K`/`Q6_K` tensor through f32 into `Q4_0`,
  collapsing 6-bit sub-block scales to 16 flat levels. The code comments at
  `:67-81` already name this "doubly-lossy" and "the top structural candidate"
  for an existing decode-degeneracy bug. Mitigation: do all correctness work on
  the 0.8B at **Q8_0**, which is retained natively, and treat the 27B Q4_K_M
  result as a separate quantization question.
- **Bit-exact decode is not currently achievable in Zerfoo for any
  architecture.** `docs/verified-models.md` records that `qwen3` greedy decode
  is not bit-exact vs llama.cpp (6/8 first-token match), and that `gemma3`
  reproduces the identical divergence on the same harness — a pre-existing
  decode issue, not a builder defect. **Do not gate the `qwen35` spike on
  bit-exact decode.** Gate M1 on layer-wise forward-pass agreement instead.
- **Speed.** 48 sequential scan layers in scalar Go on a 27B model. M4 exists
  because of this.
- **`ModelConfig` accretion.** ~60 typed fields across 9 architecture-specific
  blocks already (`model/gguf/arch.go:11-93`); `qwen35` adds four more to a
  shared struct. Cosmetic, but it is the direction of travel.

---

## 6. Why vision is a separate epic, not a phase

An inventory of the existing vision path found it to be **two disconnected
halves that never meet**, with no GGUF path at all:

- **No `mmproj-*.gguf` loading exists.** `inference/load_gguf.go:17` opens
  exactly one file. There is no projector-path option anywhere. The one
  function that opens a second GGUF
  (`inference/multimodal/gguf_loader.go:37`) reads **metadata only, no
  tensors**, and has no non-test caller.
- **The expected tensor names are invented.** `arch_llava.go:93-121` and
  `arch_qwenvl.go:91-120` look up `vision.patch_embed.weight`,
  `vision.blocks.N.attn.q_proj.weight`, `mm_projector.0.weight`. A real
  llama.cpp mmproj uses `v.patch_embd.weight`, `v.blk.N.attn_q.weight`,
  `mm.0.weight`. `MapTensorName` (`model/gguf/arch.go:713`) has no `v.`/`mm.`
  handling. Every lookup would fail.
- **The LLaVA/QwenVL graphs have no text input.** `arch_llava.go:237` sets
  `hidden := projectedVision` — the decoder consumes *only* image tokens.
  `inference.LoadFile("llava.gguf").Generate(...)` cannot succeed: the
  generator feeds `[1, seqLen]` token IDs into a graph whose first node
  rejects non-4D input (`layers/vision/clip_encoder.go:222-225`).
- **`inference.Message.Images` is dead** (`inference/inference.go:577`) — never
  read. The server fetches and size-caps images, then discards them.
- **No mRoPE, no dynamic resolution, no window attention, no pos-embed
  interpolation.** Preprocessing is fixed-square bilinear resize, and its patch
  layout (interleaved) is incompatible with `CLIPEncoder`'s (planar).
- **Zero real-GGUF test coverage.** Every green vision test uses synthetic
  tensors keyed by the repo's own invented names. The two tests that would
  touch a real file `t.Skip` unless an env var is set, and would fail if run.

Adding `qwen35` vision therefore means building the mmproj loading path,
the `v.*`/`mm.*` name mapping, a native (non-adapter) tower, real mRoPE,
image-token splicing into the decode sequence, dynamic-resolution
preprocessing, and the wiring from `Message.Images` to the graph — most of
which does not exist for *any* architecture. **15-25 days, and it fixes
LLaVA and Qwen-VL as a side effect.** It should be scoped and funded as its
own epic.

---

## 7. Practical: files, sizes, download cost

### Test artifacts (smallest first)

The decisive practical finding: **you do not need the 27B to develop this.**
The `qwen35` family publishes sub-gigabyte members with an identical
architecture — same `full_attention_interval = 4`, same
`ssm.state_size = 128`, same `group_count = 16`, same 262K context, same
`rope.dimension_sections`.

| Artifact | Size | Notes |
|---|---|---|
| `unsloth/Qwen3.5-0.8B-GGUF` `Q4_K_M` | 533 MB | 24 layers (18 DeltaNet + 6 attn), hidden 1024, tied LM head |
| `unsloth/Qwen3.5-0.8B-GGUF` **`Q8_0`** | **812 MB** | **recommended dev/CI fixture** — Q8_0 is retained natively, no Q4_K→Q4_0 round trip |
| `unsloth/Qwen3.5-0.8B-GGUF` `BF16` | 1.52 GB | for exact-reference work |
| `unsloth/Qwen3.5-2B-GGUF` `Q8_0` | 2.01 GB | intermediate scale check |
| `bartowski/Qwen_Qwen3.5-0.8B-GGUF` | 399 MB - 708 MB | alternative quant ladder incl. Q2_K/IQ |

### `ggml-org/Qwen3.8-27B-GGUF` (the actual ask)

| File | Bytes | Size |
|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 18,973,870,432 | **17.7 GiB / 18.97 GB** |
| `Qwen3.8-27B-Q8_0.gguf` | 28,595,763,552 | 26.6 GiB |
| `Qwen3.8-27B-BF16.gguf` | 53,808,281,952 | 50.1 GiB |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629,247,008 | 600 MiB (vision only) |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 931,145,888 | 888 MiB (vision only) |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1,680,271,648 | 1.56 GiB (**not needed**) |

**ggml-org publishes no quant below Q4_K_M for the 27B.** Q4_K_M at 17.7 GiB
is the smallest usable copy of *this* model — which is precisely why the 0.8B
is the development fixture.

### Download cost for the 27B Q4_K_M (18.97 GB)

Measured from the mini to HuggingFace on 2026-08-20: **10.1 MB/s** sustained
over a 200 MB range request → **~31 minutes** for the full file.

| Link speed | Time for 18.97 GB |
|---|---|
| 5 Mbps | ~8 h 25 m |
| 10 Mbps | ~4 h 13 m |
| 25 Mbps | ~1 h 41 m |
| 50 Mbps | ~51 m |
| 100 Mbps | ~25 m |
| measured 81 Mbps (10.1 MB/s) | ~31 m |

HuggingFace serves HTTP `Range`, so the download is resumable (`curl -C -`,
or `hf download` which chunks and resumes by default). On a genuinely slow
link, start it detached and do M0-M2 against the 0.8B in the meantime — the
27B is not needed until M3.

---

## 8. Bottom line

| Milestone | Days | Delivers |
|---|---|---|
| M0 recon + golden reference | 1 | pinned 0.8B fixture, llama.cpp per-layer reference |
| M1 text-only forward pass (0.8B) | 6-9 | graph builds, logits match layer-by-layer; decode correct but `O(n^2)` |
| M2 recurrent state cache | 5-8 | linear-time decode |
| M3 Qwen3.8-27B Q4_K_M end to end | 2-4 | **the ask, as a text model**, with verified-models evidence |
| **M0-M3 total** | **14-22** | **Qwen3.8-27B running as a text model** |
| M4 performance | 5-10+ | usable throughput (engine op / CUDA kernel) |
| M5 vision | 15-25 | separate epic; fixes LLaVA + Qwen-VL too |

**What already exists that helps:** generic arch-prefixed metadata extraction
(free), partial RoPE at exactly 0.25 (free), the whole GQA/QK-norm/SwiGLU
attention half from PR #989 (free), two working hybrid-loop patterns to copy,
a matrix-state scan loop in `MIMOMambaBlock` that is the right shape, and
Q4_K/Q6_K/Q8_0 decode.

**What is genuinely new:** the Gated DeltaNet delta rule (~250-400 lines), and
the recurrent state cache for decode (first of its kind in this repo).

**What is removed from scope by evidence:** the MTP head (separate file, never
touched by plain inference) and mRoPE (degenerates to 1D RoPE with no image
tokens) — two workstreams that look mandatory from the architecture
description and are not.

**Biggest risk:** the state cache. **Biggest unknown:** the exact DeltaNet
gating formulas. Both are addressed by doing M0 first.

This is a multi-week epic. It is a *tractable* multi-week epic, and the 0.8B
fixture means nearly all of it can be done on a laptop before a single byte of
the 27B is downloaded — but it should be funded as three to four weeks of one
engineer, not slipped into a sprint.
