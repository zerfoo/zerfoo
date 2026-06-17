# ADR 092: LTX-2 diffusion audio/video inference -- DiT-first, via general primitives

## Status
Proposed

## Date
2026-06-16

## Context

LTX-2 (arXiv 2601.03233, Lightricks) is an asymmetric dual-stream audio+video
**diffusion transformer (DiT)** -- a 14B-parameter video stream plus a
5B-parameter audio stream (~19B total) coupled by bidirectional audio-video
cross-attention. We want to run **LTX-2 19B** natively in Zerfoo. (LTX-2.3 22B is
out of scope -- see Deferred.)

This is structurally unlike everything Zerfoo runs today. All current
architectures are **autoregressive LLMs**: load GGUF, build a graph, decode
token-by-token with a growing KV cache, sample from a vocab. LTX-2 is a
**diffusion** model: there is no KV cache and no vocab sampling. Generation is an
iterative **flow-matching / rectified-flow** denoise of a fixed-shape latent
tensor (40 steps full, 8 distilled) with modality-aware classifier-free
guidance, followed by VAE + vocoder decode to pixels and 24kHz stereo audio. The
correct **baseline is the LTX-2 PyTorch / ComfyUI reference on the GB10 DGX
Spark** -- not Ollama, which does not run LTX. The reference path needs the
LTX2-specific diffusers dev build (`0.37.0.dev0`, `LTX2Pipeline`,
`AutoencoderKLLTX2Video`/`Audio`, `LTX2Vocoder`, `LTX2TextConnectors`),
provisioned on Spark.

**On numerical parity.** Parity here means matching the PyTorch reference
**within the per-op tolerance band defined by the PyTorch-oracle gate
(ADR-091)** -- NOT bit-exactness. Bit-exactness is impossible: production GPU is
`GPUEngine[float32]` with bf16/fp8 as reduced-precision *storage* (there is no
native `GPUEngine[bfloat16]`), the FFN is gelu-approximate, and fused kernels
reorder arithmetic. The earlier draft's "bit-comparable" framing was an
overclaim and has been removed.

What already exists in Zerfoo is encouraging. The FLOPs-dominant transformer
denoiser maps cleanly onto existing primitives: `ScaledDotProductAttention` /
`FusedSDPA` (Q,K,V separate, so cross-attention is composable), `RMSNorm`
(qk_norm=rms_norm_across_heads), `RotaryPositionalEmbedding` (split RoPE),
`Gelu` (gelu-approximate FFN -- this IS the LTX-2 activation), fused CUDA kernels
(FusedAddRMSNorm, FusedQKNormRoPE, FusedSoftmaxVMul), quantized GEMM
(Q4/Q8/K-quant/FP8), and the `Graph.Compile` -> `ExecutionPlan` ->
`NewCUDAGraphExecutor` substrate that already gives autoregressive decode
zero-relaunch CUDA-graph replay.

What is missing is a known, bounded set: no diffusion scheduler / denoise loop,
no AdaLN-Zero modulation, no timestep/sinusoidal embedding, no packaged
cross-attention module, no GroupNorm, no Conv3D, no ConvTranspose, no
convolutional/image VAE, and -- by repo policy (CONTRIBUTING.md) -- no
safetensors runtime loader. The CUDA-graph capture gate is also currently keyed
to autoregressive decode (`inputs[0]` last-axis > 1 -> skip capture, at
`ztensor/graph/cuda_graph.go:354-355`).

**Verified geometry vs. assumptions.** The architecture facts the epic builds on
are verified: 48 layers, video inner 4096 (32x128), audio inner 2048 (32x64),
caption channels 3840, in/out 128 latent channels, cross-attention 4096/2048,
FlowMatchEuler, 40-step full / 8-step distilled, Gemma3-12B encoder (hidden
3840), VAE `vae_scale_factors=[8,32,32]` with 128-latent, audio VAE latent 8, and
crucially **`patch_size=patch_size_t=audio_patch_size=audio_patch_size_t=1`** --
the DiT ingests VAE latents directly with NO spatial patch folding; all spatial
compression lives in the VAE. Three things the research could NOT independently
verify are carried as explicit "ASSUMPTION -- validate in T127.x" items, not
asserted facts: (1) the fp8 sub-format (E4M3FN vs E5M2; binary header unreadable
via WebFetch); (2) whether n>1 low-precision GPU GEMM at the denoise regime needs
new kernels; (3) LTX-2.3 22B geometry (its binary header could not be re-fetched).

The central tension is **breadth vs. bandwidth**: the full LTX-2 pipeline is a
large surface (DiT + 2 VAEs + vocoder + text encoder + connectors + scheduler +
loader). Trying to land all of it before producing any measurable result risks a
long, unverifiable build. But the transformer denoiser alone -- the ~19B,
run-40x core -- is reachable early from existing primitives, and it is where the
performance story lives.

## Decision

**Build DiT-first, with general primitives, not LTX-special-cased code.**

1. **DiT-first phasing.** Phase 1 lands the dual-stream DiT denoiser using
   existing attention/RMSNorm/RoPE/fused/quantized kernels, with the VAE,
   scheduler, and text encoder initially **stubbed or fed fixtures**, so a
   per-step benchmark against the GB10 PyTorch reference is reachable before the
   long tail. Critically, the Phase-1 milestone (T127.1.6) is a **shape /
   runs-green / self-consistency / per-step-latency** check -- it does NOT claim
   PyTorch-oracle parity, because random fixture weights cannot match the
   real-checkpoint oracle. The **full-forward oracle match moves to T127.3.4**,
   where real converted weights exist. (Per-block oracle parity on a FIXED
   injected-weight fixture, T127.1.5, IS legitimate because both sides use the
   same weights.) Subsequent phases add the flow-matching scheduler (2), the
   safetensors->GGUF loader for DiT + VAEs + vocoder (3), the video VAE decode
   (4), the Gemma text encoder + connector (5), the audio stream + cross-modal
   sync + vocoder (6), and performance (quantization + CUDA-graph capture + full
   memory-budget sizing) + parity (7). A Phase-0 gating wave first extends the
   ADR-091 oracle to the new ops, runs the fp8/perf spike, and provisions the
   PyTorch reference + fixture generator.

2. **Every new building block is a reusable framework primitive**, each carrying
   a doc note naming the other model classes it unlocks:
   - **Timestep/sinusoidal embedding + AdaLN-Zero** -> all DiT-family diffusion
     models (PixArt, SD3/Flux MM-DiT, Hunyuan, Mochi, Wan). The AdaLN node is the
     generic, parameterizable-count 6-vector zero-init convention; the LTX
     `adaln_single` + `scale_shift_table` layout is mapped onto it **in the arch
     builder**, never inside the primitive.
   - **Packaged CrossAttention** (separate context K/V projections) -> all
     encoder-decoder transformers, T5/Whisper-style models, multimodal fusion.
   - **`Scheduler[T]` flow-matching abstraction + denoise loop** -> all diffusion
     samplers (DDPM/DDIM/Euler/rectified-flow/distilled).
   - **Conv3D, ConvTranspose, GroupNorm** -> video/volumetric models and the
     whole convolutional-VAE/UNet (Stable-Diffusion-family) class.
   - **safetensors header reader + safetensors->GGUF converter (in zonnx, not
     this repo)** -> the entire HuggingFace diffusion/vision ecosystem.

3. **No LTX-specific logic leaks into framework code.** The only LTX-named code
   is the arch builder (`inference/arch_ltx2.go`), its GGUF tensor-name map, and
   the LTX VAE/vocoder decoders -- analogous to existing per-arch builders. All
   reusable mechanism lands in `layers/`, `compute/`, `generate/diffusion/`, and
   `model/gguf/`.

4. **Universal quality gates apply to every op -- but conv ops are inference-only
   for this epic.** Each new op (timestep embed, AdaLN, cross-attention,
   GroupNorm, SiLU node) must pass **gradcheck/OpInfo, the GPU/CPU
   parity-under-arena-stress harness, and the PyTorch-oracle gate (ADR-091)**
   before merge. **The conv ops (Conv2d/Conv3D/ConvTranspose) are inference-only
   for E127** because the VAE is decode-only and never trained: their gate is
   **forward-parity** (PyTorch-oracle forward + GPU/CPU parity), NOT gradcheck.
   Conv backward is a real gap (`conv2d.go:14` is documented inference-only) and
   is tracked as a **separate deferred issue for future VAE training** -- the
   epic does not silently require both "decode-only sidesteps backward" and
   "gradcheck passes" on the same op. A Phase-0 task confirms/extends the ADR-091
   oracle harness to cover conv3d/convtranspose/groupnorm/adaln/timestep-embed/
   cross-attention, so the op-parity ACs reference infrastructure that actually
   exists. Acceptance is **not single-consumer**: the Phase-1 primitives are
   validated by a standalone generic image-DiT that achieves **PyTorch-oracle
   parity against a small real DiT checkpoint** (not just a smoke test),
   satisfying the "non-Wolf path" rule.

5. **Honor the SafeTensors policy.** No safetensors runtime loader lands in
   zerfoo. The header parser is generalized from the existing
   `inference/timeseries/convert_*.go` converters, and the safetensors->GGUF
   conversion lands in **zonnx**, the sanctioned conversion home. The converter
   covers **the DiT shards AND the non-DiT components** (video VAE, audio VAE,
   HiFi-GAN vocoder) that Phases 4/6 depend on -- not just the transformer.

## Consequences

**Positive:**
- A measurable per-step benchmark against the GB10 PyTorch/ComfyUI reference is
  reachable at the end of Phase 1, before the VAE/audio long tail -- de-risking
  the whole epic.
- Zerfoo gains a full diffusion-DiT primitive set (AdaLN, timestep embed,
  cross-attention, scheduler, Conv3D, ConvTranspose, GroupNorm) that unlocks an
  entire model class beyond LTX, in line with the general-purpose doctrine.
- The compile + CUDA-graph-capture substrate is reused nearly unchanged; the
  capture gate becomes a static-shape predicate (guarded behind a NEW predicate,
  not a relaxed existing one) that also benefits any future fixed-shape workload.
- A general safetensors->GGUF path (in zonnx) opens the HF diffusion/vision
  ecosystem to the whole framework.
- The fp8-subformat and n>1-GEMM unknowns are de-risked by a Phase-0 spike, so
  the converter storage mapping and perf budget are committed on measured data,
  not assumption.

**Negative:**
- The GPU low-precision story is a known gap: production GPU is
  GPUEngine[float32] with bf16/fp8 as reduced-precision storage; there is no
  native GPUEngine[bfloat16], and quantized GPU GEMM for n>1 (the denoise regime)
  dequantizes to f32 for most K-quants. Throughput may need new kernels -- this
  is the load-bearing perf risk; it is measured early (Phase-0 spike) and flagged,
  not assumed away.
- Memory pressure is real: LTX-2 19B is 43.3GB bf16 / 27.1GB fp8 / ~20GB fp4,
  and it must co-reside with Gemma3-12B + two VAEs + vocoder + CFG activation
  expansion on GB10 unified memory. A dedicated task sizes the full resident set
  per storage tier; some components may need off-loading/streaming.
- Bimodal CFG doubles or triples the per-step forward in full 40-step mode
  (text-null + modality-null + conditional), while distilled 8-step mode runs
  CFG-off; the throughput story differs sharply between the two and is reported
  per-regime.
- Conv3D/ConvTranspose/GroupNorm and an audio VAE + HiFi-GAN vocoder are
  substantial new surface; Phases 4 and 6 are XL and may each justify a
  follow-up epic.
- Generalizing the CUDA-graph capture gate risks regressing LLM decode capture
  if done carelessly; it must be guarded behind a new predicate, not a relaxed
  existing one. The diffusion snapshotCache must honor the
  `NewCUDAGraphExecutor` (`cuda_graph.go:268`) contract -- a no-op must still
  return a non-nil empty restore closure (invoked at `cuda_graph.go:401`), not
  nil.
- A cross-repo dependency on zonnx for weight conversion (DiT + VAEs + vocoder)
  adds coordination cost.

**Deferred:**
- **LTX-2.3 (22B) support: strictly out of scope** until an independent header
  read (T127.8.1) confirms its geometry. The "2.3 geometry identical to 2"
  claim is from a single header read the research could **not** independently
  re-fetch (UNCERTAIN), and 2.3 ships no standalone config.json. No 2.3 builder
  or converter work may be scheduled or sized as if geometry parity is
  established; T127.8.1 (a byte-range header read via huggingface_hub, NOT
  WebFetch) is the precondition gate.
- Diffusion/VAE **training** (fine-tune/LoRA): this epic is inference + decode
  only. Conv2d/Conv3D/ConvTranspose **backward** is a known gap, tracked as a
  separate deferred issue; the conv ops are inference-only here and gated on
  forward-parity, not gradcheck.
- fp8 sub-format (E4M3FN vs E5M2) confirmation is an explicit validate-before-use
  task in Phase 0/3 (byte-range header read), not an assumption.

## Alternatives Considered

1. **Full pipeline before any benchmark.** Build DiT + both VAEs + vocoder +
   text encoder + scheduler + loader, then run. Rejected: maximizes time-to-first
   signal and time-to-first-bug on the largest, least-verifiable surface; a wrong
   assumption in the denoiser would not surface until everything else is built.
   DiT-first inverts this -- the FLOPs core is verified and benchmarked first.

2. **Delegate the VAE (and/or text encoder) to Python.** Keep the
   PyTorch VAE/vocoder and only port the DiT to Zerfoo. Rejected as the end state:
   it violates the Go-native, zero-CGo-by-default principle and leaves a Python
   dependency in the production path. However, the PyTorch reference IS used as
   the oracle and as fixture/stub generator during Phases 0-3 -- delegation as a
   *temporary scaffold*, not a shipped architecture. The Phase-0 task T127.1.0c
   provisions and pins exactly this reference (diffusers 0.37.0.dev0 with the
   LTX2 classes) on Spark and builds the fixture generator that emits the latent,
   text-context, and oracle reference tensors Phases 0-3 consume.

3. **Don't do it at all.** LTX-2 is far from Zerfoo's LLM-inference core, and
   diffusion is a new paradigm for the framework. Rejected: the primitives
   required (AdaLN, timestep embedding, cross-attention, scheduler, Conv3D,
   ConvTranspose, GroupNorm, safetensors->GGUF) are exactly the general building
   blocks that unlock the entire diffusion + vision + HF-ecosystem model class,
   which the framework's vision explicitly targets. The work is high-leverage
   precisely because so little of it is LTX-specific.

References: docs/adr/091-gradcheck-pytorch-oracle-verification.md (universal
quality gates / PyTorch-oracle tolerance band); docs/plan.md E127 (the
implementation epic); CONTRIBUTING.md (no SafeTensors/ONNX loaders -- use zonnx);
../CLAUDE.md (general-purpose doctrine, fix-at-the-contract-level).
