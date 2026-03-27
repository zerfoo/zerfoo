# ADR 069: TransMLA -- Retrofit MLA onto MHA/GQA Models

## Status

Accepted

## Date

2026-03-27

## Context

Zerfoo already supports Multi-Head Latent Attention (MLA) for DeepSeek V3/R1
via `layers/attention/multi_head_latent_attention.go`. MLA compresses KV into
a low-rank latent vector, reducing KV cache by 93.3% and boosting generation
throughput 5.76x vs standard MHA.

However, this benefit is exclusive to models natively trained with MLA (currently
only DeepSeek). The vast majority of deployed models (Llama, Gemma, Qwen, Mistral,
Phi) use standard MHA or GQA attention.

Two recent papers show how to retrofit MLA onto existing MHA models:

- **TransMLA** (arXiv:2502.07864, February 2025): Demonstrates that MLA's latent
  features function as a shared large KV head. Shows how to derive low-rank
  projection matrices from existing MHA weights via SVD decomposition, then
  fine-tune briefly to recover quality.

- **MHA2MLA** (arXiv:2502.14837, February 2025): Provides a practical conversion
  pipeline: (1) compute SVD of the KV projection matrices, (2) keep the top-r
  singular values as the latent dimension, (3) optionally fine-tune with a small
  dataset to recover quality lost in truncation.

The conversion produces projection matrices (wDKV, wUK, wUV) that are compatible
with Zerfoo's existing MLA layer. At inference time, only the compressed latent
vector is stored in the KV cache instead of full K/V heads.

## Decision

Add E40 (TransMLA/MHA2MLA) to implement:

1. **SVD-based weight conversion tool** (`cmd/transmla/`): Takes a standard
   GGUF model, computes SVD of K/V projection matrices per layer, truncates
   to target latent dimension, and writes a new GGUF with MLA projection
   tensors. This is a one-time offline conversion.

2. **Automatic MLA inference path**: When a GGUF model contains MLA projection
   tensors (detected by tensor name prefix "transmla."), the architecture
   builder wires the existing `MultiHeadLatentAttention` layer instead of GQA.
   No code changes needed in the MLA layer itself.

3. **Latent KV cache**: When MLA is active, the KV cache stores the compressed
   latent vector (dimension = kvLoraDim) instead of full K/V (dimension =
   numKVHeads * headDim). This is the source of the 93.3% memory reduction.

The latent dimension (rank) is configurable at conversion time. Higher rank
preserves more quality but reduces the compression ratio. Default: rank = 512
for models with hidden_size >= 4096 (e.g., Llama 3 8B).

## Consequences

**Positive:**
- Dramatically reduces KV cache memory for ALL supported model families,
  not just DeepSeek. A Llama 3 8B model would go from ~1GB KV cache at
  4K context to ~67MB (93.3% reduction).
- Reuses the existing MLA layer code -- no new attention mechanism needed.
- Conversion is offline and produces a standard GGUF file. No runtime
  conversion overhead.
- Enables running larger models or longer contexts on the DGX Spark.

**Negative:**
- SVD truncation is lossy. Quality depends on the singular value spectrum
  of the K/V matrices. Must validate perplexity before deploying converted
  models.
- Optional fine-tuning step requires training infrastructure (already available
  in Zerfoo's training package).
- Adds a new GGUF tensor naming convention ("transmla.") that must be
  documented and maintained.
- SVD computation on large models is CPU-intensive (one-time cost).
