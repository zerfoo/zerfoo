# ADR 067: MSA-Inspired Sparse Attention for Scalable Memory

## Status

Accepted

## Date

2026-03-27

## Context

Contemporary LLMs support effective context lengths of 128K-1M tokens. Applications
requiring larger memory (document QA over large corpora, agents with persistent
history, Digital Twins) need a mechanism to access information from millions of
tokens without quadratic attention cost.

The Memory Sparse Attention (MSA) paper (EverMind-AI/Shanda Group, 2025) demonstrates
that a sparse attention mechanism with document-wise RoPE can scale to 100M tokens
with less than 9% quality degradation, achieving linear complexity in both training
and inference. A 4B model using MSA outperforms RAG systems backed by 235B generators.

We evaluated three decomposable techniques from MSA for adoption in Zerfoo:

1. **KV cache compression via chunk-wise mean pooling** -- reduces KV cache memory
   by pooling tokens into fixed-size chunks (e.g., 64 tokens per chunk = 64x
   compression). Enables serving longer contexts on memory-constrained hardware.

2. **Document-wise RoPE** -- assigns independent position IDs (starting from 0) to
   each document in the memory bank, decoupling positional encoding from total
   context length. Models trained on 64K contexts can retrieve from millions of
   tokens without positional out-of-distribution issues.

3. **Tiered KV storage** -- routing keys remain GPU-resident for low-latency
   scoring, while bulk content KVs are offloaded to CPU DRAM and fetched on demand.
   This decouples capacity from VRAM limits.

## Decision

Implement the three MSA-inspired techniques incrementally in zerfoo/zerfoo (not a
new repo), as they are model-architecture and inference-pipeline concerns:

- **KV cache compression**: new `CompressedKVCache` in `generate/` with chunk-wise
  mean pooling via existing `ReduceMean` engine operation.
- **Document-wise RoPE**: new RoPE mode in `layers/embeddings/` that resets position
  IDs per document boundary, plus a global offset for the query context.
- **Sparse attention with learned routing**: new `SparseRoutedAttention` layer in
  `layers/attention/` with dedicated Router Q/K projectors, cosine similarity
  scoring, and top-k document selection.
- **Tiered storage**: extend `generate/` with GPU-resident routing key index and
  CPU-offloaded content KV store with async fetch.

New ztensor primitives required:
- Cosine similarity kernel (GPU + CPU)
- Max reduction along axis (GPU + CPU)

These are small additions to `compute.Engine[T]`, not architectural changes.

## Consequences

**Positive:**
- Enables serving contexts of 10M-100M tokens on single-node hardware (DGX Spark
  has 128GB unified memory, well-suited for tiered storage).
- KV cache compression alone provides immediate value for existing models by
  reducing memory footprint for long-context serving.
- Document-wise RoPE enables existing RoPE-based models (Llama, Gemma, Qwen,
  Mistral) to retrieve from contexts far beyond their training window.
- All techniques are composable -- each delivers value independently.

**Negative:**
- Full MSA requires model fine-tuning with router projectors and auxiliary
  contrastive loss. Off-the-shelf GGUF models cannot use learned routing without
  training.
- KV cache compression via mean pooling is lossy; fine-grained information within
  chunks may be lost.
- Adds complexity to the attention and cache layers that must be maintained.
- Sparse routing is approximate; multi-hop reasoning across tightly coupled
  documents remains challenging (acknowledged limitation in the MSA paper).
