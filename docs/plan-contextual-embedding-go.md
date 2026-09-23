# Contextual embedding in Zerfoo Go

Status: In progress. This plan implements the owner-directed path from the
experimental `zerfoo/skillrouter` adapter to Go-only inference and training.
Zerfoo APIs remain useful for arbitrary query/document retrieval; the Qwen3
checkpoint is a pinned validation fixture, not a domain type in the framework.

## 1. Load model and adapter weights

- Load the official Qwen3-Embedding-0.6B GGUF at a pinned revision. Keep GGUF
  as the runtime model format and record the exact file digest.
- Convert the released PEFT safetensors LoRA adapter to Zerfoo's GGUF adapter
  layout with a Go command. Validate every tensor name, shape, rank, alpha,
  target module, and source digest before applying it. Reject missing or
  unexpected tensors rather than silently loading a partial adapter.
- Apply adapter weights to the decoder's projection tensors before graph
  construction, or attach equivalent LoRA nodes. Check the Go-applied weights
  against a reference merged checkpoint on a deterministic sample.
- Acceptance: pinned base plus released adapter load without Python at runtime;
  a synthetic LoRA fixture passes an exact weight-delta test; wrong metadata
  and shapes fail loudly.

## 2. Expose contextual embeddings

- Add a generic full-sequence decoder graph output at the final normalized
  hidden state, before the LM head. Preserve the existing generation graph.
- Add a model API that tokenizes, truncates to a configured maximum, selects
  the final non-padding token, and L2-normalizes. Keep query/document
  formatting outside the core model; the retrieval adapter may set different
  strings for each role.
- Make full-sequence execution use the same causal attention, RoPE, Q/K norms,
  token IDs, and tensor layout as the pinned reference. Ensure repeated calls
  do not reuse a previous sequence's KV cache.
- Acceptance: output dimension 1024 for the pinned Qwen model, deterministic
  repeat calls, empty-input and over-limit errors, and a small Go example that
  satisfies `retrieval.Embedder`.

## 3. Prove inference parity

- Freeze query and document strings, token IDs, base/adapter revisions, and
  reference vectors from the experimental checkpoint. Verify tokenizer IDs
  before comparing vectors. Compare unquantized or F16 Go output with the
  PyTorch reference, then measure the extra drift from any Q8 GGUF variant.
- Run CPU checks and a GPU parity job through the DGX Spark pod API. Compare
  vector cosine/max absolute error and ranking agreement over fixed candidates;
  record latency and memory. Set tolerances from measured precision, not from
  a passing score chosen afterward.
- Acceptance: reproducible Go-vs-reference vector and ranking report, with
  all source/artifact hashes and any unresolved mismatch recorded. Do not call
  the Go model qualified until this gate passes.

## 4. Train within Zerfoo Go

- Reuse the contextual graph as a differentiable query/document encoder.
  Expose named replaceable projections for LoRA, add batch collation,
  padding masks, gradient accumulation, optimizer state, and resumable GGUF
  checkpoints. The existing `InBatchContrastive` loss supplies weighted
  positives and rectangular hard negatives; verify gradients through the
  actual encoder and LoRA layers on CPU and GB10 GPU.
- Run the pinned skillrouter split and configuration on DGX Spark using a Go
  executable. Compare loss, embeddings, and retrieval metrics to the reference
  run. No OpenAI or Anthropic models are used for data generation or
  distillation; any future model-generated examples use OpenRouter.
- Acceptance: restartable Go training, finite-gradient and parity checks, and
  a versioned adapter usable by step 1 without conversion.

## 5. Qualify and release the Go-trained model

- Evaluate with independently worded tasks, no-match and multi-skill cases,
  held-out sources, and a second non-skill record set. Measure BM25, frozen
  encoder, Go-trained encoder, hybrid ranking, task success, p50/p95 latency,
  and index memory. Tune abstention on validation only.
- Publish the model card, GGUF adapter, exact data and training provenance,
  licenses, known failures, and a Go loading example after the numerical and
  retrieval gates pass. Keep the experimental release labeled experimental
  until these checks are complete.

## Current evidence

- The generic retrieval API and contrastive loss merged in PR #1012.
- `inference.Model.Embed` still mean-pools token-table rows; it does not run
  the transformer. The Qwen3 graph currently ends in an LM head.
- The experimental skillrouter release contains a safetensors adapter and
  description-derived evaluation only. The reference model uses a 768-token
  limit, final-token pooling, and normalized 1024-dimensional vectors.
