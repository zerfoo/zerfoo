# ADR-004: Embeddable Inference Library

**Status:** Accepted
**Phase:** 8
**Date:** 2026-03-02

## Context

Running inference on an imported model required extensive manual wiring: download
ONNX files, convert with zonnx CLI, write Go code to create Engine/Graph, use
whitespace-only tokenizer, call Forward in a manual loop (no KV cache, O(n^2)),
no sampling beyond argmax, no streaming.

Phase 8 transforms zerfoo into an embeddable inference library:

```go
m, _ := inference.Load("google/gemma-3-4b-it")
resp, _ := m.Generate(ctx, "Explain quantum computing")
```

## Decision

### BPE Tokenizer

Pure Go, no CGo. Loads HuggingFace tokenizer.json format (vocab, merge rules,
pre-tokenizer config, normalizer, special tokens). BPE merge loop: split into
bytes, iteratively merge highest-priority adjacent pair. Byte-level BPE
pre-tokenization (GPT-2 style). SentencePiece .model files not supported
directly; most HuggingFace models ship tokenizer.json.

### KV Cache

GenerationContext embeds context.Context and carries *KVCache. KVCache stores
per-layer key/value tensors (appended on each step). Attention layers
(GroupQueryAttention, GlobalAttention) check for KVCache in context: if present,
append current K/V, use full cached K/V for computation. Graph.Forward()
signature unchanged (opt-in via context). Callers without KVCache get existing
full-recompute behavior.

### Generation Loop

Generator holds graph, tokenizer, engine, model config. Autoregressive loop:
1. Encode prompt to token IDs
2. Forward pass for logits [1, seqLen, vocabSize]
3. Extract last-position logits
4. Apply: temperature scaling -> top-k filtering -> top-p filtering ->
   repetition penalty -> sample (or argmax at temperature=0)
5. Check stop conditions (EOS, max tokens, stop strings)
6. Repeat with new token as input (KV cache handles prefix)

### Streaming

TokenStream interface with OnToken(token string, done bool) error. GenerateStream
delivers each decoded token as generated. Sentinel-based stop-string detection
with delta emission.

### Model Registry

Local cache under ~/.zerfoo/models/. Layout: <org>/<model_name>/ containing
model.zmf, tokenizer.json, config.json. Pull: download from HuggingFace Hub API,
convert ONNX to ZMF (zonnx as Go library), cache locally. HF_TOKEN env var for
gated models.

### HTTP Serve

net/http server. OpenAI-compatible endpoints:
- POST /v1/chat/completions (non-streaming + SSE)
- POST /v1/completions (non-streaming + SSE)
- GET /v1/models

### Constraints

- Pure Go, no CGo, no external C libraries for tokenization.
- KV cache is opt-in; does not break existing callers.
- Model registry works offline after initial pull.
- No training through high-level API.
- No multi-model serving.

## Consequences

- 3-line model loading and generation for end users.
- O(n) per generation step with KV cache (vs O(n^2) without).
- CLI commands (pull/run/serve) for interactive and server use.
- OpenAI API compatibility enables tool interoperability.
- Coverage: generate 95%, inference 96.4%, serve 96.4%.
- Embeddings not yet supported (Embed returns error).

### Key Files

- `pkg/tokenizer/bpe.go` -- BPE tokenizer
- `pkg/tokenizer/loader.go` -- tokenizer.json loader
- `generate/kvcache.go` -- KV cache
- `generate/generator.go` -- Autoregressive generation loop
- `generate/sampling.go` -- Temperature, top-k, top-p, repetition penalty
- `generate/stream.go` -- TokenStream interface
- `registry/registry.go` -- ModelRegistry, LocalRegistry
- `inference/inference.go` -- Load, Generate, Chat, GenerateStream
- `serve/server.go` -- OpenAI-compatible HTTP server
- `cmd/cli/{pull,run,serve}.go` -- CLI commands
