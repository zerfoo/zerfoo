# Updates

## 2026-03-28: Split-GGUF support and mmap-by-default

Unlocks large-model loading: any split GGUF (70B+) loads transparently.
mmap is now the default loading strategy for all GGUF models.

### Changes
- **Split-GGUF loading**: `model/gguf/split_file.go` — `ParseSplit()` detects
  `-NNNNN-of-NNNNN.gguf` naming, discovers all shards, merges tensor maps.
  `LoadTensorsMmapSplit()` mmaps each shard independently. Auto-detected in
  both `LoadGGUF` and `LoadGGUFMmap` — no API change for callers.
- **mmap by default**: `loadOptions{mmap: true}` is now the default in
  `LoadFile` and `Load`. No flag or option needed. `WithMmap(false)` to opt out.
- **MiniMax-M2 (229B) on 128 GB**: 138 GB Q4_K_M model (3 shards) downloads
  to DGX Spark. Benchmark in progress.

### Why mmap default
Near-instant startup, zero Go heap pressure, and models larger than physical
RAM work out of the box. The OS pages tensor data from NVMe on demand.
Downside is negligible: a few ms of first-touch page faults on the first
inference call, imperceptible in practice.

## 2026-03-28: Architecture Expansion -- 40 architectures, 24 model families

Session 3: 14 new architecture builders added. Full Ollama model coverage achieved.

### New Architectures
- **GPT-2**: Learned position embeddings, GELU, LayerNorm. Enables TinyStories for pkg.go.dev examples.
- **Nemotron-H / Nemotron-H-MoE**: NVIDIA hybrid Mamba-2 + Attention + MoE (128 experts, top-6).
- **MiniMax M2**: Sigmoid MoE gating with routing bias (256 experts, top-8), QK normalization.
- **OLMo2**: AI2 Open Language Model (Llama-like).
- **InternLM2**: Shanghai AI Lab (Llama-like).
- **EXAONE / EXAONE4**: LG AI Research (Llama-like).
- **StarCoder2**: BigCode, sliding window attention (Mistral-like).
- **DBRX**: Databricks fine-grained MoE (16 experts, top-4).
- **GLM4 / ChatGLM / GLM4-MoE**: Zhipu AI, dense + MoE variants.
- **Kimi K2 (kimi-linear)**: Moonshot AI, linear attention MoE -- phi(Q)(phi(K)^T V) with ELU+1 feature map.
- **LFM2 / LFM2-MoE**: Liquid AI Foundation Model, hybrid dense/MoE dispatch.

### Other Deliverables
- **pkg.go.dev Examples**: 9 Example functions with NewModel constructor (v1.32.0)
- **CI Fixed**: All 6 repos green (grpc CVE, arm64 runner, purego vet exclusions)
- **Website**: Updated models grid (18 cards), Go 1.26, new CLI commands
- **Voxtral plan**: docs/plan-voxtral.md for Mistral speech-to-text (future)

## 2026-03-28: EAGLE Training, Q4_K Optimization, Granite TS Parity

Session 2: 28 additional tasks completed (total 68 this cycle).

### New Capabilities
- **`zerfoo eagle-train`**: Train EAGLE prediction heads from corpus text. Synthetic + model-based modes. Exports to GGUF.
- **`zerfoo transmla-validate`**: Compare perplexity between original and MLA-converted models.
- **Q4_K GEMV 14x faster**: Fused sub-block dequant kernel eliminates dequant→requant→Q4_0 path. DGX: 47.8ms → 3.36ms.
- **TieredKVStore thread-safe**: sync.RWMutex for concurrent serve access.
- **Architecture validation**: 24 architectures, 13 model families verified.

### DGX Integration Results
- Gemma3-1B compressed KV cache: PASS
- Mistral 7B quality + sliding window: PASS (coherent at 103 words)
- Mistral 7B performance: 2.4 tok/s (CPU fallback, 3-run median)
- Flash decode: 8-12 tok/s (Gemma3-1B, CPU path)
- Q4_K: 3.6x slower than Q4_0 after optimization (was 51x)
- DeepSeek-R1-1.5B MoE: loads and generates correctly

### Granite TS Parity
- 10 TTM golden files from granite-tsfm 0.3.5
- Zerfoo TTM: **21x faster** than Python (0.093ms vs 1.93ms on DGX)

### Apple M4 Metal
- Metal elementwise kernels: all pass
- CPU NEON benchmarks: MatMul 29us (64x64), Q4_0 GEMV 692us, TTM 40us

### Releases
- zerfoo v1.30.0, ztensor v0.12.0, v0.13.0

## 2026-03-27: Waves 4-8 Research Inference Optimizations

39 tasks shipped across 4 PRs (#262, #263, #264, #265). Two bug fixes discovered and shipped.

### New Features (Wave 4, PR #262)
- **EAGLE speculative decoding**: `WithEAGLE(headWeightsPath)` generator option. Self-speculative decode loop with adaptive draft length.
- **QuaRot weight fusion**: `--quarot` CLI flag and `WithQuaRot()` option. Hadamard rotation fused at load time.
- **Quantized KV cache**: `WithKVDtype("q4")` / `WithKVDtype("q3")`. Q4: 7.5x memory reduction, Q3: 6.4x.
- **TransMLA CLI**: `zerfoo transmla --rank 512 --input model.gguf --output model-mla.gguf`. SVD-based MHA-to-MLA conversion.
- **TransMLA inference**: Auto-detection of TransMLA tensors in GGUF, wires MLA layer automatically.
- **Multi-LoRA serving**: Per-request adapter selection via `model: "base:adapter"` in API.
- **BitNet ternary MatMul**: Transparent ternary GEMV dispatch for BitNet b1.58 models.
- **NSA layer registry**: NativeSparseAttention registered for GGUF builder.
- **SparseRoutedAttention registry**: Registered for GGUF builder.
- **Contrastive routing loss**: Auxiliary loss for sparse attention training.
- **Async CPU-to-GPU prefetch**: TieredKVStore async prefetch with channel-based queue.
- **Async CPU expert dispatch**: Goroutine pool for hybrid MoE CPU experts.
- **Predictive expert prefetch**: 98% hit rate on DeepSeek-V3 patterns.

### Test Coverage (Wave 6, PR #263 + Wave 7, PR #264)
- QuaRot weight fusion (6 tests, roundtrip tolerance)
- Quantized KV cache (Q4/Q3 memory reduction, quality degradation bounds)
- EAGLE head (shape, determinism, batch sizes) + decode loop (verification, adaptive N)
- NSA (degenerate, divergence, gate weighting)
- SVD conversion (roundtrip, rank truncation, shapes)
- I-Quant dequantization (IQ4_NL, IQ3_S, IQ2_XXS — 8 tests)
- Radix cache + scheduler (hash collision, LRU, ordering)
- Multi-LoRA (LRU eviction, concurrent, API selection)
- Compressed KV cache, document-wise RoPE, sparse routed attention, tiered storage, BitNet loading
- TransMLA end-to-end integration test
- go vet clean for E35, E36, E37, E39, E41, E42, E44

### Bug Fixes (Wave 7-8)
- **TernaryStorage preserved in GGUF loader** (PR #264): `decodeTernaryTensor` was discarding TernaryStorage by calling `tensor.New` instead of `tensor.NewWithStorage`. BitNet GEMV dispatch now works for GGUF-loaded models.
- **Q2_K/Q3_K tensor decoders** (PR #265): Phi-3 and Llama 3.1 GGUFs use Q2_K/Q3_K for norm/bias tensors. Constants were defined but decode paths were missing. Added decoders with dequantize-to-Q4 re-quantization.

### Remaining (blocked by external dependencies)
- ztensor repo: T34.1.4, T35.1.3, T39.1.4
- DGX Spark: benchmarks, GPU integration tests
- Python: Granite TS golden files
