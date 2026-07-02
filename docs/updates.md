# Updates

## 2026-07-02: H2 strategy adopted — Trust, then Traction

- **ADR-093 accepted**: A full product review (plan, devlog, benchmarks, ADRs 068–092, the issue tracker, and every satellite repo) concluded that engineering is ahead of the 10-year vision's Year-1 targets while distribution and trust lag. The response is a "Trust, then Traction" H2 roadmap. Details in [`docs/product-strategy-2026-H2.md`](product-strategy-2026-H2.md); it supersedes the priority ordering in `docs/plan.md` but not the 10-year thesis in `docs/VISION.md`.
- **Plan reset to Phase 0** under the new strategy.
- **Tracker hygiene**: ~40 stale/noise issues closed and a new `parked` label introduced to mark deferred-but-not-dead work, so the open tracker reflects what is actually in flight.
- **Knowledge base**: `docs/lore.md` created as an append-only landmine/invariant register (seeded with 12 invariants); the ADR index was regenerated.
- **Incoming**: a containment gate for the capture-replay gradient divergence tracked in [#878](https://github.com/zerfoo/zerfoo/issues/878) — see the capture-replay note below.

## 2026-06-13 → 06-30: v1.50.x → v1.55.1 — capture-replay training runner + LTX-2 diffusion primitives

Follow-up hardening after the June training-stack campaign, plus the first primitives for the LTX-2 diffusion inference epic (ADR-092, E127 started).

- **CUDA-graph capture-replay training runner**: capture-safe cross-entropy loss and device-side Concat backward, with a device-resident loss seed so capture-on no longer host-copies inside the capture region ([#875](https://github.com/zerfoo/zerfoo/issues/875)).
- **On-device AdamW**: optimizer step runs on-device when the engine and params are GPU-resident (ztensor v1.12.0 fused AdamW kernel).
- **bf16 gradient accumulation**: device-resident bf16 accumulation now goes through the engine instead of a host round-trip.
- **Engine-op Dropout**: capture-safe `WithEngineDropout` path.
- **Diffusion primitives (E127, ADR-092)**: GroupNormalization layer and Conv3d / ConvTranspose3d forward primitives landed, with an LTX-2 VAE decoder skeleton.
- **Known issue**: capture-replay can silently diverge gradients under a corrected 1.0 loss seed ([#878](https://github.com/zerfoo/zerfoo/issues/878), open) — a containment gate is being added.

## 2026-06-11: GPU training-stack hardening campaign (ADR-091)

Every op — used by any workload or not — now has to clear a per-op verification gate before merge: finite-difference gradcheck, GPU-vs-CPU parity under arena stress, and a PyTorch-oracle harness on GB10.

- **Verification gates (ADR-091)**: gradcheck / OpInfo, parity-under-arena-stress, and PyTorch-oracle checks, wired as design.md §7.5.
- **Lifetime + memory contracts**: `SaveForBackward` for backward-consumed caches across activations, attention, normalization, FFN/FiLM/TCE/VSN, and dropout masks (ADR-006); arena pinning; and a poison mode that surfaces stale-free / double-free bugs deterministically.
- **Root causes were contract gaps, not numerics**: unordered host access to device gradients on GB10 unified memory (ztensor#137) and stale GC-finalizer frees after an arena Reset (ztensor#138), both fixed upstream (ztensor v1.11.0).
- **Measured outcome**: two consecutive clean end-to-end GB10 f32 training runs — zero NaN, epoch loss 0.778373, accuracy within 0.05pp of the CPU baseline, ~390 samples/s. Requires ztensor v1.11+ and zerfoo v1.49+.

## 2026-04-27 → 04-28: E124 layout cleanup + ADR-090 scope decision

- **ADR-090**: decided that cloud, marketplace, and compliance code is out of scope for the open-source core and is extracted to the private enterprise repo. Zerfoo core stays a general-purpose ML inference/training framework.
- **E124 package-layout cleanup**: the extraction was executed and verified across all subtrees; `docs/design.md` was realigned to mandate the post-E124 package layout and the GGUF-only doctrine.

## 2026-04-13: v1.43.0 → v1.48.0 — Gemma 4 architecture support

New architecture builders for the Gemma 4 family, released across v1.43.0–v1.48.0.

- **Dense / MoE / edge builders**: a 31B dense builder, an MoE variant builder, and an edge variant with per-layer embeddings (PLE) and shared-KV attention (ADR-085, ADR-086).
- **External-KV GQA path (ADR-087)**: `ResolveKVDonor` + a `kv_reuse_node` wire a donor layer's K/V into shared-KV attention, so edge layers can reuse a neighbor's KV cache. CUDA-graph capture support for the PLE combiner and LM head (ADR-088, ADR-089).
- **Known defect — do not rely on Gemma 4 edge decode yet**: gemma4 edge (`gemma4e`) decode output is not yet correct ([#757](https://github.com/zerfoo/zerfoo/issues/757), open); the failure is isolated to Q4_K PLE gather tables. No performance claims are made for Gemma 4 at this time.

## 2026-04-06: v1.42.1 released + DGX benchmark infrastructure

- **v1.42.1 released**: Merged release-please PR #340. Includes all E77-E84 composition phase 4 work.
- **cmd/bench_train**: New benchmark tool for PatchTST training (28K x 20ch x 10 epochs). Supports GPU/CPU engine selection.
- **DGX benchmarks started**: GPU + CPU benchmarks running on DGX. SSH became unreachable under load. Results pending collection.
- **Plan status fix**: Updated stale E77-E84 status lines (all DONE). Fixed DGX IP in hand-off notes.
- **Blockers identified**: T58.1.2 (GQA parity) needs GGUF model files on DGX. T63.2.1-T63.2.3 need CUDA CGo stubs.

## 2026-04-06: ztensor v1.4.0 + E64 — Matmul consolidation + file decomposition

- **E63 complete (2/5 tasks)**: Consolidated 14 quantized matmul methods into shared helpers (`gpu_engine_matmul.go`). Net -557 lines. ztensor PR #76, released as v1.4.0. DGX validation (T63.2.*) pending.
- **E64 complete (3/3 tasks)**: Split `gpu_engine.go` (3,521→2,245 lines) into 5 focused files: elementwise (400), reduction (221), memory (695), matmul (240), core (2,245). ztensor PR #77.
- **Pre-existing CI fixes**: Fixed `GemvQ5_0F32` test, added `metal`/`pjrt` to vet exclusion list.
- **Plan updates**: T56.1.3 marked done. T54.3.1 deferred. T56.3.1 blocked (needs engine.Narrow).
- **DGX IP updated**: 192.168.86.29

## 2026-03-29: v1.36.0 — Audio transcription, tiered KV cache, MiniMax-M2 fixes

### zerfoo v1.36.0 / ztensor v0.15.0

**New features:**
- **`Model.Transcribe`**: Speech-to-text via Whisper or Voxtral. WAV input, mel spectrogram extraction, 30-second chunking, parallel greedy decode. CLI: `zerfoo transcribe`.
- **`WithTieredKV`**: Hot/warm/cold tiered KV cache. Layers auto-demote to warm (compressed) or cold (disk) based on access frequency. Async prefetch for cold→hot. Unblocks long-context inference beyond GPU memory.
- **ztensor: `MmapStorage.SliceElements`**: Zero-copy expert weight slicing for mmap'd MoE stacked weights.
- **ztensor: Streaming GEMM**: Streaming GEMM for mmap'd tensors, enabling over-RAM CPU inference at scale.

**Bug fixes:**
- `fix(generate): Close() skips deletion of user-provided ColdDir` — TieredKVStore no longer removes a caller-supplied cold directory on close.
- `fix(attention): QK norms applied before head reshape` — fixes MiniMax-M2 output quality.
- `fix(inference): OOM eliminated during MoE graph build` — `NewFFNFromDense` removes ~857 GB phantom allocation for 256-expert models.
- `fix(inference): Zero-copy expert slicing` — `buildExpertFFN` uses pre-existing weight slices via `NewFFNFromDense`.

**Releases:** [zerfoo v1.36.0](https://github.com/zerfoo/zerfoo/releases/tag/v1.36.0) · [ztensor v0.15.0](https://github.com/zerfoo/ztensor/releases/tag/v0.15.0)

## 2026-03-29: E45 verification remediation shipped (PR #274)

Resolved all gaps found by /verify audit. `WithTieredKV` GeneratorOption now exposes
TieredKVStore through the standard generator API. `TestSchedulerImmediateEviction`
is now deterministic (select-based, 10/10 passes). CI green, merged to main, v1.36.0
release PR pending.

## 2026-03-29: MiniMax-M2 229B inference verified on 128 GB DGX Spark

First successful inference on a model larger than physical RAM. MiniMax-M2
(229B MoE, 128.8 GB Q4_K_M, 3 shards) loads in 6.3s and generates tokens on a
128 GB machine. Ollama fails to load the same model on the same hardware.

### Changes (PRs decd668, 4ed3955, 0cd50bb)
- **NewFFNFromDense constructor** (`layers/core/ffn.go`): eliminates ~857 GB
  phantom allocation during MoE graph build (256 experts × 62 layers × 3 Dense
  layers × random float32 allocation).
- **Zero-copy expert slicing** (`inference/arch_deepseek.go`): `buildExpertFFN`
  now uses pre-existing weight slices via `NewFFNFromDense` instead of calling
  `NewFFN` which allocated and immediately overwrote random weights.
- **NewLinearFromParam nil ops fix** (`layers/core/linear.go`): was missing
  `ops: engine.Ops()`, causing panic at `Sigmoid.ops.One()` in `WithSwiGLU`.
- **qkNormPreReshape for MiniMax-M2** (`layers/attention/grouped_query_attention.go`,
  `inference/arch_minimax_m2.go`): MiniMax-M2 stores qNorm weights as `[nH×hD]`
  not `[hD]`. Added flag to apply norm before head reshape.
- **README + benchmarks.md**: updated with verified inference result.

### Result
```
MiniMax-M2 229B Q4_K_M / 128 GB DGX Spark / CPU-only
Load: 6.3s | Output: "The meaning of life is a priori is something" | 0.06 tok/s
```

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
