# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, documentation site, MSA-inspired scalable
memory, and research-driven inference optimizations).

Task statuses updated 2026-03-27 based on merged PRs and git history.

**Status summary:**
- 370+ tasks completed across all plans (37 new in Waves 1-3)
- ~20 active tasks remaining from prior phase
- E34: MSA-inspired scalable memory (18 tasks)
- E35-E39: Research-driven inference optimizations (44 tasks)
- E40: TransMLA/MHA2MLA conversion (9 tasks)
- E41-E44: GGUF I-Quants, RadixAttention, Flash Decoding, Multi-LoRA (24 tasks) -- NEW
- All models produce coherent output on CPU and GPU (ztensor v0.6.3, zerfoo v1.25.5)

---

## Completed Work (Summary)

### Priority 1: Tabular and Time-Series ML

All internal-consumer-blocking work is complete:

| Epic | Description | Status |
|------|-------------|--------|
| WE1-WE13 | Tabular, Time-Series, AutoML, RL, Cross-Asset, Regime, Self-Improving, Hardware Opt, Enterprise, Continuous Learning, Perf Fixes | Complete (all tasks) |

### Priority 2: Inference Performance and Bug Fixes

| Epic | Description | Status |
|------|-------------|--------|
| E2 | New Model Architecture Support (12 archs) | Complete |
| E4 | Documentation and Developer Experience | Complete except T4.7 (video) |
| E5 | Community Infrastructure | Complete except T5.4 (Discord) |
| E9 | Multi-GPU Inference | Complete except T9.4 (benchmark) |
| E10 | Vision-Language Model Expansion | Complete |
| E14 | SOC 2 Certification | Complete |
| E17 | Zerfoo Cloud GA | Complete |
| E18 | Enterprise Features | Complete |
| E20 | Apple Metal Backend | Complete except T20.3 (benchmark) |
| E21 | Intel SYCL Backend | Complete |
| E22 | Auto-Optimization Framework | Complete |
| E24 | Custom Model Architecture SDK | Complete |
| E25 | Heterogeneous Compute | Complete |
| E27 | Ecosystem Integrations | Complete |
| E28 | Federated Learning | Complete |
| E29 | On-Device Inference | Complete except T29.4 (benchmark) |
| E33 | Performance Target 1000+ tok/s | Complete |

### Priority 0: Security Remediation

All 22 security and remediation epics (E101-E122) complete, totaling 160+
tasks across deep reviews, issue resolution, training fixes, cloud
consolidation, observability, persistent stores, god file splits, backward
pass error propagation, and CodeQL CI.

### Granite Time Series (16/18 complete)

SafeTensors-to-GGUF conversion, all three architecture builders (TTM,
FlowState, TSPulse), inference pipelines, training backends, exogenous
variable support, CLI, and REST API endpoints all shipped (PRs #187-197, #208).

### Granite Guardian (13/13 complete)

Architecture builder, prompt template engine, verdict parser, evaluator, batch
evaluation, multi-risk scanning, REST API endpoints, CLI `zerfoo guard`,
guardrails middleware, parity tests (15/15 verdicts match Ollama), latency
benchmarks (77ms median, target <100ms PASS), and safety accuracy evaluation
all shipped (PRs #200-205).

### K-Quant Infrastructure (6 tasks complete)

Native Q4_K/Q5_K/Q6_K storage with virtual transpose, merged QKV/GateUp GEMV,
SM121 GEMV kernels, Q4_0 re-quantization restored, shared memory fix for 7B+,
GPU RMSNorm fallback (PRs #179-186).

### Multi-Model Benchmarks (BMK-T2 complete)

3-run median results (2026-03-27, DGX Spark GB10, 128 tokens, greedy):
Gemma3-1B 235 tok/s (1.25x Ollama), DeepSeek-R1 186 (1.11x), Llama3.2 92
(0.99x), Mistral-7B 44 (1.00x). All models produce coherent output after GQA
repeat fix (ztensor v0.6.3) and flash attention decode fix (zerfoo v1.25.5).
25% faster on small models, parity at 7B.

### GPU Verification (E114, 7/7 complete)

VRAM bounds check, FP16 MatMul fix, kernel rebuild, BF16 tolerance, timeseries
GPU verify, full suite report -- all pass on DGX Spark.

### GPU Training (E123, 6/6 complete)

All 7 timeseries backends wired to GPU engine forward path. Batched forward
pass for PatchTST shipped. DGX benchmark verified.

### GGUF Writer Consolidation (18/18 complete)

Shared ztensor/gguf package created, all writers in zerfoo and zonnx migrated
(PRs across 3 repos).

### Documentation Site (48/48 complete)

Hugo site at zerfoo.feza.ai/docs/ with 61 pages, full content migration, repo
cleanup, link verification, Lighthouse audit, mobile/search QA all complete.

### Other Completed Epics

| Epic | Status |
|------|--------|
| E11 Community Growth (T11.2, T11.3, T11.5) | Complete |
| E12 Enterprise Support (T12.1-T12.3) | Complete |
| E13 Security Audit (T13.1-T13.4) | Complete |
| E15 Edge Deployment (T15.1, T15.2, T15.5) | Complete |
| E16 Performance 500+ tok/s (T16.1, T16.2) | Complete |
| E26 ZerfooConf (T26.1) | Complete |
| E32 Architecture Expansion (T32.1) | Complete |

---

## Active Work -- Next Phase

All models now produce coherent output on CPU and GPU after two critical fixes
(ztensor v0.6.3 repeat semantics, zerfoo v1.25.5 flash attention decode).
The next phase focuses on benchmarking, quality verification, kernel
optimization, time-series parity, scalable memory, and research-driven
inference optimizations.

### P1: Full Multi-Model Benchmark (highest priority)

All models work -- time to build the definitive comparison table.

- [ ] BMK-T1 Download missing GGUFs  Est: 1h
  Missing: gemma3-4b-Q4_K_M, qwen2.5-7b-Q4_K_M. Download to DGX Spark
  model cache. Verify each loads successfully with `zerfoo run --dry-run`.
  Acceptance: Both GGUFs load and produce coherent first token.

- [x] BMK-T2 Re-run bench-compare-ollama.sh for all models  Est: 2h  DONE 2026-03-27
  Deps: BMK-T1
  Models: Gemma3-1B, DeepSeek-R1-1.5B, Llama3.2-3B, Mistral-7B.
  3-run median on DGX Spark, 128 tokens, greedy sampling.
  Results: Gemma3 235 (1.25x), DeepSeek 186 (1.11x), Llama 92 (0.99x), Mistral 44 (1.00x).
  All models produce coherent output. JSON: results/benchmark-2026-03-27.json.

- [ ] BMK-T3 Update website and README with full comparison table  Est: 1h
  Deps: BMK-T2
  Acceptance: Published table with 6+ models, ratios, and hardware specs.

- [ ] BMK-T4 Investigate Phi3/Llama3.1 GGUF load failures  Est: 2h
  Acceptance: Root cause documented in devlog; fix shipped or issue filed.

### P2: Mistral vs Ollama Head-to-Head

- [ ] MHH-T1 Run Mistral 7B quality comparison  Est: 2h
  Acceptance: Quality scores documented; token agreement > 95% with greedy.

- [ ] MHH-T2 Profile Mistral 7B performance gap  Est: 1h
  Deps: MHH-T1
  Acceptance: Confirmed ratio documented in devlog with 3-run median.

- [ ] MHH-T3 Test sliding window attention correctness  Est: 2h
  Acceptance: Coherent output at 5000+ tokens; no degradation past window.

### P3: K-Quant Kernel Optimization

All in ztensor repo. Q4_K is 45% slower than Q4_0.

- [ ] KQ-T1 Profile Q4_K vs Q4_0 GEMV  Est: 2h  repo: ztensor
  Acceptance: Root cause of 45% slowdown identified and documented.

- [ ] KQ-T2 Optimize Q4_K GEMV kernel  Est: 4h  repo: ztensor  Deps: KQ-T1
  Acceptance: Measurable improvement in BenchmarkGEMV.

- [ ] KQ-T3 Benchmark and re-enable native Q4_K loading  Est: 2h  Deps: KQ-T2
  Target: >= 215 tok/s on Gemma 3 1B. Acceptance: All tests pass.

### P4: Granite TS Parity Tests

- [ ] GTS-T1 Generate Python golden files  Est: 4h
  Acceptance: 10 golden files checked in with input/output pairs.

- [ ] GTS-T2 Run Zerfoo against golden files  Est: 4h  Deps: GTS-T1
  Acceptance: All 10 test cases pass within 1e-4 tolerance.

- [ ] GTS-T3 Benchmark latency vs Python granite-tsfm  Est: 2h  Deps: GTS-T2
  Acceptance: Results recorded in devlog with comparison table.

### P5: MSA-Inspired Scalable Memory (E34)

Adopt three decomposable techniques from the MSA paper (EverMind-AI, 2025).
Decision rationale: docs/adr/067-msa-sparse-attention-memory.md

#### E34.1: ztensor Compute Primitives

- [x] T34.1.1 Add CosineSimilarity to Engine[T] (CPU)  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor. Input: a=[M,D], b=[N,D]. Output: [M,N].
  Acceptance: Unit test with known vectors passes within 1e-6.

- [x] T34.1.2 Add CosineSimilarity GPU kernel  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T34.1.1
  Acceptance: GPU matches CPU within 1e-4; benchmark on DGX Spark.

- [x] T34.1.3 Add MaxAxis reduction to Engine[T]  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor. Reuse SumAxis kernel pattern.
  Acceptance: Unit test on 3D tensor, CPU and GPU match.

- [ ] T34.1.4 Run go vet and tests for ztensor primitives  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T34.1.1, T34.1.2, T34.1.3

#### E34.2: KV Cache Compression

- [x] T34.2.1 Implement CompressedKVCache  Owner: TBD  Est: 4h  verifies: [UC-001, UC-002]
  File: generate/compressed_kv_cache.go. Chunk-wise mean pooling via ReduceMean.
  Acceptance: Store 128 tokens, chunkSize=64, Get() returns [batch, 2, dim].

- [x] T34.2.2 Add CompressedKVCache generator option  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T34.2.1. Add `WithCompressedKV(chunkSize)`.

- [ ] T34.2.3 Unit and integration tests for compressed cache  Owner: TBD  Est: 2h  verifies: [UC-001, UC-002]
  Deps: T34.2.1, T34.2.2

#### E34.3: Document-wise RoPE

- [x] T34.3.1 Add DocumentWiseRoPE mode  Owner: TBD  Est: 3h  verifies: [UC-001, UC-003]
  File: layers/embeddings/rotary_positional_embedding.go
  Acceptance: Position IDs reset at boundaries; query uses global offset.

- [x] T34.3.2 Wire DocumentWiseRoPE into GQA  Owner: TBD  Est: 2h  verifies: [UC-001, UC-003]
  Deps: T34.3.1

- [ ] T34.3.3 Tests for document-wise RoPE  Owner: TBD  Est: 2h  verifies: [UC-001, UC-003]
  Deps: T34.3.1, T34.3.2

#### E34.4: Sparse Routed Attention

- [x] T34.4.1 Implement SparseRoutedAttention layer  Owner: TBD  Est: 6h  verifies: [UC-003]
  Deps: T34.1.1, T34.1.3, T34.2.1, T34.3.1
  File: layers/attention/sparse_routed_attention.go

- [ ] T34.4.2 Implement auxiliary contrastive routing loss  Owner: TBD  Est: 3h  verifies: [UC-004]
  Deps: T34.4.1. File: training/loss/routing_contrastive.go

- [ ] T34.4.3 Register SparseRoutedAttention in layer registry  Owner: TBD  Est: 1h  verifies: [UC-003]
  Deps: T34.4.1

- [ ] T34.4.4 Unit tests for sparse routed attention  Owner: TBD  Est: 3h  verifies: [UC-003]
  Deps: T34.4.1, T34.4.2, T34.4.3

#### E34.5: Tiered KV Storage and Memory Parallel

- [x] T34.5.1 Implement TieredKVStore  Owner: TBD  Est: 4h  verifies: [UC-003]
  Deps: T34.2.1. File: generate/tiered_kv_store.go

- [ ] T34.5.2 Add async CPU-to-GPU fetch  Owner: TBD  Est: 2h  verifies: [UC-003]
  Deps: T34.5.1

- [ ] T34.5.3 Tests for tiered storage  Owner: TBD  Est: 2h  verifies: [UC-003]
  Deps: T34.5.1, T34.5.2

#### E34.6: Integration and Validation

- [ ] T34.6.1 Run go vet and linters on all new code  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T34.2.3, T34.3.3, T34.4.4, T34.5.3

- [ ] T34.6.2 Integration test: compressed cache with existing models  Owner: TBD  Est: 2h  verifies: [UC-001, UC-002]
  Deps: T34.6.1. Load Gemma3-1B with WithCompressedKV(64) on DGX Spark.

### P6: QuaRot + KVQuant -- Uniform 4-Bit Quantization (E35)

Apply Hadamard rotation at GGUF load time to eliminate outlier features,
enabling uniform 4-bit quantization of weights, activations, and KV cache.
3.3x prefill speedup, 3.9x memory savings.
Decision rationale: docs/adr/068-research-driven-inference-priorities.md

**References:**
- QuaRot (arXiv:2404.00456, NeurIPS 2024)
- KVQuant (UC Berkeley, NeurIPS 2024)

#### E35.1: Hadamard Rotation Infrastructure

- [x] T35.1.1 Implement Hadamard matrix generator  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor
  File: compute/hadamard.go
  Generate Walsh-Hadamard matrices of size 2^k using recursive Kronecker
  product. Normalize by 1/sqrt(N). Support dimensions 64, 128, 256, 512.
  Acceptance: H * H^T = I verified for all supported sizes within 1e-6.

- [x] T35.1.2 Add Hadamard transform to Engine[T]  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T35.1.1
  Files: compute/engine.go, compute/cpu_engine.go, compute/gpu_engine.go
  Add `HadamardTransform(ctx, a, dst) error`. CPU: dense matmul with
  precomputed H. GPU: fast Walsh-Hadamard transform kernel (O(N log N)).
  Acceptance: CPU and GPU match within 1e-4. Benchmark on DGX Spark.

- [ ] T35.1.3 Tests for Hadamard infrastructure  Owner: TBD  Est: 1h  verifies: [infrastructure]
  repo: ztensor  Deps: T35.1.1, T35.1.2
  Tests: orthogonality, inverse, dimension mismatch error.
  Acceptance: All pass on CPU and GPU.

#### E35.2: QuaRot Weight Rotation at Load Time

- [x] T35.2.1 Implement QuaRot weight fusion in GGUF loader  Owner: TBD  Est: 4h  verifies: [UC-001, UC-006]
  File: inference/quarot.go
  After loading GGUF weights, fuse Hadamard rotation into Q/K/V/O projection
  matrices and FFN gate/up/down matrices. Rotation is absorbed into weights
  via W_rotated = H * W -- zero runtime cost after fusion.
  Acceptance: Rotated model produces identical output to unrotated model
  within 1e-3 tolerance (rotation is information-preserving).

- [ ] T35.2.2 Add --quarot flag to CLI and LoadFile option  Owner: TBD  Est: 1h  verifies: [UC-001, UC-006]
  Deps: T35.2.1
  Files: inference/options.go, cmd/run.go
  Add `WithQuaRot()` option and `--quarot` CLI flag.
  Acceptance: `zerfoo run --quarot model.gguf` applies rotation at load time.

- [ ] T35.2.3 Tests for QuaRot weight fusion  Owner: TBD  Est: 2h  verifies: [UC-001, UC-006]
  Deps: T35.2.1, T35.2.2
  Test: load model with and without QuaRot, compare logits within tolerance.
  Acceptance: All tests pass.

#### E35.3: KV Cache Quantization

- [x] T35.3.1 Implement Q4 KV cache storage  Owner: TBD  Est: 4h  verifies: [UC-001, UC-002]
  File: generate/kvcache_q4.go
  New CacheProvider[T] that stores KV in 4-bit quantized format.
  On Update(): quantize incoming FP32/FP16 KV to Q4 with per-channel
  scale factors. On Get(): dequantize to FP32 for attention computation.
  Use group quantization (group_size=128) for accuracy.
  Acceptance: Unit test -- store and retrieve 256 tokens, dequantized
  values within 0.05 of originals (Q4 tolerance).

- [x] T35.3.2 Implement Q3 KV cache with non-uniform codebook  Owner: TBD  Est: 4h  verifies: [UC-001, UC-002]
  Deps: T35.3.1
  File: generate/kvcache_q3.go
  KVQuant-style 3-bit quantization with sensitivity-weighted k-means
  codebook (8 centroids per group). Pre-RoPE key quantization for
  better accuracy (quantize keys before RoPE rotation).
  Acceptance: Perplexity degradation < 0.5 on a reference model.

- [ ] T35.3.3 Add quantized KV cache generator options  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T35.3.1, T35.3.2
  Add `WithKVDtype("q4")` and `WithKVDtype("q3")` GeneratorOptions.
  Acceptance: Options create correct cache type. Existing FP16/FP32 unchanged.

- [x] T35.3.4 GPU kernels for Q4/Q3 KV dequantization  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-03-27
  repo: ztensor  Deps: T35.3.1
  CUDA kernels for fused dequant-and-gather during attention.
  Acceptance: GPU path matches CPU within Q4 tolerance.

- [ ] T35.3.5 Tests and benchmarks for quantized KV cache  Owner: TBD  Est: 2h  verifies: [UC-001, UC-002]
  Deps: T35.3.1, T35.3.2, T35.3.3, T35.3.4
  Measure: memory reduction ratio, perplexity impact, tok/s impact.
  Acceptance: >= 3x memory reduction, < 1% quality degradation.

- [ ] T35.3.6 Run go vet and linters for E35  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T35.2.3, T35.3.5

### P7: EAGLE-3 Self-Speculative Decoding (E36)

Replace two-model speculative decoding with a lightweight prediction head
that reuses penultimate-layer features. 3-6.5x speedup, no draft model.
Decision rationale: docs/adr/068-research-driven-inference-priorities.md

**Reference:** EAGLE-3 (arXiv:2503.01840, March 2025)

#### E36.1: Prediction Head

- [x] T36.1.1 Implement EAGLEHead layer  Owner: TBD  Est: 4h  verifies: [UC-001, UC-007]
  File: layers/core/eagle_head.go
  Lightweight FFN that takes the penultimate transformer layer's hidden
  state and autoregressively predicts the next hidden state (not token).
  Architecture: LayerNorm -> Linear(hidden, hidden) -> SiLU -> Linear(hidden, hidden).
  The predicted hidden state is fed through the target model's LM head
  to get draft token logits.
  Acceptance: Forward produces output of shape [batch, 1, hidden].

- [x] T36.1.2 Wire EAGLEHead into computation graph  Owner: TBD  Est: 3h  verifies: [UC-001, UC-007]
  Deps: T36.1.1
  File: inference/eagle.go
  Extract the penultimate layer's output during decode. Feed it to
  EAGLEHead to generate N draft hidden states autoregressively.
  Apply the existing LM head to each draft hidden state to get draft tokens.
  Acceptance: Draft token generation produces valid token IDs.

- [ ] T36.1.3 Tests for EAGLE head  Owner: TBD  Est: 2h  verifies: [UC-001, UC-007]
  Deps: T36.1.1, T36.1.2

#### E36.2: EAGLE Speculative Decode Loop

- [x] T36.2.1 Implement EAGLE decode loop  Owner: TBD  Est: 4h  verifies: [UC-001, UC-007]
  Deps: T36.1.2
  File: generate/eagle_speculative.go
  Modified speculative decode loop:
  1. Run target model forward for current token, capture penultimate features.
  2. Feed features to EAGLEHead, generate N draft tokens autoregressively.
  3. Verify all N drafts in one batched target forward.
  4. Accept prefix of matching tokens, reject rest.
  5. Adaptively adjust N based on rolling acceptance rate (reuse adaptive.go).
  Acceptance: Generates identical output to vanilla autoregressive (greedy).

- [ ] T36.2.2 Add WithEAGLE generator option  Owner: TBD  Est: 1h  verifies: [UC-001, UC-007]
  Deps: T36.2.1
  File: generate/generator.go
  `WithEAGLE(headWeightsPath string)` loads EAGLE head weights and enables
  the EAGLE decode loop. Falls back to vanilla if head weights not found.
  Acceptance: Option creates EAGLE-enabled generator.

- [x] T36.2.3 EAGLE head weight loading from GGUF  Owner: TBD  Est: 2h  verifies: [UC-001, UC-007]
  Deps: T36.1.1
  File: inference/eagle.go
  Load EAGLE head weights from a separate GGUF file or from extra tensors
  in the main model's GGUF (if present under "eagle." prefix).
  Acceptance: Weights loaded and shapes validated.

- [ ] T36.2.4 Benchmark EAGLE vs vanilla autoregressive  Owner: TBD  Est: 2h  verifies: [UC-001, UC-007]
  Deps: T36.2.1
  Compare tok/s with and without EAGLE on Gemma3-1B and Llama3.2-3B.
  Target: >= 2x speedup with EAGLE head trained on the same model.
  Acceptance: Results in devlog with speedup ratios.

- [ ] T36.2.5 Tests for EAGLE decode loop  Owner: TBD  Est: 2h  verifies: [UC-001, UC-007]
  Deps: T36.2.1, T36.2.2, T36.2.3

- [ ] T36.2.6 Run go vet and linters for E36  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T36.2.5

### P8: NSA -- Native Sparse Attention (E37)

Hardware-aligned sparse attention with three paths: coarse compression,
fine-grained selection, and sliding window. ACL 2025 Best Paper.
Decision rationale: docs/adr/068-research-driven-inference-priorities.md

**Reference:** NSA (arXiv:2502.11089, DeepSeek, February 2025)

#### E37.1: Three-Path Attention Kernel

- [x] T37.1.1 Implement coarse-grained token compression path  Owner: TBD  Est: 3h  verifies: [UC-001, UC-003]
  File: layers/attention/nsa_coarse.go
  Divide KV sequence into blocks of B tokens. Compute block-level
  attention scores by averaging key representations per block. Select
  top-c blocks. Attend to selected blocks with full resolution.
  Acceptance: Forward produces correct output shape. Block selection
  is differentiable via straight-through estimator.

- [x] T37.1.2 Implement fine-grained token selection path  Owner: TBD  Est: 3h  verifies: [UC-001, UC-003]
  File: layers/attention/nsa_fine.go
  Score individual tokens using Q*K similarity. Select top-f tokens
  per query position via top-k. Attend to selected tokens.
  Acceptance: Correct output. Selected indices match manual computation.

- [x] T37.1.3 Implement sliding window path  Owner: TBD  Est: 1h  verifies: [UC-001]
  File: layers/attention/nsa_window.go
  Reuse existing LocalAttention with configurable window size.
  Output is the standard sliding-window attention result.
  Acceptance: Matches existing LocalAttention output exactly.

- [x] T37.1.4 Combine three paths into NativeSparseAttention  Owner: TBD  Est: 3h  verifies: [UC-001, UC-003]
  Deps: T37.1.1, T37.1.2, T37.1.3
  File: layers/attention/native_sparse_attention.go
  Combine outputs: O = gate_coarse * O_coarse + gate_fine * O_fine + gate_window * O_window.
  Gates are learned sigmoid scalars per head.
  Constructor: `NewNativeSparseAttention[T](engine, ops, modelDim, numHeads,
  numKVHeads, blockSize, topBlocks, topTokens, windowSize, ...opts)`
  Acceptance: Forward pass with all three paths active. Gradients flow.

- [x] T37.1.5 CUDA kernel for fused NSA  Owner: TBD  Est: 6h  verifies: [infrastructure]  DONE 2026-03-27
  Deps: T37.1.4
  repo: ztensor
  File: internal/cuda/nsa_attention.cu
  Fused kernel that performs all three paths in a single GPU launch.
  Coarse scoring uses warp-level reduction. Fine selection uses shared
  memory for top-k. Window path uses register-tiled matmul.
  Acceptance: Fused kernel matches decomposed path within 1e-4. Benchmark
  shows >= 1.5x speedup over decomposed on DGX Spark.

#### E37.2: Registration and Testing

- [ ] T37.2.1 Register NativeSparseAttention in layer registry  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T37.1.4
  Acceptance: Builder creates valid layer from GGUF attributes.

- [ ] T37.2.2 Unit tests for NSA  Owner: TBD  Est: 3h  verifies: [UC-001, UC-003]
  Deps: T37.1.4
  Tests: (1) Each path independently correct. (2) Combined output correct.
  (3) Degenerate: full selection = dense attention. (4) Backward pass.
  Acceptance: All pass CPU and GPU.

- [ ] T37.2.3 Run go vet and linters for E37  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T37.2.2

### P9: CPU/GPU Hybrid MoE Inference (E38)

Place shared MoE experts on GPU, offload routed experts to CPU with
SIMD kernels. Async scheduling hides transfer latency. Enables running
DeepSeek-V3 671B on single GPU + host CPU.
Decision rationale: docs/adr/068-research-driven-inference-priorities.md

**Reference:** KTransformers (SOSP 2025)

#### E38.1: Expert Placement Strategy

- [x] T38.1.1 Implement ExpertPlacementPolicy  Owner: TBD  Est: 3h  verifies: [UC-001, UC-008]
  File: inference/moe_placement.go
  Policy that decides which experts run on GPU vs CPU based on routing
  frequency. Shared experts (always active) -> GPU. Routed experts
  (sparse activation) -> CPU. Configurable via threshold.
  Acceptance: Policy assigns experts to devices given routing statistics.

- [x] T38.1.2 Split MoE weights between GPU and CPU  Owner: TBD  Est: 3h  verifies: [UC-001, UC-008]
  Deps: T38.1.1
  File: inference/moe_loader.go
  During GGUF loading, upload shared expert weights to GPU, keep routed
  expert weights in CPU memory (mmap or heap). Build a device map
  mapping expert_id -> device.
  Acceptance: Device map correctly partitions experts.

- [ ] T38.1.3 Tests for expert placement  Owner: TBD  Est: 1.5h  verifies: [UC-001, UC-008]
  Deps: T38.1.1, T38.1.2

#### E38.2: Async CPU Expert Execution

- [ ] T38.2.1 Implement async CPU expert dispatch  Owner: TBD  Est: 4h  verifies: [UC-001, UC-008]
  Deps: T38.1.2
  File: inference/moe_async.go
  When MoE layer routes tokens to CPU-resident experts, dispatch the
  GEMM to a goroutine pool using xblas SIMD kernels. Meanwhile, GPU
  executes shared experts. Gather results when both complete.
  Use sync.WaitGroup for synchronization.
  Acceptance: Async dispatch produces identical output to serial execution.

- [ ] T38.2.2 Implement prefetch for predicted expert activation  Owner: TBD  Est: 3h  verifies: [UC-001, UC-008]
  Deps: T38.2.1
  File: inference/moe_prefetch.go
  Predict which experts will be needed in the next layer based on
  current routing decisions (experts tend to be sticky across layers).
  Begin CPU->GPU transfer of predicted expert weights asynchronously.
  Acceptance: Prefetch hit rate > 60% on DeepSeek-V3 routing patterns.

- [ ] T38.2.3 Tests and benchmarks for hybrid MoE  Owner: TBD  Est: 2h  verifies: [UC-001, UC-008]
  Deps: T38.2.1, T38.2.2
  Benchmark on DeepSeek-R1-1.5B (small MoE). Measure tok/s with
  hybrid vs GPU-only. Verify correctness via output comparison.
  Acceptance: Correct output. Performance documented in devlog.

- [ ] T38.2.4 Run go vet and linters for E38  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T38.2.3

### P10: BitNet b1.58 Ternary Inference (E39)

Support ternary weight models {-1, 0, 1} where matrix multiplication
becomes integer addition/subtraction. Eliminates GEMM entirely.
Decision rationale: docs/adr/068-research-driven-inference-priorities.md

**Reference:** BitNet b1.58 (arXiv:2402.17764, Microsoft, February 2024)

#### E39.1: Ternary Storage and Compute

- [x] T39.1.1 Implement TernaryStorage in ztensor  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor
  File: tensor/ternary_storage.go
  Pack ternary weights {-1, 0, 1} into 2 bits per weight. Storage
  format: value = 2*bit1 + bit0 - 1 (00=-1, 01=0, 10=1).
  Methods: Len(), Get(i), Set(i, val), Slice() (dequantize to float).
  Acceptance: Pack/unpack roundtrip preserves all values. 4x smaller
  than float32.

- [x] T39.1.2 Implement ternary GEMV (CPU)  Owner: TBD  Est: 3h  verifies: [UC-001, UC-009]
  repo: ztensor  Deps: T39.1.1
  File: compute/ternary_gemv.go
  Matrix-vector multiply where weight matrix is ternary: y[i] = sum of
  x[j] where w[i,j]=1, minus sum of x[j] where w[i,j]=-1, skip where
  w[i,j]=0. No floating-point multiply needed.
  ARM NEON: use bit-masking to select addends. x86 AVX2: similar.
  Acceptance: Output matches dense float32 GEMV within 1e-6.
  Benchmark: >= 2x faster than float32 GEMV on CPU.

- [x] T39.1.3 Implement ternary GEMV GPU kernel  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T39.1.1
  File: internal/cuda/ternary_gemv.cu
  CUDA kernel: each thread processes a row, uses bit operations to
  select positive/negative addends, accumulates with warp reduction.
  Acceptance: GPU matches CPU. Benchmark on DGX Spark.

- [ ] T39.1.4 Tests for ternary storage and GEMV  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor  Deps: T39.1.1, T39.1.2, T39.1.3

#### E39.2: BitNet Model Support

- [x] T39.2.1 Add TernaryStorage to GGUF loader  Owner: TBD  Est: 2h  verifies: [UC-001, UC-009]
  Deps: T39.1.1
  File: inference/load_gguf.go
  Recognize GGUF tensor type for ternary/1.58-bit weights. Create
  TernaryStorage and wire to ternary GEMV path in MatMul layer.
  Acceptance: GGUF with ternary tensors loads without error.

- [ ] T39.2.2 Wire ternary MatMul into computation graph  Owner: TBD  Est: 2h  verifies: [UC-001, UC-009]
  Deps: T39.2.1, T39.1.2
  File: layers/core/matmul.go
  When weight tensor has TernaryStorage, dispatch to ternary GEMV
  instead of standard MatMul. Transparent to architecture builders.
  Acceptance: Inference with ternary model produces coherent output.

- [ ] T39.2.3 Tests for BitNet model loading  Owner: TBD  Est: 2h  verifies: [UC-001, UC-009]
  Deps: T39.2.1, T39.2.2

- [ ] T39.2.4 Run go vet and linters for E39  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T39.2.3

### P11: TransMLA / MHA2MLA -- Retrofit MLA onto MHA Models (E40)

Convert standard MHA/GQA models (Llama, Gemma, Qwen, Mistral, Phi) to use
Multi-Head Latent Attention via SVD-based weight decomposition. Reuses the
existing MLA layer from DeepSeek V3. Reduces KV cache by 93.3%, boosts
generation throughput 5.76x. Conversion is offline; inference uses existing code.
Decision rationale: docs/adr/069-transmla-mha-to-mla-conversion.md

**References:**
- TransMLA (arXiv:2502.07864, February 2025)
- MHA2MLA (arXiv:2502.14837, February 2025)
- DeepSeek-V2 MLA (arXiv:2405.04434, May 2024)

#### E40.1: SVD Weight Conversion Tool

- [x] T40.1.1 Implement SVD decomposition for KV projections  Owner: TBD  Est: 4h  verifies: [UC-010]
  File: inference/transmla/convert.go
  For each transformer layer, compute SVD of the concatenated K/V projection
  matrix W_KV = [W_K; W_V] of shape [2*numKVHeads*headDim, hidden].
  Truncate to rank r (configurable, default 512 for hidden >= 4096).
  Produce three matrices:
  - wDKV: [hidden, r] (down-projection = V_r^T from SVD)
  - wUK: [r, numKVHeads*headDim] (up-projection for keys = U_k * S_k)
  - wUV: [r, numKVHeads*headDim] (up-projection for values = U_v * S_v)
  Use standard library math for SVD (or implement Golub-Kahan bidiagonalization).
  Acceptance: Reconstructed W_KV = wUK^T * wDKV^T + wUV^T * wDKV^T
  approximates original within truncation error.

- [x] T40.1.2 Write converted weights as GGUF  Owner: TBD  Est: 2h  verifies: [UC-010]
  Deps: T40.1.1
  File: inference/transmla/convert.go
  Write a new GGUF file containing:
  - All original tensors (embeddings, FFN, norms, LM head) unchanged.
  - Original Q projection and output projection unchanged.
  - Replace W_K and W_V with transmla.{layer}.wDKV, transmla.{layer}.wUK,
    transmla.{layer}.wUV tensors.
  - Add metadata: transmla.kv_lora_dim = r, transmla.source_arch = original arch.
  Use ztensor/gguf writer.
  Acceptance: Output GGUF loads without error. Tensor shapes match expected.

- [ ] T40.1.3 Implement transmla CLI command  Owner: TBD  Est: 1.5h  verifies: [UC-010]
  Deps: T40.1.2
  File: cmd/transmla/main.go
  CLI: `zerfoo transmla --rank 512 --input model.gguf --output model-mla.gguf`
  Flags: --rank (latent dimension), --input, --output.
  Show progress per layer. Report compression ratio on completion.
  Acceptance: CLI converts Gemma3-1B GGUF and reports compression ratio.

- [ ] T40.1.4 Tests for SVD conversion  Owner: TBD  Est: 2h  verifies: [UC-010]
  Deps: T40.1.1, T40.1.2, T40.1.3
  File: inference/transmla/convert_test.go
  Tests: (1) SVD decomposition roundtrip for small matrix within tolerance.
  (2) Rank truncation preserves top singular values. (3) GGUF output contains
  correct tensor names and shapes. (4) CLI flag parsing.
  Acceptance: All tests pass.

#### E40.2: Automatic MLA Inference Path

- [ ] T40.2.1 Detect TransMLA tensors in GGUF and wire MLA layer  Owner: TBD  Est: 3h  verifies: [UC-001, UC-010]
  Deps: T40.1.2
  File: inference/arch_common.go
  In `buildTransformerGraph()`, check if the GGUF contains tensors with
  "transmla." prefix for the current layer. If present, construct
  `MultiHeadLatentAttention` (existing layer) with wDKV, wUK, wUV from
  the transmla tensors instead of constructing GQA with separate W_K, W_V.
  Pass kvLoraDim from transmla.kv_lora_dim metadata.
  Acceptance: Converted GGUF loads and uses MLA path. Original GGUF
  continues to use GQA path. No regression on DeepSeek V3 MLA.

- [ ] T40.2.2 Validate MLA KV cache stores latent vector  Owner: TBD  Est: 1.5h  verifies: [UC-001, UC-010]
  Deps: T40.2.1
  File: layers/attention/multi_head_latent_attention.go
  Verify that when MLA is used, the KV cache stores the compressed latent
  vector (dimension = kvLoraDim) not the full K/V. The existing MLA
  Forward() should already do this -- this task validates and fixes if not.
  Measure KV cache memory with and without TransMLA conversion.
  Acceptance: KV cache dimension equals kvLoraDim. Memory reduction >= 80%.

- [ ] T40.2.3 Integration test: TransMLA end-to-end  Owner: TBD  Est: 2h  verifies: [UC-001, UC-010]
  Deps: T40.2.1, T40.2.2
  File: tests/integration/transmla_test.go
  Pipeline: (1) Load original GGUF (Gemma3-1B). (2) Convert with rank=256.
  (3) Load converted GGUF. (4) Generate 64 tokens. (5) Compare output
  against original model -- should be similar but not identical due to
  SVD truncation. (6) Verify KV cache memory reduction.
  Acceptance: Converted model produces coherent output. KV cache >= 4x smaller.

- [ ] T40.2.4 Perplexity validation script  Owner: TBD  Est: 1.5h  verifies: [UC-010]
  Deps: T40.2.1
  File: cmd/transmla/validate.go
  `zerfoo transmla validate --original model.gguf --converted model-mla.gguf`
  Run both models on a fixed evaluation set (256 tokens from wikitext).
  Report perplexity for each. Flag if delta > 0.5 ppl.
  Acceptance: Script runs and reports perplexity comparison.

- [ ] T40.2.5 Run go vet and linters for E40  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T40.1.4, T40.2.3, T40.2.4

### P12: GGUF I-Quant Dequantization (E41)

Add dequantization kernels for importance-weighted quantization types
(IQ2_XXS, IQ3_S, IQ4_NL) used by llama.cpp. Without these, Zerfoo cannot
load a large fraction of community-quantized models. Minimal kernel work --
just dequantization lookup tables and GEMV dispatch.

- [x] T41.1.1 Implement IQ4_NL dequantization (CPU)  Owner: TBD  Est: 2h  verifies: [UC-001]
  repo: ztensor
  File: tensor/iq4_storage.go
  IQ4_NL uses non-linear 4-bit quantization with a 16-entry lookup table.
  Dequant: output[i] = table[nibble[i]] * scale.
  Acceptance: Roundtrip test with known values within Q4 tolerance.

- [x] T41.1.2 Implement IQ3_S dequantization (CPU)  Owner: TBD  Est: 2h  verifies: [UC-001]
  repo: ztensor
  File: tensor/iq3_storage.go
  IQ3_S uses 3-bit quantization with importance-weighted scales per super-block.
  Acceptance: Dequantized values match llama.cpp reference within tolerance.

- [x] T41.1.3 Implement IQ2_XXS dequantization (CPU)  Owner: TBD  Est: 3h  verifies: [UC-001]
  repo: ztensor
  File: tensor/iq2_storage.go
  IQ2_XXS uses 2-bit quantization with grid-based codebook lookup.
  Most complex I-Quant format. Requires precomputed grid tables.
  Acceptance: Dequantized values match llama.cpp reference.

- [x] T41.1.4 GPU dequantization kernels for IQ types  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-03-27
  repo: ztensor  Deps: T41.1.1, T41.1.2, T41.1.3
  Files: internal/cuda/iq_dequant.cu, internal/gpuapi/kernels.go
  CUDA kernels: table lookup dequant for IQ4_NL, IQ3_S, IQ2_XXS.
  Acceptance: GPU matches CPU within tolerance.

- [x] T41.1.5 Wire I-Quant types into GGUF tensor loader  Owner: TBD  Est: 1.5h  verifies: [UC-001]  DONE 2026-03-27
  Deps: T41.1.1, T41.1.2, T41.1.3
  Files: inference/load_gguf.go, model/gguf/
  Recognize GGUF tensor types for IQ2_XXS, IQ3_S, IQ4_NL. Create
  corresponding storage and dispatch to I-Quant GEMV path.
  Acceptance: GGUF with IQ4_NL tensors loads and runs inference.

- [ ] T41.1.6 Tests for I-Quant dequantization  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T41.1.1-T41.1.5
  Test each format: pack, dequant, GEMV, roundtrip accuracy.
  Acceptance: All tests pass CPU and GPU.

- [ ] T41.1.7 Run go vet and linters for E41  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T41.1.6

### P13: RadixAttention KV Cache Upgrade (E42)

Upgrade the existing PrefixCache to a production-grade RadixAttention
system (SGLang, NeurIPS 2024). Adds LRU eviction, hash-based block
matching, and cache-aware request scheduling. Up to 6.4x throughput
improvement for shared-prefix workloads (system prompts, few-shot).

**Reference:** SGLang (arXiv:2312.07104, NeurIPS 2024)

- [x] T42.1.1 Implement hash-based radix tree for KV blocks  Owner: TBD  Est: 4h  verifies: [UC-002, UC-011]
  File: generate/radix_cache.go
  Replace the current PrefixCache radix tree with a hash-based version.
  Each node stores a hash of the token subsequence it represents.
  Matching is O(prefix_length / block_size) -- amortized O(1) per block.
  LRU eviction: track last-access time per leaf, evict coldest when full.
  Acceptance: Insert 1000 prefixes, match returns correct blocks.
  LRU eviction frees blocks when pool is exhausted.

- [x] T42.1.2 Add cache-aware request scheduling  Owner: TBD  Est: 3h  verifies: [UC-002, UC-011]
  Deps: T42.1.1
  File: serve/batcher/scheduler.go
  When multiple requests are queued, sort by prefix cache hit ratio
  (longest match / total prompt length). Requests with high cache hits
  get priority -- they skip most prefill computation.
  Acceptance: Scheduler prefers requests with cached prefixes.
  Throughput improvement measurable on repeated-prefix workload.

- [ ] T42.1.3 Tests for radix cache and scheduling  Owner: TBD  Est: 2h  verifies: [UC-002, UC-011]
  Deps: T42.1.1, T42.1.2
  Tests: (1) Hash collision handling. (2) LRU eviction correctness.
  (3) Scheduler ordering with mixed cache hit ratios.
  (4) Backward compatibility: existing PrefixCache API still works.
  Acceptance: All tests pass.

- [ ] T42.1.4 Run go vet and linters for E42  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T42.1.3

### P14: Flash Decoding (E43)

Implement split-KV decode attention (Flash Decoding / FlashAttention-3
pattern). During autoregressive decode, the single query token attends
to the full KV cache. Standard attention is memory-bound here. Flash
Decoding splits the KV across thread blocks, each computing partial
softmax, then reduces. Maximizes GPU memory bandwidth utilization.

**Reference:** FlashAttention-3 (arXiv:2407.08608, NeurIPS 2024)

- [x] T43.1.1 Implement split-KV decode kernel (CUDA)  Owner: TBD  Est: 6h  verifies: [UC-001, UC-002]
  repo: ztensor
  File: internal/cuda/flash_decode.cu
  For decode (seqLen_Q=1): split KV cache across S thread blocks.
  Each block computes partial attention: local_softmax(Q * K_chunk^T) * V_chunk.
  Reduce across blocks using online softmax (log-sum-exp correction).
  Grid: [numHeads, S] where S = ceil(seqLen_KV / chunk_size).
  Acceptance: Output matches standard attention within 1e-4.
  Benchmark: >= 1.5x speedup on seqLen_KV > 1024 vs current decode kernel.

- [x] T43.1.2 Wire flash decode into attention layer  Owner: TBD  Est: 2h  verifies: [UC-001]  DONE 2026-03-27
  Deps: T43.1.1
  File: layers/attention/scaled_dot_product_attention.go
  In tryFlashForward(): when seqLen_Q == 1 (decode mode), dispatch to
  flash_decode kernel instead of standard attention. Fall back to
  standard path on CPU or when conditions not met.
  Acceptance: Decode uses flash decode kernel on GPU. Prefill unchanged.

- [ ] T43.1.3 Tests and benchmarks for flash decode  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T43.1.1, T43.1.2
  Test: output correctness at various KV lengths (128, 1024, 8192, 32768).
  Benchmark: tok/s comparison with and without flash decode on Gemma3-1B.
  Acceptance: All tests pass. Benchmark results in devlog.

- [ ] T43.1.4 Run go vet and linters for E43  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T43.1.3

### P15: Multi-LoRA Serving (E44)

Serve multiple LoRA adapters from a single base model simultaneously.
Each request specifies an adapter name; the server loads/unloads adapters
on demand. Adapter weights managed via existing arena allocator.

**Reference:** S-LoRA (arXiv:2311.03285), LoRAFusion (arXiv:2510.00206)

- [x] T44.1.1 Implement LoRA adapter weight format  Owner: TBD  Est: 2h  verifies: [UC-012]
  File: inference/lora/adapter.go
  Load LoRA adapter weights from a GGUF file. Adapter contains delta
  matrices A and B per layer: W_adapted = W_base + B * A.
  Rank, alpha, and target layers stored in GGUF metadata.
  Acceptance: Adapter loads from GGUF. Shapes validated per layer.

- [x] T44.1.2 Implement LoRA weight merging in forward pass  Owner: TBD  Est: 3h  verifies: [UC-012]  DONE 2026-03-27
  Deps: T44.1.1
  File: inference/lora/apply.go
  During forward pass, apply LoRA: output = base_output + scale * (x @ A^T @ B^T).
  This is two small matmuls per adapted layer (rank << hidden_dim).
  Adapter is applied on-the-fly, not merged into base weights (enables
  per-request adapter selection).
  Acceptance: Forward with adapter matches merged-weight forward within 1e-5.

- [x] T44.1.3 Implement adapter cache with LRU eviction  Owner: TBD  Est: 2h  verifies: [UC-002, UC-012]  DONE 2026-03-27
  Deps: T44.1.1
  File: inference/lora/cache.go
  Cache loaded adapters in GPU memory. LRU eviction when cache exceeds
  configured max adapters. On cache miss, load adapter from disk.
  Acceptance: Cache hit returns adapter instantly. Eviction frees GPU memory.

- [ ] T44.1.4 Add adapter selection to serve API  Owner: TBD  Est: 2h  verifies: [UC-002, UC-012]
  Deps: T44.1.2, T44.1.3
  File: serve/server.go
  Add optional `model` field in ChatCompletionRequest that specifies
  "base_model:adapter_name" format (e.g., "gemma3-1b:my-lora").
  Server loads adapter on first use, caches for subsequent requests.
  Acceptance: Two requests with different adapters produce different outputs.
  Request without adapter uses base model.

- [ ] T44.1.5 Tests for Multi-LoRA serving  Owner: TBD  Est: 2h  verifies: [UC-002, UC-012]
  Deps: T44.1.1-T44.1.4
  Tests: (1) Adapter load/unload. (2) LRU eviction. (3) Per-request
  adapter selection via API. (4) Concurrent requests with different adapters.
  Acceptance: All tests pass.

- [ ] T44.1.6 Run go vet and linters for E44  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T44.1.5

---

## Backlog

### Community and DevRel

- [ ] T4.7 Record 15-minute video walkthrough of Zerfoo  Est: 4h
- [ ] T5.4 Create Discord server with channels  Est: 2h
- [ ] T11.1 Sponsor GopherCon 2027 booth  Est: 2h
- [ ] T11.4 Recruit 5 external co-maintainers  Est: 4h

### ROCm Backend (E8) [Q1-Q3 2027]

- [ ] T8.1 Acquire AMD Instinct GPU access  Est: 2h
- [ ] T8.2 Validate all purego HIP bindings on AMD hardware  Est: 6h
- [ ] T8.3 Validate rocBLAS GEMM parity with cuBLAS  Est: 4h
- [ ] T8.4 Port custom CUDA kernels to HIP in ztensor  Est: 8h
- [ ] T8.5 Benchmark ROCm vs CUDA throughput  Est: 2h
- [ ] T8.6 Add ROCm to CI pipeline  Est: 3h

### Enterprise and Compliance

- [ ] T12.4 Sign first 5 enterprise support contracts ($500K ARR)
- [ ] T19.1 Complete SOC 2 Type II audit

### Edge Deployment (E15)

- [ ] T15.3 Cross-compile and test on Raspberry Pi 5  Est: 3h
- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano  Est: 3h

### Performance

- [ ] T16.3 Benchmark 500+ tok/s (needs A100/H100, GB10 roofline ~257)  Est: 2h
- [ ] T9.4 Multi-GPU inference benchmark on Llama 3 70B (needs multi-GPU)  Est: 2h
- [ ] T20.3 Benchmark Metal vs CPU on Apple M4 Max  Est: 2h
- [ ] T29.4 Benchmark on-device inference  Est: 2h

### ZerfooConf (E26)

- [ ] T26.2 Execute ZerfooConf Day  Est: 8h
- [ ] T26.3 Plan standalone ZerfooConf 2032  Est: 6h

### Architecture Expansion (E32)

- [ ] T32.2 Validate 100+ model architectures

### FedRAMP (E30) [Q1-Q4 2034]

- [ ] T30.1 Engage FedRAMP 3PAO  Est: 4h
- [ ] T30.2 Implement FedRAMP controls (NIST 800-53)  Est: 12h
- [ ] T30.3 Complete FedRAMP authorization  Est: 4h

### IPO Preparation (E31) [Q1-Q4 2035]

- [ ] T31.1 Form board of directors
- [ ] T31.2 Engage Big 4 audit firm
- [ ] T31.3 Hire VP Sales and VP Marketing
- [ ] T31.4 Achieve $150M+ ARR
- [ ] T31.5 Draft S-1 registration

---

## Parallel Work

### Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A: ztensor Primitives | T34.1.*, T35.1.*, T39.1.* | New compute ops (cosine sim, max, Hadamard, ternary) |
| B: KV Cache Compression | T34.2.* | Compressed cache in generate/ |
| C: Document-wise RoPE | T34.3.* | RoPE mode in layers/ |
| D: Sparse Routed Attention | T34.4.* | MSA layer in layers/attention/ |
| E: Tiered Storage | T34.5.* | GPU/CPU tiered KV store |
| F: QuaRot Fusion | T35.2.* | Hadamard weight rotation at load time |
| G: KV Cache Quantization | T35.3.* | Q4/Q3 KV cache |
| H: EAGLE Head | T36.1.*, T36.2.* | Self-speculative decoding |
| I: NSA Attention | T37.1.*, T37.2.* | Three-path sparse attention |
| J: Hybrid MoE | T38.1.*, T38.2.* | CPU/GPU expert placement |
| K: BitNet | T39.1.*, T39.2.* | Ternary storage and inference |
| P: TransMLA | T40.1.*, T40.2.* | MHA-to-MLA conversion |
| Q: I-Quants | T41.1.* | IQ2/IQ3/IQ4 dequant kernels (ztensor) |
| R: RadixAttention | T42.1.* | Production KV prefix cache + scheduling |
| S: Flash Decode | T43.1.* | Split-KV decode kernel (ztensor) |
| T: Multi-LoRA | T44.1.* | Per-request adapter serving |
| L: Benchmarks | BMK-T1, BMK-T3, BMK-T4 | Model benchmarks |
| M: Mistral | MHH-T1, MHH-T2, MHH-T3 | Mistral head-to-head |
| N: K-Quant | KQ-T1, KQ-T2, KQ-T3 | Kernel optimization (ztensor) |
| O: Granite TS | GTS-T1, GTS-T2, GTS-T3 | Time-series parity |

### Sync Points

- Track D depends on Tracks A + B + C.
- Track E depends on Track B.
- Track F depends on Track A (Hadamard).
- Track G depends on Track A (Q4 dequant kernel).
- Track H is fully independent.
- Track I is fully independent (can run in parallel with D).
- Track J is fully independent.
- Track K depends on Track A (ternary storage).
- Track P (TransMLA) is fully independent -- reuses existing MLA layer.
- Tracks Q (I-Quants), R (RadixAttention), S (Flash Decode), T (Multi-LoRA) are
  all fully independent of each other and of E34-E40.
- Tracks L, M, N, O are fully independent of all research tracks.

### Waves

#### Wave 1: Foundation (10 agents)

All zero-dependency tasks. Saturates all agent slots.

- [x] T34.1.1 CosineSimilarity CPU (ztensor)  verifies: [infrastructure]
- [x] T34.1.3 MaxAxis reduction (ztensor)  verifies: [infrastructure]
- [x] T35.1.1 Hadamard matrix generator (ztensor)  verifies: [infrastructure]
- [x] T39.1.1 TernaryStorage (ztensor)  verifies: [infrastructure]
- [x] T34.2.1 CompressedKVCache  verifies: [UC-001, UC-002]
- [x] T34.3.1 DocumentWiseRoPE mode  verifies: [UC-001, UC-003]
- [x] T36.1.1 EAGLEHead layer  verifies: [UC-001, UC-007]
- [x] T37.1.1 Implement coarse compression path  verifies: [UC-001, UC-003]
- [x] T37.1.2 Implement fine selection path  verifies: [UC-001, UC-003]
- [x] T37.1.3 NSA sliding window path  verifies: [UC-001]

Note: E41-E44 tasks with no dependencies can start in any wave with
available agent slots. Distributed across waves to stay under 10 per wave.

#### Wave 2: Kernels + Wiring (10 agents)

- [x] T34.1.2 CosineSimilarity GPU kernel (ztensor)  Deps: T34.1.1
- [x] T35.1.2 Hadamard transform Engine[T] (ztensor)  Deps: T35.1.1
- [x] T39.1.2 Ternary GEMV CPU (ztensor)  Deps: T39.1.1
- [x] T39.1.3 Ternary GEMV GPU kernel (ztensor)  Deps: T39.1.1
- [x] T34.2.2 CompressedKVCache generator option  Deps: T34.2.1
- [x] T34.3.2 Wire DocumentWiseRoPE into GQA  Deps: T34.3.1
- [x] T36.1.2 Wire EAGLEHead into graph  Deps: T36.1.1
- [x] T37.1.4 Combine NSA three paths  Deps: T37.1.1, T37.1.2, T37.1.3
- [x] T38.1.1 ExpertPlacementPolicy  verifies: [UC-001, UC-008]
- [x] T35.3.1 Q4 KV cache storage  verifies: [UC-001, UC-002]
- [x] T40.1.1 SVD decomposition for KV projections  verifies: [UC-010]
- [x] T41.1.1 IQ4_NL dequantization CPU (ztensor)  verifies: [UC-001]
- [x] T41.1.2 IQ3_S dequantization CPU (ztensor)  verifies: [UC-001]
- [x] T41.1.3 IQ2_XXS dequantization CPU (ztensor)  verifies: [UC-001]
- [x] T42.1.1 Hash-based radix tree for KV blocks  verifies: [UC-002, UC-011]
- [x] T43.1.1 Split-KV flash decode kernel (ztensor)  verifies: [UC-001, UC-002]
- [x] T44.1.1 LoRA adapter weight format  verifies: [UC-012]

#### Wave 3: Integration Layer (10 agents)

- [x] T35.2.1 QuaRot weight fusion in GGUF loader  Deps: T35.1.2
- [x] T35.3.2 Q3 KV cache with codebook  Deps: T35.3.1
- [x] T35.3.4 GPU kernels Q4/Q3 KV dequant (ztensor)  Deps: T35.3.1
- [x] T36.2.1 EAGLE decode loop  Deps: T36.1.2
- [x] T36.2.3 EAGLE weight loading from GGUF  Deps: T36.1.1
- [x] T37.1.5 Fused NSA CUDA kernel (ztensor)  Deps: T37.1.4
- [x] T38.1.2 Split MoE weights GPU/CPU  Deps: T38.1.1
- [x] T39.2.1 TernaryStorage in GGUF loader  Deps: T39.1.1
- [x] T34.4.1 SparseRoutedAttention layer  Deps: T34.1.1, T34.1.3, T34.2.1, T34.3.1
- [x] T34.5.1 TieredKVStore  Deps: T34.2.1
- [x] T40.1.2 Write converted weights as GGUF  Deps: T40.1.1
- [x] T41.1.4 GPU dequant kernels for IQ types (ztensor)  Deps: T41.1.1-T41.1.3
- [x] T41.1.5 Wire I-Quant types into GGUF loader  Deps: T41.1.1-T41.1.3
- [x] T42.1.2 Cache-aware request scheduling  Deps: T42.1.1
- [x] T43.1.2 Wire flash decode into attention  Deps: T43.1.1
- [x] T44.1.2 LoRA weight merging in forward pass  Deps: T44.1.1
- [x] T44.1.3 Adapter cache with LRU eviction  Deps: T44.1.1

#### Wave 4: Wiring + Options (10 agents)

- [ ] T35.2.2 --quarot CLI flag  Deps: T35.2.1
- [ ] T35.3.3 Quantized KV cache options  Deps: T35.3.1, T35.3.2
- [ ] T36.2.2 WithEAGLE generator option  Deps: T36.2.1
- [ ] T34.4.2 Contrastive routing loss  Deps: T34.4.1
- [ ] T34.4.3 Register SparseRoutedAttention  Deps: T34.4.1
- [ ] T34.5.2 Async CPU-to-GPU fetch  Deps: T34.5.1
- [ ] T38.2.1 Async CPU expert dispatch  Deps: T38.1.2
- [ ] T39.2.2 Wire ternary MatMul into graph  Deps: T39.2.1, T39.1.2
- [ ] T37.2.1 Register NSA in layer registry  Deps: T37.1.4
- [ ] T38.2.2 Prefetch predicted experts  Deps: T38.2.1
- [ ] T40.1.3 TransMLA CLI command  Deps: T40.1.2
- [ ] T40.2.1 Detect TransMLA tensors and wire MLA  Deps: T40.1.2
- [ ] T44.1.4 Adapter selection in serve API  Deps: T44.1.2, T44.1.3

#### Wave 5: Independent Existing Work (10 agents)

These run in parallel with any wave -- no E34-E39 dependencies.

- [ ] BMK-T1 Download missing GGUFs  verifies: [UC-001]
- [ ] BMK-T3 Update website with benchmark table  Deps: BMK-T2
- [ ] BMK-T4 Investigate Phi3/Llama3.1 load failures  verifies: [UC-001]
- [ ] MHH-T1 Mistral quality comparison  verifies: [UC-001]
- [ ] MHH-T3 Sliding window correctness  verifies: [UC-001]
- [ ] KQ-T1 Profile Q4_K vs Q4_0 GEMV (ztensor)  verifies: [infrastructure]
- [ ] GTS-T1 Generate Python golden files  verifies: [UC-005]
- [ ] MHH-T2 Profile Mistral performance gap  Deps: MHH-T1
- [ ] KQ-T2 Optimize Q4_K GEMV kernel (ztensor)  Deps: KQ-T1
- [ ] GTS-T2 Run Zerfoo against golden files  Deps: GTS-T1

#### Wave 6: Tests + Benchmarks (10 agents)

- [ ] T35.1.3 Tests for Hadamard (ztensor)  Deps: T35.1.1, T35.1.2
- [ ] T35.2.3 Tests for QuaRot weight fusion  Deps: T35.2.1, T35.2.2
- [ ] T35.3.5 Tests and benchmarks for quantized KV  Deps: T35.3.1-T35.3.4
- [ ] T36.1.3 Tests for EAGLE head  Deps: T36.1.1, T36.1.2
- [ ] T36.2.4 Benchmark EAGLE vs vanilla  Deps: T36.2.1
- [ ] T36.2.5 Tests for EAGLE decode loop  Deps: T36.2.1-T36.2.3
- [ ] T37.2.2 Unit tests for NSA  Deps: T37.1.4
- [ ] T38.1.3 Tests for expert placement  Deps: T38.1.1, T38.1.2
- [ ] T38.2.3 Tests and benchmarks for hybrid MoE  Deps: T38.2.1, T38.2.2
- [ ] T39.1.4 Tests for ternary storage/GEMV  Deps: T39.1.1-T39.1.3
- [ ] T40.1.4 Tests for SVD conversion  Deps: T40.1.1-T40.1.3
- [ ] T40.2.2 Validate MLA KV cache stores latent  Deps: T40.2.1
- [ ] T40.2.4 Perplexity validation script  Deps: T40.2.1
- [ ] T41.1.6 Tests for I-Quant dequantization  Deps: T41.1.1-T41.1.5
- [ ] T42.1.3 Tests for radix cache and scheduling  Deps: T42.1.1, T42.1.2
- [ ] T43.1.3 Tests and benchmarks flash decode  Deps: T43.1.1, T43.1.2
- [ ] T44.1.5 Tests for Multi-LoRA serving  Deps: T44.1.1-T44.1.4

#### Wave 7: Final Lint + Integration (10 agents)

- [ ] T34.1.4 Run vet/tests ztensor primitives  Deps: T34.1.1-T34.1.3
- [ ] T34.2.3 Compressed cache tests  Deps: T34.2.1, T34.2.2
- [ ] T34.3.3 Document-wise RoPE tests  Deps: T34.3.1, T34.3.2
- [ ] T34.4.4 Sparse routed attention tests  Deps: T34.4.1-T34.4.3
- [ ] T34.5.3 Tiered storage tests  Deps: T34.5.1, T34.5.2
- [ ] T39.2.3 Tests for BitNet loading  Deps: T39.2.1, T39.2.2
- [ ] T35.3.6 Run go vet E35  Deps: T35.2.3, T35.3.5
- [ ] T36.2.6 Run go vet E36  Deps: T36.2.5
- [ ] T37.2.3 Run go vet E37  Deps: T37.2.2
- [ ] T38.2.4 Run go vet E38  Deps: T38.2.3
- [ ] T40.2.3 TransMLA end-to-end integration test  Deps: T40.2.1, T40.2.2
- [ ] T41.1.7 Run go vet E41  Deps: T41.1.6
- [ ] T42.1.4 Run go vet E42  Deps: T42.1.3
- [ ] T43.1.4 Run go vet E43  Deps: T43.1.3
- [ ] T44.1.6 Run go vet E44  Deps: T44.1.5

#### Wave 8: Final Integration (5 agents)

- [ ] T34.6.1 Run go vet and linters all E34  Deps: all E34 tasks
- [ ] T34.6.2 Integration test on DGX Spark  Deps: T34.6.1
- [ ] T39.2.4 Run go vet E39  Deps: T39.2.3
- [ ] KQ-T3 Benchmark and re-enable native Q4_K  Deps: KQ-T2
- [ ] GTS-T3 Benchmark vs Python granite-tsfm  Deps: GTS-T2
- [ ] T40.2.5 Run go vet E40  Deps: T40.1.4, T40.2.3, T40.2.4

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M0 | Internal Consumer Bridge | WE1, WE13 | DONE | 2026-03-19 |
| M0.5 | Advanced Tabular | WE2-WE4 | DONE | 2026-03-19 |
| M1 | Inference Excellence | E2, WE13 | 300+ tok/s; 12+ archs; 5K stars | 2026-12-31 |
| M1.5 | Scalable Memory | E34 | Compressed cache serving 1M+ tokens; document-wise RoPE validated; sparse routing layer merged | 2026-Q3 |
| M1.6 | Research Optimizations | E35-E44 | QuaRot fused; EAGLE >= 2x; NSA merged; ternary GEMV; hybrid MoE; TransMLA >= 80% KV reduction; I-Quants loading; radix cache; flash decode >= 1.5x; Multi-LoRA serving | 2026-Q4 |
| M2 | v1.0 and Ecosystem | E8-E11 | ROCm parity; 25K stars | 2027-12-31 |
| M3 | Enterprise Foundation | E12-E16, WE5 | $500K ARR; SOC 2 Type I | 2028-12-31 |
| M4 | Platform GA | E17-E19, WE6-WE7 | $2M ARR; SOC 2 Type II | 2029-12-31 |
| M5 | Training Platform | E20-E22, WE8-WE9 | $10M ARR; Metal + SYCL | 2030-12-31 |
| M6 | Industry Standard | E24-E27, WE10-WE11 | $50M ARR; ZerfooConf | 2032-12-31 |
| M7 | Platform Maturity | E28-E30, WE12 | $75M ARR; federated; on-device | 2034-12-31 |
| M8 | Market Leadership | E31-E33 | $150M+ ARR; IPO filed | 2036-12-31 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R0 | Tabular package does not improve signal quality | Critical | Medium | Architecture diversity; AutoML; walk-forward validation |
| R1 | Go ML TAM ceiling | Critical | High | OpenAI API, edge runtime, language FFI |
| R2 | Apache 2.0 fork by cloud provider | Existential | Medium-High | Innovation velocity; consider AGPL for v2 (ADR-057) |
| R3 | Latent bugs in AI-generated code | High | High | Security audit; DGX validation; fuzz testing; bug bounty |
| R4 | Maintainer burnout / bus factor of 1 | Critical | High | 5 co-maintainers by Year 2; governance by Year 4 |
| R5 | No enterprise budget owner | High | Medium-High | Position as inference infrastructure; marketplace credits |
| R6 | ROCm never reaches CUDA parity | Medium | High | 80% parity target; gate by user demand |
| R7 | Enterprise sales cycle too long | High | Medium | Marketplace; support contracts; PLG motion |
| R9 | Rust ML captures systems ML first | High | Medium | Ship v1.0 first; edge differentiator |
| R15 | Agentic coder quality drift | High | High | Human review gates; security audit; strict CI |
| R25 | ~~Mistral forward pass bug~~ | ~~High~~ | ~~High~~ | RESOLVED |
| R26 | MSA compressed cache degrades output quality | Medium | Medium | Validate with perplexity benchmarks; configurable chunk size |
| R27 | MSA routing requires model fine-tuning | High | High | KV compression and doc-wise RoPE work without training |
| R28 | QuaRot Hadamard rotation loses precision at high dimension | Medium | Low | Validate perplexity before/after rotation on each model family; skip rotation if degradation > 0.1 ppl |
| R29 | No competitive BitNet models in GGUF format | Medium | Medium | Ternary storage is small engineering cost; ready when models ship. Monitor llama.cpp BitNet support. |
| R30 | NSA fused kernel complexity causes maintenance burden | Medium | Medium | Decomposed fallback path always available; fused kernel is optional optimization |
| R31 | Hybrid MoE async scheduling adds latency on small models | Low | High | Only activate hybrid placement for models with > 8 experts; small models stay GPU-only |
| R32 | Six concurrent research epics spread too thin | High | Medium | Each epic independently shippable; prioritize E35+E36+E40 first (highest impact, lowest risk) |
| R33 | SVD truncation degrades quality for some model families | Medium | Medium | Validate perplexity per model family before publishing converted GGUFs; allow configurable rank; provide validation CLI |
| R34 | I-Quant grid tables diverge from llama.cpp upstream | Low | Medium | Pin to llama.cpp GGUF spec version; re-validate on major releases |
| R35 | Multi-LoRA adapter GPU memory fragmentation | Medium | Medium | LRU eviction + adapter memory budget cap; monitor with arena allocator stats |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Benchmark tasks: results appended to docs/devlog.md.
6. Each task committed as its own commit. One logical change per commit.

### Quality Gates

- Every implementation task must have a paired test.
- Run `go vet ./...` after every code change before committing.
- Standard library only: no testify, no cobra, no viper.
- GPU-only tests: tag with `//go:build cuda` and run only on DGX.
- Human review gate required at each milestone (M0-M8).
- Security review (/review) before each enterprise-facing release.
- Run `golangci-lint` on all changed packages before committing.
- Rebase and merge. Not squash, not merge commits.

### Code Style

- Engine[T] is law: all tensor ops through compute.Engine[T].
- Generics throughout: [T tensor.Numeric] constraints.
- Fuse, do not fragment: prefer fused ops over primitive sequences.
- No CGo in core packages; GPU via purego.
- Docstrings only on exported types and functions.

---

## References

### Research Papers (E35-E39)
- QuaRot: Outlier-Free 4-Bit Inference (arXiv:2404.00456, NeurIPS 2024)
- KVQuant: KV Cache Quantization to 10M Context (UC Berkeley, NeurIPS 2024)
- EAGLE-3: Scaling up Inference Acceleration (arXiv:2503.01840, March 2025)
- NSA: Native Sparse Attention (arXiv:2502.11089, DeepSeek, ACL 2025 Best Paper)
- KTransformers: CPU/GPU Hybrid MoE (Tsinghua MADSys, SOSP 2025)
- BitNet b1.58: 1-bit LLMs (arXiv:2402.17764, Microsoft, February 2024)
- TransMLA: Multi-Head Latent Attention Is All You Need (arXiv:2502.07864, February 2025)
- MHA2MLA: Retrofitting MLA (arXiv:2502.14837, February 2025)
- DeepSeek-V2 MLA (arXiv:2405.04434, May 2024)
- SGLang / RadixAttention (arXiv:2312.07104, NeurIPS 2024)
- S-LoRA: Serving Thousands of LoRA Adapters (arXiv:2311.03285)
- LoRAFusion: Fused LoRA Serving (arXiv:2510.00206)

### MSA Paper
- "Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to
  100M Tokens" (Chen et al., 2025, EverMind-AI/Shanda Group)
- [GitHub: EverMind-AI/MSA](https://github.com/EverMind-AI/MSA)

### Granite Time Series
- [TTM Paper (NeurIPS 2024)](https://arxiv.org/pdf/2401.03955)
- [TSPulse Paper](https://arxiv.org/pdf/2505.13033)

### Granite Guardian
- [Granite Guardian Paper (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)

### K-Quant
- Q4_K GEMV is 45% slower than Q4_0 on GB10. Infrastructure merged.

### ADRs Referenced
- ADR-037: GGUF sole model format
- ADR-057: Apache 2.0 licensing
- ADR-058: v1.0 API freeze
- ADR-059: Edge runtime architecture
- ADR-060: Cloud model repository
- ADR-061: GGUF writer in ztensor
- ADR-062: Tabular model package
- ADR-064: Hugo docs site
- ADR-065: Security middleware integration
- ADR-066: CPU training backprop
- ADR-067: MSA-inspired sparse attention for scalable memory
- ADR-068: Research-driven inference optimization priorities
- ADR-069: TransMLA -- Retrofit MLA onto MHA/GQA models

---

## Progress Log

### 2026-03-27: Added E41-E44 (I-Quants, RadixAttention, Flash Decode, Multi-LoRA)

Added four remaining high-value research items as epics E41-E44 (24 tasks total):
- E41 (P12): GGUF I-Quant dequantization -- IQ2_XXS, IQ3_S, IQ4_NL (7 tasks).
  Enables loading community-quantized models from llama.cpp ecosystem.
- E42 (P13): RadixAttention KV cache upgrade -- hash-based radix tree, LRU
  eviction, cache-aware scheduling (4 tasks). Up to 6.4x serving throughput.
- E43 (P14): Flash Decoding -- split-KV decode kernel for memory-bound decode
  phase (4 tasks). >= 1.5x decode speedup at long KV lengths.
- E44 (P15): Multi-LoRA serving -- per-request adapter selection, LRU adapter
  cache, serve API integration (6 tasks). Production multi-tenant requirement.
Added risks R34-R35. Added use cases UC-011, UC-012. All four epics are fully
independent -- tasks distributed across Waves 2-7.

### 2026-03-27: Added TransMLA/MHA2MLA epic (E40)

Added E40 (P11) with 9 tasks for retrofitting Multi-Head Latent Attention onto
existing MHA/GQA models via SVD-based weight decomposition. Reuses existing MLA
layer from DeepSeek V3. Two sub-epics: E40.1 (SVD conversion tool, 4 tasks) and
E40.2 (automatic MLA inference path, 5 tasks). Created ADR-069. Added risk R33
(SVD quality). Added use case UC-010. Updated milestone M1.6 to include TransMLA.
Tasks distributed across Waves 2-8 -- no dependencies on other research epics.

### 2026-03-27: Added research-driven inference epics (E35-E39)

Research survey of 30+ ML papers (March 2024 -- March 2027) identified five
high-impact techniques. Added 44 new tasks across 5 epics:
- E35 (P6): QuaRot + KVQuant uniform 4-bit quantization (12 tasks)
- E36 (P7): EAGLE-3 self-speculative decoding (9 tasks)
- E37 (P8): NSA native sparse attention (8 tasks)
- E38 (P9): CPU/GPU hybrid MoE inference (7 tasks)
- E39 (P10): BitNet b1.58 ternary inference (8 tasks)
Created ADR-068 for the decision. Added risks R28-R32. Added milestone M1.6
(Research Optimizations, 2026-Q4). Reorganized waves 1-8 to accommodate all
research tracks in parallel with existing work. Added use cases UC-006
through UC-009.

### 2026-03-27: Added MSA-Inspired Scalable Memory epic (E34)

Added E34 with 18 tasks across 6 sub-epics. Created ADR-067.
Added risks R26-R27. Added milestone M1.5.

### 2026-03-26: Consolidated all plans

Merged all satellite plans. 330+ tasks complete. All models coherent.

---

## Hand-off Notes

- All code is in Go 1.25 with generics. No CGo. GPU via purego/dlopen.
- DGX Spark GPU at `ssh ndungu@192.168.86.250` for CUDA testing.
- ztensor and zerfoo are separate repos with separate go.mod files.
  Primitive tasks (T34.1.*, T35.1.*, T35.3.4, T37.1.5, T39.1.*) go in ztensor.
  All other tasks go in zerfoo.
- E35 (QuaRot) and E36 (EAGLE) are highest priority -- least risk, most impact.
- E37 (NSA) subsumes E34.4 (SparseRoutedAttention) with better GPU utilization.
  Both are planned since MSA routing is different from NSA's approach.
- E38 (hybrid MoE) is architecturally invasive; start with placement policy
  before touching the inference loop.
- E39 (BitNet) depends on ternary GGUF models being available.
  Monitor llama.cpp for ternary quantization type support.
- E40 (TransMLA) reuses existing MLA layer -- no new attention code needed.
  SVD conversion is offline. Converted GGUFs are standard GGUF files with
  extra "transmla." tensors. Validate perplexity per model family.
- E41 (I-Quants) tasks go in ztensor (dequant kernels) and zerfoo (GGUF loader).
  Reference llama.cpp quantize.cpp for grid table constants.
- E42 (RadixAttention) replaces existing PrefixCache. Backward compatible API.
- E43 (Flash Decode) goes in ztensor (CUDA kernel) and zerfoo (attention wiring).
- E44 (Multi-LoRA) is entirely in zerfoo. Adapter format uses standard GGUF.
- GGUF is the sole model format (ADR-037).
- Gemma3-1B Q4_K_M is cached on DGX Spark for integration tests.

---

## Appendix

### Use Case IDs Referenced

| ID | Name | Description |
|----|------|-------------|
| UC-001 | Run inference on GGUF model | Load model, generate text, return response |
| UC-002 | Serve model via OpenAI API | HTTP server with /v1/chat/completions |
| UC-003 | Large-corpus memory QA | Query against 1M-100M token document bank |
| UC-004 | Train MSA routing | Fine-tune with router projectors and auxiliary loss |
| UC-005 | Time-series inference | Run TTM/FlowState/TSPulse on time-series data |
| UC-006 | Load QuaRot-rotated model | GGUF load with Hadamard rotation fused into weights |
| UC-007 | EAGLE self-speculative inference | Generate text with lightweight prediction head, no draft model |
| UC-008 | Hybrid MoE inference | Run large MoE model with experts split across GPU and CPU |
| UC-009 | BitNet ternary inference | Run ternary-weight model with add/sub instead of GEMM |
| UC-010 | TransMLA-converted inference | Load MHA model converted to MLA via SVD, serve with compressed KV cache |
| UC-011 | Prefix-cached serving | Serve requests that share system prompts with automatic KV cache reuse |
| UC-012 | Multi-LoRA serving | Serve multiple fine-tuned LoRA adapters from a single base model per-request |
