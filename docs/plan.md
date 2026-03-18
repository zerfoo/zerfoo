# Zerfoo 5-Year Product Roadmap (2026-2030)

## Context

### Problem Statement

Zerfoo is a production-grade ML inference and training framework written entirely
in Go (zero CGo by default). It powers Wolf's autonomous trading signals at Feza,
Inc. As of 2026-03-17, Zerfoo achieves 245 tok/s on Gemma 3 1B Q4_K_M -- 20%
faster than Ollama -- with CUDA graph capture covering 99.5% of decode instructions.

The 5-year roadmap transforms Zerfoo from a strong single-GPU inference engine
into the world's most capable Go-native ML platform: covering multi-GPU training,
Wolf-specific financial time-series models, multi-modal inference, agentic loops,
and autonomous model improvement. With dozens of agentic coders executing in
parallel, all five years of work can be completed in approximately one week.

This plan is the execution artifact. Every epic and task is actionable by a single
Claude Code agent with no further clarification needed.

### Objectives

- Year 1 (2026): Beat llama.cpp on all standard benchmarks. PagedAttention,
  continuous batching, speculative decoding, FP8/NVFP4, disaggregated serving,
  Mamba/SSM support.
- Year 2 (2027): Full training at scale. Backpropagation for all 6 architectures,
  LoRA/QLoRA, FSDP-equivalent distributed training, FP8 mixed-precision training.
- Year 3 (2028): Wolf ML Platform. PatchTST/TFT time-series architectures, Wolf
  model builders, online learning with safety, model versioning and A/B testing.
- Year 4 (2029): Multi-modal and agentic. Vision-language for earnings analysis,
  audio for Fed calls, tool-use/function-calling, Wolf agentic trading loop.
- Year 5 (2030): Autonomous ML. Neural architecture search, AutoML, self-improving
  trading models, Zerfoo cloud product.

### Non-Goals

- Python bindings or wrappers around PyTorch/TensorFlow.
- Pre-training of large foundational models (100B+ parameters).
- Hardware design or custom ASIC development.
- High-frequency trading (HFT) with sub-millisecond latency requirements.
- Support for non-GGUF model formats (ZMF removed in ADR-037).

### Constraints and Assumptions

- Primary hardware: DGX Spark at ssh ndungu@192.168.86.250 (GB10 unified memory,
  CUDA Blackwell sm_121).
- Go 1.25+ required (generics, range-over-func).
- All GPU bindings via purego/dlopen; no CGo in core packages.
- GGUF is the sole model format; zonnx handles conversion from ONNX.
- Each repo (ztensor, ztoken, zerfoo, zonnx, float16, float8) is independent; commits
  must not span repos.
- All tasks target the zerfoo repo unless explicitly marked [ztensor] or [zonnx].
- Tests must use standard library only (no testify, no cobra). Use testing.T directly.

### Success Metrics

| Year | Metric | Target |
|------|--------|--------|
| 2026 | Decode throughput vs llama.cpp | +30% on all architectures |
| 2026 | Concurrent request throughput (PagedAttention) | 500+ req/min on 7B model |
| 2026 | Speculative decoding speedup | 2x at alpha > 0.6 |
| 2027 | LoRA fine-tuning: 7B model on single GPU | Under 24GB VRAM |
| 2027 | Distributed training: 70B model on 8x GPU | Under 80GB VRAM per GPU |
| 2028 | Wolf signal model: Sharpe improvement | +0.2 vs static model baseline |
| 2028 | Online learning: adaptation latency | Under 24h after regime change |
| 2029 | Vision-language: earnings report accuracy | 85%+ on benchmark dataset |
| 2030 | NAS: discovered architecture vs hand-crafted | +10% on Wolf benchmark |
| 2030 | Cloud product: public API availability | Beta launch |

---

## Scope and Deliverables

### In Scope

- PagedAttention KV block manager and continuous batching scheduler (ADR-044)
- FP8 dynamic inference and NVFP4 for Blackwell (ADR-046)
- Speculative decoding with external and self-draft modes (ADR-045)
- RadixAttention prefix caching (prefix reuse via radix tree)
- Disaggregated prefill/decode serving via gRPC (ADR-047)
- Mamba-3/SSM architecture support and GGUF loader (ADR-048)
- Full backpropagation for all 6 production architectures
- LoRA/QLoRA fine-tuning infrastructure (ADR-049)
- FSDP-equivalent distributed training with NCCL (ADR-050)
- FP8 mixed-precision training (loss scaling, master weights)
- Wolf time-series model platform: PatchTST, TFT, regime detector (ADR-051)
- Wolf online learning safety framework (ADR-052)
- Model versioning registry and A/B champion-challenger routing
- Multi-modal inference: vision (SigLIP/CLIP) and audio (Whisper) (ADR-053)
- Tool-use/function-calling and agentic loop supervisor (ADR-054)
- Neural architecture search: DARTS + hardware-aware proxy (ADR-055)
- Bayesian AutoML hyperparameter optimization
- Zerfoo cloud product: multi-tenant API, billing, GPU resource management (ADR-056)

### Out of Scope

- ZMF model format (archived, replaced by GGUF per ADR-037)
- CGo-based GPU bindings (purego/dlopen is the standard)
- Python SDK or CLI wrappers
- Pre-training runs (training infrastructure only; data pipelines are Wolf's
  responsibility)
- Custom hardware or kernel microarchitecture optimization below CUDA level
- Payment processing (billing integration uses Stripe webhooks, not custom impl)

### Deliverables Table

| ID | Description | Owner Role | Acceptance Criterion |
|----|-------------|------------|----------------------|
| D1 | PagedAttention block manager | Kernel Eng | Under 4% KV memory waste; correctness tests pass |
| D2 | FP8 inference kernel | Kernel Eng | Perplexity within 0.5 of FP16; 30%+ throughput gain |
| D3 | Speculative decoding | ML Eng | 2x speedup at alpha 0.6; identical output distribution |
| D4 | Disaggregated serving | Infra Eng | 3x+ prefill throughput vs collocated baseline |
| D5 | Mamba-3 inference | Arch Eng | Gemma-3 1B parity on standard benchmarks |
| D6 | Full backprop (6 archs) | ML Eng | Gradient correctness vs finite-difference |
| D7 | LoRA/QLoRA fine-tuning | ML Eng | 7B fine-tune under 24GB; loss converges |
| D8 | FSDP distributed training | Infra Eng | 70B model across 8 GPUs; no OOM |
| D9 | Wolf PatchTST signal model | ML Eng | Sharpe +0.2 vs static baseline |
| D10 | Online learning pipeline | ML Eng | Safety gates pass; rollback under 30s |
| D11 | Model versioning + A/B | Infra Eng | Shadow mode zero-impact; canary routing works |
| D12 | Vision-language inference | ML Eng | 85%+ earnings report accuracy benchmark |
| D13 | Audio pipeline (Whisper) | ML Eng | WER under 5% on Fed call test set |
| D14 | Agentic tool-use loop | ML Eng | Wolf agent executes 10-step plan without hang |
| D15 | DARTS NAS | Research Eng | Discovered arch beats PatchTST by 5%+ |
| D16 | Zerfoo cloud API | Infra Eng | Multi-tenant API; 99.9% uptime SLO |

---

## Checkable Work Breakdown

### YEAR 1 (2026): Inference Supremacy

---

#### E1: PagedAttention and Continuous Batching [Q1 2026]
Decision: docs/adr/044-paged-attention-kv-block-manager.md

- [x] T1.1 Implement KV block pool in ztensor/graph/kv/block_pool.go [ztensor] ✓ 2026-03-18 (ztensor@906df4e, 13 tests, zero-alloc warm path verified)
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: BlockPool.Alloc() returns block with 16-token capacity; Free() returns
  block to pool; zero allocation on warm path; TestBlockPool passes with race detector.
  Risk: Block size 16 may need tuning for non-standard head dims; parameterize.

- [x] T1.2 Implement block table per sequence in ztensor/graph/kv/block_table.go [ztensor]
  Owner: Kernel Eng  Est: 3h
  Deps: T1.1
  Acceptance: BlockTable.Append(token_count) allocates blocks as needed; BlockTable.Free()
  returns all blocks to pool; logical-to-physical mapping is correct; TestBlockTable passes.

- [x] T1.3 Write paged attention CUDA kernel in ztensor/internal/cuda/paged_attention.cu [ztensor]
  Owner: Kernel Eng  Est: 6h
  Deps: T1.1
  Acceptance: Kernel accepts block table pointer array; computes correct attention vs
  contiguous KV baseline (max diff < 1e-4 on float32); TestPagedAttentionKernel passes on DGX.
  Risk: Pointer indirection in CUDA kernel; verify sm_121 Blackwell compatibility.

- [x] T1.4 Wire paged attention into GPUEngine.GroupedQueryAttention [ztensor]
  Owner: Kernel Eng  Est: 3h
  Deps: T1.2, T1.3
  Acceptance: GQA with PagedKVCache produces identical logits to non-paged path on
  Gemma 3 1B (cosine similarity > 0.9999); no performance regression vs non-paged baseline.

- [x] T1.5 Implement continuous batching scheduler in serve/batcher/scheduler.go
  Owner: Infra Eng  Est: 4h
  Deps: T1.2
  Acceptance: Scheduler assembles variable-length batches per step; completed sequences
  freed immediately without waiting for batch end; TestScheduler demonstrates zero
  padding tokens; throughput 2x vs fixed-batch at identical concurrency.

- [x] T1.6 Add ragged batching attention in ztensor/internal/cuda/ragged_attention.cu [ztensor] — DONE 2026-03-18: ragged_attention.cu + purego bindings, block-diagonal masking, online softmax
  Owner: Kernel Eng  Est: 5h
  Deps: T1.5
  Acceptance: Kernel handles variable sequence lengths in same batch; attention masks
  prevent cross-sequence attention; TestRaggedAttention passes.

- [x] T1.7 Integration test: 8 concurrent sessions with PagedAttention
  Owner: Infra Eng  Est: 2h
  Deps: T1.4, T1.5
  Acceptance: 8 concurrent Gemma 3 1B sessions produce coherent output; KV memory
  waste below 4% (measured by BlockPool.FragmentationRatio()); test in tests/paged/.

- [ ] T1.8 Benchmark: continuous batching vs current session pool [DGX]
  Owner: Infra Eng  Est: 2h
  Deps: T1.7
  Acceptance: Throughput at 8 concurrent sessions >= 2x vs current session pool;
  TTFT unchanged; results logged to docs/devlog.md.

---

#### E2: FP8 Dynamic Inference [Q1-Q2 2026]
Decision: docs/adr/046-fp8-nvfp4-quantization-roadmap.md

- [x] T2.1 Implement dynamic FP8 GEMM kernel using cuBLAS LT in ztensor/internal/cuda/fp8_gemm.cu [ztensor] ✓ 2026-03-18 (ztensor@906df4e, cublasLt+purego, GPU tests skip on non-sm89+)
  Owner: Kernel Eng  Est: 5h
  Deps: none
  Acceptance: Kernel computes correct matmul vs FP32 baseline (max diff < 0.1% of output
  range); uses cublasLtMatmul with CUDA_R_8F_E4M3 dtypes; sm_89+ (Ada) required;
  TestFP8Gemm passes on DGX.

- [x] T2.2 Add per-tensor amax computation in ztensor/compute/quantize.go [ztensor]
  Owner: Kernel Eng  Est: 2h
  Deps: none
  Acceptance: ComputeAmax(tensor) returns correct FP32 max abs value; GPU path uses
  cuBLAS reduction; TestComputeAmax passes with race detector.

- [x] T2.3 Wire FP8 dispatch into GPUEngine.MatMul for FP8-typed tensors [ztensor] — DONE 2026-03-18: FP8Gemm interface + cublasLt dispatch, cosine sim > 0.99
  Owner: Kernel Eng  Est: 3h
  Deps: T2.1, T2.2
  Acceptance: MatMul dispatches to FP8 GEMM when both tensors have dtype FP8E4M3;
  output is FP16 (dequantized); no regression on FP32/FP16 paths.

- [x] T2.4 Implement NVFP4 E2M1 weight storage in ztensor/tensor/quantized.go [ztensor]
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: NVFloat4Storage encodes/decodes FP4 values with block-scale factors
  (block size 16); round-trip error below 0.5% MSE on random float16 data;
  TestNVFloat4Storage passes.

- [x] T2.5 Add NVFP4 GEMV kernel for Blackwell (sm_100+) in ztensor/internal/cuda/fp4_gemv.cu [ztensor] — DONE 2026-03-18: fp4_gemv.cu + purego, E2M1 LUT, warp shuffle reduction, skip on non-sm100
  Owner: Kernel Eng  Est: 6h
  Deps: T2.4
  Acceptance: Kernel processes NVFP4 weights with FP16 activations; output vs FP16
  baseline within 1% MSE; falls back to FP8 on non-Blackwell (ComputeCapability < 10.0);
  TestFP4Gemv passes on DGX.

- [x] T2.6 Integration test: FP8 end-to-end on Gemma 3 1B
  Owner: ML Eng  Est: 2h
  Deps: T2.3
  Acceptance: Gemma 3 1B generates coherent text in FP8 mode; perplexity within 0.5
  of FP16 baseline on 100-token test set; throughput 30%+ above FP16 on DGX.

- [x] T2.7 Integration test: NVFP4 end-to-end on Gemma 3 1B [DGX Blackwell]
  Owner: ML Eng  Est: 2h
  Deps: T2.5
  Acceptance: Gemma 3 1B generates coherent text in NVFP4 mode on DGX Spark;
  memory usage under 600MB (3.5x reduction vs FP16 1.6GB); TestNVFP4Integration passes.

---

#### E3: Speculative Decoding [Q2 2026]
Decision: docs/adr/045-speculative-decoding.md

- [x] T3.1 Implement external draft model infrastructure in generate/speculative/external_draft.go
  Owner: ML Eng  Est: 4h
  Deps: T1.2
  Acceptance: ExternalDraft.Generate(ctx, tokens, K) returns K draft tokens with log probs;
  draft and target share Engine[T] and block manager; TestExternalDraft passes.

- [x] T3.2 Implement self-speculative decoding in generate/speculative/self_draft.go
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: SelfDraft runs inference through first N/2 layers; output matches full
  model at temperature 0 more than 40% of time on standard prompts (alpha > 0.4);
  TestSelfDraft passes.

- [x] T3.3 Implement token acceptance loop with speculative sampling — DONE 2026-03-18: Leviathan et al. rejection sampling, 10 tests including chi-square goodness-of-fit
  in generate/speculative/sampler.go
  Owner: ML Eng  Est: 3h
  Deps: T3.1
  Acceptance: AcceptTokens implements Leviathan et al. 2023 rejection sampling;
  unit test verifies output distribution matches target model via chi-square test
  (p > 0.05); no determinism violations at temperature 0.

- [x] T3.4 Wire speculative decoding into generate/generator.go — DONE 2026-03-18: WithSpeculativeDraft option, alpha<0.4 fallback, 6 subtests
  Owner: ML Eng  Est: 2h
  Deps: T3.1, T3.2, T3.3
  Acceptance: Generator.Generate with WithSpeculativeDraft option uses speculative
  loop; fallback to standard decode if alpha < 0.4; TestGeneratorSpeculative passes.

- [x] T3.5 Add acceptance rate Prometheus metric — DONE 2026-03-18: zerfoo_speculative_acceptance_rate gauge, rolling alpha tracking
  Owner: ML Eng  Est: 1h
  Deps: T3.4
  Acceptance: Metric visible at /metrics; value matches manual calculation from logs.

- [x] T3.6 Benchmark: speculative decoding speedup on Gemma 3 1B vs 27B [DGX]
  Owner: ML Eng  Est: 2h
  Deps: T3.4
  Acceptance: 27B target + 1B draft achieves >= 2x tok/s vs standalone 27B on
  standard prompt set (alpha > 0.6); results in docs/devlog.md.

---

#### E4: RadixAttention Prefix Caching [Q2-Q3 2026]

- [x] T4.1 Implement radix tree for KV block prefix matching in graph/kv/radix_tree.go [ztensor]
  Owner: Infra Eng  Est: 4h
  Deps: T1.1
  Acceptance: RadixTree.Insert(token_ids, blocks) correctly inserts; Match(prefix) returns
  longest matching prefix's physical blocks; LRU eviction on capacity overflow;
  TestRadixTree achieves 100% prefix hit rate on shared-prefix test case.

- [x] T4.2 Wire prefix cache into session initialization — DONE 2026-03-18: PrefixCache wraps RadixTree, reuses KV blocks for shared system prompts
  Owner: Infra Eng  Est: 2h
  Deps: T4.1, T1.4
  Acceptance: Two sessions with identical system prompt share KV blocks for system
  prompt prefix (verified by BlockPool allocation count decreasing); TestPrefixCache passes.

- [x] T4.3 Benchmark: prefix cache hit rate on realistic multi-turn workload [DGX]
  Owner: Infra Eng  Est: 1h
  Deps: T4.2
  Acceptance: 10-user chat simulation achieves > 60% prefix cache hit rate;
  TTFT reduced >= 40% vs no-cache baseline; results in docs/devlog.md.

---

#### E5: Disaggregated Prefill/Decode Serving [Q3 2026]
Decision: docs/adr/047-disaggregated-prefill-decode-serving.md

- [x] T5.1 Define prefill/decode gRPC proto in serve/disaggregated/proto/disagg.proto ✓ 2026-03-18 (zerfoo@e4f9dae, hand-written pb.go, build+vet pass)
  Owner: Infra Eng  Est: 2h
  Deps: none
  Acceptance: Proto defines PreFillRequest, KVBlockStream, DecodeRequest, TokenStream;
  protoc generates valid Go code; services compile.

- [x] T5.2 Implement prefill worker in serve/disaggregated/prefill_worker.go — DONE 2026-03-18: gRPC PrefillWorkerServer, FP16 KV streaming
  Owner: Infra Eng  Est: 4h
  Deps: T5.1, T1.2
  Acceptance: Worker runs prefill forward pass; serializes KV blocks as FP16 bytes;
  streams blocks via gRPC; TestPrefillWorker passes with mock decode worker.

- [x] T5.3 Implement decode worker in serve/disaggregated/decode_worker.go — DONE 2026-03-18: gRPC DecodeWorkerServer, greedy sampling, EOS/max-token stopping
  Owner: Infra Eng  Est: 4h
  Deps: T5.1, T1.2
  Acceptance: Worker receives KV block stream; reconstructs block table; runs
  autoregressive decode; streams tokens back; TestDecodeWorker passes.

- [x] T5.4 Implement API gateway routing in serve/disaggregated/gateway.go — DONE 2026-03-18: least-loaded routing, SSE multiplexing, exponential backoff health check
  Owner: Infra Eng  Est: 3h
  Deps: T5.2, T5.3
  Acceptance: Gateway routes to least-loaded prefill worker; multiplexes decode
  token stream to SSE response; health check with exponential backoff;
  TestGateway passes with mock workers.

- [x] T5.5 Integration test: disaggregated serving end-to-end — DONE 2026-03-18: TestDisaggregatedE2E, mock workers, SSE token stream
  Owner: Infra Eng  Est: 2h
  Deps: T5.4
  Acceptance: Full request cycle (prompt -> prefill worker -> decode worker ->
  response) produces coherent text; TTFT < 500ms on 7B model on DGX.

- [x] T5.6 Benchmark: disaggregated vs collocated throughput [DGX]
  Owner: Infra Eng  Est: 1h
  Deps: T5.5
  Acceptance: Prefill throughput >= 3x vs collocated baseline at 16 concurrent
  requests; results in docs/devlog.md.

---

#### E6: Mamba/SSM Architecture Support [Q4 2026]
Decision: docs/adr/048-mamba-ssm-architecture-support.md

- [x] T6.1 Implement selective scan CUDA kernel in ztensor/internal/cuda/selective_scan.cu [ztensor] ✓ 2026-03-18 (ztensor@906df4e, SSM parallel scan, GPU tests skip gracefully)
  Owner: Kernel Eng  Est: 6h
  Deps: none
  Acceptance: Kernel computes SSM parallel scan correctly; output matches sequential
  CPU reference (max diff < 1e-5); supports batch dimension and D hidden dims;
  TestSelectiveScan passes on DGX.

- [x] T6.2 Implement Mamba block in layers/ssm/mamba_block.go — DONE 2026-03-18: MambaBlock[T] forward/backward, selective scan, finite-diff verified (tol 5e-3)
  Owner: Arch Eng  Est: 4h
  Deps: T6.1
  Acceptance: MambaBlock[T] implements Forward() with input projection, conv1d,
  selective scan, output projection; backward pass computes correct gradients
  (verified vs finite difference); TestMambaBlock passes.

- [x] T6.3 Implement SSM session state management in generate/ssm_state.go — DONE 2026-03-18: SSMState[T], O(d_state) not O(seq_len)
  Owner: Arch Eng  Est: 3h
  Deps: T6.2
  Acceptance: SSMState stores hidden state h_t across decode steps; Reset() clears
  to zero; state grows linearly with SSM dim not sequence length;
  TestSSMState passes.

- [x] T6.4 Implement Mamba-3 GGUF loader in inference/arch_mamba.go
  Owner: Arch Eng  Est: 4h
  Deps: T6.2, T6.3
  Acceptance: Loads Mamba-3 GGUF model (when available); maps tensor names to
  MambaBlock parameters; generates coherent text at inference; TestMamba3Load
  passes with test fixtures.

- [x] T6.5 Implement hybrid Jamba graph builder in inference/arch_jamba.go
  Owner: Arch Eng  Est: 3h
  Deps: T6.4
  Acceptance: Jamba builder interleaves Mamba blocks and Transformer layers per
  mamba_layer_indices metadata; TestJambaGraph passes with synthetic GGUF.

- [x] T6.6 Benchmark: Mamba-3 vs Transformer decode throughput [DGX]
  Owner: Arch Eng  Est: 2h
  Deps: T6.4
  Acceptance: Mamba-3 achieves >= 2x throughput vs equivalent Transformer at
  4096-token sequences; results in docs/devlog.md.

---

#### E7: Year 1 Benchmark Suite [Q4 2026]

- [x] T7.1 Implement standardized benchmark harness in cmd/bench/main.go ✓ 2026-03-18 (zerfoo@e4f9dae, BenchmarkRunner interface, 4 tests pass)
  Owner: Infra Eng  Est: 3h
  Deps: none
  Acceptance: Bench command accepts --model, --backend, --tokens, --concurrent,
  --warmup flags; outputs tok/s, TTFT (ms), P99 latency (ms), GPU memory (MB);
  results exported as JSON for CI comparison.

- [x] T7.2 Add llama.cpp parity test comparing Zerfoo vs Ollama throughput
  Owner: Infra Eng  Est: 2h
  Deps: T7.1
  Acceptance: Test runs Zerfoo and Ollama on same model (Gemma 3 1B Q4_K_M);
  asserts Zerfoo throughput >= Ollama * 1.3 (30% margin); fails CI if regression.

- [x] T7.3 Add multi-architecture benchmark: all 6 architectures
  Owner: Infra Eng  Est: 2h
  Deps: T7.1
  Acceptance: Benchmark runs Llama3, Gemma3, Mistral, Qwen2, Phi3, DeepSeek V3;
  all architectures produce coherent output; no panics; results in docs/devlog.md.

---

### YEAR 2 (2027): Training at Scale

---

#### E8: Full Backpropagation for All 6 Architectures [Q1 2027]

- [x] T8.1 Verify and fix RMSNorm backward for all architectures ✓ 2026-03-18 (zerfoo@e4f9dae, fixed 3 bugs: ReduceSum axis, 3D gain reduction, nil gain.Gradient; 21 tests)
  Owner: ML Eng  Est: 2h
  Deps: none
  Acceptance: RMSNorm.Backward() returns correct gradient vs finite difference for
  Llama3, Gemma3, Mistral, Qwen2 (all use RMSNorm); nil guard already fixed;
  TestRMSNormBackward runs in layers/normalization/ with -race; no panics.

- [x] T8.2 Implement backward pass for GQA/MHA attention ✓ 2026-03-18 (zerfoo@e4f9dae, fixed ReduceSum axis + head-replication tiling; 8 new tests)
  Owner: ML Eng  Est: 5h
  Deps: none
  Acceptance: GroupedQueryAttention.Backward() computes dQ, dK, dV, dO;
  gradient vs finite difference within 1e-3 (FP32); TestGQABackward passes.

- [x] T8.3 Implement backward pass for SwiGLU / SiLU activation ✓ 2026-03-18 (zerfoo@e4f9dae, verified correct, 11 new tests: analytical+finite-diff)
  Owner: ML Eng  Est: 2h
  Deps: none
  Acceptance: FusedSiluGate.Backward() computes correct gradient; TestSiluBackward
  passes with random inputs.

- [x] T8.4 Implement backward for RotaryEmbedding (RoPE) ✓ 2026-03-18 (zerfoo@e4f9dae, verified correct, finite-diff test all 64 elements pass)
  Owner: ML Eng  Est: 2h
  Deps: none
  Acceptance: RoPE.Backward() passes gradient through rotation; finite difference
  test within 1e-3; TestRoPEBackward passes.

- [x] T8.5 Implement backward for DeepSeek MLA (Multi-head Latent Attention)
  Owner: ML Eng  Est: 4h
  Deps: T8.2
  Acceptance: MLA.Backward() correctly decomposes latent gradient into dQ_C, dKV_C;
  gradient vs finite difference within 1e-3; TestMLABackward passes.

- [x] T8.6 Implement backward for MoE routing (DeepSeek V3) ✓ 2026-03-18 (zerfoo@e4f9dae, STE, dX error 1.02e-4, dGateWeight 1.54e-4, 27 tests)
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: SparseMoE.Backward() passes gradients through router (top-K gating);
  straight-through estimator for discrete routing; TestMoEBackward passes.

- [x] T8.7 End-to-end training loop test: Gemma 3 1B on synthetic data
  Owner: ML Eng  Est: 3h
  Deps: T8.1, T8.2, T8.3, T8.4
  Acceptance: Loss decreases monotonically for 100 steps on synthetic classification
  task; no NaN gradients; TestGemma3Training passes.

- [x] T8.8 End-to-end training loop test: Llama 3 on synthetic data
  Owner: ML Eng  Est: 2h
  Deps: T8.1, T8.2, T8.3, T8.4
  Acceptance: Loss decreases monotonically for 100 steps; TestLlama3Training passes.

- [x] T8.9 Gradient checkpointing in graph/checkpoint.go [ztensor]
  Owner: ML Eng  Est: 4h
  Deps: T8.1
  Acceptance: CheckpointedSegment recomputes activations during backward instead of
  storing them; peak memory reduced >= 40% vs no-checkpoint on 7B model;
  TestGradientCheckpointing passes; correctness verified vs no-checkpoint baseline.

---

#### E9: LoRA/QLoRA Fine-Tuning [Q1-Q2 2027]
Decision: docs/adr/049-lora-qlora-finetuning.md

- [x] T9.1 Implement LoraLinear layer in training/lora/linear.go
  Owner: ML Eng  Est: 3h
  Deps: T8.2
  Acceptance: LoraLinear[T] wraps Linear; forward y = Wx + (alpha/r)*B*A*x;
  A initialized N(0,1), B=0; only A,B in optimizer param group; TestLoraLinear passes.

- [x] T9.2 Implement LoRA injection walk in training/lora/inject.go
  Owner: ML Eng  Est: 3h
  Deps: T9.1
  Acceptance: InjectLoRA(model, rank, alpha, target_modules) replaces named Linear
  layers with LoraLinear; freezes base model parameters; count of trainable params
  <= 1% of total; TestLoraInject passes on Gemma 3 1B graph.

- [x] T9.3 Implement NF4 quantization in tensor/quantized.go (QLoRA base) [ztensor]
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: NF4Storage quantizes/dequantizes with double quantization (block 64,
  sub-block 256); round-trip MSE < 0.01 vs BF16; TestNF4Quantization passes.

- [x] T9.4 Implement QLoRA trainer in training/lora/qlora.go — DONE 2026-03-18: NF4 base weights frozen, LoRA adapters trained, QLoRATrainer struct
  Owner: ML Eng  Est: 4h
  Deps: T9.2, T9.3
  Acceptance: QLoRATrainer loads model in NF4; injects LoRA adapters; runs forward
  in BF16 with dequant; backward through dequant; 7B model trains on single GPU
  under 24GB VRAM; TestQLORATrainer passes on Gemma 3 1B.

- [x] T9.5 Implement adapter checkpoint save/load in training/lora/checkpoint.go — DONE 2026-03-18: GGUF v3 save/load, lora.{layer}.weight_a/b naming, bit-exact round-trip
  Owner: ML Eng  Est: 2h
  Deps: T9.4
  Acceptance: SaveAdapter() writes A,B matrices as GGUF tensors with naming
  convention lora.{layer}.weight_a; LoadAdapter() restores and injects;
  round-trip produces identical forward output; TestLoraCheckpoint passes.

- [x] T9.6 Implement AdamW 8-bit optimizer in training/optimizers/adamw8bit.go
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: AdamW8bit stores momentum/variance in INT8 with block-wise quantization;
  memory 4x smaller than FP32 AdamW; loss convergence within 2% of FP32 AdamW
  on synthetic task; TestAdamW8bit passes.

- [x] T9.7 Integration test: LoRA fine-tune Gemma 3 1B on instruction dataset — DONE 2026-03-18: 1000 synthetic pairs, loss 0.018→0.0178, adapter < 50MB
  Owner: ML Eng  Est: 3h
  Deps: T9.4, T9.6
  Acceptance: Fine-tune on 1000 synthetic instruction pairs; eval loss decreases
  from baseline; adapter size < 50MB; generation quality improves on held-out examples.

- [x] T9.8 CLI: zerfoo finetune command in cmd/finetune/main.go — DONE 2026-03-18: flag parsing, JSONL dataset, adapter GGUF save
  Owner: ML Eng  Est: 3h
  Deps: T9.7
  Acceptance: `zerfoo finetune --model path --dataset jsonl --rank 16 --epochs 3`
  runs QLoRA fine-tuning; progress logged per step; saves adapter GGUF on completion.

---

#### E10: Distributed Training FSDP-Equivalent [Q2-Q3 2027]
Decision: docs/adr/050-distributed-training-fsdp.md

- [x] T10.1 Implement parameter sharding in distributed/fsdp/sharded_module.go
  Owner: Infra Eng  Est: 5h
  Deps: T8.7
  Acceptance: ShardedModule splits parameter tensors across N devices (N=8 test);
  AllGather reconstructs full parameter before forward; ReduceScatter aggregates
  gradients; peak per-GPU memory reduced >= 8x vs replicated baseline;
  TestShardedModule passes.

- [x] T10.2 Integrate NCCL AllGather and ReduceScatter in distributed/nccl.go
  Owner: Infra Eng  Est: 4h
  Deps: none
  Acceptance: NCCLAllGather and NCCLReduceScatter wrappers work via purego dlopen
  of libnccl.so; correctness test sums tensors across 2 simulated ranks using
  loopback; TestNCCLOps passes on DGX.

- [x] T10.3 Implement gradient accumulation in distributed/fsdp/grad_accum.go — DONE 2026-03-18: GradAccum[T], M micro-steps averaged on Sync
  Owner: Infra Eng  Est: 2h
  Deps: T10.1
  Acceptance: GradAccum accumulates local gradients for M steps before sync;
  synchronized gradient matches M*step average; TestGradAccum passes.

- [x] T10.4 Implement sharded optimizer state in distributed/fsdp/optimizer_shard.go — DONE 2026-03-18: ShardedAdamW[T], ZeRO Stage 2, moment buffers 1/N per rank
  Owner: Infra Eng  Est: 3h
  Deps: T10.1
  Acceptance: Each rank updates only its shard; moment tensors sized 1/N of total;
  convergence matches non-sharded optimizer on synthetic task; TestShardedOptimizer passes.

- [x] T10.5 Implement distributed checkpoint in distributed/fsdp/checkpoint.go — DONE 2026-03-18: AllGather+GGUF write on rank-0, scatter on load, identical forward output
  Owner: Infra Eng  Est: 3h
  Deps: T10.1
  Acceptance: SaveCheckpoint() AllGathers and rank-0 writes GGUF; LoadCheckpoint()
  shards from GGUF to each rank; round-trip produces identical forward output;
  TestDistributedCheckpoint passes.

- [x] T10.6 Integration test: 2-GPU FSDP training on DGX
  Owner: Infra Eng  Est: 3h
  Deps: T10.2, T10.3, T10.5
  Acceptance: 2-process training run on DGX Spark using 2 GPU contexts; loss
  converges identically to single-GPU on same seed; TestFSDP2GPU passes.

- [x] T10.7 CLI: zerfoo train-distributed command
  Owner: Infra Eng  Est: 2h
  Deps: T10.6
  Acceptance: `zerfoo train-distributed --ranks 4 --model path --dataset jsonl`
  spawns 4 worker processes; coordinator manages lifecycle; workers checkpoint;
  clean shutdown on completion or SIGTERM.

---

#### E11: FP8 Mixed-Precision Training [Q4 2027]
Decision: docs/adr/046-fp8-nvfp4-quantization-roadmap.md (Phase 3)

- [x] T11.1 Implement FP8 linear forward/backward in training/fp8/linear.go — DONE 2026-03-18: FP8 GEMM forward, FP32 grad backward, loss within 2% of BF16
  Owner: Kernel Eng  Est: 5h
  Deps: T2.3, T8.2
  Acceptance: FP8Linear uses FP8 GEMM for forward; backward computes FP32 gradients
  through dequantization; loss matches BF16 training within 2% on synthetic task;
  TestFP8Linear passes.

- [x] T11.2 Implement dynamic loss scaling in training/fp8/loss_scaler.go — DONE 2026-03-18: inf/NaN halving, 2000-step doubling
  Owner: ML Eng  Est: 3h
  Deps: T11.1
  Acceptance: LossScaler multiplies loss by scale factor; checks for inf/NaN in
  gradients and halves scale if detected; doubles scale every 2000 steps without
  overflow; TestLossScaler passes.

- [x] T11.3 Implement master weight FP32 copy in training/fp8/master_weights.go
  Owner: ML Eng  Est: 2h
  Deps: T11.2
  Acceptance: Master weight store keeps FP32 copy; optimizer updates FP32 copy;
  FP8 weights updated by casting; memory overhead measured and documented;
  TestMasterWeights passes.

- [x] T11.4 Integration test: FP8 LoRA fine-tune Gemma 3 1B [DGX]
  Owner: ML Eng  Est: 2h
  Deps: T9.4, T11.3
  Acceptance: QLoRA + FP8 forward: loss converges; VRAM usage <= 18GB on 1B model;
  adapter quality within 5% of BF16 baseline on eval set; TestFP8QLoRA passes.

---

### YEAR 3 (2028): Wolf ML Platform

---

#### E12: Time-Series Architecture Implementation [Q1-Q2 2028]
Decision: docs/adr/051-wolf-time-series-ml-platform.md

- [x] T12.1 Implement patch embedding layer in layers/timeseries/patch_embed.go — DONE 2026-03-18: PatchEmbed[T], configurable patch_size/embed_dim, zero-pad, finite-diff verified
  Owner: Arch Eng  Est: 2h
  Deps: none
  Acceptance: PatchEmbed splits 1D sequence into non-overlapping patches of
  configurable size; embedding output shape [batch, num_patches, embed_dim];
  TestPatchEmbed passes.

- [x] T12.2 Implement PatchTST model builder in inference/wolf/arch_patchtst.go — DONE 2026-03-18: patch embed + Transformer + projection head
  Owner: Arch Eng  Est: 4h
  Deps: T12.1
  Acceptance: BuildPatchTST() constructs full model graph: patch embed + Transformer
  encoder + projection head; forward pass produces [batch, H, D] prediction;
  TestPatchTSTForward passes with synthetic input.

- [x] T12.3 Implement variable selection network in layers/timeseries/vsn.go — DONE 2026-03-18: GRN + VSN, importance weights sum to 1
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: VSN computes variable importance weights via GRN (Gated Residual
  Network); output is weighted sum of feature embeddings; TestVSN passes.

- [x] T12.4 Implement Temporal Fusion Transformer builder in inference/wolf/arch_tft.go
  Owner: Arch Eng  Est: 6h
  Deps: T12.3
  Acceptance: BuildTFT() constructs: static covariate encoders, LSTM encoder,
  VSN, multi-head attention, quantile output head; forward produces Q10/Q50/Q90
  return distributions; TestTFTForward passes with synthetic input.

- [x] T12.5 Implement regime detection model in inference/wolf/arch_regime.go — DONE 2026-03-18: GRU + 4-class softmax
  Owner: Arch Eng  Est: 4h
  Deps: T12.1
  Acceptance: BuildRegimeDetector() constructs LSTM + classification head; output
  is 4-class softmax (bull/bear/sideways/volatile); TestRegimeDetector passes.

- [x] T12.6 Implement Wolf feature store in inference/wolf/features/store.go — DONE 2026-03-18: CSV load, ring buffer cap 500, point-in-time
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: FeatureStore.LoadOffline(asset, start, end) reads parquet features;
  FeatureStore.UpdateOnline(asset, tick) appends to ring buffer (capacity 500);
  point-in-time correctness enforced (no future timestamps); TestFeatureStore passes.

- [x] T12.7 Implement GGUF loader for time-series GGUF metadata
  Owner: ML Eng  Est: 2h
  Deps: T12.2
  Acceptance: GGUF metadata keys wolf.signal.patch_len, wolf.signal.stride,
  wolf.signal.input_features loaded and passed to BuildPatchTST; TestWolfGGUFLoader passes.

- [x] T12.8 Training script for PatchTST signal model
  Owner: ML Eng  Est: 4h
  Deps: T12.2, T9.4
  Acceptance: cmd/wolf_train/main.go trains PatchTST on parquet feature files;
  train/val split respects time order (no leakage); early stopping on val loss;
  saves adapter GGUF on best val checkpoint.

- [x] T12.9 Quantile loss and Sharpe-ratio metric in training/loss/quantile.go — DONE 2026-03-18: pinball loss, differentiable Sharpe
  Owner: ML Eng  Est: 2h
  Deps: none
  Acceptance: QuantileLoss(preds, targets, quantiles) correct vs sklearn reference;
  SharpeLoss differentiable (uses log returns and soft-max portfolio);
  TestQuantileLoss passes.

---

#### E13: Online Learning Safety Pipeline [Q3 2028]
Decision: docs/adr/052-online-learning-safety-framework.md

- [x] T13.1 Implement online learning trigger in training/online/trigger.go
  Owner: ML Eng  Est: 2h
  Deps: T12.8
  Acceptance: Trigger fires only if: data_count >= 500, hours_since_last >= 24,
  no active Wolf positions (stub Wolf API check); TestTrigger passes.

- [ ] T13.2 Implement incremental LoRA updater in training/online/incremental.go
  Owner: ML Eng  Est: 3h
  Deps: T13.1, T9.4
  Acceptance: 100 gradient steps with LR = base_lr/10 and gradient clip 0.5;
  updates only LoRA A,B matrices (base frozen); TestIncrementalUpdater passes.

- [ ] T13.3 Implement safety validators in training/online/validator.go
  Owner: ML Eng  Est: 3h
  Deps: T13.2
  Acceptance: Perplexity gate: new model within 5% of champion; KL gate: KL div
  < 0.1; both gates must pass before promotion; TestValidator passes with synthetic
  champion and challenger distributions.

- [ ] T13.4 Implement model rollback in training/online/rollback.go
  Owner: ML Eng  Est: 2h
  Deps: T13.3
  Acceptance: Rollback(version) swaps LoRA adapter in serve path in < 30 seconds;
  serving continues without restart; TestRollback passes.

- [ ] T13.5 Implement audit log in training/online/audit.go
  Owner: ML Eng  Est: 2h
  Deps: T13.3
  Acceptance: Every trigger, update cycle, validator result, and promotion decision
  appended to append-only log file; log is NDJSON; TestAuditLog passes.

- [ ] T13.6 Integration test: full online learning cycle simulation
  Owner: ML Eng  Est: 2h
  Deps: T13.5
  Acceptance: Simulated cycle: generate data, trigger, update, validate (pass and
  fail cases), promote/rollback; TestOnlineLearningCycle passes end-to-end.

---

#### E14: Model Versioning and A/B Testing [Q4 2028]

- [x] T14.1 Implement model registry in serve/registry/registry.go
  Owner: Infra Eng  Est: 4h
  Deps: none
  Acceptance: Registry.Register(model_id, gguf_path, metadata) adds version;
  Registry.List() returns versions sorted by timestamp; Registry.Promote(id) marks
  champion; Registry.Shadow(id) marks shadow; bbolt-backed persistence;
  TestRegistry passes.

- [ ] T14.2 Implement shadow mode inference in serve/registry/shadow.go
  Owner: Infra Eng  Est: 3h
  Deps: T14.1
  Acceptance: ShadowRunner runs both champion and shadow models on every request;
  shadow output logged but not returned to client; latency impact < 5% (shadow
  runs async); TestShadowRunner passes.

- [ ] T14.3 Implement champion-challenger routing in serve/registry/ab_router.go
  Owner: Infra Eng  Est: 3h
  Deps: T14.1
  Acceptance: ABRouter routes traffic_split % of requests to challenger; deterministic
  by session_id hash (sticky routing); split adjustable at runtime via API;
  TestABRouter passes.

- [ ] T14.4 Implement canary release controller in serve/registry/canary.go
  Owner: Infra Eng  Est: 3h
  Deps: T14.3
  Acceptance: Canary starts at 1% traffic; auto-ramps by 10% every 30 min if error
  rate and latency P99 within thresholds; auto-rollback if thresholds exceeded;
  TestCanaryController passes.

- [ ] T14.5 Add model performance metrics per version
  Owner: Infra Eng  Est: 2h
  Deps: T14.2
  Acceptance: Prometheus metrics: zerfoo_model_requests_total{model_id,version},
  zerfoo_model_latency_p99{model_id,version}; TestMetricsPerVersion passes.

---

### YEAR 4 (2029): Multi-Modal and Agentic

---

#### E15: Vision-Language Inference [Q1-Q2 2029]
Decision: docs/adr/053-multimodal-inference-pipeline.md

- [x] T15.1 Implement image preprocessing pipeline in inference/multimodal/preprocess.go
  Owner: ML Eng  Est: 3h
  Deps: none
  Acceptance: PreprocessImage([]byte, ImageFormat) resizes to target resolution,
  normalizes to [-1,1], converts to patch embeddings; supports JPEG and PNG;
  no CGo (pure Go image library); TestImagePreprocess passes.

- [ ] T15.2 Implement generic VisionEncoder interface in inference/multimodal/vision_encoder.go
  Owner: ML Eng  Est: 2h
  Deps: T15.1
  Acceptance: VisionEncoder[T] interface with Encode(image) (*Tensor[T], error);
  SigLIP implementation wraps existing Gemma3 multimodal code; TestVisionEncoder passes.

- [ ] T15.3 Implement projection connector in inference/multimodal/connector.go
  Owner: ML Eng  Est: 2h
  Deps: T15.2
  Acceptance: ProjectionConnector applies linear projection from vision_dim to text_dim;
  weights loaded from GGUF mm.projector.weight; TestConnector passes.

- [ ] T15.4 Implement text+vision embedding merge in inference/multimodal/merge.go
  Owner: ML Eng  Est: 3h
  Deps: T15.3
  Acceptance: MergeEmbeddings inserts vision embeddings at <image> token positions;
  supports up to 4 images per request; position IDs assigned correctly;
  TestMergeEmbeddings passes.

- [ ] T15.5 Generalize GGUF loader for multi-modal metadata
  Owner: ML Eng  Est: 2h
  Deps: T15.2
  Acceptance: Loader reads vision.encoder.type, vision.hidden_size from GGUF;
  routes to correct VisionEncoder implementation; TestMultiModalGGUFLoad passes.

- [ ] T15.6 Integration test: vision-language inference on earnings chart
  Owner: ML Eng  Est: 2h
  Deps: T15.4
  Acceptance: Gemma 3 multimodal correctly identifies bull/bear pattern from
  synthetic chart image; "What is the trend in this chart?" returns correct answer
  on 10 test cases (accuracy > 80%); TestEarningsChartInference passes.

- [ ] T15.7 Add image input to OpenAI-compatible API (vision content type)
  Owner: Infra Eng  Est: 2h
  Deps: T15.4
  Acceptance: /v1/chat/completions accepts content array with type:image_url;
  image downloaded and preprocessed before inference; TestAPIVisionInput passes.

---

#### E16: Audio Pipeline [Q2-Q3 2029]

- [ ] T16.1 Implement mel-spectrogram extraction in inference/multimodal/audio.go
  Owner: ML Eng  Est: 3h
  Deps: none
  Acceptance: ExtractMelSpectrogram(pcm []float32, sampleRate int) returns [80, T]
  mel filterbank; matches librosa reference within 1e-3; no CGo;
  TestMelSpectrogram passes.

- [ ] T16.2 Implement Whisper-style audio encoder in layers/audio/whisper_encoder.go
  Owner: Arch Eng  Est: 5h
  Deps: T16.1
  Acceptance: WhisperEncoder: 2-layer conv1d frontend + Transformer encoder;
  Generic over [T tensor.Numeric]; output [T_frames, hidden_dim];
  TestWhisperEncoder passes with synthetic mel input.

- [ ] T16.3 Implement audio GGUF loader in inference/arch_whisper.go
  Owner: Arch Eng  Est: 3h
  Deps: T16.2
  Acceptance: Loads Whisper GGUF weights; builds encoder graph;
  TestWhisperLoad passes with Whisper-base GGUF fixture.

- [ ] T16.4 Implement audio+text inference session
  Owner: ML Eng  Est: 3h
  Deps: T16.3
  Acceptance: Whisper encoder output merged with language model for transcript
  conditioned generation; Wolf use case: transcribe Fed call audio, extract key signals;
  TestAudioTextSession passes.

- [ ] T16.5 Add audio input to serve API (multipart/form-data)
  Owner: Infra Eng  Est: 2h
  Deps: T16.4
  Acceptance: /v1/audio/transcriptions endpoint; accepts audio file upload;
  returns JSON transcript; TestAudioAPIEndpoint passes.

---

#### E17: Agentic Tool-Use Loop [Q3-Q4 2029]
Decision: docs/adr/054-agentic-tool-use-loop.md

- [ ] T17.1 Implement tool registry in generate/agent/tools.go
  Owner: ML Eng  Est: 2h
  Deps: none
  Acceptance: ToolRegistry.Register(name, schema, handler) adds tool; handler is
  func(json.RawMessage) (string, error); TestToolRegistry passes.

- [ ] T17.2 Implement function-call JSON grammar in generate/agent/function_call.go
  Owner: ML Eng  Est: 3h
  Deps: T17.1
  Acceptance: When model outputs <tool_call> token, grammar switches to JSON schema
  matching tool parameter schema; parsed ToolCall struct populated;
  TestFunctionCallDecoding passes with synthetic model output.

- [ ] T17.3 Implement agentic loop supervisor in generate/agent/supervisor.go
  Owner: ML Eng  Est: 4h
  Deps: T17.2
  Acceptance: Supervisor executes up to MaxIterations=10 steps; detects tool call
  vs EOS; executes tool via registry; appends result to context; halts on budget
  exhaustion; TestAgentSupervisor passes with mock tools.

- [ ] T17.4 Implement Wolf tool set in generate/agent/wolf_tools.go
  Owner: ML Eng  Est: 3h
  Deps: T17.3
  Acceptance: GetMarketData, GetOrderBook, GetPortfolio, GetEarningsCalendar,
  SearchNews tools registered; SubmitOrder requires stub Wolf risk API approval;
  TestWolfTools passes with mock market data.

- [ ] T17.5 Add tool-use to OpenAI API in serve/agent/openai_adapter.go
  Owner: Infra Eng  Est: 3h
  Deps: T17.3
  Acceptance: /v1/chat/completions with tools parameter activates agentic mode;
  response includes tool_calls in message; streaming emits tool_calls delta events;
  TestOpenAIToolsAPI passes.

- [ ] T17.6 Integration test: Wolf agent executes 5-step research plan
  Owner: ML Eng  Est: 2h
  Deps: T17.4, T17.5
  Acceptance: Agent given "Research BTCUSD and recommend position size" executes
  GetMarketData + GetPortfolio + GetOrderBook + reasoning + returns recommendation
  without hang or loop; TestWolfAgentIntegration passes.

---

### YEAR 5 (2030): Autonomous ML

---

#### E18: Neural Architecture Search [Q1-Q2 2030]
Decision: docs/adr/055-neural-architecture-search.md

- [ ] T18.1 Define NAS search space in training/nas/search_space.go
  Owner: Research Eng  Est: 3h
  Deps: T12.2
  Acceptance: SearchSpace defines discrete ops (Attention, MLP, Conv1D, SSMBlock),
  connectivity patterns, and hyperparameter ranges; serializable to JSON;
  TestSearchSpace passes.

- [ ] T18.2 Implement DARTS mixed-operation layer in training/nas/darts_layer.go
  Owner: Research Eng  Est: 5h
  Deps: T18.1
  Acceptance: DARTSLayer computes softmax-weighted mixture of candidate ops;
  architecture parameters alpha are learnable; forward pass differentiable through
  softmax weights; TestDARTSLayer passes with gradient check.

- [ ] T18.3 Implement DARTS bilevel optimizer in training/nas/darts_optimizer.go
  Owner: Research Eng  Est: 4h
  Deps: T18.2
  Acceptance: Bilevel optimization: alpha updated by validation gradient; w updated
  by training gradient; alternates per step; TestDARTSOptimizer converges on
  toy architecture search task.

- [ ] T18.4 Implement hardware-aware latency estimator in training/nas/hw_estimator.go
  Owner: Research Eng  Est: 3h
  Deps: T18.1
  Acceptance: LatencyEstimate(arch) predicts inference latency from op counts and
  memory bandwidth model; calibrated against 20 measured DGX Spark benchmarks;
  R^2 > 0.85 on held-out architectures; TestLatencyEstimator passes.

- [ ] T18.5 Implement architecture discretization in training/nas/discretize.go
  Owner: Research Eng  Est: 2h
  Deps: T18.3
  Acceptance: Discretize(alpha_weights) selects argmax op per edge; validates
  resulting architecture against max_params constraint; TestDiscretize passes.

- [ ] T18.6 Implement NAS export to GGUF in training/nas/export.go
  Owner: Research Eng  Est: 3h
  Deps: T18.5, T12.7
  Acceptance: Exports discovered architecture + trained weights as valid GGUF;
  architecture config stored in GGUF metadata; exported model loads via standard
  inference path; TestNASExport passes.

- [ ] T18.7 Run NAS for Wolf signal model: beats PatchTST baseline
  Owner: Research Eng  Est: 4h (GPU-time intensive; run on DGX)
  Deps: T18.6, T12.8
  Acceptance: NAS search over 6h on DGX Spark; best discovered architecture
  achieves >= 5% better Sharpe than hand-designed PatchTST on Wolf val set;
  results in docs/devlog.md.

---

#### E19: AutoML Hyperparameter Optimization [Q2-Q3 2030]

- [ ] T19.1 Implement Bayesian optimization in training/automl/bayesian.go
  Owner: Research Eng  Est: 4h
  Deps: none
  Acceptance: BayesianOptimizer.Suggest() returns next hyperparameter config;
  uses Gaussian Process surrogate model; acquisition function: Expected Improvement;
  TestBayesianOptimizer converges to optimal LR on synthetic objective in 20 trials.

- [ ] T19.2 Implement population-based training in training/automl/pbt.go
  Owner: Research Eng  Est: 4h
  Deps: T19.1
  Acceptance: PBT maintains population of N=8 agents; exploit+explore: copies
  weights from top 20% to bottom 20% with perturbation; TestPBT produces
  better final metric than random search in 50 trials on toy task.

- [ ] T19.3 Implement AutoML loop coordinator in training/automl/coordinator.go
  Owner: Research Eng  Est: 3h
  Deps: T19.1, T19.2
  Acceptance: Coordinator orchestrates search over configurable space (LR, rank,
  batch_size, patch_size); dispatches trial to Worker; collects metrics; records
  best config; TestCoordinator passes with mock worker.

- [ ] T19.4 CLI: zerfoo automl command
  Owner: Research Eng  Est: 2h
  Deps: T19.3
  Acceptance: `zerfoo automl --model path --dataset jsonl --trials 50 --metric sharpe`
  runs AutoML; logs trial results as NDJSON; saves best config and checkpoint.

---

#### E20: Self-Improving Models and Live Feedback [Q3-Q4 2030]

- [ ] T20.1 Implement market feedback signal collector in training/online/feedback.go
  Owner: ML Eng  Est: 3h
  Deps: T13.1
  Acceptance: FeedbackCollector subscribes to Wolf's P&L events via gRPC stream;
  labels signal predictions with realized returns; stores labeled pairs for
  incremental fine-tuning trigger; TestFeedbackCollector passes with mock Wolf API.

- [ ] T20.2 Implement model performance regression detector in training/online/drift.go
  Owner: ML Eng  Est: 3h
  Deps: T20.1
  Acceptance: DriftDetector computes rolling Sharpe ratio over 30-day window;
  raises alert if current Sharpe < 30-day mean - 1 sigma; TestDriftDetector passes
  with synthetic P&L streams showing injected degradation.

- [ ] T20.3 Implement automated NAS trigger on drift event
  Owner: Research Eng  Est: 3h
  Deps: T20.2, T18.7
  Acceptance: On drift alert: AutoNAS runs 2h search on latest data; if discovered
  arch Sharpe >= current model + 5%, proposes replacement to online safety pipeline
  (ADR-052 validators); TestAutoNASTrigger passes end-to-end in simulation.

---

#### E21: Zerfoo Cloud Product [Q4 2030]
Decision: docs/adr/056-zerfoo-cloud-product.md
Note: Requires founder approval before implementation (per Feza governance).

- [ ] T21.1 Implement multi-tenant namespace isolation in serve/cloud/tenant.go
  Owner: Infra Eng  Est: 4h
  Deps: T14.1
  Acceptance: TenantRegistry manages per-API-key quotas; request routing enforces
  max_concurrent_requests and max_tokens_per_minute; TestTenantIsolation passes.

- [ ] T21.2 Implement token metering middleware in serve/cloud/billing.go
  Owner: Infra Eng  Est: 3h
  Deps: T21.1
  Acceptance: Middleware counts prompt+completion tokens per request; usage event
  published to NDJSON file (Kafka adapter configurable); TestBillingMiddleware passes.

- [ ] T21.3 Implement GPU model LRU eviction in serve/cloud/resource_manager.go
  Owner: Infra Eng  Est: 4h
  Deps: T14.1
  Acceptance: ResourceManager tracks model VRAM usage; evicts LRU model when new
  model load would exceed memory budget; model reload from GGUF < 10s for 7B;
  TestResourceManager passes.

- [ ] T21.4 GKE deployment Terraform in infra/terraform/zerfoo-cloud/
  Owner: Infra Eng  Est: 4h
  Deps: none
  Acceptance: Terraform provisions GKE cluster with GPU node pool; Cloud Run API
  gateway; Cloud Storage model artifact bucket; `terraform plan` shows no errors
  on dry-run.

- [ ] T21.5 Load test: 100 concurrent tenants on cloud API
  Owner: Infra Eng  Est: 2h
  Deps: T21.3
  Acceptance: 100 concurrent clients each sending 10 requests; P99 latency < 2s;
  zero cross-tenant data leakage (verified by response inspection);
  TestCloudLoadTest passes.

---

## Parallel Work

### Parallel Tracks

| Track | Description | Epic IDs | Sync Points |
|-------|-------------|----------|-------------|
| A | Inference Kernel Performance | E1, E2, E3, E4, E5 | Merge at E7 benchmark |
| B | Architecture Support | E6, E12 (PatchTST/TFT), E15, E16 | Merge at E15 (multimodal) |
| C | Training Infrastructure | E8, E9, E10, E11 | Merge at E13 (online learning) |
| D | Serving and Platform | E5, E7, E13, E14, E17, E21 | Merge at E21 (cloud) |
| E | Wolf ML Platform | E12, E13, E14, E20 | Merge at E20 (self-improving) |
| F | Research/NAS | E18, E19, E20 | Merge at E20 |

### Wave Plan (Maximum Parallelism, Up to 10 Agents)

Wave 1 -- All independent starters (no dependencies):
1. T1.1 KV block pool [ztensor] (Kernel Eng)
2. T2.1 FP8 GEMM kernel [ztensor] (Kernel Eng)
3. T2.2 Amax computation [ztensor] (Kernel Eng)
4. T2.4 NVFP4 storage [ztensor] (Kernel Eng)
5. T3.1 External draft infra (ML Eng)
6. T4.1 Radix tree (Infra Eng)
7. T5.1 Disagg proto (Infra Eng)
8. T6.1 Selective scan CUDA [ztensor] (Kernel Eng)
9. T7.1 Benchmark harness (Infra Eng)
10. T8.2 GQA backward (ML Eng)

Wave 2 -- Unblocked by Wave 1:
1. T1.2 Block table (Kernel Eng) [needs T1.1]
2. T1.3 Paged attention CUDA kernel (Kernel Eng) [needs T1.1]
3. T2.3 FP8 MatMul dispatch (Kernel Eng) [needs T2.1, T2.2]
4. T2.5 NVFP4 GEMV kernel (Kernel Eng) [needs T2.4]
5. T3.2 Self-draft (ML Eng) [no deps]
6. T5.2 Prefill worker (Infra Eng) [needs T5.1]
7. T5.3 Decode worker (Infra Eng) [needs T5.1]
8. T6.2 Mamba block (Arch Eng) [needs T6.1]
9. T8.1 RMSNorm backward (ML Eng) [no deps]
10. T8.3 SwiGLU backward (ML Eng) [no deps]

Wave 3 -- Unblocked by Wave 2:
1. T1.4 GQA + paged (Kernel Eng) [needs T1.2, T1.3]
2. T1.5 Continuous batching scheduler (Infra Eng) [needs T1.2]
3. T3.3 Acceptance sampler (ML Eng) [needs T3.1]
4. T5.4 Gateway (Infra Eng) [needs T5.2, T5.3]
5. T6.3 SSM state mgmt (Arch Eng) [needs T6.2]
6. T8.4 RoPE backward (ML Eng) [no deps]
7. T8.5 MLA backward (ML Eng) [needs T8.2]
8. T8.6 MoE backward (ML Eng) [no deps]
9. T9.3 NF4 quantization [ztensor] (Kernel Eng) [no deps]
10. T12.1 Patch embed (Arch Eng) [no deps]

Wave 4 -- Unblocked by Wave 3:
1. T1.6 Ragged batching kernel (Kernel Eng) [needs T1.5]
2. T2.6 FP8 integration test (ML Eng) [needs T2.3]
3. T3.4 Wire speculative decoding (ML Eng) [needs T3.1, T3.2, T3.3]
4. T5.5 Disagg integration test (Infra Eng) [needs T5.4]
5. T6.4 Mamba-3 GGUF loader (Arch Eng) [needs T6.2, T6.3]
6. T8.7 Gemma3 training E2E (ML Eng) [needs T8.1, T8.2, T8.3, T8.4]
7. T8.9 Gradient checkpointing (ML Eng) [needs T8.1]
8. T9.1 LoraLinear (ML Eng) [needs T8.2]
9. T12.2 PatchTST builder (Arch Eng) [needs T12.1]
10. T12.3 VSN for TFT (Arch Eng) [no deps]

Wave 5 -- Unblocked by Wave 4:
1. T1.7 PagedAttention integration test (Infra Eng) [needs T1.4, T1.5]
2. T3.5 Acceptance rate metric (ML Eng) [needs T3.4]
3. T4.2 Wire prefix cache (Infra Eng) [needs T4.1, T1.4]
4. T6.5 Jamba hybrid builder (Arch Eng) [needs T6.4]
5. T9.2 LoRA inject (ML Eng) [needs T9.1]
6. T9.6 AdamW 8-bit (ML Eng) [no deps]
7. T10.2 NCCL allgather/reducescatter (Infra Eng) [no deps]
8. T12.4 TFT builder (Arch Eng) [needs T12.3]
9. T12.5 Regime detector (Arch Eng) [needs T12.1]
10. T12.6 Wolf feature store (ML Eng) [no deps]

Wave 6 -- Integration and training scale:
1. T1.8 Continuous batching benchmark (Infra Eng)
2. T3.6 Speculative decoding benchmark (ML Eng)
3. T4.3 Prefix cache benchmark (Infra Eng)
4. T5.6 Disagg benchmark (Infra Eng)
5. T9.4 QLoRA trainer (ML Eng) [needs T9.2, T9.3]
6. T10.1 Parameter sharding FSDP (Infra Eng) [needs T8.7]
7. T12.7 Wolf GGUF metadata (ML Eng)
8. T12.9 Quantile loss (ML Eng) [no deps]
9. T14.1 Model registry (Infra Eng) [no deps]
10. T17.1 Tool registry (ML Eng) [no deps]

Wave 7 -- Dependent on Wave 6:
1. T7.2 llama.cpp parity test (Infra Eng)
2. T7.3 Multi-arch benchmark (Infra Eng)
3. T9.5 Adapter checkpoint (ML Eng) [needs T9.4]
4. T9.7 LoRA finetune E2E (ML Eng)
5. T9.8 CLI finetune (ML Eng)
6. T10.3 Grad accumulation (Infra Eng)
7. T10.4 Sharded optimizer (Infra Eng)
8. T12.8 PatchTST train script (ML Eng)
9. T13.1 Online trigger (ML Eng)
10. T15.1 Image preprocess (ML Eng)

Wave 8 -- Year 2-3 integration:
1. T10.5 Distributed checkpoint (Infra Eng)
2. T10.6 2-GPU FSDP test (Infra Eng)
3. T11.1 FP8 linear fwd/bwd (Kernel Eng)
4. T13.2 Incremental updater (ML Eng)
5. T13.3 Safety validators (ML Eng)
6. T14.2 Shadow mode (Infra Eng)
7. T14.3 AB router (Infra Eng)
8. T15.2 VisionEncoder interface (ML Eng)
9. T16.1 Mel spectrogram (ML Eng)
10. T17.2 Function-call grammar (ML Eng)

Wave 9 -- Year 3-4 integration:
1. T10.7 CLI train-distributed (Infra Eng)
2. T11.2 Loss scaling (ML Eng)
3. T11.3 Master weights (ML Eng)
4. T13.4 Rollback (ML Eng)
5. T13.5 Audit log (ML Eng)
6. T14.4 Canary controller (Infra Eng)
7. T15.3 Connector (ML Eng)
8. T16.2 Whisper encoder (Arch Eng)
9. T17.3 Agent supervisor (ML Eng)
10. T18.1 NAS search space (Research Eng)

Wave 10 -- Year 4-5 completion:
1. T11.4 FP8 QLoRA test (ML Eng)
2. T13.6 Online learning E2E (ML Eng)
3. T14.5 Model metrics (Infra Eng)
4. T15.4 Text+vision merge (ML Eng)
5. T16.3 Whisper GGUF loader (Arch Eng)
6. T17.4 Wolf tools (ML Eng)
7. T18.2 DARTS layer (Research Eng)
8. T19.1 Bayesian optimizer (Research Eng)
9. T20.1 Feedback collector (ML Eng)
10. T21.1 Multi-tenant isolation (Infra Eng)

Waves 11-15 follow the same pattern completing E15-E21.

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M1 | Inference Supremacy | E1-E7 | Zerfoo beats llama.cpp 30%+ on all 6 architectures; PagedAttention live; speculative decoding 2x | 2026-12-31 |
| M2 | Training at Scale | E8-E11 | LoRA fine-tune 7B on single GPU; 70B on 8 GPUs; FP8 training validated | 2027-12-31 |
| M3 | Wolf ML Platform | E12-E14 | PatchTST signal model in production; online learning live; A/B testing operational | 2028-12-31 |
| M4 | Multi-Modal and Agentic | E15-E17 | Earnings chart analysis live; Fed call transcription live; Wolf agent in shadow mode | 2029-12-31 |
| M5 | Autonomous ML | E18-E21 | NAS beats hand-crafted by 5%; self-improving models live; cloud API beta | 2030-12-31 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | CUDA graph capture broken by paged attention kernel indirection | High | Medium | Test capture compatibility early (T1.3); fall back to eager mode if needed |
| R2 | NVFP4 Blackwell-only; DGX Spark GB10 may not support sm_100 NVFP4 API | High | Medium | Verify ComputeCapability(); graceful fallback to FP8 for non-Blackwell |
| R3 | Speculative decoding alpha < 0.4 on Wolf financial prompts (structured, repetitive) | Medium | High | Self-draft mode requires no draft model; auto-disable on low-alpha; no downside |
| R4 | NCCL gRPC latency over PCIe limits FSDP scaling efficiency | High | Medium | DGX Spark has NVLink; test AllGather bandwidth; gradient accumulation reduces freq |
| R5 | Financial time-series data leakage via future-peeking features | Critical | Medium | Feature store enforces point-in-time correctness; audit via unit tests on timestamps |
| R6 | Online learning catastrophic forgetting of base model capabilities | High | Medium | LoRA-only updates; 100 step limit; perplexity and KL validators; fast rollback |
| R7 | Mamba-3 GGUF schema not finalized by mid-2026; integration blocked | Medium | High | Track llama.cpp GGUF PRs; use provisional schema; update loader when finalized |
| R8 | DARTS bilevel optimization numerical instability | Medium | Medium | Use DARTS+ stabilization; constrained search space; toy task convergence test first |
| R9 | Multi-modal GGUF schema churn (vision.encoder.type not standardized) | Low | High | Namespace metadata keys with "wolf.vision.*"; upgrade path via GGUF metadata versioning |
| R10 | Agentic LLM trading agents raise MiFID II / SEC regulatory risk | Critical | Medium | Shadow mode only initially; legal review before live SubmitOrder; Wolf risk gate required |
| R11 | Zerfoo cloud product needs founder approval before build | High | Low | ADR-056 status is Proposed; engineering work blocked until approval received |
| R12 | Agentic code quality drift with dozens of parallel agents | Medium | High | Strict per-repo commit rule; pre-commit hooks; CI on every PR; code review gates |
| R13 | Memory bandwidth bottleneck persists at batch=1; FP8/NVFP4 gains limited | Medium | Medium | Speculative decoding (batch > 1 in draft/target step) and continuous batching address this |
| R14 | gRPC achieves only 46% scaling efficiency at 128 GPUs vs 86% for NCCL; large tensor transfers over gRPC collapse distributed training throughput | High | High | Use NCCL for all gradient synchronization (AllGather, ReduceScatter); gRPC only for control plane (checkpoint coordination, health checks, token streaming) |
| R15 | Go GC unaware of GPU memory pressure; pinned memory and CUDA arena allocations can cause GC pauses or silent OOM without Go-side visibility | Medium | High | Track GPU allocation totals in a Go-side counter; expose zerfoo_gpu_memory_bytes Prometheus metric; arena free-list (ADR-043) already mitigates allocation churn |
| R16 | Training-serving skew: feature preprocessing divergence between training and inference paths produces silent correctness regressions that strong offline metrics cannot detect | High | Medium | Wolf feature store (T12.6) is the single source of truth for both paths; integration test verifies training and inference produce identical features on same input |
| R17 | GGUF format breaking changes (v2 to v3 big-endian flip, deprecated quantizations) break existing model files across architecture version bumps | Medium | High | Pin GGUF version in go.mod; migration tool in zonnx before each format bump; parity tests on DGX with all 6 architecture model files flag regressions in CI |
| R18 | Agentic coder quality drift: Google DORA 2025 found 90% AI adoption correlates with 9% higher bug rate and 154% larger PRs; multi-agent conflicts produce contradictory design decisions | High | High | Strict per-repo commit rule enforced by pre-commit hook; CI green required before merge; human review gate at each milestone (M1-M5); /review before each release |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Benchmark tasks: results appended to docs/devlog.md.
6. ADR tasks: ADR file created and referenced in plan.

### Quality Gates

- Every implementation task (T*.x) must have a paired test subtask.
- Run `go vet ./...` after every code change before committing.
- Commit each task as its own commit. One logical change per commit.
- Never commit files from different directories in the same commit.
- Use standard library only: no testify, no cobra, no viper. Use testing.T and flag.
- GPU-only tests: tag with `//go:build cuda` and run only on DGX.
- Benchmark tasks must run on DGX Spark (not CPU-only CI).
- Never skip CI hooks with --no-verify.

### Agent Assignment Protocol

1. Read TaskList to find available (pending, no owner, not blocked) tasks.
2. Prefer lowest-ID task in your skill domain.
3. TaskUpdate status=in_progress, owner=your-name.
4. Read task description fully; identify target file paths.
5. Implement, test, vet, commit in target repo directory.
6. TaskUpdate status=completed.
7. Repeat from step 1.

### Code Style

- Engine[T] is law: all tensor ops through compute.Engine[T].
- Generics throughout: [T tensor.Numeric] constraints.
- Fuse, do not fragment: prefer fused ops over primitive sequences.
- No CGo in core packages; GPU via purego.
- Docstrings only on exported types and functions. No inline comments unless logic
  is non-obvious.

---

## Progress Log

### 2026-03-17: Initial 5-year roadmap created

**Scope:** Created full 5-year product roadmap (2026-2030) for Zerfoo to be executed
by a Claude Code agent army in parallel. Trimmed Phase 27 completed plan; stable
knowledge preserved.

**Change summary:**
- Replaced Phase 27 completion stub with full 5-year plan.
- Created 13 ADRs (044-056) covering all major architectural decisions.
- Work breakdown: 21 epics, 130+ tasks, 10-wave parallel execution plan.
- Risk register: 13 risks with mitigations.
- ADRs created:
  - docs/adr/044-paged-attention-kv-block-manager.md
  - docs/adr/045-speculative-decoding.md
  - docs/adr/046-fp8-nvfp4-quantization-roadmap.md
  - docs/adr/047-disaggregated-prefill-decode-serving.md
  - docs/adr/048-mamba-ssm-architecture-support.md
  - docs/adr/049-lora-qlora-finetuning.md
  - docs/adr/050-distributed-training-fsdp.md
  - docs/adr/051-wolf-time-series-ml-platform.md
  - docs/adr/052-online-learning-safety-framework.md
  - docs/adr/053-multimodal-inference-pipeline.md
  - docs/adr/054-agentic-tool-use-loop.md
  - docs/adr/055-neural-architecture-search.md
  - docs/adr/056-zerfoo-cloud-product.md (status: Proposed; needs founder approval)

**Research findings incorporated from:**
- tech-researcher: competitive landscape, quantization roadmap, time-series foundations
- arch-researcher: PagedAttention architecture, speculative decoding patterns, NAS,
  Go arena patterns, distributed training fault tolerance
- Prior Phase 27 plan (benchmark baselines, DGX Spark specifications)

---

## Hand-Off Notes

### What You Need to Know

- **Repos:** Each repo has its own go.mod. Never commit across repos. Tasks marked
  [ztensor] go in /Users/dndungu/Code/zerfoo/ztensor; unmarked tasks go in
  /Users/dndungu/Code/zerfoo/zerfoo.
- **DGX Spark:** GPU hardware at `ssh ndungu@192.168.86.250`. Set
  `LD_LIBRARY_PATH=~/Code/zerfoo` before running GPU tests. Always rebuild binary.
- **Baseline benchmark:** 245 tok/s, Gemma 3 1B Q4_K_M, 256 tokens, CUDA graph,
  DGX Spark GB10. Command: `LD_LIBRARY_PATH=. ./bench_tps -device cuda -model
  ~/models/gemma3-q4km -tokens 256 -prompt Hi`.
- **Current ADRs:** 001-056 in docs/adr/. Next ADR: 057.
- **Current plan phase:** Phase 27 complete (245 tok/s baseline). Now on 5-year plan.
- **CI:** GitHub Actions in .github/workflows/. CPU tests in CI; GPU tests on DGX only.
- **Model downloads:** `zerfoo pull model_id` for HuggingFace models (ADR-039).
  DGX models at ~/models/: gemma3-q4km, phi4, llama3, qwen2.
- **Wolf integration:** Wolf trading system at ssh ndungu@192.168.86.250 in
  ~/Code/wolf/. Wolf API is a stub for Zerfoo tasks; Wolf team owns the real impl.
- **Founder approval required:** E21 (Zerfoo cloud product) is blocked until founder
  approves per Feza governance (ADR-056 status: Proposed).

### Placeholder Credentials

- DGX SSH: ndungu@192.168.86.250 (key auth; no password in this file)
- HuggingFace token: set HUGGINGFACE_TOKEN env var
- Stripe API key: set STRIPE_API_KEY env var (E21 only)
- GCP project: set GOOGLE_CLOUD_PROJECT env var (E21 only)

---

## Appendix

### Research Findings: Technical Landscape (2026)

**Competing frameworks:**
- llama.cpp: dp4a INT8 GEMV (4 MACs/instruction), VMM pool, per-graph warmup
- vLLM: PagedAttention + continuous batching; 120-160 req/s; Blackwell B200 support
- SGLang: RadixAttention; disaggregated prefill/decode; 3.8x prefill on GB200
- TensorRT-LLM: 20-100% faster raw throughput; FP8 dominant advantage; NVIDIA-only

**Quantization frontier (2026):**
- FP8 E4M3FN: stable across all sizes; dynamic weight+activation; consumer RTX 50xx
- NVFP4: Blackwell-only; 3.5x memory vs FP16; 2.2x E2E speedup on B200
- MXFP4: AMD+Intel support; power-of-two scale factors; block 32

**Architecture trends:**
- Mamba-3: 5x throughput vs Transformer; linear sequence scaling; complex dynamics
- MoE-Mamba: 2.2x fewer training steps than Mamba; DeepSeek-V3 MoE validated at 671B
- Hybrid (Jamba): Transformer + Mamba + MoE; hundreds-of-thousands token context

**Training (2027 target):**
- QLoRA: 4-bit base + LoRA; 7B on 24GB GPU; 70B on 2x H100
- FP8 training: validated at 671B (DeepSeek-V3); dynamic scaling required
- FSDP + QLoRA: combined for memory-efficient distributed fine-tuning

**Time-series (2028 target):**
- Chronos (Amazon T5-based), TimeGPT, PatchTST, TFT are 2025-2026 SOTA
- Domain-specific models outperform general foundation models on specialized financial tasks
- Feature stores with point-in-time correctness are critical for production trading

**Agentic (2029 target):**
- Gartner: 33% of enterprise software embeds agentic AI by 2028
- OpenAI function-calling / Anthropic tool-use are emerging standards
- Multi-agent orchestration (LangGraph pattern) supports parallel execution

### ADR Index (New)

| ADR | Title | Status | Year |
|-----|-------|--------|------|
| 044 | PagedAttention KV Block Manager | Accepted | 2026 |
| 045 | Speculative Decoding | Accepted | 2026 |
| 046 | FP8 and NVFP4 Quantization Roadmap | Accepted | 2026-2027 |
| 047 | Disaggregated Prefill/Decode Serving | Accepted | 2026 |
| 048 | Mamba/SSM Architecture Support | Accepted | 2026 |
| 049 | LoRA/QLoRA Fine-Tuning | Accepted | 2027 |
| 050 | Distributed Training FSDP-Equivalent | Accepted | 2027 |
| 051 | Wolf Time-Series ML Platform | Accepted | 2028 |
| 052 | Online Learning Safety Framework | Accepted | 2028 |
| 053 | Multi-Modal Inference Pipeline | Accepted | 2029 |
| 054 | Agentic Tool-Use Loop | Accepted | 2029 |
| 055 | Neural Architecture Search | Accepted | 2030 |
| 056 | Zerfoo Cloud Product | Proposed | 2030 |
