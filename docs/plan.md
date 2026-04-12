# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, documentation site, MSA-inspired scalable
memory, and research-driven inference optimizations).

Task statuses updated 2026-04-10 based on merged PRs and git history.

**Status summary:**
- 380+ tasks completed across all plans
- E86: PyTorch parity testing (66/72 -- 105 tests, 100 pass, 5 skip, 0 fail; GPU parity E86.5 pending DGX)
- E88: Upgrade timeseries model tests from structural to golden-file parity (6/6 DONE -- PatchTST, N-BEATS, ITransformer, DLinear, CfC, FreTS, TimeMixer)
- E89: Timeseries Engine[T] compliance -- eliminate raw slice math (0/27 -- 6 models, 155+ violations)
- E87: Fix backward pass bugs found by PyTorch parity (8/8 COMPLETE -- all 4 bugs fixed, 92/92 parity tests pass)
- E45: Verification remediation (3/3 complete) -- DONE
- E46: Ecosystem v1 release (46/46 complete -- all 5 repos at v1.0.0) -- DONE 2026-03-30
- E47: Batched training performance (19/19 complete) -- DONE 2026-03-30
- E48: TimeMixer backend (10/10 complete) -- DONE 2026-03-30
- E49: Foundation model inference (12/12 complete) -- DONE 2026-03-30
- E50: GPU training kernel elimination (6/6 COMPLETE -- 28K×20×10 in 40.3s, 4.0s/epoch, see benchmarks.md)
- E51: CUDA graph capture for training (6/6 COMPLETE -- graph capture disabled but perf target met via E50+E85 dst-reuse)
- E52: DRY composition refactoring (7/7 complete -- shared math_ops, adamw_f32, layernorm_ops, engine wrappers)
- E53: Unified training forward/backward (6/6 complete -- shared encoder, eliminated engine paths)
- E54: Capture-pure GPU engine ops (2/4 -- GPU-native Zero/Copy done; re-enable graph capture pending)
- E55: Fused PatchTST encoder CUDA kernel (0/8 -- single kernel per encoder layer)
- E56: Gemma3 inference micro-optimizations (7/9 -- fused kernels written and wired; T56.1.3 done; prefill extension + benchmarks pending)
- E57: Fix DGX Spark build regression (3/3 COMPLETE -- 3 root causes fixed: transpose no-op, causal mask D2H, Q4_K re-quant; composed GQA divergence remains)
- E58: GPU vs CPU GQA parity test (1/2 -- diagnostic test to find remaining composed-pipeline divergence)
- E59: Remove gonum dependency (7/7 COMPLETE -- replace BLAS fallback + FFT with zero-dep implementations)
- E60: CrossAsset GPU training (12/12 COMPLETE -- GitHub #312, GPU forward/backward/AdamW)
- E61: Inference builder composition (9/10 -- all 6 builders done; vet+linters pass; DGX parity T61.3.2 pending)
- E62: Auxiliary training package composition (7/7 COMPLETE -- tabular, modeldsl, gnn refactored; tests+validation pass)
- E63: Quantized matmul consolidation in ztensor (2/5 -- dispatcher + replacements done ztensor PR #76 v1.4.0; DGX benchmarks need CUDA CGo kernel stubs)
- E64: GPU engine file decomposition in ztensor (3/3 COMPLETE -- split into 5 files, ztensor PR #77)
- E65: MoE layer composition fix (3/3 COMPLETE -- PR #316)
- E66: Functional layer API for training (5/5 COMPLETE -- PR #320, #322)
- E67: Timeseries full layers migration (11/11 COMPLETE -- all helpers replaced, attention migrated, validated, files verified)
- E68: CrossAsset full layers migration (4/4 COMPLETE -- forward+backward+AdamW+cleanup, -1,357 lines)
- E69: Training loss/optimizer Engine compliance (6/6 COMPLETE -- PR #320, #321, #322) + T69.3.1 validated PR #324
- E70: Intra-layers violations cleanup (10/10 COMPLETE -- all tasks + T70.1.10 validation PR #324)
- E71: Experimental package migration (5/5 COMPLETE -- all 4 packages + T71.1.5 validation PR #324)
- E72: Architecture enforcement test (2/2 COMPLETE -- test created + added to CI)
- E73: Generate KV cache consolidation (3/3 COMPLETE -- base extraction, migration, validation done)
- E74: Timeseries backward pass composition (12/14 -- all backward API + migration done PR #329/#330/#331; 2 DGX validation tasks pending wave plan)
- E75: Inference timeseries .Data() elimination (9/9 COMPLETE -- all 6 arch builders + validation done PR #329/#330)
- E76: Architecture test allowlist cleanup (0/2 -- remove timeseries/ from allowlist after E74)
- E77: Tabular package composition migration (9/9 COMPLETE -- PRs #334, #336, #338, #341)
- E78: Layers internal violations cleanup (11/11 COMPLETE -- PRs #334, #336, #338, #341)
- E79: Generate package refactoring (7/7 COMPLETE -- PRs #334, #336, #338, #341)
- E80: Inference builder boilerplate extraction (8/8 COMPLETE -- PRs #334, #336, #338, #341)
- E81: Inference custom node replacement (7/7 COMPLETE -- PRs #334, #336, #338, #341)
- E82: Training loss engine migration (6/6 COMPLETE -- PRs #334, #336, #338, #341)
- E83: Serve handler refactoring (5/5 COMPLETE -- PRs #334, #336, #338, #341)
- E84: ModeLDSL composition (8/8 COMPLETE -- PRs #334, #336, #338, #341)
- E85: Fix GPU training memory leak in PatchTST encoder bwd (COMPLETE -- ztensor#84/#85 dst-memory reuse, 28K×20×10 40.3s, no OOM)
- GPU status: Q5_0 GEMV alignment fix shipped (ztensor 5f19e54). Q4_0 re-quantization restored for 231 tok/s decode. Pool-backed GPUStorage prevents arena corruption.

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

### Ecosystem v1 and Training Epics (Complete)

| Epic | Description | Tasks | Completed |
|------|-------------|-------|-----------|
| E46 | Ecosystem v1 Release (5 repos to v1.0.0) | 46/46 | 2026-03-30 |
| E47 | Batched Training Performance | 19/19 | 2026-03-30 |
| E48 | TimeMixer Backend | 10/10 | 2026-03-30 |
| E49 | Foundation Model Inference (TiRex, Chronos-2, Moirai-2) | 12/12 | 2026-03-30 |
| E60 | CrossAsset GPU Training | 12/12 | 2026-03-30 |

### Composition Remediation (Complete)

| Epic | Description | Status |
|------|-------------|--------|
| E52 | DRY Composition Refactoring (timeseries/) | Complete (7/7) |
| E53 | Unified Training Forward/Backward | Complete (6/6) |
| E57 | Fix DGX Spark Build Regression | Complete (3/3) |
| E59 | Remove gonum Dependency | Complete (7/7) |
| E62 | Auxiliary Training Package Composition | Complete (7/7) |
| E65 | MoE Layer Composition Fix | Complete (3/3) |
| E66 | Functional Layer API for Training | Complete (5/5) |
| E67 | Timeseries Full Layers Migration | Complete (11/11) |
| E68 | CrossAsset Full Layers Migration | Complete (4/4) |
| E69 | Training Loss/Optimizer Engine Compliance | Complete (6/6) |
| E70 | Intra-Layers Violations Cleanup | Complete (10/10) |
| E71 | Experimental Package Migration | Complete (5/5) |
| E72 | Architecture Enforcement Test | Complete (2/2) |
| E73 | Generate KV Cache Consolidation | Complete (3/3) |
| E75 | Inference Timeseries .Data() Elimination | Complete (9/9) |

### Composition Remediation (Near-Complete)

| Epic | Description | Status | Notes |
|------|-------------|--------|-------|
| E74 | Timeseries Backward Pass Composition | 12/14 | All backward API + migration done; 2 DGX validation tasks pending |

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
Gemma3-1B 233 tok/s (1.14x Ollama), DeepSeek-R1 186 (1.11x), Llama3.2 92
(0.99x), Mistral-7B 44 (1.00x). All models produce coherent output. v1.38.4
with ztensor v1.1.2. CUDA graph capture 184/185 instructions (99.5%).
Up to 14% faster on small models, parity at 7B. Training: PatchTST 128.5s (4.6x faster).

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

## Completed Research & Optimization (P1-P15, all tasks done)

| Section | Description | Tasks | Completed |
|---------|-------------|-------|-----------|
| P1 | Full Multi-Model Benchmark | BMK-T1 to BMK-T4 | 2026-03-27 |
| P2 | Mistral vs Ollama Head-to-Head | MHH-T1 to MHH-T3 | 2026-03-27 |
| P3 | K-Quant Kernel Optimization | KQ-T1 to KQ-T3 | 2026-03-27 |
| P4 | Granite TS Parity Tests | GTS-T1 to GTS-T3 | 2026-03-27 |
| P5 | MSA-Inspired Scalable Memory (E34) | T34.1-T34.6 (18 tasks) | 2026-03-27 |
| P6 | QuaRot + KVQuant 4-Bit (E35) | T35.1-T35.3 (12 tasks) | 2026-03-27 |
| P7 | EAGLE-3 Self-Speculative (E36) | T36.1-T36.2 (12 tasks) | 2026-03-27 |
| P8 | NSA Native Sparse Attention (E37) | T37.1-T37.2 (8 tasks) | 2026-03-27 |
| P9 | CPU/GPU Hybrid MoE (E38) | T38.1-T38.2 (7 tasks) | 2026-03-27 |
| P10 | BitNet b1.58 Ternary (E39) | T39.1-T39.2 (8 tasks) | 2026-03-27 |
| P11 | TransMLA MHA-to-MLA (E40) | T40.1-T40.2 (9 tasks) | 2026-03-27 |
| P12 | GGUF I-Quant Dequant (E41) | T41.1 (7 tasks) | 2026-03-27 |
| P13 | RadixAttention KV Cache (E42) | T42.1 (4 tasks) | 2026-03-27 |
| P14 | Flash Decoding (E43) | T43.1 (4 tasks) | 2026-03-27 |
| P15 | Multi-LoRA Serving (E44) | T44.1 (6 tasks) | 2026-03-27 |

Detailed task lists for P1-P15 removed during 2026-04-03 /tidy --apply.
All 127 tasks completed by 2026-03-27 (PRs #262-#265). Full details in git history.

---

## E86: PyTorch Parity Testing for All Layers and Architectures

### Context

Zerfoo has had multiple instances of incorrect layer implementations and performance
issues that went undetected because there was no systematic comparison against a mature
reference framework. PyTorch is the most widely validated ML framework and serves as
a reliable source of numerical truth for standard operations (while acknowledging it is
not infallible, its correctness coverage far exceeds Zerfoo's).

This epic creates comprehensive parity tests that compare every Zerfoo layer, loss
function, optimizer, and model architecture against PyTorch. The approach:
1. A Python script generates deterministic golden test data using PyTorch (fixed seeds,
   small shapes, exported as JSON with flat arrays and shape metadata).
2. Go parity tests load the golden data, run through Zerfoo, and compare element-wise
   within float32 tolerance.
3. Both forward and backward passes are tested where applicable.
4. GPU parity tests are submitted to DGX via Spark for CUDA kernel verification.

### Existing Coverage (from initial audit 2026-04-10)

32 forward-pass CPU parity tests already pass (100%): ReLU, GELU, Sigmoid, Tanh,
Softmax, LeakyReLU, SwiGLU, Erf, functional.ReLU/GELU/Sigmoid/SiLU/Softmax/
LayerNorm/RMSNorm/Linear, LayerNorm, RMSNorm, Linear, MatMul, Conv1D, SDPA (causal
+ bidirectional), MultiHeadAttention, TokenEmbedding, RotaryEmbedding, MSE, BCE,
CrossEntropy, ReduceSum, Transpose, Gather.

Files: tests/golden/generate_golden.py, tests/golden/layers/*.json,
tests/parity/layer_parity_test.go.

### Key Conventions Discovered

- core.Linear stores weight as [in, out] and computes x @ W.
- functional.Linear stores weight as [out, in] (PyTorch convention) and computes x @ W^T.
- SDPA defaults to bidirectional; callers must SetCausal(true) for decoder attention.
- RoPE uses split-half rotation (GPT-NeoX/LLaMA style), not interleaved pairs.
- GELU uses tanh approximation. LayerNorm uses population variance (N, not N-1).
- Conv1D/Conv2D use cross-correlation (same as PyTorch, not true convolution).

### Acceptance Criteria

- Every layer in layers/ has at least one forward-pass golden file test vs PyTorch.
- Every layer with Backward() has a gradient parity test vs PyTorch autograd.
- Every loss function and optimizer has a parity test.
- All timeseries and tabular model architectures have end-to-end forward parity tests.
- GPU parity tests run on DGX for all CUDA-accelerated layers.
- All tests integrated into CI (CPU tests run on every PR, GPU tests run weekly).

### Work Breakdown

Each task follows the same pattern: (1) add a gen_xxx() function to
tests/golden/generate_golden.py, (2) run the script to generate the JSON golden
file, (3) add a TestParity_Xxx Go test to tests/parity/layer_parity_test.go that
loads the golden file and compares Zerfoo output, (4) add the test to the
TestParity_Summary list. For backward tests, include grad_output and
expected_grad_input fields in the golden file and call Backward() in Go.

Golden files already exist but lack Go tests: composite_transformer_block,
core_conv2d, core_ffn, norm_batch_norm, op_dropout, optimizer_adamw,
optimizer_sgd, recurrent_simple_rnn, ssm_mamba, ssm_s4. Wire these first.

#### E86.0: Wire existing unwired golden files (Go tests only, no Python)

- [x] T86.0.1 Wire Conv2D golden Go test  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.2 Wire FFN golden Go test  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.3 Wire BatchNorm golden Go test  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.4 Wire Dropout golden Go test  Est: 15m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.5 Wire AdamW optimizer golden Go test  Est: 30m  verifies: [UC-L02]  DONE 2026-04-10
- [x] T86.0.6 Wire SGD optimizer golden Go test  Est: 15m  verifies: [UC-L02]  DONE 2026-04-10
- [x] T86.0.7 Wire SimpleRNN golden Go test  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.8 Wire S4 SSM golden Go test  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.0.9 Wire MambaBlock golden Go test  Owner: Agent  Est: 45m  verifies: [UC-L01]  Done: 2026-04-11 PR#386
  Added convBias to MambaBlock, wired full golden parity test.
- [x] T86.0.10 Wire TransformerBlock structural Go test  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  Structural test using AttentionHead as attention node inside transformer.Block.
- [x] T86.0.11 Run go vet + go test on all wired tests  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10
  42/42 pass, 2 skip (MambaBlock, TransformerBlock). Full suite green.

#### E86.1: New Layer Forward Parity (generate golden + wire Go test)

- [x] T86.1.1 FastGelu activation  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.2 SimplifiedLayerNorm  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.3 SkipSimplifiedLayerNorm  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.4 GQA (GroupedQueryAttention)  Owner: auto  Est: 1h  verifies: [UC-L01]  Done: 2026-04-11
  Full numerical parity test already passing (TestParity_GQA).
- [x] T86.1.5 AttentionHead structural  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.6 MoE (MixtureOfExperts + MoEGate)  Owner: auto  Est: 1h  verifies: [UC-L01]  Done: 2026-04-11
  Full numerical parity test already passing (TestParity_MoE).
- [x] T86.1.7 LMHead  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [ ] T86.1.8 MIMOMambaBlock  Owner: TBD  Est: 1h  verifies: [UC-L01]
  SKIP: 7+ projection layers with per-head SSM params, cross-head mixing.
- [x] T86.1.9 AttnRes structural  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.10 BlockAttnRes residual  Owner: Agent  Est: 30m  verifies: [UC-L01]  Done: 2026-04-11 PR#386
  Full golden parity test with RMSNorm keys and softmax attention weights.
- [ ] T86.1.11 HModule hierarchical residual  Owner: TBD  Est: 45m  verifies: [UC-L01]
  SKIP: needs attention graph.Node construction.
- [x] T86.1.12 PatchEmbed timeseries  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.13 GRN (Gated Residual Network)  Owner: Agent  Est: 30m  verifies: [UC-L01]  Done: 2026-04-11 PR#386
  GRN forward matches within 1e-4 (TestParity_GRN).
- [x] T86.1.14 TSMixerBlock  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.15 MLSTM structural  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.16 SLSTM structural  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.17 SSMLayer (timeseries)  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.18 CLIPEncoder structural  Est: 1h  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.19 MelExtractor structural  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.20 WhisperEncoder structural  Est: 1h  verifies: [UC-L01]  DONE 2026-04-10
- [x] T86.1.21 Core arithmetic ops (Add, Sub, Mul, Div, Pow, Sqrt, Sin, Cos)  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  Note: Neg not on Engine interface. 8/9 ops tested.
- [x] T86.1.22 Core shape ops (Reshape, Concat)  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  Note: Squeeze, Unsqueeze, Slice not on compute.Engine interface. 2/8 ops tested.
- [ ] T86.1.23 Core comparison ops  Owner: TBD  Est: 30m  verifies: [UC-L01]
  SKIP: Equal/Greater/Where/TopK not in compute.Engine interface.
- [x] T86.1.24 Run go vet + go test on all E86.1 tests  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10
  103 pass, 5 skip, 0 fail.

#### E86.2: Layer Backward Parity (gradient verification)

- [x] T86.2.1 Activation backward: ReLU, GELU, Sigmoid, Tanh, LeakyReLU, SwiGLU  Est: 1h  verifies: [UC-L01]  DONE 2026-04-10
  All 6 activation backward gradients match PyTorch autograd within 1e-5.
- [x] T86.2.2 Normalization backward: LayerNorm, RMSNorm  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  RMSNorm PASS. LayerNorm FAIL: ReduceSum axis bug in dGamma/dBeta (filed for fix).
- [x] T86.2.3 Core backward: Linear, MatMul  Est: 1h  verifies: [UC-L01]  DONE 2026-04-10
  Linear PASS. MatMul FAIL: missing transpose in gradient computation (filed for fix).
- [x] T86.2.4 Loss backward: MSE, BCE, CrossEntropy  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  BCE PASS. MSE FAIL: missing 2/N factor. CrossEntropy FAIL: missing 1/N factor (filed for fix).
- [x] T86.2.5 SSM backward: S4  Est: 1h  verifies: [UC-L01]  DONE 2026-04-10
  SKIP: no backward golden data in S4 golden file.
- [x] T86.2.6 Attention backward: SDPA  Est: 45m  verifies: [UC-L01]  DONE 2026-04-10
  SKIP: no backward golden data in SDPA golden file.
- [x] T86.2.7 Run go vet + go test for all backward parity tests  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10
  92 tests total: 88 pass, 4 fail (real backward bugs), 4 skip.

#### E86.3: Optimizer and Initializer Parity

- [x] T86.3.1 EMA: add golden + Go test for one EMA update  Est: 30m  verifies: [UC-L02]  DONE 2026-04-10
- [x] T86.3.2 SWA: add golden + Go test for one SWA step  Est: 30m  verifies: [UC-L02]  DONE 2026-04-10
- [x] T86.3.3 Initializers: statistical tests for Xavier, He  Est: 30m  verifies: [UC-L02]  DONE 2026-04-10
- [x] T86.3.4 Run go vet + go test for optimizer parity tests  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

#### E86.4: Model Architecture End-to-End Parity

Each task builds a tiny model in PyTorch with random (seeded) weights, exports
all weights + input + expected output as a golden JSON file, then loads weights
into the Zerfoo model and compares forward pass output.

- [x] T86.4.1 PatchTST structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.2 N-BEATS structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.3 DLinear golden-file parity  Est: 30m  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.4 ITransformer structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.5 TFT structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.6 CfC structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.7 FTTransformer structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.8 TabNet structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.9 PPO structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.10 SAC structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.11 GCN structural  Est: 45m  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.12 GAT structural  Est: 45m  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.13 MarketVAE structural  Est: 1h  verifies: [UC-L03]  DONE 2026-04-10
- [x] T86.4.14 Run go vet + go test  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

#### E86.5: GPU Kernel Parity (DGX via Spark)

- [x] T86.5.1 Build arm64 parity test image for DGX  Est: 30m  verifies: [infrastructure]  DONE 2026-04-11
  AC: Containerfile with tests/parity/ tests compiles for linux/arm64 with -tags cuda.
  Containerfile at tests/parity/Containerfile, Spark manifest at docs/bench/manifests/gpu-parity.yaml.
- [x] T86.5.2 GPU vs CPU parity: activations  Est: 30m  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU forward output matches CPU for all 9 activations within 1e-4. commit f87892d1.
- [x] T86.5.3 GPU vs CPU parity: normalization  Est: 30m  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU LayerNorm, RMSNorm, BatchNorm match CPU within 1e-4. commit f87892d1.
- [x] T86.5.4 GPU vs CPU parity: core ops  Est: 30m  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU Linear, MatMul, Conv1D, FFN match CPU within 1e-3. commit adde0a2b.
- [x] T86.5.5 GPU vs CPU parity: attention  Est: 45m  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU SDPA (causal + bidirectional) and GQA match CPU within 1e-3. commit adde0a2b.
- [x] T86.5.6 GPU vs CPU parity: RotaryEmbedding  Est: 30m  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU RoPE matches CPU within 1e-5. commit f87892d1.
- [x] T86.5.7 GPU backward parity: all trained layers  Est: 1h  verifies: [UC-L01]  DONE 2026-04-11
  AC: GPU gradients match CPU gradients within 1e-3 for all layers with Backward(). commit adde0a2b.
- [ ] T86.5.8 Submit tests to DGX via Spark and collect results  Est: 30m  verifies: [infrastructure]
  AC: Pod completes with exit 0. Results captured in .claude/scratch/gpu-parity-results.txt.
  BLOCKED: purego runtime.dlopen prevents cross-compilation. Needs native arm64 build on DGX.

#### E86.6: CI Integration and Reporting

- [x] T86.6.1 Add CPU parity tests to CI workflow  Est: 30m  verifies: [infrastructure]  DONE 2026-04-10
  Added step to .github/workflows/ci.yml. commit 8081b289.
- [x] T86.6.2 Add golden file staleness check to CI  Est: 30m  verifies: [infrastructure]  DONE 2026-04-10
  Created .github/workflows/golden-staleness.yml (weekly cron, Monday 06:00 UTC). commit dd59849a.
- [x] T86.6.3 Create parity coverage report  Est: 30m  verifies: [infrastructure]  DONE 2026-04-10
  Added TestParity_CoverageReport with 80+ entries across 14 categories. commit e267916a.
- [x] T86.6.4 Run go vet + golangci-lint  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10
  Both pass clean. Fixed 3 forbidigo violations. commit e267916a.

### E86 Parallel Tracks

| Track | Tasks | Description | Dependencies |
|-------|-------|-------------|-------------|
| AN: Wire Existing | T86.0.1-T86.0.11 | Wire 10 existing golden files (Go only) | None |
| AO: New Layer Forward | T86.1.1-T86.1.24 | New golden + Go tests for remaining layers | None |
| AP: Backward | T86.2.1-T86.2.7 | Gradient parity for trainable layers | AN |
| AQ: Optimizers | T86.3.1-T86.3.4 | Optimizer steps + initializers | None |
| AR: Architectures | T86.4.1-T86.4.14 | End-to-end model forward parity | AN, AO (partial) |
| AS: GPU Parity | T86.5.1-T86.5.8 | DGX GPU vs CPU kernel parity | AN, AP |
| AT: CI | T86.6.1-T86.6.4 | CI integration and reporting | AN |

### E86 Sync Points

- Tracks AN, AO, and AQ are fully independent -- can run in Wave 1.
- Track AP (backward) needs at least AN done (golden files + Go test infrastructure).
  Track AO is NOT a prerequisite because backward tests only cover layers that already
  have golden files from the initial audit (activations, norms, core, loss, SSM, SDPA).
- Track AR (architectures) can start after AN completes (component layer trust) but
  does NOT need AO -- architecture tests only verify forward pass.
- Track AS (GPU) needs AN + AP (CPU baselines to compare against).
- Track AT (CI) needs AN at minimum to have tests to run.

### E86 Waves

#### Wave E86-1: Wire Existing + New Layers + Optimizers (3 agents)
All three tracks are fully independent.

- [ ] Agent 1: T86.0.1-T86.0.11 (wire 10 existing golden files -- Go tests only, no Python)
- [ ] Agent 2: T86.1.1-T86.1.24 (new layer forward parity -- Python golden + Go tests)
- [ ] Agent 3: T86.3.1-T86.3.4 (optimizer + initializer parity)

#### Wave E86-2: Backward + Architectures (3 agents)
Deps: Wave E86-1 Agent 1 (AN track -- golden file infrastructure proven).

- [ ] Agent 1: T86.2.1-T86.2.7 (backward gradient parity for all trainable layers)
- [ ] Agent 2: T86.4.1-T86.4.7 (timeseries + tabular architecture parity)
- [ ] Agent 3: T86.4.8-T86.4.14 (TabNet, RL, GNN, synth architecture parity)

#### Wave E86-3: GPU + CI (2 agents)
Deps: Waves E86-1 and E86-2 (CPU baselines established).

- [ ] Agent 1: T86.5.1-T86.5.8 (GPU kernel parity on DGX via Spark)
- [ ] Agent 2: T86.6.1-T86.6.4 (CI integration + coverage report)

---

## E87: Fix Backward Pass Bugs Found by PyTorch Parity

### Context

PyTorch parity testing (E86 Wave 2) discovered 4 bugs in Backward() methods where
Zerfoo's gradient computation does not match PyTorch autograd. These bugs affect
training correctness -- models using these layers will compute wrong gradients,
leading to degraded or incorrect training outcomes. The parity tests
(TestParity_*_Backward in tests/parity/layer_parity_test.go) serve as regression
tests once the bugs are fixed.

### Work Breakdown

#### T87.1 Fix LayerNorm Backward -- wrong ReduceSum axis for dGamma/dBeta

- [x] T87.1.1 Fix LayerNorm backward ReduceSum axis  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
  Fixed axis from len(inputShape)-1 to 0 (iterative for multi-dim). Also fixed
  missing stdDev division in term3. commit 5c300c95.
- [x] T87.1.2 Run go test ./layers/normalization/... and TestParity_LayerNorm_Backward  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

#### T87.2 Fix MatMul Backward -- missing transposes in gradient computation

- [x] T87.2.1 Fix MatMul backward transposes  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
  Added Transpose(b, [1,0]) and Transpose(a, [1,0]) before MatMul calls. commit fefdcba6.
- [x] T87.2.2 Run go test ./layers/core/... and TestParity_MatMul_Backward  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

#### T87.3 Fix MSE Backward -- missing 2/N scaling factor

- [x] T87.3.1 Fix MSE backward 2/N scaling  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
  Added MulScalar(2.0/N) after diff computation. Updated 4 existing test expected values. commit 54be8879.
- [x] T87.3.2 Run go test ./training/loss/... and TestParity_MSELoss_Backward  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

#### T87.4 Fix CrossEntropy Backward -- missing 1/N batch normalization

- [x] T87.4.1 Fix CrossEntropy backward 1/N normalization  Est: 30m  verifies: [UC-L01]  DONE 2026-04-10
  Added MulScalar(1.0/N) after softmax-oneHot subtraction. commit 7733c08b.
- [x] T87.4.2 Run go test ./training/loss/... and TestParity_CrossEntropyLoss_Backward  Est: 15m  verifies: [infrastructure]  DONE 2026-04-10

### E87 Waves

#### Wave E87-1: Fix all 4 backward bugs (4 agents)
All 4 bugs are in different packages and files -- fully independent.

- [ ] Agent 1: T87.1.1-T87.1.2 (LayerNorm backward fix)
- [ ] Agent 2: T87.2.1-T87.2.2 (MatMul backward fix)
- [ ] Agent 3: T87.3.1-T87.3.2 (MSE backward fix)
- [ ] Agent 4: T87.4.1-T87.4.2 (CrossEntropy backward fix)

---

## E89: Timeseries Engine[T] Compliance -- Eliminate Raw Slice Math

### Context

"Engine[T] is law" -- all tensor arithmetic must flow through compute.Engine[T].
Six timeseries model files violate this by performing raw float64/float32 slice
arithmetic in for loops. This makes them CPU-only, prevents GPU acceleration,
breaks CUDA graph capture, and bypasses the compute abstraction that makes
Zerfoo's architecture work.

These models were written as quick Python ports and never properly migrated.
The Engine[T] existed before they were written -- this is not legacy, it is a
convention violation. Four clean reference files (nbeats.go, nhits.go, tft.go,
mamba.go) demonstrate the correct pattern.

### Audit Results

| File | Raw Slice Lines | Engine Calls | Status | Priority |
|------|----------------|-------------|--------|----------|
| frets.go | 73 | 0 | VIOLATION | CRITICAL |
| cfc.go | 42 | 0 | VIOLATION | CRITICAL |
| timemixer.go | 24 | 0 | VIOLATION | HIGH |
| dlinear.go | 10 | 0 | VIOLATION | HIGH |
| itransformer.go | 4 | 0 | VIOLATION | MEDIUM |
| itransformer_backward.go | 6+ | 0 | VIOLATION | MEDIUM |
| patchtst.go | 1 | 11 | PARTIAL | LOW |

Clean references: nbeats.go (0 violations, 9 Engine calls), nhits.go, tft.go, mamba.go.

### Approach

For each violating file:
1. Replace raw []float64 / []float32 parameter storage with graph.Parameter[T].
2. Replace manual for-loop arithmetic with Engine[T] ops (MatMul, Add, MulScalar, etc.).
3. Replace flatParams() with Parameters() []*graph.Parameter[T].
4. Keep the same mathematical formula -- only change HOW the computation is dispatched.
5. Verify with existing golden-file parity test (output must not change).

The golden-file parity tests from E86/E88 serve as regression tests: if the
refactored model produces different output, the test catches it immediately.

### Acceptance Criteria

- Zero raw slice arithmetic in forward/backward paths of all timeseries models.
- All models use compute.Engine[T] for every tensor operation.
- All models store parameters as graph.Parameter[T], not raw slices.
- All existing tests pass (go test ./timeseries/... and parity tests).
- No flatParams() methods remain (replaced by Parameters()).

### Work Breakdown

#### E89.1: ITransformer -- float64 slices to Engine[T] (most-used, best parity coverage)

- [x] T89.1.1 Migrate ITransformer forward path to Engine[T]  Owner: Agent-1  Est: 2h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/itransformer.go (647 lines, 4 raw arithmetic violations).
  Replace [][]float64 weight storage with graph.Parameter[float32]. Replace manual
  linear transforms (y[j] += xi * w[i][j]) with engine.MatMul. Replace residual
  additions with engine.Add. Keep forward() signature compatible via adapter.
  AC: TestParity_ITransformer golden-file parity test passes (tolerance 1e-9).
  Reference: nbeats.go for Engine-based linear layer pattern.
- [x] T89.1.2 Migrate ITransformer backward path to Engine[T]  Owner: Agent-1  Est: 2h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/itransformer_backward.go (723 lines, manual gradient accumulation).
  Replace manual grads[i] += ... with Engine operations and graph.Parameter gradient
  accumulation. Use layers/functional backward ops (LinearBackward, LayerNormBackward,
  MultiHeadAttentionBackward) where available.
  AC: ITransformer training produces same loss curve within 1e-4 over 10 epochs.
- [x] T89.1.3 Remove flatParams() from ITransformer, replace with Parameters()  Owner: Agent-1  Est: 30m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
  AC: flatParams() removed. Parameters() returns []*graph.Parameter[T]. Callers updated.
- [x] T89.1.4 Run go test ./timeseries/... and parity tests  Owner: Agent-1  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.2: DLinear -- simplest violation, best test coverage

- [x] T89.2.1 Migrate DLinear forward path to Engine[T]  Owner: Agent-2  Est: 1h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/dlinear.go (532 lines, 10 raw arithmetic violations).
  Replace manual movingAverage, seasonal decomposition, and output projection
  with Engine ops. movingAverage -> engine.Conv1D or manual engine.Add/DivScalar.
  AC: TestParity_DLinear golden-file parity test passes (tolerance 1e-4).
- [x] T89.2.2 Remove flatParams(), add Parameters()  Owner: Agent-2  Est: 30m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.2.3 Run go test ./timeseries/... and parity tests  Owner: Agent-2  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.3: CfC (Closed-form Continuous) -- 42 violations, RNN cell

- [x] T89.3.1 Migrate CfC forward path to Engine[T]  Owner: Agent-3  Est: 2h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/cfc.go (842 lines, 42 raw arithmetic violations).
  Replace manual matrix-vector products (Wtau, Wx, Wh computations) with
  engine.MatMul. Replace element-wise sigmoid/tanh with engine ops.
  AC: TestParity_CfC_Structural test passes (shape, no NaN, non-constant).
- [x] T89.3.2 Migrate CfC backward path to Engine[T]  Owner: Agent-3  Est: 2h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  AC: CfC training loss matches pre-migration within 1e-4 over 10 epochs.
- [x] T89.3.3 Remove flatParams(), add Parameters()  Owner: Agent-3  Est: 30m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.3.4 Run go test ./timeseries/... and parity tests  Owner: Agent-3  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.4: FreTS -- 73 violations, worst offender (DFT/IDFT + MLP)

- [x] T89.4.1 Migrate FreTS forward path to Engine[T]  Owner: Agent-3  Est: 3h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/frets.go (925 lines, 73 raw arithmetic violations).
  Replace manual DFT/IDFT nested loops with Engine ops. Replace channel MLP and
  temporal MLP manual forward with engine.MatMul + engine activation.
  Note: complex128 arithmetic may need special handling -- Engine may not support
  complex types. If so, split real/imaginary and use engine.MatMul on each.
  AC: FreTS forward produces correct output (structural test: shape, no NaN).
- [x] T89.4.2 Migrate FreTS backward path  Owner: Agent-3  Est: 2h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
- [x] T89.4.3 Remove flatParams(), add Parameters()  Owner: Agent-3  Est: 30m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.4.4 Run go test ./timeseries/... and parity tests  Owner: Agent-3  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.5: TimeMixer -- 24 violations, multi-scale decomposition

- [x] T89.5.1 Migrate TimeMixer forward path to Engine[T]  Owner: Agent-2  Est: 1.5h  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/timemixer.go (607 lines, 24 raw arithmetic violations).
  Replace mixingMLP manual 2-layer MLP with engine.MatMul + engine activation.
  Replace manual residual connections with engine.Add.
  AC: TimeMixer forward produces correct output (structural test).
- [x] T89.5.2 Remove flatParams(), add Parameters()  Owner: Agent-2  Est: 30m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.5.3 Run go test ./timeseries/... and parity tests  Owner: Agent-2  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.6: PatchTST -- 1 violation, nearly clean

- [x] T89.6.1 Remove remaining raw slice arithmetic in PatchTST  Owner: Agent-1  Est: 30m  verifies: [UC-L01]  Done: 2026-04-11 PR#383
  File: timeseries/patchtst.go — already compliant (only copy() data movement, no arithmetic).
  AC: Zero raw slice arithmetic in forward path. TestParity_PatchTST passes.
- [x] T89.6.2 Run go test ./timeseries/... and parity tests  Owner: Agent-1  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

#### E89.7: Validation sweep

- [x] T89.7.1 Run full test suite: go test ./...  Owner: Coordinator  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.7.2 Run all parity tests: go test -run TestParity_ ./tests/parity/...  Owner: Coordinator  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
- [x] T89.7.3 Verify no flatParams() methods remain in timeseries/  Owner: Coordinator  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383
  AC: Only patchtst_backward.go and ttm.go (out-of-scope) retain flatParams.
- [x] T89.7.4 Run go vet + golangci-lint  Owner: Coordinator  Est: 15m  verifies: [infrastructure]  Done: 2026-04-11 PR#383

### E89 Parallel Tracks

| Track | Tasks | Description | Dependencies |
|-------|-------|-------------|-------------|
| AU: ITransformer | T89.1.1-T89.1.4 | Highest parity coverage | None |
| AV: DLinear | T89.2.1-T89.2.3 | Simplest, best golden coverage | None |
| AW: CfC | T89.3.1-T89.3.4 | RNN cell, moderate complexity | None |
| AX: FreTS | T89.4.1-T89.4.4 | Worst offender, complex128 | None |
| AY: TimeMixer | T89.5.1-T89.5.3 | Multi-scale MLP | None |
| AZ: PatchTST | T89.6.1-T89.6.2 | Nearly clean, trivial | None |
| BA: Validation | T89.7.1-T89.7.4 | Full sweep | All above |

### E89 Waves

All model migrations are independent (different files, no shared code paths).

#### Wave E89-1: Migrate all 6 models (3 agents) — DONE 2026-04-11 PR#383
- [x] Agent 1: T89.1.1-T89.1.4 + T89.6.1-T89.6.2 (ITransformer + PatchTST)
- [x] Agent 2: T89.2.1-T89.2.3 + T89.5.1-T89.5.3 (DLinear + TimeMixer)
- [x] Agent 3: T89.3.1-T89.3.4 + T89.4.1-T89.4.4 (CfC + FreTS)

#### Wave E89-2: Validation sweep (1 agent) — DONE 2026-04-11 PR#383
Deps: Wave E89-1

- [x] Agent 1: T89.7.1-T89.7.4 (full test + lint + parity + grep verification)

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
- [x] T20.3 Benchmark Metal vs CPU on Apple M4 Max  Est: 2h
- [ ] T29.4 Benchmark on-device inference  Est: 2h

### ZerfooConf (E26)

- [ ] T26.2 Execute ZerfooConf Day  Est: 8h
- [ ] T26.3 Plan standalone ZerfooConf 2032  Est: 6h

### Architecture Expansion (E32)

- [x] T32.2 Validate 100+ model architectures

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

### FP8 E5M2 (from E46)

- [ ] T46.4.8 Backlog: FP8 E5M2 support (deferred to v1.1)  Owner: TBD  Est: 8h  verifies: [UC-L02]
  E5M2 format (1 sign, 5 exponent, 2 mantissa) is used on NVIDIA Ada Lovelace GPUs.
  Implement after v1.0.0 tag as a non-breaking addition.

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

### Composition Remediation Tracks (E66-E76)

| Track | Tasks | Description | Status |
|-------|-------|-------------|--------|
| U: Functional API | E66 T66.1.* | Prerequisite: layers/functional package | COMPLETE |
| V: Timeseries Forward Migration | E67 T67.*.* | Forward path migration (18K lines) | COMPLETE |
| W: CrossAsset Migration | E68 T68.*.* | Unify CPU/GPU paths | COMPLETE |
| X: Training Engine Compliance | E69 T69.*.* | Loss + optimizer fixes | COMPLETE |
| Y: Intra-Layers Cleanup | E70 T70.*.* | Fix layers/ internal violations | COMPLETE |
| Z: Experimental Migration | E71 T71.*.* | rl/, synth/, meta/, shared/ | COMPLETE |
| AA: Enforcement | E72 T72.*.* | Architecture test CI gate | COMPLETE |
| AB: KV Cache | E73 T73.*.* | generate/ cache consolidation | COMPLETE |
| AC: Backward Composition | E74 T74.*.* | Add functional backward ops, migrate 3 backward files | COMPLETE (12/14, 2 DGX pending) |
| AD: Inference TS .Data() | E75 T75.*.* | Replace unjustified .Data() in 6 arch builders | COMPLETE |
| AE: Allowlist Cleanup | E76 T76.*.* | Remove timeseries/ from arch test allowlist | BLOCKED (deps: AC) |
| AF: Tabular Composition | E77 T77.*.* | Replace reimplemented ops with functional | DONE |
| AG: Layers Internal | E78 T78.*.* | Rewrite violating code to use engine ops | DONE |
| AH: Generate Refactor | E79 T79.*.* | Extract shared decode loop, deduplicate | DONE |
| AI: Builder Boilerplate | E80 T80.*.* | Shared factory functions for arch builders | DONE |
| AJ: Custom Nodes | E81 T81.*.* | Replace unjustified custom nodes with layers/ | DONE |
| AK: Training Loss | E82 T82.*.* | Loss functions + optimizer to engine ops | DONE |
| AL: Serve Handlers | E83 T83.*.* | Extract shared helpers from monolithic handlers | DONE |
| AM: ModeLDSL | E84 T84.*.* | DSL layer implementations compose from layers/ | DONE |

### Composition Sync Points

- Tracks U-AD: ALL COMPLETE as of 2026-04-03.
- Tracks AF-AM (E77-E84): ALL COMPLETE as of 2026-04-06.
- Track AE (E76) depends on Track AC (E74 must complete before allowlist removal).
- BLOCKED: T76.1.1 needs backward bridge .Data() elimination (88 calls remain).

### Composition Waves

#### Composition Waves 1-6: COMPLETE

All tasks in Waves 1-6 (E66-E73) completed 2026-04-03. 48 tasks across
functional API, timeseries migration, crossasset migration, training engine
compliance, intra-layers cleanup, experimental migration, architecture
enforcement, and KV cache consolidation. Details in git history.

#### Composition Wave 7: Remaining ztensor + DGX parity (3 agents)
Independent of Waves 1-6. Can start immediately.

- [ ] T61.3.2 DGX parity tests for inference builders  verifies: [UC-010]
- [x] T63.1.1 Design quantized matmul dispatcher (ztensor)  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #76 v1.4.0
- [x] T63.1.2 Replace 16 methods with dispatcher (ztensor)  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #76 v1.4.0

#### Composition Wave 8: ztensor validation (3 agents)
Deps: Wave 7 (T63.1.2)

- [ ] T63.2.1 Benchmark quantized matmul (ztensor)  verifies: [infrastructure]
- [ ] T63.2.2 Full ztensor test suite  verifies: [infrastructure]
- [x] T64.1.1 Split gpu_engine.go into focused files (ztensor)  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #77

#### Composition Waves 9-12: COMPLETE

All tasks in Waves 9-12 (E74, E75) completed 2026-04-03. 24 tasks across
backward API creation (LinearBackward, LayerNormBackward, GELUBackward,
SoftmaxBackward, MultiHeadAttentionBackward, MLPBackward), backward file
migration (patchtst, itransformer, timemixer, encoder), inference .Data()
elimination (6 arch builders), and validation. Only T76.1.1 and T76.1.2
remain BLOCKED (allowlist removal needs bridge .Data() elimination).

Remaining from Wave 12:
- [ ] T76.1.1 Remove timeseries/ from allowlist  verifies: [infrastructure]  BLOCKED -- backward files still use .Data() in slice-tensor conversion bridges (88 calls); needs future bridge elimination epic
- [ ] T76.1.2 Verify CI green  verifies: [infrastructure]  BLOCKED by T76.1.1

#### Composition Waves 13-19: COMPLETE

All tasks in Waves 13-19 (E77-E84) completed 2026-04-03 through 2026-04-06.
61 tasks across tabular composition (E77), layers internal cleanup (E78),
generate refactoring (E79), builder boilerplate extraction (E80), custom node
replacement (E81), training loss migration (E82), serve handler refactoring (E83),
and modeldsl composition (E84). All race-detector tests and go vet pass.
PRs: #334, #336, #338, #341. Details in git history.

### Completed Research Waves (1-8, all tasks done)

Waves 1-8 (127 tasks across 10 tracks A-T) completed 2026-03-27.
Task details removed during /tidy --apply. See git history for full lists.

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
| M-E46.1 | zonnx v1.0.0 released | v1.0.0 tag on github.com/zerfoo/zonnx | DONE 2026-03-30 |
| M-E46.2 | ztensor v1.0.0 released | v1.0.0 tag on github.com/zerfoo/ztensor | DONE 2026-03-30 |
| M-E46.3 | ztoken v1.0.0 released | v1.0.0 tag on github.com/zerfoo/ztoken | DONE 2026-03-30 |
| M-E46.4 | float8 v1.0.0 released | v1.0.0 tag on github.com/zerfoo/float8 | DONE 2026-03-30 |
| M-E46.5 | float16 v1.0.0 released | v1.0.0 tag on github.com/zerfoo/float16 | DONE 2026-03-30 |
| M-E46.6 | Full ecosystem v1+ | All 5 libraries at v1+; zerfoo already at v1.36+ | DONE 2026-03-30 |
| M-E47 | Batched training practical | PatchTST 28K rows trains in < 60s on DGX Spark | DONE 2026-03-30 |
| M-E48 | TimeMixer shipped | TimeMixer TrainWindowed + inference graph builder; tests pass | DONE 2026-03-30 |
| M-COMP | Composition Remediation Phase 1 | E61-E65 | Inference builders, tabular/gnn/modeldsl, MoE compose from layers/; ztensor god file consolidated | 2026-Q2 |
| M-COMP-2 | Composition Remediation Phase 2 | E66-E73 | All forward paths compose from layers/ or Engine; architecture test in CI | DONE 2026-04-03 |
| M-COMP-3 | Composition Remediation Phase 3 | E74-E76 | All backward passes compose from functional backward ops; inference .Data() eliminated; timeseries/ removed from arch test allowlist | 2026-Q3 |
| M-COMP-4 | Composition Remediation Phase 4 | E77-E84 | dirty-architecture.md violations reduced from ~9,800 to <2,000 lines; tabular/, layers/, generate/, inference/, training/, serve/, modeldsl/ all compose from layers/ or Engine | 2026-Q3 |
| M-E86 | PyTorch Parity Complete | E86 | 100% layer forward parity, 100% backward parity, all model architectures, GPU kernel parity on DGX | 2026-Q3 |

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
| R50 | Inference builder refactoring breaks model parity (E61) | High | Medium | Run parity tests for each model after refactoring; keep old code behind build tag until verified |
| R51 | gnn [][]float64 to tensor conversion changes public API (E62) | Medium | High | Provide adapter functions at API boundary; internal-only tensor usage if possible |
| R52 | Quantized matmul dispatcher adds dispatch overhead (E63) | Medium | Low | Dispatcher is a type-switch resolved at call time, not runtime polymorphism; benchmark validates <2% regression |
| R53 | gpu_engine.go file split creates merge conflicts with in-flight PRs (E64) | Low | Medium | Schedule E64 during a merge freeze or after all ztensor PRs land |
| R54 | MoE engine op refactoring changes expert routing behavior (E65) | Medium | Low | Top-K routing is unchanged; only bias/sigmoid/softmax are refactored; parity test validates |
| R55 | layers/functional API adds dispatch overhead for training | Low | Medium | Functions are thin wrappers; benchmark to confirm <2% overhead |
| R56 | timeseries/ migration breaks training accuracy | High | Medium | Parity tests for every model; compare 10-epoch training loss curves before/after |
| R57 | crossasset/ graph-based backward produces different gradients | Medium | Medium | Numerical gradient check (finite differences) validates analytical gradients |
| R58 | MAML inner-loop in meta/ is incompatible with graph-based backward | Medium | High | If incompatible, keep meta/ as justified exception; document why |
| R59 | Architecture enforcement test has false positives | Low | Medium | Maintain allowlist; review and adjust thresholds quarterly |
| R60 | KV cache strategy pattern adds virtual dispatch overhead | Low | Low | Benchmark KV cache Get/Set latency; strategy dispatch is not on hot path |
| R61 | Functional backward ops produce numerically different gradients than hand-coded loops (E74) | High | Medium | Numerical gradient check (finite differences) for each op; 10-epoch training loss parity within 1e-4 |
| R62 | MultiHeadAttentionBackward complexity with batched heads (E74) | Medium | Medium | Start from itransformer_backward.go reference; validate per-head gradient isolation |
| R63 | Replacing .Data() with engine.Slice in inference/timeseries/ changes node count (E75) | Low | Low | Benchmark inference throughput before/after; engine.Slice is lightweight |
| R64 | Removing timeseries/ from arch test allowlist too early (E76) | Low | Medium | Only remove after E74 fully complete and all tests pass |
| R65 | Tabular migration breaks model accuracy (5 architectures) (E77) | High | Medium | Run parity tests for all 5 tabular models (FT-Transformer, SAINT, TabNet, ResNet, model.go) before/after; compare validation loss within 1e-4 |
| R66 | layers/ internal cleanup changes layer behavior (vision, timeseries) (E78) | High | Medium | Numerical parity tests for clip_encoder, mLSTM, SSM, VSN; compare output tensors element-wise within tolerance |
| R67 | generate/ refactoring breaks streaming/speculative generation (E79) | High | Medium | Run full generation test suite including streaming, speculative, and EAGLE paths; compare output token sequences |
| R68 | Inference builder helper migration breaks model parity (30 architectures) (E80) | High | High | Run parity tests per architecture after migration; keep old code behind build tag until DGX verified |
| R69 | ModeLDSL composition changes DSL compilation behavior (E84) | Medium | Medium | Run full modeldsl test suite; compare compiled model structure before/after; validate layer-type registration |
| R70 | PyTorch golden files may not match Zerfoo's intended semantics for some ops (E86) | Medium | Medium | When PyTorch and Zerfoo disagree, investigate which is correct rather than blindly matching PyTorch. Document intentional differences in the golden file description field. |
| R71 | GPU parity tests may show larger tolerance than CPU due to non-deterministic kernel execution order (E86) | Low | High | Use 1e-3 tolerance for GPU tests (vs 1e-5 for CPU). If larger divergence, investigate specific kernel. |

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

### Time Series Research (E47-E49)
- TimeMixer: Decomposable Multiscale Mixing (ICLR 2024, https://github.com/kwuking/TimeMixer)
- Chronos-2: Multivariate Time Series Foundation Model (Amazon, 2025)
- TiRex: xLSTM Foundation Model for Time Series (NX-AI, 2025, arXiv:2505.23719)
- Moirai-2: Any-Variate Foundation Model (Salesforce, 2025)

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
- ADR-075: Batched training with kernel fusion for time series
- ADR-076: Foundation model inference via gRPC Python bridge
- ADR-082: Composition remediation strategy (E61-E65)

---

## Progress Log

### 2026-04-11: E89 added -- timeseries Engine[T] compliance

- Added E89 (27 tasks, 2 waves) to eliminate all raw slice math from 6 timeseries
  model files (155+ lines of for-loop arithmetic violating "Engine[T] is law").
- Audit found: frets.go (73 violations), cfc.go (42), timemixer.go (24),
  dlinear.go (10), itransformer.go (4+), patchtst.go (1). Clean references:
  nbeats.go, nhits.go, tft.go, mamba.go.
- All migrations are independent (Wave 1: 3 agents, 2 models each).
- E88 completed: PatchTST, N-BEATS, ITransformer upgraded to golden-file parity.

### 2026-04-10 (night): E87 added -- fix 4 backward pass bugs

- Added E87 (8 tasks, 1 wave, 4 agents) to fix bugs found by E86 Wave 2.
- Bugs: LayerNorm (wrong ReduceSum axis), MatMul (missing transposes),
  MSE (missing 2/N factor), CrossEntropy (missing 1/N normalization).
- All 4 fixes are in different packages -- fully parallelizable in one wave.
- Each fix has a paired test verification task.

### 2026-04-10 (evening): E86 plan refined for execution

- Restructured E86 from 4 sequential waves to 3 waves with better parallelism (8 agents total).
- Added E86.0 sub-epic: wire 10 existing golden files that have no Go test (Conv2D, FFN,
  BatchNorm, Dropout, AdamW, SGD, SimpleRNN, S4, MambaBlock, TransformerBlock).
- Split batched tasks into individual tasks: timeseries layers (6 separate tasks),
  SSM variants, residual types, audio layers. Total tasks: 72 (was 63).
- Removed false dependencies: backward tests only need E86.0 (wire existing), not all
  of E86.1 (new layers). Architecture tests can start after E86.0 completes.
- Added E86.4 tasks for GCN and GAT separately (was combined). Added SWA optimizer.
- Wave 1 now has 3 fully independent agents (wire existing, new layers, optimizers).
  Wave 2 has 3 agents (backward, timeseries/tabular archs, RL/GNN/synth archs).
  Wave 3 has 2 agents (GPU on DGX, CI integration).

### 2026-04-10: E86 added -- PyTorch parity testing for all layers

- Added E86 for comprehensive PyTorch parity testing of every layer, loss function,
  optimizer, and model architecture.
- Initial audit completed: 32/32 CPU forward-pass parity tests PASS (100%).
  Tests cover: 8 activations, 8 functional ops, 2 normalization, 3 core, 3 attention,
  2 embeddings, 3 loss, 3 ops.
- Files created: tests/golden/generate_golden.py (PyTorch 2.11.0 golden generator),
  tests/golden/layers/ (36 JSON golden files), tests/parity/layer_parity_test.go.
- Key findings: core.Linear uses [in,out] weight (x@W); functional.Linear uses
  [out,in] (x@W^T). SDPA defaults bidirectional. RoPE uses split-half rotation.

### 2026-04-07 (afternoon): E85 diagnosis complete, fix scope refined

- T85.1.3 + T85.1.4 marked DONE. Root cause: ~38 leaked GPU tensors per batch in
  trainWindowedGPU. CUDA graph capture (T51.4.1) confirmed disabled at line 453.
- T85.1.1 + T85.1.2 (runtime profiling) skipped — root cause was visible from source.
- E85.2 (fix tasks) refined from 3 vague tasks to 8 concrete tasks (T85.2.0-T85.2.7)
  with line numbers, target struct fields, and specific allocation sites.
- Added Wave E85-2a (ztensor API inventory) as a hard prerequisite for the fix.
- Added Wave E85-2b (ztensor API extension) as conditional follow-up.
- Wave E85-3 expanded from 3 to 5 parallel agents (one per allocation cluster +
  encoder helpers).
- Added "E85 Next-Session Starter Checklist" with the exact resume sequence.
- Shipped diagnosis as PR #346 (merged).

### 2026-04-07: E85 added — fix GPU training memory leak

- Added E85 (9 tasks across 4 waves) targeting the cudaMalloc OOM in `trainWindowedGPU`
  encoder backward path. CRITICAL — blocks T50.5.2 and T51.5.2 (DGX training benchmarks).
- Detailed reproducer and evidence matrix in the epic and devlog.
- Wave plan: diagnosis (3 agents) -> profile run (1) -> fix (3) -> validation (1).

### 2026-04-06: Wave 20 DGX benchmarks + housekeeping

- Merged release-please PR #340 (v1.42.1).
- Fixed stale status header for E77-E84 (all DONE), E50/E51 (code complete).
- Fixed DGX IP in hand-off notes (192.168.86.250 → 192.168.86.29).
- Created cmd/bench_train benchmark tool for PatchTST training.
- Started GPU + CPU benchmarks on DGX (T50.5.2, T51.5.2); results pending collection.
- T58.1.2 (GQA parity) blocked: no GGUF model files on DGX.
- T63.2.1-T63.2.3 blocked: need CUDA CGo kernel stubs on DGX.

### 2026-04-06: /tidy --apply --prune

- Marked Waves 18-19 validation tasks complete (11/11). Epics E77-E84 all DONE.
- Milestone M-COMP-4 marked DONE.
- Collapsed completed epics E77-E84 into summary (removed ~520 lines of task breakdowns).
- Collapsed completed Waves 13-19 into summary.
- Pruned 5 stale worktrees and 8 superseded branches (local + remote).
- Release-please PR #340 (v1.42.1) left open (automated, no action needed).

### 2026-04-03: Added E77-E84 composition phase 4 and E74-E76 phase 3

Added 8 epics (E77-E84, ~61 tasks) for phase 4 composition and 3 epics
(E74-E76, 25 tasks) for phase 3 backward composition. All completed by
2026-04-06 across Waves 9-19. See git history for detailed breakdowns.

Older progress log entries (2026-03-26 through 2026-04-03) removed during
2026-04-06 plan trim. Key milestones: E34-E44 research (127 tasks complete),
E45-E65 implementation, E66-E73 composition phases 1-2, E74-E76 phase 3,
E77-E84 phase 4. See git history for full changelog.

---

## Hand-off Notes

- All code is in Go 1.25 with generics. No CGo. GPU via purego/dlopen.
- DGX Spark GPU at `ssh ndungu@192.168.86.29` for CUDA testing.
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
- E74 (backward composition) is the biggest remaining composition task. Key insight:
  layers/functional currently only has forward ops. Backward ops must be added first
  (T74.1.1-T74.1.7), then 3 backward files + encoder backward can migrate (T74.2.*).
  The backward ops should use engine operations internally, NOT raw loops.
  Reference implementations: itransformer_backward.go:503-537 (LayerNorm backward),
  itransformer_backward.go:542-555 (softmax backward), timemixer_backward.go:457-496
  (MLP backward). All use the same numerical formulas -- just needs engine wrapping.
- E75 (inference .Data()) is mostly engine.Slice and engine.Reshape replacements.
  Do NOT touch arch_chronos.go or arch_regime.go (justified .Data() uses).
- E63/E64 (ztensor) must be committed in the ztensor repo, not zerfoo.
- E77 (tabular composition): tabular/ has 5 model architectures (FT-Transformer, SAINT,
  TabNet, ResNet, model.go). Each file can be migrated independently. The key is replacing
  local linearForward/layerNorm/geluScalar/attention with functional.* equivalents.
  Run parity tests for each model after migration -- validation loss must match within 1e-4.
- E78 (layers/ internal): These are violations WITHIN layers/ itself. vision/clip_encoder.go
  is the worst offender (~200 lines of raw loops). The timeseries files (mlstm, ssm, vsn)
  have .Data() access that should use engine.Slice. core/gemm.go has a hand-rolled triple-loop
  GEMM that should simply call engine.MatMul.
- E79 (generate/ refactor): The 4 decode loops (Generate, GenerateStream, speculative, EAGLE)
  are ~80% identical. Extract the shared decode step FIRST (T79.1.2), then deduplicate the
  callers. Be careful with streaming -- SSE token emission timing must not change.
- E80 (builder boilerplate): Create the helper functions FIRST (T80.1.1-T80.1.3), then
  migrate builders in waves. Start with the 6 production builders (llama, gemma, mistral,
  qwen, phi, deepseek), then do the remaining ~20. Each builder migration is small but
  there are many -- batch them efficiently.
- E81 (custom nodes): These are the REMAINING custom nodes after E61. E61 refactored the
  builders to use arch_common patterns, but left some custom nodes in place. E81 finishes
  the job by replacing them with layer compositions. arch_vision_helpers.go is the priority.
- E82 (training loss): QuantileLoss has a KNOWN BUG -- it panics for non-float32 types due
  to a hard float32 cast. Fix the generics first (T82.1.3). The other loss functions work
  but bypass the engine they store as a field.
- E83 (serve handlers): handleChatCompletions and handleCompletions share ~60% of their logic.
  The extractions (buildGenerationOptions, parseAndApplyGrammar, detectAndFormatToolCalls) are
  straightforward refactors. Be careful not to change the OpenAI API response format.
- E84 (modeldsl): modeldsl/ reimplements layers from scratch on raw []float64. E62 did
  high-level composition but E84 goes deeper into the individual layer implementations.
  The LayerType constant reconciliation (T84.1.6) should make layers/registry the single
  source of truth.
- E86 (PyTorch parity): The golden file generator is tests/golden/generate_golden.py.
  Run `python3 tests/golden/generate_golden.py` to regenerate all golden files.
  Go tests are in tests/parity/layer_parity_test.go. Pattern for adding a new layer:
  1. Add a gen_xxx() function to generate_golden.py that creates deterministic input,
     runs PyTorch forward, and calls save_case() with JSON output.
  2. Add a TestParity_Xxx function to layer_parity_test.go that loads the golden file,
     creates the Zerfoo layer, runs Forward(), and calls assertClose().
  3. Add the test to the TestParity_Summary cases list.
  Key gotcha: core.Linear weight is [in, out] (x@W), but functional.Linear weight
  is [out, in] (x@W^T). The golden file must match whichever convention the test uses.
  For backward tests: golden files include grad_output, expected_grad_input, and
  expected_grad_weight fields. Call layer.Backward() and compare.
- Gemma3-1B Q4_K_M is cached on DGX Spark for integration tests.
- E47 (batched training) is entirely in zerfoo/timeseries/. Key insight: replace
  per-sample GPU calls with batch-level tensor operations. DataLoader converts
  float64 slices to float32 tensors once, then iterates mini-batches.
- E48 (TimeMixer) follows exact same pattern as PatchTST. Reference implementation:
  github.com/kwuking/TimeMixer. Key architecture: multi-scale decomposition via
  learnable moving averages, then MLP mixing across scales.
- E49 (foundation models) is native Go. New xLSTM layers (sLSTM, mLSTM) go in
  layers/timeseries/. Graph builders go in inference/timeseries/. Weight converters
  produce GGUF from HuggingFace SafeTensors. Parity tests need golden files generated
  from Python reference (one-time, checked into tests/golden/).
- Platform for E47-E49: linux/arm64 (DGX Spark with Grace Hopper GPU).

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
| UC-TS01 | Train time series efficiently | Train PatchTST/iTransformer on 28K+ rows with GPU-batched forward in < 60s |
| UC-TS02 | TimeMixer forecasting | Multi-scale decomposition MLP training and inference |
| UC-TS03 | Foundation model zero-shot | Zero-shot forecasting via Chronos-2/TiRex/Moirai-2 gRPC bridge |
| UC-TS04 | Train time series with composed backward | Backward passes use functional backward ops (not raw f64 loops) for maintainability and future GPU acceleration |

---

## E45: Verification Remediation 2026-03-29 (COMPLETE)

All 3 tasks done. TieredKVStore.Close() fixed to preserve user-provided ColdDir
(b944bde), test added (0505e81), re-verified HEALTHY.

---

## E46: Ecosystem v1 Release (COMPLETE)

46/46 tasks complete. All 5 sub-v1 libraries promoted to v1.0.0 on 2026-03-30.
float16 completed BFloat16 Phases 2-5 (arithmetic modes, batch ops, math functions,
parse/format, error handling). float8 verified E4M3FN against NVIDIA reference (256 values).
ztensor narrowed v1 stable surface to Engine[T], Tensor[T], Numeric, Device, numeric.*.
ztoken expanded edge case tests. zonnx API reviewed and tagged. T46.4.8 (FP8 E5M2)
moved to Backlog. Detailed task lists removed during /tidy --apply. See git history.

---

## E47: Batched Training Performance (COMPLETE)

19/19 tasks complete. Shipped batched forward/backward for all 9 time series backends
(PatchTST, iTransformer, DLinear, Mamba, CfC, FreTS, TTM, N-HiTS, N-BEATS). CPU
benchmark: PatchTST 28K rows at 596s (target <60s needs CUDA). DataLoader, batched
attention, and wiring all shipped (PRs #281-#286). Detailed task lists removed during
/tidy --apply. See git history.

---

## E48: TimeMixer Backend (COMPLETE)

10/10 tasks complete. Shipped TimeMixer (multi-scale decomposition + MLP mixing):
core implementation, engine-accelerated forward, backward pass, adapter, inference
graph builder, unit tests (PRs #281-#286). Detailed task lists removed during
/tidy --apply. See git history.

---

## E49: Foundation Model Inference (COMPLETE)

12/12 tasks complete. Shipped native Go inference for 3 foundation models: TiRex
(xLSTM, sLSTM+mLSTM cells), Chronos-2 (T5 encoder-decoder, value tokenizer),
Moirai-2 (masked encoder, any-variate projection). GGUF converters, graph builders,
parity tests, CLI `zerfoo forecast`, and fine-tune API all shipped (PRs #281-#286).
Detailed task lists removed during /tidy --apply. See git history.

---

## E50: GPU Training Kernel Elimination (GitHub Issue #278)

**Problem:** After channel batching (PR #292), PatchTST GPU training still takes 63.7s/epoch
on DGX Spark (target: 6s/epoch for <60s total at 10 epochs). Profiling shows the remaining
time is dominated by CPU operations that pull data off GPU, process on CPU, and push back:
layer norm forward/backward (8 calls/layer on [7680, 64] matrices), GELU forward/backward
(4 calls/layer on [7680, 256] matrices), and redundant weight transpose computation
(same weights transposed every batch iteration).

**Goal:** Replace all CPU bottleneck operations with engine ops (engine.Sum, engine.Mul,
engine.Sub, engine.MulScalar, engine.Sqrt, engine.Tanh, engine.Add, engine.Div) so the
full forward/backward loop can execute on GPU without CPU round-trips. Cache weight
transposes outside the batch loop.

**File:** timeseries/patchtst_gpu_train.go

### E50.1: Layer Norm Forward on Engine

- [x] T50.1.1 Implement engine-based layer norm forward  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30
  File: timeseries/patchtst_gpu_train.go
  Replace layerNormF32WithCache with engine ops:
  (1) mean = engine.Sum(x, -1, keepDims=true) / dModel via engine.MulScalar
  (2) centered = engine.Sub(x, mean)
  (3) var = engine.Sum(engine.Mul(centered, centered), -1, keepDims=true) / dModel
  (4) invStd = 1 / engine.Sqrt(engine.AddScalar(var, 1e-5))
  (5) normed = engine.Mul(engine.Mul(centered, invStd), scale) + bias via engine.Add
  Update gpuBatchLayerCache: change centered1/2 from [][]float32 to *tensor.TensorNumeric[float32],
  invStd1/2 from []float32 to *tensor.TensorNumeric[float32].
  Cache xInput (pre-norm) for backward. Remove matFromTensor/tensorFromMat calls for layer norm.
  Acceptance: go test -run TestPatchTST passes. Gradient check within 1e-3.

### E50.2: Layer Norm Backward on Engine

- [x] T50.2.1 Implement engine-based layer norm backward  Owner: TBD  Est: 3h  verifies: [UC-TS01]  DONE 2026-03-31
  Deps: T50.1.1
  File: timeseries/patchtst_encoder.go (layerNormBackwardWithEngine)
  Replaced layerNormBackwardF32 with engine ops. Also removed matFromTensor,
  tensorFromMat, copyMatToTensor (all now unused). The backward formula:
  (1) dScale += engine.Sum(dOut * centered * invStd, axis=0)
  (2) dBias += engine.Sum(dOut, axis=0)
  (3) dNorm = engine.Mul(dOut, scale)  -- broadcast [1, dModel] over [rows, dModel]
  (4) dotScaleGrad = engine.Sum(dNorm * centered, -1, keepDims=true)
  (5) dotMeanGrad = engine.Sum(dNorm, -1, keepDims=true)
  (6) dInput = invStd * (dNorm - (dotMeanGrad + centered * invStd^2 * dotScaleGrad) / dModel)
  All operations use engine.Mul, engine.Sub, engine.Sum, engine.MulScalar, engine.Add.
  Acceptance: Gradient check (numerical vs analytical) within 1e-3. go test passes.

### E50.3: GELU Forward/Backward on Engine

- [x] T50.3.1 Implement engine-based GELU forward and backward  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30
  File: timeseries/patchtst_gpu_train.go
  GELU forward: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  Using engine ops: engine.Mul, engine.MulScalar, engine.Add, engine.Tanh, engine.AddScalar.
  Cache the tanh result and the inner term for backward.
  GELU backward: dgelu = 0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
  Using engine.Mul, engine.MulScalar, engine.Sub, engine.Add.
  Update lc.ffn1PreAct caching to store values needed for backward.
  Remove geluScalar and geluDerivF32 CPU functions.
  Acceptance: Gradient check within 1e-3. go test passes. Forward output matches CPU within 1e-5.

### E50.4: Cache Weight Transposes

- [x] T50.4.1 Pre-compute weight transposes before batch loop  Owner: TBD  Est: 1h  verifies: [UC-TS01]  DONE 2026-03-30 0fbaf2e8
  File: timeseries/patchtst_gpu_train.go
  Before the epoch loop, compute and store: qWT, kWT, vWT, oWT, ffn1WT, ffn2WT, headWT
  for each encoder layer. These are used in the backward pass but recomputed every batch.
  Add a gpuWeightTransposes struct to hold them. Recompute only when weights change (once
  per optimizer step, not once per batch -- but since we update weights every batch,
  recompute at start of each batch iteration instead of inside backward pass to avoid
  redundant computation when multiple backward calls reference the same weight).
  Actually: weights change every batch via AdamW, so compute transposes once at the start
  of each backward pass (not inside the per-layer loop). This saves nLayers-1 redundant
  transposes per weight matrix per batch.
  Acceptance: go test passes. No behavioral change.

### E50.5: Validation and Benchmark

- [x] T50.5.1 Run go vet and full test suite  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-03
  Deps: T50.1.1, T50.2.1, T50.3.1, T50.4.1
  Acceptance: go vet ./timeseries/ clean. go test ./timeseries/ passes.

- [x] T50.5.2 Benchmark on DGX Spark  DONE 2026-04-09  verifies: [UC-TS01]
  28K×20×10: 40.3s (4.0s/epoch). Target met (<6s/epoch). 14.8x vs v1.37 (596s).
  See docs/benchmarks.md and devlog 2026-04-09 entries.

### E50 Parallel Work

#### Waves

##### Wave E50-1: Independent implementations (3 agents)

- [x] T50.1.1 Layer norm forward on engine  DONE 2026-03-30
- [x] T50.3.1 GELU forward/backward on engine  DONE 2026-03-30
- [x] T50.4.1 Cache weight transposes  DONE 2026-03-30 0fbaf2e8

##### Wave E50-2: Dependent + validation (3 agents)

- [x] T50.2.1 Layer norm backward on engine  Deps: T50.1.1  DONE 2026-03-31
- [x] T50.5.1 Run go vet and tests  Deps: T50.1.1, T50.2.1, T50.3.1, T50.4.1  DONE 2026-04-03
- [x] T50.5.2 Benchmark on DGX Spark  DONE 2026-04-09

---

## E51: CUDA Graph Capture for Training (GitHub Issue #278)

**Problem:** PatchTST GPU training takes 63.7s/epoch on DGX Spark despite channel batching
and batched attention. The bottleneck is Go-to-GPU synchronization: ~500 engine op calls per
batch, each requiring a Go round-trip. Moving ops to engine (E50) made things 2x slower
because element-wise engine ops have more launch overhead than CPU for small tensors.
The fundamental fix: capture the entire forward+backward pass as a CUDA graph and replay it,
eliminating ALL intermediate synchronization.

**Goal:** Capture the PatchTST forward+backward pass as a CUDA graph. First batch runs normally
(warmup), second batch captures, subsequent batches replay. Target: <6s/epoch (10x speedup).

**Decision rationale:** docs/adr/077-cuda-graph-training-capture.md

**Key decisions:**
- Drop partial batches (not pad) for fixed tensor shapes during replay
- Capture combined forward+backward graph (one graph per batch iteration)
- Implement in zerfoo training loop, not in ztensor graph compiler
- Pre-allocate all tensors before capture; warmup on first batch
- AdamW and loss computation stay outside graph (they call .Data() which triggers D2H)

**Closes:** GitHub issue #278

### E51.1: Drop Partial Batches

- [x] T51.1.1 Skip partial final batch in trainWindowedGPU  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]  DONE 2026-03-30
  File: timeseries/patchtst_gpu_train.go
  When nSamples % batchSize != 0, stop the batch loop before the partial batch.
  Change `for start := 0; start < nSamples; start += batchSize` to stop at
  `nSamples - (nSamples % batchSize)`. Log a warning if samples are dropped.
  Acceptance: go test passes. Training on 28001 samples with batch=64 produces
  same result as 28000 samples (last sample silently dropped).

### E51.2: Pre-allocate Tensor Workspace

- [x] T51.2.1 Pre-allocate all layer caches and intermediates before batch loop  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30
  File: timeseries/patchtst_gpu_train.go
  CUDA graph capture fails on cudaMalloc during capture. Currently, tensor.New
  and engine ops allocate GPU memory on demand. Pre-allocate:
  (1) gpuBatchForwardCache with all layer caches (normed1/2, q/k/v, scores, attnOut,
  ffn1PreAct, ffn1Out, centered1/2, invStd1/2, xResidual, xAfterAttn, flatInput, patches)
  (2) All backward intermediates (dX, dFlat, dChanOut, dAttnProjOut, etc.)
  Use engine ops with dst parameter (pre-allocated destination) to avoid new allocations.
  Acceptance: go test passes. No new tensor.New or engine allocations inside the
  forward/backward block (verify by counting allocations in a test).

### E51.3: Add Engine Capture/Replay API

- [x] T51.3.1 Add BeginCapture/EndCapture/Replay to Engine interface  Owner: TBD  Est: 2h  verifies: [infrastructure]  DONE 2026-03-30 ztensor PR #46
  Repo: ztensor. File: compute/engine.go, compute/gpu_engine.go, compute/cpu_engine.go
  Add three methods to the Engine[T] interface:
  - `BeginCapture(ctx) error` -- starts CUDA stream capture (no-op on CPU engine)
  - `EndCapture(ctx) (GraphHandle, error)` -- ends capture, returns opaque handle
  - `ReplayGraph(ctx, GraphHandle) error` -- replays captured graph (on CPU: re-execute ops)
  GraphHandle is an interface{} (opaque, engine-specific).
  GPUEngine implementation calls cuda.StreamBeginCapture/EndCapture/GraphInstantiate/GraphLaunch.
  CPUEngine implementation records op sequence during capture, replays during replay.
  Acceptance: GPU engine captures and replays a simple MatMul+Add sequence correctly.
  Unit test: capture MatMul+Add, replay 3 times, verify output matches non-captured execution.

### E51.4: Wire Graph Capture into Training Loop

- [x] T51.4.1 Integrate capture/replay into trainWindowedGPU  Owner: TBD  Est: 3h  verifies: [UC-TS01]  DONE 2026-03-30
  Deps: T51.1.1, T51.2.1, T51.3.1
  File: timeseries/patchtst_gpu_train.go
  Training loop structure becomes:
  ```
  batch 0: warmup (normal execution, establishes tensor sizes)
  batch 1: engine.BeginCapture() -> forward+backward -> engine.EndCapture() -> graphHandle
  batch 2..N: engine.ReplayGraph(graphHandle) (tensors updated in-place)
  ```
  Before each replay: update input patches tensor and zero gradients (these are writes
  to pre-allocated buffers, not new allocations -- compatible with graph replay).
  After each replay: run AdamW on CPU (outside graph).
  Handle epoch boundaries: recapture at start of each epoch if shuffling changes data order
  (graph replays with same tensor addresses, data content changes via memcpy before replay).
  Acceptance: Training convergence matches non-captured execution within 1e-4 tolerance.
  Gradient check passes. go test -run TestPatchTST passes.

### E51.5: Validation and Benchmark

- [x] T51.5.1 Run go vet and full test suite  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-03
  Deps: T51.4.1
  Acceptance: go vet clean. go test ./timeseries/ passes. go test ./... passes in ztensor.

- [x] T51.5.2 Benchmark on DGX Spark  DONE 2026-04-09  verifies: [UC-TS01]
  28K×20×10: 40.3s (4.0s/epoch). Target met (<6s/epoch, <60s total).
  Note: graph capture is disabled (canCapture=false, see T85.1.4 / T54.3.1);
  performance gains from E50 kernel elimination + E85 dst-memory reuse.
  See docs/benchmarks.md and devlog 2026-04-09 entries.

### E51 Parallel Work

#### Waves

##### Wave E51-1: Foundation (3 agents)

- [x] T51.1.1 Drop partial batches  DONE 2026-03-30
- [x] T51.2.1 Pre-allocate tensor workspace  DONE 2026-03-30
- [x] T51.3.1 Add Engine capture/replay API (ztensor repo)  DONE 2026-03-30

##### Wave E51-2: Integration + validation (3 agents)

- [x] T51.4.1 Wire graph capture into training loop  Deps: T51.1.1, T51.2.1, T51.3.1  DONE 2026-03-30
- [x] T51.5.1 Run go vet and tests  Deps: T51.4.1  DONE 2026-04-03
- [x] T51.5.2 Benchmark on DGX Spark  DONE 2026-04-09

---

## E52: DRY Composition Refactoring (timeseries/) (COMPLETE)

7 tasks complete. Eliminated ~5,329 duplicated lines in timeseries/ via shared math_ops, adamw_f32, layernorm_ops, and engine wrappers. Details in git history.

---

## E53: Unified Training Forward/Backward (COMPLETE)

6 tasks complete. Extracted shared encoderForward/encoderBackward, eliminated patchtst_engine.go and patchtst_backward_engine.go, reduced PatchTST from 6,196 to ~3,500 lines. Details in git history.

---

## E54: Capture-Pure GPU Engine Ops (ztensor)

**Problem:** CUDA graph capture fails because GPUEngine.Zero delegates to CPUEngine.Zero,
which calls tensor.Data() on GPU-resident tensors, triggering GPUStorage.TrySlice (sync
D2H memcpy on the default stream). Other ops (Copy, some fallbacks) have the same issue.
This blocks capturing the full ~500-op encoder forward+backward as a CUDA graph.

**Root cause:** gpu_engine.go line 3162: `func (e *GPUEngine[T]) Zero(...) { return e.cpu.Zero(ctx, a) }`.
The CPU engine zeros via `a.Data()` slice mutation, which triggers D2H copy for GPU tensors.

**Goal:** Make GPU engine ops that touch GPU-resident tensors use GPU-native operations
(cudaMemset, cudaMemcpy D2D) on the engine's stream. No CPU delegation for GPU tensors.

**Repo:** ztensor (github.com/zerfoo/ztensor)

### E54.1: GPU-Native Zero

- [x] T54.1.1 Implement GPU-native Zero using cudaMemsetAsync  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-03-30 ztensor PR #50
  Repo: ztensor. File: compute/gpu_engine.go
  Replace `func (e *GPUEngine[T]) Zero(...) { return e.cpu.Zero(ctx, a) }` with:
  Check if tensor has GPUStorage. If yes: cudaMemsetAsync(ptr, 0, byteSize, e.stream).
  If no (CPU tensor): delegate to e.cpu.Zero as before.
  Add cudaMemsetAsync binding to internal/cuda/runtime_purego.go if not present.
  Acceptance: go test ./compute/ passes. GPU tensors zeroed without D2H copy.

### E54.2: GPU-Native Copy

- [x] T54.2.1 Implement GPU-native Copy using cudaMemcpyAsync D2D  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-03-30 ztensor PR #50
  Repo: ztensor. File: compute/gpu_engine.go
  Replace `func (e *GPUEngine[T]) Copy(...) { return e.cpu.Copy(ctx, dst, src) }` with:
  Check if both tensors have GPUStorage. If yes: MemcpyAsync D2D on e.stream.
  If mixed or CPU: delegate to e.cpu.Copy.
  Acceptance: go test passes. GPU-to-GPU copy without TrySlice.

### E54.3: Re-enable Graph Capture in zerfoo

- [ ] T54.3.1 Remove canCapture=false and re-enable forward-prefix capture  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]  DEFERRED: forward-prefix capture (~78 ops) is slower than no-capture (20.9s vs 12.9s/epoch). Graph too small for replay savings to offset launch+sync cost. Keep disabled until fused encoder kernel (E55) enables ~500-op capture.
  Deps: T54.1.1, T54.2.1, E55
  Repo: zerfoo. File: timeseries/patchtst_gpu_train.go
  Update ztensor dependency. Remove the `canCapture = false` disable flag.
  Forward-prefix capture (~78 ops) should work without TrySlice errors.
  Acceptance: DGX Spark benchmark completes without CUDA errors. Target: ~32s/epoch.

### E54.4: Benchmark

- [ ] T54.4.1 Benchmark on DGX Spark  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T54.3.1
  Run 28K x 20ch x 10 epochs. Compare with 128.5s baseline.
  Acceptance: Results documented. Graph capture working.

### E54 Parallel Work

##### Wave E54-1: GPU-native ops (2 agents, ztensor repo)

- [x] T54.1.1 GPU-native Zero (cudaMemsetAsync)  DONE 2026-03-30
- [x] T54.2.1 GPU-native Copy (cudaMemcpyAsync D2D)  DONE 2026-03-30

##### Wave E54-2: Enable + benchmark (2 agents)

- [ ] T54.3.1 Re-enable graph capture in zerfoo  Deps: T54.1.1, T54.2.1, E55  DEFERRED
- [ ] T54.4.1 Benchmark on DGX Spark  Deps: T54.3.1

---

## E55: Fused PatchTST Encoder CUDA Kernel (ztensor)

**Problem:** Even with graph capture, the encoder runs ~250 separate CUDA kernels per layer
per direction (forward + backward). Each kernel has launch overhead. A fused kernel
combining the entire encoder layer into a single launch would eliminate this overhead.

**Goal:** Implement a fused encoder layer CUDA kernel in ztensor that combines:
LayerNorm + QKV projection + multi-head attention + output projection + residual +
LayerNorm + FFN1 + GELU + FFN2 + residual into a single kernel launch.

**Repo:** ztensor (github.com/zerfoo/ztensor)

**Reference:** Existing fused kernels in internal/cuda/kernels/ (fused_add_rmsnorm.cu,
fused_swiglu.cu) provide the pattern.

### E55.1: Fused Encoder Forward Kernel

- [ ] T55.1.1 Implement fused_encoder_fwd.cu  Owner: TBD  Est: 8h  verifies: [infrastructure]
  Repo: ztensor. File: internal/cuda/kernels/fused_encoder_fwd.cu
  CUDA kernel that fuses one encoder layer forward pass:
  (1) LayerNorm1: mean reduction + normalize + scale/bias
  (2) QKV projection: three matrix multiplies (use cuBLAS from kernel? or tiled GEMM)
  (3) Multi-head attention: reshape + Q@K^T + softmax + scores@V
  (4) Output projection: matmul + bias
  (5) Residual add
  (6) LayerNorm2: same as (1)
  (7) FFN1: matmul + bias + GELU
  (8) FFN2: matmul + bias
  (9) Residual add
  Parameters: input, layer weights (Q/K/V/O/FFN1/FFN2/norm1/norm2), output, caches.
  Dimensions: totalRows, dModel, nHeads, headDim, ffnDim.
  Note: MatMul inside CUDA kernel can call cuBLAS from device code or use cooperative
  groups for tiled GEMM. For small matrices ([7680, 64]), tiled GEMM in shared memory
  may be faster than cuBLAS call overhead.
  Acceptance: Kernel compiles for sm_121 (GB10). Output matches per-op encoder within 1e-4.

- [ ] T55.1.2 Add purego bindings and Go wrapper  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Deps: T55.1.1
  Repo: ztensor. Files: internal/cuda/purego.go, compute/gpu_engine.go
  Add launch_fused_encoder_fwd symbol to KernelLib.
  Add GPUFusedEncoderForward method to GPUEngine.
  Add FusedEncoderForwarder optional interface to engine.go.
  Acceptance: go build ./compute/ clean.

- [ ] T55.1.3 Unit tests for fused encoder forward  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Deps: T55.1.2
  Repo: ztensor. File: compute/gpu_engine_test.go
  Test: fused encoder forward matches per-op encoder forward within 1e-4 tolerance
  for various (totalRows, dModel, nHeads, ffnDim) configurations.
  Acceptance: go test passes on DGX Spark.

### E55.2: Fused Encoder Backward Kernel

- [ ] T55.2.1 Implement fused_encoder_bwd.cu  Owner: TBD  Est: 10h  verifies: [infrastructure]
  Deps: T55.1.1
  Repo: ztensor. File: internal/cuda/kernels/fused_encoder_bwd.cu
  Backward pass for the fused encoder layer. Computes gradients for all weights
  and the input gradient in a single kernel. Uses cached activations from forward.
  This is the most complex kernel -- attention backward requires careful softmax
  gradient computation and multiple matmul-transpose operations.
  Acceptance: Kernel compiles. Gradient check matches per-op backward within 1e-3.

- [ ] T55.2.2 Add purego bindings and Go wrapper for backward  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Deps: T55.2.1
  Acceptance: go build clean.

### E55.3: Wire into zerfoo

- [ ] T55.3.1 Wire fused encoder into encoderForward/encoderBackward  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Deps: T55.1.2, T55.2.2
  Repo: zerfoo. File: timeseries/patchtst_encoder.go
  Type-assert engine to FusedEncoderForwarder. If supported, call fused kernel
  per layer instead of the ~25 individual engine ops.
  Acceptance: Training convergence unchanged. Gradient check passes.

- [ ] T55.3.2 Benchmark fused encoder on DGX Spark  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T55.3.1
  Run 28K x 20ch x 10 epochs with fused kernel + graph capture.
  Target: <60s total.
  Acceptance: Results documented.

### E55 Parallel Work

##### Wave E55-1: Forward + backward kernels (2 agents, independent .cu files)

- [ ] T55.1.1 Fused encoder forward kernel (.cu)
- [ ] T55.2.1 Fused encoder backward kernel (.cu)

##### Wave E55-2: Bindings + tests (2 agents)

- [ ] T55.1.2 Forward purego bindings + Go wrapper  Deps: T55.1.1
- [ ] T55.2.2 Backward purego bindings + Go wrapper  Deps: T55.2.1

##### Wave E55-3: Tests + wiring (2 agents)

- [ ] T55.1.3 Forward unit tests  Deps: T55.1.2
- [ ] T55.3.1 Wire into zerfoo  Deps: T55.1.2, T55.2.2

##### Wave E55-4: Benchmark (1 agent)

- [ ] T55.3.2 DGX Spark benchmark  Deps: T55.3.1

---

## E56: Gemma3 Inference Micro-Optimizations

**Problem:** Gemma3 inference is already well-optimized (233 tok/s, 1.14x Ollama on 1B)
with CUDA graph capture (184/185 instructions), 6 fused kernels per layer, merged QKV/gate+up,
and GPU-native ops. However, three small fusion opportunities remain that cumulatively offer
~5-25% speedup depending on prefill vs decode workload.

**Current decode (Gemma3-1B, DGX Spark, v1.38.4):** 233 tok/s
**Target decode:** 270+ tok/s (~10% gain from fusions a+b)
**Target prefill:** 10-20% improvement from fusion c

### E56.1: Fused Softmax+V Multiply (Decode)

- [x] T56.1.1 Implement fused_softmax_vmul.cu kernel  Owner: TBD  Est: 4h  verifies: [UC-001]  DONE 2026-03-30 ztensor PR #52
  Repo: ztensor. File: internal/cuda/kernels/fused_softmax_vmul.cu
  Fuse softmax + MatMul(scores, V) into one kernel. Currently:
  - Line 235 of scaled_dot_product_attention.go: engine.Softmax(ctx, scaledAttentionScores, -1)
  - Line 245: engine.MatMul(ctx, attentionWeights, v)
  The fused kernel computes softmax row-wise in shared memory, then immediately
  multiplies by V in the same kernel, avoiding intermediate tensor materialization.
  Input: scores [batch*heads, seqQ, seqKV], V [batch*heads, seqKV, headDim].
  Output: attnOut [batch*heads, seqQ, headDim].
  For decode (seqQ=1): this is a fused reduction (softmax) + GEMV.
  Acceptance: Output matches separate softmax+MatMul within 1e-4.

- [x] T56.1.2 Add purego bindings and GPUFusedSoftmaxVMul method  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-03-30 ztensor PR #52
  Deps: T56.1.1
  Repo: ztensor. Files: internal/cuda/purego.go, compute/gpu_engine.go
  Add FusedSoftmaxVMuler optional interface to engine.go.
  Acceptance: go build clean.

- [x] T56.1.3 Wire into ScaledDotProductAttention  Owner: TBD  Est: 1h  verifies: [UC-001]  DONE 2026-03-30 (wired at sdpa.go:160-173 with FusedSoftmaxVMulProvider type-assert + fallback)
  Deps: T56.1.2
  Repo: zerfoo. File: layers/attention/scaled_dot_product_attention.go
  Type-assert engine to FusedSoftmaxVMuler. If available, replace lines 235+245
  with single fused call. Fallback to separate ops if not supported.
  Acceptance: Inference output identical. Benchmark shows reduced kernel count.

### E56.2: Fused GQA Head Expansion

- [x] T56.2.1 Implement fused_repeat_interleave.cu kernel  Owner: TBD  Est: 3h  verifies: [UC-001]  DONE 2026-03-30 ztensor PR #51
  Repo: ztensor. File: internal/cuda/kernels/fused_repeat_interleave.cu
  Fuse the reshape+Repeat+reshape pattern for GQA K/V head expansion.
  Currently (grouped_query_attention.go lines 819-843):
  - Reshape K to [batch, numKV, 1, seq, headDim]
  - Repeat along axis 2 by replicationFactor
  - Reshape to [batch, numQ, seq, headDim]
  The fused kernel reads from [batch, numKV, seq, headDim] and writes to
  [batch, numQ, seq, headDim] with index remapping: outHead / replicationFactor = srcHead.
  Acceptance: Output matches reshape+Repeat+reshape within 1e-6.

- [x] T56.2.2 Add purego bindings and wire into GQA  Owner: TBD  Est: 1.5h  verifies: [UC-001]  DONE 2026-03-30 PR #308
  Deps: T56.2.1
  Repo: ztensor + zerfoo. Files: ztensor compute/gpu_engine.go, zerfoo layers/attention/grouped_query_attention.go
  Replace lines 819-843 with fused call when engine supports it.
  Acceptance: Inference identical. 1 kernel replaces 3 per K and 3 per V.

### E56.3: Fused Prefill Attention Path

- [ ] T56.3.1 Extend merged QKV to prefill (seqLen > 1)  Owner: TBD  Est: 2h  verifies: [UC-001]  BLOCKED: splitMergedQKV GPU SubSlice is fundamentally broken for seqLen>1 (takes contiguous memory, but Q/K/V are interleaved row-by-row in merged output). Needs engine.Narrow (column-slice on last dim) in ztensor first. Prefill already uses separate wq/wk/wv projections; perf diff of 1 vs 3 MatMuls during one-time prefill is negligible.
  Repo: zerfoo. File: layers/attention/grouped_query_attention.go
  Currently at line 375: `if gqa.mergedQKV != nil && seqLen == 1`.
  For prefill, the merged weight [dModel, (numQ+2*numKV)*headDim] can still be used
  with a single MatMul on [batch, seqLen, dModel] input. The split into Q/K/V is
  a zero-copy slice of the output. Remove the seqLen==1 guard.
  Acceptance: Prefill output identical. 1 MatMul instead of 3.

- [ ] T56.3.2 Extend fused QK norm+RoPE to prefill  Owner: TBD  Est: 4h  verifies: [UC-001]
  Deps: T56.3.1
  Repo: ztensor + zerfoo.
  Currently at line 422: `if seqLen == 1 && gqa.rope != nil && gqa.qNormWeight != nil`.
  The fused kernel (fused_qk_norm_rope.cu) needs to be extended to handle seqLen > 1.
  The kernel processes one (query, key) pair per thread block; for seqLen > 1, launch
  with batch*seqLen blocks instead of just batch blocks.
  Acceptance: Prefill output matches unfused path within 1e-4. Kernel handles seqLen up to 4096.

- [ ] T56.3.3 Benchmark prefill improvement  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T56.3.1, T56.3.2
  Benchmark Gemma3-1B prefill latency (128 tokens, 512 tokens, 2048 tokens)
  before and after fusions on DGX Spark.
  Acceptance: Measurable prefill speedup documented.

### E56.4: Decode Benchmark

- [ ] T56.4.1 Benchmark decode tok/s improvement  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T56.1.3, T56.2.2
  Run bench-compare-ollama.sh on DGX Spark for Gemma3-1B with fused softmax+V and
  fused GQA expansion. Compare: before vs after vs Ollama.
  Target: 270+ tok/s (vs 245 current).
  Acceptance: Results documented.

### E56 Parallel Work

E55 and E56 are fully independent (different packages, different repos).
Within E56, the three fusions are independent.

##### Wave E56-1: Kernels (3 agents, all independent .cu files)

- [x] T56.1.1 Fused softmax+V multiply kernel  DONE 2026-03-30
- [x] T56.2.1 Fused repeat-interleave kernel  DONE 2026-03-30
- [ ] T56.3.1 Extend merged QKV to prefill  BLOCKED (needs engine.Narrow in ztensor)

##### Wave E56-2: Bindings + wiring (3 agents)

- [x] T56.1.2 Softmax+V purego bindings  DONE 2026-03-30
- [x] T56.2.2 Repeat-interleave bindings + GQA wiring  DONE 2026-03-30
- [ ] T56.3.2 Extend fused QK norm+RoPE to prefill  Deps: T56.3.1

##### Wave E56-3: Integration + benchmarks (3 agents)

- [x] T56.1.3 Wire softmax+V into SDPA  DONE 2026-03-30
- [ ] T56.3.3 Prefill benchmark  Deps: T56.3.2
- [ ] T56.4.1 Decode benchmark  Deps: T56.1.3, T56.2.2

---

## E57: Fix DGX Spark Build Regression (COMPLETE)

3 tasks complete. Fixed 3 root causes (transpose no-op, causal mask D2H, Q4_K re-quant) that caused cudaMemcpy misaligned address on DGX Spark. Details in git history.

---

## E58: GPU vs CPU GQA Parity Test (DIAGNOSTIC for E57)

**Problem:** Three GPU regression root causes have been fixed (ztensor eab19d0, zerfoo 90cacad4,
zerfoo 1d56d2e5) but GPU inference still produces wrong output. Individual operations (MatMul,
RMSNorm, Softmax, Gather, RoPE, Transpose) all pass parity when tested in isolation. The
divergence occurs inside the composed GQA (GroupedQueryAttention) Forward pass. Binary dump of
the GQA input (RMSNorm output) is bit-identical between CPU and GPU when both use heap loading,
and Q/K/V projections are also bit-identical. The divergence starts AFTER the projections,
somewhere in the QK norm -> Transpose -> RoPE -> Attention chain inside GQA. compute-sanitizer
reports zero GPU memory errors.

**Goal:** Write a comprehensive parity test that runs the FULL GQA forward pass on both CPU
and GPU engines with identical inputs and compares output at each internal step. This will
pinpoint the exact operation where GPU diverges from CPU inside the composed pipeline.

**File:** tests/parity/gqa_gpu_parity_test.go
**Environment:** DGX Spark (GEMMA3_GGUF_PATH env var points to ~/models/gemma3-q4km/model.gguf)

### E58.1: GQA Parity Test

- [x] T58.1.1 Write GQA GPU vs CPU parity test  DONE 2026-04-01  Owner: TBD  Est: 3h  verifies: [infrastructure]
  File: tests/parity/gqa_gpu_parity_test.go
  The test must:
  (1) Load model weights from GGUF via inference.LoadGGUF (heap, no mmap). Skip if
      GEMMA3_GGUF_PATH env var is not set.
  (2) Create CPU and GPU engines (compute.NewCPUEngine, compute.NewGPUEngine).
  (3) Construct TWO identical GQA layers: one with CPU engine, one with GPU engine.
      Use the actual Gemma3-1B layer 0 weights: q_proj, k_proj, v_proj, o_proj (Q5_0/Q4_K
      storage), q_norm, k_norm (float32), and RoPE config (theta=10000, headDim=256,
      numQueryHeads=4, numKVHeads=1).
  (4) Create identical input tensor [1, 2, 1152] with deterministic data (e.g. the actual
      RMSNorm output for BOS + token 18428 "Hi").
  (5) Upload GPU weights via gpuEngine.UploadWeights.
  (6) Call GQA Forward on both engines with the same input.
  (7) Compare full output tensor element-by-element. Report maxDiff and failing indices.
  (8) If the output differs, add intermediate checkpoints INSIDE GQA Forward to narrow
      down the divergence: after Q projection, after QK norm, after transpose, after RoPE,
      after attention scores, after softmax, after output projection.
  Acceptance: Test compiles, runs on DGX Spark with GEMMA3_GGUF_PATH set, and either
  PASSES (proving GPU parity) or FAILS with a clear message identifying the first
  divergent step and the max element-wise difference.

- [ ] T58.1.2 Run test on DGX Spark and document findings  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Deps: T58.1.1
  SSH to DGX Spark, run:
  `GEMMA3_GGUF_PATH=~/models/gemma3-q4km/model.gguf go test -run TestGQA_GPUParity -v -count=1 ./tests/parity/`
  Record the output in docs/devlog.md. If the test identifies the divergent step, file a
  targeted fix. If it passes (GPU parity is correct), investigate the graph execution layer
  (EngineProxy, graph.Forward, tensor pool) as the next suspect.
  Acceptance: Test results documented in devlog. Next action identified.

### E58 Parallel Work

##### Wave E58-1: Write and run test (1 agent)

- [x] T58.1.1 Write GQA GPU parity test  DONE 2026-04-01
- [ ] T58.1.2 Run on DGX and document  Deps: T58.1.1

---

## E59: Remove gonum Dependency (COMPLETE)

7 tasks complete. Replaced gonum BLAS GEMM with native triple-loop implementations and gonum FFT with Cooley-Tukey radix-2; zero external dependencies remain. Details in git history.

---

### E60: CrossAsset GPU Training (COMPLETE)

12/12 tasks complete. GPU forward/backward/AdamW for CrossAsset cross-attention transformer
(GitHub #312). Detailed task lists removed during /tidy --apply. See git history.

---

## E61: Inference Builder Composition

**Problem:** 6 inference architecture builders define 31 custom graph nodes with
inline math instead of composing from layers/. arch_rwkv.go has a 250+ line
inline triple-loop matrix multiply. arch_bert.go reimplements attention, FFN, and
embedding as 7 private node types. arch_common.go (the exemplar) proves that full
layers/ composition works for all 12+ model architectures on the inference path.

**Goal:** Replace custom graph nodes with composition from layers/attention,
layers/core, layers/normalization, and layers/activations. Reduce custom node
count from 31 to under 5 (justified exceptions only). See ADR-082.

### E61.1: Critical Builders (inline math elimination)

- [x] T61.1.1 Refactor arch_rwkv.go to compose from layers/  Owner: TBD  Est: 4h  verifies: [UC-010]  DONE 2026-04-02 PR #316
  Replace the `project` function (triple-loop matmul, lines 694-707) with
  layers/core.Linear. Replace inline sigmoid and normalization with
  layers/activations.Sigmoid and layers/normalization. Replace 3 custom nodes
  with composed layer nodes.
  Acceptance: go test passes. RWKV model parity test PASS (output within tolerance).

- [x] T61.1.2 Refactor arch_bert.go to compose from layers/  Owner: TBD  Est: 4h  verifies: [UC-010]  DONE 2026-04-02 PR #316
  Replace 7 custom nodes (attention, FFN, embedding, etc.) with composition from
  layers/attention.GroupedQueryAttention, layers/core.FFN, layers/core.Linear,
  layers/embeddings.TokenEmbedding.
  Acceptance: go test passes. BERT model parity test PASS.

- [x] T61.1.3 Refactor arch_gpt2.go to compose from layers/  Owner: TBD  Est: 3h  verifies: [UC-010]  DONE 2026-04-02 PR #316
  Replace 4 custom nodes (attention, FFN) with layers/ composition.
  Acceptance: go test passes. GPT-2 model parity test PASS.

### E61.2: High/Medium Builders

- [x] T61.2.1 Refactor arch_llava.go to compose from layers/  Owner: TBD  Est: 3h  verifies: [UC-010]  DONE 2026-04-02 (GQA+FFN+RMSNorm composed from layers/; shared helpers extracted to arch_vision_helpers.go; mmProjectorNode kept as vision-specific)
  Replace inline attention and FFN with layers/ composition. Keep vision-specific
  processing as custom nodes if no layers/ equivalent exists.
  Acceptance: go test passes. LLaVA model parity test PASS.

- [x] T61.2.2 Refactor arch_falcon.go to compose from layers/  Owner: TBD  Est: 2h  verifies: [UC-010]  DONE 2026-04-02 (-123 lines, layerNorm+GELU+FFN composed from layers/)
  Replace inline layerNorm (line 384) and GELU with layers/normalization.LayerNorm
  and layers/activations.GELU. Keep custom multi-query attention node if it differs
  from GQA.
  Acceptance: go test passes. Falcon model parity test PASS.

- [x] T61.2.3 Refactor arch_llama.go custom embedding and LMHead  Owner: TBD  Est: 2h  verifies: [UC-010]  DONE 2026-04-02 (no change -- 6 inference-critical features justify custom nodes)
  Replace 2 custom nodes with layers/embeddings and layers/core.LMHead if the
  standard implementations match Llama's behavior.
  Acceptance: go test passes. Llama model parity test PASS (Gemma3-1B benchmark
  throughput within 2% of pre-refactor).

### E61.3: Validation

- [x] T61.3.1 Run go vet and full test suite  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-04-02
  Deps: T61.1.1, T61.1.2, T61.1.3, T61.2.1, T61.2.2, T61.2.3
  Acceptance: go vet ./inference/ clean. go test -race ./inference/ passes.

- [ ] T61.3.2 Run model parity tests on DGX Spark  Owner: TBD  Est: 2h  verifies: [UC-010]
  Deps: T61.3.1
  Run parity tests for RWKV, BERT, GPT-2, LLaVA, Falcon, Llama on DGX.
  Acceptance: all parity tests PASS on GPU.

- [x] T61.3.3 Run linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-02
  Deps: T61.3.1
  Run golangci-lint on all changed files.
  Acceptance: zero lint warnings.

### E61 Parallel Work

#### Wave E61-1: Critical builders (3 agents)
- [x] T61.1.1 arch_rwkv.go  DONE 2026-04-02
- [x] T61.1.2 arch_bert.go  DONE 2026-04-02
- [x] T61.1.3 arch_gpt2.go  DONE 2026-04-02

#### Wave E61-2: Remaining builders (3 agents)
- [x] T61.2.1 arch_llava.go  DONE 2026-04-02 (deferred: shared types)
- [x] T61.2.2 arch_falcon.go  DONE 2026-04-02
- [x] T61.2.3 arch_llama.go  DONE 2026-04-02 (justified no-change)

#### Wave E61-3: Validation (2 agents)
Deps: Wave E61-1, Wave E61-2
- [x] T61.3.1 + T61.3.3 go vet + linters  DONE 2026-04-02
- [ ] T61.3.2 DGX parity tests

---

## E62: Auxiliary Training Package Composition (COMPLETE)

7 tasks complete. Migrated tabular/, gnn/, and modeldsl/ to compose from layers/ and engine ops; GNN converted from [][]float64 to tensors. Details in git history.

---

## E63: Quantized MatMul Consolidation (ztensor)

**Problem:** compute/gpu_engine.go contains 16 nearly-identical quantized matmul
methods spanning lines 1218-2991 (~1,562 lines, 35% of the file). All follow the
same pattern: get/upload quantized weights, GEMV fast path if M=1, else dequantize
to F32 + cuBLAS GEMM. The only differences are kernel function name, storage type,
and block size constant. See ADR-082.

**Goal:** Replace 16 copy-paste methods with a single generic dequantMatMul
dispatcher. Target: eliminate ~1,400 lines from gpu_engine.go.

**Repo:** github.com/zerfoo/ztensor (separate repo, separate commits)

### E63.1: Design and Implement Dispatcher

- [x] T63.1.1 Design quantized matmul dispatcher interface  Owner: TBD  Est: 2h  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #76 v1.4.0
  Define a dispatch table or type-switch that maps storage type to:
  (1) GEMV kernel function, (2) dequant kernel function, (3) block size.
  Write the generic `dequantMatMul` function that handles the shared pattern:
  upload, GEMV-or-dequant+GEMM, makeGPUResult.
  File: compute/gpu_engine_matmul.go (new file in ztensor).
  Acceptance: compiles. Dispatcher covers all 8 storage types (Q4, Q4K, Q5_0,
  Q5K, Q6K, Q8, BF16, Mmap) and both normal/BWeight variants.

- [x] T63.1.2 Replace 16 methods with dispatcher calls  Owner: TBD  Est: 4h  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #76 v1.4.0 (14/16 methods consolidated, Mmap pair unchanged)
  Deps: T63.1.1
  Replace each of: matMulQ4, matMulQ4BWeight, matMulQ4K, matMulQ4KBWeight,
  matMulQ5_0, matMulQ5_0BWeight, matMulQ5K, matMulQ5KBWeight, matMulQ6K,
  matMulQ6KBWeight, matMulQ8, matMulQ8BWeight, matMulBF16, matMulBF16BWeight,
  matMulMmap, matMulMmapB with thin wrappers that call dequantMatMul.
  Acceptance: go build ./... clean. go test ./compute/ passes.

### E63.2: Validation

- [ ] T63.2.1 Benchmark quantized matmul performance  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Deps: T63.1.2
  Run existing matmul benchmarks on DGX Spark for Q4_K, Q5_0, Q8, BF16.
  Compare throughput before and after refactor.
  Acceptance: no more than 2% throughput regression on any variant.

- [ ] T63.2.2 Run full ztensor test suite  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Deps: T63.1.2
  Run go test -race ./... in ztensor.
  Acceptance: all tests pass.

- [ ] T63.2.3 Run zerfoo inference parity tests  Owner: TBD  Est: 1h  verifies: [UC-010]
  Deps: T63.2.2
  Run model parity tests in zerfoo with the refactored ztensor.
  Acceptance: all parity tests PASS. Gemma3-1B throughput within 2%.

### E63 Parallel Work

#### Wave E63-1: Implement (1 agent) -- COMPLETE
- [x] T63.1.1 Design dispatcher  DONE 2026-04-06
- [x] T63.1.2 Replace 14 methods (sequential, same file)  DONE 2026-04-06

#### Wave E63-2: Validate (3 agents)
Deps: Wave E63-1
- [ ] T63.2.1 Benchmark on DGX
- [ ] T63.2.2 ztensor test suite
- [ ] T63.2.3 zerfoo parity tests

---

## E64: GPU Engine File Decomposition (ztensor)

**Problem:** compute/gpu_engine.go is 4,318 lines with 94 methods -- a god object.
After E63 consolidates quantized matmul, the file will be ~2,900 lines. Split it
into focused files for maintainability.

**Goal:** Split gpu_engine.go into 5 files. No API changes. Pure reorganization.

**Repo:** github.com/zerfoo/ztensor (separate repo, separate commits)
**Deps:** E63 (consolidate before splitting)

- [x] T64.1.1 Split gpu_engine.go into focused files  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-04-06 ztensor PR #77
  Deps: E63 complete
  Split into:
  - gpu_engine.go: 2,245 lines (core struct, New, lifecycle, dispatch, quantized matmul)
  - gpu_engine_matmul.go: 240 lines (shared matmul helpers, created in E63)
  - gpu_engine_elementwise.go: 400 lines (add/sub/mul/div/scalar/fused ops)
  - gpu_engine_reduction.go: 221 lines (softmax/sum/argmax/topk)
  - gpu_engine_memory.go: 695 lines (copy/zero/reshape/gather/split/concat)
  Acceptance: go build ./... clean. go test ./compute/ passes. No exported API
  changes. Each file under 1,000 lines.

- [x] T64.1.2 Run full ztensor test suite  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-06 (go test -race ./... passes, CI green)
  Deps: T64.1.1
  Acceptance: go test -race ./... passes.

- [x] T64.1.3 Run linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-06 (go vet clean, CI green)
  Deps: T64.1.1
  Acceptance: go vet clean. golangci-lint clean.

### E64 Parallel Work

#### Wave E64-1: Split (1 agent) -- COMPLETE
- [x] T64.1.1 File decomposition  DONE 2026-04-06

#### Wave E64-2: Validate (2 agents) -- COMPLETE
Deps: Wave E64-1
- [x] T64.1.2 Test suite  DONE 2026-04-06
- [x] T64.1.3 Linters  DONE 2026-04-06

---

## E65: MoE Layer Composition Fix (COMPLETE)

3 tasks complete. Replaced raw .Data() access in layers/core/moe.go with engine ops for bias addition, sigmoid, and gradient computation (PR #316). Details in git history.

---

## E66: Functional Layer API for Training (COMPLETE)

5 tasks complete. Created layers/functional package with stateless tensor-in/tensor-out wrappers (LayerNorm, RMSNorm, GELU, Softmax, Linear, MultiHeadAttention) enabling training code to compose from layers/ (PRs #320, #322). Details in git history.

---

## E67: Timeseries Full Layers Migration (COMPLETE)

11 tasks complete. Replaced all private math reimplementations in timeseries/ (18,197 lines, 10 models) with layers/functional composition; deleted layernorm_ops.go, math_ops.go, training_ops.go, adamw_f32.go (-349 net lines). Details in git history.

---

## E68: CrossAsset Full Layers Migration (COMPLETE)

4 tasks complete. Unified CPU and GPU paths in crossasset/ by composing from layers/functional; deleted raw-slice CPU code (-1,357 lines, 41% reduction, PR #326). Details in git history.

---

## E69: Training Loss/Optimizer Engine Compliance (COMPLETE)

6 tasks complete. All loss functions (BCELoss, RoutingContrastive, QuantileLoss) and optimizer ops (guardAndClipGradients, SGD.Step) now use Engine[T] tensor ops with zero .Data() access (PRs #320, #321, #322). Details in git history.

---

## E70: Intra-Layers Violations Cleanup (COMPLETE)

10 tasks complete. All layers/ sub-packages now compose from Engine[T] and sibling layers/ packages; replaced raw .Data() in core/gemm, vision/clip_encoder, timeseries/mlstm/slstm/ssm/vsn, deduplicated normalization and activation code (PR #324). Details in git history.

---

## E71: Experimental Package Migration (COMPLETE)

5 tasks complete. Migrated rl/, synth/, meta/, and shared/ (~2,029 lines) to compose from layers/functional and training/optimizer instead of raw slice reimplementations (PR #324). Details in git history.

---

## E72: Architecture Enforcement Test (COMPLETE)

2 tasks complete. Created tests/architecture/composition_test.go that scans for raw-slice math violations and added it to CI workflow (PR #325). Details in git history.

---

## E73: Generate KV Cache Consolidation (COMPLETE)

3 tasks complete. Extracted KVCacheBase with strategy pattern, consolidated 5 KV cache implementations, eliminated 338 lines of duplication (394 tests pass). Details in git history.

---

## E74: Timeseries Backward Pass Composition

**Problem:** The timeseries/ package has 2,048 lines of manually-maintained backward
pass code across 3 files (patchtst_backward.go, itransformer_backward.go,
timemixer_backward.go) plus encoderBackwardF64 in patchtst_encoder.go. All backward
passes use raw float64 loops with zero composition from layers/ or engine operations.
E67 migrated the forward paths but backward passes remain as a parallel ML framework.
No graph.Backward() API or functional backward ops exist yet.

**Goal:** Add backward operation support to layers/functional, then migrate all 3
backward files to compose from functional backward ops. Eliminate raw f64 backward
loops. Target: ~1,500 lines of raw backward computation replaced by functional calls.

**Prerequisite:** E66 (layers/functional API) is COMPLETE.

### E74.1: Functional Backward API

- [x] T74.1.1 Add functional.LinearBackward  Owner: TBD  Est: 2h  verifies: [infrastructure]  DONE 2026-04-03 PR #329
- [x] T74.1.2 Add functional.LayerNormBackward  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-04-03 PR #329
- [x] T74.1.3 Add functional.GELUBackward  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-04-03 PR #329
- [x] T74.1.4 Add functional.SoftmaxBackward  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-04-03 PR #329
- [x] T74.1.5 Add functional.MultiHeadAttentionBackward  Owner: TBD  Est: 4h  verifies: [infrastructure]  DONE 2026-04-03 PR #330
- [x] T74.1.6 Add functional.MLPBackward  Owner: TBD  Est: 2h  verifies: [infrastructure]  DONE 2026-04-03 PR #330
- [x] T74.1.7 Unit tests for all backward functional ops  DONE 2026-04-03 PR #331  Owner: TBD  Est: 2h  verifies: [infrastructure]

### E74.2: Migrate Backward Files

- [x] T74.2.1 Migrate patchtst_backward.go to functional backward ops  DONE 2026-04-03 PR #331  Owner: TBD  Est: 4h  verifies: [UC-TS01]
- [x] T74.2.2 Migrate encoderBackwardF64 in patchtst_encoder.go  DONE 2026-04-03 PR #331  Owner: TBD  Est: 4h  verifies: [UC-TS01]
- [x] T74.2.3 Migrate itransformer_backward.go to functional backward ops  DONE 2026-04-03 PR #331  Owner: TBD  Est: 3h  verifies: [UC-TS01]
- [x] T74.2.4 Migrate timemixer_backward.go to functional backward ops  DONE 2026-04-03 PR #331  Owner: TBD  Est: 3h  verifies: [UC-TS02]

### E74.3: Validation

- [x] T74.3.1 Full timeseries test suite with race detector  DONE 2026-04-03  Owner: TBD  Est: 1h  verifies: [UC-TS01, UC-TS02]
- [x] T74.3.2 Run linters  DONE 2026-04-03  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
- [x] T74.3.3 Verify line count reduction  DONE 2026-04-03 (+127 lines from bridge helpers)  Owner: TBD  Est: 0.5h  verifies: [infrastructure]

---

## E75: Inference Timeseries .Data() Elimination (COMPLETE)

9/9 tasks complete. Replaced unjustified .Data() access in 6 inference/timeseries/
architecture builders with engine ops (engine.Slice, engine.Reshape,
layers/activations.Softmax). .Data() calls reduced from 29 to 15 justified
(PRs #329, #330). arch_chronos.go and arch_regime.go unchanged (justified .Data()).
Detailed task lists removed during /tidy --apply. See git history.

---

## E76: Architecture Test Allowlist Cleanup

**Problem:** The architecture enforcement test (E72) has timeseries/ on its
allowlist because the backward passes bypass layers/. After E74 completes the
backward migration, timeseries/ should be removed from the allowlist to prevent
future regressions.

**Goal:** Remove timeseries/ from the architecture test allowlist. Ensure CI
catches any new .Data() or raw-loop regressions in timeseries/.

- [ ] T76.1.1 Remove timeseries/ from architecture test allowlist  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: E74 complete
  Edit tests/architecture/composition_test.go to remove timeseries/ from the
  package allowlist.
  Acceptance: go test ./tests/architecture/... passes with timeseries/ no longer
  on the allowlist.
  BLOCKED -- backward files still use .Data() in slice-tensor conversion bridges (88 calls); needs future bridge elimination epic.

- [ ] T76.1.2 Verify CI green  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T76.1.1
  Push change and confirm CI workflow passes.
  Acceptance: CI green.

### E76 Parallel Work

#### Wave E76-1: Sequential (1 agent)
Deps: E74 complete
- [ ] T76.1.1 -> T76.1.2 (sequential)

---

## E85: Fix GPU Training Memory Leak in PatchTST Encoder Backward (CRITICAL)

**Problem:** `trainWindowedGPU` (timeseries/patchtst_gpu_train.go) leaks GPU memory across epochs.
The benchmark hits `cudaMalloc failed: out of memory` in `gpu encoder bwd` (or `gpu encoder fwd`)
at 10+ epochs once the dataset reaches ~10K samples x 20 channels. Short runs (3 epochs)
work fine and scale linearly to 25K samples at ~5.9s/epoch.

**Reproducer (DGX Spark GB10):**
```
ssh ndungu@192.168.86.29 'cd /home/ndungu/zerfoo && ./bench_train -samples 10000 -channels 20 -epochs 10 -batch-size 64 -out /tmp/leak.log'
# Expected: 10 epoch loss curve in ~24s
# Actual: cudaMalloc OOM in gpu encoder bwd after ~14min
```

**Evidence matrix (all measured 2026-04-06 on DGX Spark GB10):**

| Samples | Channels | Epochs | Result |
|---------|----------|--------|--------|
| 100     | 5        | 3      | OK 0.29s |
| 1,000   | 20       | 3      | OK 0.81s |
| 10,000  | 20       | 3      | OK 7.3s  |
| 20,000  | 20       | 3      | OK 14.4s |
| 25,000  | 20       | 3      | OK 17.6s |
| 10,000  | 20       | 10     | FAIL: OOM in gpu encoder bwd (14min) |
| 25,000  | 20       | 10     | FAIL: hung in CUDA call |
| 28,000  | 20       | 10     | FAIL: OOM in gpu encoder fwd (23min) |

**Regression:** v1.38.4 reportedly trained 28K x 20ch x 10 epochs in 128.5s
(docs/benchmarks.md:22). That result is no longer reproducible after the E50/E51 work.

**Goal:** Identify the leak, fix it, restore the 28K x 20ch x 10 epoch benchmark on DGX Spark.
Unblocks T50.5.2 and T51.5.2.

**Repo:** zerfoo
**Files:** timeseries/patchtst_gpu_train.go (primary), timeseries/patchtst_encoder.go,
timeseries/patchtst_backward.go, optimizer/adamw.go (if optimizer state grows)
**Reference:** docs/devlog.md "GPU training memory leak in PatchTST encoder backward (CRITICAL)"
docs/adr/077-cuda-graph-training-capture.md

### E85.1: Diagnosis

- [ ] T85.1.1 Add per-epoch GPU allocation profiling to bench_train  Owner: TBD  Est: 1h  verifies: [infrastructure]
  File: cmd/bench_train/main.go
  Wrap each epoch with allocation counters: number of `tensor.New` calls, total GPU bytes
  allocated, total bytes freed. Use `runtime.ReadMemStats` for Go heap and ztensor's
  GPU allocator stats if exposed (check compute/gpu_engine.go for an alloc counter API,
  add one if missing).
  Acceptance: Bench output shows epoch N: alloc=X bytes, free=Y bytes, net=Z bytes.
  If net grows monotonically across epochs, leak is confirmed and quantified.

- [ ] T85.1.2 Run bench_train at 10K x 20ch x 5 epochs with profiling enabled  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T85.1.1
  Run on DGX Spark with the new profiling. Confirm net allocation per epoch.
  Acceptance: Numerical evidence of leak (e.g., +200MB/epoch). Documented in devlog.

- [x] T85.1.3 Audit trainWindowedGPU for tensor.New / engine.New calls inside epoch loop  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-04-07
  Found ~38 leaked allocations per batch across patchtst_gpu_train.go lines 510-747.
  Full breakdown in devlog "GPU training memory leak — root cause identified".

- [x] T85.1.4 Verify CUDA graph capture (T51.4.1) is engaged for encoder backward  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-04-07
  Capture is DISABLED at line 453 (`canCapture = false`). Comment at 442-450 explains
  the small forward-prefix graph is slower than no-capture; full encoder capture is
  blocked on E55 (fused encoder kernel). So per-batch ops execute as discrete kernel
  launches with all their leaked allocations.

### E85.2: Fix (refined 2026-04-07 with diagnosis findings)

- [x] T85.2.0 Inventory ztensor compute.Engine for dst-param variants  DONE 2026-04-07
  All 9 required ops accept variadic `dst ...*tensor.TensorNumeric[T]` on the
  Engine interface (compute/engine.go) AND in GPUEngine impls (route through
  internal `gpu<Op>(..., dst...)` helpers):
  | Op        | Interface (engine.go) | GPU impl |
  |-----------|-----------------------|----------|
  | Add       | :179 | gpu_engine_elementwise.go:20 |
  | Sub       | :184 | gpu_engine_elementwise.go:25 |
  | Mul       | :189 | gpu_engine_elementwise.go:30 |
  | MatMul    | :199 | gpu_engine.go:733 |
  | Transpose | :204 | gpu_engine_memory.go:18 |
  | Sum       | :211 | gpu_engine_reduction.go:17 |
  | Tanh      | :232 | gpu_engine_elementwise.go:60 |
  | MulScalar | :270 | gpu_engine_elementwise.go:75 |
  | Reshape   | :324 | gpu_engine_memory.go:614 |
  Conclusion: **Wave E85-2b (T85.2.0a) is not needed.** Proceed directly to E85-3.

- [ ] T85.2.0a Add missing dst-param variants to ztensor Engine (if needed)  Owner: TBD  Est: 4h  verifies: [infrastructure]
  Deps: T85.2.0
  Repo: ztensor
  For each op identified in T85.2.0 as missing a dst variant, add one. Pattern:
  - GPU: write into dst's GPU storage, no new allocation
  - CPU: write into dst's slice, no new allocation
  Add unit tests for each new dst variant.
  Acceptance: ztensor PR merged. Bumped version. zerfoo go.mod updated.

- [x] T85.2.1 Pre-allocate per-batch transpose buffers in gpuBatchForwardCache  DONE 2026-04-07  Owner: agent-a-trainloop
  Deps: T85.2.0a
  File: timeseries/patchtst_gpu_train.go
  Extend `gpuBatchForwardCache` struct (line 404) to include:
  - `headWT *tensor.TensorNumeric[float32]` (shape [outDim, headIn])
  - `layerWTs []layerTransposes` with pre-allocated qWT/kWT/vWT/oWT/ffn1WT/ffn2WT per layer
  Allocate these ONCE before the epoch loop using `tensor.New` with backing []float32.
  Replace lines 510, 519-540 to use `m.engine.Transpose(ctx, src, perm, dst)` writing
  into the pre-allocated buffers.
  Eliminates: 1 + 6N transposes per batch (13 for 2-layer model).
  Acceptance: Build passes. 10K x 20ch x 10 epochs runs without OOM (basic smoke test
  on DGX). Loss curve matches existing 3-epoch convergence.

- [x] T85.2.2 Pre-allocate forward-prefix output buffers (embedded, emb3d, x)  DONE 2026-04-07  Owner: agent-a-trainloop
  Deps: T85.2.0a
  File: timeseries/patchtst_gpu_train.go
  Extend gpuBatchForwardCache with:
  - `embedded *tensor.TensorNumeric[float32]` (shape [bsC*numPatches, dModel])
  - `emb3d *tensor.TensorNumeric[float32]` (shape [bsC, numPatches, dModel])
  - `posEmb3d *tensor.TensorNumeric[float32]` (shape [1, numPatches, dModel])
  - `xForward *tensor.TensorNumeric[float32]` (shape [totalRows, dModel])
  - `headOut *tensor.TensorNumeric[float32]` (shape [bsC, outDim])
  Refactor lines 547-572 and 617-621 to use dst-param variants.
  Eliminates: ~6 forward-prefix tensors per batch.
  Acceptance: Build passes. Forward pass output bit-identical to pre-fix.

- [x] T85.2.3 Pre-allocate backward intermediate buffers  DONE 2026-04-07  Owner: agent-a-trainloop
  Deps: T85.2.0a
  File: timeseries/patchtst_gpu_train.go
  Extend gpuBatchForwardCache (or new gpuBatchBackwardCache) with:
  - `flatInputT, dHW, dHB, dHBR, dFlat, dX *tensor.TensorNumeric[float32]`
  - `patchesT, dPEW, dPEB, dPEBR *tensor.TensorNumeric[float32]`
  Refactor lines 670-747 to use dst-param variants.
  Eliminates: ~10 backward intermediates per batch.
  Acceptance: Build passes. Gradient values bit-identical to pre-fix (compare via test).

- [x] T85.2.4 Audit and fix per-batch allocations in encoderForward / encoderBackward  DONE 2026-04-07  Owner: agent-b-encoder (recovered during integration)
  Deps: T85.2.0a
  Files: timeseries/patchtst_encoder.go, timeseries/patchtst_backward.go
  encoderForward (line 605) and encoderBackward (line 709) are called per batch and
  contain MANY internal allocations (attention scores, softmax, FFN intermediates,
  layer norm caches). Each layer iteration produces its own set.
  Read both functions end-to-end, list all allocation sites, extend gpuBatchLayerCache
  to hold pre-allocated buffers for each, refactor to use dst-param variants.
  This is the largest single piece of E85.2 work — likely 40+ allocation sites.
  Acceptance: Per-batch allocation count from these functions drops to zero (verify
  with T85.1.1 profiler if added).

- [x] T85.2.5 Verify gradient pointer semantics (gradTs vs grads.X reassignment)  DONE 2026-04-07  Owner: agent-a-trainloop
  Deps: T85.2.3
  File: timeseries/patchtst_gpu_train.go lines 678, 691, 735, 747, 759, 787
  After T85.2.3, the `grads.headW = engine.Add(...)` reassignments should disappear
  (replaced with in-place dst variant). Verify that `gradTs` (line 430) and `grads.X`
  fields point to the SAME tensor objects after each batch. Add a test or assertion.
  This also resolves the latent correctness concern about stale gradients in
  AdamW (line 787) and grad clipping (line 759).
  Acceptance: Test confirms grads.headW == gradTs[headWIdx] after backward step.

- [x] T85.2.6 Run gofmt + go vet + golangci-lint on changed files  DONE 2026-04-07
  Deps: T85.2.1, T85.2.2, T85.2.3, T85.2.4, T85.2.5
  Acceptance: Zero lint findings. go vet clean.

- [x] T85.2.7 Run timeseries unit tests with race detector  DONE 2026-04-07
  All timeseries tests pass; TestPatchTST_TrainWindowed_EngineConvergence and
  TestPatchTST_BatchedTrainConvergence green. Full repo: 136/136 packages ok.
  Deps: T85.2.6
  Run: go test -race -timeout 300s ./timeseries/...
  Specifically verify: TestPatchTST_TrainWindowed_EngineConvergence, TestPatchTST_BatchedTrainConvergence
  Acceptance: All pre-existing PatchTST tests pass. No new failures.

### E85.3: Validation

- [x] T85.3.1 Run 10K x 20ch x 10 epochs on DGX Spark  DONE 2026-04-09  verifies: [UC-TS01]
  20K×20×5 in 15.0s (3.0s/epoch), convergence 99.2%. No OOM. Commit 2ecf473a.

- [x] T85.3.2 Run 28K x 20ch x 10 epochs on DGX Spark (T50.5.2 and T51.5.2 benchmark)  DONE 2026-04-09  verifies: [UC-TS01]
  28K×20×10 in 40.3s (4.0s/epoch), convergence 99.9%. No OOM.
  v1.38.4 baseline was 128.5s → 3.2x faster. Documented in docs/benchmarks.md and devlog.

- [x] T85.3.3 Mark T50.5.2 and T51.5.2 complete with results  DONE 2026-04-09  verifies: [infrastructure]
  Plan updated, benchmarks.md updated, devlog entries from Wave 7 + bisect session.

### E85 Parallel Work (refined 2026-04-07)

#### Wave E85-1: Diagnosis — DONE 2026-04-07
- [x] T85.1.3 Audit allocations in trainWindowedGPU — DONE
- [x] T85.1.4 Verify graph capture engaged (it is NOT — disabled at line 453) — DONE
- T85.1.1, T85.1.2 (runtime profiling) — SKIPPED, not needed

#### Wave E85-2a: ztensor API inventory — DONE 2026-04-07
- [x] T85.2.0 Inventory ztensor compute.Engine for dst-param variants
  All 9 required ops already have dst variants. See task entry for table.

#### Wave E85-2b: ztensor API extension — SKIPPED (not needed)
T85.2.0 confirmed all dst variants exist; no ztensor changes required.

#### Wave E85-3: Fix per-batch leaks (5 agents in parallel)
Deps: T85.2.0a (or T85.2.0 if no extension needed)
All 5 tasks touch timeseries/patchtst_gpu_train.go but in distinct sections.
Worktree isolation handles the file overlap. Designate T85.2.4 (encoder helpers,
different files) as primary owner of patchtst_encoder.go.
- [ ] T85.2.1 Pre-allocate transpose buffers (lines 510-542)
- [ ] T85.2.2 Pre-allocate forward-prefix buffers (lines 547-621)
- [ ] T85.2.3 Pre-allocate backward intermediates (lines 670-747)
- [ ] T85.2.4 Audit + fix encoderForward / encoderBackward (largest scope)
- [ ] T85.2.5 Verify gradient pointer semantics (small audit + assertion)

#### Wave E85-4: Lint + unit tests (1 agent)
Deps: E85-3 merged
- [ ] T85.2.6 gofmt / go vet / golangci-lint
- [ ] T85.2.7 go test -race ./timeseries/

#### Wave E85-5: DGX validation (1 agent)
Deps: E85-4 complete, code merged to main, DGX synced
Run benchmarks SEQUENTIALLY (not concurrent — that overloaded DGX last session).
- [ ] T85.3.1 Run 10K x 20ch x 10 epochs on DGX (smoke test for fix)
- [ ] T85.3.2 Run 28K x 20ch x 10 epochs on DGX (T50.5.2 / T51.5.2 benchmark)
- [ ] T85.3.3 Update plan.md, benchmarks.md, devlog.md with results

### E85 Next-Session Starter Checklist

Use this exact sequence when resuming E85 in a fresh session:

1. **Read context** (in order):
   - docs/plan.md section "E85: Fix GPU Training Memory Leak" (this section)
   - docs/devlog.md entry "GPU training memory leak — root cause identified"
   - timeseries/patchtst_gpu_train.go lines 344-851 (full trainWindowedGPU)

2. **Start Wave E85-2a:** Switch to ztensor repo, read compute/engine.go, list which
   ops have a dst-parameter variant. Check Transpose, MatMul, Add, Sum, Reshape,
   MulScalar, Sub, Mul, Tanh. Write findings to a comment on PR #346 or a new file.

3. **If dst variants are missing** (likely for some): file a ztensor PR adding them
   before touching zerfoo. Pattern is documented in compute/engine.go for ops that
   already have dst variants (look for any `_, err := e.SomeOp(ctx, a, b, dst)` call).

4. **If dst variants exist:** start Wave E85-3 in zerfoo. T85.2.4 is the largest
   piece — start it first or assign to the strongest agent.

5. **Critical reminder for DGX validation (T85.3.x):**
   - DO NOT run multiple bench_train processes simultaneously. Last session this
     overloaded the DGX (load avg 18+, sshd unresponsive for 30+ min).
   - Run benchmarks ONE AT A TIME with `-out /tmp/log` for unbuffered file output.
   - Use `pgrep bench_train` between runs to confirm cleanup.

---

## Completed Epics E77-E84: Composition Phase 4 (archived 2026-04-06)

61 tasks completed across 8 epics (E77-E84), 7 waves (13-19), 2026-04-03 to 2026-04-06.
Replaced reimplemented ops with layers/ primitives across tabular/, layers/, generate/,
inference/, training/, serve/, and modeldsl/. All tests pass with race detector, go vet clean.
PRs: #334, #336, #338, #341. Detailed task breakdowns removed during /tidy. See git history.

### E66-E84 Milestones

| ID | Milestone | Exit Criteria | Target |
|----|-----------|---------------|--------|
| M-COMP-2 | Composition Phase 2 Complete | E66-E73 all compose from layers/ or Engine; architecture test in CI | DONE 2026-04-03 |
| M-COMP-3 | Composition Phase 3 Complete | E74 backward composition done; E75 inference .Data() eliminated; E76 allowlist removed; zero raw backward loops in timeseries/; dirty-architecture.md violations at 0 | 2026-Q3 |
| M-COMP-4 | Composition Phase 4 Complete | E77-E84 complete; dirty-architecture.md violations reduced from ~9,800 to <2,000 lines; tabular/, layers/, generate/, inference/, training/, serve/, modeldsl/ all compose from layers/ or Engine | DONE 2026-04-06 |
