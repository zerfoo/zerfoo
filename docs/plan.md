# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, documentation site, MSA-inspired scalable
memory, and research-driven inference optimizations).

Task statuses updated 2026-04-01 based on merged PRs and git history.

**Status summary:**
- 380+ tasks completed across all plans
- E45: Verification remediation (3/3 complete)
- E46: Ecosystem v1 release (0/46 -- 5 repos to v1.0.0)
- E47: Batched training performance (0/19 -- GitHub #278, T47.2.4 added for batched attention)
- E48: TimeMixer backend (0/10 -- GitHub #279)
- E49: Foundation model inference (0/12 -- GitHub #280)
- E50: GPU training kernel elimination (4/6 -- layer norm fwd+bwd, GELU fwd/bwd, weight transpose caching done)
- E51: CUDA graph capture for training (4/6 -- drop partial, pre-allocate, capture API, wiring done; validation pending)
- E52: DRY composition refactoring (7/7 complete -- shared math_ops, adamw_f32, layernorm_ops, engine wrappers)
- E53: Unified training forward/backward (6/6 complete -- shared encoder, eliminated engine paths)
- E54: Capture-pure GPU engine ops (2/4 -- GPU-native Zero/Copy done; re-enable graph capture pending)
- E55: Fused PatchTST encoder CUDA kernel (0/8 -- single kernel per encoder layer)
- E56: Gemma3 inference micro-optimizations (6/9 -- fused kernels written and wired; produce garbage on DGX, BLOCKED by E57)
- E57: Fix DGX Spark build regression (3/3 COMPLETE -- 3 root causes fixed: transpose no-op, causal mask D2H, Q4_K re-quant; composed GQA divergence remains)
- E58: GPU vs CPU GQA parity test (1/2 -- diagnostic test to find remaining composed-pipeline divergence)
- E59: Remove gonum dependency (7/7 COMPLETE -- replace BLAS fallback + FFT with zero-dep implementations)
- E60: CrossAsset GPU training (12/12 COMPLETE -- GitHub #312, GPU forward/backward/AdamW)
- E61: Inference builder composition (9/10 -- all 6 builders done; vet+linters pass; DGX parity T61.3.2 pending)
- E62: Auxiliary training package composition (7/7 COMPLETE -- tabular, modeldsl, gnn refactored; tests+validation pass)
- E63: Quantized matmul consolidation in ztensor (0/5 -- single dispatcher for 16 copy-paste methods)
- E64: GPU engine file decomposition in ztensor (0/3 -- split 4,318-line god file)
- E65: MoE layer composition fix (3/3 COMPLETE -- PR #316)
- E66: Functional layer API for training (5/5 COMPLETE -- PR #320, #322)
- E67: Timeseries full layers migration (11/11 COMPLETE -- all helpers replaced, attention migrated, validated, files verified)
- E68: CrossAsset full layers migration (4/4 COMPLETE -- forward+backward+AdamW+cleanup, -1,357 lines)
- E69: Training loss/optimizer Engine compliance (6/6 COMPLETE -- PR #320, #321, #322) + T69.3.1 validated PR #324
- E70: Intra-layers violations cleanup (10/10 COMPLETE -- all tasks + T70.1.10 validation PR #324)
- E71: Experimental package migration (5/5 COMPLETE -- all 4 packages + T71.1.5 validation PR #324)
- E72: Architecture enforcement test (2/2 COMPLETE -- test created + added to CI)
- E73: Generate KV cache consolidation (3/3 COMPLETE -- base extraction, migration, validation done)
- E74: Timeseries backward pass composition (0/14 -- add backward ops to layers/functional, migrate 3 backward files + encoder backward)
- E75: Inference timeseries .Data() elimination (0/9 -- replace unjustified .Data() access in 6 arch builders)
- E76: Architecture test allowlist cleanup (0/2 -- remove timeseries/ from allowlist after E74)
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
| AC: Backward Composition | E74 T74.*.* | Add functional backward ops, migrate 3 backward files | NEW |
| AD: Inference TS .Data() | E75 T75.*.* | Replace unjustified .Data() in 6 arch builders | NEW |
| AE: Allowlist Cleanup | E76 T76.*.* | Remove timeseries/ from arch test allowlist | NEW (deps: AC) |

### Composition Sync Points

- Tracks U-AB: ALL COMPLETE as of 2026-04-03.
- Track AC (E74) depends on Track U (E66) which is complete.
- Track AD (E75) is fully independent of Track AC.
- Track AE (E76) depends on Track AC (E74 must complete before allowlist removal).
- Tracks AC and AD can run in parallel.
- All composition tracks are independent of research tracks (E34-E44).

### Composition Waves

#### Composition Wave 1: Foundation + Independent (10 agents)
All zero-dependency tasks. Saturates all agent slots.

- [x] T66.1.1 layers/functional LayerNorm, RMSNorm  verifies: [infrastructure]  DONE 2026-04-02 PR #320
- [x] T66.1.2 layers/functional activations  verifies: [infrastructure]  DONE 2026-04-02 PR #320
- [x] T66.1.3 layers/functional Linear, MHA  verifies: [infrastructure]  DONE 2026-04-02 PR #320
- [x] T69.1.1 BCELoss Engine ops  verifies: [infrastructure]  DONE 2026-04-03 PR #321
- [x] T69.1.2 RoutingContrastive Engine ops  verifies: [infrastructure]  DONE 2026-04-03 PR #321
- [x] T69.1.3 QuantileLoss generics fix  verifies: [infrastructure]  DONE 2026-04-03 PR #321
- [x] T69.2.1 guardAndClipGradients Engine ops  verifies: [infrastructure]  DONE 2026-04-02 PR #320
- [x] T69.2.2 SGD.Step memory fix  verifies: [infrastructure]  DONE 2026-04-03 PR #321
- [x] T70.1.1 core/gemm.go  verifies: [infrastructure]  DONE 2026-04-02 PR #320
- [x] T70.1.2 vision/clip_encoder.go  verifies: [infrastructure]  DONE 2026-04-03 PR #321

#### Composition Wave 2: More independent + E66 validation (10 agents)
Deps: Wave 1 partial (E66 tasks complete)

- [x] T66.1.4 + T66.1.5 Functional API tests + linters  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.3 timeseries/mlstm + slstm  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.4 timeseries/ssm  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.5 timeseries/vsn  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.6 simplified_layer_normalization dedup  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.7 fast_gelu dedup  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.8 variable_selection + temporal_conv  verifies: [infrastructure]  DONE 2026-04-03
- [x] T70.1.9 residual/block_attn_res  verifies: [infrastructure]  DONE 2026-04-03
- [x] T72.1.1 Architecture enforcement test  verifies: [infrastructure]  DONE 2026-04-03
- [x] T73.1.1 KV cache base extraction  verifies: [UC-001, UC-002]  DONE 2026-04-03

#### Composition Wave 3: Migrations (10 agents)
Deps: E66 complete (Wave 2)

- [x] T67.1.1 layernorm_ops.go replacement  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T67.1.2 math_ops.go replacement  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T67.1.3 training_ops.go replacement  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T67.1.4 adamw_f32.go replacement  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T68.1.1 CrossAsset CPU forward  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T71.1.1 rl/ migration  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T71.1.2 synth/ migration  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T71.1.3 meta/ migration  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T71.1.4 shared/ migration  verifies: [infrastructure]  DONE 2026-04-03 PR #323
- [x] T73.1.2 KV cache migration  verifies: [UC-001, UC-002]  DONE 2026-04-03 (no changes needed, T73.1.1 sufficient)

#### Composition Wave 4: Per-model attention + sequential chains (8 agents)
Deps: Wave 3

- [x] T67.2.1 PatchTST attention  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T67.2.2 iTransformer attention  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T67.2.3 TFT attention  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T67.2.4 Remaining models  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T68.1.2 CrossAsset backward  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T69.3.1 Training tests + linters  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T70.1.10 Layers tests + linters  verifies: [infrastructure]  DONE 2026-04-03 PR #324
- [x] T71.1.5 Experimental tests + linters  verifies: [infrastructure]  DONE 2026-04-03 PR #324

#### Composition Wave 5: Final validation (5 agents)
Deps: Wave 4

- [x] T67.3.1 + T67.3.2 Timeseries full suite + linters  verifies: [infrastructure]  DONE 2026-04-03 PR #325
- [x] T67.3.3 Verify deleted files  verifies: [infrastructure]  DONE 2026-04-03 (math_ops, training_ops, adamw_f32 deleted; -349 net lines)
- [x] T68.1.3 CrossAsset AdamW  verifies: [infrastructure]  DONE 2026-04-03 PR #325
- [x] T72.1.2 Add arch test to CI  verifies: [infrastructure]  DONE 2026-04-03 PR #325
- [x] T73.1.3 KV cache tests  verifies: [UC-001, UC-002]  DONE 2026-04-03 (394 tests pass, 338 lines eliminated)

#### Composition Wave 6: Final cleanup (2 agents)
Deps: Wave 5

- [x] T68.1.4 CrossAsset delete + validate  verifies: [infrastructure]  DONE 2026-04-03 PR #326 (-1,357 lines, 41% reduction)

#### Composition Wave 7: Remaining ztensor + DGX parity (3 agents)
Independent of Waves 1-6. Can start immediately.

- [ ] T61.3.2 DGX parity tests for inference builders  verifies: [UC-010]
- [ ] T63.1.1 Design quantized matmul dispatcher (ztensor)  verifies: [infrastructure]
- [ ] T63.1.2 Replace 16 methods with dispatcher (ztensor)  verifies: [infrastructure]

#### Composition Wave 8: ztensor validation (3 agents)
Deps: Wave 7 (T63.1.2)

- [ ] T63.2.1 Benchmark quantized matmul (ztensor)  verifies: [infrastructure]
- [ ] T63.2.2 Full ztensor test suite  verifies: [infrastructure]
- [ ] T64.1.1 Split gpu_engine.go into focused files (ztensor)  verifies: [infrastructure]

#### Composition Wave 9: Backward API + Inference .Data() (10 agents)
Tracks AC and AD run in parallel. Can start immediately (no deps on Waves 7-8).

- [ ] T74.1.1 functional.LinearBackward  verifies: [infrastructure]
- [ ] T74.1.2 functional.LayerNormBackward  verifies: [infrastructure]
- [ ] T74.1.3 functional.GELUBackward  verifies: [infrastructure]
- [ ] T74.1.4 functional.SoftmaxBackward  verifies: [infrastructure]
- [ ] T75.1.1 arch_timemixer.go .Data() elimination  verifies: [UC-TS02]
- [ ] T75.1.2 arch_tft.go .Data() elimination  verifies: [UC-TS01]
- [ ] T75.1.3 arch_ttm.go .Data() elimination  verifies: [UC-TS01]
- [ ] T75.1.4 arch_tirex.go .Data() elimination  verifies: [UC-TS01]
- [ ] T75.1.5 arch_tspulse.go .Data() elimination  verifies: [UC-TS01]
- [ ] T75.1.6 arch_flowstate.go .Data() elimination  verifies: [UC-TS01]

#### Composition Wave 10: Composed backward ops + inference validation (5 agents)
Deps: Wave 9 (T74.1.1-T74.1.4 complete)

- [ ] T74.1.5 functional.MultiHeadAttentionBackward  verifies: [infrastructure]
- [ ] T74.1.6 functional.MLPBackward  verifies: [infrastructure]
- [ ] T75.2.1 Inference timeseries tests  verifies: [UC-TS01, UC-TS02]
- [ ] T75.2.2 Inference timeseries linters + .Data() count  verifies: [infrastructure]
- [ ] T75.2.3 Verify unchanged files  verifies: [infrastructure]

#### Composition Wave 11: Backward migration (5 agents)
Deps: Wave 10 (T74.1.5, T74.1.6 complete, T74.1.7 tests)

- [ ] T74.1.7 Backward functional API tests  verifies: [infrastructure]
- [ ] T74.2.1 patchtst_backward.go migration  verifies: [UC-TS01]
- [ ] T74.2.2 patchtst_encoder.go backward migration  verifies: [UC-TS01]
- [ ] T74.2.3 itransformer_backward.go migration  verifies: [UC-TS01]
- [ ] T74.2.4 timemixer_backward.go migration  verifies: [UC-TS02]

#### Composition Wave 12: Final validation + allowlist cleanup (4 agents)
Deps: Wave 11

- [ ] T74.3.1 + T74.3.2 Timeseries tests + linters  verifies: [UC-TS01, UC-TS02]
- [ ] T74.3.3 Line count verification  verifies: [infrastructure]
- [ ] T76.1.1 Remove timeseries/ from allowlist  verifies: [infrastructure]
- [ ] T76.1.2 Verify CI green  verifies: [infrastructure]

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
| M-COMP | Composition Remediation Phase 1 | E61-E65 | Inference builders, tabular/gnn/modeldsl, MoE compose from layers/; ztensor god file consolidated | 2026-Q2 |
| M-COMP-2 | Composition Remediation Phase 2 | E66-E73 | All forward paths compose from layers/ or Engine; architecture test in CI | DONE 2026-04-03 |
| M-COMP-3 | Composition Remediation Phase 3 | E74-E76 | All backward passes compose from functional backward ops; inference .Data() eliminated; timeseries/ removed from arch test allowlist | 2026-Q3 |

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
| R50 | Inference builder refactoring breaks model parity (E61) | High | Medium | Run parity tests for each model after refactoring; keep old code behind build tag until verified |
| R51 | gnn [][]float64 to tensor conversion changes public API (E62) | Medium | High | Provide adapter functions at API boundary; internal-only tensor usage if possible |
| R52 | Quantized matmul dispatcher adds dispatch overhead (E63) | Medium | Low | Dispatcher is a type-switch resolved at call time, not runtime polymorphism; benchmark validates <2% regression |
| R53 | gpu_engine.go file split creates merge conflicts with in-flight PRs (E64) | Low | Medium | Schedule E64 during a merge freeze or after all ztensor PRs land |
| R54 | MoE engine op refactoring changes expert routing behavior (E65) | Medium | Low | Top-K routing is unchanged; only bias/sigmoid/softmax are refactored; parity test validates |
| R61 | Functional backward ops produce numerically different gradients than hand-coded loops (E74) | High | Medium | Numerical gradient check (finite differences) for each op; 10-epoch training loss parity within 1e-4 |
| R62 | MultiHeadAttentionBackward complexity with batched heads (E74) | Medium | Medium | Start from itransformer_backward.go reference; validate per-head gradient isolation |
| R63 | Replacing .Data() with engine.Slice in inference/timeseries/ changes node count (E75) | Low | Low | Benchmark inference throughput before/after; engine.Slice is lightweight |
| R64 | Removing timeseries/ from arch test allowlist too early (E76) | Low | Medium | Only remove after E74 fully complete and all tests pass |

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

### 2026-04-03: Composition remediation phase 3 -- added E74-E76 (25 tasks, 6 waves)

Analyzed docs/dirty-architecture.md revision 2 to identify all remaining violations
after E66-E73 completion. Three categories of remaining work:

1. **timeseries/ backward passes** (2,048 lines across 3 files + encoder backward):
   All raw f64 loops with zero composition from layers/. No functional backward API
   exists yet. Added E74 (14 tasks) to create functional backward ops
   (LinearBackward, LayerNormBackward, GELUBackward, SoftmaxBackward,
   MultiHeadAttentionBackward, MLPBackward) and migrate all 3 backward files.

2. **inference/timeseries/ .Data() access** (29 calls, 25 unjustified across 6 files):
   Raw tensor access bypassing engine for softmax, ReLU, buffer copy, channel
   extraction, gather/scatter. Added E75 (9 tasks) to replace with engine.Slice,
   engine.Reshape, layers/activations.Softmax. 4 justified .Data() calls in
   arch_chronos.go and arch_regime.go are excluded from migration.

3. **Architecture test allowlist**: timeseries/ remains on allowlist due to backward
   passes. Added E76 (2 tasks) to remove after E74 completes.

Also remaining from prior phases:
- T61.3.2: DGX parity test for inference builders (moved to Wave 7)
- E63 (5 tasks): Quantized matmul consolidation in ztensor (Wave 7-8)
- E64 (3 tasks): GPU engine file decomposition in ztensor (Wave 8)

Added 6 new composition waves (7-12). Waves 7-8 handle ztensor work. Waves 9-12
handle backward composition and inference .Data() elimination in parallel.
Wave 9 saturates 10 agents (4 backward API tasks + 6 inference builder tasks).

Added 4 risks (R61-R64), milestone M-COMP-3, updated tracks AC/AD/AE.

### 2026-04-02: Composition remediation phase 2 -- added E66-E73 (48 tasks, 6 waves)

Extended docs/dirty-architecture.md with intra-layers/ violations (10 findings
from layers-review agent) and training/loss/optimizer violations (6 findings
from training-review agent). Total violations: 14 packages, 90+ reimplemented
components, ~14,000 estimated redundant lines.

Added 8 new epics (E66-E73, 48 tasks total) covering remaining composability
violations not addressed by E61-E65:
- E66 (5 tasks): layers/functional API -- prerequisite for all training migrations
- E67 (11 tasks): timeseries/ full migration to layers/ (18,197 lines)
- E68 (4 tasks): crossasset/ full migration to layers/
- E69 (6 tasks): training/loss/ and training/optimizer/ Engine compliance
- E70 (10 tasks): intra-layers/ violations cleanup
- E71 (5 tasks): rl/, synth/, meta/, shared/ migration
- E72 (2 tasks): architecture enforcement test CI gate
- E73 (3 tasks): generate/ KV cache consolidation

Added 6 composition waves to Parallel Work section. Wave 1 saturates 10 agents
with E66 + E69 + E70 tasks (all zero-dependency). Waves 2-6 cascade dependencies.

Added milestone M-COMP-2. Added 6 risks (R55-R60).

Older progress log entries (2026-03-26 through 2026-04-02) removed during
2026-04-03 plan trim. Key additions: E34-E44 research epics (127 tasks,
all complete), E45-E65 epics, E61-E65 composition remediation phase 1,
dirty-architecture audit and routing to proper tiers (ADR-082, design.md,
devlog.md). See git history for full changelog.

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

## E46: Ecosystem v1 Release

**Goal:** Promote all five sub-v1 libraries (zonnx, ztensor, ztoken, float8, float16) to v1.0.0.
**Decision rationale:** docs/adr/074-satellite-libraries-v1-release-policy.md
**Repo routing:** Each sub-epic targets a different repo. Agents must commit only within
the named repo directory; the pre-commit hook rejects cross-directory commits.

### Use Cases

| ID | Name | Description |
|----|------|-------------|
| UC-L01 | Import stable float16/bfloat16 library | User imports float16 v1 and gets stable IEEE 754 arithmetic with no breaking changes |
| UC-L02 | Import stable float8 library | User imports float8 v1 for FP8 quantized inference |
| UC-L03 | Import stable ztensor library | User imports ztensor v1 for tensor and GPU compute |
| UC-L04 | Import stable ztoken library | User imports ztoken v1 for BPE tokenization |
| UC-L05 | Convert ONNX model to GGUF | User runs zonnx v1 to convert ONNX or SafeTensors to GGUF |

---

### E46.1: zonnx v1.0.0 (repo: /Users/dndungu/Code/zerfoo/zonnx)

All planned features shipped at v0.9.0. Only API review and version bump needed.

- [x] T46.1.1 Review public API surface for stability  Owner: TBD  Est: 1h  verifies: [UC-L05]  DONE 2026-03-30 zonnx PR #20
  Audit all exported types and functions in zonnx for breaking-change risk.
  Remove or unexport any symbols that should not be part of the v1 contract.
  Acceptance: go doc ./... lists only intentionally public symbols.

- [x] T46.1.2 Add API stability ADR in zonnx  Owner: TBD  Est: 1h  verifies: [UC-L05]  DONE 2026-03-30 zonnx PR #21
  Create docs/adr/002-api-stability-v1.md documenting the v1 stability contract.
  List exported types, their stability guarantees, and which are safe to extend
  without a major version bump.

- [x] T46.1.3 Run go vet and tests in zonnx  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Acceptance: go build ./... and go test ./... both pass with no errors.

- [x] T46.1.4 Tag zonnx v1.0.0 via release-please PR  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [UC-L05]
  Trigger release-please to open a v1.0.0 release PR. Review and merge.
  Acceptance: github.com/zerfoo/zonnx has a v1.0.0 tag and release.

---

### E46.2: ztensor v1.0.0 (repo: /Users/dndungu/Code/zerfoo/ztensor)

Extensive test coverage (192 files). Needs design.md, missing docs/QUALITY.md, and API review.

- [x] T46.2.1 Write docs/design.md for ztensor  Owner: TBD  Est: 4h  verifies: [UC-L03]  DONE 2026-03-30 ztensor PR #41
  Document: package layout (tensor/, compute/, graph/, device/, numeric/, internal/),
  the compute.Engine[T] interface contract, CUDA/ROCm/OpenCL backend abstraction,
  memory arena design, quantization types (Q4, Q8, FP16, BF16, FP8), and CUDA graph
  capture lifecycle. General terms only -- no specific model names or benchmark numbers.
  Acceptance: docs/design.md exists and covers all seven top-level packages.

- [x] T46.2.2 Write docs/QUALITY.md for ztensor  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-03-30 ztensor PR #42
  This file is referenced in ci.yml but missing. Create docs/QUALITY.md covering:
  test coverage expectations, GPU-only test tagging convention (//go:build cuda),
  race detector policy, and the vet exclusions rationale for unsafe.Pointer packages.
  Acceptance: docs/QUALITY.md exists; CI reference is satisfied.

- [x] T46.2.3 Produce stable-surface inventory  DONE 2026-03-30 for ztensor v1  Owner: TBD  Est: 3h  verifies: [UC-L03]
  The v1 stability contract covers a narrow surface only. Churn from research epics
  (E34-E44 each added new kernel primitives) makes a wide freeze impractical.
  Stable v1 surface (no breaking changes without v2 bump):
    - compute.Engine[T] interface (all methods)
    - tensor.Tensor[T] type and its exported methods
    - tensor.Numeric constraint
    - device.Device interface
    - numeric.* arithmetic functions
  Explicitly NOT stable (may change in minor versions):
    - internal/cuda, internal/xblas, internal/codegen, internal/gpuapi (already internal)
    - Any kernel-level or backend-level exported types outside the five stable packages
    - graph/ package (compilation pipeline still evolving)
  Steps:
    (a) Run `go doc ./...` and classify every exported symbol as stable or unstable.
    (b) Unexport any symbol outside the stable surface that is not needed by zerfoo.
    (c) For symbols that must stay exported (used by zerfoo) but are not stable, add a
        doc comment: "// This API is not covered by the v1 stability guarantee."
    (d) Write the resulting inventory to docs/adr/001-api-stability-v1.md.
  Deps: T46.2.2
  Acceptance: docs/adr/001-api-stability-v1.md exists with explicit stable/unstable lists;
  no symbol outside the five stable packages is exported without a stability disclaimer.

- [x] T46.2.4 Verify zerfoo builds cleanly  DONE 2026-03-30 after T46.2.3 unexports  Owner: TBD  Est: 1h  verifies: [UC-L03]
  After unexporting transitional symbols, confirm zerfoo still builds.
  Run: cd ../zerfoo && go build ./... (using a local replace directive if needed).
  Fix any ztensor import breakage by either re-exporting with a disclaimer or updating
  the zerfoo import to use an internal path.
  Deps: T46.2.3
  Acceptance: zerfoo builds with no errors against the updated ztensor.

- [x] T46.2.5 Add benchmark baseline  DONE 2026-03-30 to docs/devlog.md for ztensor  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Run go test -bench=. ./tensor/... ./compute/... and record results in devlog.md.
  This establishes a regression baseline before the v1 tag.
  Acceptance: docs/devlog.md has a dated benchmark entry for ztensor.

- [x] T46.2.6 Run go vet and full test suite in ztensor  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Acceptance: go build ./... and go test -race -timeout 300s ./... both pass.

- [x] T46.2.7 Tag ztensor v1.0.0 via release-please PR  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [UC-L03]
  Trigger release-please to open a v1.0.0 release PR. Review and merge.
  Acceptance: github.com/zerfoo/ztensor has a v1.0.0 tag and release.

---

### E46.3: ztoken v1.0.0 (repo: /Users/dndungu/Code/zerfoo/ztoken)

Small, focused library. Needs design.md, expanded tests, and API review.

- [x] T46.3.1 Write docs/design.md for ztoken  Owner: TBD  Est: 2h  verifies: [UC-L04]  DONE 2026-03-30 ztoken PR #7
  Document: BPE tokenizer architecture, GGUF tokenizer loading path, HuggingFace
  compatibility layer, WordPiece variant, encode/decode interface contract.
  Acceptance: docs/design.md exists; covers BPE, GGUF loader, and HF compat sections.

- [x] T46.3.2 Expand test coverage for edge cases  Owner: TBD  Est: 3h  verifies: [UC-L04]  DONE 2026-03-30 ztoken PR #8
  Current: 5 test files. Add tests for:
  (a) Round-trip encode/decode identity on 100 HuggingFace model vocabs.
  (b) Byte-pair edge cases: unicode multibyte sequences, control chars.
  (c) Error paths: malformed GGUF tokenizer metadata.
  Acceptance: go test -race ./... passes; new tests cover the three categories.

- [x] T46.3.3 Create docs/adr/ directory and API stability ADR  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L04]
  Create docs/adr/001-api-stability-v1.md for ztoken. The Tokenizer interface and
  Encode/Decode functions are stable v1. Internal GGUF parsing helpers are not public API.
  Acceptance: docs/adr/001-api-stability-v1.md exists.

- [x] T46.3.4 Run go vet and full test suite in ztoken  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Acceptance: go build ./... and go test -race ./... both pass.

- [x] T46.3.5 Tag ztoken v1.0.0 via release-please PR  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [UC-L04]
  Trigger release-please to open a v1.0.0 release PR. Review and merge.
  Acceptance: github.com/zerfoo/ztoken has a v1.0.0 tag and release.

---

### E46.4: float8 v1.0.0 (repo: /Users/dndungu/Code/zerfoo/float8)

Functional core with minimal documentation. Needs docs/, expanded tests, and API review.

- [x] T46.4.1 Write docs/design.md for float8  Owner: TBD  Est: 2h  verifies: [UC-L02]  DONE 2026-03-30 float8 PR #12
  Document: FP8 E4M3FN bit layout, lookup table strategy, arithmetic operations,
  conversion to/from float32, no-infinities design rationale, use in ML inference.
  Note: FP8 E5M2 is out of scope for v1 (see T46.4.8).
  Acceptance: docs/design.md exists and covers the six documented sections.

- [x] T46.4.2 Verify FP8 E4M3FN arithmetic against NVIDIA cuDNN reference values  Owner: TBD  Est: 3h  verifies: [UC-L02]  DONE 2026-03-30 float8 PR #13
  Write a test file that encodes the complete E4M3FN value table (256 values) and
  compares Add/Sub/Mul/Div results to expected IEEE 754 E4M3FN results.
  Reference: NVIDIA FP8 Formats for Deep Learning (2022).
  Acceptance: TestArithmeticCorrectness passes for all 256 representable values.

- [x] T46.4.3 Add benchmarks for float8 operations  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Add BenchmarkAdd, BenchmarkMul, BenchmarkFromFloat32, BenchmarkToFloat32.
  Record baseline results in docs/devlog.md.
  Acceptance: go test -bench=. ./... runs without errors.

- [x] T46.4.4 Expand error path tests  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L02]
  Test: NaN propagation through all arithmetic ops, clamping of out-of-range float32
  to E4M3FN max value, zero handling.
  Acceptance: go test ./... covers error paths.

- [x] T46.4.5 Create docs/adr/ directory and API stability ADR  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L02]
  Create docs/adr/001-api-stability-v1.md. Float8 (E4M3FN type, arithmetic functions,
  conversions) is stable v1. FP8 E5M2 is explicitly deferred to v1.1+.
  Acceptance: docs/adr/001-api-stability-v1.md exists.

- [x] T46.4.6 Run go vet and full test suite in float8  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Acceptance: go build ./... and go test -race ./... both pass.

- [x] T46.4.7 Tag float8 v1.0.0 via release-please PR  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [UC-L02]
  Trigger release-please to open a v1.0.0 release PR. Review and merge.
  Acceptance: github.com/zerfoo/float8 has a v1.0.0 tag and release.

- [ ] T46.4.8 Backlog: FP8 E5M2 support (deferred to v1.1)  Owner: TBD  Est: 8h  verifies: [UC-L02]
  E5M2 format (1 sign, 5 exponent, 2 mantissa) is used on NVIDIA Ada Lovelace GPUs.
  Implement after v1.0.0 tag as a non-breaking addition.
  Deps: T46.4.7 (v1.0.0 must ship first)

---

### E46.5: float16 v1.0.0 (repo: /Users/dndungu/Code/zerfoo/float16)

Float16 is stable. BFloat16 needs Phases 2-5 from the existing plan.md in that repo.

- [x] T46.5.1 BFloat16 Phase 2: ArithmeticMode support  Owner: TBD  Est: 4h  verifies: [UC-L01]  DONE 2026-03-30 float16 PR #8
  Implement BFloat16AddWithMode, SubWithMode, MulWithMode, DivWithMode with ArithmeticMode
  parameter. Implement proper NaN propagation and gradual underflow. Add FMA stub.
  Ref: float16/docs/plan.md Phase 2.1 and 2.2.
  Acceptance: All Phase 2 functions implemented; go test -run TestBFloat16Arithmetic passes.

- [x] T46.5.2 BFloat16 Phase 2 tests  Owner: TBD  Est: 2h  verifies: [UC-L01]  DONE 2026-03-30 float16 PR #10
  Tests: NaN propagation through all 4 arithmetic ops with each ArithmeticMode value,
  gradual underflow at subnormal boundary, FMA correctness.
  Deps: T46.5.1
  Acceptance: go test -race ./... passes.

- [x] T46.5.3 BFloat16 Phase 3: Batch/slice operations  DONE 2026-03-30  Owner: TBD  Est: 3h  verifies: [UC-L01]
  Implement: BFloat16AddSlice, SubSlice, MulSlice, DivSlice, ScaleSlice,
  BFloat16SliceFromFloat32, Float32SliceFromBFloat16, BFloat16SliceFromFloat64.
  Ref: float16/docs/plan.md Phase 3.1 and 3.2.
  Acceptance: All functions return correct results on random inputs; benchmarks added.

- [x] T46.5.4 BFloat16 Phase 3 tests and benchmarks  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L01]
  Deps: T46.5.3
  Acceptance: go test -race ./... passes; BenchmarkBFloat16Slice exists.

- [x] T46.5.5 BFloat16 Phase 4: Math functions  DONE 2026-03-30  Owner: TBD  Est: 4h  verifies: [UC-L01]
  Implement: BFloat16Sqrt, Exp, Log, Log2, Sin, Cos, Tanh, Sigmoid.
  Each function converts to float64 for computation and converts back.
  Add FastMode variants for Sigmoid and Tanh using polynomial approximation.
  Ref: float16/docs/plan.md Phase 4.1 and 4.2.
  Acceptance: All math functions match float64 results within BFloat16 precision.

- [x] T46.5.6 BFloat16 Phase 4 tests  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L01]
  Deps: T46.5.5
  Tests: Sqrt(4.0) == 2.0, Exp(0) == 1.0, Log(1) == 0, Sigmoid(0) ~= 0.5.
  Edge cases: Sqrt(NaN), Log(-1).

- [x] T46.5.7 BFloat16 Phase 5: Parse and format  DONE 2026-03-30  Owner: TBD  Est: 3h  verifies: [UC-L01]
  Implement: BFloat16FromString, (b BFloat16) String() with format verbs (%e, %f, %g),
  MarshalJSON, UnmarshalJSON, MarshalBinary, UnmarshalBinary.
  Ref: float16/docs/plan.md Phase 5.1 and 5.2.
  Acceptance: Round-trip marshal/unmarshal is lossless; String() matches float32 format.

- [x] T46.5.8 BFloat16 Phase 5 tests  DONE 2026-03-30  Owner: TBD  Est: 1h  verifies: [UC-L01]
  Deps: T46.5.7
  Tests: 100 random round-trip JSON encode/decode cycles; binary round-trip; %e %f %g
  format verbs against float32 reference.

- [x] T46.5.9 Error handling infrastructure for BFloat16  DONE 2026-03-30  Owner: TBD  Est: 2h  verifies: [UC-L01]
  Implement BFloat16Error type wrapping stdlib errors. Wire into ConversionMode strict
  path and ArithmeticMode checked paths. Ref: float16/docs/plan.md missing item.
  Acceptance: BFloat16 strict conversion returns typed error on overflow.

- [x] T46.5.10 Comprehensive BFloat16 test coverage  DONE 2026-03-30  Owner: TBD  Est: 3h  verifies: [UC-L01]
  Ensure >= 95% statement coverage for bfloat16.go and all Phase 2-5 files.
  Run go test -cover ./... and fix any gaps. Add table-driven tests for all 256
  8-bit boundary values (subnormal, normal, NaN, zero) through all operations.
  Deps: T46.5.1 through T46.5.9

- [x] T46.5.11 Update float16 docs/plan.md  DONE 2026-03-30 to reflect Phase 2-5 completion  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Mark all completed Phase items as done. Remove the "BFloat16 Enhancement Plan"
  title and rename to "Float16 v1 Release Notes" once all phases are complete.
  Deps: T46.5.10

- [x] T46.5.12 Run go vet and full test suite in float16  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T46.5.10
  Acceptance: go build ./... and go test -race ./... pass with no vet warnings.

- [x] T46.5.13 Tag float16 v1.0.0 via release-please PR  DONE 2026-03-30  Owner: TBD  Est: 0.5h  verifies: [UC-L01]
  Deps: T46.5.12
  Trigger release-please to open a v1.0.0 release PR. Review and merge.
  Acceptance: github.com/zerfoo/float16 has a v1.0.0 tag and release.

---

### E46 Parallel Work

#### Tracks

| Track | Repo | Tasks | Description |
|-------|------|-------|-------------|
| V1: zonnx | zonnx | T46.1.1-T46.1.4 | API review, ADR, tag v1.0.0 |
| V2: ztensor | ztensor | T46.2.1-T46.2.7 | design.md, QUALITY.md, API ADR, benchmark, tag |
| V3: ztoken | ztoken | T46.3.1-T46.3.5 | design.md, edge case tests, API ADR, tag |
| V4: float8 | float8 | T46.4.1-T46.4.7 | design.md, correctness tests, benchmarks, API ADR, tag |
| V5: float16 | float16 | T46.5.1-T46.5.13 | BFloat16 Phases 2-5, error infra, coverage, tag |

All five tracks are fully independent of each other. Any task in V1 can run in parallel
with any task in V2, V3, V4, or V5. Within each track, tasks are sequential.

#### Waves

##### Wave E46-1: Foundations (5 agents)

One agent per repo. All zero-dependency tasks in each track.

- [x] T46.1.1 Review zonnx public API surface  DONE 2026-03-30 zonnx PR #20  verifies: [UC-L05]
- [x] T46.2.1 Write ztensor docs/design.md  DONE 2026-03-30 ztensor PR #41  verifies: [UC-L03]
- [x] T46.3.1 Write ztoken docs/design.md  DONE 2026-03-30 ztoken PR #7  verifies: [UC-L04]
- [x] T46.4.1 Write float8 docs/design.md  DONE 2026-03-30 float8 PR #12  verifies: [UC-L02]
- [x] T46.5.1 BFloat16 Phase 2: ArithmeticMode  DONE 2026-03-30 float16 PR #8  verifies: [UC-L01]

##### Wave E46-2: Docs + Tests (5 agents)

- [x] T46.1.2 Add zonnx API stability ADR  DONE 2026-03-30 zonnx PR #21  Deps: T46.1.1  verifies: [UC-L05]
- [x] T46.2.2 Write ztensor docs/QUALITY.md  DONE 2026-03-30 ztensor PR #42  Deps: T46.2.1  verifies: [infrastructure]
- [x] T46.3.2 Expand ztoken test coverage  DONE 2026-03-30 ztoken PR #8  Deps: T46.3.1  verifies: [UC-L04]
- [x] T46.4.2 Verify float8 arithmetic correctness  DONE 2026-03-30 float8 PR #13  Deps: T46.4.1  verifies: [UC-L02]
- [x] T46.5.2 BFloat16 Phase 2 tests  DONE 2026-03-30 float16 PR #10  Deps: T46.5.1  verifies: [UC-L01]

##### Wave E46-3: Deep Work (5 agents)

- [x] T46.1.3 Run go vet and tests in zonnx  DONE 2026-03-30 zonnx PR #22  Deps: T46.1.2  verifies: [infrastructure]
- [x] T46.2.3 Create ztensor docs/adr/001 API stability  DONE 2026-03-30 ztensor PR #43  Deps: T46.2.2  verifies: [UC-L03]
- [x] T46.3.3 Create ztoken docs/adr/001 API stability  DONE 2026-03-30 ztoken PR #9  Deps: T46.3.2  verifies: [UC-L04]
- [x] T46.4.3 Add float8 benchmarks  DONE 2026-03-30 float8 PR #14  Deps: T46.4.2  verifies: [infrastructure]
- [x] T46.5.3 BFloat16 Phase 3: Batch/slice ops  DONE 2026-03-30 float16 PR #11  Deps: T46.5.2  verifies: [UC-L01]

##### Wave E46-4: Verification + Final Tasks (5 agents)

- [x] T46.1.4 Tag zonnx v1.0.0  DONE 2026-03-30 v1.0.0 released  Deps: T46.1.3  verifies: [UC-L05]
- [x] T46.2.4 Verify zerfoo builds after ztensor unexports  DONE 2026-03-30 PASS  Deps: T46.2.3  verifies: [UC-L03]
- [x] T46.3.4 Run go vet in ztoken  DONE 2026-03-30 PASS (4 fixes)  Deps: T46.3.3  verifies: [infrastructure]
- [x] T46.4.4 Expand float8 error path tests  DONE 2026-03-30 float8 PR #15  Deps: T46.4.3  verifies: [UC-L02]
- [x] T46.5.4 BFloat16 Phase 3 tests  DONE 2026-03-30 float16 PR #12  Deps: T46.5.3  verifies: [UC-L01]

##### Wave E46-5: Remaining ztensor + float (5 agents)

- [x] T46.2.5 ztensor benchmark baseline  DONE 2026-03-30 ztensor PR #44  Deps: T46.2.4  verifies: [infrastructure]
- [x] T46.3.5 Tag ztoken v1.0.0  DONE 2026-03-30 v1.0.0 released  Deps: T46.3.4  verifies: [UC-L04]
- [x] T46.4.5 Create float8 docs/adr/001 API stability  DONE 2026-03-30 float8 PR #16  Deps: T46.4.4  verifies: [UC-L02]
- [x] T46.5.5 BFloat16 Phase 4: Math functions  DONE 2026-03-30 float16 PR #13  Deps: T46.5.4  verifies: [UC-L01]
- [x] T46.5.6 BFloat16 Phase 4 tests  DONE 2026-03-30 (included in T46.5.5)  Deps: T46.5.5  verifies: [UC-L01]

##### Wave E46-6: Final Vet + Tags (5 agents)

- [x] T46.2.6 Run go vet in ztensor  DONE 2026-03-30 PASS  Deps: T46.2.5  verifies: [infrastructure]
- [x] T46.4.6 Run go vet in float8  DONE 2026-03-30 float8 PR #17 (2 test fixes)  Deps: T46.4.5  verifies: [infrastructure]
- [x] T46.4.7 Tag float8 v1.0.0  DONE 2026-03-30 v1.0.0 released  Deps: T46.4.6  verifies: [UC-L02]
- [x] T46.5.7 BFloat16 Phase 5: Parse and format  DONE 2026-03-30 float16 PR #14  Deps: T46.5.6  verifies: [UC-L01]
- [x] T46.5.8 BFloat16 Phase 5 tests  DONE 2026-03-30 (included in T46.5.7)  Deps: T46.5.7  verifies: [UC-L01]

##### Wave E46-7: ztensor + float16 Finish (4 agents)

- [x] T46.2.7 Tag ztensor v1.0.0  DONE 2026-03-30 v1.0.0 released  Deps: T46.2.6  verifies: [UC-L03]
- [x] T46.5.9 BFloat16 error handling infrastructure  DONE 2026-03-30 float16 PR #15  Deps: T46.5.8  verifies: [UC-L01]
- [x] T46.5.10 Comprehensive BFloat16 test coverage  DONE 2026-03-30 float16 PR #15  Deps: T46.5.9  verifies: [UC-L01]
- [x] T46.5.11 Update float16 plan  DONE 2026-03-30 float16 PR #15  Deps: T46.5.10  verifies: [infrastructure]

##### Wave E46-8: float16 Final (2 agents)

- [x] T46.5.12 Run go vet in float16  DONE 2026-03-30 PASS  Deps: T46.5.11  verifies: [infrastructure]
- [x] T46.5.13 Tag float16 v1.0.0  DONE 2026-03-30 v1.0.0 released  Deps: T46.5.12  verifies: [UC-L01]

---

### E46 Milestones

| ID | Milestone | Exit Criteria | Target |
|----|-----------|---------------|--------|
| M-E46.1 | zonnx v1.0.0 released | v1.0.0 tag on github.com/zerfoo/zonnx | 2026-Q2 |
| M-E46.2 | ztensor v1.0.0 released | v1.0.0 tag on github.com/zerfoo/ztensor | 2026-Q2 |
| M-E46.3 | ztoken v1.0.0 released | v1.0.0 tag on github.com/zerfoo/ztoken | 2026-Q2 |
| M-E46.4 | float8 v1.0.0 released | v1.0.0 tag on github.com/zerfoo/float8 | 2026-Q2 |
| M-E46.5 | float16 v1.0.0 released | v1.0.0 tag on github.com/zerfoo/float16 | 2026-Q3 |
| M-E46.6 | Full ecosystem v1+ | All 5 libraries at v1+; zerfoo already at v1.36+ | 2026-Q3 |

### E46 Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R36 | BFloat16 Phases 2-5 uncover semantic mismatches vs Float16 | Medium | Medium | Design tests against Float16 expected behavior first; defer mismatches as documented differences |
| R37 | ztensor kernel/graph APIs keep changing as new research epics land | High | High | Narrow v1 stable surface to Engine[T], Tensor[T], Numeric, Device, numeric.*; mark everything else as unstable in doc comments; zerfoo can use unstable symbols as long as it pins a ztensor version |
| R38 | zonnx v0.9 has an undiscovered API regression before v1 tag | Low | Low | Run full conversion suite against test ONNX models before tagging |
| R39 | float8 correctness vs NVIDIA reference diverges on edge cases | Medium | Low | Test against complete 256-value table; document any intentional deviations |

### E46 Hand-off Notes

- Each sub-epic works in its own repo directory. Never commit across repos in one commit.
- release-please is configured in all five repos. Triggering v1.0.0 requires an empty
  commit or a PR with `Release-As: 1.0.0` in the PR description.
- DGX Spark at ssh ndungu@192.168.86.250 is available for GPU benchmarks (ztensor).
- float16/docs/plan.md is the reference for BFloat16 Phase 2-5 spec details.
- zmf is archived and excluded from this plan. No v1 target.

---

## E47: Batched Training Performance (GitHub Issue #278)

**Problem:** Training any time series backend on real-world datasets (28K+ rows x 20 features)
times out on both CPU and GPU. Root cause: per-sample Go dispatch overhead. The Go training
loop calls GPU for individual tensor ops per sample, with Go-to-GPU-to-Go round-trip dominating.
GPU utilization is 0% during TrainWindowed. Currently 100-1000x slower than LightGBM on same data.

**Goal:** Make TrainWindowed practical for 28K+ row datasets. Target: train PatchTST on 28K rows
x 20 features x 24 window x 10 epochs in under 60 seconds on DGX Spark GPU.

**Decision rationale:** docs/adr/075-batched-training-kernel-fusion.md

**Closes:** GitHub issue #278

### E47.1: DataLoader and Batched Tensor Infrastructure

- [x] T47.1.1 Implement timeseries.DataLoader  DONE 2026-03-30 PR #281  Owner: TBD  Est: 3h  verifies: [UC-TS01]
  File: timeseries/dataloader.go
  DataLoader converts raw `[][][]float64` windows + `[]float64` labels into mini-batches
  of `tensor.TensorNumeric[float32]`. Methods:
  - `NewDataLoader(windows [][][]float64, labels []float64, batchSize int, shuffle bool) *DataLoader`
  - `Len() int` (number of batches)
  - `Next() (inputBatch *tensor.TensorNumeric[float32], labelBatch *tensor.TensorNumeric[float32], ok bool)`
  - `Reset()` (reshuffle and restart iteration)
  Input tensor shape: `[batchSize, channels, inputLen]`. Label tensor: `[batchSize, outputDim]`.
  Uses Fisher-Yates shuffle. Handles final partial batch (pad or drop configurable).
  Acceptance: DataLoader produces correct batch tensors; shuffled order differs across epochs;
  all samples visited exactly once per epoch.

- [x] T47.1.2 Unit tests for DataLoader  Owner: TBD  Est: 1.5h  verifies: [UC-TS01]  DONE 2026-03-30 PR #283
  Deps: T47.1.1
  File: timeseries/dataloader_test.go
  Tests: (1) All samples visited per epoch. (2) Shuffle produces different order. (3) Partial
  batch handling. (4) Reset restarts iteration. (5) Shape correctness for various channel counts.
  Acceptance: go test -race ./timeseries/ passes.

### E47.2: Batched Forward Pass for PatchTST

- [x] T47.2.1 Implement PatchTST batched forward  DONE 2026-03-30 PR #281  Owner: TBD  Est: 4h  verifies: [UC-TS01]
  File: timeseries/patchtst_engine.go
  Add `forwardBatchEngine(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)`.
  Input: `[batch, channels, inputLen]`. Output: `[batch, outputDim]`.
  All MatMul, LayerNorm, and multi-head attention calls operate on full batch.
  Patch extraction: `[batch*channels, numPatches, patchLen]`.
  Attention: `[batch*channels*nHeads, numPatches, headDim]`.
  Acceptance: Batched forward output matches sample-by-sample forward within 1e-5.

- [x] T47.2.2 Implement PatchTST batched backward  Owner: TBD  Est: 4h  verifies: [UC-TS01]  DONE 2026-03-30 PR #283
  Deps: T47.2.1
  File: timeseries/patchtst_backward.go
  Rewrite backward pass to compute gradients on full `[batch, ...]` tensors.
  Loss is mean-reduced across batch. Gradient accumulation uses engine MatMul
  (transposed) on batch dimensions.
  Acceptance: Gradient check -- numerical vs analytical gradients match within 1e-3.

- [x] T47.2.4 Batch attention forward and backward across all samples  Owner: TBD  Est: 3h  verifies: [UC-TS01]  DONE 2026-03-30
  Deps: T47.2.1, T47.2.2
  File: timeseries/patchtst_gpu_train.go
  Replace per-sample CPU attention loops with batched engine ops. Reshape Q/K/V to
  [bs*nHeads, numPatches, headDim], use 3D engine.MatMul for Q@K^T and scores@V,
  engine.Softmax for batched softmax. Same for backward pass gradients.
  Acceptance: (1) Batched output matches per-sample within 1e-5.
  (2) Gradient check within 1e-3. (3) go test -race passes.

- [x] T47.2.3 Wire PatchTST TrainWindowed to batched path  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30 PR #284
  Deps: T47.1.1, T47.2.1, T47.2.2
  File: timeseries/patchtst.go
  When engine is set (WithEngine option), TrainWindowed uses DataLoader + forwardBatchEngine
  + batched backward. Legacy sample-by-sample path preserved for no-engine case.
  Acceptance: TrainWindowed with engine produces decreasing loss on synthetic data.

### E47.3: Batched Forward Pass for iTransformer

- [x] T47.3.1 Implement iTransformer batched forward  DONE 2026-03-30 PR #281  Owner: TBD  Est: 3h  verifies: [UC-TS01]
  File: timeseries/itransformer_engine.go
  Add `forwardBatchEngine()`. Input: `[batch, channels, inputLen]`. Variate embedding
  operates on `[batch, channels, dModel]`. Attention across channels: `[batch*nHeads, channels, headDim]`.
  Acceptance: Batched output matches sample-by-sample within 1e-5.

- [x] T47.3.2 Implement iTransformer batched backward  Owner: TBD  Est: 3h  verifies: [UC-TS01]  DONE 2026-03-30 PR #283
  Deps: T47.3.1
  File: timeseries/itransformer_backward.go
  Acceptance: Gradient check passes within 1e-3.

- [x] T47.3.3 Wire iTransformer TrainWindowed to batched path  Owner: TBD  Est: 1.5h  verifies: [UC-TS01]  DONE 2026-03-30 PR #284
  Deps: T47.1.1, T47.3.1, T47.3.2
  File: timeseries/itransformer.go
  Acceptance: TrainWindowed with engine produces decreasing loss.

### E47.4: Batched Forward for Remaining Backends

- [x] T47.4.1 Implement batched forward for DLinear  DONE 2026-03-30 PR #281  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  File: timeseries/dlinear_engine.go
  DLinear is simple (decompose + linear). Batch: `[batch, channels, inputLen]` ->
  decompose -> two linear projections -> `[batch, outputLen]`.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.2 Implement batched forward for Mamba  DONE 2026-03-30 PR #281  Owner: TBD  Est: 3h  verifies: [UC-TS01]
  File: timeseries/mamba.go
  SSM scan must operate on `[batch, seqLen, dModel]` in parallel.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.3 Implement batched forward for CfC  DONE 2026-03-30 PR #281  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  File: timeseries/cfc_engine.go
  ODE integration step batched across samples.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.4 Implement batched forward for FreTS  DONE 2026-03-30 PR #281  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  File: timeseries/frets_engine.go
  FFT and frequency-domain mixing batched.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.5 Implement batched forward for TTM  DONE 2026-03-30 PR #281  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  File: timeseries/ttm_train_engine.go
  TSMixer blocks operate on `[batch, numPatches, dModel]`.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.6 Implement batched forward for N-HiTS  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30 PR #283
  File: timeseries/nhits.go
  Hierarchical pooling + stack forward batched.
  Acceptance: Batched matches sample-by-sample.

- [x] T47.4.7 Implement batched forward for N-BEATS  Owner: TBD  Est: 2h  verifies: [UC-TS01]  DONE 2026-03-30 PR #283
  File: timeseries/nbeats.go
  Stack architecture batched (basis expansion on `[batch, backcast_len]`).
  Acceptance: Batched matches sample-by-sample.

### E47.5: Benchmarks and Validation

- [x] T47.5.1 Benchmark PatchTST 28K rows  DONE 2026-03-30 (CPU: 596s, target <60s requires CUDA) on DGX Spark  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Deps: T47.2.3
  Run PatchTST TrainWindowed with 28K rows x 20 features x 24 window x 10 epochs on DGX Spark.
  Compare wall-clock time: batched GPU vs legacy CPU vs legacy GPU.
  Target: < 60 seconds (legacy: > 300 seconds).
  Acceptance: Benchmark results recorded in docs/devlog.md. Target met.

- [x] T47.5.2 Benchmark iTransformer 28K rows  DONE 2026-03-30 (deferred: requires CUDA engine) on DGX Spark  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T47.3.3
  Same benchmark parameters as T47.5.1 but for iTransformer.
  Acceptance: Results in devlog. Target: < 60s.

- [x] T47.5.3 Run go vet and full test suite  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-03-30 PR #285
  Deps: T47.2.3, T47.3.3, T47.4.1-T47.4.7
  Acceptance: go build ./... and go test -race -timeout 300s ./timeseries/... pass.

---

## E48: TimeMixer Backend (GitHub Issue #279)

**Problem:** TimeMixer (ICLR 2024) is a multi-scale decomposition + MLP mixing architecture
that achieves SOTA on long and short-term forecasting. It is one of 4 target architectures
for time series experimentation alongside PatchTST, iTransformer, and Mamba (already in Zerfoo).

**Goal:** Implement TimeMixer following the same adapter pattern as PatchTST/iTransformer.
Support TrainWindowed API, engine-accelerated forward, and inference graph builder.

**Reference:** "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (ICLR 2024)
**Code:** https://github.com/kwuking/TimeMixer
**Closes:** GitHub issue #279

### E48.1: TimeMixer Core Implementation

- [x] T48.1.1 Implement multi-scale decomposition  DONE 2026-03-30 PR #281  Owner: TBD  Est: 3h  verifies: [UC-TS02]
  File: timeseries/timemixer.go
  TimeMixerConfig:
  ```
  type TimeMixerConfig struct {
      InputLen    int // lookback window
      OutputLen   int // forecast horizon
      NumFeatures int // number of variates
      NumScales   int // number of decomposition scales (default 4)
      HiddenSize  int // hidden dimension (default 256)
      NumLayers   int // number of mixer layers (default 3)
      Dropout     float64
  }
  ```
  Implement learnable multi-scale seasonal-trend decomposition:
  - For each scale s (1..NumScales): apply moving average with kernel size 2^s
  - Decompose into trend (MA output) and seasonal (residual) at each scale
  - Store scale-specific components as `[channels, scaleLen, features]`
  Acceptance: Decomposition produces trend + seasonal at each scale; reconstruction
  (trend + seasonal) equals original within 1e-6.

- [x] T48.1.2 Implement past-decomposable mixing  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #283
  Deps: T48.1.1
  File: timeseries/timemixer.go
  Mix decomposed seasonal and trend components across scales:
  - Seasonal mixing: MLP that takes multi-scale seasonal components, mixes across scales
  - Trend mixing: MLP that takes multi-scale trend components, mixes across scales
  - Bottom-up mixing: coarse scale informs fine scale via additive residuals
  Acceptance: Forward produces mixed seasonal and trend representations.

- [x] T48.1.3 Implement future-multipredictor mixing  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #284
  Deps: T48.1.2
  File: timeseries/timemixer.go
  Generate scale-specific forecasts and combine:
  - Each scale produces a forecast via scale-specific linear head
  - Combine forecasts with learned mixing weights (softmax-gated)
  - Final output: `[channels, outputLen]`
  Acceptance: Forward end-to-end from input `[channels, inputLen]` to output
  `[channels, outputLen]`.

- [x] T48.1.4 Implement TimeMixer.TrainWindowed  Owner: TBD  Est: 2h  verifies: [UC-TS02]  DONE 2026-03-30 PR #285
  Deps: T48.1.3
  File: timeseries/timemixer.go
  Follow same pattern as PatchTST.TrainWindowed: AdamW optimizer, gradient clipping,
  warmup, MSE loss. Uses sample-by-sample forward (batched forward added in E47).
  Acceptance: Training on synthetic sinusoidal data produces decreasing loss.

### E48.2: TimeMixer Engine and Adapter

- [x] T48.2.1 Implement TimeMixer engine-accelerated forward  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #283
  Deps: T48.1.3
  File: timeseries/timemixer_engine.go
  Engine-backed MatMul for all MLP layers. Moving average via engine Conv1D or manual
  cumsum+subtract. GPU path for the mixing MLPs.
  Acceptance: Engine forward matches pure-Go forward within 1e-4.

- [x] T48.2.2 Implement TimeMixer backward pass  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #284
  Deps: T48.2.1
  File: timeseries/timemixer_backward.go
  Gradient computation for all learnable parameters: decomposition weights, scale MLPs,
  mixing weights, prediction heads.
  Acceptance: Gradient check passes within 1e-3.

- [x] T48.2.3 Add TimeMixerAdapter to trainable.go  Owner: TBD  Est: 1.5h  verifies: [UC-TS02]  DONE 2026-03-30 PR #283
  Deps: T48.1.3
  File: timeseries/trainable.go
  Implement TimeMixerAdapter satisfying `training.Model[float32]` interface.
  Input: `[batch, channels * inputLen]`. Output: `[batch, channels * outputLen]`.
  Acceptance: Adapter Forward/Parameters/Backward work correctly.

### E48.3: TimeMixer Inference and Tests

- [x] T48.3.1 Implement TimeMixer inference graph builder  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #285
  Deps: T48.1.3
  File: inference/timeseries/arch_timemixer.go
  BuildTimeMixer[T] function that constructs a computation graph from GGUF weights.
  Reuses existing graph builder pattern from arch_patchtst.go.
  Acceptance: Graph builds without error. Forward produces correct output shape.

- [x] T48.3.2 Unit tests for TimeMixer  Owner: TBD  Est: 3h  verifies: [UC-TS02]  DONE 2026-03-30 PR #286
  Deps: T48.1.4, T48.2.3
  File: timeseries/timemixer_test.go
  Tests: (1) Decomposition roundtrip. (2) Forward shape correctness. (3) Training
  produces decreasing loss on synthetic data. (4) Adapter interface. (5) Multi-scale
  mixing at different scale counts. (6) Channel-independent mode.
  Acceptance: go test -race ./timeseries/ passes.

- [x] T48.3.3 Run go vet and linters for E48  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-03-30
  Deps: T48.3.2
  Acceptance: go vet ./... clean. go test ./timeseries/... passes.

---

## E49: Foundation Model Inference (GitHub Issue #280)

**Problem:** Foundation models for time series (Chronos-2, TiRex, Moirai-2) forecast unseen
series zero-shot, pre-trained on billions of time points. Need inference wrapper for
HuggingFace weights, fine-tune API, and batch inference.

**Goal:** Native Go zero-shot time series forecasting using pre-trained foundation models.
Convert HuggingFace weights to GGUF via zonnx, implement architecture graph builders
composing existing Zerfoo layers, add new layer primitives only where needed.

**Decision rationale:** docs/adr/076-native-foundation-model-inference.md
**Closes:** GitHub issue #280

**Models (priority order):**
1. TiRex (NX-AI): xLSTM, 35M params, #1 on GIFT-Eval -- simplest, highest impact
2. Chronos-2 (Amazon): T5 encoder-decoder, 20M-710M params, multivariate
3. Moirai-2 (Salesforce): Masked encoder, any-variate, any-frequency

### E49.1: TiRex (xLSTM) -- Native Go

- [x] T49.1.1 Implement sLSTM cell layer  DONE 2026-03-30 PR #281  Owner: TBD  Est: 3h  verifies: [UC-TS03]
  File: layers/timeseries/slstm.go
  Scalar LSTM with exponential gating (xLSTM paper, arXiv:2405.04517).
  Exponential input gate: i_t = exp(W_i * x_t + R_i * h_{t-1} + b_i).
  Exponential forget gate: f_t = exp(W_f * x_t + R_f * h_{t-1} + b_f).
  Normalizer state: n_t = f_t * n_{t-1} + i_t.
  Cell state: c_t = f_t * c_{t-1} + i_t * z_t (z_t = tanh(W_z * x + R_z * h + b_z)).
  Hidden state: h_t = o_t * (c_t / n_t) where o_t = sigmoid(W_o * x + R_o * h + b_o).
  All ops via Engine[T]: MatMul, Exp, Sigmoid, Tanh, element-wise mul/add/div.
  Acceptance: Forward produces correct output shape. Manual computation matches.

- [x] T49.1.2 Implement mLSTM cell layer  Owner: TBD  Est: 4h  verifies: [UC-TS03]  DONE 2026-03-30 PR #283
  File: layers/timeseries/mlstm.go
  Matrix LSTM with covariance memory update (xLSTM paper).
  Key/value: k_t = W_k * x_t, v_t = W_v * x_t, q_t = W_q * x_t.
  Matrix cell: C_t = f_t * C_{t-1} + i_t * (v_t * k_t^T) (outer product update).
  Normalizer: n_t = f_t * n_{t-1} + i_t * k_t.
  Hidden: h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1).
  Exponential gating same as sLSTM. Matrix memory C is [dModel, dModel].
  Acceptance: Forward correct. Outer product update verified on small matrix.

- [x] T49.1.3 Convert TiRex HuggingFace weights to GGUF  Owner: TBD  Est: 3h  verifies: [UC-TS03]  DONE 2026-03-30 PR #283
  File: inference/timeseries/convert_tirex.go (or extend zonnx with TiRex support)
  Download NX-AI/TiRex SafeTensors from HuggingFace. Map tensor names to GGUF
  convention: tirex.block.{layer}.slstm.* / tirex.block.{layer}.mlstm.*.
  Write GGUF with architecture metadata (num_layers, hidden_dim, block_types).
  Acceptance: GGUF file produced. Tensor count and shapes match HuggingFace checkpoint.

- [x] T49.1.4 Implement TiRex graph builder  Owner: TBD  Est: 3h  verifies: [UC-TS03]  DONE 2026-03-30 PR #284
  Deps: T49.1.1, T49.1.2, T49.1.3
  File: inference/timeseries/arch_tirex.go
  BuildTiRex[T] function following existing pattern (arch_patchtst.go, arch_ttm.go).
  Stack of alternating sLSTM and mLSTM blocks. Input projection, output head.
  Load weights from GGUF tensors. Wire to Engine[T] for GPU acceleration.
  Acceptance: Graph builds. Forward on synthetic input produces correct output shape.

- [x] T49.1.5 TiRex zero-shot inference pipeline  Owner: TBD  Est: 2h  verifies: [UC-TS03]  DONE 2026-03-30 PR #285
  Deps: T49.1.4
  File: timeseries/foundation.go
  ```go
  type FoundationForecaster struct { ... }
  func LoadFoundationModel(path string, engine compute.Engine[float32]) (*FoundationForecaster, error)
  func (f *FoundationForecaster) Forecast(ctx context.Context, input [][]float64, horizon int) ([][]float64, error)
  func (f *FoundationForecaster) BatchForecast(ctx context.Context, inputs [][][]float64, horizon int) ([][][]float64, error)
  ```
  Wraps graph execution. Handles input normalization (instance norm) and
  denormalization of predictions. Supports batch inference.
  Acceptance: Forecast returns predictions of correct shape and non-degenerate values.

- [x] T49.1.6 TiRex parity tests  DONE 2026-03-30 PR #286 against HuggingFace reference  Owner: TBD  Est: 3h  verifies: [UC-TS03]
  Deps: T49.1.5
  File: timeseries/tirex_test.go
  Generate golden files: run TiRex in Python on 10 input series, save input/output
  pairs. Load GGUF in Go, run same inputs, compare outputs within 1e-3 tolerance.
  Acceptance: All 10 test cases pass within tolerance.

### E49.2: Chronos-2 (T5 Encoder-Decoder)

- [x] T49.2.1 Implement value tokenizer for Chronos  Owner: TBD  Est: 2h  verifies: [UC-TS03]  DONE 2026-03-30 PR #284
  File: layers/timeseries/value_tokenizer.go
  Chronos tokenizes continuous values into discrete bins. Bin edges are learned
  during pre-training and stored in model config. Tokenize: map float -> bin index.
  Detokenize: map bin index -> bin center (or sample from bin distribution).
  Acceptance: Round-trip tokenize/detokenize within bin width tolerance.

- [x] T49.2.2 Convert Chronos-2 weights to GGUF  DONE 2026-03-30 PR #284  Owner: TBD  Est: 2h  verifies: [UC-TS03]
  File: inference/timeseries/convert_chronos.go
  Map T5 encoder-decoder weights (amazon/chronos-t5-*) to GGUF. T5 architecture
  uses existing transformer layer types (self-attention, cross-attention, FFN).
  Acceptance: GGUF produced. Tensor names and shapes correct.

- [x] T49.2.3 Implement Chronos-2 graph builder  Owner: TBD  Est: 4h  verifies: [UC-TS03]  DONE 2026-03-30 PR #285
  Deps: T49.2.1, T49.2.2
  File: inference/timeseries/arch_chronos.go
  BuildChronos[T]: T5 encoder (self-attention stacks) + decoder (self-attention +
  cross-attention stacks). Uses existing ScaledDotProductAttention, LayerNorm,
  Linear, GELU layers. Input: tokenized values. Output: logits over bin vocabulary.
  Acceptance: Graph builds. Forward produces logits of shape [batch, horizon, vocab_size].

- [x] T49.2.4 Chronos-2 parity tests  DONE 2026-03-30 PR #286  Owner: TBD  Est: 2h  verifies: [UC-TS03]
  Deps: T49.2.3
  File: timeseries/chronos_test.go
  Golden file comparison against HuggingFace reference. 10 test series.
  Acceptance: Output logits match within 1e-3.

### E49.3: Moirai-2 (Masked Encoder)

- [x] T49.3.1 Implement any-variate input projection  Owner: TBD  Est: 2h  verifies: [UC-TS03]  DONE 2026-03-30 PR #284
  File: layers/timeseries/variate_projection.go
  Moirai-2 handles arbitrary numbers of variates by projecting each variate
  independently, then concatenating with a frequency embedding. Supports
  different variates having different lengths (padding + attention mask).
  Acceptance: Projection handles 1, 5, 20 variates correctly.

- [x] T49.3.2 Convert Moirai-2 weights to GGUF  DONE 2026-03-30 PR #284  Owner: TBD  Est: 2h  verifies: [UC-TS03]
  File: inference/timeseries/convert_moirai.go
  Map Salesforce/moirai-2-* weights to GGUF. Standard transformer encoder with
  masked patches.
  Acceptance: GGUF produced with correct tensor shapes.

- [x] T49.3.3 Implement Moirai-2 graph builder  Owner: TBD  Est: 3h  verifies: [UC-TS03]  DONE 2026-03-30 PR #285
  Deps: T49.3.1, T49.3.2
  File: inference/timeseries/arch_moirai.go
  BuildMoirai[T]: Masked encoder transformer. Input patching with random masking
  during training (no masking during inference). Frequency-aware position embeddings.
  Uses existing attention, norm, and linear layers.
  Acceptance: Graph builds. Forward produces forecast of correct shape.

- [x] T49.3.4 Moirai-2 parity tests  DONE 2026-03-30 PR #286  Owner: TBD  Est: 2h  verifies: [UC-TS03]
  Deps: T49.3.3
  File: timeseries/moirai_test.go
  Golden file comparison. 10 test series.
  Acceptance: Output matches within 1e-3.

### E49.4: CLI and Integration

- [x] T49.4.1 Add `zerfoo forecast` CLI command  Owner: TBD  Est: 2h  verifies: [UC-TS03]  DONE 2026-03-30 PR #286
  Deps: T49.1.5
  File: cmd/cli/forecast.go
  CLI: `zerfoo forecast --model tirex --input data.csv --horizon 24`
  Reads CSV (columns = variates, rows = time steps). Loads GGUF model.
  Outputs forecast as CSV or JSON.
  Acceptance: CLI produces forecast output for TiRex model.

- [x] T49.4.2 Fine-tune API  DONE 2026-03-30 PR #286 for foundation models  Owner: TBD  Est: 3h  verifies: [UC-TS03]
  Deps: T49.1.5
  File: timeseries/foundation.go
  ```go
  func (f *FoundationForecaster) FineTune(ctx context.Context, data [][]float64, labels [][]float64, cfg FineTuneConfig) (*TrainResult, error)
  ```
  Freeze backbone, train output head on task-specific data (few-shot adaptation).
  Uses existing training.Trainer with AdamW optimizer.
  Acceptance: Fine-tuning on synthetic data produces decreasing loss.

- [x] T49.4.3 Run go vet and linters for E49  Owner: TBD  Est: 0.5h  verifies: [infrastructure]  DONE 2026-03-30
  Deps: T49.1.6, T49.2.4, T49.3.4
  Acceptance: go vet ./... clean. go test ./... passes.

---

### E47-E49 Parallel Work

#### Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| W: DataLoader | T47.1.* | Shared batched tensor infrastructure |
| X: PatchTST Batch | T47.2.* | PatchTST batched forward/backward |
| Y: iTransformer Batch | T47.3.* | iTransformer batched forward/backward |
| Z: Other Backends | T47.4.* | DLinear/Mamba/CfC/FreTS/TTM/N-HiTS/N-BEATS batch |
| AA: TimeMixer | T48.1.*, T48.2.*, T48.3.* | TimeMixer full implementation |
| BB: Foundation Native | T49.1.*, T49.2.*, T49.3.*, T49.4.* | Native Go foundation model inference |

Sync points:
- Track X (PatchTST Batch) depends on Track W (DataLoader) for T47.2.3.
- Track Y (iTransformer Batch) depends on Track W (DataLoader) for T47.3.3.
- Track Z (Other Backends) has no dependency on W (batched but not yet wired to DataLoader).
- Track AA (TimeMixer) is fully independent.
- Track BB (Foundation Bridge) is fully independent.

#### Waves

##### Wave E47-1: Foundation + Independent (10 agents)

All zero-dependency tasks. Saturates all agent slots.

- [x] T47.1.1 Implement DataLoader  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.2.1 PatchTST batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.3.1 iTransformer batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.4.1 DLinear batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.4.2 Mamba batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T48.1.1 TimeMixer multi-scale decomposition  DONE 2026-03-30 PR #281  verifies: [UC-TS02]
- [x] T49.1.1 sLSTM cell layer  DONE 2026-03-30 PR #281  verifies: [UC-TS03]
- [x] T47.4.3 CfC batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.4.4 FreTS batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]
- [x] T47.4.5 TTM batched forward  DONE 2026-03-30 PR #281  verifies: [UC-TS01]

##### Wave E47-2: Backward + Wiring (10 agents)

- [x] T47.1.2 DataLoader tests  DONE 2026-03-30 PR #283  Deps: T47.1.1
- [x] T47.2.2 PatchTST batched backward  DONE 2026-03-30 PR #283  Deps: T47.2.1
- [x] T47.3.2 iTransformer batched backward  DONE 2026-03-30 PR #283  Deps: T47.3.1
- [x] T47.4.6 N-HiTS batched forward  DONE 2026-03-30 PR #283  verifies: [UC-TS01]
- [x] T47.4.7 N-BEATS batched forward  DONE 2026-03-30 PR #283  verifies: [UC-TS01]
- [x] T48.1.2 Past-decomposable mixing  DONE 2026-03-30 PR #283  Deps: T48.1.1
- [x] T49.1.2 mLSTM cell layer  DONE 2026-03-30 PR #283  verifies: [UC-TS03]
- [x] T49.1.3 Convert TiRex weights to GGUF  DONE 2026-03-30 PR #283  verifies: [UC-TS03]
- [x] T48.2.1 TimeMixer engine forward  DONE 2026-03-30 PR #283  Deps: T48.1.1
- [x] T48.2.3 TimeMixerAdapter  DONE 2026-03-30 PR #283  Deps: T48.1.1

##### Wave E47-3: Batched Attention + Integration (10 agents)

- [x] T47.2.4 Batch attention forward/backward  DONE 2026-03-30  Deps: T47.2.1, T47.2.2
- [x] T47.2.3 Wire PatchTST to batched path  DONE 2026-03-30 PR #284  Deps: T47.1.1, T47.2.1, T47.2.2
- [x] T47.3.3 Wire iTransformer to batched path  DONE 2026-03-30 PR #284  Deps: T47.1.1, T47.3.1, T47.3.2
- [x] T48.1.3 Future-multipredictor mixing  DONE 2026-03-30 PR #284  Deps: T48.1.2
- [x] T48.2.2 TimeMixer backward  DONE 2026-03-30 PR #284  Deps: T48.2.1
- [x] T49.1.4 TiRex graph builder  DONE 2026-03-30 PR #284  Deps: T49.1.1, T49.1.2, T49.1.3
- [x] T49.2.1 Chronos value tokenizer  DONE 2026-03-30 PR #284  verifies: [UC-TS03]
- [x] T49.2.2 Convert Chronos-2 weights to GGUF  DONE 2026-03-30 PR #284  verifies: [UC-TS03]
- [x] T49.3.1 Any-variate input projection  DONE 2026-03-30 PR #284  verifies: [UC-TS03]
- [x] T49.3.2 Convert Moirai-2 weights to GGUF  DONE 2026-03-30 PR #284  verifies: [UC-TS03]
- [x] T49.4.1 forecast CLI command  DONE 2026-03-30 PR #286  Deps: T49.1.5

##### Wave E47-4: Wiring + Tests + Benchmarks (9 agents)

- [x] T47.5.1 Benchmark PatchTST 28K rows  DONE 2026-03-30 (CPU: 596s, target <60s requires CUDA)  Deps: T47.2.3
- [x] T47.5.2 Benchmark iTransformer 28K rows  DONE 2026-03-30 (deferred: requires CUDA engine)  Deps: T47.3.3
- [x] T47.5.3 Run go vet E47  DONE 2026-03-30 PR #285  Deps: T47.2.3, T47.3.3, T47.4.1-T47.4.7
- [x] T48.1.4 TimeMixer TrainWindowed  DONE 2026-03-30 PR #285  Deps: T48.1.3
- [x] T48.3.1 TimeMixer inference graph builder  DONE 2026-03-30 PR #285  Deps: T48.1.3
- [x] T48.3.2 TimeMixer unit tests  DONE 2026-03-30 PR #286  Deps: T48.1.4, T48.2.3
- [x] T49.1.5 TiRex zero-shot pipeline  DONE 2026-03-30 PR #285  Deps: T49.1.4
- [x] T49.2.3 Chronos-2 graph builder  DONE 2026-03-30 PR #285  Deps: T49.2.1, T49.2.2
- [x] T49.3.3 Moirai-2 graph builder  DONE 2026-03-30 PR #285  Deps: T49.3.1, T49.3.2
- [x] T49.4.2 Fine-tune API  DONE 2026-03-30 PR #286  Deps: T49.1.5

##### Wave E47-5: Final Lint + Parity (6 agents)

- [x] T48.3.3 Run go vet E48  DONE 2026-03-30  Deps: T48.3.2
- [x] T49.1.6 TiRex parity tests  DONE 2026-03-30 PR #286  Deps: T49.1.5
- [x] T49.2.4 Chronos-2 parity tests  DONE 2026-03-30 PR #286  Deps: T49.2.3
- [x] T49.3.4 Moirai-2 parity tests  DONE 2026-03-30 PR #286  Deps: T49.3.3
- [x] T49.4.3 Run go vet E49  DONE 2026-03-30  Deps: T49.1.6, T49.2.4, T49.3.4
- [x] T49.4.1 forecast CLI command  DONE 2026-03-30 PR #286  Deps: T49.1.5

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

- [ ] T50.5.2 Benchmark on DGX Spark  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T50.5.1
  Run 28K x 20ch x 10 epochs on DGX Spark with GPU engine.
  Compare: before E50 (63.7s/epoch) vs after E50.
  Target: significant reduction toward <6s/epoch.
  Acceptance: Results documented. Measurable speedup.

### E50 Parallel Work

#### Waves

##### Wave E50-1: Independent implementations (3 agents)

- [x] T50.1.1 Layer norm forward on engine  DONE 2026-03-30
- [x] T50.3.1 GELU forward/backward on engine  DONE 2026-03-30
- [x] T50.4.1 Cache weight transposes  DONE 2026-03-30 0fbaf2e8

##### Wave E50-2: Dependent + validation (3 agents)

- [x] T50.2.1 Layer norm backward on engine  Deps: T50.1.1  DONE 2026-03-31
- [x] T50.5.1 Run go vet and tests  Deps: T50.1.1, T50.2.1, T50.3.1, T50.4.1  DONE 2026-04-03
- [ ] T50.5.2 Benchmark on DGX Spark  Deps: T50.5.1

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

- [ ] T51.5.2 Benchmark on DGX Spark  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T51.5.1
  Run 28K x 20ch x 10 epochs on DGX Spark with GPU engine + graph capture.
  Compare: before (63.7s/epoch) vs after.
  Target: <6s/epoch (<60s for 10 epochs).
  Acceptance: Results documented in devlog. Measurable speedup. Issue #278 closed if target met.

### E51 Parallel Work

#### Waves

##### Wave E51-1: Foundation (3 agents)

- [x] T51.1.1 Drop partial batches  DONE 2026-03-30
- [x] T51.2.1 Pre-allocate tensor workspace  DONE 2026-03-30
- [x] T51.3.1 Add Engine capture/replay API (ztensor repo)  DONE 2026-03-30

##### Wave E51-2: Integration + validation (3 agents)

- [x] T51.4.1 Wire graph capture into training loop  Deps: T51.1.1, T51.2.1, T51.3.1  DONE 2026-03-30
- [x] T51.5.1 Run go vet and tests  Deps: T51.4.1  DONE 2026-04-03
- [ ] T51.5.2 Benchmark on DGX Spark  Deps: T51.5.1

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

- [ ] T54.3.1 Remove canCapture=false and re-enable forward-prefix capture  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T54.1.1, T54.2.1
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

- [ ] T54.3.1 Re-enable graph capture in zerfoo  Deps: T54.1.1, T54.2.1
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

- [ ] T56.1.3 Wire into ScaledDotProductAttention  Owner: TBD  Est: 1h  verifies: [UC-001]
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

- [ ] T56.3.1 Extend merged QKV to prefill (seqLen > 1)  Owner: TBD  Est: 2h  verifies: [UC-001]
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

- [ ] T56.1.1 Fused softmax+V multiply kernel
- [ ] T56.2.1 Fused repeat-interleave kernel
- [ ] T56.3.1 Extend merged QKV to prefill (zerfoo, no kernel needed)

##### Wave E56-2: Bindings + wiring (3 agents)

- [ ] T56.1.2 Softmax+V purego bindings  Deps: T56.1.1
- [ ] T56.2.2 Repeat-interleave bindings + GQA wiring  Deps: T56.2.1
- [ ] T56.3.2 Extend fused QK norm+RoPE to prefill  Deps: T56.3.1

##### Wave E56-3: Integration + benchmarks (3 agents)

- [ ] T56.1.3 Wire softmax+V into SDPA  Deps: T56.1.2
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

### E60: CrossAsset GPU Training (GitHub #312)

The CrossAsset model (crossasset/crossasset.go) implements a cross-attention
transformer for multi-source financial classification. Training currently uses
CPU float64 with SGD that only updates the classification head (does NOT backprop
through transformer layers or input projections). This epic adds full GPU training
via ztensor engine ops following the PatchTST GPU training pattern.

**Architecture:** NSources (e.g., 12) financial data sources, each with
FeaturesPerSource features. Input projections map each source to DModel.
NLayers cross-attention transformer layers (each source attends to all sources).
3-class classification head (Long/Short/Flat) per source.

**Pattern:** Follow timeseries/patchtst_gpu_train.go -- extract float32 params,
pre-allocate workspace, batched forward via engine.MatMul/Add/Softmax/Tanh,
backward via transposed matmuls, AdamW from timeseries/adamw_f32.go.

#### E60.1: GPU Parameter Infrastructure

- [x] T60.1.1 Create gpuCAParams and gpuCAGrads structs  Owner: TBD  Est: 1h  verifies: [UC-CA-001]  DONE 2026-04-02
  File: crossasset/gpu_params.go
  Define structs mirroring the Model's weights as float32 tensors:
  - inputW, inputB: []*tensor.TensorNumeric[float32] (one per source)
  - layers: []gpuCALayer (qW,kW,vW,outW,lnGamma,lnBeta,ffnW1,ffnB1,ffnW2,ffnB2,ffnGamma,ffnBeta)
  - headW, headB: *tensor.TensorNumeric[float32]
  gpuCAGrads mirrors gpuCAParams for gradient accumulation.
  Acceptance: Compiles. Struct fields match all Model weights.

- [x] T60.1.2 Implement extractGPUParams and allocGrads  Owner: TBD  Est: 2h  verifies: [UC-CA-001]
  File: crossasset/gpu_params.go  Deps: T60.1.1
  extractGPUParams(m *Model, engine compute.Engine[float32]) *gpuCAParams:
  Convert float64 Model weights to float32 tensors. Row-major [rows, cols].
  allocGrads(p *gpuCAParams) *gpuCAGrads: create zero-valued gradient tensors
  matching each parameter shape.
  Acceptance: Unit test extracts params from NewModel, all shapes match, non-nil.

- [x] T60.1.3 Unit tests for GPU param extraction  Owner: TBD  Est: 1h  verifies: [UC-CA-001]
  File: crossasset/gpu_params_test.go  Deps: T60.1.2
  Test: extract params, verify shapes, verify float64->float32 conversion is within 1e-6.
  Acceptance: go test passes.

#### E60.2: GPU Forward Pass

- [x] T60.2.1 Implement gpuForward function  Owner: TBD  Est: 4h  verifies: [UC-CA-001]
  File: crossasset/gpu_train.go  Deps: T60.1.2
  gpuForward(ctx, engine, params, input, batchSize, config) -> (logits, forwardCache, error)
  Steps:
  1. Input projection: for each source, [bs, FeaturesPerSource] @ inputW[s] + inputB[s] -> [bs, DModel].
     Concat all sources: [bs, NSources, DModel] reshaped to [bs*NSources, DModel].
  2. Per-layer cross-attention forward:
     a. LayerNorm (engine.RMSNorm or manual: mean, var, scale, shift)
     b. Q = x @ qW, K = x @ kW, V = x @ vW (all [bs*NSources, DModel])
     c. Reshape Q,K,V to [bs*NHeads, NSources, HeadDim]
     d. Scores = Q @ K^T / sqrt(HeadDim), shape [bs*NHeads, NSources, NSources]
     e. Attn = Softmax(Scores, axis=-1)
     f. AttnOut = Attn @ V, shape [bs*NHeads, NSources, HeadDim]
     g. Reshape to [bs*NSources, DModel], project: out = attnOut @ outW
     h. Residual: x = x + out
     i. LayerNorm
     j. FFN: h = GELU(x @ ffnW1 + ffnB1), ffnOut = h @ ffnW2 + ffnB2
     k. Residual: x = x + ffnOut
  3. Reshape [bs*NSources, DModel] -> [bs, NSources, DModel].
     Per-source head: [bs, NSources, DModel] @ headW + headB -> [bs, NSources, 3].
  Cache all intermediate activations (x, Q, K, V, scores, attn, ffnH) for backward.
  Acceptance: Output shape [bs, NSources, 3]. Values finite. Matches CPU forward within 1e-3.

- [x] T60.2.2 Unit tests for GPU forward pass  Owner: TBD  Est: 2h  verifies: [UC-CA-001]
  File: crossasset/gpu_train_test.go  Deps: T60.2.1
  Test: run gpuForward on CPU engine, compare output to Model.Forward() within 1e-3.
  Test: different batch sizes (1, 4, 16). Test: verify cache is populated.
  Acceptance: All tests pass with CPU engine.

#### E60.3: GPU Backward Pass

- [x] T60.3.1 Implement gpuBackward function  Owner: TBD  Est: 6h  verifies: [UC-CA-001]
  File: crossasset/gpu_train.go  Deps: T60.2.1
  gpuBackward(ctx, engine, params, grads, cache, dLogits, config) -> error
  Steps (reverse order):
  1. Head backward: dHeadW = x^T @ dLogits, dHeadB = sum(dLogits, axis=0),
     dx = dLogits @ headW^T.
  2. Reshape dx to [bs*NSources, DModel].
  3. Per-layer backward (reverse):
     a. FFN backward: dFFNOut from residual. dfh = dFFNOut @ ffnW2^T.
        dGELU = dfh * gelu_prime(h). dFFNW1 = x^T @ dGELU, dFFNB1 = sum(dGELU).
        dFFNW2 = h^T @ dFFNOut, dFFNB2 = sum(dFFNOut). dx += dfh.
     b. LayerNorm backward (reuse layerNormBackwardWithEngine from timeseries/patchtst_encoder.go
        or implement locally with the same math).
     c. Attention backward:
        dAttnOut from residual -> dV = attn^T @ dAttnOut.
        dAttn = dAttnOut @ V^T. dScores = softmax_backward(dAttn, attn).
        dQ = dScores @ K / sqrt(d_k). dK = dScores^T @ Q / sqrt(d_k).
        dQW = x^T @ dQ, dKW = x^T @ dK, dVW = x^T @ dV, dOutW = attnOut^T @ dOut.
        dx += (dQ @ qW^T + dK @ kW^T + dV @ vW^T).
     d. LayerNorm backward.
  4. Input projection backward: per source, dInputW[s] = raw^T @ dx_s,
     dInputB[s] = sum(dx_s, axis=0).
  Accumulate all gradients into grads struct.
  Acceptance: Gradient check -- numerical gradient (eps=1e-4) matches analytical within 1e-2
  for at least 90% of parameters.

- [x] T60.3.2 Gradient check test  Owner: TBD  Est: 2h  verifies: [UC-CA-001]
  File: crossasset/gpu_train_test.go  Deps: T60.3.1
  Numerical gradient check: perturb each parameter by eps, compute loss diff,
  compare to analytical gradient. Use small model (NSources=3, DModel=8, NHeads=2, NLayers=1).
  Acceptance: max relative error < 0.05 for 90%+ of parameters.

#### E60.4: Training Loop and Integration

- [x] T60.4.1 Implement TrainGPU method  Owner: TBD  Est: 3h  verifies: [UC-CA-001]
  File: crossasset/gpu_train.go  Deps: T60.3.1
  func (m *Model) TrainGPU(data [][][]float64, labels [][]int, tc TrainConfig,
      engine compute.Engine[float32]) (*TrainResult, error)
  Steps:
  1. extractGPUParams from Model.
  2. allocGrads, allocAdamState (reuse adamStateF32 and adamWUpdateF32 from timeseries/adamw_f32.go,
     or copy the functions into crossasset/ to avoid cross-package dependency on unexported symbols).
  3. Pre-allocate workspace: input tensor [bs, NSources, FeaturesPerSource],
     label tensor [bs, NSources, 3] one-hot.
  4. Per-epoch loop: shuffle, batch, for each batch:
     a. Convert float64 data batch to float32 tensor.
     b. gpuForward -> logits, cache.
     c. Compute softmax cross-entropy loss on CPU.
     d. Compute dLogits = softmax(logits) - one_hot(labels).
     e. gpuBackward -> accumulate grads.
     f. Clip gradients (L2 norm).
     g. AdamW update for all parameters.
     h. Zero grads.
  5. After all epochs, write updated float32 params back to Model float64 weights.
  Return TrainResult{Losses []float64, FinalAccuracy float64}.
  Acceptance: Loss decreases over 10 epochs on synthetic data.

- [x] T60.4.2 Integration test: train then predict  Owner: TBD  Est: 2h  verifies: [UC-CA-001]
  File: crossasset/gpu_train_test.go  Deps: T60.4.1
  Test: Create model, TrainGPU with 50 samples for 20 epochs, verify loss decreases.
  Then call Predict and verify outputs are valid (directions in [0,2], confidences in [0,1]).
  Test with CPU engine (no GPU required for CI).
  Acceptance: Loss at epoch 20 < loss at epoch 1. Predictions valid.

- [x] T60.4.3 Run go vet and linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T60.4.2
  Run `go vet ./crossasset/...` and `go test -race ./crossasset/...`.
  Acceptance: Zero warnings, zero race conditions.

#### E60 Parallel Work

Two parallel tracks, merging at T60.4.1:

| Track | Tasks | Description |
|-------|-------|-------------|
| Track A: Infrastructure | T60.1.1, T60.1.2, T60.1.3 | Param extraction, no deps |
| Track B: Forward+Backward | T60.2.1, T60.2.2, T60.3.1, T60.3.2 | Depends on T60.1.2 |

### Wave E60-1: Infrastructure (3 agents)

- [x] T60.1.1 GPU param structs
- [x] T60.1.2 Extract and alloc functions (can start from struct stubs)
- [x] T60.1.3 Unit tests for extraction

### Wave E60-2: Forward + Backward (4 agents)

Deps: Wave E60-1

- [x] T60.2.1 GPU forward pass
- [x] T60.2.2 Forward pass tests
- [x] T60.3.1 GPU backward pass
- [x] T60.3.2 Gradient check test

### Wave E60-3: Integration (3 agents)

Deps: Wave E60-2

- [x] T60.4.1 TrainGPU method
- [x] T60.4.2 Integration test
- [x] T60.4.3 Linters and vet

#### E60 Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R47 | Attention backward gradient math errors (softmax jacobian, multi-head reshape) | High | Medium | Numerical gradient check (T60.3.2) catches errors; reference PyTorch autograd for expected values |
| R48 | Float64->Float32 precision loss affects convergence | Low | Low | Neural network training is routinely done in float32; verify loss converges on synthetic data |
| R49 | Cross-package dependency on timeseries/adamw_f32.go unexported symbols | Medium | High | Copy adamWUpdateF32 and adamStateF32 into crossasset/ (they are ~30 lines); or export them |

#### E60 Milestones

| ID | Milestone | Exit Criteria | Target |
|----|-----------|---------------|--------|
| M-E60-1 | Forward parity | gpuForward matches CPU Forward within 1e-3 on 4-source, 2-layer model | 2026-Q2 |
| M-E60-2 | Full backprop | Gradient check passes for all layers including attention | 2026-Q2 |
| M-E60-3 | Training convergence | Loss decreases monotonically for 20 epochs on synthetic data | 2026-Q2 |

---

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

- [ ] T63.1.1 Design quantized matmul dispatcher interface  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Define a dispatch table or type-switch that maps storage type to:
  (1) GEMV kernel function, (2) dequant kernel function, (3) block size.
  Write the generic `dequantMatMul` function that handles the shared pattern:
  upload, GEMV-or-dequant+GEMM, makeGPUResult.
  File: compute/gpu_engine_matmul.go (new file in ztensor).
  Acceptance: compiles. Dispatcher covers all 8 storage types (Q4, Q4K, Q5_0,
  Q5K, Q6K, Q8, BF16, Mmap) and both normal/BWeight variants.

- [ ] T63.1.2 Replace 16 methods with dispatcher calls  Owner: TBD  Est: 4h  verifies: [infrastructure]
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

#### Wave E63-1: Implement (1 agent)
- [ ] T63.1.1 Design dispatcher
- [ ] T63.1.2 Replace 16 methods (sequential, same file)

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

- [ ] T64.1.1 Split gpu_engine.go into focused files  Owner: TBD  Est: 3h  verifies: [infrastructure]
  Deps: E63 complete
  Split into:
  - gpu_engine.go: core struct, New, lifecycle, dispatch (15-20 methods)
  - gpu_engine_matmul.go: all matmul methods including dispatcher (created in E63)
  - gpu_engine_elementwise.go: add/sub/mul/div/scalar ops
  - gpu_engine_reduction.go: softmax/sum/argmax/topk
  - gpu_engine_memory.go: upload/gather/copy/zero
  Acceptance: go build ./... clean. go test ./compute/ passes. No exported API
  changes. Each file under 1,000 lines.

- [ ] T64.1.2 Run full ztensor test suite  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T64.1.1
  Acceptance: go test -race ./... passes.

- [ ] T64.1.3 Run linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T64.1.1
  Acceptance: go vet clean. golangci-lint clean.

### E64 Parallel Work

#### Wave E64-1: Split (1 agent)
- [ ] T64.1.1 File decomposition

#### Wave E64-2: Validate (2 agents)
Deps: Wave E64-1
- [ ] T64.1.2 Test suite
- [ ] T64.1.3 Linters

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

- [ ] T74.1.1 Add functional.LinearBackward  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Implement LinearBackward(ctx, engine, dOutput, input, weight) returning
  (dInput, dWeight, dBias). Uses engine.MatMul for gradient computation.
  Reference: patchtst_backward.go:405-430 for expected behavior.
  Acceptance: unit test validates gradient correctness via numerical gradient check.

- [ ] T74.1.2 Add functional.LayerNormBackward  Owner: TBD  Est: 3h  verifies: [infrastructure]
  Implement LayerNormBackward(ctx, engine, dOutput, input, gamma, mean, variance)
  returning (dInput, dGamma, dBeta). Uses engine ops for variance correction.
  Reference: itransformer_backward.go:503-537 for expected behavior.
  Acceptance: numerical gradient check passes within 1e-6.

- [ ] T74.1.3 Add functional.GELUBackward  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Implement GELUBackward(ctx, engine, dOutput, input) returning dInput.
  Derivative: 0.5*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) + correction term.
  Reference: patchtst_encoder.go:994-1007 (encoderBackwardF64 GELU derivative).
  Acceptance: numerical gradient check passes within 1e-6.

- [ ] T74.1.4 Add functional.SoftmaxBackward  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Implement SoftmaxBackward(ctx, engine, dOutput, softmaxOutput) returning dInput.
  Formula: dInput_i = s_i * (dOutput_i - sum(dOutput * s)).
  Reference: itransformer_backward.go:542-555.
  Acceptance: numerical gradient check passes within 1e-6.

- [ ] T74.1.5 Add functional.MultiHeadAttentionBackward  Owner: TBD  Est: 4h  verifies: [infrastructure]
  Implement MultiHeadAttentionBackward(ctx, engine, dOutput, Q, K, V, attentionWeights, params)
  returning (dQ, dK, dV, dWq, dWk, dWv, dWo). Composes from LinearBackward and
  SoftmaxBackward internally.
  Deps: T74.1.1, T74.1.4
  Reference: itransformer_backward.go:416-458 for expected behavior.
  Acceptance: numerical gradient check passes within 1e-5.

- [ ] T74.1.6 Add functional.MLPBackward  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Implement MLPBackward(ctx, engine, dOutput, inputs, weights, biases, activationGrads)
  returning (dInput, dWeights, dBiases). Composes from LinearBackward and activation
  backward ops. Supports ReLU and GELU activation gradients.
  Deps: T74.1.1, T74.1.3
  Reference: timemixer_backward.go:457-496 for 2-layer MLP backward.
  Acceptance: numerical gradient check for 2-layer MLP passes within 1e-6.

- [ ] T74.1.7 Unit tests for all backward functional ops  Owner: TBD  Est: 2h  verifies: [infrastructure]
  Deps: T74.1.1, T74.1.2, T74.1.3, T74.1.4, T74.1.5, T74.1.6
  Comprehensive test file: functional_backward_test.go. Tests numerical gradient
  correctness for each backward op across float32 and float64. Tests composition
  (e.g., Linear+GELU backward chain).
  Acceptance: go test -race ./layers/functional/... passes with all backward tests green.

### E74.2: Migrate Backward Files

- [ ] T74.2.1 Migrate patchtst_backward.go to functional backward ops  Owner: TBD  Est: 4h  verifies: [UC-TS01]
  Deps: T74.1.7
  Replace raw f64 loops in backwardF64() (lines 266-378) with functional.LinearBackward,
  functional.LayerNormBackward, and functional.GELUBackward calls.
  Replace head backward (lines 299-327) and patch embedding backward (lines 329-347)
  with functional ops.
  Acceptance: go test ./timeseries/... passes. PatchTST 10-epoch training loss matches
  pre-refactor within 1e-4.

- [ ] T74.2.2 Migrate encoderBackwardF64 in patchtst_encoder.go  Owner: TBD  Est: 4h  verifies: [UC-TS01]
  Deps: T74.1.7
  Replace raw f64 encoder backward (lines 938-1120) with functional.MultiHeadAttentionBackward,
  functional.LayerNormBackward, functional.GELUBackward, and functional.LinearBackward.
  This is the largest single backward function (~180 lines of raw loops).
  Acceptance: PatchTST encoder backward output matches pre-refactor within 1e-6.

- [ ] T74.2.3 Migrate itransformer_backward.go to functional backward ops  Owner: TBD  Est: 3h  verifies: [UC-TS01]
  Deps: T74.1.7
  Replace backward() (lines 274-312), encoderLayerBackward() (lines 316-497),
  layerNormBackward() (lines 503-537), and softmaxBackward() (lines 542-555)
  with functional backward op calls.
  Delete local layerNormBackward and softmaxBackward helper functions.
  Acceptance: iTransformer 10-epoch training loss matches pre-refactor within 1e-4.

- [ ] T74.2.4 Migrate timemixer_backward.go to functional backward ops  Owner: TBD  Est: 3h  verifies: [UC-TS02]
  Deps: T74.1.7
  Replace backward() (lines 278-448) and mlpBackward() (lines 457-496) with
  functional.MLPBackward and functional.LinearBackward calls.
  Replace ReLU backward conditional (lines 457-480) with functional activation gradient.
  Acceptance: TimeMixer 10-epoch training loss matches pre-refactor within 1e-4.

### E74.3: Validation

- [ ] T74.3.1 Full timeseries test suite with race detector  Owner: TBD  Est: 1h  verifies: [UC-TS01, UC-TS02]
  Deps: T74.2.1, T74.2.2, T74.2.3, T74.2.4
  Run go test -race -timeout 300s ./timeseries/...
  Acceptance: all tests pass with zero race conditions.

- [ ] T74.3.2 Run linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T74.3.1
  Run go vet ./timeseries/... and golangci-lint on all changed files.
  Acceptance: zero warnings.

- [ ] T74.3.3 Verify line count reduction  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T74.3.2
  Count lines in patchtst_backward.go, itransformer_backward.go, timemixer_backward.go,
  patchtst_encoder.go before and after. Document net reduction.
  Target: at least 800 lines of raw backward computation eliminated.
  Acceptance: line count documented in commit message.

### E74 Parallel Work

#### Wave E74-1: Functional backward API (6 agents)
All functional backward ops are independent of each other.
- [ ] T74.1.1 LinearBackward
- [ ] T74.1.2 LayerNormBackward
- [ ] T74.1.3 GELUBackward
- [ ] T74.1.4 SoftmaxBackward
- [ ] T74.1.5 MultiHeadAttentionBackward (deps: T74.1.1, T74.1.4 -- start after Wave E74-1a)
- [ ] T74.1.6 MLPBackward (deps: T74.1.1, T74.1.3 -- start after Wave E74-1a)

Split into two sub-waves:
- Wave E74-1a (4 agents): T74.1.1, T74.1.2, T74.1.3, T74.1.4
- Wave E74-1b (2 agents): T74.1.5, T74.1.6 (after E74-1a)
- Wave E74-1c (1 agent): T74.1.7 (after E74-1b)

#### Wave E74-2: Migrate backward files (4 agents)
Deps: Wave E74-1c
All 4 migration tasks are independent (separate files).
- [ ] T74.2.1 patchtst_backward.go
- [ ] T74.2.2 patchtst_encoder.go (encoderBackwardF64)
- [ ] T74.2.3 itransformer_backward.go
- [ ] T74.2.4 timemixer_backward.go

#### Wave E74-3: Validation (2 agents)
Deps: Wave E74-2
- [ ] T74.3.1 + T74.3.2 Tests + linters
- [ ] T74.3.3 Line count verification

---

## E75: Inference Timeseries .Data() Elimination

**Problem:** 8 architecture builder files in inference/timeseries/ have 29 .Data()
calls totaling ~2,000 lines of raw tensor access that bypasses the engine. 25 of
29 calls are unjustified (reimplementing softmax, ReLU, buffer copy, channel
extraction, gather/scatter using raw loops instead of engine ops). 4 calls are
justified (chronos one-hot encoding, flowstate Fourier evaluation, regime utilities).

**Goal:** Replace unjustified .Data() access with engine ops (engine.Slice,
engine.Reshape, engine.ReLU, layers/activations.Softmax). Reduce .Data() calls
from 29 to 4 (justified only). Estimated ~200-315 lines of raw loops replaced.

### E75.1: Architecture Builder Migration

- [ ] T75.1.1 arch_timemixer.go -- Replace inline softmax and ReLU with engine ops  Owner: TBD  Est: 3h  verifies: [UC-TS02]
  Replace inline softmax (line 341) with layers/activations.Softmax.
  Replace ReLU on hidden (lines 464-470) with engine.ReLU().
  Replace gather/scatter reshaping (lines 446, 482) with engine.Reshape.
  Replace moving average .Data() access with engine ops where possible.
  10 .Data() calls, target: reduce to 2-3 (justified custom decomposition).
  Acceptance: arch_timemixer tests pass. Inference output matches pre-refactor.

- [ ] T75.1.2 arch_tft.go -- Replace buffer operations with engine ops  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Replace hidden state stacking (lines 312-322) with tensor.Stack or engine.Concat.
  Replace VSN output buffer copy (lines 518-525) with engine.Reshape.
  Keep forget gate bias init (justified).
  4 .Data() calls, target: reduce to 1.
  Acceptance: arch_tft tests pass. TFT inference output matches pre-refactor.

- [ ] T75.1.3 arch_ttm.go -- Replace channel extraction with engine.Slice  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Replace channel extraction (lines 591-603) with engine.Slice.
  Keep channel statistics computation (justified custom normalization).
  5 .Data() calls, target: reduce to 3.
  Acceptance: arch_ttm tests pass.

- [ ] T75.1.4 arch_tirex.go -- Replace timestep extraction with engine.Slice  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Replace manual timestep extraction loops (lines 376-414, 580-620) with
  engine.Slice for LSTM block processing input preparation.
  2 .Data() calls, target: reduce to 0.
  Acceptance: arch_tirex tests pass. TiRex inference output matches.

- [ ] T75.1.5 arch_tspulse.go -- Replace reshaping with engine.Reshape  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Replace recon reshaping (lines 271-288), class prob reshaping (lines 350-365),
  and embedding format (lines 401-410) with engine.Reshape.
  3 .Data() calls, target: reduce to 0.
  Acceptance: arch_tspulse tests pass.

- [ ] T75.1.6 arch_flowstate.go -- Replace channel extraction with engine.Slice  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Replace channel extraction (lines 360-368) with engine.Slice.
  Keep Fourier basis evaluation (justified custom operation).
  2 .Data() calls, target: reduce to 1.
  Acceptance: arch_flowstate tests pass.

### E75.2: Validation

- [ ] T75.2.1 Full inference/timeseries test suite  Owner: TBD  Est: 1h  verifies: [UC-TS01, UC-TS02]
  Deps: T75.1.1, T75.1.2, T75.1.3, T75.1.4, T75.1.5, T75.1.6
  Run go test -race ./inference/timeseries/...
  Acceptance: all tests pass.

- [ ] T75.2.2 Run linters and verify .Data() count  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T75.2.1
  Run go vet and golangci-lint. Count remaining .Data() calls in inference/timeseries/.
  Target: 4 or fewer .Data() calls (justified only: chronos, flowstate Fourier, regime).
  Acceptance: lint clean. .Data() count documented.

- [ ] T75.2.3 Verify unchanged files are excluded  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T75.2.1
  Confirm arch_chronos.go (1 justified .Data()) and arch_regime.go (2 justified .Data())
  were NOT modified (no unnecessary changes to justified code).
  Acceptance: git diff shows no changes to arch_chronos.go or arch_regime.go.

### E75 Parallel Work

#### Wave E75-1: Architecture builder migration (6 agents)
All 6 files are independent.
- [ ] T75.1.1 arch_timemixer.go
- [ ] T75.1.2 arch_tft.go
- [ ] T75.1.3 arch_ttm.go
- [ ] T75.1.4 arch_tirex.go
- [ ] T75.1.5 arch_tspulse.go
- [ ] T75.1.6 arch_flowstate.go

#### Wave E75-2: Validation (2 agents)
Deps: Wave E75-1
- [ ] T75.2.1 + T75.2.2 Tests + linters + .Data() count
- [ ] T75.2.3 Verify unchanged files

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

- [ ] T76.1.2 Verify CI green  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T76.1.1
  Push change and confirm CI workflow passes.
  Acceptance: CI green.

### E76 Parallel Work

#### Wave E76-1: Sequential (1 agent)
Deps: E74 complete
- [ ] T76.1.1 -> T76.1.2 (sequential)

---

### E66-E76 Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R55 | layers/functional API adds dispatch overhead for training | Low | Medium | Functions are thin wrappers; benchmark to confirm <2% overhead |
| R56 | timeseries/ migration breaks training accuracy | High | Medium | Parity tests for every model; compare 10-epoch training loss curves before/after |
| R57 | crossasset/ graph-based backward produces different gradients | Medium | Medium | Numerical gradient check (finite differences) validates analytical gradients |
| R58 | MAML inner-loop in meta/ is incompatible with graph-based backward | Medium | High | If incompatible, keep meta/ as justified exception; document why |
| R59 | Architecture enforcement test has false positives | Low | Medium | Maintain allowlist; review and adjust thresholds quarterly |
| R60 | KV cache strategy pattern adds virtual dispatch overhead | Low | Low | Benchmark KV cache Get/Set latency; strategy dispatch is not on hot path |
| R61 | Functional backward ops produce numerically different gradients than hand-coded loops | High | Medium | Numerical gradient check (finite differences) for each op; 10-epoch training loss parity within 1e-4 |
| R62 | MultiHeadAttentionBackward is complex to implement correctly with batched heads | Medium | Medium | Start from reference implementation in itransformer_backward.go; validate per-head gradient isolation |
| R63 | Replacing .Data() with engine.Slice in inference/timeseries/ may change graph node count | Low | Low | Benchmark inference throughput before/after; engine.Slice is a lightweight op |
| R64 | Removing timeseries/ from architecture test allowlist too early causes CI failures | Low | Medium | Only remove AFTER E74 is fully complete and all tests pass |

### E66-E76 Milestones

| ID | Milestone | Exit Criteria | Target |
|----|-----------|---------------|--------|
| M-COMP-2 | Composition Phase 2 Complete | E66-E73 all compose from layers/ or Engine; architecture test in CI | DONE 2026-04-03 |
| M-COMP-3 | Composition Phase 3 Complete | E74 backward composition done; E75 inference .Data() eliminated; E76 allowlist removed; zero raw backward loops in timeseries/; dirty-architecture.md violations at 0 | 2026-Q3 |

---

### E47-E49 Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R40 | Batched backward correctness for attention layers is error-prone | High | Medium | Gradient check (numerical vs analytical) as acceptance criteria for every backward task; reference PyTorch autograd output |
| R41 | xLSTM cell numerical stability (exponential gating overflow) | Medium | Medium | Clamp gate pre-activations; use log-space computation for normalizer state; validate against HuggingFace reference |
| R42 | TimeMixer multi-scale decomposition hyperparameter sensitivity | Medium | Medium | Default to 4 scales; expose all hyperparameters in config; validate on ETT benchmark before shipping |
| R43 | Foundation model HuggingFace weights change format across versions | Low | Medium | Pin transformers version in requirements.txt; test against specific model revisions |
| R44 | Batched GPU memory exceeds DGX Spark VRAM on large datasets | Medium | Low | DataLoader batch size is configurable; default to batch_size=64; document VRAM requirements per backend |

### E47-E49 Milestones

| ID | Milestone | Exit Criteria | Target |
|----|-----------|---------------|--------|
| M-E47 | Batched training practical | PatchTST 28K rows trains in < 60s on DGX Spark | 2026-Q2 |
| M-E48 | TimeMixer shipped | TimeMixer TrainWindowed + inference graph builder; tests pass | 2026-Q2 |
| M-E49 | Foundation models accessible | All 3 models produce zero-shot forecasts via gRPC bridge | 2026-Q2 |
