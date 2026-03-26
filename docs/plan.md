# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, and documentation site).

Task statuses updated 2026-03-26 based on merged PRs and git history.

**Status summary:**
- 305+ tasks completed across all plans
- ~60 tasks remaining (details below)

---

## Completed Work (Summary)

### Priority 1: Tabular and Time-Series ML

All internal-consumer-blocking work is complete:

| Epic | Description | Status |
|------|-------------|--------|
| WE1 | Tabular Model Package | Complete (W1.1.1-W1.1.4) |
| WE2 | Advanced Tabular Architectures | Complete (W2.1.1-W2.1.4) |
| WE3 | Time-Series Architectures | Complete (W2.2.1-W2.2.3) |
| WE4 | Tabular AutoML Extension | Complete (W2.3.1) |
| WE5 | Transfer Learning for Tabular | Complete (W5.1.1-W5.1.3) |
| WE6 | Reinforcement Learning Package | Complete (W6.1.1-W6.1.3) |
| WE7 | Cross-Asset and Causal Models | Complete (W7.1.1-W7.1.2, W7.3.1-W7.3.2) |
| WE8 | Regime Detection and Synthetic Data | Complete (W8.1.1, W8.2.1-W8.2.2) |
| WE9 | Self-Improving Systems | Complete (W9.1.1, W9.2.1, W9.3.1) |
| WE10 | Hardware Optimization for Tabular | Complete (W10.1.1-W10.1.3) |
| WE11 | Enterprise Features (Internal) | Complete (W10.2.1-W10.2.3) |
| WE12 | Continuous Learning and Provenance | Complete (W11.1.1-W11.1.2, W11.2.1, W11.3.1) |
| WE13 | Performance and Test Fixes | Complete (W3.1.1-W3.1.5) |

### Priority 2: Inference Performance and Bug Fixes

| Epic | Description | Status |
|------|-------------|--------|
| E2 | New Model Architecture Support | Complete (T2.1-T2.12, all 12 archs validated) |
| E4 | Documentation and Developer Experience | Complete except T4.7 (video) |
| E5 | Community Infrastructure | Complete except T5.4 (Discord) |
| E9 | Multi-GPU Inference | Complete (T9.1-T9.3) except T9.4 (benchmark) |
| E10 | Vision-Language Model Expansion | Complete (T10.1-T10.3) |
| E14 | SOC 2 Certification | Complete (T14.1-T14.4) |
| E17 | Zerfoo Cloud GA | Complete (T17.1-T17.7) |
| E18 | Enterprise Features | Complete (T18.1-T18.5) |
| E20 | Apple Metal Backend | Complete except T20.3 (benchmark) |
| E21 | Intel SYCL Backend | Complete (T21.1-T21.2) |
| E22 | Auto-Optimization Framework | Complete (T22.1-T22.3) |
| E24 | Custom Model Architecture SDK | Complete (T24.1-T24.3) |
| E25 | Heterogeneous Compute | Complete (T25.1-T25.2) |
| E27 | Ecosystem Integrations | Complete (T27.1-T27.3) |
| E28 | Federated Learning | Complete (T28.1-T28.4) |
| E29 | On-Device Inference | Complete (T29.1-T29.3) except T29.4 (benchmark) |
| E33 | Performance Target 1000+ tok/s | Complete (T33.1-T33.2) |

### Priority 0: Security Remediation

| Epic | Description | Status |
|------|-------------|--------|
| E101 | GitHub Issues Resolution | Complete (T101.1-T101.15) |
| E102 | Attention Residuals (AttnRes) | Complete (T102.1-T102.5) |
| E103 | Throughput Regression Fix | Complete (T103.1-T103.5, 244 tok/s restored) |
| E104 | HRM Segfault (#105) | Complete (T104.1-T104.4) |
| E105 | NaN/Inf Training (#121) | Complete (T105.1-T105.8) |
| E106 | Security Deep Review v1.10.0 | Complete (T106.1-T106.37, 37 tasks) |
| E107 | v1.11.0 Remediation + #123 | Complete (T107.1-T107.12) |
| E108 | Deep Review v1.11.1 | Complete (T108.1-T108.32, 32 tasks) |
| E109 | Deep Review v1.12.0 | Complete (T109.1-T109.9) |
| E110 | Issues #152-#156 | Complete (T110.1-T110.6) |
| E111 | Verification Remediation | Complete (T111.1-T111.3) |
| E112 | CPU Training Performance (#157) | Complete (T112.1-T112.6) |
| E113 | GPU Engine Training | Complete (T113.1-T113.3) |
| E115 | Deep Review v1.15.1 | Complete (T115.1-T115.9) |
| E116 | Deep Review v1.15.1 Remaining | Complete (T116.1-T116.8) |
| E117 | Cloud Package Consolidation | Complete (T117.1-T117.3) |
| E118 | Observability Metrics | Complete (T118.1-T118.2) |
| E119 | Persistent Store Interfaces | Complete (T119.1-T119.4) |
| E120 | God File Splits | Complete (T120.1-T120.3) |
| E121 | Backward Pass Error Propagation | Complete (T121.1-T121.2) |
| E122 | CodeQL in CI | Complete (T122.1) |

### Granite Time Series (PRs #187-197, #208)

16 of 18 tasks complete. SafeTensors-to-GGUF conversion, all three architecture
builders (TTM, FlowState, TSPulse), inference pipelines, training backends, and
exogenous variable support all shipped. TTM GPU training path fixed (PR #208).

Completed tasks: GTS-T1.1 through GTS-T4.3 (Waves 1-2), GTS-T5.1 (CLI),
GTS-T5.2 (REST API endpoints).

### Granite Guardian (PRs #200-205)

10 of 13 tasks complete. Granite architecture builder, prompt template engine,
verdict parser, evaluator, batch evaluation, multi-risk scanning, REST API
endpoints, and CLI `zerfoo guard` command all shipped.

Completed tasks: GG-T1.1 through GG-T3.2.

### K-Quant Optimization (PRs #179-186)

Infrastructure complete. Native Q4_K/Q5_K/Q6_K storage with virtual transpose
(PR #179), merged QKV/GateUp GEMV for Q4K (PR #181), Q6_K/Q5_K/Q4_K SM121
GEMV kernels (PR #182). Q4_K re-quantization restored for performance (PRs
#183-184). Q4 GEMV shared memory fix for 7B+ models (PR #186). GPU RMSNorm
fallback to prevent CUDA graph D2H (PR #185).

Completed: T1.1, T1.2, T2.1 (reverted), T2.2, T2.3, T5.1 (investigation).

### Multi-Model Benchmarks (PRs #174-178)

Environment setup complete (T1.1, T1.2, T1.4). Benchmark script written (T2.1,
T2.2). Small models benchmarked (T3.1): Gemma3-1B 236/204 (1.16x), DeepSeek-R1
193/185 (1.04x), Llama3.2 96/98 (0.98x). Medium models partially benchmarked
(T3.2): Mistral-7B 12/47 (0.25x regression). Benchmark claims updated (PR #176-178).

Completed: T1.1, T1.2, T1.4, T2.1, T2.2, T3.1, T3.2 (partial).

### GPU Training for All Backends (PRs #170)

All 7 timeseries backends wired to GPU engine forward path (PR #170).

Completed: T123.1-T123.5.

### E12: Enterprise Support Tier

Completed: T12.1 (SLA tiers), T12.2 (ticketing), T12.3 (deployment guide).

### E11: Community Growth

Completed: T11.2 (tutorial series), T11.3 (KubeCon CFP), T11.5 (LangChain-Go).

### E13: Security Audit

Completed: T13.1 (deep review), T13.2 (remediation plan), T13.3 (SBOM), T13.4 (fuzz).

### E15: Edge Deployment

Completed: T15.1 (edge binary), T15.2 (optimized format), T15.5 (ARM64 CI).

### E16: Performance 500+ tok/s

Completed: T16.1 (warp-specialized GEMV), T16.2 (KV cache FP8).

### E26: ZerfooConf

Completed: T26.1 (plan ZerfooConf Day).

### E32: Architecture Expansion

Completed: T32.1 (automated architecture builder from GGUF metadata).

---

## Active Work

### Granite Time Series -- 2 tasks remaining

- [ ] GTS-T5.3 Accuracy parity tests against Python granite-tsfm
  Owner: ML Eng  Est: 8h
  Files: tests/parity/granite_ts_test.go
  Deps: GTS-T2.4, GTS-T3.3, GTS-T4.3
  Description: Parity tests that verify Zerfoo's Granite TS output matches
  Python granite-tsfm reference. For each model family:
  1. Run Python reference on a fixed input, save output as golden file.
  2. Run Zerfoo on same input, compare within tolerance (1e-4 for F32).
  Test on ETTh1 (forecasting), synthetic anomalies (detection), UCR subset
  (classification). Golden files checked into tests/parity/testdata/.
  Acceptance:
  - TTM forecast MAE within 1e-4 of Python reference on 3 test cases.
  - FlowState forecast within 1e-4 on 3 test cases across sampling rates.
  - TSPulse anomaly scores within 1e-4 on 2 test cases.
  - TSPulse classification logits within 1e-4 on 2 test cases.

- [ ] GTS-T5.4 Benchmark suite and performance optimization
  Owner: Kernel Eng  Est: 8h
  Files: tests/benchmark/granite_ts_bench_test.go
  Deps: GTS-T5.3
  Description: Comprehensive benchmarks comparing Zerfoo vs Python granite-tsfm:
  - **Latency**: Single-series inference time (ms) for each model family.
  - **Throughput**: Series/second for batch sizes 1, 8, 32, 128.
  - **Accuracy**: MAE/MSE on ETTh1, Weather, Electricity standard benchmarks.
  - **Memory**: Peak RSS during inference.
  Target: 5-10x latency improvement over Python (tiny models where Python/PyTorch
  overhead dominates). Optimize hot paths: CUDA graph capture, fuse ops, F16.
  Acceptance:
  - Benchmark results published in tests/benchmark/results/.
  - Zerfoo TTM inference latency < 2ms per series on GPU (batch=1).
  - Zerfoo FlowState inference latency < 5ms per series on GPU (batch=1).
  - Throughput > 10,000 series/sec on GPU (batch=128) for TTM.
  - All accuracy metrics within 1% of Python reference.

### Granite Guardian -- 3 tasks remaining

- [x] GG-T3.3 Guardrails middleware for chat completions
  Owner: ML Eng  Est: 6h
  Files: serve/guardian_middleware.go, serve/guardian_middleware_test.go
  Deps: GG-T2.4
  Description: Optional guardrails middleware that wraps `/v1/chat/completions`.
  Evaluates user prompt before generation and (optionally) assistant response
  after generation. If content is flagged, return 400 with verdict details.
  Configuration:
  ```go
  type GuardianMiddlewareConfig struct {
      Model         string   // Guardian model path
      Risks         []string // risk categories to check
      CheckInput    bool     // scan user prompts (default: true)
      CheckOutput   bool     // scan assistant responses (default: false)
      BlockOnFlag   bool     // return error if flagged (default: true)
  }
  ```
  Acceptance:
  - Chat request with harmful prompt is blocked with 400 + verdict JSON.
  - Chat request with safe prompt passes through normally.
  - Output checking scans assistant response before returning to client.
  - Middleware can be disabled at runtime via config.
  - go test with httptest passes.

- [ ] GG-T4.1 Parity tests against Ollama granite3-guardian
  Owner: ML Eng  Est: 8h
  Files: tests/parity/guardian_test.go, tests/parity/testdata/guardian/
  Deps: GG-T2.2
  Description: Parity tests that verify Zerfoo's Guardian verdicts match Ollama.
  Test cases: 5 harmful inputs, 5 safe inputs, 3 RAG hallucination cases,
  2 edge cases. Golden files checked into tests/parity/testdata/guardian/.
  Acceptance:
  - Verdict (Yes/No) matches Ollama on all 15 test cases.
  - Confidence values within 0.05 of Ollama reference.
  - Tests run on DGX Spark.

- [ ] GG-T4.2 Latency benchmarks
  Owner: Kernel Eng  Est: 6h
  Files: tests/benchmark/guardian_bench_test.go
  Deps: GG-T2.3
  Description: Benchmark Guardian evaluation latency: single evaluation,
  multi-risk scan, batch throughput. Compare against Ollama on DGX Spark.
  Target: single evaluation < 100ms on GPU.
  Acceptance:
  - Benchmark results for all three scenarios.
  - Comparison table vs. Ollama.
  - Single evaluation latency < 100ms on GPU.

- [ ] GG-T4.3 Safety benchmark accuracy evaluation
  Owner: ML Eng  Est: 6h
  Files: tests/benchmark/guardian_accuracy_test.go
  Deps: GG-T2.2
  Description: Evaluate on HarmBench, ToxiGen, XSTest subsets. Binary
  classification metrics: precision, recall, F1, balanced accuracy.
  Acceptance:
  - F1 scores within 2% of IBM's reported numbers.
  - False positive rate on XSTest safe subset < 5%.
  - Results logged with per-category breakdown.

### K-Quant Optimization -- 9 tasks remaining

#### E4: Q4_K GEMV Kernel Optimization [ztensor repo]

- [ ] T4.1 Profile gemv_q4k.cu vs gemv_q4.cu  Est: 2h  repo: ztensor
  Use `ncu` (Nsight Compute) on DGX Spark. Collect register usage, occupancy,
  memory throughput, warp stalls, L1/L2 cache hit rates.
  Acceptance: profiling report with root cause identified.

- [ ] T4.2 Optimize Q4_K GEMV kernel  Est: 4h  repo: ztensor
  Apply optimizations based on T4.1 profiling: reduce register usage, improve
  coalesced memory access, use shared memory for sub-block scales, reduce warp
  divergence, tune thread block size.
  Acceptance: measurable improvement in BenchmarkGEMV.

- [ ] T4.3 Benchmark Q4_K GEMV at target  Est: 1h  repo: ztensor
  Target: >= 215 tok/s on Gemma 3 1B (within 10% of Q4_0's 236 tok/s).
  Acceptance: benchmark result recorded.

- [ ] T4.4 Re-enable native Q4_K loading in zerfoo  Est: 1h  depends: T4.3
  Remove Q4_K re-quantization in model/gguf/loader.go. All infrastructure
  (virtual transpose, merged QKV) already merged and ready.
  Acceptance: Gemma 3 1B Q4_K_M >= 215 tok/s, all tests pass.

#### E5: CUDA Graph Capture Fix for Mistral 7B [zerfoo repo]

- [x] T5.2 Eliminate the D2H copy  Est: 3h  depends: T5.1 (done)
  Fix identified call site from T5.1 investigation. Options: replace Slice()
  with GPU-side view, pre-allocate buffer, restructure KV cache.
  Acceptance: CUDA graph capture succeeds for Mistral 7B.

- [x] T5.3 Benchmark Mistral 7B with CUDA graphs  Est: 1h  depends: T5.2
  Target: >= 40 tok/s (from 11.6 baseline, parity with Ollama's 46.8).
  Acceptance: benchmark result recorded.

#### E6: Mistral Architecture Detection and Tokenizer Fix [zerfoo repo]

- [x] T6.1 Detect Mistral from GGUF metadata  Est: 2h
  Parse GGUF metadata to identify Mistral models (general.name, vocab size,
  sliding window config, tokenizer.ggml.pre).
  Acceptance: Mistral 7B GGUF correctly identified.

- [x] T6.2 Apply correct tokenizer for Mistral  Est: 2h  depends: T6.1
  Configure BPE tokenizer with Mistral's vocabulary and special tokens.
  Acceptance: Mistral 7B produces coherent text output.

- [ ] T6.3 Benchmark Mistral 7B end-to-end  Est: 1h  depends: T5.3, T6.2
  Full end-to-end test: load, generate 128 tokens, verify coherent output.
  Target: >= 40 tok/s with CUDA graph fix.
  Acceptance: coherent output + tok/s >= 40.

### Multi-Model Benchmarks -- 9 tasks remaining

- [ ] T1.3 Acquire GGUF files for all 13 Zerfoo models  Est: 1h
  Missing: gemma3-4b, qwen2.5-7b, mixtral-8x7b, command-r-35b, falcon-7b,
  mamba-2.8b, rwkv-7b. Note: phi3 and llama3.1 GGUFs have format mismatches.

- [ ] T3.3 Benchmark large models (35B+)  Est: 1h
  Models: Mixtral 8x7B, Command R 35B. May not fit in 128GB.

- [ ] T3.4 Benchmark alternative architectures (Mamba, RWKV)  Est: 30m
  Mamba 2.8B (Zerfoo only, no Ollama comparison).

- [ ] T4.1 Compile results into comparison table  Est: 30m  depends: T3.1-T3.4
  Merge all JSON results, calculate Zerfoo/Ollama ratios, flag regressions.

- [ ] T4.2 Update website benchmarks page  Est: 30m  depends: T4.1
  Update zerfoo.github.io benchmarks page with full comparison table.

- [ ] T4.3 Update README.md benchmark claims  Est: 15m  depends: T4.1

- [ ] T4.4 File GitHub issues for regressions  Est: 30m  depends: T4.1
  Mistral 7B 0.25x regression already known. File issues for any others.

- [ ] T4.5 Record results in devlog  Est: 15m  depends: T4.1

### GPU Verification (E114) -- 7 tasks remaining

#### Wave 59: DGX Sync and Kernel Rebuild

- [x] T114.1 Sync DGX repos and rebuild kernel library
  Owner: Infra Eng  Est: 1h
  Steps: SSH to DGX, pull both repos, rebuild libkernels.so.
  Acceptance: Both repos at latest main, go build clean.

#### Wave 60: Fix Segfaults

- [x] T114.2 Add VRAM bounds check for large cuBLAS MatMul
  Owner: Kernel Eng  Est: 2h  depends: T114.1
  Files: ztensor/compute/gpu_engine.go
  Acceptance: Error return (not segfault) for 128256x4096 on GB10.

- [x] T114.3 Fix FP16 MatMul segfault on aarch64 purego
  Owner: Kernel Eng  Est: 3h  depends: T114.1
  Files: ztensor/compute/gpu_engine.go, ztensor/internal/cuda/cublas_purego.go
  Acceptance: TestGPUEngine_FP16 passes on DGX without segfault.

#### Wave 61: Kernel + BF16 Fixes

- [x] T114.4 Verify custom CUDA kernels load after rebuild
  Owner: Kernel Eng  Est: 1h  depends: T114.1
  Acceptance: Cos, Softmax, Transpose, Gather GPU parity tests PASS on DGX.

- [ ] T114.5 Fix BF16 MatMul tolerance or add tensor core detection
  Owner: Kernel Eng  Est: 1h  depends: T114.1
  Acceptance: BF16 MatMul tests pass on DGX.

#### Wave 62: Timeseries GPU Verify

- [x] T114.6 Run timeseries GPU engine tests on DGX
  Owner: ML Eng  Est: 1h  depends: T114.1
  Acceptance: All 7 timeseries backends pass engine training on DGX GPU.

#### Wave 63: Full GPU Suite

- [ ] T114.7 Run full GPU test suite and produce report
  Owner: QA Eng  Est: 2h  depends: T114.2-T114.6
  Acceptance: 0 segfaults, all float32/BF16/kernel/timeseries tests pass.

### GPU Training DGX Benchmark (E123)

- [x] T123.6 Run GPU training benchmark on DGX and verify GPU utilization
  Owner: ML Eng  Est: 2h  depends: T123.1-T123.5 (done), T114.1
  Steps: Run iTransformer on 28K rows on DGX, monitor nvidia-smi.
  Acceptance: GPU util >50%, training <60s, close issue #166.

---

## Backlog

### Batched GPU Training for TrainWindowed (Issue #169)

4 tasks remaining. PatchTST per-sample GPU overhead makes 28K rows impractical.
Fix: batch N samples into single tensor for GPU forward pass.

#### E1: Batched Forward Pass

- [x] T1.1 Add batched forward method to PatchTST engine path  Est: 2h
  Files: timeseries/patchtst_engine.go
  Description: Add `forwardBatchF64WithCacheEngine()` that packs batchWindows
  into a single [batchSize, channels, inputLen] tensor, calls Forward, unpacks.
  Acceptance: method returns correct predictions for batch of 4 samples.

- [x] T1.2 Modify trainWindowedEngine batch loop to use batched forward  Est: 1h
  Deps: T1.1
  Replace per-sample loop with batched forward call.
  Acceptance: same loss trajectory as per-sample version.

#### E2: Testing and Verification

- [x] T2.1 Add numerical parity test  Est: 1h
  Batched forward produces same predictions as per-sample (within 1e-5).

- [x] T2.2 Add benchmark test  Est: 30m
  Batched vs per-sample with 256 samples. Must be at least 10x faster.

### Shared GGUF Writer in ztensor

18 tasks remaining across 3 repos (ztensor, zerfoo, zonnx). Consolidates 5
hand-rolled GGUF writers into a single shared `gguf/` package in ztensor.
Decision: docs/adr/061-gguf-writer-in-ztensor.md

#### E1: Create ztensor/gguf Package [ztensor repo]

- [x] T1.1 Create gguf/constants.go  Est: 30m
  GGUF v3 magic, version, alignment, type constants.

- [x] T1.2 Create gguf/writer.go  Est: 2h  depends: T1.1
  Writer struct with metadata and tensor methods. Based on zonnx/pkg/gguf/writer.go
  plus AddMetadataUint64 and AddMetadataUint32Array.

- [x] T1.3 Write unit tests for gguf/writer.go  Est: 1h30m  depends: T1.2
  >= 90% line coverage.

- [x] T1.4 Create gguf/reader.go for round-trip testing  Est: 1h30m  depends: T1.1
  Minimal reader for ztensor's own round-trip tests.

- [x] T1.5 Round-trip integration test: Writer -> Reader  Est: 1h  depends: T1.2, T1.4

- [x] T1.6 Run go vet and golangci-lint  Est: 15m  depends: T1.2-T1.5

#### E2: Migrate zerfoo Writers [zerfoo repo]

- [x] T2.1 Update zerfoo go.mod  Est: 15m  depends: E1
- [x] T2.2 Migrate training/lora/checkpoint.go  Est: 1h  depends: T2.1
- [x] T2.3 Migrate training/nas/export.go  Est: 1h  depends: T2.1
- [x] T2.4 Migrate distributed/fsdp/checkpoint.go  Est: 1h  depends: T2.1
- [x] T2.5 Migrate cmd/ts_train/main.go  Est: 45m  depends: T2.1
- [x] T2.6 Migrate inference test helpers  Est: 45m  depends: T2.1
- [x] T2.7 Implement SaveModel in training/adapter.go  Est: 1h30m  depends: T2.1
- [x] T2.8 Run go vet and golangci-lint  Est: 15m  depends: T2.2-T2.7

#### E3: Migrate zonnx Writer [zonnx repo]

- [x] T3.1 Update zonnx go.mod  Est: 15m  depends: E1
- [x] T3.2 Migrate zonnx converter  Est: 1h  depends: T3.1
- [x] T3.3 Delete zonnx pkg/gguf/writer.go  Est: 30m  depends: T3.2
- [x] T3.4 Run go vet and golangci-lint  Est: 15m  depends: T3.2-T3.3

### Documentation Site Migration

48 tasks remaining across 12 epics. Consolidate overlapping docs, set up Hugo
site at zerfoo.feza.ai/docs/, migrate all user-facing content, delete from repo.
Decision: docs/adr/064-docs-site-hugo.md

#### E0: Documentation Audit and Cleanup

- [x] T0.1 Consolidate getting-started docs  Est: 45m
- [x] T0.2 Consolidate GPU setup docs  Est: 30m
- [x] T0.3 Consolidate enterprise deployment docs  Est: 45m
- [x] T0.4 Consolidate benchmark docs  Est: 30m
- [x] T0.5 Delete docsite/ directory  Est: 5m
- [x] T0.6 Verify internal docs classification  Est: 15m

#### E1: Hugo Infrastructure Setup

- [x] T1.1 Initialize Hugo project in zerfoo.github.io  Est: 1h
- [x] T1.2 Customize Hugo Book theme  Est: 1h  depends: T1.1
- [x] T1.3 GitHub Actions CI/CD  Est: 30m  depends: T1.1
- [x] T1.4 Navigation structure  Est: 30m  depends: T1.1

#### E2-E9: Content Migration (30 tasks)

Getting Started (3 tasks), Tutorials (4), API Reference (6), Cookbooks (2),
Blog (2), Architecture + Deployment (5), Reference + Ecosystem (7), zonnx (3).

Full task details in git history (commit that created plan-site.md).

#### E10: Repo Cleanup

- [x] T10.1 Delete migrated user-facing docs from repo  Est: 30m
- [x] T10.2 Update README.md links to website  Est: 20m
- [x] T10.3 Update CONTRIBUTING.md links  Est: 10m

#### E11: Final Verification

- [x] T11.1 Verify all site links  Est: 30m
- [x] T11.2 Verify code examples compile  Est: 45m
- [ ] T11.3 Run Lighthouse audit  Est: 15m
- [ ] T11.4 Test mobile responsiveness  Est: 15m
- [ ] T11.5 Test search functionality  Est: 15m

---

## Long-Term Roadmap (Remaining Tasks)

### Community and DevRel

- [ ] T4.7 Record 15-minute video walkthrough of Zerfoo
  Owner: DevRel  Est: 4h
  Acceptance: Video covers installation, model loading, text generation, API serving.

- [ ] T5.4 Create Discord server with channels
  Owner: DevRel  Est: 2h
  Acceptance: Discord server with roles, channels, GitHub notification bot.

- [ ] T11.1 Sponsor GopherCon 2027 booth
  Owner: DevRel  Est: 2h

- [ ] T11.4 Recruit 5 external co-maintainers
  Owner: Lead Eng  Est: 4h

### ROCm Backend (E8) [Q1-Q3 2027]

- [ ] T8.1 Acquire AMD Instinct GPU access  Est: 2h
- [ ] T8.2 Validate all purego HIP bindings on AMD hardware  Est: 6h  depends: T8.1
- [ ] T8.3 Validate rocBLAS GEMM parity with cuBLAS  Est: 4h  depends: T8.1
- [ ] T8.4 Port custom CUDA kernels to HIP in ztensor  Est: 8h  depends: T8.2
- [ ] T8.5 Benchmark ROCm vs CUDA throughput  Est: 2h  depends: T8.4
- [ ] T8.6 Add ROCm to CI pipeline  Est: 3h  depends: T8.4

### Enterprise and Compliance

- [ ] T12.4 Sign first 5 enterprise support contracts ($500K ARR)
  Owner: Biz Dev  depends: T12.1, T12.2

- [ ] T19.1 Complete SOC 2 Type II audit
  Owner: Compliance  depends: T14.4

### Edge Deployment (E15)

- [ ] T15.3 Cross-compile and test on Raspberry Pi 5  Est: 3h
- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano  Est: 3h

### Performance

- [ ] T16.3 Benchmark 500+ tok/s [DGX, high-bandwidth GPU]
  Est: 2h. Blocker: GB10 roofline ~257 tok/s. Needs A100/H100.

- [ ] T9.4 Multi-GPU inference benchmark on Llama 3 70B
  Est: 2h. Blocker: DGX Spark has single GB10 GPU.

- [ ] T20.3 Benchmark Metal vs CPU on Apple M4 Max  Est: 2h  depends: T20.2

- [ ] T29.4 Benchmark on-device inference  Est: 2h  depends: T29.2, T29.3

### ZerfooConf (E26)

- [ ] T26.2 Execute ZerfooConf Day  Est: 8h  depends: T26.1
- [ ] T26.3 Plan standalone ZerfooConf 2032  Est: 6h  depends: T26.2

### FedRAMP (E30) [Q1-Q4 2034]

- [ ] T30.1 Engage FedRAMP 3PAO  Est: 4h  depends: T19.1
- [ ] T30.2 Implement FedRAMP controls (NIST 800-53)  Est: 12h  depends: T30.1
- [ ] T30.3 Complete FedRAMP authorization  Est: 4h  depends: T30.2

### IPO Preparation (E31) [Q1-Q4 2035]

- [ ] T31.1 Form board of directors
- [ ] T31.2 Engage Big 4 audit firm
- [ ] T31.3 Hire VP Sales and VP Marketing
- [ ] T31.4 Achieve $150M+ ARR
- [ ] T31.5 Draft S-1 registration  depends: T31.1, T31.2, T31.4

### Architecture Expansion (E32)

- [ ] T32.2 Validate 100+ model architectures  depends: T32.1 (done)

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M0 | Internal Consumer Bridge | WE1, WE13 | DONE | 2026-03-19 |
| M0.5 | Advanced Tabular | WE2-WE4 | DONE | 2026-03-19 |
| M1 | Inference Excellence | E2, WE13 | 300+ tok/s; 12+ archs; 5K stars | 2026-12-31 |
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
| R23 | GPU segfaults on large MatMul | Critical | High | E114 T114.2 adds VRAM bounds check |
| R24 | Stale DGX kernel library | High | High | E114 T114.1 syncs repos |

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

### Granite Time Series
- [TTM Paper (NeurIPS 2024)](https://arxiv.org/pdf/2401.03955)
- [TSPulse Paper](https://arxiv.org/pdf/2505.13033)
- [HuggingFace: granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- [HuggingFace: granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1)
- [HuggingFace: granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)

### Granite Guardian
- [Granite Guardian 3.3 8B](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)
- [Granite Guardian Paper (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)
- [Ollama granite3-guardian](https://ollama.com/library/granite3-guardian)

### K-Quant
- Q4_K GEMV kernel is 45% slower than Q4_0 GEMV on GB10.
- Infrastructure (virtual transpose, merged QKV/GateUp) is merged and ready.
- All K-quants reverted to Q4_0 re-quantization pending kernel optimization.

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
