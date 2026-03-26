# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, and documentation site).

Task statuses updated 2026-03-26 based on merged PRs and git history.

**Status summary:**
- 320+ tasks completed across all plans
- ~30 active tasks remaining (details below)

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

### Multi-Model Benchmarks (partial)

Environment setup, benchmark script, small model results: Gemma3-1B 236 tok/s
(1.16x Ollama), DeepSeek-R1 193 (1.04x), Llama3.2 96 (0.98x). Medium model:
Mistral-7B 44 tok/s with CUDA graph fix but garbage output (PRs #174-178).

### GPU Verification (E114, 7/7 complete)

VRAM bounds check, FP16 MatMul fix, kernel rebuild, BF16 tolerance, timeseries
GPU verify, full suite report — all pass on DGX Spark.

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

## Active Work — Next Phase

### P1: Mistral Forward Pass Fix (highest priority)

Mistral 7B tokenizer now matches HuggingFace exactly (token IDs verified), but
inference still produces garbage output. The issue is in the model forward pass,
not tokenization. CUDA graph fix restored 44 tok/s (from 11.6), but output is
incoherent. This is the highest-impact bug — Mistral is a major model family.

- [ ] MFP-T1 Audit buildMistralGraph weight name mapping  Est: 2h
  Files: inference/arch_mistral.go, inference/load_gguf.go
  Description: Mistral GGUFs from HuggingFace use "llama.*" tensor name
  prefixes. Verify that buildMistralGraph maps every GGUF tensor name to
  the correct graph node. Log all tensor names at load time and compare
  against expected layer structure (32 layers, GQA with 8 KV heads).
  Acceptance: Printed tensor mapping matches expected Mistral 7B architecture.

- [ ] MFP-T2 Compare layer activations against llama.cpp reference  Est: 4h
  Files: inference/arch_mistral.go, tests/parity/mistral_activations_test.go
  Deps: MFP-T1
  Description: Add activation dumping at key points (after embedding, after
  first RMSNorm, after first attention, after first FFN, final logits).
  Run same prompt through llama.cpp with `--log-activations`, compare values.
  Identify the first layer where activations diverge.
  Acceptance: Root cause layer identified; divergence documented in devlog.

- [ ] MFP-T3 Verify BOS token handling in generation pipeline  Est: 1h
  Files: generate/generator.go, generate/prompt.go
  Description: Mistral requires BOS token (token ID 1) prepended to input.
  Verify the generation pipeline adds BOS for Mistral. Check if the
  tokenizer config from GGUF metadata sets add_bos_token correctly.
  Acceptance: BOS handling confirmed correct or fixed.

- [ ] MFP-T4 Fix and validate Mistral end-to-end  Est: 2h
  Deps: MFP-T2, MFP-T3
  Description: Apply fix identified in MFP-T2/T3. Run end-to-end generation
  on DGX Spark: load Mistral 7B Q4_K_M, generate 128 tokens with greedy
  sampling, verify coherent English output.
  Acceptance: Coherent output, >= 40 tok/s, benchmark recorded in devlog.

### P2: Granite Time Series — 2 tasks remaining

- [ ] GTS-T5.3 Accuracy parity tests against Python granite-tsfm  Est: 8h
  Owner: ML Eng
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

- [ ] GTS-T5.4 Benchmark suite and performance optimization  Est: 8h
  Owner: Kernel Eng
  Files: tests/benchmark/granite_ts_bench_test.go
  Deps: GTS-T5.3
  Description: Comprehensive benchmarks comparing Zerfoo vs Python granite-tsfm:
  latency (single-series), throughput (batch 1/8/32/128), accuracy
  (MAE/MSE on ETTh1, Weather, Electricity), memory (peak RSS).
  Target: 5-10x latency improvement over Python.
  Acceptance:
  - TTM inference < 2ms/series on GPU (batch=1).
  - FlowState inference < 5ms/series on GPU (batch=1).
  - Throughput > 10,000 series/sec on GPU (batch=128) for TTM.
  - All accuracy within 1% of Python reference.

### P3: Multi-Model Benchmarks — 6 tasks remaining

- [ ] BMK-T1 Acquire remaining GGUF files  Est: 1h
  Missing: gemma3-4b, qwen2.5-7b, mixtral-8x7b, command-r-35b, falcon-7b,
  mamba-2.8b, rwkv-7b. Note: phi3 and llama3.1 GGUFs have format mismatches
  that need GGUF parser investigation.

- [ ] BMK-T2 Benchmark medium models (4B-7B)  Est: 1h
  Deps: BMK-T1
  Models: Gemma3-4B, Qwen2.5-7B, Falcon-7B. Run on DGX Spark.

- [ ] BMK-T3 Benchmark large models (8x7B, 35B)  Est: 1h
  Deps: BMK-T1
  Models: Mixtral 8x7B, Command R 35B. May not fit in 128GB.

- [ ] BMK-T4 Benchmark alternative architectures  Est: 30m
  Deps: BMK-T1
  Mamba 2.8B, RWKV 7B (Zerfoo only, no Ollama comparison).

- [ ] BMK-T5 Compile and publish results  Est: 1h
  Deps: BMK-T2, BMK-T3, BMK-T4
  Merge JSON results, calculate ratios, update website benchmarks page,
  update README.md claims, file GitHub issues for regressions, record in devlog.

- [ ] BMK-T6 Investigate Phi3/Llama3.1 GGUF load failures  Est: 2h
  Deps: BMK-T1
  Both models fail to load from HuggingFace GGUFs. Investigate parser
  compatibility — may be missing GGUF v3 metadata fields or tensor types.

### P4: K-Quant Kernel Optimization — 4 tasks remaining

All in ztensor repo except T4.4.

- [ ] KQ-T1 Profile Q4_K vs Q4_0 GEMV with Nsight Compute  Est: 2h
  repo: ztensor
  Use `ncu` on DGX Spark. Collect register usage, occupancy, memory
  throughput, warp stalls, L1/L2 cache hit rates.
  Acceptance: Profiling report with root cause of 45% slowdown identified.

- [ ] KQ-T2 Optimize Q4_K GEMV kernel  Est: 4h
  repo: ztensor  Deps: KQ-T1
  Apply optimizations: reduce register usage, improve coalesced access,
  shared memory for sub-block scales, reduce warp divergence, tune block size.
  Acceptance: Measurable improvement in BenchmarkGEMV.

- [ ] KQ-T3 Benchmark Q4_K GEMV at target  Est: 1h
  repo: ztensor  Deps: KQ-T2
  Target: >= 215 tok/s on Gemma 3 1B (within 10% of Q4_0's 236 tok/s).

- [ ] KQ-T4 Re-enable native Q4_K loading in zerfoo  Est: 1h
  Deps: KQ-T3
  Remove Q4_K re-quantization in model/gguf/loader.go. All infrastructure
  (virtual transpose, merged QKV) already merged and ready.
  Acceptance: Gemma 3 1B Q4_K_M >= 215 tok/s, all tests pass.

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
| R25 | Mistral forward pass bug erodes multi-model credibility | High | High | P1 priority fix; activation comparison vs llama.cpp |

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
