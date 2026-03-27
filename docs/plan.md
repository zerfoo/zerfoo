# Zerfoo Work Plan

## Overview

This is the single consolidated plan for the Zerfoo ML framework. It combines
the main 5-year product roadmap with all satellite plans (Granite Time Series,
Granite Guardian, K-Quant optimization, multi-model benchmarks, batched GPU
training, GGUF writer consolidation, and documentation site).

Task statuses updated 2026-03-26 based on merged PRs and git history.

**Status summary:**
- 330+ tasks completed across all plans
- ~20 active tasks remaining (details below)
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

All models now produce coherent output on CPU and GPU after two critical fixes
(ztensor v0.6.3 repeat semantics, zerfoo v1.25.5 flash attention decode).
The next phase focuses on comprehensive benchmarking, quality verification,
kernel optimization, and time-series parity.

### P1: Full Multi-Model Benchmark (highest priority)

All models work — time to build the definitive comparison table.

- [x] BMK-T1 Download missing GGUFs  Est: 1h  2026 03 27
  Qwen 2.5 7B Q4_K_M downloaded from bartowski (4.4GB, single file).
  Gemma 3 4B Q4_K_M needs HuggingFace auth (gated model).
  Also verified: Phi 3.5 mini loads (was blocked by BMK-T4 metadata fix).
  Llama 3.1 8B loads via mmap. Cleaned 130GB of old ZMF/ONNX files from DGX.

- [x] BMK-T2 Re-run bench-compare-ollama.sh for all models  Est: 2h  DONE 2026-03-27
  Deps: BMK-T1
  Models: Gemma3-1B, DeepSeek-R1-1.5B, Llama3.2-3B, Mistral-7B.
  3-run median on DGX Spark, 128 tokens, greedy sampling.
  Results: Gemma3 235 (1.25x), DeepSeek 186 (1.11x), Llama 92 (0.99x), Mistral 44 (1.00x).
  All models produce coherent output. JSON: results/benchmark-2026-03-27.json.

- [ ] BMK-T3 Update website and README with full comparison table  Est: 1h
  Deps: BMK-T2
  Merge JSON results, calculate Zerfoo/Ollama ratios, update docs/benchmarks/
  page and README.md performance claims. File GitHub issues for any model
  where Zerfoo < 0.95x Ollama.
  Acceptance: Published table with 6+ models, ratios, and hardware specs.

- [x] BMK-T4 Investigate Phi3/Llama3.1 GGUF load failures  Est: 2h  2026 03 27
  Root cause: GetUint32/GetFloat32 only matched uint32/float32 type assertions.
  HuggingFace GGUFs for Phi3/Llama3.1 store dimensions as uint64. Added type
  switch to handle uint64/int32/int64/float64. Fix shipped in commit 1648db9.

### P2: Mistral vs Ollama Head-to-Head

Mistral is the highest-profile model family after Llama. Verify the quality
and performance claims extend beyond Gemma.

- [ ] MHH-T1 Run Mistral 7B quality comparison  Est: 2h
  Run identical prompts (5 diverse: factual, creative, code, reasoning,
  instruction-following) through both Zerfoo and Ollama on same GGUF.
  Score output quality subjectively (1-5) and compare token-for-token
  agreement with greedy sampling.
  Acceptance: Quality scores documented; token agreement > 95% with greedy.

- [ ] MHH-T2 Profile Mistral 7B performance gap  Est: 1h
  Deps: MHH-T1
  Current: Zerfoo ~44 tok/s vs Ollama ~37 tok/s (estimated). Verify the
  advantage ratio. If Zerfoo < Ollama, profile to identify bottleneck.
  Acceptance: Confirmed ratio documented in devlog with 3-run median.

- [ ] MHH-T3 Test sliding window attention correctness  Est: 2h
  Mistral uses sliding window attention (4096 tokens). Generate a long
  prompt (> 4096 tokens) and verify output remains coherent past the
  window boundary. Compare against Ollama on same prompt.
  Acceptance: Coherent output at 5000+ tokens; no degradation past window.

### P3: K-Quant Kernel Optimization

All in ztensor repo except KQ-T4. Q4_K is 45% slower than Q4_0 — closing
this gap improves quality (K-quants are more accurate) AND potentially speed.

- [ ] KQ-T1 Profile Q4_K vs Q4_0 GEMV  Est: 2h
  repo: ztensor
  ncu/nsys not available for purego kernels. Use Go benchmarks with timer
  instrumentation: measure kernel dispatch, memory transfer, and compute
  separately. Compare register pressure via PTX disassembly.
  Acceptance: Root cause of 45% slowdown identified and documented.

- [ ] KQ-T2 Optimize Q4_K GEMV kernel  Est: 4h
  repo: ztensor  Deps: KQ-T1
  Apply optimizations: reduce register usage, improve coalesced access,
  shared memory for sub-block scales, reduce warp divergence, tune block size.
  Acceptance: Measurable improvement in BenchmarkGEMV.

- [ ] KQ-T3 Benchmark and re-enable native Q4_K loading  Est: 2h
  Deps: KQ-T2
  Target: >= 215 tok/s on Gemma 3 1B (within 10% of Q4_0's 236 tok/s).
  If met, remove Q4_K re-quantization in model/gguf/loader.go. All
  infrastructure (virtual transpose, merged QKV) already merged.
  Acceptance: Gemma 3 1B Q4_K_M >= 215 tok/s natively, all tests pass.

### P4: Granite TS Parity Tests

- [ ] GTS-T1 Generate Python golden files  Est: 4h
  Run Python granite-tsfm reference on fixed inputs for TTM (ETTh1
  forecasting, 3 cases), FlowState (3 cases across sampling rates),
  TSPulse (2 anomaly detection, 2 classification). Save outputs as
  JSON golden files in tests/parity/testdata/.
  Acceptance: 10 golden files checked in with input/output pairs.

- [ ] GTS-T2 Run Zerfoo against golden files  Est: 4h
  Deps: GTS-T1
  Files: tests/parity/granite_ts_test.go
  Load same GGUF models, run same inputs, compare within tolerance (1e-4
  for F32). Test all three families: TTM, FlowState, TSPulse.
  Acceptance: All 10 test cases pass within 1e-4 tolerance.

- [ ] GTS-T3 Benchmark latency vs Python granite-tsfm  Est: 2h
  Deps: GTS-T2
  Measure single-series latency (batch=1) and throughput (batch=128) on
  DGX Spark GPU. Compare against Python granite-tsfm on same hardware.
  Target: 5-10x latency improvement.
  Acceptance: Results recorded in devlog with comparison table.

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
| R25 | ~~Mistral forward pass bug~~ | ~~High~~ | ~~High~~ | RESOLVED: GQA repeat fix (ztensor v0.6.3) + flash attn decode fix (v1.25.5) |

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
- ncu/nsys not directly usable with purego kernels; profile via Go benchmarks + PTX.

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
