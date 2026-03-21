# Zerfoo Product Roadmap (2026-2036)

## Context

### Problem Statement

Zerfoo is a production-grade ML inference and training framework written entirely in
Go (zero CGo by default). The framework runs 18+ model architectures at 245 tok/s
on Gemma 3 1B Q4_K_M (20% faster than Ollama). The 5-year technical foundation is
complete: PagedAttention, FP8/NVFP4, speculative decoding, LoRA/QLoRA, FSDP,
multi-modal, agentic tool-use, NAS/AutoML, and computation graph compilation.

An internal consumer needs tabular and time-series ML capabilities that Zerfoo does
not yet provide. This consumer currently misuses LLM inference for tabular prediction
(formatting numeric features as text prompts) and has hand-rolled pure Go CNN/TabNet
implementations instead of using Zerfoo's Trainer[T] and GPU acceleration. Bridging
this gap is the top priority and blocks the consumer from going live.

The roadmap serves two masters:
1. **Internal consumer** -- needs tabular/time-series ML for signal prediction
2. **Open-source community** -- needs LLM inference excellence for Go developers

Internal consumer needs take priority when they conflict (per Chairman directive).

See docs/VISION.md for the full 10-year product vision and revenue model.
See docs/design.md for the technical architecture (29 sections, 62 ADRs).

### Objectives

- **Immediate (2026 Q2-Q3):** Tabular ensemble. Advanced tabular architectures
  (FTTransformer, TabNet variant, SAINT, ResNet). Time-series architectures
  (TFT, N-BEATS, PatchTST). AutoML for tabular/time-series search. Fix GPU tests.
- **Near-term (2026 Q3-Q4):** 300+ tok/s. Transfer learning for tabular.
- **Year 1-2 (2026-2027):** 12+ LLM architectures validated. ROCm parity.
  v1.0 stable release. 25,000+ stars.
- **Year 3-4 (2028-2029):** Enterprise foundation. $500K-$2M ARR. SOC 2.
  Cloud marketplace. Edge deployment.
- **Year 5-6 (2030-2031):** Training platform. $10M-$25M ARR. Multi-accelerator.
- **Year 7-10 (2032-2036):** Industry standard. $100M-$150M+ ARR. IPO readiness.

### Non-Goals

- Pre-training at scale (100B+ parameters). Focus is inference + fine-tuning.
- Python API or Python bindings. Go-first; Python users use the OpenAI-compatible API.
- Custom ASIC backends. Support NVIDIA, AMD, Intel, Apple only.
- Runtime ONNX execution. zonnx converts ONNX to GGUF at build time.
- Internal consumer repo tasks. Only zerfoo and ztensor tasks are in this plan.

### Constraints and Assumptions

- Primary hardware: DGX Spark at ssh ndungu@192.168.86.250 (GB10, sm_121).
- Go 1.25+ required (generics, range-over-func).
- All GPU bindings via purego/dlopen; no CGo in core packages.
- GGUF is the sole model format; zonnx handles ONNX conversion.
- Each repo (ztensor, ztoken, zerfoo, zonnx, float16, float8) is independent.
- Apache 2.0 license for all core repos (see ADR-057).
- Tests use standard library only (no testify, no cobra).
- Agentic coders execute parallel waves; human review gates at milestones.
- metee v1.0.1 is stable and provides LightGBM/XGBoost bindings.

### Success Metrics

| Year | Metric | Target |
|------|--------|--------|
| 2026 Q2 | Internal consumer uses tabular.Train | Shipped |
| 2026 Q3 | Advanced tabular architectures | 7+ models |
| 2026 | Decode tok/s (1B Q4_K_M) | 300+ |
| 2026 | GitHub stars (all repos) | 5,000+ |
| 2026 | Supported LLM architectures | 12+ |
| 2027 | GitHub stars | 25,000+ |
| 2027 | v1.0 stable release | Shipped |
| 2028 | ARR | $500K |
| 2030 | ARR | $10M |
| 2036 | ARR | $150M+ |

### Research Findings

Research conducted by three parallel agents on 2026-03-18. Key findings:

**Technical Landscape:**
- Ollama (165K stars) wraps llama.cpp C++ -- not native Go. Zerfoo is the only
  framework combining native Go + zero CGo + library-first + competitive tok/s.
- No Go-native tabular ML framework exists. Zerfoo would be the first.
- Enterprise ML tooling valuations are strong (W&B $50M ARR, Replicate $5.3M ARR
  acquired for $550M, Modal $1.1B valuation).

**Risks:**
- Go ML TAM ceiling is the top risk. Mitigation: expand via OpenAI API and edge runtime.
- Apache 2.0 fork risk. Mitigation: innovation velocity (ADR-057).
- AI-generated code quality. Mitigation: security audit, DGX validation, fuzz testing.

**Architecture Patterns:**
- v1.0 API: freeze Engine[T], extension interfaces (ADR-058).
- Plugin architecture: in-process init() registration (Go database/sql pattern).
- Cloud: Model Repository pattern, Kubernetes operator, token billing (ADR-060).
- Edge: build-tag-gated minimal binary, pre-optimized GGUF models (ADR-059).

---

## Discovery Summary

**Work type:** Engineering (primary), Strategy (enterprise/cloud phases)

**Use cases discovered:** 37 total
- P0 (core workflows): 7 (UC-001 through UC-005, UC-010, UC-015 through UC-018)
- P1 (important features): 19 (UC-006 through UC-009, UC-019 through UC-034)
- P2 (secondary): 11 (UC-011 through UC-014, UC-022, UC-035 through UC-037)

**Wiring status:**
- WIRED: 26 use cases (interfaces exist, tests pass)
- PLANNED: 10 use cases (no code yet -- tabular advanced, timeseries, transfer learning)
- STUB: 1 use case (UC-035 ROCm -- bindings exist, no hardware validation)

**Gaps identified:**
- Tabular ensemble (UC-018): W1.1.4 planned, blocks internal consumer advanced use
- Advanced tabular architectures (UC-025 through UC-027): 3 PLANNED use cases
- Time-series forecasting (UC-028 through UC-030): 3 PLANNED use cases
- AutoML tabular/timeseries (UC-031): extends existing automl package
- Transfer learning (UC-032, UC-033): PLANNED, blocks per-source specialization
- ROCm validation (UC-035): STUB, needs AMD hardware

**Reference:** .claude/scratch/usecases-manifest.json

---

## Scope and Deliverables

### In Scope

- Tabular ensemble model (internal consumer blocker)
- Advanced tabular architectures (FTTransformer, TabNet, SAINT, ResNet)
- Time-series architectures (TFT, N-BEATS, PatchTST)
- AutoML extension for tabular/time-series architecture search
- Performance optimization to 300+ tok/s (Year 1), 500+ (Year 3), 1000+ (Year 7)
- ROCm parity and hardware validation
- v1.0 stable release and documentation
- Enterprise support, SOC 2, cloud marketplace
- Edge deployment, on-device inference
- Transfer learning, RL, cross-asset models, regime detection (later phases)
- Provenance tracking, continuous learning, meta-learning (later phases)

### Out of Scope

- ZMF model format (archived, replaced by GGUF per ADR-037)
- CGo-based GPU bindings (purego/dlopen is the standard)
- Python SDK or CLI wrappers
- Pre-training runs for 100B+ models
- Custom hardware or kernel microarchitecture below CUDA level
- Payment processing (billing uses Stripe webhooks)
- Internal consumer repo tasks (separate repo, separate plan)

### Deliverables Table

| ID | Description | Owner Role | Acceptance Criterion |
|----|-------------|------------|----------------------|
| D0 | Tabular model package for internal consumer | ML Eng | tabular.Train + Predict + Save/Load + Ensemble working on GPU |
| D1 | 12+ model architectures validated | Arch Eng | All produce coherent output; parity tests pass on DGX |
| D2 | 300+ tok/s decode | Kernel Eng | Gemma 3 1B Q4_K_M >= 300 tok/s on DGX Spark |
| D3 | v1.0 stable release | Lead Eng | API freeze, 2-year guarantee, release-please tag |
| D4 | Documentation site | DevRel | Quickstart, API ref, cookbook, architecture tour live |
| D5 | 5,000+ GitHub stars | DevRel | Organic stars across all repos |
| D6 | ROCm CUDA parity | Kernel Eng | All GPU ops pass on AMD Instinct; benchmark within 20% |
| D7 | Enterprise support tier | Biz Dev | SLA contracts, Slack channel, ticketing system live |
| D8 | SOC 2 Type II | Compliance | Audit report issued by 3PAO |
| D9 | Zerfoo Cloud GA | Platform Eng | Multi-tenant, marketplace listed, 99.9% uptime SLO |
| D10 | Zerfoo Runtime | Arch Eng | <10MB ARM64 binary, inference on Raspberry Pi 5 |
| D11 | Kubernetes operator | Platform Eng | ZerfooInferenceService CRD, autoscaling, canary |
| D12 | Apple Metal backend | Kernel Eng | All GPU ops pass on M-series; benchmark published |

---

## Checkable Work Breakdown

### PRIORITY 1: Tabular and Time-Series ML (Internal Consumer Blocker)

These tasks are the highest priority and must be completed before any remaining
10-year roadmap tasks. Decision rationale: docs/adr/062-tabular-model-package.md

---

#### WE1: Tabular Model Package [2026 Q2 -- CRITICAL]

Completed: W1.1.1-W1.1.4. Trimmed 2026-03-19.

---

#### WE2: Advanced Tabular Architectures [2026 Q3]

Completed: W2.1.1-W2.1.4. Trimmed 2026-03-19.

---

#### WE3: Time-Series Architectures [2026 Q3]

Completed: W2.2.1-W2.2.3. Trimmed 2026-03-19.

---

#### WE4: Tabular AutoML Extension [2026 Q3-Q4]

Completed: W2.3.1. Trimmed 2026-03-19.

---

#### WE5: Transfer Learning for Tabular (Internal Consumer) [2027-2028]

Completed: W5.1.1-W5.1.3. Trimmed 2026-03-19.

---

#### WE6: Reinforcement Learning Package [2028-2029]

Completed: W6.1.1-W6.1.3. Trimmed 2026-03-19.

---

#### WE7: Cross-Asset and Causal Models [2029-2030]

Completed: W7.1.1-W7.1.2, W7.3.1-W7.3.2. Trimmed 2026-03-19.

---

#### WE8: Regime Detection and Synthetic Data [2030-2031]

Completed: W8.1.1, W8.2.1-W8.2.2. Trimmed 2026-03-19.

---

#### WE9: Self-Improving Systems [2031-2032]

Completed: W9.1.1, W9.2.1, W9.3.1. Trimmed 2026-03-19.

---

#### WE10: Hardware Optimization for Tabular [2032-2034]

Completed: W10.1.1-W10.1.3. Trimmed 2026-03-19.

---

#### WE11: Enterprise Features (Internal Consumer) [2032-2034]

Completed: W10.2.1-W10.2.3. Trimmed 2026-03-19.

---

#### WE12: Continuous Learning and Provenance [2034-2036]

Completed: W11.1.1-W11.1.2, W11.2.1, W11.3.1. Trimmed 2026-03-19.

---

### PRIORITY 2: Inference Performance and Bug Fixes

These tasks overlap with the 10-year roadmap but are also needed by the internal
consumer. They can run in parallel with Priority 1 tasks.

---

#### WE13: Performance and Test Fixes [2026 Q2 -- parallel with WE1]

Completed: W3.1.1-W3.1.5. Trimmed 2026-03-19.

---

### PRIORITY 3: 10-Year Product Roadmap (Remaining Tasks)

These tasks continue the original 10-year roadmap. They are lower priority than
Priority 1 and 2 tasks but should be scheduled when agent capacity is available.

---

#### E2: New Model Architecture Support [Q1-Q3 2026]

Completed: T2.1-T2.12 (all architectures validated including Mamba 3 parity on DGX).
Trimmed 2026-03-20.

---

#### E4: Documentation and Developer Experience [Q1-Q3 2026]

Completed: T4.1-T4.6 (docs site, quickstart, API ref, architecture tour, cookbook,
benchmark guide). Trimmed 2026-03-18.

- [ ] T4.7 Record 15-minute video walkthrough of Zerfoo
  Owner: DevRel  Est: 4h  delivers: [Zerfoo video walkthrough on YouTube]
  Deps: none (T4.2 complete)
  Acceptance: Video covers installation, model loading, text generation, and
  OpenAI API serving. Published on YouTube.

---

#### E5: Community Infrastructure [Q1-Q2 2026]

Completed: T5.1-T5.3, T5.5-T5.7 (CONTRIBUTING.md, starter issues, Discussions,
blog posts, GopherCon proposal, example apps). Trimmed 2026-03-18.

- [ ] T5.4 Create Discord server with channels
  Owner: DevRel  Est: 2h  delivers: [Zerfoo Discord community server]
  Deps: none
  Acceptance: Discord server with roles, channels, and bot for GitHub notifications.

---

#### E8: ROCm Backend Hardware Validation [Q1-Q3 2027]

- [ ] T8.1 Acquire AMD Instinct GPU access
  Owner: Infra Eng  Est: 2h  verifies: [UC-035]
  Deps: none
- [ ] T8.2 Validate all purego HIP bindings on AMD hardware
  Owner: Kernel Eng  Est: 6h  verifies: [UC-035]
  Deps: T8.1
- [ ] T8.3 Validate rocBLAS GEMM parity with cuBLAS
  Owner: Kernel Eng  Est: 4h  verifies: [UC-035]
  Deps: T8.1
- [ ] T8.4 Port custom CUDA kernels to HIP in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h  verifies: [UC-035]
  Deps: T8.2
- [ ] T8.5 Benchmark ROCm vs CUDA throughput [AMD]
  Owner: Kernel Eng  Est: 2h  verifies: [UC-035]
  Deps: T8.4
- [ ] T8.6 Add ROCm to CI pipeline
  Owner: Infra Eng  Est: 3h  verifies: [infrastructure]
  Deps: T8.4

---

#### E9: Multi-GPU Inference [Q2-Q3 2027]

Completed: T9.1-T9.3 (tensor parallelism, pipeline parallelism, --gpus flag).
Trimmed 2026-03-18.

- [ ] T9.4 Benchmark: multi-GPU inference on Llama 3 70B [DGX, multi-GPU]
  Owner: Infra Eng  Est: 2h  verifies: [UC-024]
  Deps: none (T9.3 complete)
  Blocker: DGX Spark has single GB10 GPU. Needs multi-GPU system.

---

#### E10: Vision-Language Model Expansion [Q3-Q4 2027]

Completed: T10.1-T10.3. Trimmed 2026-03-19.

---

#### E11: Community Growth to 25,000 Stars [Q1-Q4 2027]

- [ ] T11.1 Sponsor GopherCon 2027 booth
  Owner: DevRel  Est: 2h  delivers: [GopherCon 2027 booth presence]
  Deps: none
- [x] T11.2 Publish tutorial series (5 parts) (2026-03-19)
  Owner: DevRel  Est: 8h  delivers: [5-part tutorial series published]
  Deps: none
- [x] T11.3 Submit KubeCon 2027 talk (2026-03-19)
  Owner: DevRel  Est: 3h  delivers: [KubeCon 2027 CFP submission]
  Deps: none
- [ ] T11.4 Recruit 5 external co-maintainers
  Owner: Lead Eng  Est: 4h  delivers: [5 external co-maintainers onboarded]
  Deps: none
- [x] T11.5 Integrate with LangChain-Go and Weaviate (2026-03-18)
  Owner: DevRel  Est: 6h  verifies: [UC-001, UC-007]
  Deps: none

---

#### E12: Enterprise Support Tier [Q1-Q2 2028]

- [x] T12.1 Define enterprise support SLA tiers (2026-03-19)
  Owner: Biz Dev  Est: 2h  delivers: [enterprise SLA tier definitions]
  Deps: none
- [x] T12.2 Set up enterprise ticketing system (2026-03-19)
  Owner: Biz Dev  Est: 3h  delivers: [enterprise ticketing system live]
  Deps: T12.1
- [x] T12.3 Create enterprise deployment guide (2026-03-19)
  Owner: DevRel  Est: 4h  delivers: [enterprise deployment guide published]
  Deps: none
- [ ] T12.4 Sign first 5 enterprise support contracts ($500K ARR)
  Owner: Biz Dev  Est: ongoing  delivers: [$500K ARR enterprise contracts]
  Deps: T12.1, T12.2

---

#### E13: Security Audit and Hardening [Q2-Q3 2028]

Completed: T13.3 (SBOM generation), T13.4 (fuzz testing). Trimmed 2026-03-18.
T13.1 superseded by internal deep review (E106). T13.2 superseded by E106 tasks.

- [x] T13.1 Deep security review of v1.10.0 (2026-03-21)
  Owner: Lead Eng  Est: 2h  delivers: [deep review report in .claude/scratch/deep-review-report.md]
  Result: 10-agent review found 2 Critical, 11 High, 24 Medium, 9 Low, 4 Info. E106 created.
- [x] T13.2 Plan remediation for all findings (2026-03-21)
  Owner: Lead Eng  Est: 1h  verifies: [infrastructure]
  Deps: T13.1
  Result: E106 created with 37 tasks across 8 waves. ADR-065 created.

---

#### E14: SOC 2 Certification [Q3-Q4 2028]

Completed: T14.1-T14.4. Trimmed 2026-03-20.

---

#### E15: Edge Deployment (Zerfoo Runtime) [Q2-Q4 2028]

Decision: docs/adr/059-edge-runtime-architecture.md

Completed: T15.1 (build-tag-gated edge binary), T15.2 (pre-optimized model format),
T15.5 (ARM64 CI). Trimmed 2026-03-18.

- [ ] T15.3 Cross-compile and test on Raspberry Pi 5
  Owner: Arch Eng  Est: 3h  verifies: [UC-022]
  Deps: none (T15.1 complete)
- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano
  Owner: Arch Eng  Est: 3h  verifies: [UC-022]
  Deps: none (T15.1 complete)

---

#### E16: Performance Optimization to 500+ tok/s [Q3-Q4 2028]

Completed: T16.1 (warp-specialized GEMV), T16.2 (KV cache FP8 quantization).
Trimmed 2026-03-18.

- [ ] T16.3 Benchmark: 500+ tok/s [DGX, high-bandwidth GPU]
  Owner: Kernel Eng  Est: 2h  verifies: [UC-002]
  Deps: T16.1
  Blocker: GB10 roofline is ~257 tok/s (200 GB/s BW, 778 MB model). 500 tok/s
  needs A100/H100 class memory bandwidth.

---

#### E17: Zerfoo Cloud GA [Q1-Q3 2029]

Completed: T17.1-T17.7. Trimmed 2026-03-20.

---

#### E18: Enterprise Features [Q2-Q4 2029]

Completed: T18.1-T18.5. Trimmed 2026-03-20.

---

#### E19: SOC 2 Type II Completion [Q1-Q2 2029]

- [ ] T19.1 Complete SOC 2 Type II audit
  Owner: Compliance  Est: 2h  delivers: [SOC 2 Type II audit report]
  Deps: T14.4

---

#### E20: Apple Metal Backend [Q1-Q2 2030]

Completed: T20.1-T20.2 (Metal compute shader bindings, kernel ports). Trimmed 2026-03-20.

- [ ] T20.3 Benchmark Metal vs CPU on Apple M4 Max
  Owner: Kernel Eng  Est: 2h  verifies: [UC-037]
  Deps: T20.2

---

#### E21: Intel SYCL Backend [Q2-Q3 2030]

Completed: T21.1-T21.2. Trimmed 2026-03-19.

---

#### E22: Auto-Optimization Framework [Q3-Q4 2030]

Completed: T22.1-T22.3. Trimmed 2026-03-19.

---

#### E24: Custom Model Architecture SDK [Q1-Q3 2031]

Completed: T24.1-T24.3. Trimmed 2026-03-19.

---

#### E25: Heterogeneous Compute [Q2-Q4 2031]

Completed: T25.1-T25.2. Trimmed 2026-03-19.

---

#### E26: ZerfooConf [2031-2032]

- [x] T26.1 Plan ZerfooConf Day (2026-03-19)
  Owner: DevRel  Est: 4h  delivers: [ZerfooConf Day event plan]
  Deps: none
- [ ] T26.2 Execute ZerfooConf Day
  Owner: DevRel  Est: 8h  delivers: [ZerfooConf Day executed with 200+ attendees]
  Deps: T26.1
- [ ] T26.3 Plan standalone ZerfooConf 2032
  Owner: DevRel  Est: 6h  delivers: [ZerfooConf 2032 plan and venue]
  Deps: T26.2

---

#### E27: Ecosystem Integrations [Q1-Q4 2031]

Completed: T27.1-T27.3. Trimmed 2026-03-19.

---

#### E28: Federated Learning [Q1-Q3 2033]

Completed: T28.1-T28.4. Trimmed 2026-03-19.

---

#### E29: On-Device Inference [Q2-Q4 2033]

Completed: T29.1-T29.3 (gomobile bindings, iOS demo, Android demo). Trimmed 2026-03-20.

- [ ] T29.4 Benchmark on-device inference
  Owner: Arch Eng  Est: 2h  verifies: [UC-022]
  Deps: T29.2, T29.3

---

#### E30: FedRAMP Authorization [Q1-Q4 2034]

- [ ] T30.1 Engage FedRAMP 3PAO
  Owner: Compliance  Est: 4h  delivers: [FedRAMP 3PAO engagement]
  Deps: T19.1
- [ ] T30.2 Implement FedRAMP controls (NIST 800-53)
  Owner: Infra Eng  Est: 12h  delivers: [NIST 800-53 controls implemented]
  Deps: T30.1
- [ ] T30.3 Complete FedRAMP authorization
  Owner: Compliance  Est: 4h  delivers: [FedRAMP ATO issued]
  Deps: T30.2

---

#### E31: IPO Preparation [Q1-Q4 2035]

- [ ] T31.1 Form board of directors
  Owner: CEO  Est: ongoing  delivers: [independent board seated]
  Deps: none
- [ ] T31.2 Engage Big 4 audit firm
  Owner: CFO  Est: 4h  delivers: [Big 4 audit engagement]
  Deps: none
- [ ] T31.3 Hire VP Sales and VP Marketing
  Owner: CEO  Est: ongoing  delivers: [VP Sales and VP Marketing hired]
  Deps: none
- [ ] T31.4 Achieve $150M+ ARR
  Owner: CEO  Est: ongoing  delivers: [$150M+ ARR achieved]
  Deps: all
- [ ] T31.5 Draft S-1 registration
  Owner: CFO  Est: 8h  delivers: [S-1 registration filed]
  Deps: T31.1, T31.2, T31.4

---

#### E32: Architecture Expansion to 100+ Models [2035-2036]

- [x] T32.1 Implement automated architecture builder from GGUF metadata (2026-03-19)
  Owner: Arch Eng  Est: 8h  verifies: [UC-001, UC-002]
  Deps: none
- [ ] T32.2 Validate 100+ model architectures
  Owner: Arch Eng  Est: ongoing  verifies: [UC-001, UC-002]
  Deps: T32.1

---

#### E33: Performance Target 1000+ tok/s [2032-2035]

Completed: T33.1-T33.2. Trimmed 2026-03-19.

---

#### E103: Throughput Regression Investigation and Fix [2026 Q2]

During T16.3 benchmarking on DGX Spark (2026-03-19), a ~40% throughput drop was
observed: 229 tok/s (old binary at 4e85b12) vs 136 tok/s (current HEAD at b81b616).

The devlog entry (2026-03-19) traced this to a dirty working tree where
`decodeQ4KTensor` in model/gguf/loader.go was experimentally changed to keep
native Q4KStorage instead of re-quantizing to Q4_0. Native Q4_K falls through to
cuBLAS SGEMM (~134 tok/s) while the optimized Q4_0 GEMV kernel achieves ~223 tok/s.
No commit introduced the regression.

However, the investigation is incomplete:
1. **benchmarks.md still reports 136 tok/s as "Current Baseline"** with a stale
   regression alert. This is misleading and needs updating with a verified clean build.
2. **No clean-build verification has been run** on current HEAD to confirm the
   committed code actually achieves 229-245 tok/s.
3. **A secondary gap exists**: peak was 245 tok/s (commit 4e85b12, 256 tokens) but
   the last clean measurement was 229 tok/s. The 6.5% difference (245 vs 229) may
   be measurement noise or a real micro-regression worth investigating.
4. **The current HEAD has ~90 new commits** since the 245 tok/s baseline. While the
   dirty tree explains 229 vs 136, none of the new code has been profiled for
   decode-phase overhead.

- [x] T103.1 Verify clean-build throughput on DGX Spark [DGX] (2026-03-20)
  Owner: Kernel Eng  Est: 1h  verifies: [UC-002, UC-003]
  Result: 156 tok/s on clean HEAD — regression confirmed real (not dirty tree).

- [x] T103.2 Bisect ztensor regression (2026-03-20)
  Owner: Kernel Eng  Est: 2h  verifies: [UC-002, UC-003]
  Result: Commit 33b54d9 (CUDA graph full-capture bypass) caused 245->195 tok/s.
  Reverted in ztensor commit 4d56fd6.

- [x] T103.3 Bisect zerfoo regression (2026-03-20)
  Owner: Kernel Eng  Est: 2h  verifies: [UC-002]
  Result: Q5_K/Q6_K loader change (float32 instead of Q4_0 re-quant) caused 245->187.
  Fixed in zerfoo commit 21c9f45 + test update 22b1c31.

- [x] T103.4 Update benchmarks.md with verified measurements (2026-03-20)
  Owner: Kernel Eng  Est: 30m  verifies: [infrastructure]
  Result: Updated current baselines to 244.45 tok/s (3 runs). Removed regression alert.

- [x] T103.5 Verify fix restores baseline (2026-03-20)
  Owner: Kernel Eng  Est: 30m  verifies: [UC-002, UC-003]
  Result: 244.45 / 244.18 / 244.62 tok/s on DGX Spark. Baseline restored (95% roofline).

---

#### E104: GitHub Issues Resolution (#105 HRM segfault) [2026 Q2]

GitHub issue #105: HRM Forward() segfaults on linux/arm64 (DGX Spark GB10)
with CGO_ENABLED=0 and tabular build tag. The crash occurs during the
recurrent loop in HRM.Forward() when uninitialized hidden states (containing
garbage data) are passed to ARM64 NEON assembly operations that have no
null/alignment checks.

Root cause (from code analysis):
1. NewHModule/NewLModule create HiddenState via tensor.New with nil data,
   which allocates uninitialized memory (Go zeroes the slice, but the tensor
   is [1, modelDim] with batch=1 regardless of actual batch size).
2. HRM.Forward() passes HiddenState directly to the recurrent loop without
   resizing to match the input batch dimension.
3. When Engine.Add() operates on tensors with mismatched shapes (batch=1 vs
   batch=N), the ARM64 NEON assembly in xblas/elementwise_arm64.s dereferences
   out-of-bounds memory, causing the segfault.

Files to fix:
- layers/hrm/h_module.go (HiddenState init + batch resize)
- layers/hrm/l_module.go (HiddenState init + batch resize)
- model/hrm/hrm.go (Forward: resize hidden states before recurrent loop)
- model/hrm/hrm_test.go (integration test for recurrent loop)

- [x] T104.1 Fix HModule/LModule hidden state initialization and batch handling (2026-03-20)
  Owner: ML Eng  Est: 2h  verifies: [UC-025, UC-016]
  Deps: none
  Acceptance:
  - In layers/hrm/h_module.go NewHModule: zero-initialize HiddenState explicitly
    (pass make([]T, modelDim) instead of nil to tensor.New). Use shape [1, modelDim].
  - In layers/hrm/l_module.go NewLModule: same zero-initialization fix.
  - In HModule.Forward: at entry, check if HiddenState batch dim matches input
    batch dim. If not, resize HiddenState to [batchSize, modelDim] with zeros.
  - In LModule.Forward: same batch dim resize.
  - In model/hrm/hrm.go Forward: before the recurrent loop, resize both
    m.HModule.HiddenState and m.LModule.HiddenState to match projectedInput
    batch dimension.
  - Tests: TestHModule_Forward and TestLModule_Forward pass with batch > 1.
  - `go build -tags tabular ./...` compiles cleanly.
  - `go vet ./model/hrm/ ./layers/hrm/` clean.

- [x] T104.2 Add integration test for HRM recurrent loop with real batch sizes (2026-03-20)
  Owner: ML Eng  Est: 1h  verifies: [UC-025]
  Deps: T104.1
  Acceptance:
  - New test TestHRM_Forward_RecurrentLoop in model/hrm/hrm_test.go.
  - Creates HRM with small config (dim=16, nSteps=2, tSteps=3).
  - Calls Forward() with batch of 32 samples and 10 features.
  - Verifies: no panic, output shape is [32, 1], output contains finite values.
  - Run: `go test -tags tabular -run TestHRM_Forward_RecurrentLoop -race ./model/hrm/`

- [x] T104.3 Verify fix on DGX Spark (linux/arm64) [DGX] (2026-03-20)
  Owner: ML Eng  Est: 1h  verifies: [UC-025, UC-016]
  Deps: T104.1, T104.2
  Result: 43 tests PASS on linux/arm64 (Go 1.26.1), no segfault, race clean.
  Acceptance:
  - SSH to DGX Spark (ndungu@192.168.86.250).
  - Sync zerfoo repo. Build: `CGO_ENABLED=0 go build -tags tabular -o train-hrm ./cmd/train/`.
  - Run the exact reproduction command from issue #105:
    `./train-hrm -backend hrm -data data.csv -asset COIN -horizon 15m -epochs 5 -hrm-n 2 -hrm-t 4 -hrm-dim 32`
  - Verify: no segfault, training completes, metrics printed.
  - Run tests on DGX: `go test -tags tabular -race -timeout 120s ./model/hrm/ ./layers/hrm/`

- [x] T104.4 Close GitHub issue #105 with fix evidence (2026-03-20)
  Owner: ML Eng  Est: 15m  delivers: [issue #105 closed with fix commit]
  Deps: T104.3
  Result: Comment posted with fix commit + DGX verification. Issue was already closed.
  Acceptance:
  - Post comment on #105 citing the fix commit, test output, and DGX verification.
  - Close the issue.

---

#### E105: Fix NaN/Inf in Windowed Training -- Issue #121 [2026 Q2 -- CRITICAL]

GitHub issue #121: All four timeseries backends (DLinear, N-HiTS, CfC, PatchTST)
produce NaN or -Inf weights when training on real-world high-dimensional data
(1,623 features, 50K rows, feature scales spanning 10 orders of magnitude).

Root cause analysis:
1. No input normalization -- features range from 0.0001 (funding rates) to
   millions (volumes). Without z-score normalization, gradients explode.
2. No NaN/Inf detection -- once a weight becomes NaN, training continues
   and poisons all subsequent computations. Should halt early with error.
3. No LR warmup -- large initial gradients cause divergence in early epochs.
Note: Gradient clipping IS already present (GradClip=1.0 default in
DefaultTrainConfig). The issue incorrectly claims it is missing.

Files affected: timeseries/dlinear.go, nhits.go, cfc.go, patchtst.go and
their corresponding test files.

- [x] T105.1 Add WarmupEpochs to TrainConfig and z-score normalization helper (2026-03-21)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: none
  Acceptance:
  - Add WarmupEpochs int field to TrainConfig (default 5 in DefaultTrainConfig).
  - Add normalizeWindows(windows [][][]float64) ([][][]float64, means [][]float64, stds [][]float64)
    helper that computes per-channel mean and std across all samples, returns
    normalized windows with (x - mean) / (std + 1e-8).
  - Add isFinite(v float64) bool helper.
  - Unit tests for normalizeWindows with multi-scale input.
  - `go vet ./timeseries/` clean.

- [x] T105.2 Apply normalization + NaN detection + warmup to DLinear TrainWindowed (2026-03-21)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: T105.1
  Acceptance:
  - At entry of TrainWindowed, call normalizeWindows on input windows.
  - Apply linear LR warmup: lr_effective = lr * min(1.0, float64(epoch+1) / float64(warmupEpochs)).
  - After each epoch loss computation, check math.IsNaN(epochLoss) || math.IsInf(epochLoss, 0).
    If true, return error: "dlinear: training diverged at epoch %d: loss=%v".
  - Existing gradient clipping (GradClip=1.0) remains unchanged.
  - Existing tests still pass.

- [x] T105.3 Apply normalization + NaN detection + warmup to N-HiTS TrainWindowed (2026-03-21)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: T105.1
  Acceptance:
  - Same normalization, warmup, and NaN detection pattern as T105.2.
  - Error message prefix: "nhits:".
  - Existing tests still pass.

- [x] T105.4 Apply normalization + NaN detection + warmup to CfC TrainWindowed (2026-03-21)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: T105.1
  Acceptance:
  - Same normalization, warmup, and NaN detection pattern as T105.2.
  - Error message prefix: "cfc:".
  - Existing tests still pass.

- [x] T105.5 Apply normalization + NaN detection + warmup to PatchTST TrainWindowed (2026-03-21)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: T105.1
  Acceptance:
  - Same normalization, warmup, and NaN detection pattern as T105.2.
  - Error message prefix: "patchtst:".
  - Existing tests still pass.

- [x] T105.6 Add multi-scale divergence regression tests for all 4 backends (2026-03-21)
  Owner: ML Eng  Est: 1.5h  verifies: [UC-026]
  Deps: T105.2, T105.3, T105.4, T105.5
  Acceptance:
  - New test TestXxx_TrainWindowed_MultiScale in each backend test file.
  - Create synthetic data with 100+ features spanning 10 orders of magnitude
    (0.0001 to 1,000,000), 500+ samples, 20 epochs.
  - Assert: training completes without error, final loss is finite (not NaN/Inf).
  - Assert: all model weights are finite after training.
  - Run: `go test -run MultiScale -timeout 120s ./timeseries/`

- [x] T105.7 Run go vet and linter on timeseries package (2026-03-21)
  Owner: ML Eng  Est: 15m  verifies: [infrastructure]
  Deps: T105.6
  Acceptance:
  - `go vet ./timeseries/` clean.
  - `golangci-lint run ./timeseries/` clean.

- [x] T105.8 Close GitHub issue #121 with fix evidence (2026-03-21)
  Owner: ML Eng  Est: 15m  delivers: [issue #121 closed with fix commit]
  Deps: T105.7
  Acceptance:
  - Post comment on #121 citing fix commits, test output, and what changed.
  - Close the issue.

---

#### E101: GitHub Issues Resolution [2026 Q2]

Completed: T101.1-T101.15 (15 tasks across 4 waves). Trimmed 2026-03-20.
Knowledge preserved in checkpoint and devlog.

---

#### E102: Attention Residuals (AttnRes) [2026 Q2]

Completed: T102.1-T102.5 (5 tasks across 2 waves). Trimmed 2026-03-20.
Implements arXiv:2603.15031. See layers/residual/ package and inference/residual.go.

---

### PRIORITY 0: Security Remediation (Deep Review v1.10.0)

A deep security review of v1.10.0 (10 agents, 350+ files read) found 2 Critical,
11 High, 24 Medium findings. The security/ package has production-quality primitives
but none are wired into the default server. All API endpoints are unauthenticated.
Decision rationale: docs/adr/065-security-middleware-integration.md
Source: .claude/scratch/deep-review-report.md

---

#### E106: Security -- Critical and High Fixes [2026 Q2 -- CRITICAL]

##### Wave 30: Critical Security Fixes (5 agents)

- [x] T106.1 Wire authentication middleware into serve.Server
  Owner: Security Eng  Est: 2h  verifies: [UC-003, UC-004]
  Deps: none
  Files: serve/server.go, cmd/cli/serve.go
  Acceptance:
  - Add WithAPIKey(key string) ServerOption that stores hashed key on Server.
  - Add authMiddleware that checks Bearer token with subtle.ConstantTimeCompare.
  - Skip auth for /metrics, /healthz, /readyz, /openapi.yaml paths.
  - In Handler(), compose: recovery > auth > logging > handler.
  - In cmd/cli/serve.go, add --api-key flag and ZERFOO_API_KEY env var fallback.
  - Test: request without key returns 401. Request with wrong key returns 401.
    Request with correct key returns 200. Skipped paths return 200 without key.
  - go vet ./serve/ ./cmd/cli/ clean.

- [x] T106.2 Replace X-Tenant-ID header with context.Context in cloud/server.go
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: cloud/server.go
  Acceptance:
  - In authMiddleware, replace r.Header.Set("X-Tenant-ID", tenant.ID) with
    ctx := context.WithValue(r.Context(), tenantKey{}, tenant); r = r.WithContext(ctx).
  - In rateLimitMiddleware and billingMiddleware, replace
    r.Header.Get("X-Tenant-ID") with tenantFromContext(r.Context()).
  - Remove the X-Tenant-ID header entirely from all middleware.
  - Test: verify tenant is correctly propagated through the middleware chain.
  - go vet ./cloud/ clean.

- [x] T106.3 Fix path traversal in FileSystemRepository
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/repository/repository.go
  Acceptance:
  - Change modelDir(id string) to return (string, error).
  - Add containment check: filepath.Clean(joined) must have baseDir as prefix.
  - Update all callers (modelPath, metadataPath, Get, Save, Delete, List) to
    handle the error and return 400 for traversal attempts.
  - Test: model ID "../../etc" returns error. Normal ID "gemma-3-1b" works.
  - go vet ./serve/repository/ clean.

- [x] T106.4 Add SSRF protection to vision image fetch
  Owner: Security Eng  Est: 1.5h  verifies: [UC-001]
  Deps: none
  Files: serve/vision.go
  Acceptance:
  - In downloadImage, resolve hostname to IP via net.DefaultResolver.LookupHost.
  - Block if any resolved IP is loopback, private, link-local, or link-local-multicast.
  - Block hostnames "metadata.google.internal" and "169.254.169.254".
  - Replace http.DefaultClient with dedicated client: Timeout 30s, CheckRedirect max 3.
  - Test: URL to 127.0.0.1 returns error. URL to 169.254.169.254 returns error.
    URL to public host succeeds. Test with mock server.
  - go vet ./serve/ clean.

- [x] T106.5 Fix Server.unloaded data race
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Change field from `unloaded bool` to `unloaded atomic.Bool`.
  - Replace all reads with s.unloaded.Load(), writes with s.unloaded.Store(true).
  - Run go test -race ./serve/ -- no race detected.
  - go vet ./serve/ clean.

##### Wave 31: High Security Fixes -- Serving (5 agents)

- [x] T106.6 Add request body size limits to inference endpoints
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - At top of handleChatCompletions, handleCompletions, handleEmbeddings:
    r.Body = http.MaxBytesReader(w, r.Body, 10<<20).
  - Test: POST body > 10 MB returns 413 Request Entity Too Large.
  - go vet ./serve/ clean.

- [x] T106.7 Add embedding lookup bounds check
  Owner: ML Eng  Est: 30m  verifies: [UC-001, UC-002]
  Deps: none
  Files: inference/arch_llama.go
  Acceptance:
  - Before indexing embData at line 288, add:
    if id < 0 || id >= vocabSize { return nil, fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize) }
  - Same check for Q8 path at line 282 and GPU path at line 247.
  - Test: verify out-of-range token ID returns error, not panic.
  - go vet ./inference/ clean.

- [x] T106.8 Add TLS support to serve CLI
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: cmd/cli/serve.go
  Acceptance:
  - Add --tls-cert and --tls-key flags.
  - When both provided, use httpServer.ListenAndServeTLS(cert, key).
  - When only one provided, return error "both --tls-cert and --tls-key required".
  - Test: verify TLS flags are parsed; verify error on mismatched flags.
  - go vet ./cmd/cli/ clean.

- [x] T106.9 Add server-side max_tokens cap
  Owner: ML Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add WithMaxTokens(n int) ServerOption.
  - In handleChatCompletions and handleCompletions, clamp req.MaxTokens to
    min(req.MaxTokens, s.maxTokens) where default is 8192.
  - Test: request with max_tokens=100000 gets clamped to 8192.
  - go vet ./serve/ clean.

- [x] T106.10 Add rate limiting middleware to serve.Server
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add WithRateLimiter(rl *security.RateLimiter) ServerOption.
  - In Handler(), if rateLimiter is set, wrap handler with rate limit middleware.
  - Use security.ClientIP(r) for per-IP limiting.
  - Return 429 Too Many Requests when limit exceeded.
  - Test: exceed rate limit, verify 429 response.
  - go vet ./serve/ clean.

##### Wave 32: High Security Fixes -- Inference and Registry (5 agents)

- [ ] T106.11 Convert panics to error returns in layers/core/
  Owner: ML Eng  Est: 2h  verifies: [UC-001, UC-002]
  Deps: none
  Files: layers/core/dense.go, cast.go, matmul.go, mul.go, sub.go, concat.go,
         unsqueeze.go, reshape.go, rotary_embedding.go
  Acceptance:
  - Replace every panic() call with return nil, fmt.Errorf(...) in Forward/Backward.
  - For WithBias() functional option panic in dense.go, move validation to NewDense
    and return error from constructor.
  - All existing tests still pass.
  - go vet ./layers/core/ clean.

- [ ] T106.12 Convert panics to error returns in layers/attention/
  Owner: ML Eng  Est: 1h  verifies: [UC-001, UC-002]
  Deps: none
  Files: layers/attention/attention_head.go
  Acceptance:
  - Replace all panic() calls in attention_head.go (lines 58-78) with error returns.
  - Existing tests still pass.
  - go vet ./layers/attention/ clean.

- [x] T106.13 Cap GenerateBatch concurrency
  Owner: ML Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: inference/inference.go
  Acceptance:
  - Add maxBatchConcurrency field to Model (default 8).
  - In GenerateBatch, use errgroup with SetLimit(m.maxBatchConcurrency).
  - Test: batch of 100 prompts runs with only 8 concurrent goroutines.
  - go vet ./inference/ clean.

- [ ] T106.14 Add SHA-256 checksum verification to HuggingFace downloads
  Owner: ML Eng  Est: 2h  verifies: [UC-005]
  Deps: none
  Files: registry/pull.go
  Acceptance:
  - During download, compute SHA-256 hash via io.TeeReader.
  - After download, fetch expected hash from HF API siblings response.
  - Compare computed vs expected. If mismatch, delete file and return error.
  - Write to temp file first, rename on success (also fixes F-4 atomic write).
  - Test: mock server returns mismatched hash -- verify file deleted and error returned.
  - go vet ./registry/ clean.

- [x] T106.15 Fix RegisterAlias concurrent map race
  Owner: ML Eng  Est: 30m  verifies: [UC-001]
  Deps: none
  Files: inference/inference.go
  Acceptance:
  - Add sync.RWMutex protecting modelAliases map.
  - RegisterAlias: Lock(). ResolveAlias: RLock().
  - Test: run RegisterAlias and ResolveAlias concurrently with -race, no race.
  - go vet ./inference/ clean.

##### Wave 33: Medium Security Fixes -- Data Exposure (5 agents)

- [ ] T106.16 Sanitize inference error messages to clients
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add sanitizeError(err error) string function that maps CUDA OOM to "server
    overloaded", file-not-found to "model not available", and all others to
    "inference failed". Log the original error with slog.Error.
  - Apply to handleChatCompletions, handleCompletions, handleEmbeddings, handleAudio.
  - Apply to streaming error frames in streamChatCompletion and streamCompletion.
  - Test: verify CUDA error message is not leaked to client.
  - go vet ./serve/ clean.

- [ ] T106.17 Add security headers middleware
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add securityHeadersMiddleware that sets X-Content-Type-Options: nosniff,
    X-Frame-Options: DENY, Cache-Control: no-store on all responses.
  - Wire into Handler() middleware chain.
  - Test: verify headers present on response.
  - go vet ./serve/ clean.

- [ ] T106.18 Add request ID correlation middleware
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add requestIDMiddleware that reads X-Request-Id header or generates UUID.
  - Store in context, include in all log entries, return in response header.
  - Test: verify X-Request-Id in response matches request header when provided,
    or is a valid UUID when not provided.
  - go vet ./serve/ clean.

- [ ] T106.19 Fix streaming chat template bypass
  Owner: ML Eng  Est: 1h  verifies: [UC-001, UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In streamChatCompletion, replace manual message concatenation (line 766)
    with model.FormatMessages(messages) or equivalent exported method.
  - Verify system prompts and role boundaries are preserved in streaming output.
  - Test: streaming chat with system prompt produces same prompt as non-streaming.
  - go vet ./serve/ clean.

- [ ] T106.20 Add request drain on model delete
  Owner: ML Eng  Est: 1h  verifies: [UC-003]
  Deps: T106.5
  Files: serve/server.go
  Acceptance:
  - Add sync.WaitGroup or atomic counter for in-flight requests.
  - Increment in handleChatCompletions/handleCompletions entry, decrement in defer.
  - In handleModelDelete, set unloaded=true first (reject new requests),
    then Wait() for in-flight requests to complete, then call model.Close().
  - Test: verify model delete waits for in-flight request to complete.
  - go vet ./serve/ clean.

##### Wave 34: Medium Security Fixes -- GGUF and Infrastructure (5 agents)

- [ ] T106.21 Add integer overflow checks to GGUF tensor parsing
  Owner: ML Eng  Est: 1h  verifies: [UC-001]
  Deps: none
  Files: model/gguf/loader.go, model/gguf/parser.go
  Acceptance:
  - In loader.go, use int64 for numElements. Reject any dimension > MaxInt32.
    Reject total elements > 1<<34 (~16 billion, ~64 GB at float32).
  - In parser.go, reject tensorCount > 100,000 and metadataKVCount > 1,000,000.
  - Test: crafted dimensions that would overflow are rejected with clear error.
  - go vet ./model/gguf/ clean.

- [ ] T106.22 Add size limit to OCI blob download
  Owner: ML Eng  Est: 30m  verifies: [UC-005]
  Deps: none
  Files: registry/oci.go
  Acceptance:
  - Replace io.ReadAll(resp.Body) with io.ReadAll(io.LimitReader(resp.Body, 20<<30+1)).
  - If len(data) > 20 GB, return error.
  - Test: verify oversized response is rejected.
  - go vet ./registry/ clean.

- [ ] T106.23 Fix JSON injection in support API error response
  Owner: Security Eng  Est: 15m  verifies: [infrastructure]
  Deps: none
  Files: support/api.go
  Acceptance:
  - Replace string concatenation at line 165 with json.NewEncoder(w).Encode.
  - Test: error message containing double quotes is properly escaped.
  - go vet ./support/ clean.

- [ ] T106.24 Add pod securityContext to Helm deployment
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: deploy/helm/zerfoo/templates/deployment.yaml, deploy/helm/zerfoo/values.yaml
  Acceptance:
  - Add securityContext: runAsNonRoot: true, runAsUser: 1000,
    readOnlyRootFilesystem: true, allowPrivilegeEscalation: false,
    capabilities: drop: ["ALL"].
  - Add corresponding values to values.yaml.
  - Verify: helm template renders correctly with security context.

- [ ] T106.25 Restrict Cloud Run IAM from allUsers
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: infra/terraform/zerfoo-cloud/cloud_run.tf
  Acceptance:
  - Replace allUsers IAM binding with authenticated service account.
  - Add google_service_account resource for API invoker.
  - Verify: terraform plan shows IAM change.

##### Wave 35: Medium Fixes -- Training and Layers (5 agents)

- [ ] T106.26 Fix worker pool Close() data race
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: internal/workerpool/pool.go
  Acceptance:
  - Replace `closed bool` with sync.Once.
  - Close() uses p.once.Do(func() { close(p.tasks); p.wg.Wait() }).
  - Test: concurrent Close() calls do not panic.
  - go vet ./internal/workerpool/ clean.

- [ ] T106.27 Add gradient clipping and NaN guard to AdamW
  Owner: ML Eng  Est: 1h  verifies: [UC-016]
  Deps: none
  Files: training/optimizer/adamw.go
  Acceptance:
  - Add optional MaxGradNorm float64 field to AdamW config.
  - Before parameter update, compute gradient norm and clip if > MaxGradNorm.
  - Check for NaN/Inf in gradients; return error if detected.
  - Test: gradient with NaN returns error. Gradient exceeding norm is clipped.
  - go vet ./training/optimizer/ clean.

- [ ] T106.28 Fix S4 backward nil gradient panic
  Owner: ML Eng  Est: 30m  verifies: [UC-001]
  Deps: none
  Files: layers/ssm/s4.go
  Acceptance:
  - Before accessing Gradient.Data(), check for nil. If nil, initialize to zeros.
  - Test: S4 backward on first call does not panic.
  - go vet ./layers/ssm/ clean.

- [ ] T106.29 Fix LoRA backward nil gradient Add
  Owner: ML Eng  Est: 30m  verifies: [UC-016]
  Deps: none
  Files: training/lora/ (identify exact file)
  Acceptance:
  - Before engine.Add(grad, dB), check if grad is nil. If nil, set grad = dB directly.
  - Test: LoRA backward on first call does not panic.
  - go vet ./training/lora/ clean.

- [ ] T106.30 Fix PatchTST inference projection head
  Owner: ML Eng  Est: 1.5h  verifies: [UC-026]
  Deps: none
  Files: inference/timeseries/arch_patchtst.go
  Acceptance:
  - Replace current projection path (lines 303-357) with channel-independent
    projection: [batch*numVars, d_model] @ [d_model, horizon] = [batch*numVars, horizon],
    then reshape to [batch, numVars, horizon] and transpose to [batch, horizon, numVars].
  - Remove the ReduceMean that averages unrelated variables.
  - Test: output shape is [batch, horizon, numVars] with no cross-variable mixing.
  - go vet ./inference/timeseries/ clean.

##### Wave 36: Tech Debt and Quality (5 agents)

- [ ] T106.31 Replace stdlib log with structured logger in 7 files
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: none
  Files: layers/attention/grouped_query_attention.go, generate/generator.go,
         generate/tensor_cache.go, generate/megakernel.go, inference/load_gguf.go,
         model/gguf/loader.go, serve/disaggregated/gateway.go
  Acceptance:
  - Replace `import "log"` with structured logger from ztensor/log.
  - Pass logger via constructor or use package-level default.
  - No stdlib log imports remain in production code (test files exempt).
  - go vet on each changed package clean.

- [ ] T106.32 Cache ZERFOO_DEBUG_ONNX env var check
  Owner: ML Eng  Est: 15m  verifies: [UC-002]
  Deps: none
  Files: generate/generator.go
  Acceptance:
  - Add package-level var debugOnnx = os.Getenv("ZERFOO_DEBUG_ONNX") != "".
  - Replace os.Getenv("ZERFOO_DEBUG_ONNX") calls at lines 342, 375, 436 with debugOnnx.
  - go vet ./generate/ clean.

- [ ] T106.33 Fix flaky TestMAML_MetaConvergence
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: meta/meta_test.go
  Acceptance:
  - Set fixed random seed (e.g., 42) for deterministic test.
  - Increase tolerance or epochs to ensure convergence within test bounds.
  - Test passes reliably: go test -count=5 -run TestMAML_MetaConvergence ./meta/
  - go vet ./meta/ clean.

- [ ] T106.34 Redact tenant API keys from Config()/List() responses
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: cloud/tenant.go
  Acceptance:
  - In Tenant.Config(), set APIKey to empty string or redacted placeholder.
  - In TenantManager.List(), same redaction.
  - Test: Config() and List() do not return raw API keys.
  - go vet ./cloud/ clean.

- [ ] T106.35 Add OCI reference path traversal check
  Owner: Security Eng  Est: 15m  verifies: [UC-005]
  Deps: none
  Files: registry/oci.go
  Acceptance:
  - In parseReference, reject repository names containing "..".
  - Test: reference with ".." in repository returns error.
  - go vet ./registry/ clean.

##### Wave 37: Verification and Lint (2 agents)

- [ ] T106.36 Run go test -race on all changed packages
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: T106.1-T106.35
  Acceptance:
  - go test -race -timeout 300s ./serve/ ./cloud/ ./inference/ ./layers/core/
    ./layers/attention/ ./layers/ssm/ ./registry/ ./model/gguf/ ./generate/
    ./training/optimizer/ ./training/lora/ ./internal/workerpool/ ./support/
    ./meta/ ./inference/timeseries/
  - All tests pass with no races detected.

- [ ] T106.37 Run go vet and linter on entire codebase
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: T106.36
  Acceptance:
  - go vet ./... clean.
  - golangci-lint run ./... clean (or only warnings in unmodified code).

---

## Parallel Work

### Parallel Tracks

| Track | Description | Epic/Group IDs | Sync Points |
|-------|-------------|----------------|-------------|
| A | Security Remediation (ACTIVE) | E106 | Merge at Wave 37 lint pass |
| B | Community + DevRel | E4, E5, E11 | Merge at content published |
| C | Backend Expansion | E8, E20 | Merge at ROCm parity |
| D | Platform and Enterprise | E12-E19 | Merge at cloud GA |
| E | 10-Year Long-Tail | E26-E33 | Merge at milestones |

### Waves

#### Wave 23: Throughput Regression Fix (completed 2026-03-20)

- [x] T103.1 Verify clean-build throughput on DGX — 156 tok/s (regression confirmed)
- [x] T103.2 Bisect ztensor regression — commit 33b54d9 (CUDA graph bypass)
- [x] T103.3 Bisect zerfoo regression — Q5_K/Q6_K loader change
- [x] T103.4 Update benchmarks.md — 244 tok/s verified
- [x] T103.5 Verify fix — 244.45 tok/s (95% roofline)

#### Wave 24: HRM Segfault Fix (completed 2026-03-20)

- [x] T104.1 Fix HModule/LModule hidden state init + batch handling — auto-resize in Forward
- [x] T104.2 Add HRM recurrent loop integration test — batch=32, 10 features, nSteps=2, tSteps=3

#### Wave 25: DGX Verification + Issue Close (completed 2026-03-20)

- [x] T104.3 Verify fix on DGX Spark — 43 tests PASS, no segfault, race clean
- [x] T104.4 Close GitHub issue #105 — comment posted with fix evidence

#### Wave 26: TrainConfig + Normalization Helper (completed 2026-03-21)

- [x] T105.1 Add WarmupEpochs to TrainConfig + normalizeWindows helper

#### Wave 27: Per-Backend NaN Fix (completed 2026-03-21)

- [x] T105.2 DLinear: normalization + NaN detection + warmup
- [x] T105.3 N-HiTS: normalization + NaN detection + warmup
- [x] T105.4 CfC: normalization + NaN detection + warmup
- [x] T105.5 PatchTST: normalization + NaN detection + warmup

#### Wave 28: Multi-Scale Tests + Lint (completed 2026-03-21)

- [x] T105.6 Multi-scale divergence regression tests (all 4 backends)
- [x] T105.7 Run go vet + linter on timeseries package

#### Wave 29: Close Issue (completed 2026-03-21)

- [x] T105.8 Close GitHub issue #121

#### Wave 30: Critical Security Fixes (5 agents)

- [x] T106.1 Wire authentication middleware into serve.Server
- [x] T106.2 Replace X-Tenant-ID header with context in cloud/server.go
- [x] T106.3 Fix path traversal in FileSystemRepository
- [x] T106.4 Add SSRF protection to vision image fetch
- [x] T106.5 Fix Server.unloaded data race

#### Wave 31: High Security Fixes -- Serving (5 agents)

- [x] T106.6 Add request body size limits to inference endpoints
- [x] T106.7 Add embedding lookup bounds check
- [x] T106.8 Add TLS support to serve CLI
- [x] T106.9 Add server-side max_tokens cap
- [x] T106.10 Add rate limiting middleware

#### Wave 32: High Security Fixes -- Inference and Registry (5 agents)

- [x] T106.11 Convert panics to errors in layers/core/
- [x] T106.12 Convert panics to errors in layers/attention/
- [x] T106.13 Cap GenerateBatch concurrency
- [x] T106.14 Add SHA-256 checksum to HuggingFace downloads + atomic write
- [x] T106.15 Fix RegisterAlias concurrent map race

#### Wave 33: Medium Fixes -- Data Exposure (5 agents)

- [ ] T106.16 Sanitize inference error messages
- [ ] T106.17 Add security headers middleware
- [ ] T106.18 Add request ID correlation middleware
- [ ] T106.19 Fix streaming chat template bypass
- [ ] T106.20 Add request drain on model delete

#### Wave 34: Medium Fixes -- GGUF and Infrastructure (5 agents)

- [ ] T106.21 Add integer overflow checks to GGUF parsing
- [ ] T106.22 Add size limit to OCI blob download
- [ ] T106.23 Fix JSON injection in support API error response
- [ ] T106.24 Add pod securityContext to Helm deployment
- [ ] T106.25 Restrict Cloud Run IAM from allUsers

#### Wave 35: Medium Fixes -- Training and Layers (5 agents)

- [ ] T106.26 Fix worker pool Close() data race
- [ ] T106.27 Add gradient clipping and NaN guard to AdamW
- [ ] T106.28 Fix S4 backward nil gradient panic
- [ ] T106.29 Fix LoRA backward nil gradient Add
- [ ] T106.30 Fix PatchTST inference projection head

#### Wave 36: Tech Debt and Quality (5 agents)

- [ ] T106.31 Replace stdlib log with structured logger
- [ ] T106.32 Cache ZERFOO_DEBUG_ONNX env var check
- [ ] T106.33 Fix flaky TestMAML_MetaConvergence
- [ ] T106.34 Redact tenant API keys from Config/List
- [ ] T106.35 Add OCI reference path traversal check

#### Wave 37: Verification and Lint (2 agents)

- [ ] T106.36 Run go test -race on all changed packages
- [ ] T106.37 Run go vet + linter on entire codebase

Remaining roadmap tasks are blocked by hardware access or human actions.
See Hand-Off Notes.

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M0 | Internal Consumer Bridge | WE1, WE13 | tabular.Ensemble working; GPU tests green | 2026-06-30 |
| M0.5 | Advanced Tabular | WE2, WE3, WE4 | 7+ tabular/timeseries architectures; AutoML extension | 2026-09-30 |
| M1 | Inference Excellence | E2, WE13 | 300+ tok/s; 12+ archs validated; 5K stars | 2026-12-31 |
| M2 | v1.0 and Ecosystem | E8, E9, E10, E11 | ROCm parity; 25K stars | 2027-12-31 |
| M3 | Enterprise Foundation | E12-E16, WE5 | $500K ARR; SOC 2 Type I; transfer learning | 2028-12-31 |
| M4 | Platform GA | E17-E19, WE6, WE7 | $2M ARR; SOC 2 Type II; RL + cross-asset | 2029-12-31 |
| M5 | Training Platform | E20-E22, WE8, WE9 | $10M ARR; Metal + SYCL; regime + NAS | 2030-12-31 |
| M6 | Industry Standard | E24-E27, WE10, WE11 | $50M ARR; ZerfooConf; hardware optimization | 2032-12-31 |
| M7 | Platform Maturity | E28-E30, WE12 | $75M ARR; federated; on-device; continuous learning | 2034-12-31 |
| M8 | Market Leadership | E31-E33 | $150M+ ARR; IPO filed; 100+ architectures | 2036-12-31 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R0 | Tabular package does not improve internal consumer signal quality | Critical | Medium | Focus on architecture diversity (7+ models); AutoML finds best fit; walk-forward validation as quality gate. |
| R1 | Go ML TAM ceiling | Critical | High | Expand beyond Go devs via OpenAI API, edge runtime, language FFI. |
| R2 | Apache 2.0 fork by cloud provider | Existential | Medium-High | Innovation velocity; consider AGPL for v2. See ADR-057. |
| R3 | Latent bugs in AI-generated code | High | High | Security audit (Year 3); DGX validation; fuzz testing; bug bounty. |
| R4 | Maintainer burnout / bus factor of 1 | Critical | High | 5 co-maintainers by Year 2; governance by Year 4. |
| R5 | No enterprise budget owner for "Go ML library" | High | Medium-High | Position as "inference infrastructure"; POC program; marketplace credits. |
| R6 | ROCm never reaches CUDA parity | Medium | High | 80% parity target; gate by user demand; drop if < 5% adoption. |
| R7 | Enterprise sales cycle too long | High | Medium | Marketplace consumption; support contracts first; PLG motion. |
| R8 | SaaS multiples compressed | High | Medium | Maintain optionality: acquisition, PE, or continued private. |
| R9 | Rust ML captures "systems language ML" first | High | Medium | Ship v1.0 first; Go has Python interop advantage; edge differentiator. |
| R10 | NVIDIA CUDA licensing changes | High | Low-Medium | GRAL insulates; Metal + SYCL fallback. |
| R11 | Go generics limitations | Medium | Medium | Extension interface pattern (ADR-058). |
| R12 | Cloud marketplace revenue share erodes margins | Medium | Medium | SaaS listings (3%); enterprise self-managed. See ADR-060. |
| R13 | GopherCon talk rejected | Low | Medium | Multiple conferences; sponsor booth; host meetups. |
| R14 | FedRAMP cost exceeds budget | Medium | Medium | Delay to Year 8-9; evaluate demand first; partner with GovCloud MSP. |
| R15 | Agentic coder quality drift | High | High | Human review gates; security audit; strict CI; /review before releases. |
| R16 | Unauthenticated API server in production | Critical | High | E106 Wave 30 fixes this. Wire security/ middleware into serve.Server. See ADR-065. |
| R17 | SSRF via vision image fetch exposes cloud metadata | High | Medium | E106 T106.4 adds private IP blocking and DNS resolution validation. |
| R18 | Path traversal in model repository enables arbitrary file deletion | High | Medium | E106 T106.3 adds containment validation to FileSystemRepository. |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Benchmark tasks: results appended to docs/devlog.md and docs/benchmarks.md.
6. ADR tasks: ADR file created and referenced in plan.
7. Documentation tasks: content reviewed and published on docs site.
8. Enterprise tasks: customer validation or contract execution confirmed.

### Quality Gates

- Every implementation task must have a paired test.
- Run `go vet ./...` after every code change before committing.
- Commit each task as its own commit. One logical change per commit.
- Never commit files from different directories in the same commit.
- Use standard library only: no testify, no cobra, no viper. Use testing.T and flag.
- GPU-only tests: tag with `//go:build cuda` and run only on DGX.
- Benchmark tasks must run on DGX Spark (not CPU-only CI).
- Never skip CI hooks with --no-verify.
- Human review gate required at each milestone (M0-M8).
- Security review (/review) before each enterprise-facing release.
- Run `golangci-lint` on all changed packages before committing.

### Agent Assignment Protocol

1. Read the wave plan to find the current wave and available tasks.
2. Prefer Priority 1 (W-series) tasks over Priority 2 and 3 (T-series).
3. Within same priority, prefer lowest-ID task in your skill domain.
4. Claim a task by marking it in progress.
5. Read task description fully; identify target file paths.
6. Implement, test, vet, commit in target repo directory.
7. Mark task completed.
8. Repeat from step 1.

### Code Style

- Engine[T] is law: all tensor ops through compute.Engine[T].
- Generics throughout: [T tensor.Numeric] constraints.
- Fuse, do not fragment: prefer fused ops over primitive sequences.
- No CGo in core packages; GPU via purego.
- Docstrings only on exported types and functions. No inline comments unless logic
  is non-obvious.
- Rebase and merge. Not squash, not merge commits.

---

## Progress Log

### 2026-03-21: E106 created -- Security remediation from deep review v1.10.0

Deep review of v1.10.0 (10 agents, 350+ files, ~180K lines) found 2 Critical,
11 High, 24 Medium, 9 Low, 4 Info findings. Created E106 with 37 tasks across
8 waves (Wave 30-37). Primary theme: security/ package primitives exist but are
not wired into serve.Server. Created ADR-065 (security middleware integration).
Waves 30-36 run 5 agents each (35 parallel tasks), Wave 37 runs 2 agents for
verification. Total estimated effort: ~30 hours across 37 tasks.

Also trimmed completed E105 progress log entry (E105 is done, issue #121 closed).

### 2026-03-21: E105 created -- GitHub issue #121 NaN/Inf in windowed training

Created E105 (8 tasks, 4 waves) to fix NaN/Inf divergence in all 4 timeseries
backends (DLinear, N-HiTS, CfC, PatchTST) when training on high-dimensional
real-world data (issue #121). Root cause: no input normalization when features
span 10 orders of magnitude. Gradient clipping was already present (GradClip=1.0)
contrary to the issue report. Fix adds z-score normalization, NaN/Inf early halt,
and LR warmup. Wave 26 (config), Wave 27 (4 parallel backend fixes),
Wave 28 (tests + lint), Wave 29 (close issue).

### 2026-03-20: E104 created -- GitHub issue #105 HRM segfault

Created E104 (4 tasks) to fix HRM Forward() segfault on linux/arm64 (issue #105).
Root cause: uninitialized hidden states with batch=1 passed to recurrent loop
that expects batch=N, causing out-of-bounds NEON memory access.
Wave 24 (2 agents parallel) + Wave 25 (DGX verification + issue close).

E103 completed earlier today: throughput regression fixed (244 tok/s restored).
Trimmed E103 context to results only.

### 2026-03-20: E103 throughput regression plan created

Created E103 (5 tasks) to investigate and close the throughput regression reported
during T16.3 benchmarking. The devlog already traced the 229->136 drop to a dirty
working tree (Q4_K re-quantization disabled experimentally), but benchmarks.md is
stale and no clean-build verification has been run.

Trimmed completed epics: E2 (T2.12 marked done), E14, E17, E18, E20, E29, E101,
E102. Removed stale wave summary sections (Waves 5-22) that duplicated task status
already captured in the main work breakdown.

T103.1 marked complete based on prior DGX session verification.

### 2026-03-19: Waves 10-22 executed (97 tasks total across all sessions)

All codeable tasks complete. 107/132 tasks done (81.1%).
Remaining 25 tasks blocked by hardware access or human actions.

---

## Hand-Off Notes

### Current State (2026-03-21)

- **Score:** 121/181 tasks complete (66.9%). E106 added 37 security tasks.
- **Active epic:** E106 (security remediation, 37 tasks, 8 waves).
- **Last completed:** E105 (NaN/Inf fix in windowed training, issue #121 closed).
- **DGX Spark access:** ssh ndungu@192.168.86.250. GB10 GPU, sm_121, 128GB LPDDR5x.
- **Throughput:** 244 tok/s restored (E103 fixed two compounding regressions).
- **Peak throughput:** 245 tok/s (commit 4e85b12, Gemma 3 1B Q4_K_M, 256 tokens, CUDA graphs).
- **Roofline:** GB10 max ~257 tok/s at 200 GB/s bandwidth for 778 MB model.
- **500+ tok/s:** Blocked by hardware. Needs A100 (2 TB/s) or H100 (3.35 TB/s).

### Blocked Tasks Summary

| Blocker | Tasks | Unblock Action |
|---------|-------|----------------|
| AMD Instinct GPU | T8.1-T8.6 | Acquire AMD GPU access |
| Multi-GPU system | T9.4 | Access DGX A100/H100 |
| Raspberry Pi 5 | T15.3 | Acquire hardware |
| Jetson Orin Nano | T15.4 | Acquire hardware |
| High-BW GPU | T16.3 | Access A100/H100 |
| Apple M4 Max | T20.3 | Access M4 Max system |
| Mobile devices | T29.4 | iOS + Android test devices |
| Human action | T4.7, T5.4, T11.1, T11.4, T12.4 | Manual execution required |
| Security remediation | T106.1-T106.37 | Execute E106 waves 30-37 via /apply |
| SOC 2 Type II | T19.1 | Observation period completion |
| Upstream deps | T26.2, T26.3, T30.1-T30.3, T31.1-T31.5, T32.2 | Sequential human milestones |

### Key Files

- Plan: docs/plan.md (this file)
- Architecture: docs/design.md
- Devlog: docs/devlog.md
- Benchmarks: docs/benchmarks.md
- Use cases: .claude/scratch/usecases-manifest.json
- Deep review report: .claude/scratch/deep-review-report.md
- Security middleware ADR: docs/adr/065-security-middleware-integration.md
- ADRs: docs/adr/ (62 ADRs, 037-062)
