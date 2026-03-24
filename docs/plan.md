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

#### E107: v1.11.0 Review Remediation + Issue #123 [2026 Q2]

Post-E106 deep review of v1.11.0 found 2 High, 7 Medium, 8 Low remaining findings.
Plus GitHub issue #123: NHiTS nil pointer dereference in linearForward on
TrainWindowed with 132-channel data.
Source: .claude/scratch/deep-review-report.md (2026-03-23)

##### Wave 38: Critical Gaps (5 agents)

- [x] T107.1 Add MaxBytesReader, sanitizeError, and inflight tracking to handleClassify
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/classify.go
  Acceptance:
  - Add r.Body = http.MaxBytesReader(w, r.Body, 10<<20) at handler entry.
  - Add s.inflight.Add(1) and defer s.inflight.Done().
  - Replace err.Error() with s.sanitizeError(err) in error response.
  - Add isMaxBytesError check returning 413 for oversized requests.
  - Test: oversized body returns 413. Internal error details not leaked.
  - go vet ./serve/ clean. go test -race ./serve/ pass.

- [x] T107.2 Convert panic to error return in reducesum Backward (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-001, UC-002]
  Deps: none
  Files: layers/reducesum/reducesum.go
  Acceptance:
  - Replace panic(fmt.Sprintf("unsupported axis %d", r.axis)) at line 111 with
    return nil, fmt.Errorf("reducesum: unsupported axis %d for backward", r.axis).
  - Test: Backward with invalid axis returns error, not panic.
  - go vet ./layers/reducesum/ clean.

- [x] T107.3 Convert panic to error returns in rl/replay.go (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: rl/replay.go
  Acceptance:
  - Replace panic() at lines 20 and 60 with error returns.
  - Update callers to handle errors.
  - go vet ./rl/ clean. go test -race ./rl/ pass.

- [x] T107.4 Fix NHiTS nil pointer dereference in linearForward (issue #123)
  Owner: ML Eng  Est: 2h  verifies: [UC-026]
  Deps: none
  Files: timeseries/nhits.go
  Acceptance:
  - Root cause: linearForward at line 253 dereferences l.weights which is nil
    when the stack's MLP dimensions are miscalculated for high-channel data.
  - Add nil check for l.weights and l.biases at top of linearForward:
    if l.weights == nil || l.biases == nil { return nil, fmt.Errorf("nhits: nil weight/bias in linear layer") }
  - Investigate and fix the root cause: verify newNHiTSStack creates correctly
    sized MLP layers when channels=132 and various inputLen (15-240).
  - Add test: NewNHiTS with 132 channels, TrainWindowed with windows 15,30,60,120,240
    must not panic. Forward pass must produce finite output.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestNHiTS pass.

- [x] T107.5 Fix DNS rebinding TOCTOU in SSRF validation
  Owner: Security Eng  Est: 1.5h  verifies: [UC-001]
  Deps: none
  Files: serve/vision.go
  Acceptance:
  - Replace validateImageURL + separate HTTP request with a custom net.Dialer
    that validates the resolved IP at connect time (in DialContext).
  - The dialer checks ip.IsLoopback, ip.IsPrivate, ip.IsLinkLocalUnicast,
    ip.IsLinkLocalMulticast, and blocks cloud metadata IPs.
  - Remove the separate validateImageURL call (replaced by dialer validation).
  - Test: DNS rebinding scenario (mock that returns different IPs) is blocked.
  - go vet ./serve/ clean. go test -race ./serve/ pass.

##### Wave 39: Medium Fixes (5 agents)

- [x] T107.6 Fix ListByCustomer sort algorithm
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: support/ticket.go
  Acceptance:
  - Replace single-pass pairwise swap at lines 158-162 with slices.SortFunc
    or sort.Slice using ticket.UpdatedAt for ordering.
  - Test: ListByCustomer with 5+ tickets returns correctly ordered list.
  - go vet ./support/ clean. go test -race ./support/ pass.

- [x] T107.7 Add inflight tracking to handleEmbeddings
  Owner: Security Eng  Est: 15m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add s.inflight.Add(1) and defer s.inflight.Done() at top of handleEmbeddings.
  - go vet ./serve/ clean.

- [x] T107.8 Add NaN detection to normalizeWindows input
  Owner: ML Eng  Est: 30m  verifies: [UC-026]
  Deps: none
  Files: timeseries/dlinear.go
  Acceptance:
  - In normalizeWindows, before computing mean/std, scan input for NaN using
    isFinite helper. Return error if any NaN detected.
  - Change return signature to include error.
  - Update all callers (dlinear, nhits, cfc, patchtst TrainWindowed) to handle error.
  - Test: input with NaN returns error.
  - go vet ./timeseries/ clean.

- [x] T107.9 Disable public IPs in AWS QuickStart template
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: marketplace/aws/cfn/quickstart.yaml
  Acceptance:
  - Set AssignPublicIp: DISABLED in the ECS task networking config.
  - Verify ALB still routes traffic to private task IPs.

- [x] T107.10 Restrict Azure ARM template firewall rules
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: marketplace/azure/arm/template.json
  Acceptance:
  - Replace 0.0.0.0/0 source address prefix with parameterized CIDR.
  - Add parameter AllowedSourceCIDR with sensible default.

##### Wave 40: Verification (2 agents)

- [x] T107.11 Run go test -race on all changed packages
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: T107.1-T107.10
  Acceptance:
  - go test -race -timeout 300s ./serve/ ./layers/reducesum/ ./rl/ ./timeseries/
    ./support/ -- all pass, no races.

- [x] T107.12 Close GitHub issue #123 with fix evidence
  Owner: ML Eng  Est: 15m  delivers: [issue #123 closed with fix commit]
  Deps: T107.4, T107.11
  Acceptance:
  - Post comment on #123 citing fix commit, test output, and root cause.
  - Close the issue.

---

#### E108: Deep Review v1.11.1 Remediation [2026 Q2 -- CRITICAL]

Deep review of v1.11.1 (5 agents, 350+ files, ~298K lines) found 5 Critical,
15 High, 24 Medium, 11 Low findings. E106/E107 already addressed auth middleware,
body size limits, TLS, rate limiting, error sanitization, security headers,
request IDs, panic-to-error conversions, and inflight tracking. E108 covers the
REMAINING findings: cloud billing bypass, SAML hardening, DP seed fix, pprof
exposure, scope enforcement, batch template fix, marketplace retry, infra hardening.
Source: .claude/scratch/deep-review-report.md (2026-03-23)

##### Wave 41: Critical Security Fixes (5 agents)

- [x] T108.1 Fix streaming billing bypass in cloud billing middleware (C1) (2026-03-23)
  Owner: Security Eng  Est: 3h  verifies: [UC-003]
  Deps: none
  Files: cloud/server.go, serve/cloud/billing.go, generate/session.go
  Acceptance:
  - Add a tokenCounter callback to generate/session.go that tracks actual
    prompt_tokens and completion_tokens during generation (not from response body).
  - Surface counts via context value or return struct from GenerateStream.
  - In cloud/server.go billingMiddleware, read token counts from context instead
    of parsing JSON response body. Works for both streaming and non-streaming.
  - In serve/cloud/billing.go, same approach.
  - Test: streaming request (stream:true) produces correct billing record.
  - Test: non-streaming request still produces correct billing record.
  - go vet ./cloud/ ./serve/cloud/ ./generate/ clean.

- [x] T108.2 Implement SAML XML signature verification (C2) (2026-03-23)
  Owner: Security Eng  Est: 3h  verifies: [UC-003]
  Deps: none
  Files: cloud/sso.go, go.mod
  Acceptance:
  - In ValidateAssertion, verify the XML digital signature against the IdP
    certificate stored in SAMLMetadata.Certificate using crypto/x509 and
    encoding/xml. Use standard library only (no goxmldsig dependency).
  - Parse the SignatureValue and DigestValue from the SAML Response XML.
  - Verify the digest of the signed element matches DigestValue.
  - Verify the signature over the SignedInfo using the IdP certificate.
  - Remove the "In production" comment at line 179.
  - Test: assertion with valid signature passes. Tampered assertion fails.
    Assertion without signature fails.
  - go vet ./cloud/ clean.

- [x] T108.3 Fix differential privacy hardcoded seed (C4) (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: none
  Files: federated/dp.go
  Acceptance:
  - Replace math/rand seeded with 42 with crypto/rand-seeded source.
  - Use crypto/rand to generate 8 bytes, convert to int64 for rand.NewSource.
  - Validate DPConfig: Epsilon > 0, 0 < Delta < 1, ClipNorm > 0.
    Return error from NewDPStrategy if validation fails.
  - Add upper bound check on privacy budget accumulation (H14, M9).
  - Test: two NewDPStrategy calls produce different noise. Invalid config returns error.
  - go vet ./federated/ clean. go test -race ./federated/ pass.

- [x] T108.4 Remove pprof from public health server (C5) (2026-03-23)
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: health/server.go
  Acceptance:
  - Remove pprof handler registrations from Handler() method.
  - Add optional EnablePprof bool field to health.Server config.
  - When enabled, register pprof on a separate localhost-only mux.
  - Test: GET /debug/pprof/ returns 404 on default health server.
  - go vet ./health/ clean.

- [x] T108.5 Implement cloud responseCapture http.Flusher (H13) (2026-03-23)
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: cloud/server.go
  Acceptance:
  - Add Flush() method to responseCapture that delegates to the wrapped
    ResponseWriter if it implements http.Flusher.
  - This unblocks SSE streaming through the cloud middleware chain.
  - Test: streaming response through cloud middleware produces SSE chunks.
  - go vet ./cloud/ clean.

##### Wave 42: High Security Fixes -- Cloud and Billing (5 agents)

- [x] T108.6 Pre-authorize token budget before request execution (H6, F2) (2026-03-23)
  Owner: Security Eng  Est: 2h  verifies: [UC-003]
  Deps: T108.1
  Files: cloud/server.go, cloud/tenant.go
  Acceptance:
  - In billingMiddleware, before calling next.ServeHTTP:
    1. Parse request body to extract max_tokens (default to s.maxTokens).
    2. Call tenant.ConsumeTokens(estimatedTokens). If false, return 429.
    3. After response, reconcile actual vs estimated usage.
  - Check ConsumeTokens return value (currently discarded at line 145-147).
  - Test: tenant with exhausted budget receives 429 before inference runs.
  - go vet ./cloud/ clean.

- [x] T108.7 Make Azure webhook signature validation mandatory (H8) (2026-03-23)
  Owner: Security Eng  Est: 1h  verifies: [infrastructure]
  Deps: none
  Files: marketplace/azure/webhook.go
  Acceptance:
  - When h.Secret is empty, return 500 "webhook secret not configured".
  - Add io.LimitReader(r.Body, 1<<20) for 1 MB body limit (M2).
  - Add timestamp validation: reject webhooks older than 5 minutes (M14).
  - Add operation ID deduplication with sync.Map and 10-minute TTL.
  - Test: empty secret returns 500. Expired timestamp returns 400.
    Replayed operation ID returns 409.
  - go vet ./marketplace/azure/ clean.

- [x] T108.8 Enable GKE private cluster and master authorized networks (H9) (2026-03-23)
  Owner: Infra Eng  Est: 1h  verifies: [infrastructure]
  Deps: none
  Files: infra/terraform/zerfoo-cloud/main.tf
  Acceptance:
  - Add private_cluster_config block with enable_private_nodes = true,
    master_ipv4_cidr_block = "172.16.0.0/28".
  - Add master_authorized_networks_config restricted to VPC CIDR (10.0.0.0/20).
  - Scope node OAuth scopes to devstorage.read_only, logging.write, monitoring (M15).
  - Validate: terraform plan shows expected changes.

- [x] T108.9 Hash tenant API keys (H10) (2026-03-23)
  Owner: Security Eng  Est: 2h  verifies: [UC-003]
  Deps: none
  Files: cloud/tenant.go
  Acceptance:
  - Store SHA-256 hash of API key in Tenant struct, not raw key.
  - In TenantManager.Create, hash the key before storing.
  - In GetByAPIKey, hash the input key and compare hashes.
  - Keep constant-time comparison on the hashes (not raw keys).
  - Use byAPIKey map keyed by hash for O(1) lookup (H15).
  - Test: Tenant struct does not contain raw API key. Lookup by key works.
  - go vet ./cloud/ clean.

- [x] T108.10 Add exponential backoff retry to marketplace metering (H11) (2026-03-23)
  Owner: ML Eng  Est: 2h  verifies: [infrastructure]
  Deps: none
  Files: marketplace/aws/metering.go, marketplace/azure/metering.go,
         marketplace/gcp/metering.go
  Acceptance:
  - Add retry helper: 3 attempts with exponential backoff (1s, 2s, 4s) + jitter.
  - Wrap SubmitUsage calls in all three providers with retry.
  - Log each retry attempt at warn level.
  - If all retries fail, log at error level with full context.
  - Test: mock server returning 429 then 200 succeeds on retry.
  - go vet ./marketplace/... clean.

##### Wave 43: High Security Fixes -- Serve and Auth (5 agents)

- [x] T108.11 Enforce scope-based authorization on endpoints (H3) (2026-03-23)
  Owner: Security Eng  Est: 2h  verifies: [UC-003]
  Deps: none
  Files: serve/server.go, security/apikey.go
  Acceptance:
  - Wire KeyStore into Server via WithKeyStore(ks *security.KeyStore) option.
  - In authMiddleware, after validating Bearer token, look up key in KeyStore.
  - Add scope requirements: DELETE /v1/models requires ScopeAdmin.
    POST /v1/* requires ScopeInference. GET /v1/models requires ScopeReadOnly.
  - Return 403 Forbidden when scope is insufficient.
  - When KeyStore is nil (static API key mode), skip scope checks.
  - Test: inference key can POST but not DELETE. Admin key can do both.
  - go vet ./serve/ clean.

- [x] T108.12 Warn or refuse startup when API key is empty (H4) (2026-03-23)
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go, cmd/cli/serve.go
  Acceptance:
  - When neither --api-key flag nor ZERFOO_API_KEY env var is set, log a
    WARN-level message: "serve: no API key configured, all endpoints are public".
  - Add --allow-no-auth flag (default false). When false and no key is set,
    refuse to start with error: "set --api-key, ZERFOO_API_KEY, or --allow-no-auth".
  - Test: no key + no flag = error. No key + --allow-no-auth = warning + starts.
  - go vet ./serve/ ./cmd/cli/ clean.

- [x] T108.13 Fix ClientIP to validate trusted proxies (H5) (2026-03-23)
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: security/network.go
  Acceptance:
  - Add trustedProxies []string parameter to NewRateLimiter or separate config.
  - In ClientIP, only trust X-Forwarded-For if RemoteAddr is in trustedProxies.
  - When not behind a trusted proxy, always use RemoteAddr.
  - Update serve/server.go to pass trusted proxy config to rate limiter.
  - Test: X-Forwarded-For from untrusted IP is ignored.
  - go vet ./security/ ./serve/ clean.

- [x] T108.14 Fix batch path chat template formatting (H12, F1, F8) (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [UC-001, UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In handleChatCompletions batch path (lines 611-622), replace raw message
    concatenation with s.model.FormatMessages(messages) or equivalent method
    that applies the model's chat template.
  - Ensure batch results include accurate prompt_tokens and completion_tokens.
  - Test: batched chat with system prompt produces correctly formatted input.
    Token counts are non-zero.
  - go vet ./serve/ clean.

- [x] T108.15 Add SAML XXE protection and replay prevention (H1, H2) (2026-03-23)
  Owner: Security Eng  Est: 2h  verifies: [UC-003]
  Deps: T108.2
  Files: cloud/sso.go
  Acceptance:
  - In ParseSAMLMetadata and ValidateAssertion, reject input containing
    "<!DOCTYPE" or "<!ENTITY" (XXE prevention).
  - Validate NotBefore timestamp with 5-minute clock skew tolerance.
  - Track assertion IDs in sync.Map with 10-minute TTL to prevent replay.
  - Reject assertions with previously seen IDs.
  - Test: input with DOCTYPE rejected. Replayed assertion ID rejected.
    Assertion before NotBefore rejected.
  - go vet ./cloud/ clean.

##### Wave 44: Medium Fixes -- Infrastructure and Headers (5 agents)

- [x] T108.16 Add CSP, HSTS, Referrer-Policy to security headers (M12 remainder) (2026-03-23)
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In securityHeadersMiddleware, add:
    Content-Security-Policy: default-src 'none'; frame-ancestors 'none'
    Strict-Transport-Security: max-age=63072000; includeSubDomains
    Referrer-Policy: no-referrer
    Permissions-Policy: camera=(), microphone=(), geolocation=()
  - Test: verify all 7 headers present on response.
  - go vet ./serve/ clean.

- [x] T108.17 Add Vary: Origin to CORS middleware (M13) (2026-03-23)
  Owner: Security Eng  Est: 15m  verifies: [infrastructure]
  Deps: none
  Files: security/network.go
  Acceptance:
  - In CORSPolicy.Middleware, set Vary: Origin header when origin matches.
  - Test: response includes Vary: Origin when CORS headers are set.
  - go vet ./security/ clean.

- [x] T108.18 Pin GitHub Actions to commit SHA (M18) (2026-03-23)
  Owner: Infra Eng  Est: 1h  verifies: [infrastructure]
  Deps: none
  Files: .github/workflows/ci.yml, .github/workflows/release-please.yml,
         .github/workflows/arm64-build.yml, .github/workflows/benchmark.yml
  Acceptance:
  - Replace all tag-based action references (actions/checkout@v4, etc.)
    with SHA-pinned references. Add tag as comment for readability.
  - Verify: all workflows still trigger correctly.

- [x] T108.19 Add NetworkPolicy to Helm chart (M16) (2026-03-23)
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: deploy/helm/zerfoo/templates/networkpolicy.yaml, deploy/helm/zerfoo/values.yaml
  Acceptance:
  - Create networkpolicy.yaml template gated by networkPolicy.enabled value.
  - Restrict ingress to specified namespace (default: ingress-nginx).
  - Add networkPolicy section to values.yaml (enabled: false by default).
  - Verify: helm template renders correctly.

- [x] T108.20 Support API auth middleware + body size limits (C3, M1) (2026-03-23)
  Owner: Security Eng  Est: 2h  verifies: [UC-003]
  Deps: none
  Files: support/api.go
  Acceptance:
  - Wrap all support routes with auth middleware requiring valid Bearer token.
  - In each handler, verify authenticated tenant ID matches customer_id param.
  - Add http.MaxBytesReader(w, r.Body, 1<<20) to POST handlers.
  - Test: unauthenticated request returns 401. Request for another tenant's
    tickets returns 403. Oversized body returns 413.
  - go vet ./support/ clean.

##### Wave 45: Medium Fixes -- Correctness and Performance (5 agents)

- [x] T108.21 Fix batch scheduler context coupling (M22) (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/batch.go
  Acceptance:
  - In executeBatch, use context.Background() or merged context from all live
    requests instead of live[0].ctx for the batch handler call.
  - Cancel merged context only when ALL live requests are canceled.
  - Test: first request disconnect does not cancel remaining batch members.
  - go vet ./serve/ clean.

- [x] T108.22 Make session pool size configurable (F5, M23) (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: inference/load_gguf.go, inference/inference.go
  Acceptance:
  - Add WithSessionPoolSize(n int) LoadOption.
  - Replace hardcoded 8 in make(chan ..., 8) with configurable value.
  - Initialize session pool in assembleModel (M23 fix).
  - Default: 16 sessions. Minimum: 1.
  - Test: LoadFile with custom pool size creates pool of correct size.
  - go vet ./inference/ clean.

- [x] T108.23 Fix isOOMError false positives (F4) (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Replace strings.Contains(msg, "cuda") with specific OOM patterns:
    "out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "CUBLAS_STATUS_ALLOC_FAILED".
  - Test: "cuda driver version mismatch" returns 500 not 503.
    "out of memory" returns 503.
  - go vet ./serve/ clean.

- [x] T108.24 Fix checkStop O(n^2) decoding (A7) (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [UC-001, UC-002]
  Deps: none
  Files: generate/generator.go
  Acceptance:
  - Maintain a running decoded string across decode steps.
  - On each step, decode only the new token and append to the running string.
  - Check stop strings against the running string instead of full re-decode.
  - Test: generation with stop strings produces correct output.
  - Benchmark: 4096-token generation with stop string is measurably faster.
  - go vet ./generate/ clean.

- [x] T108.25 Add graceful shutdown timeout (L9) (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: cmd/cli/serve.go
  Acceptance:
  - Replace context.Background() in shutdown with 30-second timeout context.
  - Log warning if shutdown timeout expires.
  - Test: verify shutdown completes within timeout.
  - go vet ./cmd/cli/ clean.

##### Wave 46: Low Fixes and Tech Debt (5 agents)

- [x] T108.26 Add streaming chunk OpenAI-required fields (A9) (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-001, UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In streamChatCompletion and streamCompletion, add id, object, created,
    model fields to each SSE chunk.
  - Test: streaming response chunks include all required OpenAI fields.
  - go vet ./serve/ clean.

- [x] T108.27 Validate temperature/TopP/TopK ranges (L1) (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Reject temperature < 0. Clamp TopP to [0, 1]. Reject TopK < 0.
  - Return 400 with descriptive error for invalid values.
  - Test: negative temperature returns 400. TopP=1.5 clamped to 1.0.
  - go vet ./serve/ clean.

- [x] T108.28 Register healthz/readyz on main serve mux (A2) (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Register handleHealthz and handleReadyz on the main serve mux.
  - handleHealthz returns 200 {"status":"ok"}.
  - handleReadyz returns 200 if model loaded, 503 otherwise.
  - Test: GET /healthz returns 200. GET /readyz returns 200 with loaded model.
  - go vet ./serve/ clean.

- [x] T108.29 Remove prefix cache dead computation (F7) (2026-03-23)
  Owner: ML Eng  Est: 15m  verifies: [infrastructure]
  Deps: none
  Files: generate/session.go
  Acceptance:
  - Remove unused seqLen computation at lines 133-136.
  - go vet ./generate/ clean.

- [x] T108.30 Use UTC for billing timestamps (F11) (2026-03-23)
  Owner: ML Eng  Est: 15m  verifies: [infrastructure]
  Deps: none
  Files: cloud/billing.go
  Acceptance:
  - Replace time.Now() with time.Now().UTC() in billing record creation.
  - go vet ./cloud/ clean.

##### Wave 47: Verification (2 agents)

- [x] T108.31 Run go test -race on all changed packages (2026-03-23)
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: T108.1-T108.30
  Acceptance:
  - go test -race -timeout 300s ./serve/ ./cloud/ ./serve/cloud/ ./generate/
    ./inference/ ./security/ ./support/ ./federated/ ./health/ ./marketplace/...
  - All tests pass with no races detected.

- [x] T108.32 Run go vet and linter on entire codebase (2026-03-23)
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: T108.31
  Acceptance:
  - go vet ./... clean.
  - golangci-lint run ./... clean (or only warnings in unmodified code).

---

#### E109: Deep Review v1.12.0 Remediation [2026 Q2 -- CRITICAL]

Deep review of v1.12.0 (3 agents, 240 tool calls, 1332 files, ~291K lines) found
4 Critical functional bugs, 4 High error leaks, 13 Medium, 7 Low findings. E108
already addressed billing bypass, SAML hardening, scope enforcement, security
headers, and infrastructure hardening. E109 covers the NEW findings: session pool
leak, batch serialization, normalization mismatch, streaming error ordering, and
raw error leaks in 4 handlers.
Source: .claude/scratch/deep-review-report.md (2026-03-24)

##### Wave 48: Critical Functional Fixes (4 agents)

- [x] T109.1 Defer releaseSession in Model.Generate and GenerateStream (C-001)
  Owner: ML Eng  Est: 15m  verifies: [UC-001, UC-002]
  Deps: none
  Files: inference/inference.go
  Acceptance:
  - In Model.Generate (line 453), change direct call to defer:
    `sess := m.acquireSession()` then `defer m.releaseSession(sess)`.
  - In Model.GenerateStream (line 517), same change: `defer m.releaseSession(sess)`.
  - Matches existing pattern in Model.Chat (line 543-544) which already defers.
  - Test: session pool size does not decrease after a panic during generation.
  - go vet ./inference/ clean. go test -race ./inference/ pass.

- [x] T109.2 Fix GenerateBatch to use session pool instead of generator mutex (C-002)
  Owner: ML Eng  Est: 30m  verifies: [UC-001, UC-002]
  Deps: T109.1
  Files: inference/inference.go
  Acceptance:
  - In GenerateBatch (line 490), replace `m.generator.Generate(ctx, p, sc)` with
    `sess := m.acquireSession(); text, err := sess.Generate(ctx, p, sc); m.releaseSession(sess)`.
  - Use defer for releaseSession.
  - This allows true parallel generation up to the session pool size.
  - Test: GenerateBatch with 4 prompts completes faster than 4 sequential Generate calls.
  - go vet ./inference/ clean. go test -race ./inference/ pass.

- [x] T109.3 Store normalization statistics and apply in PredictWindowed (C-003)
  Owner: ML Eng  Est: 2h  verifies: [UC-026]
  Deps: none
  Files: timeseries/dlinear.go, timeseries/nhits.go, timeseries/cfc.go,
         timeseries/patchtst.go
  Acceptance:
  - Add normMeans [][]float64 and normStds [][]float64 fields to DLinear, NHiTS,
    CfC, and PatchTST structs.
  - In each TrainWindowed, store the returned means and stds from normalizeWindows:
    `windows, m.normMeans, m.normStds = normalizeWindows(windows)`.
  - Add applyNormalization helper that applies stored means/stds to input windows.
  - In each PredictWindowed, if normMeans is non-nil, normalize inputs before forward.
  - Include normMeans and normStds in Save/Load (JSON serialization).
  - Test: train on data with large scale differences, predict on same data, verify
    predictions match training-time forward pass output.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ pass.

- [x] T109.4 Check http.Flusher before WriteHeader in streaming handlers (C-004)
  Owner: ML Eng  Est: 15m  verifies: [UC-001, UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In streamChatCompletion (line 1077), move the Flusher type assertion
    (lines 1083-1087) BEFORE the WriteHeader(200) call (line 1081).
  - Same fix in streamCompletion if the same pattern exists.
  - Test: streaming endpoint with non-Flusher writer returns 500 (not 200 with error body).
  - go vet ./serve/ clean. go test -race ./serve/ pass.

##### Wave 49: High Error Leak Fixes (4 agents)

- [x] T109.5 Sanitize errors in repository handler (H-001)
  Owner: Security Eng  Est: 15m  verifies: [UC-003]
  Deps: none
  Files: serve/repository/handler.go
  Acceptance:
  - Replace all writeError(w, http.StatusInternalServerError, err.Error()) at
    lines 38, 62, 139, 146, 170 with
    writeError(w, http.StatusInternalServerError, "internal server error").
  - Keep the 400-status error messages that describe user input issues (lines 89, 102, 117).
  - Test: trigger a 500 error, verify response body does not contain filesystem paths.
  - go vet ./serve/repository/ clean.

- [x] T109.6 Sanitize Azure webhook processEvent error (H-002)
  Owner: Security Eng  Est: 5m  verifies: [infrastructure]
  Deps: none
  Files: marketplace/azure/webhook.go
  Acceptance:
  - At line 158, replace `http.Error(w, processErr.Error(), http.StatusInternalServerError)`
    with `http.Error(w, "webhook processing failed", http.StatusInternalServerError)`.
  - go vet ./marketplace/azure/ clean.

- [x] T109.7 Use sanitizeError in audio transcription handler (H-003)
  Owner: Security Eng  Est: 5m  verifies: [UC-003]
  Deps: none
  Files: serve/audio.go
  Acceptance:
  - At line 75, replace `writeError(w, inferenceErrorStatus(err), err.Error())`
    with `writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))`.
  - go vet ./serve/ clean.

- [x] T109.8 Sanitize gRPC error in disaggregated gateway SSE stream (H-004)
  Owner: Security Eng  Est: 5m  verifies: [UC-003]
  Deps: none
  Files: serve/disaggregated/gateway.go
  Acceptance:
  - At line 316, replace `fmt.Fprintf(w, "event: error\ndata: %s\n\n", err.Error())`
    with `fmt.Fprintf(w, "event: error\ndata: %s\n\n", "inference failed")`.
  - go vet ./serve/disaggregated/ clean.

##### Wave 50: Verification (1 agent)

- [x] T109.9 Run go test -race and go vet on all changed packages
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: T109.1-T109.8
  Acceptance:
  - go test -race -timeout 300s ./inference/ ./serve/ ./serve/repository/
    ./serve/disaggregated/ ./marketplace/azure/ ./timeseries/ -- all pass.
  - go vet ./... clean.

---

#### E110: GitHub Issues #152-#156 Resolution [2026 Q2]

Five open issues: 2 bugs (NHiTS segfault regression, FreTS NaN), 1 API gap
(CfC engine config), 2 features (iTransformer, Mamba/SSM backend).

##### Wave 51: Bug Fixes + API Gap (3 agents)

- [x] T110.1 Fix NHiTS segfault regression in linearForward (issue #152) (2026-03-24)
  Owner: ML Eng  Est: 2h  verifies: [UC-026]
  Deps: none
  Files: timeseries/nhits.go, timeseries/nhits_test.go
  Acceptance:
  - Root cause: newNHiTSStack creates MLP layers with dimension mismatch when
    inputLen is small relative to poolKernel (pooledLen rounds down to 0, producing
    flatDim=0 which creates a 0-column weight matrix; linearForward then dereferences
    the nil tensor from a 0-dim MatMul).
  - Fix: in newNHiTSStack, clamp pooledLen to minimum of 1. Validate flatDim > 0
    and return a clear error if configuration is incompatible.
  - Add guard in linearForward: if weights.Shape()[1] == 0, return error.
  - Regression test: NHiTS with inputLen=10, channels=10, 3 stacks, 20 epochs.
    Must not panic. Training completes with finite loss.
  - DGX validation: reproduce the exact command from #152 on linux/arm64.
  - Close #152 with fix evidence.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestNHiTS pass.

- [x] T110.2 Implement FreTS backend with normalization and NaN protection (issue #153) (2026-03-24)
  Owner: ML Eng  Est: 4h  verifies: [UC-026]
  Deps: none
  Files: timeseries/frets.go (new), timeseries/frets_test.go (new)
  Acceptance:
  - FreTS (Frequency-enhanced Time Series, ICML 2023) uses discrete Fourier
    transform for channel and temporal mixing.
  - Implement: NewFreTS(config FreTSConfig) with channels, inputLen, outputLen,
    topK (number of frequency components), hiddenSize.
  - Forward: real FFT -> select top-K frequencies -> channel mixing MLP ->
    temporal mixing MLP -> inverse FFT -> linear projection to outputLen.
  - TrainWindowed: normalizeWindows (store stats), warmupLR, NaN/Inf detection,
    AdamW with gradient clipping. Match pattern of existing backends.
  - PredictWindowed: apply stored normalization, forward, return predictions.
  - Save/Load: JSON weights including normMeans/normStds.
  - Tests: convergence on synthetic data, NaN protection on multi-scale data,
    save/load round-trip.
  - Close #153 with evidence.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestFreTS pass.

- [x] T110.3 Add WithEngine option to CfC constructor (issue #154) (2026-03-24)
  Owner: ML Eng  Est: 30m  verifies: [UC-026]
  Deps: none
  Files: timeseries/cfc.go, timeseries/cfc_engine.go
  Acceptance:
  - Add CfCOption type (functional options pattern matching DLinear).
  - Add WithCfCEngine(engine, ops) CfCOption.
  - Change NewCfC signature: NewCfC(config CfCConfig, opts ...CfCOption).
  - Remove or deprecate SetEngine method.
  - Test: NewCfC with WithCfCEngine, verify engine-accelerated TrainWindowed works.
  - Close #154 with evidence.
  - go vet ./timeseries/ clean.

##### Wave 52: New Architectures (2 agents)

- [x] T110.4 Implement iTransformer backend (issue #155) (2026-03-24)
  Owner: ML Eng  Est: 6h  verifies: [UC-026]
  Deps: none
  Files: timeseries/itransformer.go (new), timeseries/itransformer_test.go (new)
  Acceptance:
  - iTransformer (ICLR 2024): inverts attention axis -- each variate is a token,
    attention across variates, FFN learns per-variate temporal patterns.
  - Implement: NewITransformer(config, engine, ops) with nLayers, nHeads, dModel,
    dFF, inputLen, outputLen, channels.
  - Forward: embed each variate (linear inputLen -> dModel) -> transformer encoder
    layers (self-attention over variates, FFN) -> linear projection to outputLen.
  - TrainWindowed: normalizeWindows, warmupLR, NaN/Inf detection, graph.Backward()
    for gradients (engine path), forwardF64 fallback (CPU path).
  - PredictWindowed: apply stored normalization.
  - Tests: convergence on multivariate synthetic data, multi-channel prediction.
  - Close #155 with evidence.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestITransformer pass.

- [x] T110.5 Implement Mamba/SSM backend (issue #156) (2026-03-24)
  Owner: ML Eng  Est: 6h  verifies: [UC-026]
  Deps: none
  Files: timeseries/mamba.go (new), timeseries/mamba_test.go (new)
  Acceptance:
  - Mamba (NeurIPS 2023): selective state space model with input-dependent
    selection mechanism. O(L) complexity for long sequences.
  - Implement: NewMamba(config, engine, ops) with dModel, dState, dConv,
    expandFactor, nLayers, inputLen, outputLen, channels.
  - Forward: linear expansion -> causal conv1d -> SSM (discretize, selective scan)
    -> linear projection -> residual + norm -> output projection.
  - SSM scan: A, B, C, D matrices. Selective: B and C are input-dependent via
    linear projections. Discretize via zero-order hold.
  - TrainWindowed: normalizeWindows, warmupLR, NaN/Inf detection, AdamW.
  - PredictWindowed: apply stored normalization.
  - Tests: convergence, long sequence (inputLen=512) does not OOM.
  - Close #156 with evidence.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestMamba pass.

##### Wave 53: Verification (1 agent)

- [x] T110.6 Run go test -race on timeseries and verify all issues closed (2026-03-24)
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: T110.1-T110.5
  Acceptance:
  - go test -race -timeout 300s ./timeseries/ -- all pass.
  - go vet ./... clean.
  - Verify issues #152-#156 are closed on GitHub.

---

#### E111: Verification Remediation [2026 Q2]

Full-system learning verification (2026-03-24) found 1 bug fixed inline,
plus 2 gaps worth tracking. Source: .claude/scratch/verify-report.md

- [x] T111.1 Fix SimpleRNN bias gradient never computed (2026-03-24)
  Owner: ML Eng  Est: 15m  verifies: [UC-001]
  Deps: none
  Files: layers/recurrent/rnn.go
  Result: Added r.bias.Backward() call in Backward(). All tests pass.

- [ ] T111.2 Implement BatchNorm backward pass for training use
  Owner: ML Eng  Est: 2h  verifies: [UC-016]
  Deps: none
  Files: layers/normalization/batch_norm.go
  Acceptance:
  - BatchNorm.Backward() computes gradients for scale, bias, and input.
  - Gradient matches finite-difference within 1e-3 tolerance.
  - Loss decreases after optimizer step with BatchNorm in the graph.
  - go vet ./layers/normalization/ clean.

- [ ] T111.3 Re-run /verify to confirm all gaps resolved
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: T111.2

---

#### E112: CPU Training Performance -- Issue #157 [2026 Q2 -- CRITICAL]

GitHub issue #157: ALL timeseries backends time out (>300s) on 1K rows x 5 features
x 5 epochs when using the CPU (pure-Go) training path. Target: <1 second.

Root cause (3 researchers, code-level audit):
- ITransformer (~17K params): forward finite differences at lines 416-430. Each
  gradient step does nParams+1 forward passes over the entire batch. Estimated
  381 BILLION ops per epoch. This is the primary offender (~300-1900s alone).
- PatchTST (~15K params): central finite differences in BOTH CPU path
  (patchtst_engine.go:248-265) AND engine path (patchtst_engine.go:133-158).
  2*nParams forward passes per sample. The engine path is equally broken.
- CfC (~6.8K params): full Jacobian [outDim][nParams] via backwardSample (line 362).
  5x slower than necessary (outDim=5). ~290KB transient allocs per sample.

Already fast (proper analytical backprop): DLinear (110 params), NHiTS, FreTS (351 params).
Mamba uses graph.Backward (engine-only, no CPU fallback but CPUEngine works without GPU).

Secondary issues across all backends: flatParams() alloc per batch, math.Pow per
parameter per batch, double decompose() in DLinear, no batch parallelism.

Decision rationale: docs/adr/066-cpu-training-backprop.md

##### Wave 55: Analytical Backprop for Slow Backends (4 agents)

- [x] T112.1 Replace ITransformer finite-difference gradients with analytical backprop (2026-03-24)
  Owner: ML Eng  Est: 4h  verifies: [UC-026]
  Deps: none
  Files: timeseries/itransformer.go
  Acceptance:
  - Remove the forward finite-difference loop at lines 416-430 (the for-each-parameter
    forward-pass perturbation).
  - Implement backwardSample(input, pred, labels, config) that computes gradients via
    chain rule through: output projection, layer norm, FFN (fc2 -> ReLU -> fc1),
    multi-head self-attention (Q/K/V projections, softmax, output projection),
    layer norm, variate embedding.
  - Each layer's backward follows the pattern: dOutput -> dWeight += input^T @ dOutput,
    dBias += sum(dOutput, axis=0), dInput = dOutput @ weight^T.
  - Attention backward: dSoftmax -> mask -> dScores, then dQ/dK/dV via projections.
  - Finite-difference gradient test: for 10 random parameters, verify analytical gradient
    matches (f(x+eps) - f(x-eps)) / 2eps within 1e-4 relative tolerance.
  - Benchmark: 100 samples x 5 channels x 10 epochs completes in <2s on CPU.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestITransformer pass.

- [x] T112.2 Replace PatchTST finite-difference gradients with analytical backprop (BOTH paths) (2026-03-24)
  Owner: ML Eng  Est: 5h  verifies: [UC-026]
  Deps: none
  Files: timeseries/patchtst_engine.go
  Acceptance:
  - Remove the central finite-difference loop in trainWindowedCPU (lines 248-265).
  - ALSO remove the finite-difference loop in trainWindowedEngine (lines 133-158) --
    the engine path is equally broken (uses numerical gradients despite having an engine).
  - Implement backward pass through: output head (linear), transformer encoder layers
    (layer norm, multi-head attention, FFN), positional embedding addition, patch
    embedding (linear). Reuse the forward cache (activations stored during forward).
  - For the engine path, express backward ops as engine tensor ops (MatMul backward,
    etc.) following the NHiTS stackBackwardEngine pattern.
  - For the CPU path, use raw slice loops matching the FreTS backward() pattern.
  - Finite-difference gradient test: for 10 random parameters, verify analytical gradient
    matches (f(x+eps) - f(x-eps)) / 2eps within 1e-4 relative tolerance.
  - Benchmark: 100 samples x 5 channels x 10 epochs completes in <2s on CPU.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestPatchTST pass.

- [x] T112.3 Refactor CfC backwardSample to return single gradient vector (not Jacobian) (2026-03-24)
  Owner: ML Eng  Est: 3h  verifies: [UC-026]
  Deps: none
  Files: timeseries/cfc.go
  Acceptance:
  - Change backwardSample signature from returning [][]float64 (Jacobian) to accepting
    dLoss/dOutput []float64 upstream gradient and returning single []float64 gradient.
  - Propagate dLoss/dOutput through: output projection backward, then BPTT through CfC
    layers (reverse time steps): dH -> (1-f)*dtanh*(W_h^T @ dH) + f*dH_prev, where
    f = exp(-dt/tau).
  - Update TrainWindowed caller (lines 298-312) to compute dLoss/dOutput = 2*diff/N
    and pass it to backwardSample instead of post-multiplying Jacobian rows.
  - Finite-difference gradient test: 10 random parameters within 1e-4 relative tolerance.
  - Benchmark: 100 samples x 5 channels x 10 epochs completes in <2s on CPU.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestCfC pass.

- [x] T112.4 Add CPU fallback for Mamba training (auto-create CPUEngine when nil) (2026-03-24)
  Owner: ML Eng  Est: 1h  verifies: [UC-026]
  Deps: none
  Files: timeseries/mamba.go
  Acceptance:
  - In TrainWindowed, if m.engine is nil, create a temporary CPUEngine:
    engine := compute.NewCPUEngine[float32](numeric.Float32Ops{}).
  - Use the engine-based training path (which already has proper graph.Backward).
  - Do NOT store the temporary engine on the struct (avoid side effects).
  - Benchmark: 100 samples x 5 channels x 10 epochs completes in <2s on CPU.
  - go vet ./timeseries/ clean. go test -race ./timeseries/ -run TestMamba pass.

##### Wave 56: Benchmarks and Verification (2 agents)

- [ ] T112.5 Add CPU training benchmark test for all 7 backends
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: T112.1, T112.2, T112.3, T112.4
  Files: timeseries/benchmark_cpu_test.go
  Acceptance:
  - New test TestAllBackends_CPUTrainingBenchmark that trains each backend on
    1000 samples x 5 channels x 5 epochs and asserts completion in <10s total.
  - Report per-backend timing: samples/sec and total wall time.
  - Assert no NaN/Inf in final loss.
  - go test -tags tabular -run TestAllBackends_CPUTrainingBenchmark -timeout 30s ./timeseries/

- [ ] T112.6 Close GitHub issue #157 with fix evidence
  Owner: ML Eng  Est: 15m  delivers: [issue #157 closed with perf fix]
  Deps: T112.5
  Acceptance:
  - Post comment on #157 citing fix commits, per-backend speedup, and benchmark results.
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

- [x] T106.11 Convert panics to error returns in layers/core/
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

- [x] T106.12 Convert panics to error returns in layers/attention/
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

- [x] T106.14 Add SHA-256 checksum verification to HuggingFace downloads
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

- [x] T106.16 Sanitize inference error messages to clients
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

- [x] T106.17 Add security headers middleware
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add securityHeadersMiddleware that sets X-Content-Type-Options: nosniff,
    X-Frame-Options: DENY, Cache-Control: no-store on all responses.
  - Wire into Handler() middleware chain.
  - Test: verify headers present on response.
  - go vet ./serve/ clean.

- [x] T106.18 Add request ID correlation middleware
  Owner: Security Eng  Est: 1h  verifies: [UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - Add requestIDMiddleware that reads X-Request-Id header or generates UUID.
  - Store in context, include in all log entries, return in response header.
  - Test: verify X-Request-Id in response matches request header when provided,
    or is a valid UUID when not provided.
  - go vet ./serve/ clean.

- [x] T106.19 Fix streaming chat template bypass
  Owner: ML Eng  Est: 1h  verifies: [UC-001, UC-003]
  Deps: none
  Files: serve/server.go
  Acceptance:
  - In streamChatCompletion, replace manual message concatenation (line 766)
    with model.FormatMessages(messages) or equivalent exported method.
  - Verify system prompts and role boundaries are preserved in streaming output.
  - Test: streaming chat with system prompt produces same prompt as non-streaming.
  - go vet ./serve/ clean.

- [x] T106.20 Add request drain on model delete
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

- [x] T106.21 Add integer overflow checks to GGUF tensor parsing
  Owner: ML Eng  Est: 1h  verifies: [UC-001]
  Deps: none
  Files: model/gguf/loader.go, model/gguf/parser.go
  Acceptance:
  - In loader.go, use int64 for numElements. Reject any dimension > MaxInt32.
    Reject total elements > 1<<34 (~16 billion, ~64 GB at float32).
  - In parser.go, reject tensorCount > 100,000 and metadataKVCount > 1,000,000.
  - Test: crafted dimensions that would overflow are rejected with clear error.
  - go vet ./model/gguf/ clean.

- [x] T106.22 Add size limit to OCI blob download
  Owner: ML Eng  Est: 30m  verifies: [UC-005]
  Deps: none
  Files: registry/oci.go
  Acceptance:
  - Replace io.ReadAll(resp.Body) with io.ReadAll(io.LimitReader(resp.Body, 20<<30+1)).
  - If len(data) > 20 GB, return error.
  - Test: verify oversized response is rejected.
  - go vet ./registry/ clean.

- [x] T106.23 Fix JSON injection in support API error response
  Owner: Security Eng  Est: 15m  verifies: [infrastructure]
  Deps: none
  Files: support/api.go
  Acceptance:
  - Replace string concatenation at line 165 with json.NewEncoder(w).Encode.
  - Test: error message containing double quotes is properly escaped.
  - go vet ./support/ clean.

- [x] T106.24 Add pod securityContext to Helm deployment
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: deploy/helm/zerfoo/templates/deployment.yaml, deploy/helm/zerfoo/values.yaml
  Acceptance:
  - Add securityContext: runAsNonRoot: true, runAsUser: 1000,
    readOnlyRootFilesystem: true, allowPrivilegeEscalation: false,
    capabilities: drop: ["ALL"].
  - Add corresponding values to values.yaml.
  - Verify: helm template renders correctly with security context.

- [x] T106.25 Restrict Cloud Run IAM from allUsers
  Owner: Infra Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: infra/terraform/zerfoo-cloud/cloud_run.tf
  Acceptance:
  - Replace allUsers IAM binding with authenticated service account.
  - Add google_service_account resource for API invoker.
  - Verify: terraform plan shows IAM change.

##### Wave 35: Medium Fixes -- Training and Layers (5 agents)

- [x] T106.26 Fix worker pool Close() data race
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: internal/workerpool/pool.go
  Acceptance:
  - Replace `closed bool` with sync.Once.
  - Close() uses p.once.Do(func() { close(p.tasks); p.wg.Wait() }).
  - Test: concurrent Close() calls do not panic.
  - go vet ./internal/workerpool/ clean.

- [x] T106.27 Add gradient clipping and NaN guard to AdamW
  Owner: ML Eng  Est: 1h  verifies: [UC-016]
  Deps: none
  Files: training/optimizer/adamw.go
  Acceptance:
  - Add optional MaxGradNorm float64 field to AdamW config.
  - Before parameter update, compute gradient norm and clip if > MaxGradNorm.
  - Check for NaN/Inf in gradients; return error if detected.
  - Test: gradient with NaN returns error. Gradient exceeding norm is clipped.
  - go vet ./training/optimizer/ clean.

- [x] T106.28 Fix S4 backward nil gradient panic
  Owner: ML Eng  Est: 30m  verifies: [UC-001]
  Deps: none
  Files: layers/ssm/s4.go
  Acceptance:
  - Before accessing Gradient.Data(), check for nil. If nil, initialize to zeros.
  - Test: S4 backward on first call does not panic.
  - go vet ./layers/ssm/ clean.

- [x] T106.29 Fix LoRA backward nil gradient Add
  Owner: ML Eng  Est: 30m  verifies: [UC-016]
  Deps: none
  Files: training/lora/ (identify exact file)
  Acceptance:
  - Before engine.Add(grad, dB), check if grad is nil. If nil, set grad = dB directly.
  - Test: LoRA backward on first call does not panic.
  - go vet ./training/lora/ clean.

- [x] T106.30 Fix PatchTST inference projection head
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

- [x] T106.31 Replace stdlib log with structured logger in 7 files
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

- [x] T106.32 Cache ZERFOO_DEBUG_ONNX env var check
  Owner: ML Eng  Est: 15m  verifies: [UC-002]
  Deps: none
  Files: generate/generator.go
  Acceptance:
  - Add package-level var debugOnnx = os.Getenv("ZERFOO_DEBUG_ONNX") != "".
  - Replace os.Getenv("ZERFOO_DEBUG_ONNX") calls at lines 342, 375, 436 with debugOnnx.
  - go vet ./generate/ clean.

- [x] T106.33 Fix flaky TestMAML_MetaConvergence
  Owner: ML Eng  Est: 30m  verifies: [infrastructure]
  Deps: none
  Files: meta/meta_test.go
  Acceptance:
  - Set fixed random seed (e.g., 42) for deterministic test.
  - Increase tolerance or epochs to ensure convergence within test bounds.
  - Test passes reliably: go test -count=5 -run TestMAML_MetaConvergence ./meta/
  - go vet ./meta/ clean.

- [x] T106.34 Redact tenant API keys from Config()/List() responses
  Owner: Security Eng  Est: 30m  verifies: [UC-003]
  Deps: none
  Files: cloud/tenant.go
  Acceptance:
  - In Tenant.Config(), set APIKey to empty string or redacted placeholder.
  - In TenantManager.List(), same redaction.
  - Test: Config() and List() do not return raw API keys.
  - go vet ./cloud/ clean.

- [x] T106.35 Add OCI reference path traversal check
  Owner: Security Eng  Est: 15m  verifies: [UC-005]
  Deps: none
  Files: registry/oci.go
  Acceptance:
  - In parseReference, reject repository names containing "..".
  - Test: reference with ".." in repository returns error.
  - go vet ./registry/ clean.

##### Wave 37: Verification and Lint (2 agents)

- [x] T106.36 Run go test -race on all changed packages
  Owner: ML Eng  Est: 1h  verifies: [infrastructure]
  Deps: T106.1-T106.35
  Acceptance:
  - go test -race -timeout 300s ./serve/ ./cloud/ ./inference/ ./layers/core/
    ./layers/attention/ ./layers/ssm/ ./registry/ ./model/gguf/ ./generate/
    ./training/optimizer/ ./training/lora/ ./internal/workerpool/ ./support/
    ./meta/ ./inference/timeseries/
  - All tests pass with no races detected.

- [x] T106.37 Run go vet and linter on entire codebase
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
| A | Security Remediation (ACTIVE) | E106, E107, E108 | Merge at Wave 47 lint pass |
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

- [x] T106.16 Sanitize inference error messages
- [x] T106.17 Add security headers middleware
- [x] T106.18 Add request ID correlation middleware
- [x] T106.19 Fix streaming chat template bypass
- [x] T106.20 Add request drain on model delete

#### Wave 34: Medium Fixes -- GGUF and Infrastructure (5 agents)

- [x] T106.21 Add integer overflow checks to GGUF parsing
- [x] T106.22 Add size limit to OCI blob download
- [x] T106.23 Fix JSON injection in support API error response
- [x] T106.24 Add pod securityContext to Helm deployment
- [x] T106.25 Restrict Cloud Run IAM from allUsers

#### Wave 35: Medium Fixes -- Training and Layers (5 agents)

- [x] T106.26 Fix worker pool Close() data race
- [x] T106.27 Add gradient clipping and NaN guard to AdamW
- [x] T106.28 Fix S4 backward nil gradient panic
- [x] T106.29 Fix LoRA backward nil gradient Add
- [x] T106.30 Fix PatchTST inference projection head

#### Wave 36: Tech Debt and Quality (5 agents)

- [x] T106.31 Replace stdlib log with structured logger
- [x] T106.32 Cache ZERFOO_DEBUG_ONNX env var check
- [x] T106.33 Fix flaky TestMAML_MetaConvergence
- [x] T106.34 Redact tenant API keys from Config/List
- [x] T106.35 Add OCI reference path traversal check

#### Wave 37: Verification and Lint (2 agents)

- [x] T106.36 Run go test -race on all changed packages
- [x] T106.37 Run go vet + linter on entire codebase

#### Wave 38: v1.11.0 Critical Gaps (5 agents)

- [x] T107.1 Add MaxBytesReader + sanitizeError + inflight to handleClassify
- [x] T107.2 Convert reducesum Backward panic to error
- [x] T107.3 Convert rl/replay.go panics to errors
- [x] T107.4 Fix NHiTS nil pointer in linearForward (issue #123)
- [x] T107.5 Fix DNS rebinding TOCTOU in SSRF validation

#### Wave 39: v1.11.0 Medium Fixes (5 agents)

- [x] T107.6 Fix ListByCustomer sort algorithm
- [x] T107.7 Add inflight tracking to handleEmbeddings
- [x] T107.8 Add NaN detection to normalizeWindows input
- [x] T107.9 Disable public IPs in AWS QuickStart
- [x] T107.10 Restrict Azure ARM template firewall

#### Wave 40: Verification (2 agents)

- [x] T107.11 Run go test -race on all changed packages
- [x] T107.12 Close GitHub issue #123

#### Wave 41: E108 Critical Security Fixes (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.1 Fix streaming billing bypass (C1)
- [x] T108.2 Implement SAML signature verification (C2)
- [x] T108.3 Fix DP hardcoded seed + config validation (C4, H14, M9)
- [x] T108.4 Remove pprof from public health server (C5)
- [x] T108.5 Implement cloud responseCapture http.Flusher (H13)

#### Wave 42: E108 High Fixes -- Cloud and Billing (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.6 Pre-authorize token budget before request (H6, F2)
- [x] T108.7 Mandatory Azure webhook signature + replay protection (H8, M2, M14)
- [x] T108.8 Enable GKE private cluster + restrict OAuth scopes (H9, M15)
- [x] T108.9 Hash tenant API keys + O(1) lookup (H10, H15)
- [x] T108.10 Add marketplace metering retry with backoff (H11)

#### Wave 43: E108 High Fixes -- Serve and Auth (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.11 Enforce scope-based authorization (H3)
- [x] T108.12 Warn/refuse startup without API key (H4)
- [x] T108.13 Fix ClientIP trusted proxy validation (H5)
- [x] T108.14 Fix batch path chat template formatting (H12, F1, F8) (2026-03-23)
- [x] T108.15 SAML XXE protection + replay prevention (H1, H2) (2026-03-23)

#### Wave 44: E108 Medium Fixes -- Infrastructure (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.16 Add CSP, HSTS, Referrer-Policy headers (M12)
- [x] T108.17 Add Vary: Origin to CORS middleware (M13)
- [x] T108.18 Pin GitHub Actions to commit SHA (M18)
- [x] T108.19 Add NetworkPolicy to Helm chart (M16)
- [x] T108.20 Support API auth + body size limits (C3, M1)

#### Wave 45: E108 Medium Fixes -- Correctness (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.21 Fix batch scheduler context coupling (M22)
- [x] T108.22 Make session pool size configurable (F5, M23)
- [x] T108.23 Fix isOOMError false positives (F4)
- [x] T108.24 Fix checkStop O(n^2) decoding (A7)
- [x] T108.25 Add graceful shutdown timeout (L9)

#### Wave 46: E108 Low Fixes and Tech Debt (5 agents) -- COMPLETE (2026-03-23)

- [x] T108.26 Add streaming chunk OpenAI fields (A9)
- [x] T108.27 Validate temperature/TopP/TopK ranges (L1)
- [x] T108.28 Register healthz/readyz on main mux (A2)
- [x] T108.29 Remove prefix cache dead computation (F7)
- [x] T108.30 Use UTC for billing timestamps (F11)

#### Wave 47: E108 Verification (2 agents)

- [x] T108.31 Run go test -race on all changed packages (2026-03-23)
- [x] T108.32 Run go vet and linter on entire codebase (2026-03-23)

#### Wave 48-50: E109 Deep Review v1.12.0 Remediation (completed 2026-03-23)

- [x] T109.1-T109.9 Security fixes across serve, cloud, inference, training packages

#### Wave 51: E110 Bug Fixes + API Gap (3 agents, completed 2026-03-24)

- [x] T110.1 Fix NHiTS segfault regression in linearForward (issue #152)
- [x] T110.2 Implement FreTS backend with normalization and NaN protection (issue #153)
- [x] T110.3 Add WithCfCEngine option to CfC constructor (issue #154)

#### Wave 52: E110 New Architectures (2 agents, completed 2026-03-24)

- [x] T110.4 Implement iTransformer backend (issue #155)
- [x] T110.5 Implement Mamba/SSM backend wrapping layers/ssm.MambaBlock (issue #156)

#### Wave 53: E110 Verification (completed 2026-03-24)

- [x] T110.6 Full timeseries test suite pass (74s, all pass, race clean, vet clean)

E110 complete (all 6 tasks done).

#### Wave 54: E111 Verification Remediation (completed 2026-03-24)

- [x] T111.1 Fix SimpleRNN bias gradient never computed

E111 T111.1 complete. T111.2-T111.3 remain (BatchNorm backward, re-verify).

#### Wave 55: E112 Analytical Backprop (4 agents)

- [x] T112.1 Replace ITransformer finite-difference gradients with analytical backprop (2026-03-24)
- [x] T112.2 Replace PatchTST finite-difference gradients with analytical backprop (BOTH paths) (2026-03-24)
- [x] T112.3 Refactor CfC backwardSample to return single gradient vector (not Jacobian) (2026-03-24)
- [x] T112.4 Add CPU fallback for Mamba training (auto-create CPUEngine when nil) (2026-03-24)

#### Wave 56: E112 Benchmarks and Verification (2 agents)

- [ ] T112.5 Add CPU training benchmark test for all 7 backends
- [ ] T112.6 Close GitHub issue #157 with fix evidence

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
| R19 | Streaming billing bypass allows unlimited free inference | Critical | High | E108 T108.1 instruments generation layer for token counting. |
| R20 | SAML signature not verified allows auth bypass | Critical | High | E108 T108.2 adds XML signature verification. |
| R21 | Pprof heap dumps expose API keys and tenant data | Critical | Medium | E108 T108.4 removes pprof from public health server. |
| R22 | Cloud SSE streaming broken (responseCapture lacks Flusher) | High | High | E108 T108.5 adds http.Flusher delegation. |

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

### 2026-03-23: E108 created -- Deep review v1.11.1 remediation

Full deep review of v1.11.1 (5 agents, 350+ files, ~298K lines) found 5 Critical,
15 High, 24 Medium, 11 Low findings. Cross-referenced against E106 (37 tasks, all
complete) and E107 (12 tasks, 10 complete). Created E108 with 32 tasks across 7
waves (41-47) covering the REMAINING findings not addressed by E106/E107.

Key new findings:
- C1: Streaming responses completely bypass cloud billing (revenue-critical)
- C2: SAML signature verification still missing (auth bypass)
- C4: Federated DP uses hardcoded seed=42 (privacy guarantee defeated)
- C5: pprof endpoints exposed without auth (secrets in heap dumps)
- H6/F2: Token budget checked post-response, return value discarded
- H13: Cloud responseCapture breaks SSE streaming (all cloud streaming returns 500)
- H11: Zero retry logic in marketplace metering (revenue loss)
- H3: Scope-based authorization exists but never enforced

E107 T107.2 (reducesum panic) and T107.3 (rl/replay panics) still open.

### 2026-03-23: E107 created -- v1.11.0 review remediation + issue #123

Post-E106 deep review of v1.11.0 (4 agents) confirmed all 37 E106 fixes are correct.
Found 2 High gaps: handleClassify missed during E106 (no MaxBytesReader, no sanitizeError,
no inflight tracking), and 1 remaining panic in layers/reducesum/reducesum.go.
Also found 7 Medium findings (DNS rebinding, broken ticket sort, RL panics, marketplace
firewall rules, NaN detection gap) and 8 Low findings.

GitHub issue #123: NHiTS nil pointer dereference in linearForward with 132-channel data.
Created E107 with 12 tasks across 3 waves (38-40).

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

### Current State (2026-03-23)

- **Score:** 158/225 tasks complete (70.2%). E108 added 32 tasks.
- **Active epics:** E107 (2 tasks remaining: T107.2, T107.3), E108 (32 tasks, 7 waves).
- **Last completed:** E106 (security remediation, 37/37 tasks, v1.11.0 released).
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
| E107 remainder | T107.2, T107.3 | Execute remaining E107 tasks |
| E108 remediation | T108.1-T108.32 | Execute E108 waves 41-47 via /apply |
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
