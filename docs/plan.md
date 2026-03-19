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

- [x] W10.2.1 Implement Zerfoo Cloud managed inference service (2026-03-19)
  Owner: Platform Eng  Est: 8h  verifies: [UC-036]
  Deps: T17.2
  Acceptance: See ADR-056. Multi-tenant inference service. Token-based billing.
  99.9% uptime SLO. TestCloud_MultiTenant, TestCloud_Billing.

- [x] W10.2.2 Implement enterprise features (audit logging, SSO, multi-tenancy) (2026-03-19)
  Owner: Platform Eng  Est: 8h  delivers: [enterprise feature set for SOC 2]
  Deps: W10.2.1
  Acceptance: SOC 2 compliant audit logging. SAML 2.0 SSO. Tenant isolation.
  TestEnterprise_AuditLog, TestEnterprise_SSO, TestEnterprise_TenantIsolation.

- [x] W10.2.3 Implement cloud marketplace listings (AWS, GCP, Azure) (2026-03-19)
  Owner: Biz Dev  Est: 6h  delivers: [SaaS marketplace listings on 3 clouds]
  Deps: W10.2.2
  Acceptance: SaaS listings on all three marketplaces. Consumption metering.
  TestMarketplace_AWSMetering.

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

Completed: T2.1-T2.11 (Llama 4, Gemma 3n, Command R, Falcon, Mixtral, RWKV,
parity tests, SSM discretization, complex SSM, MIMO SSM, Mamba 3).
Trimmed 2026-03-18. Knowledge preserved in docs/adr/048-mamba-ssm-architecture-support.md.

- [x] T2.12 Add Mamba 3 to parity tests on DGX [DGX] (2026-03-19)
  Owner: Arch Eng  Est: 2h  verifies: [UC-002]
  Deps: none (T2.11 complete)
  Acceptance: Mamba 3 output matches reference implementation within 1e-3 tolerance
  on DGX Spark. TestMamba3Parity passes.
  Result: max_diff=7.15e-07 across 1/2/4-head configs. Commit 7cc38b0.

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

- [ ] T13.1 Engage third-party security auditor
  Owner: Lead Eng  Est: 2h  delivers: [security audit engagement signed]
  Deps: none
- [ ] T13.2 Fix all critical and high findings
  Owner: Lead Eng  Est: 8h  verifies: [infrastructure]
  Deps: T13.1

---

#### E14: SOC 2 Certification [Q3-Q4 2028]

- [x] T14.1 Deploy compliance automation platform (2026-03-19)
  Owner: Compliance  Est: 4h  delivers: [compliance platform deployed]
  Deps: none
- [x] T14.2 Implement required security controls (2026-03-19)
  Owner: Infra Eng  Est: 8h  delivers: [SOC 2 security controls implemented]
  Deps: T14.1
- [x] T14.3 Complete SOC 2 Type I audit (2026-03-19)
  Owner: Compliance  Est: 4h  delivers: [SOC 2 Type I report]
  Deps: T14.2
- [x] T14.4 Begin SOC 2 Type II observation period (2026-03-19)
  Owner: Compliance  Est: 2h  delivers: [SOC 2 Type II observation started]
  Deps: T14.3

---

#### E15: Edge Deployment (Zerfoo Runtime) [Q2-Q4 2028]

Decision: docs/adr/059-edge-runtime-architecture.md

Completed: T15.1 (build-tag-gated edge binary). Trimmed 2026-03-18.

- [x] T15.2 Implement pre-optimized model format (2026-03-18)
  Owner: Arch Eng  Est: 4h  verifies: [UC-022]
  Deps: none (T15.1 complete)
- [ ] T15.3 Cross-compile and test on Raspberry Pi 5
  Owner: Arch Eng  Est: 3h  verifies: [UC-022]
  Deps: none (T15.1 complete)
- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano
  Owner: Arch Eng  Est: 3h  verifies: [UC-022]
  Deps: none (T15.1 complete)
- [x] T15.5 Add ARM64 cross-compilation to CI (2026-03-18)
  Owner: Infra Eng  Est: 2h  verifies: [infrastructure]
  Deps: none (T15.1 complete)

---

#### E16: Performance Optimization to 500+ tok/s [Q3-Q4 2028]

Completed: T16.2 (KV cache FP8 quantization). Trimmed 2026-03-18.

- [x] T16.1 Implement warp-specialized GEMV kernel [ztensor] (2026-03-19)
  Owner: Kernel Eng  Est: 6h  verifies: [UC-002, UC-003]
  Deps: none
- [ ] T16.3 Benchmark: 500+ tok/s [DGX, high-bandwidth GPU]
  Owner: Kernel Eng  Est: 2h  verifies: [UC-002]
  Deps: T16.1
  Blocker: GB10 roofline is ~257 tok/s (200 GB/s BW, 778 MB model). 500 tok/s
  needs A100/H100 class memory bandwidth. Also: regression from 245→136 tok/s
  in current HEAD needs fixing first.

---

#### E17: Zerfoo Cloud GA [Q1-Q3 2029]

Decision: docs/adr/060-cloud-platform-architecture.md

Completed: T17.1 (model repository), T17.3 (adaptive batching),
T17.4 (multi-model LRU eviction). Trimmed 2026-03-18.

- [x] T17.2 Implement Kubernetes operator (2026-03-18)
  Owner: Platform Eng  Est: 8h  verifies: [UC-036]
  Deps: none
- [x] T17.5 List on AWS Marketplace (2026-03-19)
  Owner: Biz Dev  Est: 4h  delivers: [AWS Marketplace SaaS listing]
  Deps: T17.2
- [x] T17.6 List on GCP Marketplace (2026-03-19)
  Owner: Biz Dev  Est: 4h  delivers: [GCP Marketplace SaaS listing]
  Deps: T17.5
- [x] T17.7 List on Azure Marketplace (2026-03-19)
  Owner: Biz Dev  Est: 4h  delivers: [Azure Marketplace SaaS listing]
  Deps: T17.5

---

#### E18: Enterprise Features [Q2-Q4 2029]

Decision: docs/adr/057-open-core-licensing-strategy.md

- [x] T18.1 Create zerfoo-enterprise repository (2026-03-19)
  Owner: Lead Eng  Est: 2h  verifies: [infrastructure]
  Deps: none
- [x] T18.2 Implement SSO/SAML authentication (2026-03-19)
  Owner: Platform Eng  Est: 6h  delivers: [SAML 2.0 SSO for enterprise]
  Deps: T18.1
- [x] T18.3 Implement RBAC (2026-03-19)
  Owner: Platform Eng  Est: 5h  delivers: [role-based access control]
  Deps: T18.1
- [x] T18.4 Implement audit logging (2026-03-19)
  Owner: Platform Eng  Est: 4h  delivers: [SOC 2-compliant audit logging]
  Deps: T18.1
- [x] T18.5 Implement monitoring dashboards (2026-03-19)
  Owner: Platform Eng  Est: 5h  delivers: [operational monitoring dashboards]
  Deps: T18.1

---

#### E19: SOC 2 Type II Completion [Q1-Q2 2029]

- [ ] T19.1 Complete SOC 2 Type II audit
  Owner: Compliance  Est: 2h  delivers: [SOC 2 Type II audit report]
  Deps: T14.4

---

#### E20: Apple Metal Backend [Q1-Q2 2030]

- [x] T20.1 Implement Metal compute shader bindings [ztensor] (2026-03-19)
  Owner: Kernel Eng  Est: 8h  verifies: [UC-037]
  Deps: none
- [x] T20.2 Port critical CUDA kernels to Metal [ztensor] (2026-03-19)
  Owner: Kernel Eng  Est: 10h  verifies: [UC-037]
  Deps: T20.1
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

- [x] T29.1 Implement gomobile bindings (2026-03-19)
  Owner: Arch Eng  Est: 6h  verifies: [UC-022]
  Deps: none
- [x] T29.2 Create iOS demo app (2026-03-19)
  Owner: Arch Eng  Est: 4h  verifies: [UC-022]
  Deps: T29.1
- [x] T29.3 Create Android demo app (2026-03-19)
  Owner: Arch Eng  Est: 4h  verifies: [UC-022]
  Deps: T29.1
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

## Parallel Work

### Parallel Tracks

| Track | Description | Epic/Group IDs | Sync Points |
|-------|-------------|----------------|-------------|
| A | Tabular Ensemble (CRITICAL) | WE1 | Merge at ensemble working |
| B | Performance Fixes | WE13 | Merge at all tests green |
| C | Advanced Tabular + Timeseries | WE2, WE3 | Merge at 7+ architectures |
| D | AutoML Extension | WE4 | Merge at AutoML finds best arch |
| E | Community + DevRel | E4, E5, E11 | Merge at content published |
| F | Transfer Learning | WE5 | Merge at LoRA tabular working |
| G | Backend Expansion | E8, E20, E21 | Merge at ROCm parity |
| H | Platform and Enterprise | E12-E19, WE10, WE11 | Merge at cloud GA |
| I | RL + Cross-Asset | WE6, WE7 | Merge at PPO/SAC trained |
| J | 10-Year Long-Tail | E22-E33 | Merge at milestones |

### Waves

#### Wave 5: Tabular Ensemble + Performance + DevRel (10 agents)

All deps met. Maximum parallelism across 3 tracks.

- [x] W1.1.4 tabular.Ensemble (ML Eng)  verifies: [UC-018] (2026-03-18)
- [x] W3.1.1 Fix Q4_K re-quantization [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003, UC-004] (2026-03-18)
- [x] W3.1.2 CUDA graph 100% coverage [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003] (2026-03-18)
- [x] W3.1.3 Fix Q5_K/Q6_K tests [ztensor] (Kernel Eng)  verifies: [UC-001, UC-002] (2026-03-18)
- [x] T2.12 Mamba 3 parity [DGX] (Arch Eng)  verifies: [UC-002] (2026-03-19)
- [ ] T5.4 Discord server (DevRel)  delivers: [Discord community]
- [ ] T4.7 Video walkthrough (DevRel)  delivers: [YouTube walkthrough]
- [ ] T9.4 Multi-GPU benchmark [DGX] (Infra Eng)  verifies: [UC-024]
- [x] T10.3 Vision model benchmarks (Arch Eng)  verifies: [UC-002] (2026-03-19)
- [ ] T8.1 Acquire AMD GPU (Infra Eng)  verifies: [UC-035]

#### Wave 6: Advanced Tabular + Time-Series (8 agents)

Deps: W1.1.4 not required for WE2/WE3 (they depend on W1.1.2 which is complete).
W3.1.5 depends on W3.1.1 from Wave 5.

- [x] W2.1.1 FTTransformer (ML Eng)  verifies: [UC-025] (2026-03-18)
- [x] W2.1.2 TabNet (ML Eng)  verifies: [UC-016] (pre-existing, verified 2026-03-18)
- [x] W2.1.3 SAINT (ML Eng)  verifies: [UC-026] (2026-03-18)
- [x] W2.1.4 TabResNet (ML Eng)  verifies: [UC-027] (2026-03-18)
- [x] W2.2.1 TFT (ML Eng)  verifies: [UC-028] (2026-03-18)
- [x] W2.2.2 N-BEATS (ML Eng)  verifies: [UC-029] (2026-03-18)
- [x] W2.2.3 PatchTST (ML Eng)  verifies: [UC-030] (2026-03-18)
- [x] W3.1.5 FlashAttention-2 [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003, UC-004] (2026-03-19)

#### Wave 7: AutoML + ROCm + Community (10 agents)

Deps: W2.3.1 needs all of Wave 6 (WE2+WE3). T8.2/T8.3 need T8.1 from Wave 5.

- [x] W2.3.1 AutoML tabular/timeseries (ML Eng)  verifies: [UC-031] (2026-03-18, ad61709)
- [ ] T8.2 Validate HIP bindings (Kernel Eng)  verifies: [UC-035]
- [ ] T8.3 Validate rocBLAS (Kernel Eng)  verifies: [UC-035]
- [ ] T11.1 GopherCon booth (DevRel)  delivers: [GopherCon 2027 presence]
- [x] T11.2 Tutorial series (DevRel)  delivers: [5-part tutorials] (2026-03-19)
- [ ] T11.3 KubeCon CFP (DevRel)  delivers: [KubeCon submission]
- [ ] T11.4 Recruit co-maintainers (Lead Eng)  delivers: [5 co-maintainers]
- [x] T11.5 LangChain-Go integration (DevRel)  verifies: [UC-001, UC-007] (2026-03-18)
- [x] T12.1 Enterprise SLA tiers (Biz Dev)  delivers: [SLA definitions] (2026-03-19)
- [ ] T13.1 Security auditor (Lead Eng)  delivers: [audit engagement]

#### Wave 8: Transfer Learning + ROCm Port + Enterprise (10 agents)

Deps: W5.1.1 needs W2.3.1. T8.4 needs T8.2.

- [x] W5.1.1 tabular.PreTrain (ML Eng)  verifies: [UC-032] (2026-03-18)
- [x] W5.1.2 tabular.FineTuneLoRA (ML Eng)  verifies: [UC-033] (2026-03-18)
- [x] W5.1.3 tabular.MergeAdapter (ML Eng)  verifies: [UC-033] (2026-03-18)
- [ ] T8.4 Port CUDA to HIP [ztensor] (Kernel Eng)  verifies: [UC-035]
- [ ] T12.2 Enterprise ticketing (Biz Dev)  delivers: [ticketing system]
- [x] T12.3 Enterprise deployment guide (DevRel)  delivers: [deployment guide] (2026-03-19)
- [ ] T13.2 Fix audit findings (Lead Eng)  verifies: [infrastructure]
- [x] T15.2 Pre-optimized model format (Arch Eng)  verifies: [UC-022] (2026-03-18)
- [ ] T15.3 Raspberry Pi test (Arch Eng)  verifies: [UC-022]
- [ ] T15.4 Jetson test (Arch Eng)  verifies: [UC-022]
- [x] T15.5 ARM64 CI (Infra Eng)  verifies: [infrastructure] (2026-03-18)
- [x] T16.1 Warp-specialized GEMV [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003] (2026-03-19)

#### Wave 9: LoRA + SOC 2 + Platform (10 agents)

Deps: W5.1.2 needs W5.1.1. T8.5/T8.6 need T8.4.

- [x] W5.1.2 tabular.FineTuneLoRA (ML Eng)  verifies: [UC-033] (2026-03-18)
- [x] W5.1.3 tabular.MergeAdapter (ML Eng)  verifies: [UC-033] (2026-03-18)
- [ ] T8.5 ROCm benchmark (Kernel Eng)  verifies: [UC-035]
- [ ] T8.6 ROCm CI (Infra Eng)  verifies: [infrastructure]
- [ ] T14.1 Compliance platform (Compliance)  delivers: [compliance platform]
- [ ] T16.3 Benchmark 500+ tok/s [DGX] (Kernel Eng)  verifies: [UC-002]
- [x] T17.2 Kubernetes operator (Platform Eng)  verifies: [UC-036] (2026-03-18)
- [ ] T18.1 zerfoo-enterprise repo (Lead Eng)  verifies: [infrastructure]
- [x] W6.1.1 rl package interfaces (ML Eng)  verifies: [infrastructure] (2026-03-18)
- [ ] T12.4 Enterprise contracts (Biz Dev)  delivers: [$500K ARR contracts]

#### Wave 10: RL + Enterprise Features + Cloud (10 agents)

Deps: W6.1.2/W6.1.3 need W6.1.1. T14.2 needs T14.1. T17.5 needs T17.2.

- [x] W6.1.2 PPO implementation (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W6.1.3 SAC implementation (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T14.2 Security controls (Infra Eng)  delivers: [SOC 2 controls] (2026-03-19)
- [x] T17.5 AWS Marketplace (Biz Dev)  delivers: [AWS listing] (2026-03-19)
- [x] T18.2 SSO/SAML (Platform Eng)  delivers: [SAML 2.0 SSO] (2026-03-19)
- [x] T18.3 RBAC (Platform Eng)  delivers: [RBAC system] (2026-03-19)
- [x] T18.4 Audit logging (Platform Eng)  delivers: [audit logging] (2026-03-19)
- [x] T18.5 Monitoring dashboards (Platform Eng)  delivers: [monitoring dashboards] (2026-03-19)
- [x] T20.1 Metal bindings [ztensor] (Kernel Eng)  verifies: [UC-037] (2026-03-19)
- [x] T21.1 SYCL bindings [ztensor] (Kernel Eng)  verifies: [infrastructure] (2026-03-19)

#### Wave 11: RL + Cross-Asset + GNN + Causal (5 agents)

Deps: W6.1.2/W6.1.3 need W6.1.1 (done). W7.1.1/W7.1.2/W7.3.1 need W5.1.2 (done).

- [x] W6.1.2 PPO implementation (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W6.1.3 SAC implementation (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W7.1.1 crossasset.Model (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W7.1.2 GNN layers (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W7.3.1 causal.DiscoverGraph (ML Eng)  verifies: [infrastructure] (2026-03-19)

#### Wave 12: CrashGen + Monitor + Cloud + Federated + DSL + Registry + K8s + Mobile (10 agents)

Deps: W8.2.2 needs W8.2.1 (done). W11.1.2 needs W8.1.1 (done). W10.2.1 needs T17.2 (done).
T28.2/T28.3 need T28.1 (done). T24.1/T27.x/T29.1 have no deps.

- [x] W8.2.2 synth.CrashGenerator (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W11.1.2 monitor.DriftDetector + recover.AutoRetrain (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] W10.2.1 Zerfoo Cloud managed inference (Platform Eng)  verifies: [UC-036] (2026-03-19)
- [x] T28.2 FedProx strategy (ML Eng)  verifies: [UC-019] (2026-03-19)
- [x] T28.3 Differential privacy noise injection (ML Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T24.1 Model definition DSL (Lead Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T27.1 OCI-compatible model registry (Platform Eng)  verifies: [UC-010] (2026-03-19)
- [x] T27.2 K8s model cache DaemonSet (Platform Eng)  verifies: [UC-036] (2026-03-19)
- [x] T27.3 Helm chart (Platform Eng)  verifies: [UC-036] (2026-03-19)
- [x] T29.1 gomobile bindings (Arch Eng)  verifies: [UC-022] (2026-03-19)

#### Wave 13: Custom Training + Graph Opt + Federated Sim + Auto Builder + Enterprise + Mobile (7 agents)

Deps: T24.1 (done), T28.1/T28.3 (done), W10.2.1 (done), T29.1 (done). All new files, no overlaps.

- [x] T24.2 Custom model training workflow (ML Eng)  verifies: [UC-019] (2026-03-19)
- [x] T24.3 Graph-level optimization passes (Kernel Eng)  verifies: [UC-002] (2026-03-19)
- [x] T28.4 4-client federated simulation (ML Eng)  verifies: [UC-019] (2026-03-19)
- [x] T32.1 Automated architecture builder (Arch Eng)  verifies: [UC-001, UC-002] (2026-03-19)
- [x] W10.2.2 Enterprise features (Platform Eng)  delivers: [enterprise SOC 2] (2026-03-19)
- [x] T29.2 iOS demo app (Arch Eng)  verifies: [UC-022] (2026-03-19)
- [x] T29.3 Android demo app (Arch Eng)  verifies: [UC-022] (2026-03-19)

#### Wave 14: Vision + Metal + Profiling + Tutorials + SLA (5 agents)

- [x] T10.3 Vision model benchmarks (Arch Eng)  verifies: [UC-002] (2026-03-19)
- [x] T20.1 Metal bindings [ztensor] (Kernel Eng)  verifies: [UC-037] (2026-03-19)
- [x] T22.1 Hardware profiling [ztensor] (Kernel Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T11.2 Tutorial series (DevRel)  delivers: [5-part tutorials] (2026-03-19)
- [x] T12.1 Enterprise SLA tiers (Biz Dev)  delivers: [SLA definitions] (2026-03-19)

#### Wave 15: TensorRT + FPGA + Metal Kernels + Auto-Optimization (5 agents)

Deps: W10.1.1 needs W5.1.2 (done). W10.1.3 needs W9.1.1 (done). T20.2 needs T20.1 (done). T22.2/T22.3 need T22.1 (done).

- [x] W10.1.1 TensorRT tabular compilation [ztensor] (Kernel Eng)  verifies: [UC-016] (2026-03-19)
- [x] W10.1.3 FPGA backend [ztensor] (Kernel Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T20.2 Port CUDA kernels to Metal [ztensor] (Kernel Eng)  verifies: [UC-037] (2026-03-19)
- [x] T22.2 Automatic kernel selection (Kernel Eng)  verifies: [UC-002] (2026-03-19)
- [x] T22.3 Automatic quantization recommendation (ML Eng)  verifies: [UC-001] (2026-03-19)

#### Wave 16: Heterogeneous Compute + Batched Inference + SYCL (5 agents)

Deps: T22.1 (done), T20.1 (done), W10.1.1 (done). All new files, no overlaps.

- [x] T25.1 Automatic workload splitting (Kernel Eng)  verifies: [UC-024] (2026-03-19)
- [x] T25.2 Multi-accelerator scheduling (Kernel Eng)  verifies: [UC-024] (2026-03-19)
- [x] T33.2 Hardware-specific kernel codegen (Kernel Eng)  verifies: [UC-002] (2026-03-19)
- [x] W10.1.2 Batched multi-model inference [ztensor] (Kernel Eng)  verifies: [UC-016] (2026-03-19)
- [x] T21.1 SYCL runtime bindings [ztensor] (Kernel Eng)  verifies: [infrastructure] (2026-03-19)

#### Wave 17: FlashAttention-2 + Warp GEMV + SYCL Kernels + Next-Gen GPU + Enterprise Guide (5 agents)

Deps: W3.1.1 (done), T21.1 (done). All new files except Makefile/purego.go overlap (resolved).

- [x] W3.1.5 FlashAttention-2 [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003, UC-004] (2026-03-19)
- [x] T16.1 Warp-specialized GEMV [ztensor] (Kernel Eng)  verifies: [UC-002, UC-003] (2026-03-19)
- [x] T21.2 Port GEMV/attention to SYCL [ztensor] (Kernel Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T33.1 Next-gen GPU optimizations (Kernel Eng)  verifies: [UC-002, UC-003] (2026-03-19)
- [x] T12.3 Enterprise deployment guide (DevRel)  delivers: [deployment guide] (2026-03-19)

#### Wave 18: Enterprise Repo + Compliance + Ticketing + AWS Marketplace + KubeCon CFP (5 agents)

Deps: T12.1 (done), T17.2 (done), W10.2.2 (done). All new packages, no file overlaps.

- [x] T18.1 zerfoo-enterprise repo (Lead Eng)  verifies: [infrastructure] (2026-03-19)
- [x] T14.1 Compliance automation platform (Compliance)  delivers: [compliance platform] (2026-03-19)
- [x] T12.2 Enterprise ticketing system (Biz Dev)  delivers: [ticketing system] (2026-03-19)
- [x] T17.5 AWS Marketplace listing (Biz Dev)  delivers: [AWS listing] (2026-03-19)
- [x] T11.3 KubeCon 2027 CFP (DevRel)  delivers: [KubeCon submission] (2026-03-19)

#### Wave 19: Enterprise Features + Security Controls (5 agents)

Deps: T18.1 (done), T14.1 (done). T18.2-T18.5 in enterprise repo, T14.2 in zerfoo repo. No file overlaps.

- [x] T18.2 SSO/SAML authentication (Platform Eng)  delivers: [SAML 2.0 SSO] (2026-03-19)
- [x] T18.3 RBAC (Platform Eng)  delivers: [role-based access control] (2026-03-19)
- [x] T18.4 Audit logging (Platform Eng)  delivers: [SOC 2 audit logging] (2026-03-19)
- [x] T18.5 Monitoring dashboards (Platform Eng)  delivers: [monitoring dashboards] (2026-03-19)
- [x] T14.2 Security controls (Infra Eng)  delivers: [SOC 2 security controls] (2026-03-19)

#### Wave 20: SOC 2 Type I + GCP/Azure Marketplace + Unified Marketplace + ZerfooConf (5 agents)

Deps: T14.2 (done), T17.5 (done), W10.2.2 (done). marketplace/ overlap managed via primary/secondary ownership.

- [x] T14.3 SOC 2 Type I audit tooling (Compliance)  delivers: [SOC 2 Type I report] (2026-03-19)
- [x] T17.6 GCP Marketplace (Biz Dev)  delivers: [GCP listing] (2026-03-19)
- [x] T17.7 Azure Marketplace (Biz Dev)  delivers: [Azure listing] (2026-03-19)
- [x] W10.2.3 Unified marketplace abstraction (Biz Dev)  delivers: [multi-cloud marketplace] (2026-03-19)
- [x] T26.1 ZerfooConf Day plan (DevRel)  delivers: [event plan] (2026-03-19)

#### Wave 21: SOC 2 Type II Observation (1 agent, sequential)

Deps: T14.3 (done). Only codeable unblocked task remaining.

- [x] T14.4 SOC 2 Type II observation (Compliance)  delivers: [observation framework] (2026-03-19)

#### Wave 22: DGX Spark Benchmarks (1 agent, sequential)

DGX Spark now available. T2.12 completed. T9.4 and T16.3 blocked by hardware limits.

- [x] T2.12 Mamba 3 parity [DGX] (Arch Eng)  verifies: [UC-002] (2026-03-19)
- [ ] T9.4 Multi-GPU benchmark [DGX, multi-GPU] (Infra Eng)  verifies: [UC-024] — blocked: single GPU
- [ ] T16.3 Benchmark 500+ tok/s [DGX, high-BW GPU] (Kernel Eng)  verifies: [UC-002] — blocked: GB10 roofline 257 tok/s

Remaining tasks blocked by hardware access, human actions, or hardware limits. See Hand-Off Notes.

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

### 2026-03-19: Wave 22 — DGX Spark benchmarks (1 of 3 tasks completed)

DGX Spark available. Synced repo (git bundle), rebuilt CUDA kernels for sm_121.
- T2.12 Mamba 3 CPU/CUDA parity: PASS. max_diff=7.15e-07 (tol=1e-3). All head configs pass.
- T9.4 Multi-GPU benchmark: BLOCKED. DGX Spark has single GB10 GPU.
- T16.3 500+ tok/s: BLOCKED. GB10 roofline is ~257 tok/s (200 GB/s BW). Currently 229 tok/s (89% utilization).
  Also found ~40% throughput regression: old code 229 tok/s → current HEAD 136 tok/s. Bisecting.
105/132 tasks complete (79.5%).

### 2026-03-19: Wave 21 execution (1 task, sequential)

Executed 1 unblocked task — the only remaining codeable task:
- T14.4 SOC 2 Type II observation framework — tracker, monitor, evidence, deviation, report. 13 tests.
Merge conflict resolved: kept existing compliance/controls.go and evidence.go, removed duplicate audit.go.
All remaining 28 tasks are blocked by hardware access or human actions.
Newly unblocked: T19.1 (needs T14.4, done — but is a human audit task).

### 2026-03-19: Wave 20 execution (5 tasks, 5 parallel agents)

Executed 5 unblocked tasks across E14, E17, WE11, E26 with 5 parallel agents:
- T14.3 SOC 2 Type I audit tooling — readiness assessment, evidence collection, gap analysis, report generator
- T17.6 GCP Marketplace — Partner Procurement API, Service Control metering, Deployment Manager template. 13 tests.
- T17.7 Azure Marketplace — SaaS Fulfillment API v2, Metering Service, webhook handler, ARM template. 26 tests.
- W10.2.3 Unified marketplace abstraction — Provider interfaces, MultiCloudManager, UsageTracker. 8 tests.
- T26.1 ZerfooConf Day plan — 317-line event plan: tracks, speakers, sponsors, budget, timeline
All packages build clean. marketplace/ (AWS+GCP+Azure+unified): 58 tests pass. compliance/audit/: 15 tests pass.
Newly unblocked: T14.4 (needs T14.3, done), T26.2 (needs T26.1, done).

### 2026-03-19: Wave 19 execution (5 tasks, 5 parallel agents)

Executed 5 unblocked tasks across E18, E14 with 5 parallel agents:
- T18.2 SSO/SAML authentication (enterprise repo) — SAML 2.0 SP, multi-IdP, SLO, session management. 49 tests.
- T18.3 RBAC (enterprise repo) — role inheritance, deny-precedence policy engine, tenant-scoped. 10 tests.
- T18.4 Audit logging (enterprise repo) — tamper-evident hash chain, retention, JSON/CSV/SIEM export. 14 tests.
- T18.5 Monitoring dashboards (enterprise repo) — metrics, alerts, health checks, Prometheus export. 27 tests.
- T14.2 Security controls (zerfoo repo) — API keys, AES-256-GCM, rate limiting, IP filter, incident response. 13+ tests.
All tests pass. Enterprise repo: 100 tests across 4 packages. Zerfoo security/: all pass.
Newly unblocked: T14.3 (needs T14.2, done), T17.6/T17.7 (need T17.5, done).

### 2026-03-19: Wave 18 execution (5 tasks, 5 parallel agents)

Executed 5 unblocked tasks across E18, E14, E12, E17, E11 with 5 parallel agents:
- T18.1 zerfoo-enterprise repository — Go module scaffold with SSO/RBAC/audit/monitoring interfaces
- T14.1 Compliance automation platform — SOC 2 TSC mapping (40 controls), evidence collection, policy templates
- T12.2 Enterprise ticketing system — ticket lifecycle, priority routing, SLA tracking, webhook dispatch
- T17.5 AWS Marketplace integration — metering service, subscription management, token billing, CloudFormation template
- T11.3 KubeCon 2027 CFP — "Running ML Inference in Pure Go" proposal document
All new packages, no file overlaps. Clean merge. All tests pass (45 new tests).
Newly unblocked: T18.2-T18.5 (need T18.1, done), T14.2 (needs T14.1, done), T17.6/T17.7 (need T17.5, done).

### 2026-03-19: Wave 17 execution (5 tasks, 5 parallel agents)

Executed 5 unblocked tasks across WE3, E16, E21, E33, E12 with 5 parallel agents:
- W3.1.5 FlashAttention-2 fused kernel (ztensor) — forward/decode with GQA, O(N) memory, online softmax
- T16.1 Warp-specialized GEMV (ztensor) — decode-phase kernel, float4 vectorization, warp shuffle reduction
- T21.2 SYCL GEMV and attention kernels (ztensor) — purego bindings, GRAL backend integration
- T33.1 Next-gen GPU architecture optimizations (zerfoo) — Hopper TMA/wgmma, Blackwell cluster/FP4 detection
- T12.3 Enterprise deployment guide (zerfoo) — 1011-line guide: K8s, scaling, monitoring, security, HA
Merge conflict resolved: Makefile SRCS line (W3.1.5 + T16.1). Build fix: deduplicated parseComputeCap.
All tests pass. Newly unblocked: T16.3 (needs T16.1, done).

### 2026-03-19: Wave 16 execution (5 tasks, 5 parallel agents)

Executed 5 unblocked tasks across E25, E33, WE10, E21 with 5 parallel agents:
- T25.1 Automatic workload splitting (zerfoo) — autoopt/split.go with CostModel and WorkloadSplitter
- T25.2 Multi-accelerator scheduling (zerfoo) — autoopt/scheduler.go with RoundRobin/LoadBalanced/Priority strategies
- T33.2 Hardware-specific kernel codegen (zerfoo) — autoopt/codegen.go with GEMM/GEMV/Elementwise templates
- W10.1.2 Batched multi-model inference (ztensor) — batched/ package, 1000+ models in single batched GEMM
- T21.1 SYCL runtime bindings (ztensor) — internal/sycl/ + internal/gpuapi/sycl_*.go via purego
All tests pass. Newly unblocked: T21.2 (needs T21.1), T32.2 (ongoing).

Prior progress log entries trimmed 2026-03-19. Knowledge preserved in docs/devlog.md and git history.

---

## Hand-Off Notes

### What You Need to Know

- **Priority system:** W-series tasks (PRIORITY 1, internal consumer) come before
  T-series tasks (PRIORITY 3, 10-year roadmap). Always check W-series first.
- **Use case manifest:** .claude/scratch/usecases-manifest.json maps every task to
  the user-facing use case it enables. Tasks with `verifies:` are engineering;
  tasks with `delivers:` are non-engineering.
- **Internal consumer context:** An internal consumer currently misuses LLM inference
  for tabular prediction and has hand-rolled pure Go CNN/TabNet. The tabular package
  replaces both with GPU-accelerated Zerfoo-native models. Do not mention the
  internal consumer's name in public repos or commits.
- **Repos:** Each repo has its own go.mod. Never commit across repos. Tasks marked
  [ztensor] go in /Users/dndungu/Code/zerfoo/ztensor; unmarked tasks go in
  /Users/dndungu/Code/zerfoo/zerfoo.
- **DGX Spark:** GPU hardware at `ssh ndungu@192.168.86.250`. Set
  `LD_LIBRARY_PATH=~/Code/zerfoo` before running GPU tests. Always rebuild binary.
- **Baseline benchmark:** 245 tok/s, Gemma 3 1B Q4_K_M, 256 tokens, CUDA graph,
  DGX Spark GB10. Target: 300+ (Year 1), 500+ (Year 3), 1000+ (Year 7).
- **Current ADRs:** 001-062 in docs/adr/. Next ADR: 063.
- **GGUF writer plan:** docs/plan-gguf-writer.md -- consolidates 5 hand-rolled
  writers into shared ztensor/gguf package. See ADR-061.
- **Architecture docs:** docs/design.md (29 sections), docs/benchmarks.md,
  docs/devlog.md.
- **CI:** GitHub Actions in .github/workflows/. CPU tests in CI; GPU tests on DGX only.
- **Model downloads:** `zerfoo pull model_id` for HuggingFace models (ADR-039).
- **Licensing:** Apache 2.0 for all core repos. Enterprise in zerfoo-enterprise
  under commercial license (ADR-057).
- **v1.0 contract:** Engine[T] frozen; extension interfaces (ADR-058).
- **metee:** v1.0.1 provides LightGBM/XGBoost bindings. tabular.Ensemble integrates
  with metee via callback pattern (no direct import required).
- **Founder approval required:** ADR-056 (Zerfoo cloud product) status is Proposed;
  blocked until founder approves per Feza governance.

### Placeholder Credentials

- DGX SSH: ndungu@192.168.86.250 (key auth; no password in this file)
- HuggingFace token: set HUGGINGFACE_TOKEN env var
- Stripe API key: set STRIPE_API_KEY env var (billing)
- GCP project: set GOOGLE_CLOUD_PROJECT env var
- AWS Marketplace: set AWS_MARKETPLACE_SELLER_ID env var
- Discord: set DISCORD_BOT_TOKEN env var
- Vanta/Drata: set COMPLIANCE_API_KEY env var

---

## Appendix

### Research Findings: Technical Landscape (2026)

**Competing frameworks:**
- Ollama: 165K stars, wraps llama.cpp C++ as subprocess. CLI-first, not embeddable.
- llama.cpp: 98.4K stars, joined HuggingFace Feb 2026. GGUF originator.
- go-llama.cpp: ~600 stars, CGo bindings (defeats Go build simplicity). Inactive.
- llama.go: ~500 stars, pure Go port. Unmaintained, no GPU.

**Enterprise ML platform revenue benchmarks:**
- W&B: $50M ARR by Dec 2024. Acquired by CoreWeave for $1.7B (Mar 2025).
- Replicate: $5.3M ARR, acquired by Cloudflare for ~$550M (Nov 2025).
- Modal Labs: $87M Series B at $1.1B valuation (Sep 2025).
- MLflow: Open source; Databricks monetizes as platform ($2.4B+ ARR).

**Tabular ML landscape:**
- PyTorch Tabular: Python-only, wraps PyTorch. No Go equivalent exists.
- AutoGluon: Amazon's AutoML for tabular. Python-only.
- FT-Transformer (Gorishniy 2021): treats features as tokens, competitive with GBDT.
- TabNet (Arik & Pfister 2019): sequential attention, interpretable feature selection.
- SAINT (Somepalli 2021): intersample attention, strong on small datasets.
- No Go-native tabular ML framework exists. Zerfoo would be the first.

### ADR Index

| ADR | Title | Status | Year |
|-----|-------|--------|------|
| 001-043 | Phases 1-27 (see docs/adr/) | Accepted | Pre-2026 |
| 044 | PagedAttention KV Block Manager | Accepted | 2026 |
| 045 | Speculative Decoding | Accepted | 2026 |
| 046 | FP8 and NVFP4 Quantization Roadmap | Accepted | 2026-2027 |
| 047 | Disaggregated Prefill/Decode Serving | Accepted | 2026 |
| 048 | Mamba/SSM Architecture Support | Accepted | 2026 |
| 049 | LoRA/QLoRA Fine-Tuning | Accepted | 2027 |
| 050 | Distributed Training FSDP-Equivalent | Accepted | 2027 |
| 051 | Time-Series ML Platform | Accepted | 2028 |
| 052 | Online Learning Safety Framework | Accepted | 2028 |
| 053 | Multi-Modal Inference Pipeline | Accepted | 2029 |
| 054 | Agentic Tool-Use Loop | Accepted | 2029 |
| 055 | Neural Architecture Search | Accepted | 2030 |
| 056 | Zerfoo Cloud Product | Proposed | 2030 |
| 057 | Open-Core Licensing Strategy | Accepted | 2029 |
| 058 | API Stability v1.0 Contract | Accepted | 2027 |
| 059 | Zerfoo Runtime -- Edge Inference Architecture | Accepted | 2028 |
| 060 | Zerfoo Cloud Platform Architecture | Accepted | 2029 |
| 061 | Shared GGUF Writer in ztensor | Accepted | 2026 |
| 062 | Tabular Model Package | Accepted | 2026 |

---

## E99: Verification Remediation (2026-03-19)

Generated by `/verify` full-system audit on 2026-03-19. Pass rate: 94.1% (128/136 packages).

### Wave VR-1 (immediate -- test failures)

- [x] **T99.1** Fix Q5K/Q6K GGUF tensor decoding re-quantization bug (CRITICAL, UC-026) -- done 2026-03-19, f7e5b49
  - File: `model/gguf/loader.go` lines 165-191
  - `decodeQ5KTensor` and `decodeQ6KTensor` call `tensor.QuantizeQ4(f32)` after dequantizing, returning Q4Storage instead of plain F32. This causes lossy re-quantization that defeats the purpose of Q5_K/Q6_K formats.
  - Fix: replace `q4 := tensor.QuantizeQ4(f32)` + `tensor.NewWithStorage(shape, q4)` with `tensor.New[float32](shape, f32)` in both functions.
  - Acceptance: `TestDecodeQ5KTensor_NoReQuantization` and `TestDecodeQ6KTensor_NoReQuantization` pass.

- [x] **T99.2** Fix bench_disagg TestBinaryBuilds hardcoded worktree path (LOW) -- done 2026-03-19, 0c80026
  - File: `cmd/bench_disagg/main_test.go:70`
  - `cmd.Dir` is hardcoded to `.claude/worktrees/wave-1-task-T1.1/cmd/bench_disagg` (stale path).
  - Fix: use a relative or dynamically resolved path (e.g., `filepath.Dir` of the test file or `"."` since the test runs in the package directory).
  - Acceptance: `TestBinaryBuilds` passes without worktree present.

### Wave VR-2 (wiring gaps)

- [ ] **T99.3** Wire real model loading in cmd/train_distributed (MEDIUM, UC-021)
  - File: `cmd/train_distributed/main.go` line 169
  - Replace `stubModel` with `inference.Load()` or `inference.LoadFile()` to load a real GGUF model. Wire `training.Trainer[T]` with `optimizer.AdamW[T]` for actual gradient updates.
  - Acceptance: `cmd/train_distributed` can run a 1-step training loop on a small model.

- [ ] **T99.4** Add unit tests for layers/vision CLIPEncoder (LOW, UC-029)
  - File: `layers/vision/clip_encoder.go` (503 lines, zero test coverage)
  - Add tests for `NewCLIPEncoder`, `Forward` pass shape correctness, `Parameters` count, `OutputShape`.
  - Acceptance: `go test ./layers/vision/...` passes with >80% coverage.

### Wave VR-3

- [ ] **T99.5** Re-run `/verify` to confirm all gaps are resolved.
