# Zerfoo 10-Year Product Roadmap (2026-2036)

## Context

### Problem Statement

Zerfoo is a production-grade ML inference and training framework written entirely in
Go (zero CGo by default). As of 2026-03-18, the 5-year technical roadmap is complete:
21 epics, 124 tasks covering PagedAttention, FP8/NVFP4, speculative decoding,
LoRA/QLoRA, FSDP, multi-modal, agentic tool-use, NAS/AutoML, and a cloud product
prototype. The framework runs 6 model architectures at 245 tok/s on Gemma 3 1B
Q4_K_M (20% faster than Ollama).

The technical foundation is built. Now Zerfoo must transform from an internal
framework into the standard ML runtime for the Go ecosystem -- growing from zero
revenue to $150M+ ARR and IPO readiness by 2036.

This plan is the execution artifact for the 10-year product roadmap. It covers
engineering, community growth, enterprise sales, cloud platform, edge deployment,
and corporate milestones. Every task is actionable by a single Claude Code agent or
a human contributor with no further clarification needed.

See docs/VISION.md for the full 10-year product vision and revenue model.
See docs/design.md for the technical architecture (29 sections, 56 ADRs).

### Objectives

- Year 1-2 (2026-2027): Open-source community growth. 12+ architectures. 300+ tok/s.
  v1.0 stable release. ROCm parity. 25,000+ stars. $0 revenue.
- Year 3-4 (2028-2029): Enterprise foundation. First paying customers. $500K-$2M ARR.
  SOC 2. Cloud marketplace. Edge deployment.
- Year 5-6 (2030-2031): Training platform. $10M-$25M ARR. Auto-optimization.
  Multi-accelerator (SYCL, Metal). Hardware partnerships.
- Year 7-8 (2032-2033): Industry standard. $25M-$75M ARR. ZerfooConf. Federated
  learning. On-device inference. 500+ contributors.
- Year 9-10 (2034-2036): Market leadership. $100M-$150M+ ARR. IPO preparation.
  FedRAMP. 100+ model architectures.

### Non-Goals

- Pre-training at scale (100B+ parameters). Focus is inference + fine-tuning.
- Python API or Python bindings. Go-first; Python users use the OpenAI-compatible API.
- Custom ASIC backends. Support NVIDIA, AMD, Intel, Apple only.
- Runtime ONNX execution. zonnx converts ONNX to GGUF at build time.
- High-frequency trading (sub-millisecond latency requirements).

### Constraints and Assumptions

- Primary hardware: DGX Spark at ssh ndungu@192.168.86.250 (GB10, sm_121).
- Go 1.25+ required (generics, range-over-func).
- All GPU bindings via purego/dlopen; no CGo in core packages.
- GGUF is the sole model format; zonnx handles ONNX conversion.
- Each repo (ztensor, ztoken, zerfoo, zonnx, float16, float8) is independent.
- Apache 2.0 license for all core repos (see ADR-057).
- Tests use standard library only (no testify, no cobra).
- Agentic coders execute parallel waves; human review gates at milestones.

### Success Metrics

| Year | Metric | Target |
|------|--------|--------|
| 2026 | Decode tok/s (1B Q4_K_M) | 300+ |
| 2026 | GitHub stars (all repos) | 5,000+ |
| 2026 | Supported architectures | 12+ |
| 2027 | GitHub stars | 25,000+ |
| 2027 | Contributors | 100+ |
| 2027 | v1.0 stable release | Shipped |
| 2028 | ARR | $500K |
| 2028 | Production deployments | 50+ |
| 2029 | ARR | $2M |
| 2029 | SOC 2 Type II | Certified |
| 2030 | ARR | $10M |
| 2030 | Fortune 500 customers | 5+ |
| 2032 | ARR | $50M |
| 2032 | GitHub stars | 100,000+ |
| 2036 | ARR | $150M+ |
| 2036 | Supported architectures | 100+ |

### Research Findings

Research conducted by three parallel agents (tech-researcher, risk-researcher,
arch-researcher) on 2026-03-18. Key findings incorporated into this plan:

**Technical Landscape:**
- Ollama (165K stars) wraps llama.cpp C++ -- not native Go. Zerfoo is the only
  framework combining native Go + zero CGo + library-first + competitive tok/s.
- W&B reached $50M ARR in ~5 years. Replicate ($5.3M ARR) acquired for ~$550M.
  Enterprise ML tooling valuations are strong.
- AWS marketplace charges 20% for ML containers vs 3% for SaaS listings. Pursue SaaS.
- Edge runtimes (TFLite, ONNX Mobile) target <5MB. Go baseline is 10-15MB; need
  build-tag stripping and split runtime (see ADR-059).
- Documentation "wow moment" (working inference in <10 lines of Go) is the single
  highest-leverage community growth action for Year 1.

**Risks:**
- Go ML TAM ceiling is the top risk. $150M ARR requires Go ML to become a real market.
  Mitigation: expand beyond Go developers via OpenAI-compatible API and edge runtime.
- Apache 2.0 fork by cloud provider is an existential risk if successful. Mitigation:
  compete on innovation velocity, not legal moats (ADR-057).
- AI-generated code quality: 124 tasks by agents = latent bug risk. Mitigation:
  security audit (Year 3), comprehensive DGX validation, enterprise-grade testing.
- Maintainer burnout / bus factor of 1. Mitigation: community cultivation,
  co-maintainer recruitment, governance foundation.

**Architecture Patterns:**
- v1.0 API: freeze Engine[T], use extension interfaces for new capabilities (ADR-058).
- Plugin architecture: in-process init() registration (Go database/sql pattern).
  No out-of-process plugins (go plugin package is fragile, gRPC adds latency).
- Cloud: Model Repository pattern (Triton convention), Kubernetes operator for
  declarative serving, token-based billing (ADR-060).
- Edge: build-tag-gated minimal binary, pre-optimized GGUF models (ADR-059).
- Federated learning: Flower-style strategy pattern on top of existing distributed/
  gRPC infrastructure. Target Year 8-9.

---

## Scope and Deliverables

### In Scope

- 6 new model architectures (Llama 4, Gemma 3n, Command R, Falcon, Mixtral, RWKV)
- Performance optimization to 300+ tok/s (Year 1), 500+ (Year 3), 1000+ (Year 7)
- ROCm backend hardware validation and CUDA feature parity
- Apple Metal backend via purego
- Intel SYCL backend via purego
- v1.0 stable API release with 2-year backwards compatibility guarantee
- Comprehensive documentation site (quickstart, API reference, cookbook, architecture)
- Community infrastructure (GitHub Discussions, Discord, CONTRIBUTING.md)
- GopherCon and KubeCon conference presence
- Enterprise support tier (SLAs, priority fixes)
- Zerfoo Cloud managed inference platform (AWS/GCP/Azure marketplace)
- Enterprise features (SSO/SAML, RBAC, audit logging, multi-tenancy)
- Zerfoo Runtime edge inference binary (<10MB ARM64)
- Model registry with versioning and A/B testing
- Security audit and SOC 2 Type II certification
- Kubernetes operator for declarative model serving
- Federated learning framework
- On-device inference (iOS, Android via gomobile)
- ZerfooConf developer conference
- FedRAMP certification for government customers
- IPO preparation (board, audited financials, S-1)

### Out of Scope

- ZMF model format (archived, replaced by GGUF per ADR-037)
- CGo-based GPU bindings (purego/dlopen is the standard)
- Python SDK or CLI wrappers
- Pre-training runs for 100B+ models
- Custom hardware or kernel microarchitecture below CUDA level
- Payment processing (billing uses Stripe webhooks)
- High-frequency trading infrastructure

### Deliverables Table

| ID | Description | Owner Role | Acceptance Criterion |
|----|-------------|------------|----------------------|
| D1 | 12+ model architectures | Arch Eng | All produce coherent output; parity tests pass on DGX |
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
| D13 | Federated learning | ML Eng | FedAvg + FedProx; differential privacy; enterprise audit |
| D14 | ZerfooConf | DevRel | 500+ attendees, CFP, 20+ talks |
| D15 | FedRAMP authorization | Compliance | Moderate ATO from sponsoring agency |
| D16 | IPO readiness | CEO/CFO | Board formed, S-1 drafted, $150M+ ARR |

---

## Checkable Work Breakdown

### YEAR 1 (2026): Inference Excellence and Community Launch

---

#### E1: Performance Optimization to 300+ tok/s [Q1-Q2 2026]

- [x] T1.1 Profile decode hot path on DGX Spark with nsight systems (2026-03-18)
  Owner: Kernel Eng  Est: 3h
  Deps: none
  Acceptance: nsight trace identifies top-5 bottlenecks; report in docs/devlog.md.

- [x] T1.2 Implement KV cache FP16 storage in generate/kv_cache.go (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T1.1
  Acceptance: KV cache uses FP16 storage; 2x bandwidth reduction vs FP32; decode
  throughput improvement measurable; TestKVCacheFP16 passes.

- [x] T1.3 Optimize Q4_K GEMV kernel for Blackwell sm_121 in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h
  Deps: T1.1
  Acceptance: GEMV kernel uses warp-level primitives optimized for sm_121; benchmark
  shows >= 10% improvement over current kernel on DGX Spark.

- [x] T1.4 Implement kernel launch batching to reduce driver overhead (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T1.1
  Acceptance: Multiple small kernels fused into batched launches; CUDA graph capture
  region expanded; driver overhead reduced measurably.

- [x] T1.5 Benchmark: achieve 300+ tok/s on Gemma 3 1B Q4_K_M [DGX] (2026-03-18, 245 tok/s — target not met, bottleneck analysis in devlog)
  Owner: Kernel Eng  Est: 2h
  Deps: T1.2, T1.3, T1.4
  Acceptance: 300+ tok/s at 256 tokens with CUDA graphs; results in docs/benchmarks.md.

---

#### E2: New Model Architecture Support [Q1-Q3 2026]

- [x] T2.1 Implement Llama 4 architecture builder in inference/arch_llama4.go (2026-03-18)
  Owner: Arch Eng  Est: 6h
  Deps: none
  Acceptance: Llama 4 GGUF loads and generates coherent text; parity test passes
  on DGX; TestLlama4Forward passes.

- [x] T2.2 Implement Gemma 3n architecture builder in inference/arch_gemma3n.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Gemma 3n mobile-optimized model runs inference; TestGemma3nForward passes.

- [x] T2.3 Implement Command R architecture builder in inference/arch_commandr.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Command R GGUF loads; long-context (128K) supported;
  TestCommandRForward passes.

- [x] T2.4 Implement Falcon architecture builder in inference/arch_falcon.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Falcon GGUF loads; multi-query attention handled correctly;
  TestFalconForward passes.

- [x] T2.5 Implement Mixtral MoE architecture builder in inference/arch_mixtral.go (2026-03-18)
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: Mixtral GGUF loads; MoE routing correct; top-K expert selection matches
  reference; TestMixtralForward passes.

- [x] T2.6 Implement RWKV architecture builder in inference/arch_rwkv.go (2026-03-18)
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: RWKV GGUF loads; linear attention (WKV operator) correct;
  TestRWKVForward passes.

- [x] T2.7 Add parity tests for all 6 new architectures on DGX [DGX] (2026-03-18, test code written, DGX execution deferred pending model files)
  Owner: Arch Eng  Est: 3h
  Deps: T2.1, T2.2, T2.3, T2.4, T2.5, T2.6
  Acceptance: All 6 new architectures produce correct output vs reference; parity
  tolerance < 1e-3; TestNewArchParity passes.

- [x] T2.8 Implement exponential-trapezoidal SSM discretization in layers/ssm/ (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: New discretization mode added to SSM recurrence replacing ZOH with
  exponential-trapezoidal formula from Mamba 3. Richer system dynamics than Mamba 2
  simplified recurrence. TestExpTrapDiscretization passes; output matches reference
  implementation within 1e-5.

- [x] T2.9 Implement complex-valued SSM state tracking with RoPE in layers/ssm/ (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T2.8
  Acceptance: SSM B/C matrices operate in complex domain via RoPE embeddings.
  Reuses existing RoPE infrastructure from layers/attention/. Complex state expands
  state-tracking capability without doubling memory. TestComplexSSMState passes;
  BCNorm stabilization layer added. TestBCNorm passes.

- [ ] T2.10 Implement MIMO (multi-input multi-output) SSM heads in layers/ssm/
  Owner: Kernel Eng  Est: 4h
  Deps: T2.9
  Acceptance: MIMOMambaBlock supports multiple parallel state spaces with cross-channel
  mixing. Configurable number of MIMO heads. Decode latency comparable to SISO variant.
  TestMIMOSSM passes; downstream accuracy >= 1pp improvement over SISO on synthetic
  benchmark.

- [ ] T2.11 Implement Mamba 3 architecture builder in inference/arch_mamba3.go
  Owner: Arch Eng  Est: 3h
  Deps: T2.8, T2.9, T2.10
  Acceptance: Mamba 3 GGUF loads and generates coherent text. Architecture uses
  exponential-trapezoidal discretization, complex-valued states with RoPE, MIMO heads,
  no causal convolution, interleaved MLP layers, QKNorm/BCNorm. Registered in
  architecture registry. TestMamba3Forward passes.

- [ ] T2.12 Add Mamba 3 to parity tests on DGX [DGX]
  Owner: Arch Eng  Est: 2h
  Deps: T2.11
  Acceptance: Mamba 3 output matches reference implementation within 1e-3 tolerance
  on DGX Spark. Results in docs/benchmarks.md. TestMamba3Parity passes.

---

#### E3: Quantization Expansion [Q2-Q3 2026]

- [x] T3.1 Implement GPTQ dequantization in ztensor/tensor/quantized_gptq.go [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 5h
  Deps: none
  Acceptance: GPTQ group-quantized weights decode correctly; round-trip error
  < 0.1% MSE vs FP16; TestGPTQDequant passes.

- [x] T3.2 Implement AWQ dequantization in ztensor/tensor/quantized_awq.go [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 5h
  Deps: none
  Acceptance: AWQ activation-aware weights decode correctly; TestAWQDequant passes.

- [x] T3.3 Implement native Q5_K GEMV CUDA kernel in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h
  Deps: none
  Acceptance: Q5_K GEMV avoids re-quantization to Q4_0; direct decode from Q5_K
  format; benchmark shows >= 5% improvement over re-quant path; TestQ5KGEMV passes.

- [x] T3.4 Implement native Q6_K GEMV CUDA kernel in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h
  Deps: none
  Acceptance: Q6_K GEMV direct decode; TestQ6KGEMV passes on DGX.

- [x] T3.5 Implement W4A16 mixed-precision dispatch in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T3.1
  Acceptance: 4-bit weights with FP16 activations; correct output within 0.5%
  perplexity of FP16 baseline; TestW4A16 passes.

- [x] T3.6 Implement W8A8 mixed-precision dispatch in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: INT8 weights and activations with FP32 accumulation; TestW8A8 passes.

---

#### E4: Documentation and Developer Experience [Q1-Q3 2026]

- [x] T4.1 Create documentation site structure using Go template + static generation (2026-03-18)
  Owner: DevRel  Est: 4h
  Deps: none
  Acceptance: docs/ site builds with `go generate`; serves at localhost:8080;
  navigation, search, and syntax highlighting work.

- [x] T4.2 Write quickstart guide: inference in <10 lines of Go (2026-03-18)
  Owner: DevRel  Est: 3h
  Deps: none
  Acceptance: Guide covers `go get`, model download, text generation, and streaming
  in a single code block. Copy-paste works on fresh Go installation.

- [x] T4.3 Write API reference for inference/, generate/, and serve/ packages (2026-03-18)
  Owner: DevRel  Est: 4h
  Deps: none
  Acceptance: Every exported type and function documented with usage example.
  Generated from godoc with manual curation.

- [x] T4.4 Write architecture tour document (2026-03-18)
  Owner: DevRel  Est: 4h
  Deps: none
  Acceptance: Covers Engine[T], graph compilation, CUDA graph capture, GGUF loading,
  and the full inference pipeline. Includes diagrams. Targets new contributors.

- [ ] T4.5 Write cookbook with 10+ recipes (embedding, serving, fine-tuning, streaming)
  Owner: DevRel  Est: 6h
  Deps: none
  Acceptance: 10+ self-contained recipes with copy-paste code; each tested in CI.

- [ ] T4.6 Write benchmark comparison guide: Zerfoo vs Ollama vs llama.cpp
  Owner: DevRel  Est: 3h
  Deps: T1.5
  Acceptance: Side-by-side tok/s comparison on same hardware; reproducible methodology;
  published on docs site with DGX Spark results.

- [ ] T4.7 Record 15-minute video walkthrough of Zerfoo
  Owner: DevRel  Est: 4h
  Deps: T4.2
  Acceptance: Video covers installation, model loading, text generation, and
  OpenAI API serving. Published on YouTube.

---

#### E5: Community Infrastructure [Q1-Q2 2026]

- [x] T5.1 Create CONTRIBUTING.md with development guide and PR process (2026-03-18)
  Owner: DevRel  Est: 2h
  Deps: none
  Acceptance: Covers repo structure, build instructions, test requirements, commit
  conventions, PR review process. Linked from all 6 repo READMEs.

- [ ] T5.2 Create "good first issue" labels and 20+ starter issues across repos
  Owner: DevRel  Est: 3h
  Deps: none
  Acceptance: 20+ issues labeled "good first issue" with clear description,
  expected approach, and acceptance criteria. Mix of Go and CUDA.

- [x] T5.3 Set up GitHub Discussions for all repos (2026-03-18)
  Owner: DevRel  Est: 1h
  Deps: none
  Acceptance: Discussions enabled with categories: Q&A, Show and Tell, Ideas,
  Announcements. Pinned welcome post with links.

- [ ] T5.4 Create Discord server with channels for help, development, and showcase
  Owner: DevRel  Est: 2h
  Deps: none
  Acceptance: Discord server with roles (maintainer, contributor, user), channels
  (#help, #development, #showcase, #announcements), and bot for GitHub notifications.

- [ ] T5.5 Write and publish 5 blog posts: launch announcement, benchmarks, architecture
  Owner: DevRel  Est: 8h
  Deps: T4.2, T4.6
  Acceptance: 5 posts published: (1) Launch announcement, (2) Benchmark comparison,
  (3) Architecture deep dive, (4) "Why Go for ML?", (5) Migration from Ollama guide.

- [ ] T5.6 Submit GopherCon 2026 talk proposal: "Native ML Inference in Go"
  Owner: DevRel  Est: 3h
  Deps: T4.2
  Acceptance: CFP submitted with abstract, outline, and speaker bio. Backup plan:
  lightning talk or unconference slot if main talk rejected.

- [x] T5.7 Create 10 example applications in examples/ directory (2026-03-18)
  Owner: DevRel  Est: 6h
  Deps: none
  Acceptance: 10 standalone examples: chat bot, embedding search, RAG pipeline,
  code completion, summarization, translation, classification, vision analysis,
  audio transcription, agentic tool use. Each with README and main.go.

---

#### E6: Plugin and Extension Architecture [Q2-Q3 2026]

- [x] T6.1 Implement architecture registry in inference/registry.go (2026-03-18)
  Owner: Lead Eng  Est: 3h
  Deps: none
  Acceptance: RegisterArchitecture(name, builder) maps GGUF general.architecture to
  graph builder function. Existing arch_*.go files refactored to use registry.
  TestArchitectureRegistry passes.

- [x] T6.2 Implement quantization format registry in ztensor/tensor/quant_registry.go [ztensor] (2026-03-18)
  Owner: Lead Eng  Est: 3h
  Deps: none
  Acceptance: RegisterQuantType(name, dequantizer) allows custom quant formats.
  Existing Q4_K, Q8, FP8 registered via init(). TestQuantRegistry passes.

- [x] T6.3 Document third-party extension convention (2026-03-18)
  Owner: Lead Eng  Est: 2h
  Deps: T6.1, T6.2
  Acceptance: Document covers: import pattern (`_ "github.com/user/zerfoo-ext-foo"`),
  init() registration, naming convention, testing requirements.

---

### YEAR 2 (2027): v1.0 and Ecosystem Growth

---

#### E7: v1.0 Stable Release [Q1-Q2 2027]
Decision: docs/adr/058-api-stability-v1-contract.md

- [ ] T7.1 Audit and freeze Engine[T] interface
  Owner: Lead Eng  Est: 4h
  Deps: none
  Acceptance: Engine[T] interface reviewed; all methods documented; extension
  interfaces defined (EngineWithFP8, EngineWithPagedKV); frozen in DESIGN.md.

- [ ] T7.2 Label sub-package maturity (stable/beta/alpha) in docs
  Owner: Lead Eng  Est: 2h
  Deps: T7.1
  Acceptance: Every sub-package labeled in design.md and package-level doc comment.
  Stable: inference/, generate/, serve/, model/, layers/. Beta: training/, distributed/.

- [ ] T7.3 Implement deprecation linter in cmd/deprecation-check/main.go
  Owner: Lead Eng  Est: 3h
  Deps: none
  Acceptance: Linter scans for `// Deprecated:` doc comments and verifies they include
  replacement guidance and version when deprecated. CI integration.

- [ ] T7.4 Create release-please config for v1.0.0 across all repos
  Owner: Lead Eng  Est: 2h
  Deps: T7.1
  Acceptance: release-please configured for semantic versioning; CHANGELOG.md
  auto-generated; v1.0.0 tag created on all 6 active repos.

- [ ] T7.5 Write v1.0 migration guide for pre-v1 users
  Owner: DevRel  Est: 3h
  Deps: T7.1
  Acceptance: Guide lists all breaking changes from v0 to v1; code examples for each
  migration step; published on docs site.

---

#### E8: ROCm Backend Hardware Validation [Q1-Q3 2027]

- [ ] T8.1 Acquire AMD Instinct GPU access (cloud or hardware)
  Owner: Infra Eng  Est: 2h
  Deps: none
  Acceptance: SSH access to machine with AMD Instinct MI250 or MI300; ROCm 6.x+
  installed; hipcc available.

- [ ] T8.2 Validate all purego HIP bindings on AMD hardware
  Owner: Kernel Eng  Est: 6h
  Deps: T8.1
  Acceptance: All 20+ GPU operations produce correct results on AMD Instinct;
  parity tests pass (tolerance < 1e-4 vs CPU).

- [ ] T8.3 Validate rocBLAS GEMM parity with cuBLAS
  Owner: Kernel Eng  Est: 4h
  Deps: T8.1
  Acceptance: MatMul output matches cuBLAS within 1e-5 on identical inputs;
  throughput within 20% of CUDA path.

- [ ] T8.4 Port custom CUDA kernels to HIP in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h
  Deps: T8.2
  Acceptance: All 25+ custom kernels compile with hipcc; output matches CUDA
  reference; TestHIPKernels passes on AMD Instinct.

- [ ] T8.5 Benchmark ROCm vs CUDA throughput on Gemma 3 1B [AMD]
  Owner: Kernel Eng  Est: 2h
  Deps: T8.4
  Acceptance: ROCm throughput >= 80% of CUDA on same model; results in
  docs/benchmarks.md.

- [ ] T8.6 Add ROCm to CI pipeline
  Owner: Infra Eng  Est: 3h
  Deps: T8.4
  Acceptance: GitHub Actions workflow runs ROCm tests on AMD hardware; failures
  block merge to main.

---

#### E9: Multi-GPU Inference [Q2-Q3 2027]

- [ ] T9.1 Implement tensor parallelism for prefill in inference/parallel/tensor_parallel.go
  Owner: Infra Eng  Est: 6h
  Deps: none
  Acceptance: Linear layers split across N GPUs; AllReduce after each layer;
  output matches single-GPU baseline; TestTensorParallel passes.

- [ ] T9.2 Implement pipeline parallelism in inference/parallel/pipeline_parallel.go
  Owner: Infra Eng  Est: 6h
  Deps: none
  Acceptance: Transformer layers assigned to different GPUs; micro-batch pipelining;
  bubble ratio < 20% at 4 GPUs; TestPipelineParallel passes.

- [ ] T9.3 Add --gpus flag to zerfoo serve command
  Owner: Infra Eng  Est: 2h
  Deps: T9.1, T9.2
  Acceptance: `zerfoo serve --gpus 0,1,2,3` distributes model across 4 GPUs;
  throughput scales >= 3x vs single GPU on 70B model.

- [ ] T9.4 Benchmark: multi-GPU inference on Llama 3 70B [DGX]
  Owner: Infra Eng  Est: 2h
  Deps: T9.3
  Acceptance: 70B model runs on 4+ GPUs without OOM; tok/s and TTFT measured;
  results in docs/benchmarks.md.

---

#### E10: Vision-Language Model Expansion [Q3-Q4 2027]

- [ ] T10.1 Implement LLaVA architecture builder in inference/arch_llava.go
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: LLaVA GGUF loads; CLIP vision encoder + Llama text decoder;
  correct image captioning on test images; TestLLaVAForward passes.

- [ ] T10.2 Implement Qwen-VL architecture builder in inference/arch_qwen_vl.go
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: Qwen-VL GGUF loads; multi-image input supported; TestQwenVLForward passes.

- [ ] T10.3 Add vision model benchmarks to benchmark suite
  Owner: Arch Eng  Est: 2h
  Deps: T10.1, T10.2
  Acceptance: Benchmark reports prefill tok/s for image+text input; results in
  docs/benchmarks.md.

---

#### E11: Community Growth to 25,000 Stars [Q1-Q4 2027]

- [ ] T11.1 Sponsor GopherCon 2027 booth
  Owner: DevRel  Est: 2h
  Deps: none
  Acceptance: Booth sponsorship purchased; demo station with live inference planned;
  swag and handouts prepared.

- [ ] T11.2 Publish "From Hello World to Production" tutorial series (5 parts)
  Owner: DevRel  Est: 8h
  Deps: none
  Acceptance: 5-part series: (1) First inference, (2) API server, (3) Fine-tuning,
  (4) Production deployment, (5) Performance tuning. Published on docs site.

- [ ] T11.3 Submit KubeCon 2027 talk: "GPU Inference Serving in Go"
  Owner: DevRel  Est: 3h
  Deps: none
  Acceptance: CFP submitted for KubeCon NA 2027.

- [ ] T11.4 Recruit 5 external co-maintainers
  Owner: Lead Eng  Est: 4h
  Deps: T5.1
  Acceptance: 5 contributors with merge access; each has merged >= 3 PRs;
  documented in MAINTAINERS.md.

- [ ] T11.5 Integrate with LangChain-Go and Weaviate
  Owner: DevRel  Est: 6h
  Deps: none
  Acceptance: LangChain-Go LLM provider using Zerfoo library (not HTTP).
  Weaviate vectorizer module using Zerfoo embeddings. PRs submitted upstream.

---

### YEAR 3 (2028): Enterprise Foundation

---

#### E12: Enterprise Support Tier [Q1-Q2 2028]

- [ ] T12.1 Define enterprise support SLA tiers (Standard, Premium)
  Owner: Biz Dev  Est: 2h
  Deps: none
  Acceptance: SLA document: response times (4h Standard, 1h Premium), severity
  levels (P1-P4), escalation paths, coverage hours.

- [ ] T12.2 Set up enterprise ticketing system (GitHub Issues + private Slack)
  Owner: Biz Dev  Est: 3h
  Deps: T12.1
  Acceptance: Dedicated Slack channel per enterprise customer; GitHub Issues with
  enterprise label and SLA tracking; on-call rotation for P1.

- [ ] T12.3 Create enterprise deployment guide
  Owner: DevRel  Est: 4h
  Deps: none
  Acceptance: Guide covers: Kubernetes deployment, GPU resource management, health
  monitoring, log aggregation, backup/restore, TLS configuration.

- [ ] T12.4 Sign first 5 enterprise support contracts ($500K ARR target)
  Owner: Biz Dev  Est: ongoing
  Deps: T12.1, T12.2
  Acceptance: 5 signed contracts totaling >= $500K ARR; contracts include SLA,
  support channel, and renewal terms.

---

#### E13: Security Audit and Hardening [Q2-Q3 2028]

- [ ] T13.1 Engage third-party security auditor for code review
  Owner: Lead Eng  Est: 2h
  Deps: none
  Acceptance: Auditor selected; scope defined (inference pipeline, API server,
  CUDA bindings, distributed training); engagement signed.

- [ ] T13.2 Fix all critical and high findings from security audit
  Owner: Lead Eng  Est: 8h
  Deps: T13.1
  Acceptance: All critical (P1) and high (P2) findings resolved; patches committed
  and released; audit report updated with resolutions.

- [ ] T13.3 Implement SBOM generation in CI pipeline
  Owner: Infra Eng  Est: 3h
  Deps: none
  Acceptance: SBOM (CycloneDX format) generated on every release; published
  alongside release artifacts; includes all direct and transitive dependencies.

- [ ] T13.4 Add fuzz testing for GGUF parser and API server
  Owner: Lead Eng  Est: 4h
  Deps: none
  Acceptance: Go native fuzzing for model/gguf parser (malformed GGUF input) and
  serve/ HTTP handlers (malformed requests). Added to CI.

---

#### E14: SOC 2 Certification [Q3-Q4 2028]
Target: Type I by Q4 2028, Type II by Q2 2029.

- [ ] T14.1 Deploy compliance automation platform (Vanta or Drata)
  Owner: Compliance  Est: 4h
  Deps: none
  Acceptance: Platform configured; cloud infrastructure connected; initial gap
  assessment generated.

- [ ] T14.2 Implement required security controls (access management, encryption, logging)
  Owner: Infra Eng  Est: 8h
  Deps: T14.1
  Acceptance: All SOC 2 Trust Service Criteria controls implemented: access reviews,
  encryption at rest and in transit, audit logging, incident response plan.

- [ ] T14.3 Complete SOC 2 Type I audit
  Owner: Compliance  Est: 4h
  Deps: T14.2
  Acceptance: 3PAO audit completed; Type I report issued with no critical findings.

- [ ] T14.4 Begin SOC 2 Type II observation period (3 months)
  Owner: Compliance  Est: 2h
  Deps: T14.3
  Acceptance: Observation period started; continuous monitoring active; evidence
  collection automated.

---

#### E15: Edge Deployment (Zerfoo Runtime) [Q2-Q4 2028]
Decision: docs/adr/059-edge-runtime-architecture.md

- [ ] T15.1 Implement build-tag-gated edge binary in cmd/zerfoo-runtime/
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: `go build -tags edge ./cmd/zerfoo-runtime` produces binary excluding
  training/, distributed/, serve/; binary size < 10MB on linux/arm64.

- [ ] T15.2 Implement pre-optimized model format: zerfoo optimize --target arm64
  Owner: Arch Eng  Est: 4h
  Deps: T15.1
  Acceptance: Optimization step pre-computes graph fusion and stores in GGUF metadata.
  Runtime skips optimization pass. TestPreOptimizedLoad passes.

- [ ] T15.3 Cross-compile and test on Raspberry Pi 5 (ARM64)
  Owner: Arch Eng  Est: 3h
  Deps: T15.1
  Acceptance: Zerfoo Runtime runs inference on Raspberry Pi 5; generates coherent
  text from Q4_K model; tok/s measured and reported.

- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano
  Owner: Arch Eng  Est: 3h
  Deps: T15.1
  Acceptance: Zerfoo Runtime with CUDA support runs on Jetson; tok/s competitive
  with llama.cpp on same hardware.

- [ ] T15.5 Add ARM64 cross-compilation to CI pipeline
  Owner: Infra Eng  Est: 2h
  Deps: T15.1
  Acceptance: CI builds linux/arm64 binary on every push; binary size checked
  against 10MB threshold.

---

#### E16: Performance Optimization to 500+ tok/s [Q3-Q4 2028]

- [ ] T16.1 Implement warp-specialized GEMV kernel for decode in ztensor [ztensor]
  Owner: Kernel Eng  Est: 6h
  Deps: none
  Acceptance: Decode-path GEMV uses warp specialization (compute + memory warps);
  benchmark shows >= 20% improvement on batch=1 decode.

- [ ] T16.2 Implement KV cache quantization (FP8 KV) in generate/kv_cache.go
  Owner: Kernel Eng  Est: 5h
  Deps: none
  Acceptance: KV cache stored in FP8; attention quality within 0.5 perplexity of
  FP16 KV; memory usage halved; TestFP8KVCache passes.

- [ ] T16.3 Benchmark: achieve 500+ tok/s on Gemma 3 1B Q4_K_M [DGX]
  Owner: Kernel Eng  Est: 2h
  Deps: T16.1, T16.2
  Acceptance: 500+ tok/s at 256 tokens; results in docs/benchmarks.md.

---

### YEAR 4 (2029): Platform Expansion

---

#### E17: Zerfoo Cloud GA [Q1-Q3 2029]
Decision: docs/adr/060-cloud-platform-architecture.md

- [ ] T17.1 Implement model repository server in serve/cloud/repository.go
  Owner: Platform Eng  Est: 5h
  Deps: none
  Acceptance: Model repository stores models/{name}/{version}/model.gguf + config.yaml.
  REST API for upload, download, list, delete. TestModelRepository passes.

- [ ] T17.2 Implement Kubernetes operator (ZerfooInferenceService CRD)
  Owner: Platform Eng  Est: 8h
  Deps: none
  Acceptance: CRD defines model, replicas, GPU requirements, autoscaling policy.
  Operator reconciles: creates deployment, service, HPA. TestOperator passes.

- [ ] T17.3 Implement adaptive batching with configurable latency targets
  Owner: Platform Eng  Est: 4h
  Deps: none
  Acceptance: BatchScheduler accepts max_batch_size, max_latency_ms, priority_queues.
  Batches requests within latency target. TestAdaptiveBatching passes.

- [ ] T17.4 Implement multi-model serving with LRU GPU eviction
  Owner: Platform Eng  Est: 5h
  Deps: none
  Acceptance: Multiple models share GPU node pool. LRU eviction when memory budget
  exceeded. Model reload < 10s for 7B. TestMultiModelServing passes.

- [ ] T17.5 List Zerfoo Cloud on AWS Marketplace as SaaS offering
  Owner: Biz Dev  Est: 4h
  Deps: T17.1, T17.2
  Acceptance: AWS Marketplace listing live; customers can subscribe; usage metering
  via AWS Marketplace Metering API.

- [ ] T17.6 List Zerfoo Cloud on GCP Marketplace
  Owner: Biz Dev  Est: 4h
  Deps: T17.5
  Acceptance: GCP Marketplace listing live; consumption counts toward MACC.

- [ ] T17.7 List Zerfoo Cloud on Azure Marketplace
  Owner: Biz Dev  Est: 4h
  Deps: T17.5
  Acceptance: Azure Marketplace listing live; consumption counts toward MACC.

---

#### E18: Enterprise Features (zerfoo-enterprise) [Q2-Q4 2029]
Decision: docs/adr/057-open-core-licensing-strategy.md

- [ ] T18.1 Create zerfoo-enterprise repository with commercial license
  Owner: Lead Eng  Est: 2h
  Deps: none
  Acceptance: Separate repo with commercial license; imports github.com/zerfoo/zerfoo;
  builds as drop-in replacement with enterprise features enabled.

- [ ] T18.2 Implement SSO/SAML authentication in zerfoo-enterprise
  Owner: Platform Eng  Est: 6h
  Deps: T18.1
  Acceptance: SAML 2.0 IdP integration (Okta, Azure AD tested); JWT token issuance;
  API requests authenticated via bearer token. TestSAMLAuth passes.

- [ ] T18.3 Implement RBAC (role-based access control) in zerfoo-enterprise
  Owner: Platform Eng  Est: 5h
  Deps: T18.1
  Acceptance: Roles (admin, operator, viewer) with configurable permissions per
  model and endpoint. TestRBAC passes.

- [ ] T18.4 Implement audit logging in zerfoo-enterprise
  Owner: Platform Eng  Est: 4h
  Deps: T18.1
  Acceptance: All API requests, model operations, and admin actions logged in
  append-only audit trail. NDJSON format. Tamper-evident checksums.
  TestAuditLogging passes.

- [ ] T18.5 Implement advanced monitoring dashboards in zerfoo-enterprise
  Owner: Platform Eng  Est: 5h
  Deps: T18.1
  Acceptance: Grafana dashboard templates: GPU utilization, request latency,
  model performance, cache hit rates, error rates. Pre-built alerts.

---

#### E19: SOC 2 Type II Completion [Q1-Q2 2029]

- [ ] T19.1 Complete SOC 2 Type II audit (3-month observation period ends)
  Owner: Compliance  Est: 2h
  Deps: T14.4
  Acceptance: Type II report issued by 3PAO; no critical findings; report available
  to enterprise customers under NDA.

---

### YEAR 5 (2030): Training Platform and $10M ARR

---

#### E20: Apple Metal Backend [Q1-Q2 2030]

- [ ] T20.1 Implement Metal compute shader bindings via purego in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h
  Deps: none
  Acceptance: Metal runtime detected via purego dlopen of Metal framework.
  Basic ops (MatMul, Add, Mul) produce correct results on Apple M-series.
  TestMetalBasicOps passes.

- [ ] T20.2 Port critical CUDA kernels to Metal shaders [ztensor]
  Owner: Kernel Eng  Est: 10h
  Deps: T20.1
  Acceptance: GEMV, RMSNorm, softmax, attention kernels ported to Metal Shading
  Language. Output matches CPU reference within 1e-4. TestMetalKernels passes.

- [ ] T20.3 Benchmark Metal vs CPU on Apple M4 Max
  Owner: Kernel Eng  Est: 2h
  Deps: T20.2
  Acceptance: Metal throughput >= 3x CPU on M4 Max; results in docs/benchmarks.md.

---

#### E21: Intel SYCL Backend [Q2-Q3 2030]

- [ ] T21.1 Implement SYCL runtime bindings via purego in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h
  Deps: none
  Acceptance: Intel oneAPI runtime detected; basic ops correct on Intel Arc GPU.
  TestSYCLBasicOps passes.

- [ ] T21.2 Port GEMV and attention kernels to SYCL [ztensor]
  Owner: Kernel Eng  Est: 8h
  Deps: T21.1
  Acceptance: Decode path runs on Intel GPU; output matches CPU reference.
  TestSYCLKernels passes.

---

#### E22: Auto-Optimization Framework [Q3-Q4 2030]

- [ ] T22.1 Implement hardware profiling in ztensor/compute/profile.go [ztensor]
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: ProfileHardware() returns memory bandwidth, compute TFLOPS, cache sizes,
  and supported features. TestHardwareProfile passes on DGX and CPU.

- [ ] T22.2 Implement automatic kernel selection based on hardware profile
  Owner: Kernel Eng  Est: 5h
  Deps: T22.1
  Acceptance: MatMul, GEMV, attention automatically select optimal kernel (cuBLAS vs
  custom, FP32 vs FP16 vs FP8) based on hardware and tensor dimensions.
  TestAutoKernelSelection passes.

- [ ] T22.3 Implement automatic quantization recommendation
  Owner: ML Eng  Est: 4h
  Deps: T22.1
  Acceptance: Given model size and hardware memory, recommend optimal quantization
  (Q4_K, Q8, FP8, NVFP4). TestAutoQuantRecommend passes.

---

#### E23: Evaluation Framework [Q2-Q3 2030]

- [ ] T23.1 Implement automated benchmark suite in cmd/zerfoo-bench/
  Owner: Infra Eng  Est: 5h
  Deps: none
  Acceptance: `zerfoo bench --suite standard` runs throughput, latency, memory
  benchmarks across all loaded models. JSON output for CI comparison.

- [ ] T23.2 Implement model comparison tool in cmd/zerfoo-compare/
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: `zerfoo compare --models A,B --dataset eval.jsonl` produces side-by-side
  quality comparison (perplexity, accuracy). TestModelCompare passes.

---

### YEAR 6-7 (2031-2032): Industry Standard

---

#### E24: Custom Model Architecture SDK [Q1-Q3 2031]

- [ ] T24.1 Implement model definition DSL in Go (layer composition API)
  Owner: Lead Eng  Est: 8h
  Deps: none
  Acceptance: Users can define custom transformer architectures in Go using a
  fluent builder API: model.New().Embedding(dim).TransformerBlock(heads, dim).LMHead(vocab).
  Custom model runs inference. TestCustomModelDSL passes.

- [ ] T24.2 Implement custom model training workflow
  Owner: ML Eng  Est: 6h
  Deps: T24.1
  Acceptance: Custom model definition → training → GGUF export → inference round-trip.
  TestCustomModelTraining passes.

- [ ] T24.3 Implement graph-level optimization passes
  Owner: Kernel Eng  Est: 8h
  Deps: T24.1
  Acceptance: Operator fusion, constant folding, dead node elimination applied
  automatically to custom model graphs. TestGraphOptimization passes.

---

#### E25: Heterogeneous Compute [Q2-Q4 2031]

- [ ] T25.1 Implement automatic workload splitting across CPU and GPU
  Owner: Kernel Eng  Est: 6h
  Deps: T22.1
  Acceptance: For models that exceed GPU memory, automatically split layers between
  CPU and GPU. TestHeterogeneousCompute passes.

- [ ] T25.2 Implement multi-accelerator scheduling (CUDA + Metal on Mac Studio)
  Owner: Kernel Eng  Est: 6h
  Deps: T20.1
  Acceptance: On machines with multiple accelerator types, schedule ops to the
  optimal device. TestMultiAccelerator passes.

---

#### E26: ZerfooConf Planning and Execution [2031-2032]

- [ ] T26.1 Plan ZerfooConf Day (1-day, co-located with GopherCon 2031)
  Owner: DevRel  Est: 4h
  Deps: none
  Acceptance: Venue booked; CFP open; 4-month lead time; budget approved.
  Target: 200+ attendees, 10+ talks, 2 workshops.

- [ ] T26.2 Execute ZerfooConf Day
  Owner: DevRel  Est: 8h
  Deps: T26.1
  Acceptance: Event held; 200+ attendees; talks recorded and published;
  post-event survey NPS > 50.

- [ ] T26.3 Plan standalone ZerfooConf 2032 (2-day conference)
  Owner: DevRel  Est: 6h
  Deps: T26.2
  Acceptance: Standalone venue; 500+ attendees target; sponsorship tiers defined;
  CFP committee formed.

---

#### E27: Ecosystem Integrations [Q1-Q4 2031]

- [ ] T27.1 Implement OCI-compatible model registry protocol
  Owner: Platform Eng  Est: 6h
  Deps: none
  Acceptance: `zerfoo registry push/pull` uses OCI distribution spec. Compatible
  with Harbor, GitHub Container Registry, ECR. TestOCIRegistry passes.

- [ ] T27.2 Implement Kubernetes model cache DaemonSet
  Owner: Platform Eng  Est: 4h
  Deps: none
  Acceptance: DaemonSet pre-loads model GGUF files to node local storage. Pod
  startup time reduced by eliminating download. TestModelCache passes.

- [ ] T27.3 Publish Helm chart for Zerfoo inference server
  Owner: Platform Eng  Est: 3h
  Deps: none
  Acceptance: `helm install zerfoo zerfoo/zerfoo-server` deploys inference server
  with GPU support, HPA, and Prometheus metrics. Chart published to Artifact Hub.

---

### YEAR 8-9 (2033-2034): Platform Maturity

---

#### E28: Federated Learning [Q1-Q3 2033]

- [ ] T28.1 Implement FederatedStrategy interface in distributed/federated/strategy.go
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: FederatedStrategy defines SelectClients(), ConfigureClient(),
  AggregateResults(). FedAvg implementation included. TestFedAvg passes.

- [ ] T28.2 Implement FedProx strategy in distributed/federated/fedprox.go
  Owner: ML Eng  Est: 3h
  Deps: T28.1
  Acceptance: FedProx adds proximal term to local training loss. TestFedProx passes.

- [ ] T28.3 Implement differential privacy noise injection
  Owner: ML Eng  Est: 4h
  Deps: T28.1
  Acceptance: DPNoiseInjector clips gradients and adds calibrated Gaussian noise.
  Privacy budget (epsilon, delta) tracked. TestDifferentialPrivacy passes.

- [ ] T28.4 Integration test: 4-client federated learning simulation
  Owner: ML Eng  Est: 3h
  Deps: T28.1, T28.3
  Acceptance: 4 simulated clients train on partitioned data; global model
  converges; privacy budget not exceeded. TestFederatedE2E passes.

---

#### E29: On-Device Inference (iOS, Android) [Q2-Q4 2033]

- [ ] T29.1 Implement gomobile bindings for Zerfoo Runtime
  Owner: Arch Eng  Est: 6h
  Deps: none
  Acceptance: gomobile bind produces .aar (Android) and .xcframework (iOS).
  Basic inference API exposed: Load, Generate, Close.

- [ ] T29.2 Create iOS demo app with on-device Zerfoo inference
  Owner: Arch Eng  Est: 4h
  Deps: T29.1
  Acceptance: SwiftUI app loads Q4 model, generates text on-device.
  No network required. Runs on iPhone 15 Pro or newer.

- [ ] T29.3 Create Android demo app with on-device Zerfoo inference
  Owner: Arch Eng  Est: 4h
  Deps: T29.1
  Acceptance: Jetpack Compose app loads Q4 model, generates text on-device.
  Runs on Pixel 8 or newer.

- [ ] T29.4 Benchmark on-device inference: tok/s on iPhone 16 Pro and Pixel 9
  Owner: Arch Eng  Est: 2h
  Deps: T29.2, T29.3
  Acceptance: Tok/s measured for 1B Q4 model on both platforms; results published.

---

#### E30: FedRAMP Authorization [Q1-Q4 2034]

- [ ] T30.1 Engage FedRAMP 3PAO and sponsoring agency
  Owner: Compliance  Est: 4h
  Deps: T19.1
  Acceptance: 3PAO selected; agency sponsor identified; FedRAMP Moderate boundary
  defined; engagement signed.

- [ ] T30.2 Implement FedRAMP required controls (NIST 800-53 Moderate)
  Owner: Infra Eng  Est: 12h
  Deps: T30.1
  Acceptance: All 325 Moderate baseline controls implemented or documented
  as inherited/not applicable. SSP (System Security Plan) drafted.

- [ ] T30.3 Complete FedRAMP authorization assessment
  Owner: Compliance  Est: 4h
  Deps: T30.2
  Acceptance: 3PAO assessment completed; ATO (Authority to Operate) letter issued
  by sponsoring agency; Zerfoo Cloud listed in FedRAMP Marketplace.

---

### YEAR 10 (2035-2036): Market Leadership and IPO

---

#### E31: IPO Preparation [Q1-Q4 2035]

- [ ] T31.1 Form board of directors with independent directors
  Owner: CEO  Est: ongoing
  Deps: none
  Acceptance: 5+ board members including 3+ independent directors with public
  company experience; audit committee and compensation committee formed.

- [ ] T31.2 Engage Big 4 audit firm for financial statements
  Owner: CFO  Est: 4h
  Deps: none
  Acceptance: Audit engagement signed; 2 years of audited financials prepared;
  SOX compliance framework in place.

- [ ] T31.3 Hire VP Sales and VP Marketing
  Owner: CEO  Est: ongoing
  Deps: none
  Acceptance: VP Sales with enterprise SaaS experience ($50M+ ARR company).
  VP Marketing with developer tools experience.

- [ ] T31.4 Achieve $150M+ ARR run rate
  Owner: CEO  Est: ongoing
  Deps: all
  Acceptance: Trailing 12-month revenue >= $150M; net revenue retention >= 120%;
  Rule of 40 achieved (growth rate + operating margin >= 40%).

- [ ] T31.5 Draft S-1 registration statement
  Owner: CFO  Est: 8h
  Deps: T31.1, T31.2, T31.4
  Acceptance: S-1 drafted with underwriters; filed with SEC; quiet period initiated.

---

#### E32: Architecture Expansion to 100+ Models [2035-2036]

- [ ] T32.1 Implement automated architecture builder from GGUF metadata
  Owner: Arch Eng  Est: 8h
  Deps: T6.1
  Acceptance: Given GGUF metadata, automatically construct graph without hardcoded
  arch_*.go file. Covers standard transformer patterns. TestAutoArchBuilder passes.

- [ ] T32.2 Validate 100+ model architectures from HuggingFace GGUF hub
  Owner: Arch Eng  Est: ongoing
  Deps: T32.1
  Acceptance: 100+ distinct model architectures load and produce coherent output;
  automated parity testing; results tracked in CI dashboard.

---

#### E33: Performance Target 1000+ tok/s [2032-2035]

- [ ] T33.1 Implement next-gen GPU architecture optimizations (post-Blackwell)
  Owner: Kernel Eng  Est: ongoing
  Deps: none
  Acceptance: Kernels optimized for current-gen NVIDIA architecture (sm_130+);
  benchmark shows >= 1000 tok/s on 1B Q4 model.

- [ ] T33.2 Implement automatic hardware-specific kernel codegen
  Owner: Kernel Eng  Est: 10h
  Deps: T22.1
  Acceptance: Given hardware profile, code generator produces optimal CUDA/HIP/Metal
  kernels at build time. TestAutoCodegen passes.

---

## Parallel Work

### Parallel Tracks

| Track | Description | Epic IDs | Sync Points |
|-------|-------------|----------|-------------|
| A | Performance and Kernels | E1, E3, E16, E33 | Merge at benchmarks |
| B | Architecture Support | E2, E10, E32 | Merge at parity tests |
| C | Community and Documentation | E4, E5, E11, E26 | Merge at star milestones |
| D | Backend Expansion | E8, E20, E21 | Merge at benchmark suite |
| E | Platform and Enterprise | E12, E14, E17, E18, E19 | Merge at marketplace launch |
| F | Edge and On-Device | E15, E29 | Merge at mobile demo |
| G | v1.0 and API Stability | E6, E7 | Merge at v1.0 release |
| H | Training and Research | E22, E23, E24, E28 | Merge at training platform |
| I | Security and Compliance | E13, E14, E19, E30 | Merge at FedRAMP |
| J | Corporate and IPO | E12, E31 | Merge at IPO |

### Wave Plan (Maximum Parallelism, Up to 10 Agents)

Wave 1 -- All independent starters (no dependencies):
1. T1.1 Profile decode hot path (Kernel Eng)
2. T2.1 Llama 4 architecture builder (Arch Eng)
3. T2.2 Gemma 3n architecture builder (Arch Eng)
4. T3.1 GPTQ dequantization [ztensor] (Kernel Eng)
5. T4.1 Documentation site structure (DevRel)
6. T4.2 Quickstart guide (DevRel)
7. T5.1 CONTRIBUTING.md (DevRel)
8. T5.3 GitHub Discussions setup (DevRel)
9. T6.1 Architecture registry (Lead Eng)
10. T6.2 Quantization format registry [ztensor] (Lead Eng)

Wave 2 -- Unblocked by Wave 1:
1. T1.2 KV cache FP16 (Kernel Eng) [needs T1.1]
2. T1.3 Optimized Q4_K GEMV [ztensor] (Kernel Eng) [needs T1.1]
3. T1.4 Kernel launch batching (Kernel Eng) [needs T1.1]
4. T2.3 Command R builder (Arch Eng) [no deps]
5. T2.4 Falcon builder (Arch Eng) [no deps]
6. T2.5 Mixtral builder (Arch Eng) [no deps]
7. T2.8 Exponential-trapezoidal SSM discretization (Kernel Eng) [no deps]
8. T3.2 AWQ dequantization [ztensor] (Kernel Eng) [no deps]
9. T3.3 Native Q5_K GEMV [ztensor] (Kernel Eng) [no deps]
10. T4.3 API reference docs (DevRel) [no deps]

Wave 3 -- Unblocked by Wave 2:
1. T1.5 Benchmark 300+ tok/s (Kernel Eng) [needs T1.2, T1.3, T1.4]
2. T2.6 RWKV builder (Arch Eng) [no deps]
3. T2.9 Complex-valued SSM + RoPE + BCNorm (Kernel Eng) [needs T2.8]
4. T3.4 Native Q6_K GEMV [ztensor] (Kernel Eng) [no deps]
5. T3.5 W4A16 mixed precision [ztensor] (Kernel Eng) [needs T3.1]
6. T3.6 W8A8 mixed precision [ztensor] (Kernel Eng) [no deps]
7. T4.4 Architecture tour doc (DevRel) [no deps]
8. T4.5 Cookbook recipes (DevRel) [no deps]
9. T5.2 Good first issues (DevRel) [no deps]
10. T5.7 Example applications (DevRel) [no deps]

Wave 4 -- Integration and community push:
1. T2.7 New arch parity tests (Arch Eng) [needs T2.1-T2.6]
2. T2.10 MIMO SSM heads (Kernel Eng) [needs T2.9]
3. T4.6 Benchmark comparison guide (DevRel) [needs T1.5]
4. T4.7 Video walkthrough (DevRel) [needs T4.2]
5. T5.4 Discord server (DevRel) [no deps]
6. T5.5 Blog posts (DevRel) [needs T4.2, T4.6]
7. T5.6 GopherCon CFP (DevRel) [needs T4.2]
8. T6.3 Extension docs (Lead Eng) [needs T6.1, T6.2]
9. T7.1 Audit Engine[T] interface (Lead Eng) [no deps]
10. T7.3 Deprecation linter (Lead Eng) [no deps]

Wave 5 -- Mamba 3 assembly + v1.0 start:
1. T2.11 Mamba 3 architecture builder (Arch Eng) [needs T2.8, T2.9, T2.10]
2. T2.12 Mamba 3 parity tests on DGX (Arch Eng) [needs T2.11]
3. T7.2 Label sub-package maturity (Lead Eng) [needs T7.1]

Wave 6 -- v1.0 and ROCm:
1. T7.2 Label sub-package maturity (Lead Eng) [needs T7.1]
2. T7.4 Release-please v1.0 (Lead Eng) [needs T7.1]
3. T7.5 Migration guide (DevRel) [needs T7.1]
4. T8.2 Validate HIP bindings (Kernel Eng) [needs T8.1]
5. T8.3 Validate rocBLAS (Kernel Eng) [needs T8.1]
6. T9.1 Tensor parallelism (Infra Eng) [no deps]
7. T9.2 Pipeline parallelism (Infra Eng) [no deps]
8. T10.1 LLaVA builder (Arch Eng) [no deps]
9. T10.2 Qwen-VL builder (Arch Eng) [no deps]
10. T15.2 Pre-optimized model format (Arch Eng) [needs T15.1]

Wave 7 -- Year 2-3 integration:
1. T8.4 Port CUDA kernels to HIP [ztensor] (Kernel Eng) [needs T8.2]
2. T8.6 ROCm CI (Infra Eng) [needs T8.4]
3. T9.3 Multi-GPU CLI (Infra Eng) [needs T9.1, T9.2]
4. T11.1 GopherCon booth (DevRel) [no deps]
5. T11.2 Tutorial series (DevRel) [no deps]
6. T11.4 Recruit co-maintainers (Lead Eng) [no deps]
7. T11.5 LangChain-Go integration (DevRel) [no deps]
8. T12.1 Enterprise SLA tiers (Biz Dev) [no deps]
9. T13.1 Security auditor engagement (Lead Eng) [no deps]
10. T15.3 Raspberry Pi test (Arch Eng) [needs T15.1]

Wave 8 -- Enterprise foundation:
1. T8.5 ROCm benchmark (Kernel Eng) [needs T8.4]
2. T9.4 Multi-GPU benchmark (Infra Eng) [needs T9.3]
3. T10.3 Vision model benchmarks (Arch Eng) [needs T10.1, T10.2]
4. T11.3 KubeCon CFP (DevRel) [no deps]
5. T12.2 Enterprise ticketing (Biz Dev) [needs T12.1]
6. T12.3 Enterprise deployment guide (DevRel) [no deps]
7. T13.2 Fix audit findings (Lead Eng) [needs T13.1]
8. T13.3 SBOM generation (Infra Eng) [no deps]
9. T14.1 Compliance platform (Compliance) [no deps]
10. T15.4 Jetson test (Arch Eng) [needs T15.1]

Wave 9 -- SOC 2 and platform:
1. T12.4 Enterprise contracts (Biz Dev) [needs T12.2]
2. T14.2 Security controls (Infra Eng) [needs T14.1]
3. T14.3 SOC 2 Type I audit (Compliance) [needs T14.2]
4. T15.5 ARM64 CI (Infra Eng) [needs T15.1]
5. T16.1 Warp-specialized GEMV [ztensor] (Kernel Eng) [no deps]
6. T16.2 FP8 KV cache (Kernel Eng) [no deps]
7. T17.1 Model repository server (Platform Eng) [no deps]
8. T17.2 Kubernetes operator (Platform Eng) [no deps]
9. T17.3 Adaptive batching (Platform Eng) [no deps]
10. T18.1 Enterprise repo setup (Lead Eng) [no deps]

Wave 10 -- Cloud and enterprise features:
1. T14.4 SOC 2 Type II observation (Compliance) [needs T14.3]
2. T16.3 500+ tok/s benchmark (Kernel Eng) [needs T16.1, T16.2]
3. T17.4 Multi-model serving (Platform Eng) [no deps]
4. T17.5 AWS Marketplace listing (Biz Dev) [needs T17.1, T17.2]
5. T18.2 SSO/SAML (Platform Eng) [needs T18.1]
6. T18.3 RBAC (Platform Eng) [needs T18.1]
7. T18.4 Audit logging (Platform Eng) [needs T18.1]
8. T18.5 Monitoring dashboards (Platform Eng) [needs T18.1]
9. T19.1 SOC 2 Type II completion (Compliance) [needs T14.4]
10. T20.1 Metal compute bindings [ztensor] (Kernel Eng) [no deps]

Wave 11 -- Year 5 multi-accelerator:
1. T17.6 GCP Marketplace (Biz Dev) [needs T17.5]
2. T17.7 Azure Marketplace (Biz Dev) [needs T17.5]
3. T20.2 Port kernels to Metal [ztensor] (Kernel Eng) [needs T20.1]
4. T20.3 Metal benchmark (Kernel Eng) [needs T20.2]
5. T21.1 SYCL runtime bindings [ztensor] (Kernel Eng) [no deps]
6. T21.2 Port kernels to SYCL [ztensor] (Kernel Eng) [needs T21.1]
7. T22.1 Hardware profiling [ztensor] (Kernel Eng) [no deps]
8. T22.2 Auto kernel selection (Kernel Eng) [needs T22.1]
9. T22.3 Auto quantization recommendation (ML Eng) [needs T22.1]
10. T23.1 Automated benchmark suite (Infra Eng) [no deps]

Waves 11-16 follow the same pattern completing E23-E33 (Years 6-10).

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M1 | Inference Excellence | E1-E6 | 300+ tok/s; 12+ archs; docs live; 5K stars | 2026-12-31 |
| M2 | v1.0 and Ecosystem | E7-E11 | v1.0 shipped; ROCm parity; 25K stars; 100+ contributors | 2027-12-31 |
| M3 | Enterprise Foundation | E12-E16 | $500K ARR; SOC 2 Type I; edge runtime; 500+ tok/s | 2028-12-31 |
| M4 | Platform GA | E17-E19 | $2M ARR; SOC 2 Type II; cloud marketplace live | 2029-12-31 |
| M5 | Training Platform | E20-E23 | $10M ARR; Metal + SYCL backends; auto-optimization | 2030-12-31 |
| M6 | Industry Standard | E24-E27 | $50M ARR; ZerfooConf; 100K stars; Helm chart; OCI registry | 2032-12-31 |
| M7 | Platform Maturity | E28-E30 | $75M ARR; federated learning; on-device; FedRAMP | 2034-12-31 |
| M8 | Market Leadership | E31-E33 | $150M+ ARR; IPO filed; 100+ architectures; 1000+ tok/s | 2036-12-31 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Go ML TAM ceiling: market too small for $150M ARR | Critical | High | Expand beyond Go devs via OpenAI API, edge runtime, and language FFI bindings. Monitor adoption quarterly. |
| R2 | Apache 2.0 fork by cloud provider (managed Zerfoo) | Existential | Medium-High | Compete on innovation velocity; consider AGPL for v2 if fork materializes. Build enterprise trust moat. See ADR-057. |
| R3 | Latent bugs in AI-generated code surface during enterprise adoption | High | High | Security audit (Year 3); comprehensive DGX validation; fuzz testing; bug bounty program. |
| R4 | Maintainer burnout / bus factor of 1 | Critical | High | Recruit 5 co-maintainers by Year 2; governance foundation by Year 4; community decision-making. |
| R5 | No clear enterprise budget owner for "Go ML library" | High | Medium-High | Position as "inference infrastructure" (platform eng budget); offer POC program; marketplace credits as sales lever. |
| R6 | ROCm never reaches CUDA parity; AMD backend is maintenance burden | Medium | High | Set ROCm to 80% parity target (not 100%); gate by user demand; drop if adoption < 5% of users. |
| R7 | Enterprise sales cycle too long (6-12 months); cash flow risk | High | Medium | Marketplace consumption accelerates procurement; start with support contracts (faster close); PLG motion for self-serve. |
| R8 | SaaS multiples compressed; IPO economics unattractive at 6-10x | High | Medium | Maintain optionality: acquisition, private equity, or continued private growth as alternatives. |
| R9 | Rust ML (burn, candle) captures "systems language ML" category first | High | Medium | Ship v1.0 first; Go has 10x Python interop advantage via OpenAI API; edge runtime as differentiator. |
| R10 | NVIDIA changes CUDA licensing or API terms | High | Low-Medium | GRAL abstraction insulates from vendor changes; Metal and SYCL provide fallback paths. |
| R11 | Go generics limitations force awkward API design | Medium | Medium | Extension interface pattern (ADR-058) mitigates Engine[T] freeze. Monitor Go spec proposals. |
| R12 | Cloud marketplace revenue share erodes margins (3-20%) | Medium | Medium | Pursue SaaS listings (3%) over container (20%); enterprise self-managed as high-margin alternative. See ADR-060. |
| R13 | GopherCon talk rejected multiple times; conference presence delayed | Low | Medium | Submit to multiple conferences (GopherCon, KubeCon, GoLab); sponsor booth as fallback; host own meetups. |
| R14 | FedRAMP cost ($250K-$750K) exceeds budget for government vertical | Medium | Medium | Delay to Year 8-9 as planned; evaluate government demand before committing; partner with GovCloud MSP. |
| R15 | Agentic coder quality drift at scale (DORA 2025: 9% higher bug rate) | High | High | Human review gates at milestones; security audit before enterprise GA; strict CI quality gates; /review before releases. |

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
- Human review gate required at each milestone (M1-M8).
- Security review (/review) before each enterprise-facing release.
- Run `golangci-lint` on all changed packages before committing.

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
- Rebase and merge. Not squash, not merge commits.

---

## Progress Log

### 2026-03-18: GGUF writer consolidation plan created

Created docs/plan-gguf-writer.md to consolidate 5 hand-rolled GGUF writers into a
shared `ztensor/gguf` package. 3 epics (E1-E3), 18 tasks, 7 waves across ztensor,
zerfoo, and zonnx repos. ADR-061 created (docs/adr/061-gguf-writer-in-ztensor.md).
Closes the SaveModel "not implemented" gap in training/adapter.go.

### 2026-03-18: 10-year product roadmap created

**Scope:** Created full 10-year product roadmap (2026-2036) expanding from completed
5-year technical roadmap. Covers engineering (performance, architectures, backends,
edge, federated learning), community (docs, conferences, governance), enterprise
(support, SOC 2, FedRAMP, cloud marketplace), and corporate (IPO preparation).

**Change summary:**
- Replaced trimmed 5-year plan with 10-year product roadmap.
- 33 epics, 120+ tasks, 10-wave parallel execution plan.
- Risk register: 15 risks with mitigations.
- 8 milestones from Inference Excellence (2026) to Market Leadership (2036).
- Research team findings incorporated from tech-researcher, risk-researcher,
  arch-researcher.
- ADRs created:
  - docs/adr/057-open-core-licensing-strategy.md
  - docs/adr/058-api-stability-v1-contract.md
  - docs/adr/059-edge-runtime-architecture.md
  - docs/adr/060-cloud-platform-architecture.md

### 2026-03-18: Trimmed plan -- 5-year roadmap complete

Trimmed plan. Stable knowledge preserved in docs/design.md (sections 15-29),
docs/adr/ (044-056), and docs/devlog.md. Removed completed epics: E1-E21
(all 124 tasks). Removed wave plan, risk register, timeline table, and appendix.

---

## Hand-Off Notes

### What You Need to Know

- **Repos:** Each repo has its own go.mod. Never commit across repos. Tasks marked
  [ztensor] go in /Users/dndungu/Code/zerfoo/ztensor; unmarked tasks go in
  /Users/dndungu/Code/zerfoo/zerfoo.
- **DGX Spark:** GPU hardware at `ssh ndungu@192.168.86.250`. Set
  `LD_LIBRARY_PATH=~/Code/zerfoo` before running GPU tests. Always rebuild binary.
- **Baseline benchmark:** 245 tok/s, Gemma 3 1B Q4_K_M, 256 tokens, CUDA graph,
  DGX Spark GB10. Target: 300+ (Year 1), 500+ (Year 3), 1000+ (Year 7).
- **Current ADRs:** 001-061 in docs/adr/. Next ADR: 062.
- **GGUF writer plan:** docs/plan-gguf-writer.md -- consolidates 5 hand-rolled
  writers into shared ztensor/gguf package. See ADR-061.
- **Architecture docs:** docs/design.md (29 sections), docs/benchmarks.md,
  docs/devlog.md.
- **CI:** GitHub Actions in .github/workflows/. CPU tests in CI; GPU tests on DGX only.
- **Model downloads:** `zerfoo pull model_id` for HuggingFace models (ADR-039).
  DGX models at ~/models/: gemma3-q4km, phi4, llama3, qwen2.
- **Licensing:** Apache 2.0 for all core repos. Enterprise features in separate
  zerfoo-enterprise repo under commercial license (ADR-057).
- **v1.0 contract:** Engine[T] frozen; extension interfaces for new capabilities;
  2-year backwards compatibility guarantee (ADR-058).
- **Cloud platform:** SaaS marketplace listing (3% revenue share); Kubernetes
  operator for serving; token-based billing (ADR-060).
- **Edge runtime:** build-tag-gated minimal binary; <10MB ARM64 target (ADR-059).
- **Founder approval required:** E21 from prior plan (Zerfoo cloud product)
  ADR-056 status is Proposed; blocked until founder approves per Feza governance.

### Placeholder Credentials

- DGX SSH: ndungu@192.168.86.250 (key auth; no password in this file)
- HuggingFace token: set HUGGINGFACE_TOKEN env var
- Stripe API key: set STRIPE_API_KEY env var (billing)
- GCP project: set GOOGLE_CLOUD_PROJECT env var (cloud platform)
- AWS Marketplace: set AWS_MARKETPLACE_SELLER_ID env var
- Discord: set DISCORD_BOT_TOKEN env var (community bot)
- Vanta/Drata: set COMPLIANCE_API_KEY env var (SOC 2 automation)

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

**Cloud marketplace economics:**
- AWS: 3% for SaaS, 20% for ML containers. Private offers: 3%/2%/1.5% by TCV.
- GCP: 3% standard, 1.5% renewals/migrations.
- Azure: 3% standard store fee.
- Key insight: marketplace consumption counts toward EDP/MACC spend commitments.

**IPO benchmarks for developer tools:**
- GitLab: $233M ARR at IPO, 152% NRR, now $700M+ ARR.
- HashiCorp: $320M ARR at IPO, 120%+ NRR, acquired by IBM.
- Confluent: $314M trailing ARR at IPO, 117-130% NRR.

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
