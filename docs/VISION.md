# Zerfoo — Ten-Year Product Vision (2026–2036)

*Last updated: 2026-03-17. Throughput targets and architecture counts amended
2026-08-20 per ADR-093 (GB10 roofline ~257 tok/s) — see the amendment note
directly below. The thesis, the ten-year arc, and the revenue model are
unchanged.*

## Amendment — 2026-08-20 (per ADR-093)

**Why this note exists.** ADR-093 (*H2 2026 Product Strategy — Trust, then
Traction*, accepted 2026-07-02) rule 1 makes verification the gate for every
public claim: **no public claim may exceed `docs/verified-models.md`.** This
document was last written on 2026-03-17 and carried absolute throughput
targets and a static architecture count that the ruling invalidates. The
targets below are amended rather than deleted, so the original ambition and
the reason it changed both stay legible.

**The original targets, preserved:** Year 1 "300+ tok/s on Gemma 3 1B
Q4_K_M (target: 30% above Ollama)"; Year 2 "400+ tok/s on Gemma 3 1B
Q4_K_M"; Year 3 "500+ tok/s on small models"; and a Key Metrics row reading
300 / 500 / 750 / 1,000 / 1,500+ decode tok/s for 2026 / 2028 / 2030 / 2032 /
2036. Those numbers were written as absolutes, independent of hardware.

**Why they changed.** Decode of a 1B Q4_K_M model is memory-bandwidth bound.
Gemma 3 1B Q4_K_M is ~778 MB of weights; the GB10's LPDDR5x delivers
~200 GB/s, which puts the roofline at **~257 tok/s** on that hardware. The
measured 241 tok/s is 94% of the ceiling — near-optimal, not a shortfall.
500 tok/s would require ~390 GB/s, roughly 2x what the GB10 has, so it is
not a target that engineering effort can reach on this device
(`docs/devlog.md`, 2026-03-19 "T16.3 benchmark 500+ tok/s — physically
impossible on GB10"; issue #711 closed won't-fix; `docs/product-strategy-2026-H2.md`
Part 3, explicit non-goals: "No throughput moonshots"). Absolute tok/s
targets above the roofline are therefore **banned as public claims**. Where a
future year's target exceeds the GB10 roofline it is now stated as
hardware-conditional (the bandwidth the target implies) and paired with a
roofline-utilization target, which is the part Zerfoo's engineering actually
controls.

**Architecture counts.** Static counts in this document are replaced with a
pointer to **`docs/verified-models.md`, the single source of truth** for what
Zerfoo can claim to support. As of 2026-08-20 that matrix carries 10 candidate
rows, of which 5 are `verified` (Gemma 3 1B, Llama 3.2 3B, Mistral 7B,
DeepSeek-R1-Distill 1.5B on the `qwen2` builder, MiniMax-M2 229B CPU/over-RAM)
— a count of *registered architecture builders* is a much larger number and is
not a support claim. Forward-year architecture targets below are counted in
`verified` matrix rows, not builders.

**What ADR-093 does NOT change.** It explicitly upholds this document's
ten-year thesis and its revenue/IPO arc; VISION.md is listed there as the
"unchanged 10-year thesis". ADR-093 supersedes only (a) the priority ordering
for H2 2026 — trust and adoption before capability expansion — and (b) the
absolute throughput targets amended here.

## The Thesis

There is no production-grade ML inference framework written in Go. Python dominates ML, C++ dominates inference runtimes, and Go developers who want to embed model inference into their applications are forced to shell out to Python, bind to C++ via CGo, or use HTTP APIs to external services.

**Zerfoo fills this gap.** It is a Go-native ML framework that makes model inference a library call — as natural as `json.Unmarshal` or `sql.Open`. Import the package, load a model, generate text. No Python runtime, no CGo build complexity, no sidecar processes.

By 2036, Zerfoo will be the standard ML runtime for the Go ecosystem — the way PyTorch defines ML in Python, Zerfoo will define ML in Go. Every Go service that needs intelligence will import Zerfoo.

## Where We Are (March 2026)

Zerfoo has grown from a proof-of-concept into a substantial framework:

- **7 repositories (6 active, 1 archived)**, ~50,000+ lines of Go, ~5,000 lines of CUDA C, ~2,000 lines of ARM/x86 assembly
- **Model support as evidenced by `docs/verified-models.md`** — the single
  source of truth for support claims (Llama, Gemma 3, Mistral, Qwen 2, Phi,
  DeepSeek and more have builders; the matrix says which are *verified* today,
  with parity and benchmark evidence per row). Quote the matrix, not a count.
- **241 tok/s** on Gemma 3 1B Q4_K_M — 1.28x Ollama's 188 on the same GB10
  hardware, and 94% of that hardware's ~257 tok/s memory-bandwidth roofline
  (`docs/verified-models.md`, `docs/benchmarks.md`, 2026-03-31)
- **25+ custom CUDA kernels** with zero-CGo purego bindings
- **OpenAI-compatible API server** with streaming, batching, speculative decoding
- **Distributed training** via gRPC and NCCL
- **ARM NEON SIMD assembly** for critical CPU paths (GEMM, RMSNorm, RoPE, SiLU, softmax)
- **One-line API**: `model, _ := zerfoo.Load("google/gemma-3-4b")`
- **Embeddings, structured output, tool calling** all shipped

This is real. The framework runs real models at competitive speeds on real hardware.

## Ten-Year Roadmap

### Year 1 (2026): Inference Excellence

**Goal: Be the fastest Go-native inference runtime. Establish the community.**

Technical:
- Grow to 12+ `verified` rows in `docs/verified-models.md` (add Llama 4, Gemma 3n, Phi-4, Command R, Falcon, Mixtral) — a registered builder is not a support claim; a verified matrix row is
- Continuous batching (PagedAttention-style) for production serving
- Prefill/decode split — tensor parallelism for prefill, CUDA graph replay for decode
- Quantization expansion: GPTQ, AWQ, native Q5_K/Q6_K GEMV, W4A16 and W8A8 mixed precision
- Decode within 10% of the reference hardware's memory-bandwidth roofline on Gemma 3 1B Q4_K_M, and measurably ahead of Ollama on that same hardware (GB10: roofline ~257 tok/s; 241 tok/s = 94%, 1.28x Ollama, already met as of 2026-03-31). *Amended 2026-08-20 per ADR-093 — was "300+ tok/s (target: 30% above Ollama)"; 300 tok/s exceeds the GB10 roofline and is unreachable on the reference device at any level of effort. Absolute figures above ~257 tok/s require higher-bandwidth hardware (A100 ~2 TB/s, H100 ~3.35 TB/s), so they are a hardware statement, not a roadmap target.*

Community:
- 5,000+ GitHub stars across all repos
- Comprehensive documentation: getting started, API reference, architecture tour
- 10+ example applications
- GopherCon talk submission
- Active GitHub Discussions and contributor pipeline

Revenue: **$0**. Community adoption is the investment.

### Year 2 (2027): Ecosystem and v1.0

**Goal: Ship v1.0 with stable APIs. Become the default recommendation when Go developers ask "how do I run ML models?"**

Technical:
- **v1.0 stable release** — backwards-compatible public API guaranteed for 2 years
- LoRA and QLoRA fine-tuning: load a base model, train adapters, hot-swap at inference
- Gradient checkpointing for fine-tuning larger models on consumer GPUs
- Mixed precision training (FP16/BF16 forward, FP32 master weights)
- ROCm (AMD GPU) backend at CUDA feature parity
- OpenCL backend for cross-vendor GPU support
- Multi-GPU inference (tensor parallelism, pipeline parallelism)
- Vision-language models: LLaVA, Gemma 3 + SigLIP, Qwen-VL
- Hold ≥90% roofline utilization on Gemma 3 1B Q4_K_M across every supported reference device, and publish the per-device ceiling alongside every figure. *Amended 2026-08-20 per ADR-093 — was "400+ tok/s on Gemma 3 1B Q4_K_M"; 400 tok/s implies ≥311 GB/s of memory bandwidth and so is reachable only on hardware Zerfoo does not currently target.*

Community:
- 25,000+ GitHub stars
- 100+ contributors
- GopherCon presence (talk accepted)
- First production deployments by external teams
- Comprehensive tutorial series: "from hello world to production"
- Community channels (Discord/GitHub Discussions) active

Revenue: **$0**. Building towards enterprise readiness.

### Year 3 (2028): Enterprise Foundation

**Goal: First paying customers. Prove Zerfoo is production-ready at enterprise scale.**

Technical:
- Distributed training at scale: multi-node LoRA, RLHF, DPO
- Model hub integration: `zerfoo pull`, `zerfoo push` with HuggingFace and private registries
- Advanced serving: speculative decoding, KV cache quantization, request scheduling
- Edge deployment: ARM-optimized builds for Raspberry Pi, Jetson, mobile
- 500+ tok/s on small models **on hardware whose bandwidth admits it** (500 tok/s on a 778 MB 1B Q4_K_M model requires ~390 GB/s — A100/H100-class, not GB10-class), with ≥90% roofline utilization wherever Zerfoo runs; competitive on 7B–70B models. *Amended 2026-08-20 per ADR-093 — was an unconditional "500+ tok/s on small models". That figure is physically impossible on the GB10 (roofline ~257 tok/s) and is banned as a public claim; it survives here only as a hardware-conditional statement.*

Enterprise:
- Launch enterprise support tier (SLAs, priority bug fixes, dedicated Slack)
- **$500K ARR** from support contracts
- 50+ production deployments
- Security audit and SOC 2 preparation
- Enterprise documentation: deployment guides, compliance, migration paths

### Year 4 (2029): Platform Expansion

**Goal: Transition from library to platform. Multiple revenue streams.**

Technical:
- Zerfoo Cloud: managed inference on AWS/GCP/Azure marketplaces
- Enterprise features: audit logging, SSO/SAML, multi-tenancy, RBAC
- Model registry: versioned model storage, A/B testing, canary deployment
- Advanced quantization: automatic mixed-precision selection, calibration tools
- Benchmark suite: reproducible, CI-integrated performance tracking across models

Enterprise:
- **$2M ARR** (support + cloud marketplace)
- Cloud marketplace revenue sharing with AWS/GCP/Azure
- 200+ production deployments
- SOC 2 Type II certified

Community:
- 50,000+ GitHub stars
- 250+ contributors
- Conference keynotes (GopherCon, KubeCon)

### Year 5 (2030): Training Platform

**Goal: Full-cycle ML in Go. Train, fine-tune, evaluate, deploy — all in Zerfoo.**

Technical:
- Full training platform: pre-training for small models, LoRA/QLoRA/RLHF/DPO for all sizes
- Online learning: update model weights from streaming data without full retraining
- Evaluation framework: automated benchmark suites, A/B testing, model comparison
- Auto-optimization: automatic kernel selection, graph optimization, hardware-specific tuning
- Multi-accelerator: NVIDIA (CUDA), AMD (ROCm), Intel (SYCL), Apple (Metal)

Enterprise:
- **$10M ARR**
- Training-as-a-service on Zerfoo Cloud
- Hardware co-optimization partnerships with NVIDIA and AMD
- 500+ production deployments
- Fortune 500 customers

### Year 6–7 (2031–2032): Industry Standard

**Goal: Zerfoo is the PyTorch of Go. Enterprise dominance.**

Technical:
- Custom model architectures definable in Go (not just loading pre-trained)
- Compiler-level optimizations: graph-level fusion, operator scheduling, memory planning
- Heterogeneous compute: split workloads across CPU, GPU, and accelerator automatically
- Zerfoo Runtime: lightweight inference-only binary for edge and embedded deployment
- 50+ `verified` rows in `docs/verified-models.md`

Enterprise:
- **$25–50M ARR**
- Zerfoo Cloud available in all major cloud regions
- Hardware vendor partnerships generating co-marketing and referral revenue
- Enterprise consulting practice: custom model integration, performance tuning

Ecosystem:
- 100,000+ GitHub stars
- 500+ contributors, self-sustaining community
- LangChain-Go, Weaviate, and other ecosystem integrations mature
- Third-party companies building products on Zerfoo
- Annual ZerfooConf developer conference

### Year 8–9 (2033–2034): Platform Maturity

**Goal: Dominant market position. IPO-ready metrics.**

Technical:
- Zerfoo v3.0: optimized for next-generation GPU architectures
- On-device inference: iOS, Android, embedded systems via Zerfoo Runtime
- Federated learning: train across distributed nodes without centralizing data
- Model compression: automated pruning, distillation, quantization pipelines

Enterprise:
- **$75–100M ARR**
- 1,000+ production deployments across Fortune 500
- Government and defense contracts (FedRAMP certification)
- IPO preparation: audited financials, board formation, S-1 drafting

### Year 10 (2035–2036): Market Leadership

**Goal: Zerfoo is to Go what PyTorch is to Python. $100M+ ARR. IPO or strategic realization.**

Technical:
- All major model architectures verified — 100+ `verified` rows in `docs/verified-models.md`
- Automatic hardware optimization across all accelerator types
- Zerfoo Runtime deployed on billions of edge devices
- Research partnerships with universities and AI labs

Enterprise:
- **$150M+ ARR**
- Market leader in Go ML, expanding into Rust and other systems languages
- IPO at 10–15x ARR = **$1.5–2.25B standalone valuation**

## Key Metrics Trajectory

| Metric | 2026 | 2028 | 2030 | 2032 | 2036 |
|--------|------|------|------|------|------|
| GitHub stars | 5K | 25K | 50K | 100K | 200K+ |
| Contributors | 10 | 100 | 250 | 500 | 1,000+ |
| Production deployments | 0 | 50 | 500 | 1,000 | 5,000+ |
| Verified architectures (matrix rows) ‡ | 12 | 20 | 30 | 50 | 100+ |
| Decode, 1B Q4_K_M — roofline utilization † | ≥90% | ≥90% | ≥90% | ≥90% | ≥90% |
| Decode, 1B Q4_K_M — absolute tok/s, hardware-conditional † | ≥90% of device roofline (GB10 ≈257) | 500 @ ≥390 GB/s | 750 @ ≥583 GB/s | 1,000 @ ≥778 GB/s | 1,500 @ ≥1.17 TB/s |
| Revenue (ARR) | $0 | $500K | $10M | $50M | $150M+ |
| Enterprise customers | 0 | 5 | 50 | 200 | 500+ |

† *Amended 2026-08-20 per ADR-093.* The original row read
`Decode tok/s (1B Q4_K_M) | 300 | 500 | 750 | 1,000 | 1,500+` as absolutes.
Decode of a 778 MB 1B Q4_K_M model is bandwidth-bound at roughly 0.78 GB per
token, so an absolute tok/s figure is a statement about the device, not about
Zerfoo: 500 tok/s needs ~390 GB/s, 750 needs ~583 GB/s, 1,000 needs ~778 GB/s,
1,500 needs ~1.17 TB/s. The GB10 reference device has ~200 GB/s (roofline
~257 tok/s), so every figure from 300 up is unreachable there. The ambition is
kept, restated as the bandwidth class it requires; the row Zerfoo is actually
accountable for is roofline utilization. Public claims may not exceed
`docs/verified-models.md` either way.

‡ *Amended 2026-08-20 per ADR-093.* Counted as `verified` rows in
`docs/verified-models.md` (5 of 10 candidate rows as of 2026-08-20), not as
registered architecture builders — a builder that exists is not a support
claim.

## Design Principles (Ranked)

1. **Inference throughput** — throughput is the north star metric, measured as
   utilization of the device's memory-bandwidth roofline rather than as an
   absolute tok/s number (*amended 2026-08-20 per ADR-093*: absolutes above the
   roofline are hardware statements, not engineering targets). Every design
   decision is evaluated against it. Note that ADR-093 supersedes this ranking
   for H2 2026 specifically — trust (verified claims) before traction before
   capability expansion; the ranking here resumes after that phase.
2. **Embeddability** — Zerfoo must work as a Go library import, not just a CLI or server.
3. **Zero-CGo default** — `go build` must work without a C compiler. GPU support is runtime-detected.
4. **Type safety** — Go generics for compile-time correctness. No `interface{}` in hot paths.
5. **Production readiness** — Metrics, logging, health checks, graceful shutdown, TLS. Not afterthoughts.
6. **Simplicity** — Fewer abstractions are better. Don't over-engineer for hypothetical futures.

## Competitive Landscape

| Framework | Language | Strengths | Zerfoo's Advantage |
|-----------|----------|-----------|-------------------|
| llama.cpp | C++ | Raw performance, huge community | Go embeddability, no CGo, cleaner API |
| Ollama | Go (wraps llama.cpp) | Easy CLI, Docker | Native Go (no C++ subprocess), library-first |
| vLLM | Python | Continuous batching, PagedAttention | No Python runtime, embeddable in Go services |
| TensorRT-LLM | C++/Python | NVIDIA-optimized | Vendor-neutral (CUDA + ROCm + OpenCL), simpler |
| ONNX Runtime | C++ | Broad model support | Go-native, no CGo, better DX for Go developers |

By Year 5, Zerfoo's competitive position shifts from "alternative to Ollama" to "the only production ML runtime for Go." The competitive set becomes irrelevant as Zerfoo defines its own category.

## Non-Goals

- **Pre-training at scale** — Zerfoo is not for training GPT-5. Focus is inference + fine-tuning.
- **Python API** — Go-first. If Python users want Zerfoo, they can use the OpenAI-compatible API.
- **Custom hardware backends** — Support NVIDIA, AMD, Intel, and Apple. Don't chase niche ASICs.
- **ONNX runtime replacement** — zonnx converts ONNX to GGUF at build time. Runtime ONNX execution is not a goal.

## Revenue Model

**Years 1–2: $0.** Open source (Apache 2.0), community adoption investment.

**Years 3–5: $500K–$10M ARR.** Enterprise support, cloud marketplace, consulting.

**Years 6–10: $10M–$150M ARR.** Platform revenue, hardware partnerships, enterprise features.

| Stream | Model | Timeline |
|--------|-------|----------|
| Enterprise support | Annual contracts, SLAs, priority bug fixes | Year 3+ |
| Consulting | Custom model integration, performance tuning | Year 3+ |
| Cloud marketplace | Pay-per-use managed inference (AWS/GCP/Azure) | Year 4+ |
| Enterprise features | Proprietary add-ons (audit, SSO, multi-tenancy, RBAC) | Year 4+ |
| Hardware partnerships | Co-optimization with GPU/accelerator vendors | Year 5+ |
| Training platform | Managed fine-tuning and training infrastructure | Year 5+ |

Licensing remains Apache 2.0 for the core framework. Enterprise features may be offered under a commercial license (open-core model) from Year 4.

## Target Market

### Primary: Go developers who need ML inference
- Backend engineers adding AI features to Go services
- Platform teams replacing Python ML microservices with Go
- Infrastructure teams needing embeddable inference without Python/C++ dependencies

### Secondary: ML engineers seeking alternatives to Python
- Teams frustrated with Python deployment complexity
- Organizations standardizing on Go for production services

### Tertiary: Organizations evaluating Ollama/llama.cpp
- Teams that need library-level integration, not just a server
- Performance-sensitive workloads where the measured per-model throughput advantage matters (*amended 2026-08-20 per ADR-093*: the advantage is model-dependent and must be quoted from `docs/verified-models.md` — 1.28x Ollama on Gemma 3 1B, parity on Llama 3.2 3B and Mistral 7B as of the 2026-03 runs — never as a blanket "20%+")

### Expanding TAM (Year 5+)
- Edge/embedded developers needing on-device ML (Zerfoo Runtime)
- Enterprise ML platform teams (Zerfoo Cloud replaces custom infrastructure)
- Rust/Zig developers (language expansion via FFI or native ports)
