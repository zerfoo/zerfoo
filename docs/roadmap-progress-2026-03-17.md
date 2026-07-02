# Zerfoo Five-Year Technical Roadmap — Progress Report

**Date:** 2026-03-17
**Reported by:** Zerfoo CEO → Chairman of the Board, Feza, Inc.

---

## Roadmap Overview

The five-year plan has 6 phases mapped to revenue milestones. We are **4 months into Year 1**.

| Year | Phase | Revenue Target |
|------|-------|----------------|
| Year 1 | Community adoption | $0 |
| Year 2 | Ecosystem (v1.0 stable) | $0 |
| Year 3 | Enterprise support | $500K ARR |
| Year 4 | Cloud marketplace | $2M ARR |
| Year 5 | Platform | $10M ARR |

---

## Phase 1: Inference Excellence — ON TRACK (85%)

**Timeline:** NOW (Year 1)

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| 6 transformer architectures | Done | Llama 3, Gemma 3, Mistral, Qwen 2, Phi, DeepSeek V3 — all production-ready |
| Beat Ollama throughput | Done | 241 tok/s vs Ollama 197 tok/s on Gemma 3 1B Q4_K_M (+18%, v1.38.4) |
| CUDA graph capture | Done | 99.5% instruction coverage on GGUF inference path |
| Quantized GEMM/GEMV | Done | Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, FP8, FP16, BF16, NF4 — 10+ formats |
| GGUF loading all architectures | 90% | Working but Q5_K/Q6_K re-quantization bug needs fix |
| Fused transformer ops | Done | Flash attention, RMSNorm+MatMul, SwiGLU, fused FFN |
| Speculative decoding | Done (bonus) | Adaptive draft length, self-draft and external-draft modes |

**Exceeded expectations:** Mamba/Jamba SSM architectures, multimodal (vision + audio), and time-series architectures were not in the Phase 1 plan but are implemented.

---

## Phase 2: Developer Experience — IN PROGRESS (60%)

**Timeline:** NOW (Year 1) — ahead of schedule

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| One-line inference API | Partial | Generator API exists but could be simpler for common cases |
| Embedding extraction API | Done | `/v1/embeddings` endpoint in serve/ |
| Structured output / JSON mode | Done | Grammar-constrained decoding + JSON schema support |
| HuggingFace model download | Not started | No auto-download pipeline yet |
| Error messages and docs | Partial | Functional but no comprehensive getting-started guide |
| Example applications | Not started | No standalone example apps |

---

## Phase 3: Community and Ecosystem — NOT STARTED (0%)

**Timeline:** 3-6 months — correctly deferred

| Deliverable | Status |
|-------------|--------|
| GopherCon talk proposals | Not started |
| Comprehensive documentation | Not started |
| Contributor guide | Not started |
| Benchmark comparison publication | Not started |
| Tutorial series | Not started |
| Community channels | Not started |

---

## Phase 4: Multi-Backend GPU — AHEAD OF SCHEDULE (80%)

**Timeline:** 6-12 months — pulled forward by 6 months

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| CUDA backend | Done | 80 kernel files, cuBLAS, cuDNN, TensorRT, 25+ custom kernels |
| ROCm (HIP) backend | Done | Full HIP runtime, rocBLAS, MIOpen, flash attention |
| OpenCL (CLBlast) backend | Done | Complete runtime, CLBlast integration, DNN backend |
| GRAL abstraction layer | Done | Vendor-neutral interfaces across all 3 backends |
| Cross-backend benchmarks | Not done | No published comparison yet |

---

## Phase 5: Training and Fine-Tuning — AHEAD OF SCHEDULE (70%)

**Timeline:** 12-18 months — pulled forward by 12 months

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| Backpropagation all layers | Done | DefaultBackpropStrategy + OneStepApproximation |
| LoRA / QLoRA fine-tuning | Done | LoRA injection, NF4-quantized QLoRA, checkpoint management |
| Distributed training (gRPC/NCCL) | Done | Both strategies production-ready, FSDP implemented |
| RLHF training loop | Not started | No RLHF implementation found |
| Fine-tuning tutorials | Not started | No tutorials yet |

**Bonus implementations not in roadmap:** FP8 training with loss scaling, online learning with drift detection, AutoML (Bayesian optimization, population-based training), NAS (DARTS-based).

---

## Phase 6: Monetization — NOT STARTED (0%)

**Timeline:** 18-24 months — correctly deferred

| Deliverable | Status |
|-------------|--------|
| Enterprise support tier | Not started |
| Consulting offering | Not started |
| Cloud marketplace listing | Not started |
| Enterprise features (audit, SSO, multi-tenant) | Multi-tenant partially exists in serve/ |

---

## Codebase Vital Signs

| Metric | Value |
|--------|-------|
| Total Go code (non-test) | ~75,000+ lines |
| Active repositories | 6 |
| CUDA kernels | 80+ |
| Neural network operations | 100+ across 18 sub-packages |
| Quantization formats | 10+ |
| GPU backends | 4 (CUDA, ROCm, OpenCL, TensorRT) |
| Optimizers | 5 (AdamW, AdamW8bit, SGD, EMA, SWA) |
| KV cache strategies | 3 (standard, paged, GPU) |
| Distributed strategies | 2 (gRPC, NCCL) + FSDP |

### Repository Status

| Repo | Latest Tag | Vet | Tests | Clean |
|------|-----------|-----|-------|-------|
| float16 | v0.2.1 | Pass | Fail (1 pkg) | Yes |
| float8 | v0.3.1 | Pass | Pass | Yes |
| ztensor | v0.2.0 | Fail (16 warnings) | Pass (22 pkgs) | No |
| ztoken | v0.2.0 | Pass | Pass | Yes |
| zerfoo | v1.4.1 | Pass | Fail (4/84 pkgs) | Yes |
| zonnx | v0.6.0 | Pass | Pass | Yes |

---

## Open Issues (Priority Order)

### HIGH

1. **float16 rounding regression** — `TestShouldRound` fails for 4 rounding modes (NearestAway, TowardZero, TowardPositive, TowardNegative). Foundational dependency — blocks downstream correctness.

2. **zerfoo GGUF re-quantization bug** — Q5_K and Q6_K tensors are being incorrectly re-quantized to Q4_0 during model loading. Affects inference accuracy for models using these quantization formats.

### MEDIUM

3. **ztensor go vet warnings** — 16 "possible misuse of unsafe.Pointer" across GPU runtime bindings (CUDA, HIP, OpenCL, cuDNN, TensorRT). Likely intentional purego patterns but need suppression or fixes for CI credibility.

### LOW

4. **zerfoo TestBinaryBuilds stale path** — References a worktree path that no longer exists.
5. **zerfoo flaky tests** — `TestGenerate_returnsResult` and `TestSchedulerImmediateEviction` have timing/race-condition issues. Not production bugs.
6. **ztensor untracked `.claude/worktrees/`** — Needs `.gitignore` entry.

---

## Assessment

Zerfoo is significantly ahead of its five-year roadmap. At month 4 of Year 1:

- **Phase 1** (Inference Excellence) is 85% complete and on track
- **Phase 4** (Multi-Backend GPU) is 80% complete — 6 months ahead of schedule
- **Phase 5** (Training and Fine-Tuning) is 70% complete — 12 months ahead of schedule

### Key Risks

1. **Test failures** — 2 HIGH-priority correctness bugs (float16 rounding, GGUF Q5_K/Q6_K re-quantization) need immediate attention.
2. **Developer experience gap** — The framework is powerful but not yet accessible. No HuggingFace download, no examples, no tutorials. This will limit community adoption.
3. **No community infrastructure** — Phase 3 is entirely unstarted. Without docs, tutorials, and community channels, the technical excellence will not translate to adoption or the Year 2 star targets.

### Recommendation

Fix the HIGH bugs this week, then shift focus to Phase 2 (Developer Experience) and Phase 3 (Community and Ecosystem). The technical foundation is exceptionally strong — the bottleneck is now accessibility, not capability.
