# Zerfoo Product Strategy & Roadmap — 2026 H2

**Date:** 2026-07-02
**Status:** ACTIVE — supersedes the priority ordering in docs/plan.md; does not replace docs/VISION.md (the 10-year thesis stands)
**Basis:** full review of docs/plan.md, docs/devlog.md (Apr–Jun 2026), docs/benchmarks.md, ADRs 068–092, the open issue tracker (100 issues), the codebase (inference/, generate/, serve/, training/, internal/), and the satellite repos (ztensor, ztoken, zonnx, float8, float16, zerfoo.github.io)

---

## Part 1 — What we have actually built

The March 2026 vision doc's claims hold up under code-level audit, and in most areas the reality now exceeds them.

### Verified, differentiated capability (the moats)

1. **Pure-Go, zero-CGo GPU inference that beats Ollama.** 241 tok/s on Gemma 3 1B Q4_K_M vs Ollama's 188 (1.28x) on the same GB10 hardware; 99.5% CUDA-graph decode coverage; 28 custom CUDA kernels bound via purego. `go build` works with no C compiler anywhere. Nobody else in the Go ecosystem has this.
2. **Over-RAM inference.** MiniMax-M2 229B MoE (128.8 GB Q4_K_M) loads in 6.3s and infers on a 128 GB machine via mmap streaming — Ollama returns a 500 on the same hardware. This is a category-defining demo that has never been published.
3. **A full training stack in Go, now hardened.** Backprop, LoRA/QLoRA, fp8/bf16 mixed precision, FSDP, gRPC/NCCL distributed, CUDA-graph capture-replay training. The June GPU-hardening campaign (plan-gpu-training-hardening.md, ADR-091) closed the memory/autograd lifetime bug class with a SaveForBackward contract, arena pinning, and poison mode — validated by two consecutive clean GB10 f32 training runs within 0.05pp of the CPU baseline.
4. **Per-op verification infrastructure PyTorch took a decade to build.** Gradcheck + OpInfo registry, GPU/CPU parity under arena stress, and a PyTorch-oracle harness on real hardware — standing gates for every new op (ADR-091, shipped in ztensor v1.11+).
5. **Breadth no Go project has attempted:** ~45 registered architecture builders (Llama 4, Gemma 3/3n/4, DeepSeek V3, Mamba 3, RWKV, MiniMax-M2, Kimi-linear…), time-series foundation models (Chronos-2, TiRex, Moirai-2) native in Go, multimodal (SigLIP vision, Whisper/Voxtral speech), speculative decoding incl. EAGLE, grammar-constrained JSON, tool calling, an OpenAI-compatible server with disaggregated prefill/decode.

### Honest maturity grading

| Tier | What | Evidence bar met |
|---|---|---|
| **Verified** | Gemma 3 1B, DeepSeek-R1 1.5B, Llama 3.2 3B, Mistral 7B, MiniMax-M2 229B (CPU); PatchTST/iTransformer training; eager-mode GPU f32 training | Benchmarked on GB10, parity-tested, reproducible |
| **Built, unverified** | ~40 other arch builders; most serve/ features; LoRA/fp8/FSDP beyond smoke tests | Unit/parity tests only; no published benchmark |
| **Broken or untrustworthy** | Gemma 4 edge decode (degenerate output, #757); CUDA-graph capture **training** (silent gradient divergence, #878); flash paths under replay (#870, #865) | Known defects on record |
| **Scaffold** | ROCm, OpenCL, PJRT backends; megakernel codegen; amd64 SIMD beyond GEMM; LTX-2 diffusion | Code exists, never hardware-validated |

The gap between tier 1 and tier 2 is the single most important product fact: **we market breadth ("40 architectures") that we have verified for ~5.**

---

## Part 2 — Where we left off (deep review findings)

### 2.1 The adoption gap is total, and it is the bottleneck

- **6 GitHub stars, 2 forks, 0 external contributors, 0 production users.**
- zerfoo.github.io is an empty `index.html` from August 2025. There is no website, no getting-started guide, no examples directory.
- Complete GTM collateral exists as unpublished drafts: a Show HN post, an r/golang post, a Golang Weekly pitch, a GopherCon proposal, and a full enterprise pricing sheet ($2k/$10k/mo tiers) — none ever posted.
- The March 2026 progress report already diagnosed this ("the bottleneck is now accessibility, not capability") and recommended shifting to DX + community. The four months since went instead into Gemma 4 edge (E92–E99), architectural cleanup (E124), GPU training hardening (#847), and LTX-2 (E127). The diagnosis was correct and the prescription was not followed.

Every quarter of world-class engineering with zero distribution compounds the risk that someone else (or Ollama itself) occupies the "ML in Go" position we defined.

### 2.2 Correctness debt that blocks a credible launch

1. **#878 — CUDA-graph capture-replay training silently produces wrong gradients** (losses ascend 10–20x, model collapses) on v1.50.2+. Silent-wrong-answer class: the worst kind. Capture-on training is advertised by the feature's existence and unusable in practice.
2. **#870 / #865 — flash attention paths under capture/replay**: illegal memory access on replay; flash-decode launches on a private stream with un-ordered scratch frees. Decode is our core mission; any stream-based consumer of SDPA decode is exposed.
3. **#757 / #766 — Gemma 4 edge decode has never produced correct output.** Ten weeks stalled with a fully specified fix candidate (T99.2.2.9: stop the lossy Q4_K→f32→Q4_0 re-quantization of `ple_embed_tokens` in `model/gguf/loader.go`). The devlog rightly bans Gemma 4 from marketing until fixed.
4. **Kernel-numerics tail from the hardening plan**: fixed-order fp32 reductions (T3.2), oracle-gating remaining kernels (T3.3), fused-encoder audit (T3.4), deterministic mode (T4.1) — all open on the #847 umbrella.

Pattern: all of 1–2 are one bug class — **buffer lifetime + stream ordering under capture/replay** — the same class the June campaign closed for eager mode. The fix approach (contract-level, harness-gated) is proven; it just hasn't been applied to the capture path.

### 2.3 Operational drag

- **Issue tracker: ~75% noise.** Generated org-fiction under the retracted E94 epic (Draft S-1, $150M ARR, FedRAMP "Est: 4h", ZerfooConf 2032), ~18 "(COMPLETE)" epics still open, stale handover notes. Maybe 20–25 of 100 open issues are live engineering.
- **Plan drift:** plan.md's status summary contradicts its own body (E92 "0/25" vs 25/25 done); wave checkboxes diverged from task checkboxes; hand-off notes recommend epics completed in March; updates.md stale since April 6.
- **Chronic "code done, DGX validation pending"** across E58/E61/E63/E86/E90, rooted in the purego darwin→linux cross-compile blocker. The workaround (native arm64 build on the DGX) is proven but was never institutionalized as CI.
- **No lore.md** despite hard-won invariants (SaveForBackward lifetimes, dst-form contract, host-access ordering, nonCapturableOps) scattered across devlog entries and five ADRs.
- **Dangling structure:** `../CLAUDE.md` referenced but missing; ADR index stops at 037 of 92; ztoken pinned at v0.3.4 while its repo shows v1.0.0; the ADR-090 enterprise extraction awaits its v3.0.0 major bump.

### 2.4 The three-fronts risk

Active work is split across (a) LTX-2 diffusion (E127, ~5/35), (b) capture-path correctness (#847 tail), and (c) latent Gemma 4 debt — while the highest-leverage work (launch) has no front at all. LTX-2 is a sound strategic bet (first non-autoregressive model class, reusable DiT primitives, a "local video gen in Go" headline), but it is a **second-priority** bet: it cannot matter publicly before the framework has any public.

---

## Part 3 — Product strategy: Trust, then Traction

### Thesis (unchanged)

Zerfoo is the ML runtime for the Go ecosystem: model inference as a library call — no Python, no CGo, no sidecar. The 10-year vision (docs/VISION.md) stands. What changes is the H2 2026 execution posture: **we stop widening the capability lead and start converting it.**

### Positioning

- **Primary wedge:** "Run LLMs inside your Go service, faster than Ollama, with `go build`." Target: backend Go engineers adding AI features; platform teams replacing Python sidecars.
- **Signature demos:** (1) beat-Ollama benchmark, reproducible via Spark manifests; (2) 229B model on a 128 GB box where Ollama fails; (3) OpenAI-compatible server in ~10 lines of Go; (4) fine-tune LoRA on your own data without leaving Go.
- **Unique niches nobody serves:** time-series foundation models (Chronos/TiRex/Moirai) native in Go; embeddable guardian/safety classification; (H2 bet) diffusion in Go via LTX-2.

### The three pillars, in priority order

**P1 — Trust: make every public claim true.**
No launch survives a user hitting silent-wrong-gradients or degenerate decode in week one. Close the capture/replay bug cluster, fix or demote Gemma 4 edge, finish the kernel-numerics tail, and replace "40 architectures" with a published, honest **verified-model matrix** (per-model: parity ✓, benchmark ✓, GPU ✓). The ADR-091 harnesses make this cheap now; verification is a product feature — "every op gradchecked against a PyTorch oracle" is itself a marketing claim no competitor makes.

**P2 — Traction: launch, for real.**
Website + docs site with a 5-minute quickstart, `examples/` (chat CLI, embedded HTTP service, RAG, structured output, LoRA fine-tune, time-series forecast), refreshed reproducible benchmarks, then actually post the Show HN / r/golang / Golang Weekly pieces. Submit to the next open CFP windows (GopherCon EU/US 2027, KubeCon 2027) and Go meetups now. Success is measured in external users, not internal epics.

**P3 — Moat: deepen only where differentiated.**
Finish mmap/E125 (the over-RAM story), continue LTX-2 as the single capability-expansion track (DiT-first per ADR-092), keep fast-follow support for frontier open-weight releases (the Gemma 4 muscle, once decode is fixed). Everything else — ROCm/OpenCL validation, PJRT/Trainium, megakernels, RLHF — is **parked** until hardware, a partner, or a user demands it.

### Explicit non-goals for H2 2026

- No enterprise sales motion, pricing publication, or zerfoo-enterprise buildout (the ADR-090 extraction ships as repo hygiene only). Revenue remains $0 by design; VISION.md Year-3 economics start after adoption exists.
- No new GPU backends, no multi-GPU epics (single-GB10 reality), no new model classes beyond LTX-2.
- No throughput moonshots: 500+ tok/s on GB10 is physically impossible (roofline ~257); the devlog already banned that target. Perf work is limited to regressions and the flash-decode ≥1.5x item.

---

## Part 4 — Roadmap

### Phase 0 — Stabilize the ground (July, ~2 weeks)

*Goal: an honest tracker, an honest plan, and the worst bug contained.*

- Bulk-close issue noise: E94 org-fiction (#697–#725 cluster), "(COMPLETE)" epics, stale handovers. Target: open issues ≈ live engineering (~25).
- Contain #878: until fixed, capture-replay training fails loudly (error or documented hard gate) rather than silently corrupting. Ship in a patch release.
- Plan hygiene: fix plan.md summary drift, mark superseded epics (E58, E90 residuals), file the missing bridge-elimination epic for E76 or close it as won't-fix; refresh updates.md; create docs/lore.md seeded from the invariants in devlog/ADR 088–092; fix or remove the dangling `../CLAUDE.md` reference; regenerate the ADR index.
- Institutionalize DGX validation: a Spark-submitted native arm64 build+test job (closes the purego cross-compile blocker class; unblocks T86.5.8 and the stale DGX-validation backlog in one move).

### Phase 1 — Trust (July–August)

*Goal: zero known silent-correctness bugs; a published verified-model matrix.*

- Fix the capture/replay lifetime cluster in contract order: #865 (stream-ordered flash-decode scratch) → #870 (replay-stable FusedSDPA scratch or `ErrCaptureIncompatible`) → #878 (captured seed/accumulator aliasing). Each with an ADR-091 harness fixture, per the June playbook.
- Execute T99.2.2.9 (native Q4_K embedding storage). Time-boxed: if gemma4e decode is still degenerate after this candidate, **demote Gemma 4 edge to experimental** in the registry and move on — ten more weeks of hypothesis-hunting is not a P1 use of the runway.
- Kernel-numerics tail: T3.2 fixed-order reductions, T3.3 oracle-gate remaining kernels, T3.4 fused encoder, T4.1 deterministic mode. Closes #847.
- Define and publish the **verified-model matrix**: pick ~10 flagship models (Gemma 3 1B/4B, Llama 3.2 3B, Llama 4, Mistral 7B, Qwen 2, Phi-4, DeepSeek-R1-distill, MiniMax-M2, Chronos-2) and run each through parity + benchmark on the GB10 via Spark. This matrix — not the 45-builder count — becomes the public support claim.
- Re-run and publish the Ollama comparison at current versions with reproduction manifests (fixes the stale 14%-vs-28% inconsistency in the draft posts).

### Phase 2 — Traction (September–October)

*Goal: Zerfoo exists in public. First external users and contributors.*

- Ship zerfoo.dev (or the github.io site): quickstart, install, verified-model matrix, benchmark methodology, API tour. Hugo scaffolding per ADR-064.
- `examples/` with 6+ runnable apps (chat CLI, embedded inference in an HTTP service, RAG with embeddings, structured-output extraction, LoRA fine-tune, TS forecasting).
- DX pass on the golden path: `zerfoo pull` → `zerfoo run` → library quickstart; error-message audit; register the orphaned `forecast` CLI command; README rewrite around the wedge.
- **Launch week:** Show HN + r/golang + Golang Weekly, with the over-RAM MiniMax demo as the second-day follow-up. Enable GitHub Discussions; add CONTRIBUTING.md and 10–15 curated good-first-issues (the E124 GELU residue and doc gaps are ideal).
- Submit CFPs (GopherCon 2027, KubeCon 2027, FOSDEM Go devroom, local meetups now).
- Ship the deferred major version alongside: ADR-084/090 extractions (crossasset already out; cloud/marketplace/compliance to zerfoo-enterprise) land as the long-anticipated major bump with a declared stable public API surface.

### Phase 3 — Moat + community loop (November–December)

*Goal: convert launch attention into a contributor pipeline; land one new-frontier proof.*

- Community iteration: triage external issues within 48h, monthly release notes (revive updates.md), 2–3 tutorial posts (e.g. "replacing a Python inference sidecar", "fine-tuning in pure Go").
- LTX-2 milestone (E127, DiT-first per ADR-092): DiT denoiser forward parity vs the GB10 PyTorch reference + per-step benchmark. Full video pipeline (VAE, audio, text encoder) continues into 2027 — the H2 deliverable is the parity-proven denoiser, which validates all the reusable diffusion primitives.
- mmap/E125 completion (cudaHostRegister binding, layer prefetch, 138 GB stress test) — feeds the over-RAM story.
- Fast-follow one major open-weights release in the quarter (whatever ships: Llama 4.x, Gemma 4 dense, Qwen 3) through the verified-matrix gate, as the repeatable "new model in N days" demonstration.

### 2027 H1 preview (directional)

Community-driven prioritization once real users exist; ROCm validation if AMD hardware materializes; PJRT/Trainium only with a partner; LoRA/fine-tuning DX as the second wedge; LTX-2 full pipeline; revisit VISION.md Year-2 targets against actual traction data.

---

## Part 5 — Metrics and governance

### H2 2026 scorecard (replaces the fantasy metrics in the tracker)

| Metric | Now | EOY 2026 target |
|---|---|---|
| Open silent-correctness bugs | 1 (#878) | 0 |
| Verified-model matrix entries | ~5 (unpublished) | 10+ published |
| GitHub stars (zerfoo/zerfoo) | 6 | 500+ |
| External contributors | 0 | 3+ |
| External production users (self-reported) | 0 | 2+ |
| Docs site + examples | none | live, 6+ examples |
| Launch posts published | 0 | HN + r/golang + newsletter |
| CFPs submitted | 0 | 3+ |
| Issue tracker signal ratio | ~25% | >90% |
| Gemma 3 1B decode (regression floor) | 241 tok/s | ≥241 (no regressions) |

### Operating rules

1. **Verification is the gate for claims.** A model enters the public matrix only with parity + benchmark evidence on record; an op merges only through the ADR-091 gates. Marketing copy may not exceed the matrix.
2. **One capability-expansion front at a time** (currently LTX-2). New model-class or backend epics require retiring or finishing the current front.
3. **Time-box correctness hunts.** After a specified fix candidate fails, demote the feature to experimental rather than continuing open-ended hypothesis testing (the gemma4e lesson: 21 hypotheses, 10 weeks).
4. **The tracker is the truth.** Issues are engineering-real or they are closed. Aspirational/business planning lives in strategy docs, not GitHub issues.
5. **Consumer workloads harden the framework at the contract level** (Wolf rule, unchanged): fixes are general guarantees, never consumer special cases; acceptance always includes a non-Wolf path.

### Kill / pivot criteria

- If the launch (Phase 2) yields <100 stars and zero engaged users in 60 days, the problem is positioning, not capability — pause capability work and iterate on the wedge (candidate pivots: time-series-first for Go fintech/observability shops; edge/embedded-first via zerfoo-edge).
- If #878-class capture bugs recur after the Phase 1 fixes, disable capture-mode training in the public API until a redesigned capture contract ships — eager GPU training is already fast enough to be honest about.

---

## Appendix — Backlog dispositions

- **Bulk-close (Phase 0):** #697–#700, #707–#708, #714–#725 (E94 org-fiction); #540, #544–#545, #558–#569, #571 ("(COMPLETE)" epics); #839, #696 (stale handover/archived).
- **Park (explicitly deferred, keep open, label `parked`):** ROCm suite #701–#706 (no hardware); #712 multi-GPU; #709/#710 RPi/Jetson; #726 FP8 E5M2; E126/PJRT (#T126.x); E55 fused-encoder kernel; perf micro-opts #524–#539, #543, #557, #640, #606, #692.
- **Close as won't-fix / superseded:** #711 (500+ tok/s — physically impossible on GB10, per devlog 2026-03-19); E58/E90 residual DGX tasks (superseded by later fixes / wolf extraction).
- **Live engineering (H2 scope):** #878, #870, #865, #847 tail (Phase 1); #757/#766 (time-boxed, Phase 1); #887, #888 (E127, Phase 3); #802/E125 (Phase 3); #767/E124 residue + #773/#774/#796/#799 (good-first-issue candidates); #733/#734, #730/#731 (fold into the arm64 CI job).
