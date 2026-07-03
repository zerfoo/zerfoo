# Zerfoo Work Plan -- Phase 1: Trust

**Last updated:** 2026 07 02
**Phase:** 1 of 4 (H2 2026) -- previous: Phase 0 complete 2026 07 02 (see Progress Log)
**Strategy source of truth:** docs/product-strategy-2026-H2.md (read it first) and docs/adr/093-h2-2026-trust-then-traction-strategy.md
**Phase 0 record:** all 19 tasks done same-day; full history in this file's git history (commit b718cd07 and earlier), docs/devlog.md 2026 07 02 entries (three), and the Progress Log below.

---

## Context

### How this plan works (read this if you are a new session)

Zerfoo's H2 2026 direction is set by docs/product-strategy-2026-H2.md ("Trust, then Traction", ADR-093). docs/plan.md is scoped to ONE phase at a time; each phase's plan ends with a task to plan the next phase. Reading order for a fresh session: (1) the strategy doc, (2) this file, (3) docs/devlog.md newest-first, (4) ADR-091/092/093, (5) docs/lore.md before debugging anything.

### Problem statement

Phase 1 makes every public claim true. Three correctness debts block a credible launch: the capture/replay lifetime bug cluster (#865 -> #870 -> #878, silent-wrong-gradients class, currently CONTAINED by the v1.56.0 loud-fail gate), the gemma4e degenerate decode (#757, fix candidate #766 unexecuted since April), and the kernel-numerics tail (#847, now including #922 -- a real kernels-package CUDA-context-poisoning bug the new standing DGX gate caught on its first honest run). In parallel, the public support claim moves from "45 architectures" to an evidence-backed verified-model matrix. Exit state: zero known silent-correctness bugs, matrix published, benchmarks reproducible -- the preconditions for Phase 2 (launch).

### Objectives (Phase 1 exit state)

1. #865, #870, #878 root-caused and fixed at the contract level, each with an ADR-091 harness fixture; the T129.1 fixture (tests/training/capture_replay_divergence_878_test.go) flips GREEN on the GB10; the v1.56.0 containment gate is removed; issues closed.
2. gemma4e decode fixed via T99.2.2.9 (#766) OR gemma4 edge demoted to experimental -- time-boxed, one candidate, no open-ended hypothesis hunting (ADR-093 rule 3).
3. #847 closed: #922 bisected and fixed, fixed-order fp32 reductions, remaining kernels oracle-gated, fused-encoder audit, ZTENSOR_DETERMINISTIC mode.
4. Verified-model matrix (~10 flagship models, each with parity + benchmark + GPU evidence) published; Ollama comparison re-run with reproduction manifests; T86.5.8 closed.
5. ztensor#171 fixed (darwin dev hosts can run tests again -- prerequisite for Phase 2 external contributors).
6. Phase 2 (Traction) is planned.

### Non-goals (Phase 1)

- Launch/GTM work: website, examples, posts (Phase 2). LTX-2/E127 and mmap/E125 (Phase 3). New backends, new model classes, perf moonshots (parked).
- The -tags cuda CGo build path (#921): decide wire-in vs document-as-DGX-only during E135, but continuous validation of it is not required for exit.

### Constraints and assumptions

- Single GB10 via Spark; ALL GPU runs through `scripts/dgx-validate.sh` (the standing gate, operational since Phase 0) or bench manifests; one GPU pod at a time; no interactive-SSH workloads.
- Model parity + benchmarks need GGUF files provisioned on the DGX host at /var/lib/zerfoo/models (T136.2, kind: human) -- the single external dependency of this phase.
- Cross-repo: kernel-numerics work lands mostly in ztensor; releases ship dependency-ordered (ztensor then zerfoo) via release-please; grep docs/lore.md (#capture #arena #dst) before touching any of it.
- darwin dev hosts cannot run GPU-importing tests until T137.1 lands (ztensor#171); use Linux CI or the DGX gate meanwhile.

### Success metrics

| Metric | Target | Measured by |
|---|---|---|
| Silent-correctness bugs | 0 open | #878/#870/#865 closed; fixture green on GB10 |
| Standing gate | full-scope green | dgx-validate.sh default run: failures [] |
| #847 umbrella | closed | oracle suite green; deterministic double-run proof |
| Verified matrix | 10+ models published | docs/verified-models.md with per-model evidence links |
| Benchmarks | reproducible | manifests in docs/bench/; benchmarks.md refreshed at current versions |
| gemma4 honesty | fixed or demoted | decode verified on GB10 OR registry marks experimental |

---

## Discovery Summary

Work type: engineering (all tasks carry verifies:) except T136.2 (operations, human) and T138.1 (strategy).

Inputs from Phase 0 (all discovered/verified 2026 07 02, sources in devlog):
- #878 mechanism hypothesis: device-resident loss seed / gradAccumulator state (built via engine.Fill) captured with stale/aliased state across replays. Red-proof fixture exists and is env-gated (ZERFOO_RUN_878_FIXTURE=1 + ZERFOO_UNSAFE_CAPTURE_TRAINING=1). Contract order for the cluster: #865 (stream ordering + scratch lifetime in flash-decode) -> #870 (FusedSDPA replay scratch) -> #878 (captured training state) -- same lifetime/stream class the June campaign fixed for eager mode (lore L-0006).
- #922: kernels package IMA cascade on GB10; TestKernelSoftmax passes in isolation; one early test poisons the context. Bisect via `scripts/dgx-validate.sh -pkgs "-v -run <subset> ./internal/cuda/kernels/"`.
- #766 fix candidate fully specified: keep native Q4_K storage for embedding-shaped tensors in model/gguf/loader.go decodeQ4KTensor (opt-in path already exists behind ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1 + isEmbeddingShape; the lossy Q4_K->f32->Q4_0 round-trip is still the default). Verification needs the gemma4e GGUF on the DGX host.
- #847 remainder breakdown lives in docs/plan-gpu-training-hardening.md (T3.2/T3.3/T3.4/T4.1) -- authoritative for E135 details.
- Candidate matrix models (verified-generate today per benchmarks.md: Gemma 3 1B, DeepSeek-R1 1.5B, Llama 3.2 3B, Mistral 7B, MiniMax-M2 229B CPU): extend to ~10 with Qwen 2, Phi-4, Llama 4 variant, Gemma 3 4B, Chronos-2 (timeseries, non-LLM evidence per the general-purpose rule).
- Use cases: UC-H2-003 (capture-replay training trains correctly), UC-H2-004 (verified-model matrix published with evidence), UC-H2-005 (gemma4e correct or honestly demoted), UC-H2-006 (darwin dev hosts run tests). Manifest: .claude/scratch/usecases-manifest.json.

---

## Scope and Deliverables

| ID | Description | Owner | Acceptance |
|---|---|---|---|
| D1 | Capture cluster fixed (#865/#870/#878), gate removed, released | TBD | T129.1 fixture GREEN on GB10 via the standing gate; issues closed; release tagged |
| D2 | gemma4e disposition executed | TBD | decode verified on GB10 with #766 fix, OR registry+docs mark experimental; #757/#766 closed either way |
| D3 | #847 closed incl. #922 | TBD | oracle suite green over kernel inventory; deterministic double-run bitwise-identical; standing gate full-scope green |
| D4 | Verified-model matrix + refreshed benchmarks | TBD | docs/verified-models.md live; benchmarks.md re-run at current versions with manifests; T86.5.8 closed |
| D5 | ztensor#171 darwin fix released | TBD | tests/training runs (skips cleanly) on a mac; ztensor tagged; zerfoo bumped |
| D6 | Phase 2 plan | TBD | docs/plan.md replaced, ends with plan-Phase-3 task |

---

## Checkable Work Breakdown

### E133: Capture/replay lifetime cluster -- root-cause fixes in contract order

Component: training/attention + ztensor boundary. Acceptance: all three issues closed with ADR-091 fixtures; fixture green on GB10; gate removed. Decision rationale: ADR-091 (every fix ships with a harness fixture), lore L-0006 (pointer stability under replay). Grep docs/lore.md #capture #arena before starting.

- [x] T133.1 Fix #865: flash-decode stream ordering + scratch lifetime  Owner: agent  Est: 4h  verifies: [UC-H2-003]  kind: agent  (done 2026-07-02, PR #928)
  - layers/attention/flash_decode.go launches FlashDecodeSplitKV on a private stream with no ordering against the engine stream that produced Q/K/V, and defer-frees scratch (partialO, partialLSE) non-stream-ordered. Fix: accept the engine stream (compute.StreamProvider pattern, mirroring the flash-forward fix in PR #866) + pool/epoch-managed or event-ordered frees. Add a parity-harness fixture reproducing the race shape.
  - Acceptance: fixture red on pre-fix behavior, green after; decode SDPA path exercised via the standing gate.
- [x] S133.1.1 Tests + lint  Owner: agent  Est: 1h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.1]  (done 2026-07-02, in PR #928: flash_decode_race_test.go + CI green)
- [x] T133.2 Fix #870: FusedSDPA replay-stable scratch  Owner: agent  Est: 4h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.1]  (done 2026-07-03, PR #933: persistent gpuScratchBuffer on the node; 511/511 replays green on GB10)
  - Illegal memory access on replay ~#141/511 with FusedSDPA under CaptureReplayRunner: per-call scratch freed/reused between replays. Fix: graph-owned/persistent scratch keyed to the captured graph's lifetime, or make FusedSDPA return ErrCaptureIncompatible and fall back to the discrete chain (which is clean under replay). Prefer the persistent-scratch fix; the error fallback is the floor.
  - Acceptance: capture-replay with FusedSDPA=true completes 511 replays on GB10 without IMA (or is cleanly refused); ADR-091 fixture committed.
- [x] S133.2.1 Tests + lint  Owner: agent  Est: 1h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.2]  (done 2026-07-03, in PR #933: flash_capture_replay_test.go ADR-091 fixture + CI green)
- [ ] T133.3 Fix #878: captured training-state aliasing  Owner: TBD  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.2]
  - Root-cause the silent gradient divergence: suspect gradAccumulator.seeds / device-resident loss seed built via engine.Fill being captured with stale or aliased state across replays. Fix at the CONTRACT level (allocation-stable captured operands; SaveForBackward-style pinning for captured state), not consumer special-casing. Localize with ZTENSOR_ARENA_POISON=1 if needed.
  - Acceptance: tests/training/capture_replay_divergence_878_test.go (ZERFOO_RUN_878_FIXTURE=1) GREEN on GB10 via `scripts/dgx-validate.sh -pkgs "-v -run TestCaptureReplayGradientDivergence878 ./tests/training/"`; capture-on and eager trajectories converge alike.
- [ ] S133.3.1 GB10 green proof + devlog  Owner: TBD  Est: 1h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.3]
- [ ] T133.4 Remove the containment gate; release; close the cluster  Owner: TBD  Est: 2h  verifies: [UC-H2-003]  kind: agent  blocked-by: [S133.3.1]
  - Remove the T129.2 loud-fail gate from NewCaptureReplayRunner (keep the constructor's error return; keep ZERFOO_DISABLE_CUDA_GRAPH); un-gate the fixture (drop ZERFOO_UNSAFE_CAPTURE_TRAINING from it; keep ZERFOO_RUN_878_FIXTURE as a long-test gate only if runtime demands). Ship release; close #865, #870, #878 with fix summaries; update docs/lore.md L-0006 with the resolution.

### E134: gemma4e decode -- time-boxed disposition

Component: model/gguf + inference. Acceptance: #757 and #766 closed, either way. ADR-093 rule 3 applies: ONE candidate, then demote. Blocked on T136.2 for on-GPU decode verification (model file).

- [ ] T134.1 Execute T99.2.2.9: native Q4_K embedding storage as default for gemma4e  Owner: TBD  Est: 4h  verifies: [UC-H2-005]  kind: agent  blocked-by: [T136.2]
  - Promote the existing opt-in path (ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1 + isEmbeddingShape in model/gguf/loader.go) to default-on for embedding-shaped Q4_K tensors (keep the env var as opt-OUT); extend the same guard to Q5_K/Q6_K loader symmetry if trivially in reach. Verify decode on GB10: gemma4e E2B Q4_K_M, greedy, coherent-English check per docs/QUALITY.md correctness criteria.
  - Acceptance: decode output coherent on GB10 (fix confirmed) OR degenerate persists (proceed to T134.2). Record either outcome in devlog with the exact prompt/output.
- [ ] T134.2 Disposition: close out or demote to experimental  Owner: TBD  Est: 2h  verifies: [UC-H2-005]  kind: agent  blocked-by: [T134.1]
  - If fixed: close #757/#766, update docs/updates.md and the matrix candidacy. If NOT fixed: mark gemma4/gemma4e builders experimental (registry warning at load + README/design.md note "decode correctness unverified"), close #766 as attempted, re-scope #757 to a parked epic with the full H1-H21 history pointer, and exclude Gemma 4 from all public claims (standing rule from devlog 2026-04-16).

### E135: Kernel numerics tail -- close #847 (incl. #922)

Component: kernels (mostly ztensor) + internal/cuda/kernels. Task-level detail lives in docs/plan-gpu-training-hardening.md (T3.2/T3.3/T3.4/T4.1) -- keep that doc as the authoritative sub-breakdown and check its boxes too. Grep docs/lore.md #arena #capture first.

- [x] T135.1 Bisect + fix #922: first-faulting kernels test on GB10  Owner: agent  Est: 1d  verifies: [UC-H2-003, infrastructure]  kind: agent  (done 2026-07-02, commits 047f2cc3/9c8e4e45/7ea2a605; residuals routed to T135.2/T135.3)
  - Bisect package test order with `scripts/dgx-validate.sh -pkgs "-v -run <subset> ./internal/cuda/kernels/"` (binary-search the test list; log head was truncated, first visible failures in elementwise_test.go). Identify the test that first triggers the IMA, fix the kernel/launcher (oracle-gate the fix per ADR-091). Also add a zero-tests-matched guard to dgx-validate-inpod.sh (footgun from devlog).
  - Acceptance: full `./internal/cuda/kernels/` green on GB10; standing gate default scope green end-to-end.
- [x] T135.2 Fixed-order fp32 tree reductions (plan-gpu-training-hardening T3.2)  Owner: agent  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  (done 2026-07-02, ztensor v1.19.2; zerfoo bumped b9613f7b)
  - ztensor-side: audit every reduction (ReduceSum, softmax denominator, norm stats, optimizer clipping norms) for accumulation order/dtype; convert to fixed-order pairwise/tree fp32 accumulation. Oracle diffs must tighten.
- [x] S135.2.1 Tests + lint (ztensor)  Owner: agent  Est: 2h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.2]  (done 2026-07-02, ztensor main CI green through v1.19.2 release)
- [x] T135.3 Oracle-gate remaining kernels (T3.3)  Owner: agent  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.1]  (done 2026-07-03, PR wave-2-task-T135.3: sgemv_m1.cu alignment fix, gemv_q4k_sm121.cu build fix, .so rebuilt via Spark build pod, honest per-op tolerance table docs/kernel-tolerances.md, full ./internal/cuda/kernels/ green on GB10)
  - Sweep the kernel inventory through the ztensor oracle harness on GB10; fix out-of-tolerance ops; commit the standing per-op tolerance table.
- [ ] T135.4 Fused encoder fwd/bwd audit (T3.4)  Owner: TBD  Est: 1d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.1]
- [x] T135.5 ZTENSOR_DETERMINISTIC mode (T4.1)  Owner: agent  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.2]  (done 2026-07-03: ztensor#179 (branch feat-deterministic-mode, awaiting review) + zerfoo branch wave-3-task-T135.5; GB10 proof 2 pods x 2 processes, 3/3 epoch losses bitwise-identical under the flag; honest exclusion: FusedEncoderBackward atomicAdd path errors under the flag; scope table in ztensor docs/design.md; devlog 2026-07-03)
  - Acceptance: two seeded GB10 epochs bitwise-identical per-epoch losses; scope documented honestly.
- [ ] T135.6 Close #847 + #921 disposition + devlog  Owner: TBD  Est: 2h  verifies: [infrastructure]  kind: agent  blocked-by: [T135.3, T135.4, T135.5]
  - Close the umbrella with a completion summary; decide #921 (wire -tags cuda into the pod with an nvcc image + in-tree build, or close as documented build-on-DGX-only policy); update plan-gpu-training-hardening.md status to COMPLETE.

### E136: Verified-model matrix + reproducible benchmarks

Component: docs + tests/parity + bench. The public support claim becomes this matrix (strategy doc P1). Non-Wolf validation rule applies: include at least one non-LLM (timeseries) entry.

- [x] T136.1 Matrix schema + flagship list + gate criteria  Owner: agent  Est: 2h  verifies: [UC-H2-004]  kind: agent  (done 2026-07-02, docs/verified-models.md, commits d92a17b3/4f50661d)
  - Create docs/verified-models.md: per-model row = architecture, GGUF source, quant, parity evidence (test + date + ref), benchmark evidence (tok/s + manifest + date), GPU-verified flag. Candidate list (~10): Gemma 3 1B + 4B, Llama 3.2 3B, Llama 4 (smallest runnable), Mistral 7B, Qwen 2, Phi-4, DeepSeek-R1-distill 1.5B, MiniMax-M2 229B (CPU/over-RAM), Chronos-2 (timeseries). Marketing may not exceed this file (ADR-093 rule 1); add that sentence to the file header.
- [ ] T136.2 Provision GGUF models on the DGX host  Owner: David  Est: 2h  verifies: [UC-H2-004]  kind: human
  - Host-side (requires host access; agents cannot do this): create /var/lib/zerfoo/models on the DGX and stage the GGUF files for the T136.1 list (pull via zerfoo CLI or copy). Then re-add the models hostPath mount to docs/bench/manifests/validate-arm64.yaml (path must pre-exist -- lore: Spark/Podman fails on missing hostPath). Also stage the gemma4e E2B GGUF for T134.1.
  - blocked: awaiting host provisioning by a person.
- [ ] T136.3 Run the parity subset for the matrix on GB10; close T86.5.8  Owner: TBD  Est: 4h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.1, T136.2]
  - Standing gate with models mounted: parity stage runs for every matrix model present. Record per-model results in docs/verified-models.md; close the T86.5.8 issue (#572 epic if fully satisfied) referencing the run.
- [ ] T136.4 Re-run the Ollama comparison with reproduction manifests  Owner: TBD  Est: 1d  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.2]
  - Current zerfoo release vs current Ollama on the GB10 via Spark bench manifests (bench-spark.sh / bench_tps), 3-run medians, same models as docs/benchmarks.md. Update benchmarks.md (keep history), fix the 14%-vs-28% inconsistency in docs/distribution/ drafts with the fresh numbers. GPU-serial; coordinate with E133-E135 GB10 usage.
- [ ] S136.4.1 Devlog + benchmarks.md entries  Owner: TBD  Est: 1h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.4]
- [ ] T136.5 Surface the matrix  Owner: TBD  Est: 2h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.3]
  - README section + design.md pointer replace architecture-count claims with the matrix link; updates.md entry.

### E137: darwin dev-host fix (ztensor#171)

- [x] T137.1 Fix ztensor device.init darwin dlopen SIGSEGV; release; bump  Owner: agent  Est: 4h  verifies: [UC-H2-006]  kind: agent  (done 2026-07-02, ztensor v1.19.1 + zerfoo 215fcefb; ztensor#171 closed)
  - In ztensor: guard the darwin dlopen probe so a missing/incompatible CUDA library yields a clean "no GPU" error instead of a SIGSEGV in init (repro: any test importing compute on darwin + Go 1.26.2 + ztensor v1.19.0). Ship ztensor release; bump zerfoo go.mod; verify tests/training runs (skips) on a mac.
  - Acceptance: `go test ./tests/training/ -run TestCaptureReplayGradientDivergence878` SKIPS (not crashes) on darwin; ztensor#171 closed.

### E138: Plan Phase 2

- [ ] T138.1 Produce the Phase 2 (Traction) plan  Owner: TBD  Est: 2h  delivers: [docs/plan.md replaced with the Phase 2 plan]  kind: any  blocked-by: [T133.4, T135.6, T136.5]
  - Run /plan with docs/product-strategy-2026-H2.md Part 4 Phase 2 as scope: website/docs site (Hugo per ADR-064; zerfoo.github.io is an empty scaffold), examples/ (6+ runnable apps), DX golden-path pass (pull->run->library quickstart; register the orphaned forecast CLI command), launch week (publish the docs/distribution/ drafts with T136.4's fresh numbers), GitHub Discussions + CONTRIBUTING + good-first-issues (E124 residue #773/#774/#796/#799), CFP submissions, and the ADR-084/090 major-version bump with the enterprise-repo extraction (human-led push). End with a task to plan Phase 3 (Moat).

---

## Parallel Work

| Track | Tasks | Notes |
|---|---|---|
| A: Capture cluster | T133.1 -> T133.2 -> T133.3 -> T133.4 | strictly sequential (contract order); GB10 runs serialize |
| B: gemma4e | T134.1 -> T134.2 | gated on T136.2 (model on host) |
| C: Kernel numerics | T135.1; T135.2 parallel; then T135.3/T135.4/T135.5; T135.6 last | T135.1 unblocks the gate's full-scope green |
| D: Matrix + bench | T136.1; T136.2 (human); then T136.3/T136.4; T136.5 | T136.2 is the external dependency -- surface it to David EARLY |
| E: darwin fix | T137.1 | independent; ztensor repo |
| F: Next plan | T138.1 | after A, C, D converge |

GB10 serialization: one GPU pod at a time across ALL tracks (E133 proofs, E135 bisect/oracle runs, E136 parity/bench). The coordinator owns GPU scheduling order: T135.1 first (unblocks gate green), then interleave.

### Waves

### Wave 1: Fan-out (5 agents + 1 human ask) -- COMPLETE 2026-07-02 (except human T136.2)
- [x] T133.1 #865 flash-decode stream ordering  verifies: [UC-H2-003]
- [x] T135.1 #922 bisect + fix  verifies: [UC-H2-003]
- [x] T135.2 fixed-order reductions (ztensor)  verifies: [UC-H2-003]
- [x] T136.1 matrix schema + list  verifies: [UC-H2-004]
- [x] T137.1 ztensor#171 darwin fix  verifies: [UC-H2-006]
- [ ] T136.2 model provisioning  kind: human  (ask David at wave start; everything in Tracks B/D waits on it) -- STILL OPEN, blocking T134.1/T136.3/T136.4

### Wave 2: Build-out (5 agents)
- [x] S133.1.1, T133.2 (chain)  verifies: [UC-H2-003] -- DONE 2026-07-03 (PR #933)
- [x] S135.2.1  verifies: [UC-H2-003] -- DONE 2026-07-02 (ztensor v1.19.2 CI)
- [x] T135.3 oracle-gate kernels  (after T135.1) -- DONE 2026-07-03
- [x] T135.4 fused encoder audit  (after T135.1)  2026 07 02  (DONE devlog entry; FFN/SwiGLU + FusedSDPA gradcheck added; FFN GELU-mode Backward bug fixed [always called SwiGLU backward regardless of useGELU]; fused PatchTST encoder backward wiring gap filed on #522 [E55, parked])
- [ ] T134.1 gemma4e fix attempt  (after T136.2)

### Wave 3: Deep fixes (4 agents)
- [ ] S133.2.1, T133.3 (chain)  verifies: [UC-H2-003]
- [x] T135.5 deterministic mode  (after T135.2) -- DONE 2026-07-03 (ztensor#179 + wave-3-task-T135.5; GB10 bitwise proof)
- [ ] T134.2 gemma4e disposition  (after T134.1)
- [ ] T136.3 matrix parity runs  (after T136.1, T136.2)

### Wave 4: Proof + ship (4 agents)
- [ ] S133.3.1 GB10 green proof
- [ ] T133.4 remove gate + release + close cluster
- [ ] T136.4 + S136.4.1 Ollama re-run  (GPU-serial with S133.3.1)
- [ ] T135.6 close #847/#921

### Wave 5: Surface + plan next (2 agents)
- [ ] T136.5 publish matrix
- [ ] T138.1 plan Phase 2

---

## Timeline and Milestones

| ID | Milestone | Member tasks | Exit criteria |
|---|---|---|---|
| M-P1-1 | Standing gate full-scope green | T135.1 | dgx-validate.sh default run: failures [] |
| M-P1-2 | Capture cluster closed | E133 | fixture green on GB10; gate removed; release tagged |
| M-P1-3 | Kernel numerics done | E135 | #847 closed; deterministic double-run proof |
| M-P1-4 | Matrix + benchmarks live | E136 | verified-models.md published; benchmarks.md refreshed; T86.5.8 closed |
| M-P1-5 | gemma4 honest | E134 | fixed or demoted, issues closed |
| M-P1-6 | Phase 2 planned | E138 | new plan.md |

Estimated wall-clock: 2-4 weeks; the long poles are GB10 serialization and the human-gated T136.2. Surface T136.2 to David in the first status update.

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | #878 root cause is deeper than the seed-aliasing hypothesis | High | Med | poison + deterministic modes localize; the fixture names the failure; if a new class emerges, file it with a fixture and re-plan the task, not the phase |
| R2 | T136.2 (human provisioning) stalls the matrix and gemma4e | Med | Med | ask at wave start; Tracks A/C/E proceed regardless; matrix can publish CPU-verified rows first |
| R3 | GB10 contention across tracks + Wolf | Med | High | coordinator schedules; T135.1 first; one pod at a time (SPARK_GPU_MAX=1) |
| R4 | gemma4e candidate fails and pressure builds to keep hunting | Med | Med | ADR-093 rule 3 is pre-committed: demote and move on; T134.2 exists precisely for this |
| R5 | ztensor/zerfoo version skew during parallel kernel + capture work | Med | Med | dependency-ordered releases; pin commit SHAs on branches until tags ship |
| R6 | Removing the containment gate too early | High | Low | T133.4 is blocked by the GB10 green proof (S133.3.1), not by code review alone |

---

## Operating Procedure

1. Definition of done: acceptance met; ADR-091 gates for any op/kernel change (gradcheck + parity + oracle); tests green; gofmt/vet/lint clean; PR rebase-merged, CI green; GPU acceptance = an actual dgx-validate.sh (or bench manifest) run recorded in devlog; observed, not expected.
2. One top-level directory per commit; conventional commits; no AI attribution; release-please tags; ztensor before zerfoo.
3. Grep docs/lore.md before debugging (#capture #arena #dst #gb10); quote L-NNNN IDs in commit messages when a rule applies; append new landmines via /lore.
4. All GPU work through Spark; one pod at a time; coordinate across tracks and with Wolf.
5. Plan checkboxes + Progress Log updated as tasks complete; findings to devlog (newest first); benchmarks to benchmarks.md.

---

## Carried-Forward (NOT in this phase)

- **E127 LTX-2 diffusion (Phase 3; ADR-092).** ~5/35 as of 2026 06 17. Next CPU-doable: flow-matching Euler scheduler (generate/diffusion), GroupNorm registry wiring, zonnx safetensors->GGUF converter (preserve per-tensor dtype: F32+BF16+F8_E4M3), weight-accurate VAE decoder. Landmines: oracle harness lives in ztensor; Conv3d GGUF "Conv" name collision needs rank-dispatch; parity baseline is PyTorch/ComfyUI on GB10. Issues #887, #888.
- **E125 mmap remaining (Phase 3).** #802 parked; needs cudaHostRegister purego binding in ztensor first.
- **E124 residue (Phase 2 good-first-issues).** #767, #773/#774/#796/#799; enterprise-repo push + major bump ship with Phase 2 launch (ADR-090).
- **Parked (label `parked`, 33 issues):** ROCm #701-#706, multi-GPU #712, edge #709/#710/#714, FP8-E5M2 #726, PJRT/E126, E55 fused-encoder epic #522+, E54 #517/#520/#521, E76 #570/#733/#734, perf micro-opts. Revive triggers documented on the issues.

---

## Progress Log

### 2026 07 03 -- Change Summary: T135.3 oracle-gate kernels done (Wave 2)

- T135.3 done: sgemv_m1.cu float4 vectorized-load misalignment fixed (row
  pointer alignment guard + scalar fallback, replacing the `N % 4 == 0`
  assumption that faulted for odd N and poisoned the CUDA context); found
  and fixed a SEPARATE build-blocking bug in gemv_q4k_sm121.cu (missing
  `cooperative_groups/reduce.h` include, nvcc 13.1 does not pull it in
  transitively) while rebuilding libkernels.so; rebuilt + deployed the .so
  via a one-shot Spark build pod (nvcr.io/nvidia/pytorch:26.02-py3 devel
  image, writable /opt/zerfoo/lib mount -- the sanctioned rebuild path,
  documented in docs/kernel-tolerances.md); replaced the flat 1e-4 relative
  bound in TestSgemvM1_MultipleSizes / TestGemvQ4KF32_* with a combined
  abs+rel tolerance (gemvReductionAbsTol=1e-5, gemvReductionRelTol=1e-4)
  honestly sized against measured GB10 worst-case error (catastrophic
  cancellation on near-zero reduction rows, not nondeterminism). Full
  `./internal/cuda/kernels/` green on GB10 (ref 368d68d1, pod
  zerfoo-validate-wave2taskT13-1783060264). Standing tolerance table:
  docs/kernel-tolerances.md. PR: wave-2-task-T135.3, references #847.

### 2026 07 02 (late) -- Change Summary: Wave 1 complete (T132.1 -> Wave 2 dispatched)

- Wave 1 done same-day, 5/5 agent tasks: T133.1 (#865 flash-decode stream ordering, PR #928, includes S133.1.1 race-shape fixture flash_decode_race_test.go), T135.1 (#922 bisected to the unguarded TestFP16GracefulWithoutCUDA null-pointer launch; fixed + zero-match gate guard; residuals routed to T135.2/T135.3), T135.2+S135.2.1 (ztensor v1.19.2 fixed-order pairwise fp32 reductions, zerfoo bumped), T136.1 (docs/verified-models.md schema + candidates), T137.1 (ztensor v1.19.1 darwin dlopen guard, ztensor#171 closed, zerfoo bumped).
- M-P1-1 correction (from T135.1 devlog): standing-gate full-scope green additionally depends on T135.2 (.so rebuild) and T135.3 (sgemv_m1 misalignment + oracle tolerances), not T135.1 alone.
- T136.2 (human: GGUF provisioning on DGX at /var/lib/zerfoo/models) remains the open external dependency; Tracks B (gemma4e) and D (matrix/bench GPU runs) are blocked on it.
- Wave 2 dispatched: T133.2 (FusedSDPA replay scratch), T135.3 (oracle-gate kernels), T135.4 (fused encoder audit).

### 2026 07 02 -- Change Summary: Phase 0 complete; plan advanced to Phase 1 (T132.1)

- Phase 0 executed in one day, all 19 tasks done: E128 tracker hygiene (57 issues closed across two passes, 33 parked, final state 60 open = 27 live + 33 parked, no over-closure); E129 #878 containment (fixture PR #916, loud-fail gate PR #920, released v1.56.0, #878 commented); E130 documentation truth (lore.md PR #914, ADR index PR #913, design.md PR #912, updates.md PR #918, ecosystem CLAUDE.md created, E76 parked); E131 standing DGX arm64 validation gate (PR #915 + 9 hardening iterations; operational; caught real bugs); E132 this plan.
- New issues from Phase 0 evidence: #921 (-tags cuda unbuildable from module checkout), #922 (kernels-package CUDA context poisoning on GB10 -- one early test faults, isolated tests pass), ztensor#171 (darwin dlopen SIGSEGV kills any importing test binary).
- Fixes shipped along the way: linux dlsym RTLD_DEFAULT test skip (7e29ef6c), dgx-validate events-on-failure/-keep/-delete/-pkgs/pre-pull, purego-scope + committed in-pod stage (Spark drops YAML block scalars), QUALITY.md vet policy in-pod.
- This Phase 1 plan created from docs/product-strategy-2026-H2.md Part 4 Phase 1: E133 capture cluster (contract order #865->#870->#878), E134 gemma4e time-boxed, E135 kernel numerics (#847+#922), E136 verified-model matrix + benchmark refresh (human-gated T136.2), E137 ztensor#171, E138 plan Phase 2.
- Use-case manifest updated: UC-H2-001/002 WIRED; UC-H2-003..006 added as PLANNED.

---

## Hand-off Notes

1. Read docs/product-strategy-2026-H2.md first; ADR-093 is the decision record. docs/lore.md is mandatory pre-debugging reading for anything touching capture, arena, dst, or the GB10.
2. The ONE external dependency is T136.2 (GGUF provisioning on the DGX host) -- a human task. Ask David early; do not let Tracks B/D silently stall on it.
3. GPU runs: `scripts/dgx-validate.sh` is the standing gate (supports -pkgs for targeted runs, -keep/-delete for pod inspection, -dry-run). One pod at a time. Known footgun: zero-matched tests count as pass (guard added in T135.1).
4. The #878 fixture is the phase's keystone proof: tests/training/capture_replay_divergence_878_test.go, gated by ZERFOO_RUN_878_FIXTURE=1 (+ ZERFOO_UNSAFE_CAPTURE_TRAINING=1 until T133.4 removes the gate). Expected RED on GB10 today; the phase exits with it GREEN.
5. docs/plan-gpu-training-hardening.md holds the authoritative T3.x/T4.x detail for E135; check its boxes as you go and mark it COMPLETE in T135.6.
6. Phase 0's full record: git history of this file (b718cd07 and earlier), three devlog entries dated 2026 07 02, PRs #912-#920, release v1.56.0.

---

## Appendix

- docs/product-strategy-2026-H2.md -- strategy, phases, metrics, kill criteria.
- docs/adr/093 -- strategy decision + one-phase plan scoping. docs/adr/091 -- verification gates. docs/lore.md -- landmines (L-0001..L-0012).
- docs/plan-gpu-training-hardening.md -- E135 sub-breakdown (T3.2/T3.3/T3.4/T4.1).
- Issue clusters: #865/#870/#878 (capture), #757/#766 (gemma4e), #847/#921/#922 (kernels), #572/T86.5.8 (parity), ztensor#171 (darwin).
- .claude/scratch/usecases-manifest.json -- UC-H2-003..006 active this phase.
