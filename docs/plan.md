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
6. Deep-review 002 (docs/deep-reviews/002-full-codebase.md) tiers 1-2 closed: all 9 High findings (F1/F2/F3, DIST-1/2, CUDA-1/2, OCI-1/2, SERVE-1/2, CONC-H1/H2) fixed and re-verified; CICD-1/2 fixed; already-written security capabilities (rate limiter, keystore, mTLS) wired into the CLI. See ADR-094.
7. Phase 2 (Traction) is planned.

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
| Security posture (deep-review 002) | 0 open High findings; CICD-1/2 fixed | every High-finding fix has a repro test; re-verified against docs/deep-reviews/002-full-codebase.md roadmap tiers 1-2 |

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

Inputs from deep-review 002 (2026-07-03, HEAD 5817d590, full text docs/deep-reviews/002-full-codebase.md; policy decision docs/adr/094-untrusted-boundary-security-hardening.md):
- 9 High findings, no Critical: F1 (GGUF element-count overflow, check-after-multiply bug, 4 duplicate loader sites), F2 (GGUF offset signed-conversion, default mmap path), DIST-1 (unauth/unencrypted distributed worker, 0.0.0.0 bind), CUDA-1 (CWD dlopen of libkernels.so -> local code exec), OCI-1 (OCI pull never verifies blob digest, library-only reachability), SERVE-1 (pre-auth metric-label cardinality, conditional on a real collector), SERVE-2 (LoRA adapter-name path traversal, conditional on WithAdapterCache), CONC-H1 (SpeculativeGenerate bypasses graphMu), CONC-H2 (model-delete TOCTOU + WaitGroup misuse).
- 10 Medium, 7 Low, 3 Info findings across DIST-2, OCI-2, CUDA-2, SERVE-3/3b/4/5/6/7, SSRF-1, HF-1, CONC-M1/M2/L1, CICD-1..6, L1-L3, SLSA gaps -- full detail and exact fix diffs in the review doc.
- Positive baseline (do not regress): secure-by-default serve auth, constant-time key compare, SHA-256 hashed keys, best-in-class connect-time SSRF defense (defeats DNS rebinding), correct path-traversal defense on the HF/OCI download path, textbook AES-256-GCM, clean secrets/dependency hygiene.
- Architectural theme: security capabilities (rate limiter, keystore, mTLS, incident responder) are implemented correctly but never wired into the shipped CLI -- ADR-094 makes "wire the defense you write" a standing rule.
- New use cases: UC-H2-007 (GGUF loader crash-safe), UC-H2-008 (distributed wire authenticated/encrypted), UC-H2-009 (native lib loading trusted-path-only), UC-H2-010 (concurrency race-free), UC-H2-011 (HTTP resource/traversal bounded), UC-H2-012 (CI/CD supply chain hardened). All P0/P1, PLANNED.

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
| D7 | Deep-review 002 tiers 1-2 closed | TBD | 9 High findings + CICD-1/2 fixed with repro tests; unwired defenses (rate limiter/keystore/mTLS) exposed via CLI flags; docs/deep-reviews/002-full-codebase.md status header updated |

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
- [x] T133.3 Fix #878: captured training-state aliasing  Owner: agent  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.2]  (done 2026-07-03, PR #937: root cause confirmed -- the d(loss)=1 seed in buildOnesSeed was the one piece of cross-step captured state still arena-pooled via engine.Fill; re-homed into a nil-pool GPUStorage, same pattern as newPersistentGradTensor; red-proof to bit-identical-trajectory green on GB10; also fixed the merged T129.1 fixture, which had never actually exercised the bug, and a second .so symbol-drift regression from T135.3)
  - Root-cause the silent gradient divergence: suspect gradAccumulator.seeds / device-resident loss seed built via engine.Fill being captured with stale or aliased state across replays. Fix at the CONTRACT level (allocation-stable captured operands; SaveForBackward-style pinning for captured state), not consumer special-casing. Localize with ZTENSOR_ARENA_POISON=1 if needed.
  - Acceptance: tests/training/capture_replay_divergence_878_test.go (ZERFOO_RUN_878_FIXTURE=1) GREEN on GB10 via `scripts/dgx-validate.sh -pkgs "-v -run TestCaptureReplayGradientDivergence878 ./tests/training/"`; capture-on and eager trajectories converge alike.
- [x] S133.3.1 GB10 green proof + devlog  Owner: agent  Est: 1h  verifies: [UC-H2-003]  kind: agent  blocked-by: [T133.3]  (done 2026-07-03, delivered inline in PR #937: pods ...1783116105/...1783117013 bit-identical baseline/eager-reset/capture-reset trajectories; devlog entry included)
- [x] T133.4 Remove the containment gate; release; close the cluster  Owner: agent  Est: 2h  verifies: [UC-H2-003]  kind: agent  (done 2026-07-03, PR #946 -- gate removed, fixture ungated (SKIPs cleanly, not crashes, with the gate gone), docs/lore.md L-0006 updated; coordinator ran GB10 proof: {"build":"pass","vet":"pass","cuda_tests":"pass","failures":[]}; #865/#870/#878 closed with fix-summary comments referencing PRs #928/#933/#937; release-please will tag from the merged conventional commits)
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
- [x] T135.4 Fused encoder fwd/bwd audit (T3.4)  Owner: agent  Est: 1d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.1]  (done 2026-07-02: see devlog "Fused encoder fwd/bwd audit -- FFN GELU-backward bug + zero gradcheck coverage on FusedSDPA/FFN closed (T135.4)"; this box was left stale despite the Waves-section entry and devlog already recording completion -- fixed as part of T135.6's doc-consistency pass)
- [x] T135.5 ZTENSOR_DETERMINISTIC mode (T4.1)  Owner: agent  Est: 1.5d  verifies: [UC-H2-003]  kind: agent  blocked-by: [T135.2]  (done 2026-07-03: ztensor#179 (branch feat-deterministic-mode, awaiting review) + zerfoo branch wave-3-task-T135.5; GB10 proof 2 pods x 2 processes, 3/3 epoch losses bitwise-identical under the flag; honest exclusion: FusedEncoderBackward atomicAdd path errors under the flag; scope table in ztensor docs/design.md; devlog 2026-07-03)
  - Acceptance: two seeded GB10 epochs bitwise-identical per-epoch losses; scope documented honestly.
- [x] T135.6 Close #847 + #921 disposition + devlog  Owner: agent  Est: 2h  verifies: [infrastructure]  kind: agent  blocked-by: [T135.3, T135.4, T135.5]  (done 2026-07-10, PR wave-sec3-task-T135.6: docs/plan-gpu-training-hardening.md marked COMPLETE incl. retroactively checking T3.2/S3.2.1 + stale Wave 1-4 boxes that lagged the per-task entries; #921 recommendation -- close as documented DGX-only build policy, not wire -tags cuda into the standing gate; fork-parity symbol check added -- internal/cuda/kernels/symbol_parity_test.go TestForkParitySymbols, red-proofed against the T135.3 drift class; #847 close-summary comment prepared in devlog for the coordinator to post; #847/#921 NOT closed here -- coordinator action. NEW FINDING carried forward, not fixed: zerfoo's own internal/cuda/kernels/Makefile:7 still has --use_fast_math -- T3.1's removal only ever landed in ztensor's copy; since the deployed .so is now built from zerfoo's own Makefile [T135.3], the live artifact is not actually fast-math-free. Recommend a fresh follow-up issue/task, not reopening E135.)
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

- [ ] T138.1 Produce the Phase 2 (Traction) plan  Owner: TBD  Est: 2h  delivers: [docs/plan.md replaced with the Phase 2 plan]  kind: any  blocked-by: [T133.4, T135.6, T136.5, T145.2]
  - Run /plan with docs/product-strategy-2026-H2.md Part 4 Phase 2 as scope: website/docs site (Hugo per ADR-064; zerfoo.github.io is an empty scaffold), examples/ (6+ runnable apps), DX golden-path pass (pull->run->library quickstart; register the orphaned forecast CLI command), launch week (publish the docs/distribution/ drafts with T136.4's fresh numbers), GitHub Discussions + CONTRIBUTING + good-first-issues (E124 residue #773/#774/#796/#799), CFP submissions, and the ADR-084/090 major-version bump with the enterprise-repo extraction (human-led push). End with a task to plan Phase 3 (Moat).

### E139: Untrusted-input hardening -- GGUF loader + OCI registry (deep-review 002)

Component: model/gguf + model/registry. Acceptance: F1/F2/F3 closed with a shared, tested helper; OCI-1/OCI-2 closed. Decision rationale: ADR-094 (treat the model file as first-class untrusted input). Source: docs/deep-reviews/002-full-codebase.md F1/F2/F3/OCI-1/OCI-2.

- [x] T139.1 Fix F1: GGUF element-count integer-overflow bypasses the size cap  Owner: agent  Est: 3h  verifies: [UC-H2-007]  kind: agent  (done 2026-07-03, PR #945: shared computeNumElements helper, all 4 duplicate sites deduplicated; attack shape returns error not panic)
  - `model/gguf/loader.go:55-64` (dup in `loader_mmap.go:23-33`, `split_file.go:150-159,219-228`) multiplies then checks with strict `>`, so the running product can land on exactly `1<<34` and a later dimension overflows int64 to negative, passing the check and reaching `make([]byte, negativeSize)`. Fix: check `numElements > (1<<34)/int64(d)` BEFORE multiplying, at all four sites; extract into one shared `computeNumElements(ti) (int64, error)` helper so the fix exists in exactly one place (per ADR-094).
  - Acceptance: a crafted `Dimensions = [131072, 131072, 2147483647]` GGUF returns an error, not a panic, on every one of the four call sites.
- [x] T139.2 Fix F2: GGUF tensor offset signed-conversion out-of-bounds panic  Owner: agent  Est: 2h  verifies: [UC-H2-007]  kind: agent  (done 2026-07-03, PR #940: full bounds check before mmap slice at both sites; repro'd the exact panic pre-fix, confirmed clean error post-fix)
  - `model/gguf/loader_mmap.go:51-57` (dup `split_file.go:171-177`, the DEFAULT mmap load path) converts a file-controlled `uint64` offset to `int64` with no unsigned validation; a huge offset wraps negative and slips past the single `end >` check. Fix: validate `ti.Offset <= math.MaxInt64` and bounds-check `offset >= 0 && offset <= len(mapped) && sz <= len(mapped)-offset` before slicing.
  - Acceptance: `ti.Offset = 0x8000000000000000` returns an error, not a slice-bounds panic.
- [x] T139.3 Fix F3: cap GGUF tensor dimension count  Owner: agent  Est: 1h  verifies: [UC-H2-007]  kind: agent  (done 2026-07-10, PR #951: maxTensorDims=8 cap in parser.go, the sole TensorInfo construction site from untrusted input; attack shape and boundary cases tested)
  - `parser.go:131-136` reads an unbounded number of dimensions per tensor. Cap `numDims` at 8 and return an error above that.
- [x] S139.3.1 Malformed-GGUF fuzz corpus + table tests  Owner: agent  verifies: [UC-H2-007]  kind: agent  blocked-by: [T139.1, T139.2, T139.3]  (done 2026-07-10, PR #972: FuzzParse over parse+load seeded with F1/F2/F3 shapes; bounded 30s fuzz step in ci.yml; found+relaxed an over-strict empty-name assertion)
  - Go native fuzzing (`go test -fuzz`) seeded with the F1/F2/F3 repro shapes above, plus a table test of legitimate tensor shapes proving no regression. Wire into CI as a bounded fuzz run (time-boxed), not a one-shot manual repro.
- [x] T139.4 Fix OCI-1: verify blob digest on OCI pull  Owner: agent  Est: 2h  verifies: [UC-H2-007]  kind: agent  (done 2026-07-10, PR #949: sha256Digest recompute-and-compare wired into Pull before disk write; mismatch test confirms no file is written)
  - `model/registry/oci.go:199-207` `Pull` writes `getBlob` bytes to disk with no recompute-and-compare against `ggufLayer.Digest`, even though `sha256Digest` (`:367`) already exists (used only on push). Fix: recompute `sha256Digest(data)` and reject a mismatch.
- [x] T139.5 Fix OCI-2: reject non-https registry URLs  Owner: TBD  Est: 1h  verifies: [UC-H2-007]  kind: agent  blocked-by: [T139.4]  (done 2026-07-10, PR #968: reject non-https OCI registry URLs by default, WithInsecureHTTP() opt-in; NewRegistry now returns an error)
  - `oci.go:46-55` accepts plain `http://`. Reject non-`https://` unless an explicit insecure flag is set.
- [ ] S139.5.1 Tests + lint  Owner: TBD  Est: 1h  verifies: [UC-H2-007]  kind: agent  blocked-by: [T139.5]
  - Unit tests: digest-mismatch rejected, http:// rejected without the flag, https:// and matching-digest accepted unchanged.

### E140: Distributed-training wire security (deep-review 002)

Component: distributed/. Acceptance: DIST-1 closed (mTLS wired, non-loopback requires TLS); DIST-2 closed (coordinator authenticates registration). Decision rationale: ADR-094 (the distributed wire is untrusted network, fail closed not open).

- [x] T140.1 Fix DIST-1: wire mTLS into the worker gRPC server  Owner: agent  Est: 4h  verifies: [UC-H2-008]  kind: agent  (done 2026-07-03, PR #947: TLS wired into worker_node.go, non-loopback bind without TLS refuses to start; also threaded TLS into GrpcStrategyConfig so the client-side dial is no longer forced insecure)
  - `distributed/worker_node.go:64` creates `grpc.NewServer()` with zero opts; `distributed/tlsconfig.go`'s `RequireAndVerifyClientCert` builder is correct but has no caller. Fix: if `wn.config.TLS != nil`, wire `wn.config.TLS.ServerCredentials()` via `grpc.Creds`; else refuse to start on a non-loopback `WorkerAddress`.
  - Acceptance: a loopback bind with no TLS config still starts (dev UX unchanged); a routable bind with no TLS config returns a startup error; a routable bind with TLS config starts and requires client certs.
- [x] T140.2 Default the worker CLI example to loopback; add --tls-* flags  Owner: agent  Est: 2h  verifies: [UC-H2-008]  kind: agent  blocked-by: [T140.1]  (done 2026-07-10, PR #955: 127.0.0.1:9001 default, --tls-cert/--tls-key/--tls-ca flags wired via buildTLSConfig, openssl cert-gen docs in Usage())
  - `cmd/cli/worker.go:127` ships `--worker-address 0.0.0.0:9001` in its documented example. Change the example/default to `127.0.0.1:9001`; add `--tls-cert`/`--tls-key`/`--tls-ca` flags wired to `distributed.TLSConfig`; document a cert-gen helper for multi-host runs.
- [x] T140.3 Fix DIST-2: authenticate coordinator worker registration  Owner: agent  Est: 3h  verifies: [UC-H2-008]  kind: agent  blocked-by: [T140.1]  (done 2026-07-10, PR #956: coordinator wires the same TLSConfig pattern as the worker plus a defense-in-depth auth interceptor rejecting RPCs with no verified client cert, so RegisterWorker never discloses the peer list pre-auth)
  - `coordinator.go:71-93` is unauthenticated unless `SetServerOptions` is called (no caller exists today); `RegisterWorker` (`:160`) trusts any caller to claim a rank and returns the full peer list. Fix: require TLS creds by default (mirror T140.1's server-credential wiring) and gate `RegisterWorker` on a valid client cert/token before returning peers.
- [ ] S140.3.1 Tests + lint  Owner: TBD  Est: 2h  verifies: [UC-H2-008]  kind: agent  blocked-by: [T140.3]
  - Integration test: a routable-bind worker/coordinator refuses to start without TLS; a TLS-configured pair completes a handshake and an AllReduce round-trip; an unauthenticated connection attempt to the coordinator is rejected before the peer list is returned.

### E141: Native library loading hardening (deep-review 002)

Component: internal/cuda + internal/*/purego. Acceptance: CUDA-1 closed (no CWD/bare-soname dlopen); CUDA-2 closed or explicitly mitigated (absolute vendor paths / validated resolution). Decision rationale: ADR-094 (vetted absolute paths only).

- [x] T141.1 Fix CUDA-1: remove CWD dlopen candidates for the kernel library  Owner: agent  Est: 2h  verifies: [UC-H2-009]  kind: agent  (done 2026-07-03, PR #943: kernelLibPaths now trusted-absolute-only, with a validated non-world-writable env override for dev builds)
  - `internal/cuda/purego.go:164-169` includes `"./libkernels.so"` (and a bare `"libkernels.so"`) in `kernelLibPaths`, loaded by `DlopenKernels`; `dlopen` runs ELF constructors on load, so a planted file in the process's working directory gets code execution. Fix: drop both entries; keep only the trusted absolute path (`/opt/zerfoo/lib/libkernels.so`); if a configurable override is needed, read one absolute path from an env var and `os.Stat`-verify it is not world-writable before loading.
  - Acceptance: `kernelLibPaths` contains zero relative or CWD-implying entries; a test asserts the loader rejects a relative-path candidate.
- [ ] T141.2 Fix CUDA-2: prefer absolute vendor-library paths  Owner: TBD  Est: 3h  verifies: [UC-H2-009]  kind: agent  blocked-by: [T141.1]
  - All native libs (`libcudart.so.12`, `libcublas.so.12`, HIP, rocBLAS, MIOpen, OpenCL across `internal/*/purego.go`) load by bare soname, hijackable via `LD_LIBRARY_PATH`/RPATH given env control. Fix: prefer absolute vendor install paths where known (mirror T141.1's pattern); where a bare soname is unavoidable (system libs), document the residual trust assumption explicitly rather than leaving it silent.
- [x] S141.2.1 Tests + lint  Owner: agent  Est: 1h  verifies: [UC-H2-009]  kind: agent  blocked-by: [T141.2]  (done 2026-07-10, in PR #958: dlopen_security_test.go per package, covering absolute-path-first ordering, override vetting, CWD-relative rejection)
  - Unit test asserting the dlopen path-candidate list contains no relative paths and no unconditional bare-soname-only fallback without a documented rationale.

### E142: Concurrency correctness (deep-review 002)

Component: inference/, serve/. Acceptance: CONC-H1/H2 closed (High); CONC-M1/M2/L1 closed (Medium/Low). Decision rationale: existing `graphMu` design is correct; these are entrypoints that escaped it, not a broken model -- fix the entrypoints, do not redesign the lock.

- [x] T142.1 Fix CONC-H1: route SpeculativeGenerate through graphMu  Owner: agent  Est: 3h  verifies: [UC-H2-010]  kind: agent  (done 2026-07-03, PR #948: LockGraph/UnlockGraph exposed on Generator; race+corruption reproduced pre-fix, clean post-fix)
  - `inference/inference.go:872-882` builds a speculative generator over the shared singleton graphs and calls `Forward` with no lock, while every normal generation path serializes on `gen.mu` (`generate/session.go:91`). Two concurrent requests (one speculative, one normal) then mutate the same stateful graph simultaneously. Fix: expose `LockGraph()`/`UnlockGraph()` on `Generator` and wrap `SpeculativeGenerate`'s call with it.
  - Acceptance: with `WithDraftModel` enabled, a concurrent normal + speculative request pair under `-race` shows no race and no corruption.
- [x] T142.2 Fix CONC-H2: model-delete TOCTOU + WaitGroup misuse  Owner: agent  Est: 3h  verifies: [UC-H2-010]  kind: agent  (done 2026-07-03, PR #944: replaced inflight WaitGroup with an RWMutex across all 7 affected handlers, incl. guard.go/classify.go beyond the review's literal scope since they shared the same hazard)
  - `serve/handlers.go:19,172,343` call `s.inflight.Add(1)` at handler entry with no `unloaded` recheck, racing `handlers.go:331-333`'s `unloaded.Store(true)` -> `inflight.Wait()` -> `s.model.Close()`. Fix: recheck `s.unloaded.Load()` immediately after `Add(1)` and back out cleanly if set (or replace the Add/Wait pair with an RWMutex: `RLock` per handler, `Lock` before close).
  - Acceptance: a race test interleaving `DELETE /v1/models/{id}` with a concurrent chat request under `-race` shows neither a use-after-close nor a `WaitGroup misuse` panic.
- [x] T142.3 Fix CONC-M1: RateLimiter unbounded bucket map  Owner: agent  Est: 2h  verifies: [UC-H2-010]  kind: agent  (done 2026-07-10, PR #957: RateLimiter.Start/Stop schedules Cleanup() on a ticker + maxBuckets eviction backstop; wired into server lifecycle mirroring BatchScheduler's Start/Stop)
  - `serve/security/network.go:98` `Cleanup()` exists but is never scheduled outside tests, so `rl.buckets` grows one permanent entry per distinct client IP. Fix: a background ticker calling `Cleanup()` every `cleanTTL`, plus a size cap as a backstop.
- [x] T142.4 Fix CONC-M2: KeyStore field race on Revoked/ExpiresAt  Owner: agent  Est: 2h  verifies: [UC-H2-010]  kind: agent  (done 2026-07-10, PR #953: Lookup returns a value copy under RLock rather than the live pointer -- chosen over atomic.Bool because APIKey is JSON-persisted to bbolt and atomic.Bool has no exported fields; race repro'd pre-fix, clean post-fix)
  - `authMiddleware` (`server.go:282-288`) reads `key.Valid()`/`HasScope()` lock-free while `Revoke`/`Rotate` write `k.Revoked`/`ExpiresAt` under lock (`apikey.go:186,205`) -- a data race plus a brief post-revoke authorization window. Fix: make `Revoked` an `atomic.Bool` (simplest), or have `Lookup` return a value copy under the existing lock.
- [x] T142.5 Fix CONC-L1: executeBatch ctx-wait goroutine leak  Owner: agent  Est: 1h  verifies: [UC-H2-010]  kind: agent  (done 2026-07-10, PR #950: added a select on batchCtx.Done() so per-request goroutines are reaped when the batch completes, not only on the request's own ctx; regression test reproduced the exact leaked-goroutine count pre-fix)
  - `batch.go:166-173` accumulates per-request ctx-wait goroutines transiently. Fix: add a `select` on `batchCtx.Done()` so they are reaped when the batch returns.
- [ ] S142.5.1 Race-detector test suite for this epic  Owner: TBD  Est: 2h  verifies: [UC-H2-010]  kind: agent  blocked-by: [T142.1, T142.2, T142.3, T142.4, T142.5]
  - `go test -race` covering: delete-during-inflight-request, speculative+normal concurrent generate, concurrent revoke+authenticate, sustained multi-IP rate-limit traffic (bucket count stays bounded), batch cancellation goroutine count returns to baseline.

### E143: HTTP server hardening (deep-review 002)

Component: serve/. Acceptance: SERVE-1/2 closed (High/conditional-High); SERVE-3/3b/4/5/6/7, SSRF-1, HF-1 closed (Medium/Low/Info). Decision rationale: mirror the traversal defense already correct on the HF/OCI download path (`pull.go:198-209`) rather than inventing a new pattern.

- [x] T143.1 Fix SERVE-2: LoRA adapter-name path traversal  Owner: agent  Est: 2h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-03, PR #942: anchored regex + containment check mirroring pull.go's pattern)
  - `serve/adapter.go:33,44` splits the `model` field on `:` and joins the remainder verbatim into a file path with no validation; `filepath.Join` cleans `../` so a crafted name escapes the adapter directory. Fix: an anchored name regex (`^[A-Za-z0-9_-]{1,64}$`) plus a `filepath.Clean` + prefix-containment check, mirroring `model/registry/pull.go:198-209`.
- [x] T143.2 Fix SERVE-1: bound error-metric label cardinality  Owner: agent  Est: 2h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-03, PR #941: normalizeRoute built from the actual mux routes; also fixed a residual leak in the review's own sketch where /v1/models/{id} would still echo the attacker-chosen id)
  - `serve/metrics.go:93-95`'s `RecordError` encodes the raw, pre-auth `r.URL.Path` into a permanent counter name (`server.go:366`, outside `authMiddleware`). Fix: normalize to a matched route template, collapsing unknown paths to a single `"other"` bucket.
- [x] T143.3 Fix SERVE-3: cap /v1/embeddings input array size  Owner: agent  Est: 1h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #954: maxEmbeddingsBatch=256 matching the existing classify/guard cap style; also fixed the test model to carry real embedding weights so the 200 path was actually exercised, not just 500s)
  - `handlers.go:342-400` has no element-count cap while `/v1/classify`/`/v1/guard/batch` cap at 256. Fix: mirror `maxClassifyBatch = 256`.
- [x] T143.4 Fix SERVE-3b: cap per-request image-fetch fan-out  Owner: TBD  Est: 2h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #967: cap images/request (16), 64MB shared byte budget, 60s wall-clock bound on the fan-out)
  - `handlers.go:99-110` -> `vision.go:245`: a 10 MB body can carry thousands of `image_url` entries, each fetched sequentially (up to 20 MB, 30s timeout). Fix: cap images per request (<=16), cap total decoded bytes, bound concurrency and overall wall-clock via `r.Context()` deadline.
- [x] T143.5 Fix SERVE-4: GuardianMiddleware unbounded body read  Owner: TBD  Est: 1h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #964: http.MaxBytesReader (10MB) + SSE passthrough (no buffering of text/event-stream))
  - `guardian_middleware.go:42,95-126` does `io.ReadAll(r.Body)` with no cap and buffers the full response, breaking SSE. Latent (not wired into `NewServer` today, but fix before it is). Fix: `http.MaxBytesReader` before read; skip response buffering for `text/event-stream`.
- [x] T143.6 Fix SERVE-5: validate sampling parameter bounds  Owner: TBD  Est: 1h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #969: reject max_tokens<=0, clamp temperature<=2)
  - `handlers.go:59-62` only clamps the upper bound of `max_tokens`; `temperature` (`types.go:167-182`) has no upper bound. Fix: reject `max_tokens <= 0`; clamp `temperature <= 2`.
- [x] T143.7 Fix SSRF-1: extend the SSRF IP blocklist  Owner: TBD  Est: 1h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #961: isBlockedIP now blocks IsUnspecified (0.0.0.0/::) and CGNAT 100.64.0.0/10)
  - `vision.go:37-55`'s `isBlockedIP` omits `IsUnspecified()` (`0.0.0.0`/`::`) and CGNAT `100.64.0.0/10`. Fix: add both. The core connect-time SSRF defense is otherwise excellent (keep it as-is).
- [x] T143.8 Fix SERVE-6: stop logging percent-decoded request paths  Owner: TBD  Est: 30m  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #966: log r.URL.EscapedPath() instead of decoded path (log-injection fix))
  - `server.go:376` logs the decoded `r.URL.Path` (log-injection risk). Fix: log `r.URL.EscapedPath()` or strip control characters.
- [x] T143.9 Fix SERVE-7: remove dead validation code; evaluate /metrics gating  Owner: TBD  Est: 1h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #960: removed dead validateImageURL/ssrfValidator (grep-confirmed zero callers); /metrics gating deferred as tracked tech debt (it is intentional, test-asserted behavior))
  - `validateImageURL`/`ssrfValidator` (`vision.go:127,136`) are never called -- remove to avoid a false sense that URL-level validation runs. Separately evaluate gating `/metrics` (`server.go:262`) behind auth or a separate listener; if not done this phase, file as tech debt.
- [x] T143.10 Fix HF-1: support an out-of-band expected hash for HF downloads  Owner: TBD  Est: 2h  verifies: [UC-H2-011]  kind: agent  (done 2026-07-10, PR #962: optional ExpectedHashes pin verified over the server ETag for HF pulls)
  - `pull.go:254-260`'s integrity check is trust-on-first-download (expected SHA comes from the same server's ETag; absent ETag warns and accepts). Fix: accept an optional out-of-band expected-hash parameter/pin that, when provided, is verified instead of trusting the ETag.
- [ ] S143.10.1 API tests for this epic  Owner: TBD  Est: 3h  verifies: [UC-H2-011]  kind: agent  blocked-by: [T143.1, T143.2, T143.3, T143.4, T143.5, T143.6, T143.7, T143.8, T143.9, T143.10]
  - Real-request tests against the HTTP boundary: adapter traversal name rejected (400, no file opened), oversized embeddings/image-fan-out batch rejected, SSE response not buffered by GuardianMiddleware, negative max_tokens rejected, request to `http://0.0.0.0/` blocked by the SSRF dialer.

### E144: CI/CD and supply-chain hardening (deep-review 002)

Component: .github/workflows, deploy/. Acceptance: CICD-1/2 closed (this phase, per Objectives); CICD-3/4/5/6 closed or tracked. Decision rationale: match the least-privilege pattern already used correctly by `benchmark.yml`, `codeql.yml`, `release-please.yml`.

- [x] T144.1 Fix CICD-1: add permissions blocks to PR-triggered workflows  Owner: agent  Est: 1h  verifies: [UC-H2-012]  kind: agent  (done 2026-07-03, PR #938: contents:read added to ci.yml/arm64-build.yml/golden-staleness.yml; no job in any of the three needed a broader scope)
  - `ci.yml`, `arm64-build.yml`, `golden-staleness.yml` run on `pull_request` (build/run PR code) with no `permissions:` block, so jobs inherit the repo/org default `GITHUB_TOKEN` scope. Fix: add `permissions: contents: read` to all three, matching the workflows that already do this correctly.
- [x] T144.2 Fix CICD-2: pin mutable tool/package installs in CI  Owner: agent  Est: 1h  verifies: [UC-H2-012]  kind: agent  (done 2026-07-03, PR #939: govulncheck pinned to v1.5.0, torch/numpy pinned to verified-current versions on the CPU index)
  - `ci.yml:38` installs `govulncheck@latest`; `golden-staleness.yml:15` installs unpinned `torch`/`numpy` via pip. Fix: pin exact versions (or hash-checked installs) for both.
- [x] T144.3 Fix CICD-3: scope the benchmark PR-comment token  Owner: TBD  Est: 1h  verifies: [UC-H2-012]  kind: agent  (done 2026-07-10, PR #959: split comment-posting into a separate job holding the sole pull-requests:write; build job runs contents:read only)
  - `benchmark.yml` grants `pull-requests: write` to the whole job that builds/runs PR code (same-repo branch PRs get the broad write scope; fork PRs already get read-only). Fix: scope the write permission to only the commenting step/job.
- [x] T144.4 Fix CICD-4: digest-pin container base images  Owner: TBD  Est: 1h  verifies: [UC-H2-012]  kind: agent  (done 2026-07-10, PR #963: deploy/aws/Dockerfile base images digest-pinned (@sha256))
  - `deploy/aws/Dockerfile:2,20` pins by mutable tag, not digest. Fix: `FROM ...@sha256:<digest>`.
- [x] T144.5 Fix CICD-5: add .dockerignore  Owner: TBD  Est: 30m  verifies: [UC-H2-012]  kind: agent  (done 2026-07-10, PR #965: .dockerignore at repo build-context root; go build -deps verified nothing needed is excluded)
  - `deploy/aws/Dockerfile:10` `COPY . .` has no `.dockerignore`. Add one excluding `.git`, test fixtures, local scratch.
- [x] T144.6 Fix CICD-6: track the bbolt advisory instead of blanket continue-on-error  Owner: agent  verifies: [UC-H2-012]  kind: agent  (done 2026-07-10, PR #973: scripts/govulncheck-gate.sh allowlists only the no-fix bbolt ID and fails on anything else; removing the blanket swallow surfaced a real reachable HTTP/2 DoS GO-2026-4918 -> bumped golang.org/x/net to v0.53.0 in the same PR; note GO-2026-4923 has since been WITHDRAWN as a false positive)
  - `go.etcd.io/bbolt v1.4.3` has GO-2026-4923 with no fix available; the CI vuln check is currently blanket `continue-on-error`. Fix: add a scoped govulncheck ignore for only that advisory ID, and re-enable failing on any other vulnerability.
- [x] T144.7 SLSA: add artifact signing + SBOM to the release pipeline  Owner: agent  verifies: [UC-H2-012]  kind: agent  blocked-by: [T144.1]  (done 2026-07-10, PR #970: keyless cosign signing of the release SBOM via GitHub OIDC in a least-privilege id-token:write job; SBOM was already wired; documented that wiring GoReleaser to build+sign actual binaries/images is the remaining human follow-up since no built-artifact pipeline exists yet)
  - Build is reproducible-ish (pinned deps, no `replace`) but unsigned with no provenance attestation (~SLSA L1-L2). Add cosign signing + SBOM generation to `release-please.yml`. Time-boxed stretch item: if not completed this phase, file as tracked tech debt rather than extending the phase.

### E145: Security review closeout

Component: docs. Acceptance: tiers 1-2 of deep-review 002 are re-verified closed; unwired defenses are exposed via CLI flags; the review doc and lore are updated.

- [ ] T145.1 Wire unwired security defenses into the CLI  Owner: TBD  Est: 3h  verifies: [UC-H2-008, UC-H2-010]  kind: agent  blocked-by: [T140.1, T142.3]
  - Add `--rate-limit`, `--keystore`, and `--tls-*` flags to `serve`/`worker` (ADR-094's "ship the defense you write" rule) so operators can turn on the already-correct `serve/security/` and `distributed/tlsconfig.go` capabilities without hand-wiring Go code. Document in README/design.md.
- [ ] T145.2 Close deep-review 002 tiers 1-2  Owner: TBD  Est: 2h  verifies: [infrastructure]  kind: agent  blocked-by: [T139.5, T140.3, T141.2, T142.5, T143.10, T144.6, T145.1]
  - Re-verify every High finding has a passing repro test proving the fix (not just that code changed). Append a docs/devlog.md entry summarizing the closeout. Promote the two lore candidates the review flagged via /lore: the four-way-duplicated-loader-guard landmine, and the "security code in serve/security and distributed/tlsconfig.go is real but unwired" invariant. File GitHub issues for any tier-3/tech-debt findings not completed in this phase (SERVE-3/3b/4/5/6/7 residue, SSRF-1, HF-1, CONC-L1, L1/L2/L3, SLSA if T144.7 slipped). Update docs/deep-reviews/002-full-codebase.md's header with a remediation status line.

---

## Parallel Work

| Track | Tasks | Notes |
|---|---|---|
| A: Capture cluster | T133.1 -> T133.2 -> T133.3 -> T133.4 | strictly sequential (contract order); GB10 runs serialize |
| B: gemma4e | T134.1 -> T134.2 | gated on T136.2 (model on host) |
| C: Kernel numerics | T135.1; T135.2 parallel; then T135.3/T135.4/T135.5; T135.6 last | T135.1 unblocks the gate's full-scope green |
| D: Matrix + bench | T136.1; T136.2 (human); then T136.3/T136.4; T136.5 | T136.2 is the external dependency -- surface it to David EARLY |
| E: darwin fix | T137.1 | independent; ztensor repo |
| F: Next plan | T138.1 | after A, C, D, and G-M converge |
| G: Untrusted input | T139.1/2/3 -> S139.3.1; T139.4 -> T139.5 -> S139.5.1 | model/gguf + model/registry; no GPU needed |
| H: Distributed security | T140.1 -> T140.2; T140.1 -> T140.3 -> S140.3.1 | distributed/ only; no GPU needed |
| I: Native lib loading | T141.1 -> T141.2 -> S141.2.1 | internal/*/purego.go only; no GPU needed |
| J: Concurrency | T142.1/2/3/4/5 (parallel) -> S142.5.1 | inference/ + serve/; no GPU needed |
| K: HTTP hardening | T143.1..10 (parallel) -> S143.10.1 | serve/ only; no GPU needed |
| L: CI/CD supply chain | T144.1..7 (parallel) | .github/workflows + deploy/; no GPU needed |
| M: Security closeout | T145.1 (after H, J); T145.2 (after G, H, I, J, K, L, T145.1) | |

GB10 serialization: one GPU pod at a time across ALL tracks (E133 proofs, E135 bisect/oracle runs, E136 parity/bench). The coordinator owns GPU scheduling order: T135.1 first (unblocks gate green), then interleave.

Tracks G-M (deep-review 002 remediation) touch no GPU-dependent code at all (loader/network/CLI/CI-only) and can be dispatched as a fully separate, fully parallel wave set that runs concurrently with the GB10-serial Tracks A-D -- no GB10 scheduling coordination needed. Note some file overlap between Tracks J and K (both touch `serve/handlers.go`); this does not block parallel dispatch (isolated worktrees), but expect a rebase pass when merging.

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
- [x] S133.2.1, T133.3 (chain)  verifies: [UC-H2-003] -- DONE 2026-07-03 (PR #937)
- [x] T135.5 deterministic mode  (after T135.2) -- DONE 2026-07-03 (ztensor#179 + wave-3-task-T135.5; GB10 bitwise proof)
- [ ] T134.2 gemma4e disposition  (after T134.1)
- [ ] T136.3 matrix parity runs  (after T136.1, T136.2)

### Wave 4: Proof + ship (4 agents)
- [x] S133.3.1 GB10 green proof -- DONE 2026-07-03 (PR #937)
- [x] T133.4 remove gate + release + close cluster -- DONE 2026-07-03 (PR #946; #865/#870/#878 closed)
- [ ] T136.4 + S136.4.1 Ollama re-run  (GPU-serial with S133.3.1)
- [x] T135.6 close #847/#921  -- DONE 2026-07-10 (docs/devlog + plan-gpu-training-hardening.md COMPLETE; #847/#921 disposition prepared for coordinator, not executed)

### Wave 5: Surface + plan next (2 agents)
- [ ] T136.5 publish matrix
- [ ] T138.1 plan Phase 2

### Wave Sec-1: Security fixes, tier 1 + top tier 2 (10 agents; no GPU; can run any time in parallel with Waves 1-5) -- 10/10 DONE 2026-07-03
- [x] T139.1 fix F1 GGUF overflow  verifies: [UC-H2-007] -- DONE (PR #945)
- [x] T139.2 fix F2 GGUF offset  verifies: [UC-H2-007] -- DONE (PR #940)
- [x] T140.1 wire mTLS into worker  verifies: [UC-H2-008] -- DONE (PR #947)
- [x] T141.1 remove CWD dlopen  verifies: [UC-H2-009] -- DONE (PR #943)
- [x] T142.1 fix CONC-H1 speculative lock  verifies: [UC-H2-010] -- DONE (PR #948)
- [x] T142.2 fix CONC-H2 delete TOCTOU  verifies: [UC-H2-010] -- DONE (PR #944)
- [x] T143.1 fix SERVE-2 adapter traversal  verifies: [UC-H2-011] -- DONE (PR #942)
- [x] T143.2 fix SERVE-1 metric cardinality  verifies: [UC-H2-011] -- DONE (PR #941)
- [x] T144.1 fix CICD-1 permissions blocks  verifies: [UC-H2-012] -- DONE 2026-07-03 (PR #938)
- [x] T144.2 fix CICD-2 pin installs  verifies: [UC-H2-012] -- DONE (PR #939)

### Wave Sec-2: Security fixes, tier 2 remainder + tier 3 start (10 agents) -- 9/10 DONE 2026-07-10 (T139.5 not yet dispatched, blocked-by T139.4 which is now done)
- [x] T139.3 fix F3 dim cap  verifies: [UC-H2-007] -- DONE (PR #951)
- [x] T139.4 fix OCI-1 digest verify  verifies: [UC-H2-007] -- DONE (PR #949)
- [x] T139.5 fix OCI-2 https-only  (after T139.4)  verifies: [UC-H2-007] -- DONE 2026-07-10 (PR #968, in Wave Sec-3)
- [x] T140.2 worker loopback default + --tls flags  (after T140.1)  verifies: [UC-H2-008] -- DONE (PR #955)
- [x] T140.3 fix DIST-2 coordinator auth  (after T140.1)  verifies: [UC-H2-008] -- DONE (PR #956)
- [x] T141.2 fix CUDA-2 vendor paths  (after T141.1)  verifies: [UC-H2-009] -- DONE (PR #958)
- [x] T142.3 fix CONC-M1 rate-limiter cleanup  verifies: [UC-H2-010] -- DONE (PR #957)
- [x] T142.4 fix CONC-M2 keystore race  verifies: [UC-H2-010] -- DONE (PR #953)
- [x] T142.5 fix CONC-L1 batch goroutine leak  verifies: [UC-H2-010] -- DONE (PR #950)
- [x] T143.3 fix SERVE-3 embeddings cap  verifies: [UC-H2-011] -- DONE (PR #954)

### Wave Sec-3: Security fixes, tier 3 remainder (10 agents) -- 10/10 DONE 2026-07-10 (+ T139.5 leftover from Sec-2)
- [x] T143.4 fix SERVE-3b image fan-out cap  verifies: [UC-H2-011] -- DONE (PR #967)
- [x] T143.5 fix SERVE-4 guardian body cap  verifies: [UC-H2-011] -- DONE (PR #964)
- [x] T143.6 fix SERVE-5 sampling bounds  verifies: [UC-H2-011] -- DONE (PR #969)
- [x] T143.7 fix SSRF-1 blocklist  verifies: [UC-H2-011] -- DONE (PR #961)
- [x] T143.8 fix SERVE-6 log escaping  verifies: [UC-H2-011] -- DONE (PR #966)
- [x] T143.9 fix SERVE-7 dead code + metrics gating  verifies: [UC-H2-011] -- DONE (PR #960; /metrics gating deferred as tracked tech debt)
- [x] T143.10 fix HF-1 hash pin  verifies: [UC-H2-011] -- DONE (PR #962)
- [x] T144.3 fix CICD-3 token scope  verifies: [UC-H2-012] -- DONE (PR #959)
- [x] T144.4 fix CICD-4 digest-pin images  verifies: [UC-H2-012] -- DONE (PR #963)
- [x] T144.5 fix CICD-5 .dockerignore  verifies: [UC-H2-012] -- DONE (PR #965)

### Wave Sec-4: Tests, tech-debt tier, closeout -- DONE 2026-07-10 (S-tests folded into their fix PRs)
- [x] S139.3.1 GGUF fuzz corpus  (after T139.1/2/3)  verifies: [UC-H2-007] -- DONE (PR #972)
- [x] S139.5.1, S140.3.1, S141.2.1, S142.5.1, S143.10.1 -- satisfied inside their fix PRs (#968/#956/#958/#944+#948+#953+#957+#950/#967+#969); T145.2 re-verifies each
- [x] T144.6 fix CICD-6 bbolt tracking  verifies: [UC-H2-012] -- DONE (PR #973, + x/net v0.53.0 bump for GO-2026-4918)
- [x] T144.7 SLSA signing + SBOM  (after T144.1)  verifies: [UC-H2-012] -- DONE (PR #970)

### Wave Sec-5: Closeout (2 agents)
- [ ] T145.1 wire flags into CLI  (after T140.1, T142.3)
- [ ] T145.2 close deep-review 002 tiers 1-2  (after everything above)

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
| M-P1-7 | Security tiers 1-2 closed | E139-E145 | 9 High findings + CICD-1/2 fixed with repro tests; CLI flags wired; review doc status updated |

Estimated wall-clock: 2-4 weeks; the long poles are GB10 serialization and the human-gated T136.2. Surface T136.2 to David in the first status update. Tracks G-M (security) need no GPU and can run entirely concurrently with Tracks A-D, so M-P1-7 should not extend the phase's wall-clock if dispatched alongside Wave 2/3.

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
| R7 | Distributed TLS requirement (T140.1) breaks an existing multi-host deployment that relied on the old no-TLS default | Med | Low | loopback binds are explicitly exempted; document the cert-gen helper and the policy change prominently in T140.2 and release notes |
| R8 | Security fix touches the same file as an in-flight non-security fix (e.g. serve/handlers.go shared by E142/E143 and any parallel Phase-1 work) | Low | Med | isolated worktrees mean no blocking; expect a short rebase pass when merging, not a conflict that stalls a task |
| R9 | Tier-3/tech-debt security findings (SERVE-3b, CONC-L1, SLSA signing, etc.) balloon past the phase's time-box | Med | Med | ADR-093 rule 3 discipline applies here too: file remaining findings as tracked GitHub issues in T145.2 rather than open-endedly extending the phase |

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

### 2026 07 10 (latest) -- Change Summary: Wave Sec-4 done; E135/#847 closed; only Wave Sec-5 closeout remains

- **Wave Sec-4 complete**: S139.3.1 GGUF fuzz corpus + bounded CI fuzz (PR #972; the fuzzer immediately found and relaxed an over-strict empty-tensor-name assertion), T144.6 scoped govulncheck gate (PR #973), T144.7 keyless cosign SBOM signing (PR #970). The per-epic S-tests (S139.5.1/S140.3.1/S141.2.1/S142.5.1/S143.10.1) were satisfied inside their fix PRs rather than re-dispatched; T145.2 re-verifies each.
- **Real vuln surfaced + fixed by the new gate**: removing the blanket `continue-on-error` on the CI vuln check (T144.6) unmasked GO-2026-4918, a reachable HTTP/2 infinite-loop DoS via `golang.org/x/net v0.48.0` (reached through model/registry/oci.go). Bumped x/net to v0.53.0 (pulling x/sys->v0.43.0, x/text->v0.36.0) in the same PR so the gate lands green. Also learned the original bbolt advisory GO-2026-4923 was WITHDRAWN as a false positive; the scoped gate is still the right improvement (no more blanket swallow).
- **E135 closed**: #847 (kernel-numerics umbrella) and #921 (-tags cuda CGo path, dispositioned as documented DGX-only build policy) both closed via T135.6. The fork-parity symbol check (internal/cuda/kernels/symbol_parity_test.go) is in place, red-proofed against the T135.3 .so-drift incident. One residual filed as **#974**: zerfoo's kernel-fork Makefile never got ztensor#143's global `--use_fast_math` removal, so the deployed .so isn't fast-math-free (real GPU follow-up: Makefile port + rebuild + oracle re-run).
- **Deferred as tracked tech debt** (to be filed as issues in T145.2): unauthenticated `/metrics` gating (intentional/test-asserted; risks Prometheus scraping), and T144.4's per-arch (not multi-arch index) distroless digest pin.
- **Remaining = Wave Sec-5 only**: T145.1 (wire --rate-limit/--keystore/--tls-* CLI flags -- all underlying mechanisms now exist), T145.2 (re-verify every High finding's repro test, update docs/deep-reviews/002-full-codebase.md status header, file the deferred tech-debt issues). That closes Objective 6 / D7 and unblocks T138.1 (plan Phase 2).


### 2026 07 10 (later) -- Change Summary: Wave Sec-3 complete (10/10 + T139.5); deep-review 002 tiers 1-3 fixes all landed

- **Wave Sec-3 is 10/10 DONE** plus the T139.5 leftover from Sec-2, all merged: T139.5 OCI-2 https-only (PR #968), T143.4 SERVE-3b image fan-out cap (PR #967), T143.5 SERVE-4 guardian body cap + SSE passthrough (PR #964), T143.6 SERVE-5 sampling bounds (PR #969), T143.7 SSRF-1 unspecified+CGNAT blocklist (PR #961), T143.8 SERVE-6 log-injection fix (PR #966), T143.9 SERVE-7 dead-code removal (PR #960), T143.10 HF-1 out-of-band hash pin (PR #962), T144.3 CICD-3 token scoping (PR #959), T144.4 CICD-4 digest-pinned images (PR #963), T144.5 CICD-5 .dockerignore (PR #965).
- Heavy file overlap this wave (three agents on serve/vision.go, two on serve/handlers.go, two on serve/server.go) -- merged the 7 distinct-file PRs first, then the vision.go/handlers.go cluster; GitHub auto-rebased all cleanly since each agent kept its diff to a distinct region. Post-merge `go build ./... && go vet ./... && go test ./serve/... ./model/...` all clean on main (the Wave Sec-1 collision lesson).
- Deferred as tracked tech debt (per agent recommendation, conservative path): T143.9's `/metrics` auth-gating -- unauthenticated `/metrics` is intentional, test-asserted behavior (scope_auth_test.go) and gating it risks breaking Prometheus scraping; file a follow-up issue in T145.2. T144.4's gcr.io distroless pin is per-arch (amd64) rather than the multi-arch index digest, because this environment has no docker/crane/skopeo -- revisit if multi-arch runtime portability is needed.
- **Remaining for Objective 6 / D7 closeout:** T135.6 (kernel umbrella + fork-parity check, in flight), then Wave Sec-4 (S139.3.1 GGUF fuzz corpus + the per-epic S-tests, T144.6 bbolt advisory, T144.7 SLSA signing) and Wave Sec-5 (T145.1 wire --rate-limit/--keystore/--tls CLI flags, T145.2 re-verify every High finding + update the review doc status header + file tech-debt issues). After that, T138.1 (plan Phase 2) unblocks.


### 2026 07 10 -- Change Summary: Wave Sec-2 complete (9/10); ztensor v1.57.0 released

- **Wave Sec-2 is 9/10 DONE** (deep-review 002 remediation, dispatched as a fully parallel no-GPU wave, same pattern as Sec-1): T139.3 F3 dim cap (PR #951), T139.4 OCI-1 digest verify (PR #949), T140.2 worker CLI loopback+TLS flags (PR #955), T140.3 DIST-2 coordinator auth (PR #956, added a defense-in-depth auth interceptor beyond transport TLS alone), T141.2 CUDA-2 vendor paths + S141.2.1 (PR #958, covered every native loader package: cublas, cudnn, hip, rocblas, miopen, tensorrt, opencl -- not just the kernel library T141.1 fixed), T142.3 CONC-M1 rate-limiter cleanup (PR #957, wired Start/Stop into the server lifecycle so T145.1's future --rate-limit flag has something to call), T142.4 CONC-M2 keystore race (PR #953, chose value-copy-from-Lookup over atomic.Bool after confirming the latter would silently break bbolt JSON persistence), T142.5 CONC-L1 batch goroutine leak (PR #950), T143.3 SERVE-3 embeddings cap (PR #954, also fixed the shared test model to carry real embedding weights so the 200 path was actually exercised). T139.5 (OCI-2 https-only) is now unblocked by T139.4 but was not dispatched in this wave -- pick it up in Sec-3.
- Coordinator ran a full `go build ./... && go vet ./... && go test ./...` on `main` after all 8 merges (per the Wave Sec-1 lesson in devlog.md) -- all clean, no parallel-edit collisions this time.
- Merged the automated release-please PR (v1.57.0, manifest + changelog only, tagged) that had been sitting open since Wave Sec-1.
- Remaining for Objective 6 / D7: T139.5 + S139.5.1, T145.1/T145.2 fully, plus everything in Waves Sec-3/Sec-4/Sec-5 (T143.4-10 + S143.10.1, T144.3-7, T145.1/T145.2 closeout).

### 2026 07 03 (latest) -- Change Summary: Wave Sec-1 complete (10/10); T133.4 closes the capture cluster; E133 fully done

- **E133 (capture/replay cluster) is CLOSED.** T133.4 (PR #946) removed the T129.2 containment gate and un-gated the #878 fixture (drops ZERFOO_UNSAFE_CAPTURE_TRAINING; ZERFOO_RUN_878_FIXTURE stays as a long-test opt-in). The agent's sandbox had curl denied by permission policy, so it correctly stopped short of GB10 validation/issue-closure/release rather than route around the restriction; the coordinator ran `scripts/dgx-validate.sh` directly ({"build":"pass","vet":"pass","cuda_tests":"pass","failures":[]}, fixture SKIPs cleanly with the gate gone) and closed #865 (already closed via T133.1), #870, #878 with fix-summary comments referencing PRs #928/#933/#937. Release-please will tag from the merged conventional commits.
- **Wave Sec-1 is 10/10 DONE** (deep-review 002, dispatched as a fully parallel no-GPU wave): T139.1 F1 overflow (PR #945), T139.2 F2 offset bounds (PR #940), T140.1 DIST-1 mTLS (PR #947), T141.1 CUDA-1 CWD dlopen removal (PR #943), T142.1 CONC-H1 speculative lock (PR #948), T142.2 CONC-H2 delete TOCTOU (PR #944, extended scope to guard.go/classify.go beyond the review's literal enumeration since they shared the same WaitGroup hazard), T143.1 SERVE-2 adapter traversal (PR #942), T143.2 SERVE-1 metric cardinality (PR #941, also fixed a residual leak in the review's own fix sketch), T144.1 CICD-1 permissions (PR #938), T144.2 CICD-2 pinning (PR #939). All 9 High findings from deep-review 002 that were in scope for tier 1/2 are now fixed with repro tests; CICD-1/2 (Objective 6's explicit CICD bar) are fixed.
- Infra note: multiple parallel agents (and the coordinator) hit intermittent `gh` CLI 401s during this wave -- confirmed NOT a real credential or rate-limit issue (token valid, 4997/5000 core rate remaining) but transient macOS Keychain contention under ~10+ concurrent `gh` processes; every occurrence resolved on retry within seconds. No action needed, but worth knowing if it recurs at larger fan-out.
- Remaining for Objective 6 / D7: T139.3/4/5 + S139.3.1/S139.5.1 (OCI + F3), T140.2/3 + S140.3.1 (worker CLI flags + coordinator auth), T141.2 + S141.2.1 (CUDA-2), T142.3/4/5 + S142.5.1 (CONC-M1/M2/L1), T143.3-10 + S143.10.1 (remaining SERVE/SSRF/HF findings), T144.3-7 (remaining CICD + SLSA), T145.1/T145.2 (CLI flag wiring + closeout) -- these are Waves Sec-2 through Sec-5.

### 2026 07 03 (later still) -- Change Summary: T133.3 closes the capture cluster's keystone bug (#878)

- T133.3 + S133.3.1 done (PR #937): root-caused #878 to `buildOnesSeed`'s device-resident `d(loss)=1` seed being the one piece of cross-step captured training state still allocated via the engine's arena pool (`engine.Fill`) -- every other cross-step state (grad accumulators, fused-AdamW moments) was already allocation-stable. Fixed at the contract level by re-homing the seed into a nil-pool `GPUStorage`, mirroring `newPersistentGradTensor`'s existing pattern. Red-proof (sharpened fixture) diverged at step 6 (max |diff| 0.0248) before the fix; after the fix, baseline/eager-reset/capture-reset trajectories are bit-identical on GB10 (pods `...1783116105`/`...1783117013`). Also repaired the merged T129.1 fixture, which had never actually exercised the bug (capture-illegal ReLU fallback, missing `ResetPool` call, too-coarse assertion), and a second `.so` symbol-drift regression from T135.3's rebuild (5 missing symbols: transpose_2d/nd_bf16, dropout_f32, fused_adamw_f32, tiny_batched_gemm_f32 -- ported from ztensor v1.19.2). Added `-env "K=V ..."` plumbing to dgx-validate.sh for env-gated fixtures.
- E133 (capture/replay cluster) now has only T133.4 (remove containment gate, release, close #865/#870/#878) remaining.
- Filed a follow-up task (not in this plan; tracked in session) to automate a fork-parity symbol check between zerfoo's and ztensor's kernel .so builds, so the class of drift T133.3 caught by accident is caught by CI instead.

### 2026 07 03 (later) -- Change Summary: deep-review 002 merged into Phase 1 scope (E139-E145)

- Ran /plan against docs/deep-reviews/002-full-codebase.md (full-codebase security + architecture audit, 9 High findings, 0 Critical, 10 Medium, 7 Low, 3 Info). Added Objective 6 (deep-review 002 tiers 1-2 closed) and Deliverable D7. Added six new epics: E139 (untrusted-input hardening: GGUF loader F1/F2/F3 + OCI-1/2), E140 (distributed-wire security: DIST-1/2), E141 (native library loading: CUDA-1/2), E142 (concurrency correctness: CONC-H1/H2/M1/M2/L1), E143 (HTTP server hardening: SERVE-1..7, SSRF-1, HF-1), E144 (CI/CD supply-chain hardening: CICD-1..6, SLSA), E145 (security review closeout). New Waves Sec-1..Sec-5 added; these tracks (G-M) need no GPU and can run fully in parallel with the existing GB10-serial Tracks A-D.
- Created docs/adr/094-untrusted-boundary-security-hardening.md, capturing the policy: treat the GGUF file and distributed wire as first-class untrusted input, native library loading trusts only vetted absolute paths, the distributed wire fails closed on a routable bind, and "ship the defense you write" (rate limiter/keystore/mTLS must be CLI-reachable, not just correct library code).
- Added six use cases to .claude/scratch/usecases-manifest.json: UC-H2-007 (GGUF loader crash-safe), UC-H2-008 (distributed wire authenticated/encrypted), UC-H2-009 (native lib loading trusted-path-only), UC-H2-010 (concurrency race-free), UC-H2-011 (HTTP resource/traversal bounded), UC-H2-012 (CI/CD supply chain hardened) -- all P0/P1, PLANNED.
- T138.1 (plan Phase 2) now additionally blocked-by T145.2 (security closeout) so Phase 2 planning waits on the security tiers 1-2 being genuinely closed, not just Tracks A/C/D.
- Risk register: added R7 (TLS-requirement deployment break), R8 (file-overlap between E142/E143, non-blocking), R9 (tier-3/tech-debt scope creep, mitigated by ADR-093 rule 3 discipline already in force).
- No trim performed: no completed epics were fully closed out of scope this pass (E133/E135/E136/E137 remain open pending T133.3/T133.4/T134.x/T135.6/T136.x); trimming deferred to when E133-E138 fully close.

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

### SESSION HANDOFF 2026-07-11 (live state for the next session)

Phase 1 security remediation (E139-E145) is essentially complete; only the Wave Sec-5 closeout remains. Pick up here:

- **main is clean at `e1d35140`.** `go build/vet/test ./...` were green after the last merge. Only open PR is release `#952` (chore(main): release 1.57.1) -- release-please, merge when ready.
- **T145.1 (wire --rate-limit/--keystore serve CLI flags) is DONE and up as PR #977 -- MERGE IT.** Code-complete, committed, pushed on branch `wave-sec5-task-T145.1`; `go build ./...`, `go vet ./cmd/cli/... ./serve/...`, `go test ./cmd/cli/... ./serve/...` all green locally (cmd/cli 2.5s). Adds `--rate-limit`/`--rate-limit-burst` -> `serve.WithRateLimiter(security.NewRateLimiter(...))` (burst defaults to ceil(rps)); `--keystore` -> opens scoped bbolt store before model load + `serve.WithKeyStore`, counts as configured auth; registers `bboltCloser` with the shutdown coordinator BEFORE the server (reverse-order shutdown: drain listener -> stop rate-limiter goroutine T142.3 -> close bbolt last). NOTE: `--tls-*` flags already existed pre-task; only rate-limit + keystore were unwired. Next session: review + **rebase-merge** #977, then run the standing full `go build/vet/test ./...` on main, flip the T145.1 box. `refs/claims/T145.1` is still held from this session -- release it (`/claim T145.1 --release`) after merge.
- **T145.2 (deep-review 002 closeout) is BLOCKED on T145.1 merging.** Once #977 merges: re-verify each of the 9 High findings still has a passing repro test on main, update the status header of docs/deep-reviews/002-full-codebase.md to "remediated" with the PR map, and reference the deferred tech-debt issues already filed: **#974** (fast-math Makefile fork-drift), **#975** (/metrics gating, SERVE-7 residual), **#976** (distroless multi-arch index digest, CICD-4 residual). Then flip T145.1 + T145.2 boxes.
- **After Sec-5 closes, T138.1 (plan Phase 2) unblocks** except it also lists T136.5 as a blocker, and T136.5 -> T136.3 -> **T136.2 (human GGUF provisioning on the DGX, still OPEN)**. So either (a) get David to do T136.2, or (b) descope the matrix-dependent blocker from T138.1 and plan Phase 2 now. Per the "plan the next phase and repeat" directive, confirm scope with David before starting Phase 2 planning.
- **Worktree hygiene:** ~40 stale agent worktrees from merged Sec-1..Sec-4 tasks are still on disk (`git worktree list`). Safe to `git worktree prune` / remove the merged ones; keep only agent-a2bf0851ce89533da if you choose to finish T145.1 there. `.claude-checkpoint.*.md` files in the root are untracked scratch -- ignorable.
- **Standing lesson (do not skip):** after any dense same-file parallel wave, run full `go build ./...` on main after merges -- individual PR CI passed green while their COMBINATION broke main's build once (the `math` import collision in model/gguf, commit ca1eb41d).

### Standing notes

1. Read docs/product-strategy-2026-H2.md first; ADR-093 is the decision record. docs/lore.md is mandatory pre-debugging reading for anything touching capture, arena, dst, or the GB10.
2. The ONE external dependency is T136.2 (GGUF provisioning on the DGX host) -- a human task. Ask David early; do not let Tracks B/D silently stall on it.
3. GPU runs: `scripts/dgx-validate.sh` is the standing gate (supports -pkgs for targeted runs, -keep/-delete for pod inspection, -dry-run). One pod at a time. Known footgun: zero-matched tests count as pass (guard added in T135.1).
4. The #878 fixture is the phase's keystone proof: tests/training/capture_replay_divergence_878_test.go, gated by ZERFOO_RUN_878_FIXTURE=1 (+ ZERFOO_UNSAFE_CAPTURE_TRAINING=1 until T133.4 removes the gate). Expected RED on GB10 today; the phase exits with it GREEN.
5. docs/plan-gpu-training-hardening.md holds the authoritative T3.x/T4.x detail for E135; check its boxes as you go and mark it COMPLETE in T135.6.
6. Phase 0's full record: git history of this file (b718cd07 and earlier), three devlog entries dated 2026 07 02, PRs #912-#920, release v1.56.0.
7. E139-E145 (security) are the task-level breakdown of docs/deep-reviews/002-full-codebase.md; read that doc for the full CWE/CVSS/attack-narrative detail and exact fix diffs behind each task -- the plan tasks summarize file:line and the fix shape, the review doc has the complete reasoning and verification evidence. ADR-094 is the policy decision behind these epics. These tracks need no GPU and should be dispatched alongside, not after, the GB10-bound tracks.

---

## Appendix

- docs/product-strategy-2026-H2.md -- strategy, phases, metrics, kill criteria.
- docs/adr/093 -- strategy decision + one-phase plan scoping. docs/adr/091 -- verification gates. docs/adr/094 -- untrusted-boundary security hardening policy (E139-E145). docs/lore.md -- landmines (L-0001..L-0013).
- docs/plan-gpu-training-hardening.md -- E135 sub-breakdown (T3.2/T3.3/T3.4/T4.1).
- docs/deep-reviews/002-full-codebase.md -- full security/architecture audit behind E139-E145: CWE/CVSS classification, attack narratives, exact fix diffs, verification evidence for every finding.
- Issue clusters: #865/#870/#878 (capture), #757/#766 (gemma4e), #847/#921/#922 (kernels), #572/T86.5.8 (parity), ztensor#171 (darwin).
- .claude/scratch/usecases-manifest.json -- UC-H2-003..012 active this phase (007-012 added for deep-review 002).
