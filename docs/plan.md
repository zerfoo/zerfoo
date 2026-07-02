# Zerfoo Work Plan -- Phase 0: Stabilize the Ground

**Last updated:** 2026 07 02
**Phase:** 0 of 4 (H2 2026)
**Strategy source of truth:** docs/product-strategy-2026-H2.md (read it first) and docs/adr/093-h2-2026-trust-then-traction-strategy.md
**Previous plan:** the 4,472-line plan covering E1-E127 was trimmed on 2026 07 02; it is preserved in git history at commit 0dc963ca. Completed-epic knowledge lives in docs/devlog.md, docs/benchmarks.md, and docs/adr/. Live epics not in this phase are in the Carried-Forward section below.

---

## Context

### How this plan works (read this if you are a new session)

Zerfoo's H2 2026 direction is set by docs/product-strategy-2026-H2.md ("Trust, then Traction", ADR-093). docs/plan.md is deliberately scoped to ONE phase at a time; each phase's plan ends with a task to plan the next phase by reading the strategy doc. Reading order for a fresh session:

1. docs/product-strategy-2026-H2.md -- what we are doing and why (state of the union, strategy, 4-phase roadmap, metrics, backlog dispositions).
2. This file -- the current phase's executable tasks.
3. docs/devlog.md (newest first) -- what happened recently; the 2026 07 02 entry summarizes the review that produced this plan.
4. docs/adr/091, 092, 093 -- the standing verification gates, the LTX-2 scope, and the strategy decision.

### Problem statement

The product review of 2026 07 02 found world-class capability with zero adoption, three correctness debts that block a credible launch, and operational drag that makes the project illegible to new contributors and future sessions: ~75 of 100 open issues are noise, plan/docs contradict reality, and DGX validation work chronically stalls on a solved-but-never-institutionalized cross-compile blocker. Phase 0 clears the ground so Phase 1 (Trust: fix the correctness debt, publish a verified-model matrix) executes on an honest foundation.

### Objectives (Phase 0 exit state)

1. Issue tracker contains only live engineering (~25 issues); noise closed with recorded rationale; deferred work labeled `parked`.
2. #878 (capture-replay training silently produces wrong gradients) is CONTAINED: capture-replay training fails loudly with a clear error until fixed. Shipped in a release.
3. Documentation tells the truth: updates.md current, lore.md exists, ADR index regenerated, design.md consistency fixes, dangling references repaired.
4. A one-command DGX arm64 validation job exists and has run green once, retiring the purego cross-compile blocker class (closes T86.5.8).
5. Phase 1 is planned.

### Non-goals (Phase 0)

- Fixing #878/#870/#865 root causes, gemma4e decode, or kernel numerics -- that is Phase 1. Phase 0 only contains the damage.
- Any launch/GTM work (Phase 2), LTX-2 progress (Phase 3), new features, new backends.

### Constraints and assumptions

- Single GB10 GPU via Spark (http://192.168.86.250:8080); one GPU pod at a time (SPARK_GPU_MAX=1); never benchmark via interactive SSH (see CLAUDE.md Hardware section; 2026 04 07 outage).
- Branch -> PR -> CI green -> rebase-merge; release-please tags releases; pre-commit hook rejects commits mixing top-level directories.
- gh CLI authenticated with rights to close/label issues on zerfoo/zerfoo.
- Go standard library only (no cobra/viper/testify).

### Success metrics

| Metric | Target | Measured by |
|---|---|---|
| Open issues | ~25, all live engineering | gh issue list count + devlog record (T128.5) |
| #878 exposure | 0 silent-corruption paths reachable | loud-fail test red/green (S129.2.1) + patch release tag |
| DGX validation | 1 green run, T86.5.8 closed | Spark pod result JSON in devlog (T131.3) |
| Docs truth | updates.md/adr index/lore.md current | file diffs merged (E130) |
| Phase 1 readiness | plan.md replaced with Phase 1 plan | T132.1 done |

---

## Discovery Summary

Work type: mixed -- operations (E128, E130), engineering (E129, E131), strategy (E132).

Discovery was performed as a four-agent deep review on 2026 07 02 (plan state, devlog+issues, code capability inventory, ADRs+ecosystem). Full findings are in docs/product-strategy-2026-H2.md Parts 1-2 and the devlog entry of 2026 07 02. Key inputs to this phase:

- Issue dispositions (close/park/keep lists): strategy doc Appendix. Derived from reading all 100 open issues.
- #878 technical state: capture-on training runs but losses ascend 10-20x while the identical eager config converges; hypothesis is captured seed/accumulator aliasing (gradAccumulator.seeds built via engine.Fill). The known consumer works around it by disabling capture. Repro shapes [ns,3] and [B*ns,3] documented in the issue.
- DGX validation blocker: purego darwin->linux/arm64 cross-compile fails (runtime.dlopen linknames need cgo); native arm64 build on the GB10 was proven in E96/T96.1.1 but never scripted as a standing job. Stalled dependents: T86.5.8 and the E58/E61/E63/E90 validation backlog (closed as superseded in T128.3 once the standing job exists).
- Engineering use cases for this phase: UC-H2-001 (capture-replay fails loudly when unsafe), UC-H2-002 (one-command DGX arm64 validation). Manifest: .claude/scratch/usecases-manifest.json.

---

## Scope and Deliverables

### In scope

- Bulk issue triage per the strategy doc appendix (legibility is a prerequisite for external contributors in Phase 2 and for future sessions).
- A loud-failure gate on capture-replay training (silent-wrong-answer bugs are the one class that must never reach a user).
- Documentation truth restoration (updates.md, lore.md, ADR index, design.md consistency, dangling refs).
- A Spark-submitted native-arm64 build+test job (unblocks all current and future GPU validation work).
- Planning Phase 1.

### Out of scope

- Root-cause fixes for #878/#870/#865 (Phase 1, where each gets an ADR-091 harness fixture).
- gemma4e decode fix T99.2.2.9 (Phase 1, time-boxed).
- Website, examples, launch posts (Phase 2). LTX-2/E127 tasks (Phase 3).

### Deliverables

| ID | Description | Owner | Acceptance |
|---|---|---|---|
| D1 | Triaged tracker: ~25 live issues, closures commented, parked label applied | TBD | gh issue list matches strategy-doc dispositions; devlog record |
| D2 | Capture-replay loud-fail gate, released | TBD | red/green test; release tagged; #878 updated |
| D3 | Docs truth: updates.md, lore.md, adr/README.md, design.md fixes | TBD | PRs merged; no known doc/reality contradictions remain |
| D4 | scripts/dgx-validate.sh + Spark manifest, one green run | TBD | pod Succeeded; JSON report in devlog; T86.5.8 closed |
| D5 | Phase 1 plan in docs/plan.md | TBD | plan passes /plan structure; references strategy doc Phase 1 |

---

## Checkable Work Breakdown

### E128: Issue tracker hygiene

Component: operations. Acceptance: open-issue set equals the "live engineering" list in the strategy doc appendix; every closure has a comment linking docs/product-strategy-2026-H2.md.

- [x] T128.1 Bulk-close E94 org-fiction issues  2026 07 02  (DONE: 16 closed; #714 spared as real engineering and parked instead)  Owner: TBD  Est: 1h  delivers: [tracker: business-fiction cluster closed]  kind: agent
  - Close #697, #698, #699, #700, #707, #708, #714, #715, #716, #718, #719, #720, #721, #722, #723, #724, #725 with comment: "Closed per ADR-093 / docs/product-strategy-2026-H2.md Appendix: aspirational business planning lives in strategy docs, not the tracker. The parent epic E94 was retracted 2026-04-13."
  - Acceptance: all listed issues closed with the comment; none referenced by live code or plan.
- [x] T128.2 Close stale "(COMPLETE)" epics and handover notes  2026 07 02  (DONE: 18 closed incl. #839, #696; 3 spot-checks confirmed genuinely complete)  Owner: TBD  Est: 1h  delivers: [tracker: completed-epic bookkeeping closed]  kind: agent
  - Close #540, #544, #545, #558, #559, #560, #561, #562, #563, #564, #565, #566, #567, #568, #569, #571 (epics whose titles state COMPLETE), #839 (April session handover, superseded by devlog), #696 (archived E94 epic). Comment: "Closing completed/archived epic bookkeeping per ADR-093 tracker-hygiene pass; history preserved in the issue itself and docs/devlog.md."
  - Acceptance: all listed issues closed; spot-check 3 for genuinely-complete status before closing (if any is NOT complete, leave open and flag in T128.5).
- [x] T128.3 Close won't-fix and superseded issues  2026 07 02  (DONE: 8 closed -- #711 won't-fix; #727/#730/#731/#543/#692 superseded by E131 job; parent epics #541/#679 closed as emptied)  Owner: TBD  Est: 45m  delivers: [tracker: impossible/superseded items closed]  kind: agent
  - Close #711 (500+ tok/s: physically impossible on GB10, roofline ~257 tok/s, devlog 2026 03 19). Close stale DGX-validation residue superseded by the E131 standing job: #727 (T61.3.2), #730/#731 (T63.2.x). Close E58/E90 residuals if their issues exist. Comment cites the superseding mechanism (E131 job) or the physical limit.
  - Acceptance: closures commented; anything ambiguous left open and flagged in T128.5.
- [x] T128.4 Apply `parked` label to deferred-by-design issues  2026 07 02  (DONE: label created; 31 parked incl. E55 epic #522+tasks, #520 T54.3.1, #714; PJRT search had 0 open hits)  Owner: TBD  Est: 45m  delivers: [tracker: parked work labeled, not deleted]  kind: agent
  - Create label `parked` (grey). Apply to: ROCm suite #701-#706, #712 (multi-GPU), #709/#710 (RPi/Jetson), #726 (FP8 E5M2), E125 #802, E126/PJRT issues, E55-related, perf micro-opts #524-#539, #543, #557, #606, #640, #692. Add one comment per issue: "Parked per docs/product-strategy-2026-H2.md (hardware/partner/user-demand gated). Not scheduled for H2 2026."
  - Acceptance: label exists and applied; parked issues excluded from live-count in T128.5.
- [ ] T128.5 Verification sweep and devlog record  Owner: TBD  Est: 45m  delivers: [devlog entry: final tracker state]  kind: agent  blocked-by: [T128.1, T128.2, T128.3, T128.4]
  - Run gh issue list; confirm remaining open issues are the live-engineering set (#878, #870, #865, #847, #757, #766, #754, #751, #750, the #742-#748 cluster pending Phase 1 decisions, #887, #888, E124 residue #773/#774/#796/#799, #767, #733/#734, plus anything T128.2/T128.3 flagged). Record the final list and counts in docs/devlog.md (newest first).
  - Acceptance: devlog entry exists; open count is within 20-30; no closure regretted on spot-check.

### E129: Contain #878 (capture-replay silent gradient divergence)

Component: training. Acceptance: on the released version, no user can silently train with wrong gradients via capture-replay; eager path untouched. Root-cause fix is Phase 1, NOT here. Decision rationale: docs/adr/093-h2-2026-trust-then-traction-strategy.md (silent-wrong-answer classes are contained immediately, fixed at contract level next).

- [ ] T129.1 Reproduce #878 divergence as a small fixture  Owner: TBD  Est: 1.5h  verifies: [UC-H2-001]  kind: agent
  - Build a minimal training loop (synthetic graph, shapes from the issue: [ns,3] logits) that demonstrates capture-on loss ascent vs eager convergence, OR confirm and document the existing repro from the issue. GPU-gated portions t.Skip on non-GPU; the GPU variant runs via the E131 job when available. This fixture becomes the Phase 1 red/green regression proof.
  - Acceptance: fixture committed with a comment linking #878; divergence characterized in the issue.
- [ ] T129.2 Loud-failure gate on capture-replay training  Owner: TBD  Est: 1.5h  verifies: [UC-H2-001]  kind: agent  blocked-by: [T129.1]
  - In training/capture_replay.go (CaptureReplayRunner construction): return an explicit error naming #878 unless ZERFOO_UNSAFE_CAPTURE_TRAINING=1 is set. Error text: "capture-replay training is disabled pending zerfoo#878 (silent gradient divergence); set ZERFOO_UNSAFE_CAPTURE_TRAINING=1 to override at your own risk". Document in the package doc and the design.md capture-training section. Inference-side CUDA graph capture is NOT affected.
  - Acceptance: constructing the runner without the env var errors; with it, behavior unchanged; docs updated.
- [ ] S129.2.1 Tests + lint for the gate  Owner: TBD  Est: 45m  verifies: [UC-H2-001]  kind: agent  blocked-by: [T129.2]
  - Unit tests: default errors with actionable message; env override works; eager trainer unaffected. gofmt, go vet, golangci-lint clean.
- [ ] T129.3 Merge, release, and update #878  Owner: TBD  Est: 30m  verifies: [UC-H2-001]  kind: agent  blocked-by: [S129.2.1]
  - Rebase-merge the PR; let release-please tag the patch release; comment on #878 with the containment release version and a pointer to the Phase 1 fix plan.
  - Acceptance: release tagged; #878 comment posted.

### E130: Documentation truth restoration

Component: docs. Acceptance: no known contradiction between docs and reality remains from the 2026 07 02 review list.

- [ ] T130.1 Refresh docs/updates.md  Owner: TBD  Est: 1h  delivers: [user-facing changelog current through v1.55.1]  kind: agent
  - Append entries (newest first) covering 2026 04 07 -> 2026 07 02: Gemma 4 architecture support landing (with the explicit decode-correctness caveat), E124 layout cleanup + ADR-090 enterprise extraction decision, GPU training hardening campaign outcome (clean GB10 f32 runs, ADR-091 gates), capture-replay training containment (E129), LTX-2 epic start (ADR-092), release cadence v1.43 -> v1.55.1. Source: git log, docs/devlog.md, release notes.
  - Acceptance: updates.md current; no claim exceeds verified reality (no Gemma 4 performance claims).
- [x] T130.2 Create docs/lore.md  2026 07 02  (DONE PR #914: L-0001..L-0012, all sources verified; noted #766 native-Q4_K path now exists behind env flag but lossy default remains)  Owner: TBD  Est: 1.5h  delivers: [greppable landmine/invariant register]  kind: agent
  - Seed with the invariants scattered across devlog/ADRs, one entry each, tagged: SaveForBackward lifetime contract (ztensor ADR-006); dst-form "ops write into dst's storage" contract; host-access ordering on unified memory (ztensor#137); arena reset-epoch frees (ztensor#138); nonCapturableOps registry + capture whack-a-mole pattern (ADR-088/089); capture-replay pointer-stability contract; GB10 unified-memory gotchas ("GPU" tensors may be CPUStorage); Spark-only benchmarks (2026 04 07 outage); "a sentinel satisfiable by unrelated state is worse than no sentinel" (PR #365 false-fix lesson); "reproduce before blaming a recent commit"; GGUF loader re-quantization hazard (Q4_K -> Q4_0, #766); pre-commit hook rejects cross-directory commits.
  - Acceptance: docs/lore.md exists, topic-ordered, each entry has tags and a source link (issue/ADR/devlog date).
- [x] T130.3 Regenerate docs/adr/README.md  2026 07 02  (DONE PR #913: 93 ADRs indexed, 89 Accepted / 3 Proposed / 1 revised; follow-up: ADR-089's own Status field is stale)  Owner: TBD  Est: 1h  delivers: [complete ADR index 001-093 with status]  kind: agent
  - Index all 93 ADR files with number, title, date, and Status (read from each file). Note superseded chains where present. Currently the index stops at 037 and has no status column.
  - Acceptance: index lists every file in docs/adr/; statuses match the files.
- [ ] T130.4 Repair dangling ecosystem references  Owner: TBD  Est: 30m  delivers: [no dangling doc references]  kind: agent
  - zerfoo/CLAUDE.md says "See ../CLAUDE.md for the full ecosystem vision and project map (covers 7 repos)" but that file does not exist. Either write a minimal /Users/dndungu/Code/zerfoo/CLAUDE.md (repo map + one-paragraph ecosystem summary pointing at zerfoo/docs/VISION.md and docs/product-strategy-2026-H2.md) or fix the reference in zerfoo/CLAUDE.md. Note the stale parent-level docs/ folder (Aug 2025 era, pre-extraction) as archived in whichever file is written.
  - Acceptance: every reference in CLAUDE.md resolves.
- [ ] T130.5 Decide and record E76 disposition  Owner: TBD  Est: 30m  delivers: [E76 disposition recorded]  kind: agent
  - E76 (architecture-test allowlist cleanup, #733/#734, #570) is blocked on a bridge-elimination epic that was never filed (88 .Data() calls in timeseries backward bridges). Decision per strategy: park it. Label `parked`, comment with the rationale and the trigger to revive (a timeseries-refactor epic or external contributor interest).
  - Acceptance: issues labeled and commented; no phantom blocker remains in the plan.
- [x] T130.6 design.md consistency pass  2026 07 02  (DONE PR #912: ONNX wording, companion repos incl. ztensor/ztoken, ADR-090 extraction note, strategy pointer; also fixed 1.6 table + 2.3 note)  Owner: TBD  Est: 1.5h  delivers: [design.md matches current doctrine]  kind: agent
  - Fix: section 1.1 and 10.3 ONNX-era wording (GGUF is the sole format; zonnx converts at build time); section 11.1 companion-repo list (add ztensor, ztoken; mark zmf removed); section 29 cloud product (annotate as extracted per ADR-090, pointer to feza-ai/zerfoo-enterprise); add a pointer near the top to docs/product-strategy-2026-H2.md for product direction. Keep design.md Tier-1 (architecture only, no benchmark numbers or debugging findings).
  - Acceptance: the four listed inconsistencies gone; no new operational detail added to design.md.

### E131: DGX arm64 validation job (institutionalize the workaround)

Component: infrastructure. Acceptance: one command submits a Spark pod that builds zerfoo natively on the GB10 and runs the GPU validation suite; one green run recorded; T86.5.8 closed. Root cause being retired: purego darwin->linux/arm64 cross-compile is impossible without cgo, so all GPU validation must build ON the DGX -- proven manually in E96, never scripted.

- [ ] T131.1 Spark pod manifest + native build/test script  Owner: TBD  Est: 3h  verifies: [UC-H2-002]  kind: agent
  - Add docs/bench/manifests/validate-arm64.yaml (pattern: existing patchtst-train.yaml; cgroup limits memory 32Gi, cpu 8, nvidia.com/gpu 1) and scripts/dgx-validate.sh which: fetches the repo at a given ref inside the pod, builds natively (go build ./...), runs the GPU test subset (-tags cuda unit tests + the GPU-capable tests/parity subset when model files are present), emits a JSON pass/fail report to pod logs, and exits nonzero on failure. Unique pod names including the git SHA. Follow CLAUDE.md Spark API conventions; sm_121 kernels dir noted in docs/plan-gpu-training-hardening.md Hand-Off Note 4.
  - Acceptance: script submits, polls to completion, streams logs, propagates exit code; manifest enforces limits.
- [ ] S131.1.1 Lint + dry-run tests for the script  Owner: TBD  Est: 45m  verifies: [UC-H2-002]  kind: agent  blocked-by: [T131.1]
  - shellcheck clean; a -dry-run flag prints the manifest and API calls without submitting; unit-test any Go-side report parsing.
- [ ] T131.2 Document the job as the standing GPU validation gate  Owner: TBD  Est: 45m  verifies: [infrastructure]  kind: agent  blocked-by: [T131.1]
  - Add a "GPU validation" subsection to CLAUDE.md Build & Test and to the design.md CI section: all GPU-dependent acceptance runs through scripts/dgx-validate.sh; interactive SSH remains debugging-only. Note the one-GPU-pod-at-a-time rule.
  - Acceptance: docs merged; contradictory stale guidance removed.
- [ ] T131.3 First green run; close the stalled-validation backlog  Owner: TBD  Est: 1.5h  verifies: [UC-H2-002]  kind: agent  blocked-by: [S131.1.1]
  - Run the job on main. Record the JSON report in docs/devlog.md. Close the T86.5.8 issue (GPU parity on DGX) referencing the run; confirm the T128.3 closures of superseded validation tasks were correct. GPU-serial: coordinate with any Wolf GPU work.
  - Acceptance: pod phase Succeeded; devlog entry with report; T86.5.8 closed.

### E132: Plan Phase 1

- [ ] T132.1 Produce the Phase 1 (Trust) plan  Owner: TBD  Est: 2h  delivers: [docs/plan.md replaced with the Phase 1 plan]  kind: any  blocked-by: [T129.3, T131.3]
  - Run /plan with: docs/product-strategy-2026-H2.md Part 4 Phase 1 as the scope source; this plan's Progress Log as prior state. Phase 1 contents to break down: capture/replay root-cause fixes in contract order #865 -> #870 -> #878 (each with an ADR-091 harness fixture; the T129.1 fixture is the #878 red proof); time-boxed gemma4e fix T99.2.2.9 / #766 (demote gemma4 edge to experimental if it fails -- ADR-093 rule 3); kernel-numerics tail closing #847 (T3.2 fixed-order fp32 reductions, T3.3 oracle-gate remaining kernels, T3.4 fused encoder audit, T4.1 ZTENSOR_DETERMINISTIC -- breakdown in docs/plan-gpu-training-hardening.md); verified-model matrix definition + GB10 runs for ~10 flagship models via the E131 job; Ollama comparison re-run with reproduction manifests. End the Phase 1 plan with a task to plan Phase 2 (Traction).
  - Acceptance: new docs/plan.md passes the /plan structure; every Phase 1 item above is either a task or an explicit deferral with rationale.

---

## Parallel Work

Tracks (independent unless noted):

| Track | Tasks | Notes |
|---|---|---|
| A: Tracker | T128.1-T128.4 then T128.5 | pure gh CLI; fully parallel within wave |
| B: Containment | T129.1 -> T129.2 -> S129.2.1 -> T129.3 | sequential chain; T129.3 waits for release CI |
| C: Docs | T130.1-T130.6 | all six mutually independent |
| D: DGX job | T131.1 -> S131.1.1 -> {T131.2, T131.3} | T131.3 is GPU-serial (one pod at a time) |
| E: Next plan | T132.1 | after Tracks B and D complete |

Sync points: T128.5 gates on all of Track A; T132.1 gates on T129.3 + T131.3. Agents run in isolated worktrees, so file conflicts do not create dependencies; the E128 tasks all mutate GitHub state, so their issue lists must stay disjoint (they are).

### Waves

### Wave 1: Fan-out (9 agents)
- [ ] T128.1 Close E94 org-fiction  delivers: [tracker]
- [ ] T128.2 Close "(COMPLETE)" epics  delivers: [tracker]
- [ ] T128.3 Close won't-fix/superseded  delivers: [tracker]
- [ ] T128.4 Apply parked label  delivers: [tracker]
- [ ] T129.1 #878 repro fixture  verifies: [UC-H2-001]
- [ ] T130.2 Create lore.md  delivers: [lore register]
- [ ] T130.3 Regenerate ADR index  delivers: [ADR index]
- [ ] T130.6 design.md consistency pass  delivers: [design.md truth]
- [ ] T131.1 Spark manifest + validate script  verifies: [UC-H2-002]

### Wave 2: Build-out (6 agents)
- [ ] T128.5 Tracker verification sweep + devlog  (after T128.1-4)
- [ ] T129.2 Loud-fail gate  (after T129.1)
- [ ] T130.1 Refresh updates.md
- [ ] T130.4 Repair dangling references
- [ ] T130.5 E76 disposition
- [ ] S131.1.1 Script lint + dry-run tests  (after T131.1)

### Wave 3: Verify + document (3 agents)
- [ ] S129.2.1 Gate tests + lint  (after T129.2)
- [ ] T131.2 Document the standing gate  (after T131.1)
- [ ] T131.3 First green DGX run + close T86.5.8  (after S131.1.1; GPU-serial)

### Wave 4: Release + plan next (2 agents)
- [ ] T129.3 Merge + release + #878 update  (after S129.2.1)
- [ ] T132.1 Plan Phase 1  (after T129.3, T131.3)

---

## Timeline and Milestones

| ID | Milestone | Member tasks | Depends on | Exit criteria |
|---|---|---|---|---|
| M-P0-1 | Tracker legible | E128 | -- | open issues ~25, devlog record |
| M-P0-2 | #878 contained and released | E129 | -- | loud-fail release tagged, #878 commented |
| M-P0-3 | Docs tell the truth | E130 | -- | all six doc fixes merged |
| M-P0-4 | Standing DGX validation gate | E131 | -- | one green run, T86.5.8 closed |
| M-P0-5 | Phase 1 planned | E132 | M-P0-2, M-P0-4 | new plan.md in place |

Estimated wall-clock: Waves 1-2 in 2-3 working days, Waves 3-4 in 2-3 more (release CI and single-GPU serialization are the long poles). Phase 0 total: under 2 weeks, matching the strategy doc.

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Bulk-close accidentally closes a live issue | Med | Med | closures follow the pre-reviewed lists in the strategy doc appendix; T128.2 spot-checks; T128.5 verification sweep; closing is reversible |
| R2 | Loud-fail gate breaks a legitimate capture-training consumer | Med | Low | env-var override documented in the error text; the known consumer already runs capture-off (per #878); release notes call it out |
| R3 | DGX job contends with Wolf GPU work | Med | Med | SPARK_GPU_MAX=1 serializes; T131.3 coordinates timing; the job is short (build + tests, not benches) |
| R4 | Scope creep: agents start "fixing" #878/#870 root causes in Phase 0 | Med | Med | task text forbids it; Phase 1 owns fixes; reviewer checks PR scope |
| R5 | Native DGX build script hits toolchain gaps (Go version, kernel .so paths) | Med | Med | E96/T96.1.1 proved the manual path; the script mirrors it; -dry-run flag for iteration; sm_121 kernels dir documented in plan-gpu-training-hardening.md Hand-Off Note 4 |
| R6 | Session context loss between waves | Low | Med | this plan + strategy doc + devlog entry are self-contained; ADR-093 records the decision chain |

---

## Operating Procedure

1. Definition of done per task: acceptance criteria met; tests written and green; gofmt/go vet/golangci-lint clean; PR rebase-merged with CI green; GPU-dependent acceptance verified by an actual Spark run recorded in devlog; reported honestly (observed, not expected).
2. Small focused commits; never mix top-level directories in one commit (pre-commit hook enforces).
3. All DGX work via Spark; one GPU pod at a time; no interactive-SSH benchmarks, ever.
4. release-please handles tags; do not hand-tag.
5. New ops/kernels (none expected this phase) must pass the ADR-091 gates before merge.
6. Update this plan's checkboxes and Progress Log as tasks complete; measurement results go to docs/devlog.md newest-first.

---

## Carried-Forward Live Epics (NOT in this phase)

Preserved from the previous plan so nothing is lost; do not work these in Phase 0.

- **E127 LTX-2 diffusion (Phase 3 of H2; ADR-092).** ~5/35 done as of 2026 06 17. Landed: GroupNormalization layer, Conv3d/ConvTranspose3d forward-only (groups=1 for transpose), LTX VAE decoder skeleton, bench_gemm + fp8/Q4_K spike manifest, ztensor oracle ops (GroupNorm, CrossAttention, AdaLN, TimestepEmbed). Next CPU-doable per the 2026 06 17 handover: flow-matching Euler scheduler in generate/diffusion/ (T127.2.x), GroupNorm registry wiring, safetensors->GGUF converter in zonnx (T127.3.x; must preserve per-tensor dtype -- checkpoints are mixed F32+BF16+F8_E4M3), weight-accurate VAE decoder (T127.4.4). GB10-gated: fp8/Q4_K GEMM spike (T127.1.0b PART 2), diffusers dev-build provisioning, torch-oracle replays. Landmines: oracle harness lives in ztensor; Conv3d GGUF registration blocked on a "Conv" name collision with Conv2d (needs a rank-dispatch builder); bench binaries must build natively on GB10; parity baseline is PyTorch/ComfyUI on GB10 via Spark, not Ollama. Geometry: 48 DiT blocks, video inner 4096, audio 2048, patch_size 1, FlowMatchEuler 40/8-step, Gemma3-12B text encoder. Issues: #887 (conv backward, deferred), #888 (LTX-2.3 22B geometry gate).
- **E99 / gemma4e decode correctness (Phase 1, time-boxed).** T99.2.2 open since 2026 04 21; queued fix T99.2.2.9 / #766: keep native Q4_K storage for embedding-shaped tensors in the GGUF loader (model/gguf/loader.go decodeQ4KTensor unconditionally re-quantizes Q4_K -> f32 -> Q4_0, doubly lossy for the 262144x8960 ple_embed_tokens gather table), gated ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1. H1-H21 hypothesis history in devlog 2026 04 14 - 04 21. T99.1.3 (capture-on generate verify) is blocked behind it. ADR-093 rule: if this candidate fails, demote gemma4 edge to experimental and move on.
- **#847 kernel-numerics tail (Phase 1).** Open tasks T3.2 (fixed-order fp32 tree reductions), T3.3 (oracle-gate remaining kernels), T3.4 (fused encoder audit), T4.1 (ZTENSOR_DETERMINISTIC) in docs/plan-gpu-training-hardening.md, which remains the authoritative breakdown for that campaign's remainder.
- **E124 residue (Phase 2 good-first-issues).** 5 inline GELU sites deferred pending a Gelu Node constraint change (tensor.Float -> tensor.Numeric); issues #773/#774/#796/#799, #767. Also the human-led push of extracted enterprise packages to feza-ai/zerfoo-enterprise (ADR-090) and the associated major version bump -- scheduled with the Phase 2 launch.
- **E125 mmap remaining work (Phase 3).** #802; blocked on a cudaHostRegister purego binding in ztensor; then layer prefetch, unified-memory fallback, MiniMax-M2 138GB stress test.
- **E126 PJRT (parked).** Blocked on vendoring a CPU plugin .so and a first-token-logits accessor; revive only with a hardware partner (strategy doc, explicit non-goal).
- **E86 T86.5.8 (closed by this phase).** The one remaining PyTorch-parity task (GPU parity on DGX) is retired by T131.3.

---

## Progress Log

### 2026 07 02 -- Change Summary: plan reset to Phase 0 under the H2 strategy

- Performed the trim pass on the previous 4,472-line plan (E1-E127): completed epics removed (preserved in git history at commit 0dc963ca); Tier-2 knowledge captured in docs/adr/093-h2-2026-trust-then-traction-strategy.md (NEW); Tier-3 findings appended to docs/devlog.md (2026 07 02 entry); live-epic handover content condensed into the Carried-Forward section above.
- Created this Phase 0 plan from docs/product-strategy-2026-H2.md Part 4 Phase 0: E128 tracker hygiene (T128.1-T128.5), E129 #878 containment (T129.1-T129.3), E130 documentation truth (T130.1-T130.6), E131 DGX arm64 validation job (T131.1-T131.3), E132 plan Phase 1 (T132.1).
- ADRs created: docs/adr/093-h2-2026-trust-then-traction-strategy.md (strategy decision; also establishes plan.md one-phase scoping).
- Use-case manifest updated: UC-EXT-01..03 marked WIRED (completed via E91); UC-H2-001 (capture-replay loud fail) and UC-H2-002 (one-command DGX validation) added as PLANNED.
- Why: the 2026 07 02 product review (docs/product-strategy-2026-H2.md Parts 1-2) found launch-blocking correctness debt, a noise-dominated tracker, and doc/reality drift; Phase 0 clears the ground before the Phase 1 Trust work.

---

## Hand-off Notes

1. Read docs/product-strategy-2026-H2.md before touching anything -- it is the source of truth for H2 2026 priorities, non-goals, and the issue dispositions this plan executes. ADR-093 is the decision record; the devlog entry of 2026 07 02 is the evidence trail.
2. This plan covers Phase 0 only. When it completes, T132.1 produces the Phase 1 plan; do not start Phase 1 fixes (capture root causes, gemma4e, kernel numerics) early -- containment and hygiene first.
3. Hardware: single GB10 via Spark at http://192.168.86.250:8080 (CLAUDE.md has API conventions and the outage history that mandates Spark-only). One GPU pod at a time, coordinated with Wolf.
4. Cross-repo: hard bugs historically live at the zerfoo/ztensor boundary (dst contract, arena lifetime, stream ordering). ztensor is at v1.19.0; releases ship in dependency order (ztensor then zerfoo) via release-please.
5. The previous plan's full history (E1-E127, all progress logs) is at git commit 0dc963ca -- consult it rather than re-deriving; docs/plan-gpu-training-hardening.md remains a live standalone plan for the #847 remainder.
6. Repo conventions: rebase-and-merge only; no cobra/viper/testify; pre-commit hook rejects cross-directory commits; never add AI attribution to commits.

---

## Appendix

- docs/product-strategy-2026-H2.md -- H2 strategy, roadmap, metrics, kill criteria, full issue dispositions.
- docs/adr/093-h2-2026-trust-then-traction-strategy.md -- decision record for the strategy and the one-phase plan scoping.
- docs/adr/091-gradcheck-pytorch-oracle-verification.md -- the standing per-op verification gates every Phase 1 fix must use.
- docs/adr/092-ltx2-diffusion-dit-first.md -- LTX-2 scope (Phase 3).
- docs/plan-gpu-training-hardening.md -- authoritative remainder breakdown for #847 (Phase 1 input).
- docs/devlog.md 2026 07 02 entry -- the review findings that produced this plan.
- .claude/scratch/usecases-manifest.json -- use-case registry (UC-H2-001, UC-H2-002 active this phase).
- Issue clusters referenced: #878/#870/#865 (capture lifetime), #757/#766 (gemma4e), #847 (hardening umbrella), #887/#888 (LTX-2).
