# Zerfoo Work Plan -- Phase 1: Trust (closeout)

**Last updated:** 2026 08 14
**Phase:** 1 of 4 (H2 2026) -- Phase 0 complete 2026 07 02; Phase 1 opened 2026 07 02, now in closeout
**Strategy source of truth:** docs/product-strategy-2026-H2.md (read it first) and docs/adr/093-h2-2026-trust-then-traction-strategy.md
**Prior record:** this file's git history (commit 3fffa972 and earlier) holds the full Phase 0 and Phase 1 task-level history; docs/devlog.md holds the investigation record; see Trim Note below.

---

## Trim Note (2026 08 14)

This plan was trimmed on 2026 08 14: completed epics E133 (capture/replay cluster), E134 (gemma4e disposition), E135 (kernel numerics, #847), E137 (darwin fix), and E139-E145 (deep-review 002 security remediation, Objective 6 / D7) were removed along with Waves 1-5 and Sec-1..Sec-5. Their knowledge is preserved as follows:

- Task-level history and per-PR annotations: git history of docs/plan.md (commit 3fffa972 and earlier).
- Investigation record: docs/devlog.md entries 2026 07 02 through 2026 08 10 (newest first).
- Decisions: docs/adr/091 (verification gates), docs/adr/093 (strategy), docs/adr/094 (untrusted-boundary hardening).
- Security remediation evidence: docs/deep-reviews/002-full-codebase.md (Remediation Status header maps finding -> PR -> repro test).
- Landmines: docs/lore.md L-0001..L-0017.
- Kernel-hardening sub-breakdown: docs/plan-gpu-training-hardening.md (marked COMPLETE).

No new Tier-1 (design.md) or Tier-2 (ADR) knowledge required routing: all stable knowledge from the completed epics was already captured in the docs above at completion time.

---

## Context

### How this plan works (read this if you are a new session)

Zerfoo's H2 2026 direction is set by docs/product-strategy-2026-H2.md ("Trust, then Traction", ADR-093). docs/plan.md is scoped to ONE phase at a time; each phase's plan ends with a task to plan the next phase. Reading order for a fresh session: (1) the strategy doc, (2) this file, (3) docs/devlog.md newest-first, (4) ADR-091/093/094, (5) docs/lore.md before debugging anything.

**This session's sandbox may BE the DGX host** (`hostname` = `aitopatom-bfc8`, GB10 GPU). Probe with `hostname` and `nvidia-smi` before assuming you need SSH or that GPU tests are unavailable. Two prior sessions wrongly claimed the opposite; see devlog 2026 08 10.

### Problem statement (what remains, per the 2026 08 14 audit)

Phase 1 ("make every public claim true") is ~90 percent complete: the capture/replay cluster, gemma4e disposition, kernel-numerics tail, darwin fix, and the full deep-review 002 security closeout are all done and merged. What remains is the phase's core deliverable and its surrounding hygiene:

1. **The verified-model matrix is not run.** T136.3/T136.4 were dispatched 2026 08 11; both worker sessions died on wait-states and delivered nothing. docs/verified-models.md still carries 4 `pending T136.3` markers. Worse, the dead lane flagged a suspected **vacuous-parity hazard**: models are staged flat in `/var/lib/zerfoo/models`, and a model-lookup path that resolves "first .gguf in the directory" (the `findGGUF` pattern, inference/inference.go:317) could make every parity subtest silently load the same alphabetically-first file, producing a fake-green matrix. This MUST be ruled out or fixed before any parity run counts.
2. **Public claims exceed the evidence** (violates ADR-093 rule 1): README claims "41 model architectures" and "faster than Ollama on all 4 benchmarked models" while verified-models.md verifies ~5 rows on March-era v1.38.4 numbers; docs/VISION.md still carries the 300/500+ tok/s targets that ADR-093 declared physically impossible on GB10 (roofline ~257) and banned.
3. **Cross-repo limbo:** ztensor PR #179 (ZTENSOR_DETERMINISTIC, the T135.5 deliverable) was never merged -- the feature exists only on branch `feat-deterministic-mode`; release-please PRs sit unmerged in ztensor (#178, since 2026 07 03) and zmf (#14, since March); zmf branch `fix/attribute-tensor` holds 4 unmerged commits; ztensor#180 (GQA fused-kernel SIGSEGV) is open with the zerfoo perf path disabled as mitigation.
4. **Known GPU bugs filed with repros, unfixed:** zerfoo#981 (KV-cache multi-head key offset), #982 (intermittent Gather cudaMemcpy failure), #983 (PatchTST GPU training convergence).
5. **Workspace residue on the DGX host:** an empty locked worktree from the dead parity lane, a stale ztensor checkout on an obsolete April branch, two attic clones with unmined stashes, stray bundles.

Exit state unchanged: zero known silent-correctness bugs, matrix published, benchmarks reproducible, claims matching evidence -- the preconditions for Phase 2 (launch).

### Objectives (Phase 1 exit state -- remaining)

1. Verified-model matrix published from a harness proven to load the right file per row (T136.6 gate); Ollama comparison re-run with reproduction manifests; T86.5.8 closed. (Objective 4 of the original phase plan; all others are closed.)
2. Every public claim (README, VISION.md, design.md, updates.md) is backed by the matrix or removed (ADR-093 rule 1).
3. Cross-repo release state is clean: ztensor#179 merged or honestly re-scoped, release-please PRs landed, orphan branches dispositioned.
4. zerfoo#981/#982/#983 fixed or honestly annotated on the affected matrix rows; ztensor#180 fixed and the fused GQA path re-enabled (or explicitly carried as tracked debt with the mitigation documented).
5. Phase 2 (Traction) is planned.

### Non-goals (Phase 1)

- Launch/GTM work: website, examples, posts (Phase 2). LTX-2/E127 and mmap/E125 (Phase 3). New backends, new model classes, perf moonshots (parked).
- Llama 4 Scout (~65GB) and MiniMax-M2 (~129GB) provisioning: deferred by cost, not blocked; matrix ships with 9 GPU rows + honest absence notes. Use docs/lore.md L-0016 download recipe if picked up.
- #974 (fast-math residue in zerfoo's kernel Makefile): tracked issue, not phase-blocking.

### Constraints and assumptions

- Single GB10 via Spark; ALL GPU runs through `scripts/dgx-validate.sh` or bench manifests; one GPU pod at a time (SPARK_GPU_MAX=1); no interactive-SSH workloads.
- 9 GGUF models verified intact in /var/lib/zerfoo/models (33GB, re-checked 2026 08 14): Gemma 3 1B/4B, Gemma 4 Edge (Q4_K_M + Q8_0), Llama 3.2 3B, Mistral 7B, Qwen 2 7B, Phi-4 Q4_K, DeepSeek-R1-Distill-1.5B.
- Cross-repo: ztensor releases before zerfoo (dependency order) via release-please.
- **Dispatch lesson from the dead 2026 08 11 lane:** parity/benchmark work should run IN-SESSION on the DGX host (submit pod, poll with bounded retries, write results to disk incrementally), never as a fire-and-wait worker that idles on a wait-state. Checkpoint results per-model, not at the end.

### Success metrics

| Metric | Target | Measured by |
|---|---|---|
| Harness honesty | per-row model identity proven | T136.6 test red-proofs the flat-dir hazard, then green |
| Verified matrix | 9 GPU rows + honest gaps published | docs/verified-models.md, zero `pending T136.3` markers |
| Benchmarks | reproducible at current versions | manifests in docs/bench/; benchmarks.md refreshed |
| Claim consistency | zero claims exceeding the matrix | grep sweep of README/VISION/design/updates finds no unbacked counts or comparisons |
| Cross-repo limbo | 0 stale release/feature PRs | ztensor #178/#179, zmf #14 merged or closed with rationale |
| Known-bug honesty | #981/#982/#983 fixed or annotated | issue state + matrix row annotations |
| Phase 2 | planned | docs/plan.md replaced, ends with plan-Phase-3 task |

---

## Discovery Summary

Work type: engineering (tasks carry verifies:) except T150.2 (operations, human) and T138.1 (strategy, kind: plan). example-app is NOT on PATH in this environment: no acc: fields; free-text acceptance criteria govern.

Inputs from the 2026 08 14 audit (seat audit, hq brain/zerfoo-audit-2026-08-14.md; key facts restated here so this plan is self-contained):

- The 2026 08 11 T136.3 lane (sessions zerfoo-02/zerfoo-03) died on wait-states, delivered no PR. zerfoo-02's parting observation: flat models dir + first-alphabetical GGUF resolution risks all parity subtests loading the same file. `findGGUF` (inference/inference.go:317) returns the first .gguf in a directory; audit of how tests/parity resolves each matrix row's file is the first step of T136.6.
- ztensor local+remote state: PR #179 open (feature branch only), #178 release-please open since 2026 07 03 (main carries unreleased fixes), ~40 stale wave-* branches (rebase-merged history means `--merged` shows none; pruning needs per-branch content check), issue #180 open (RepeatInterleaveF32 null-pointer CUDA launch, SIGSEGV, any GQA model; zerfoo mitigated by disabling the fused path in layers/attention/grouped_query_attention.go).
- zmf: branch `fix/attribute-tensor` = 4 unmerged commits (Attribute_Tensor protobuf + Q4_0/Q8_0 DataType enums, March); zonnx has an untracked `cmd/zmf-quantize/main.go` spike (March 5) that likely depends on those enums; zmf release-please #14 (0.6.0) open since March.
- DGX-host workspace: `~/Code/zerfoo/` is canonical (go.work over zerfoo/ztensor/zonnx/zmf/metee). Residue: locked empty worktree `.claude/worktrees/t136-3-parity` (zero commits over main), stray `.claude-checkpoint.e11b361c-121.md`, ztensor checkout on obsolete `chore/nccl-purego` (content merged to main under rebased SHAs) with 3 March stashes, `~/Code/_attic/zerfoo-home-102dirty-15stash` (41 commits ahead of upstream, 15 stashes, gemma4e-era work now mostly obsolete) and `~/Code/_attic/zerfoo-stale-726behind` (9 stashes), `~/zerfoo-*.bundle` files.
- Use cases: UC-H2-004 (matrix) remains PLANNED; UC-H2-013 (public claims never exceed verified evidence) added this pass. UC-H2-003/005/006/007..012 delivered by the completed epics. Manifest: .claude/scratch/usecases-manifest.json.

---

## Scope and Deliverables

| ID | Description | Owner | Acceptance |
|---|---|---|---|
| D4 | Verified-model matrix + refreshed benchmarks (carried) | TBD | verified-models.md live with zero pending markers; benchmarks.md re-run at current versions with manifests; T86.5.8 closed |
| D6 | Phase 2 plan (carried) | TBD | docs/plan.md replaced, ends with plan-Phase-3 task |
| D8 | Trust surface consistent | TBD | README/VISION/design/updates claims match the matrix; VISION.md throughput targets amended per ADR-093 |
| D9 | Cross-repo release hygiene | TBD | ztensor #179 + #178, zmf #14 resolved; orphan branches dispositioned; stale branches pruned |
| D10 | GQA fused path restored or tracked | TBD | ztensor#180 fixed + released + zerfoo re-enabled with GB10 proof, OR documented as tracked debt with mitigation noted in the matrix |
| D11 | DGX workspace clean | TBD | residue removed; attic disposition decided by David |

---

## Checkable Work Breakdown

### E136: Verified-model matrix + reproducible benchmarks (remaining) -- fidelity: executable

Component: docs + tests/parity + bench. The public support claim becomes this matrix (strategy doc P1). Completed: T136.1 (schema, 2026 07 02), T136.2 (9/11 models staged, 2026 08 10).

- [ ] T136.6 Harness honesty: prove each parity row loads its own model file  Owner: TBD  Est: 3h  verifies: [UC-H2-004]  kind: agent
  - Audit how the parity suite resolves each matrix model's GGUF against the flat /var/lib/zerfoo/models directory. If any path funnels through directory-scan resolution (`findGGUF`, inference/inference.go:317, or a glob), replace it with explicit per-row filenames (matrix row -> exact file path table, checked into tests/parity or docs/bench/manifests). Add an identity assertion to the parity runner: for each row, assert the resolved absolute path matches the row's declared file AND log the GGUF header's architecture string; fail the subtest on mismatch.
  - Acceptance: a red-proof first -- demonstrate (or conclusively refute, with the resolution code path quoted in the PR) that two different matrix rows pointed at the flat dir can load the same file; then the identity assertion makes that class impossible. CI-runnable without GPU (path resolution is host-side).
- [ ] S136.6.1 Tests + lint  Owner: TBD  Est: 1h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.6]
  - Unit test for the resolver (unknown row -> error, not fallback; duplicate resolution across rows -> error). gofmt/vet/lint clean.
- [ ] T136.3 Run the parity subset for the matrix on GB10; close T86.5.8  Owner: TBD  Est: 4h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.6]
  - Standing gate with models mounted: parity stage runs for every matrix model present, using the T136.6 explicit-path table. Record per-model results in docs/verified-models.md AS EACH MODEL COMPLETES (checkpoint continuously -- the 2026 08 11 lane died with zero results banked). Close T86.5.8 (#572 epic if fully satisfied) referencing the run.
  - Acceptance: verified-models.md has zero `pending T136.3` markers; every GPU row cites pod ID + date + commit; rows for the 2 unfetched models and Chronos-2 carry honest absence notes.
- [ ] T136.4 Re-run the Ollama comparison with reproduction manifests  Owner: TBD  Est: 1d  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.6]
  - Current zerfoo release vs current Ollama on the GB10 via Spark bench manifests (bench-spark.sh / bench_tps), 3-run medians, same models as docs/benchmarks.md. Note in benchmarks.md that the fused GQA RepeatInterleave path is DISABLED pending ztensor#180 (numbers reflect what users get today); do not wait for E148. Update benchmarks.md (keep history); fix the 14%-vs-28% inconsistency in docs/distribution/ drafts with the fresh numbers. GPU-serial with T136.3.
- [ ] S136.4.1 Devlog + benchmarks.md entries  Owner: TBD  Est: 1h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.4]
- [ ] T136.5 Surface the matrix  Owner: TBD  Est: 2h  verifies: [UC-H2-004, UC-H2-013]  kind: agent  blocked-by: [T136.3]
  - README section + design.md pointer replace architecture-count claims with the matrix link; updates.md entry.

### E146: Trust-surface reconciliation (ADR-093 rule 1) -- fidelity: executable

Component: docs. Acceptance: no public claim exceeds docs/verified-models.md.

- [ ] T146.1 Amend docs/VISION.md throughput targets and architecture counts  Owner: TBD  Est: 2h  verifies: [UC-H2-013]  kind: agent
  - Replace the Year-1 "300+ tok/s" and Year-3 "500+ tok/s" targets with the ADR-093 ruling (GB10 roofline ~257 tok/s; targets restated as roofline-relative or hardware-conditional). Replace static architecture counts with a pointer to the matrix. Add a "targets amended 2026-08-14 per ADR-093" note preserving the original ambition language. Do NOT rewrite the thesis -- ADR-093 explicitly upholds it.
- [ ] T146.2 Claim-consistency sweep  Owner: TBD  Est: 2h  verifies: [UC-H2-013]  kind: agent  blocked-by: [T136.5]
  - Grep README.md, docs/design.md, docs/updates.md, docs/distribution/ for: architecture counts (6/40/41/45 variants), "faster than Ollama", tok/s figures, model-family claims. Every hit either cites the current matrix/benchmarks.md or is removed. Record the sweep command + hit list in the PR body so it is repeatable at Phase 2 launch.

### E147: Cross-repo release and PR hygiene -- fidelity: executable

Component: ztensor, zmf, zonnx repos. Acceptance: D9. Dependency order: ztensor releases before zerfoo bumps.

- [ ] T147.1 Resolve ztensor#179 (ZTENSOR_DETERMINISTIC)  Owner: TBD  Est: 3h  verifies: [UC-H2-004]  kind: agent
  - The T135.5 deliverable exists only on branch feat-deterministic-mode (PR open since 2026 07 03; main has moved). Rebase on ztensor main, re-run the GB10 bitwise-identical proof (2 seeded runs, per-epoch losses identical; honest exclusion note for the FusedEncoderBackward atomicAdd path stays), merge. If the rebase surfaces conflicts beyond 2h of work, instead re-scope honestly: comment on #179 with the state, and correct the T135.5 completion claim in this plan's git history via a devlog note (the plan said "done"; the honest status is "implemented, unmerged").
  - Acceptance: #179 merged with fresh GB10 proof cited, OR #179 commented + devlog correction landed. No silent limbo.
- [ ] T147.2 Land the release-please PRs; release; bump  Owner: TBD  Est: 2h  verifies: [infrastructure]  kind: agent  blocked-by: [T147.1]
  - Merge ztensor #178 (or its successor after #179 lands) -> ztensor release tagged; merge zmf #14 (0.6.0). Bump zerfoo's ztensor dependency to the new tag; go build/vet/test -short green; standing gate green if kernel code moved.
- [ ] T147.3 Disposition zmf fix/attribute-tensor + zonnx zmf-quantize spike  Owner: TBD  Est: 2h  verifies: [infrastructure]  kind: agent
  - zmf branch fix/attribute-tensor (4 commits: Attribute_Tensor proto, Q4_0/Q8_0 enums) and zonnx untracked cmd/zmf-quantize/main.go are one abandoned March quantization thread. Decide as a unit: if the Q4_0/Q8_0 enums are needed by any current GGUF/ZMF path, rebase + PR + merge both; otherwise close the branch with a pointer comment and delete the untracked spike (paste its 1137 bytes into the closing comment first -- nothing is lost). Also remove zonnx's stray zonnx-converter.log and sync the local zonnx checkout to origin/main.
- [ ] T147.4 Prune stale branches (zerfoo + ztensor)  Owner: TBD  Est: 2h  verifies: [infrastructure]  kind: agent  blocked-by: [T147.1, T147.2]
  - zerfoo: delete origin/diag/gelu-internal-trace, origin/diag/ln-backward-trace after confirming their commits are merged or obsolete; disposition origin/feat-T131.1-dgx-validate (2 commits from 2026 07 02: dgx-validate.sh submit script + validate-arm64 manifest -- check whether equivalent functionality already landed on main via other PRs; merge or close with rationale). ztensor: for each of the ~40 wave-*/feat-*/fix-*/diag-* branches, `git cherry origin/main <branch>` to detect unmerged content; delete fully-landed ones, list survivors with one-line rationale in the PR/issue. Keep origin/handover (canonical session notes).
  - Acceptance: branch count reduced with a receipts list (branch -> disposition) posted as a gist or issue comment; zero branches deleted that carried unmerged non-obsolete work.

### E148: ztensor#180 -- GQA fused-kernel SIGSEGV -- fidelity: executable

Component: ztensor internal CUDA kernels + zerfoo layers/attention. Acceptance: D10. The bug: RepeatInterleaveF32 launches with a null-pointer CUDA call, SIGSEGV, crashes any GQA model (Llama/Mistral/Qwen/Gemma) on GPU via the fused path; zerfoo currently mitigates by disabling the fused path (PR #980). Full repro in the issue.

- [ ] T148.1 Fix RepeatInterleaveF32; release ztensor  Owner: TBD  Est: 4h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T147.2]
  - Root-cause via the issue's repro on GB10 (one pod). Fix the kernel launch, add an oracle-gated regression test per ADR-091 (gradcheck + parity for the op). Release ztensor.
  - Acceptance: repro green on GB10; oracle test in ztensor CI; release tagged.
- [ ] T148.2 Re-enable the fused path in zerfoo; GB10 proof  Owner: TBD  Est: 2h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T148.1]
  - Bump ztensor; revert the PR #980 mitigation in layers/attention/grouped_query_attention.go; TestGPUParity_GQA green on GB10 via the standing gate; note the perf-path restoration in benchmarks.md (do NOT retroactively edit T136.4's published numbers -- add a dated line).
  - Time-box: if T148.1 exceeds its estimate by 2x, stop, document findings on the issue, and carry D10 as tracked debt with the mitigation note added to the matrix rows for GQA models (ADR-093 rule 3).

### E149: Filed-bug disposition (#981, #982, #983) -- fidelity: executable

Component: inference (KV cache), internal/cuda (Gather), timeseries (PatchTST). Each issue has a full repro from 2026 08 10. ADR-093 rule 3 applies per bug: ONE time-boxed root-cause attempt, then park with an honest annotation. GPU-serial; schedule after T136.3/T136.4 so the matrix ships first.

- [ ] T149.1 #981 KV-cache multi-head prefill+decode key offset bug  Owner: TBD  Est: 4h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T136.3]
  - Retrieved keys shifted/misplaced after decode append (FP16 GPU multi-head). This is silent-wrong-output class -- highest priority of the three. Fix with an ADR-091 fixture, or park with: which matrix models/configs are affected, and an annotation on their verified-models.md rows.
- [ ] T149.2 #982 intermittent Gather cudaMemcpy invalid-argument  Owner: TBD  Est: 3h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T149.1]
  - Rotating sub-case failure in TestGatherInt64Parity. Fix or park with frequency data + affected-path annotation.
- [ ] T149.3 #983 PatchTST tiny-training GPU convergence failure  Owner: TBD  Est: 3h  verifies: [UC-H2-004]  kind: agent  blocked-by: [T149.2]
  - Fix or park; if parked, the timeseries row/absence note in the matrix states it.

### E150: DGX workspace hygiene -- fidelity: executable

Component: the DGX host filesystem (not the repo). Acceptance: D11.

- [ ] T150.1 Mechanical cleanup of the canonical workspace  Owner: TBD  Est: 1h  verifies: [infrastructure]  kind: agent
  - In ~/Code/zerfoo/zerfoo: `git worktree unlock .claude/worktrees/t136-3-parity && git worktree remove` it (zero commits over main -- verified 2026 08 14); delete the stray .claude-checkpoint.e11b361c-121.md; restore .claude/scheduled_tasks.lock or commit its deletion. In ~/Code/zerfoo/ztensor: switch to main (`git checkout main && git pull`) -- the nccl branch content is on main under rebased SHAs (verified); inspect the 3 stashes (`git stash show -p`), keep only if content is absent from main, else drop; delete or commit the 2 untracked kernel files after checking against main. Delete ~/zerfoo-latest.bundle and ~/zerfoo-transpose-fix.bundle (March 4, superseded by pushed history -- verify with `git bundle list-heads` against origin first).
  - Acceptance: `git -C ~/Code/zerfoo/<repo> status` clean on main for all five repos; a one-line disposition list per item in the receipt/PR.
- [ ] T150.2 FOUNDER: attic clone disposition  Owner: David  Est: 30m decision  verifies: [infrastructure]  kind: human
  - Decision needed: delete ~/Code/_attic/zerfoo-home-102dirty-15stash and ~/Code/_attic/zerfoo-stale-726behind after a salvage pass? Prep (agent, before asking): enumerate the 15+9 stashes and the 41 ahead-of-upstream commits, mark each SALVAGE (content absent from origin and still relevant) or OBSOLETE (gemma4e-era work superseded by the 2026 08 10 demotion; benchmark JSONs superseded by T136.4). Route the go/no-go via Blink proposal with the salvage list attached; deletion is Tier-3 (destructive).

### E138: Plan Phase 2 (Traction) -- fidelity: outline

Intent: Phase 2 turns verified capability into users -- website/docs site (Hugo per ADR-064; zerfoo.github.io is an empty scaffold), 6+ runnable examples, DX golden-path pass, launch week publishing the docs/distribution/ drafts with T136.4's fresh numbers, GitHub Discussions + CONTRIBUTING + good-first-issues (E124 residue #773/#774/#796/#799), CFP submissions, ADR-084/090 major-version bump + enterprise-repo extraction. Exit criteria: launch executed, kill-criterion clock started (ADR-093: <100 stars + zero engaged users 60 days post-launch -> pivot the wedge). Strategy source: docs/product-strategy-2026-H2.md Part 4 Phase 2.

- [ ] T138.1 PLAN: expand Phase 2 to executable fidelity  Owner: TBD  Est: 2h  delivers: [docs/plan.md replaced with the Phase 2 plan]  kind: plan  blocked-by: [T136.5, T146.2]
  - Run /plan scoped to Phase 2 with the strategy doc Part 4 as input, informed by T136.3/T136.4's actual numbers. End with a task to plan Phase 3 (Moat).

---

## Parallel Work

| Track | Tasks | Notes |
|---|---|---|
| N: Harness + matrix | T136.6 -> S136.6.1 -> T136.3 -> T136.5; T136.4 after T136.6 | GPU-serial: T136.3 then T136.4 |
| O: Trust surface | T146.1 (now); T146.2 (after T136.5) | docs only, no GPU |
| P: Cross-repo hygiene | T147.1 -> T147.2 -> T147.4; T147.3 independent | T147.1 GB10 proof is GPU-serial |
| Q: GQA kernel | T148.1 -> T148.2 (after T147.2) | GPU-serial; time-boxed |
| R: Bug disposition | T149.1 -> T149.2 -> T149.3 (after T136.3) | GPU-serial; each time-boxed |
| S: Workspace | T150.1 (now); T150.2 (founder, anytime) | host-side, no GPU |
| F: Next plan | T138.1 | after N and O converge |

GPU queue order (one pod at a time): T136.3 -> T136.4 -> T147.1 proof -> T148.1/T148.2 -> T149.x. T136.6, T146.1, T147.3, T150.1 need no GPU and start immediately.

### Waves

### Wave 6: Unblock + hygiene fan-out (4 agents + 1 founder ask)
- [ ] T136.6 + S136.6.1 harness honesty  verifies: [UC-H2-004]
- [ ] T146.1 VISION.md amendment  verifies: [UC-H2-013]
- [ ] T147.1 ztensor#179 resolution  verifies: [UC-H2-004]
- [ ] T147.3 zmf/zonnx disposition  verifies: [infrastructure]
- [ ] T150.1 workspace cleanup  verifies: [infrastructure]
- [ ] T150.2 founder attic ask (route via Blink at wave start; does not block anything)  kind: human

### Wave 7: The matrix (GPU-serial, run in-session on the DGX host)
- [ ] T136.3 parity runs  (after T136.6)
- [ ] T136.4 + S136.4.1 Ollama re-run  (after T136.6; after T136.3 in the GPU queue)
- [ ] T136.5 publish matrix  (after T136.3)
- [ ] T147.2 release-please + bumps  (after T147.1; no GPU unless kernel code moved)

### Wave 8: Restore + disposition (GPU-serial)
- [ ] T146.2 claim sweep  (after T136.5)
- [ ] T147.4 branch pruning  (after T147.1, T147.2)
- [ ] T148.1 -> T148.2 ztensor#180 fix + re-enable  (after T147.2)
- [ ] T149.1 -> T149.2 -> T149.3 bug disposition  (after T136.3)

### Wave 9: Phase 2
- [ ] T138.1 plan Phase 2  (after T136.5, T146.2)

---

## Timeline and Milestones

| ID | Milestone | Member tasks | Exit criteria |
|---|---|---|---|
| M-P1-4 | Matrix + benchmarks live (carried) | T136.6, T136.3/4/5 | verified-models.md zero pending markers; benchmarks.md refreshed; T86.5.8 closed |
| M-P1-8 | Trust surface consistent | E146 | claim sweep clean; VISION amended |
| M-P1-9 | Cross-repo limbo cleared | E147, E148, E149 | no stale PRs; #180 fixed or tracked; #981-983 dispositioned |
| M-P1-6 | Phase 2 planned (carried) | T138.1 | new plan.md |

Estimated wall-clock: 3-6 working days of GPU-serial work plus review latency. M-P1-4 + M-P1-8 + M-P1-6 constitute Phase 1 exit (gate G-ZERFOO in hq); M-P1-9 should land within the same window but D10 may honestly convert to tracked debt under its time-box.

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R3 | GB10 contention across tracks + Wolf | Med | High | coordinator owns the GPU queue order above; one pod at a time (SPARK_GPU_MAX=1) |
| R10 | Vacuous parity: matrix goes green while loading the wrong files | High | Med | T136.6 is a hard gate before T136.3; red-proof required, not just review |
| R11 | Dispatch lane dies on a wait-state again, delivering nothing | Med | Med | run parity in-session on the DGX host; per-model incremental writes to verified-models.md; bounded polls, never open-ended waits |
| R12 | "Done" claims that never merged (T135.5/#179 class) recur | Med | Med | task completion requires the merged PR/tag cited in the checkbox annotation; T147.1 corrects the existing instance |
| R13 | Bug-fix scope creep on #981/#982/#983 or #180 | Med | Med | ADR-093 rule 3 pre-committed: one time-boxed attempt each, then park with honest matrix annotation |
| R14 | Branch pruning deletes unmerged work | High | Low | `git cherry` content check per branch; disposition list receipt; when in doubt, keep and list |

---

## Operating Procedure

1. Definition of done: acceptance met; ADR-091 gates for any op/kernel change (gradcheck + parity + oracle); tests green; gofmt/vet/lint clean; PR rebase-merged, CI green; GPU acceptance = an actual dgx-validate.sh (or bench manifest) run recorded in devlog; observed, not expected.
2. One top-level directory per commit; conventional commits; no AI attribution; release-please tags; ztensor before zerfoo.
3. Grep docs/lore.md before debugging (#capture #arena #dst #gb10); quote L-NNNN IDs in commit messages when a rule applies; append new landmines via /lore.
4. All GPU work through Spark; one pod at a time; coordinate across tracks and with Wolf.
5. Plan checkboxes + Progress Log updated as tasks complete; findings to devlog (newest first); benchmarks to benchmarks.md. Checkpoint results to disk as they are produced, never only at task end.

---

## Carried-Forward (NOT in this phase)

- **E127 LTX-2 diffusion (Phase 3; ADR-092).** ~5/35 as of 2026 06 17. Issues #887, #888.
- **E125 mmap remaining (Phase 3).** #802 parked; needs cudaHostRegister purego binding in ztensor first.
- **E124 residue (Phase 2 good-first-issues).** #767, #773/#774/#796/#799; enterprise-repo push + major bump ship with Phase 2 launch (ADR-090).
- **Tracked security tech debt:** #974 (zerfoo kernel Makefile --use_fast_math residue -- real GPU follow-up: Makefile port + .so rebuild + oracle re-run), #975 (/metrics auth gating), #976 (multi-arch distroless digest pin).
- **Parked (label `parked`, 33 issues):** ROCm #701-#706, multi-GPU #712, edge #709/#710/#714, FP8-E5M2 #726, PJRT/E126, E55 fused-encoder epic #522+, E54 #517/#520/#521, E76 #570/#733/#734, perf micro-opts, #757 gemma4e (full H1-H21 history pointer). Revive triggers documented on the issues.
- **Deferred model provisioning:** Llama 4 Scout, MiniMax-M2 (lore L-0016 recipe); Chronos-2 has no GGUF distribution (structural).

---

## Progress Log

### 2026 08 14 -- Change Summary: plan trimmed + closeout scope merged from the seat audit

Trimmed completed epics E133/E134/E135/E137/E139-E145 and Waves 1-5/Sec-1..5 (history: this file's git log at 3fffa972, devlog, deep-review 002 doc). Added E146 (trust-surface reconciliation), E147 (cross-repo PR/branch hygiene incl. ztensor#179 correction), E148 (ztensor#180 GQA kernel), E149 (#981/#982/#983 disposition), E150 (workspace hygiene, incl. founder attic ask); added T136.6 harness-honesty gate ahead of T136.3 (vacuous-parity hazard from the dead 2026 08 11 lane); E138 restated as an outline epic with T138.1 as its planning task, now also blocked by T146.2. Use-case manifest: statuses updated, UC-H2-013 added. No new ADRs (ADR-091/093/094 already govern every decision here). Older progress entries: git history of this file.

---

## Hand-off Notes

### Current state (2026 08 14, post-audit)

- **main is green and in sync** at 3fffa972 (v1.59.0). No open zerfoo PRs. The `origin/handover` branch holds the 2026 08 11 session notes (docs/handover.md) -- still accurate except its "T136.3 ready to pick up" line, which understates the harness hazard now gated by T136.6.
- **The 9 matrix models are verified intact** in /var/lib/zerfoo/models (33GB, checked 2026 08 14).
- **First moves for a fresh session:** Wave 6 -- T136.6 (harness honesty) unblocks everything on the matrix path; T146.1/T147.3/T150.1 are independent and immediate. Route T150.2 to David via Blink early; it blocks nothing.
- **Landmines:** the sandbox may BE the DGX (probe, don't assume); huggingface downloads need the L-0016 curl pattern; zero-matched tests count as pass in the gate (guard exists, stay alert); one GPU pod at a time; two concurrent GPU processes poison the CUDA context.

---

## Appendix

- docs/product-strategy-2026-H2.md -- strategy, phases, metrics, kill criteria. docs/adr/093 -- strategy decision + one-phase plan scoping.
- docs/adr/091 -- verification gates. docs/adr/094 -- untrusted-boundary security policy.
- docs/verified-models.md -- the matrix (T136.1 schema; marketing may not exceed this file).
- docs/deep-reviews/002-full-codebase.md -- security audit + remediation status (closed 2026 08 09).
- docs/lore.md -- landmines L-0001..L-0017. docs/handover.md (origin/handover branch) -- 2026 08 11 session notes.
- Issue clusters: #981/#982/#983 (filed GPU bugs), ztensor#180 (GQA fused kernel), #572/T86.5.8 (parity), #974/#975/#976 (security tech debt).
- .claude/scratch/usecases-manifest.json -- UC-H2-004 and UC-H2-013 active this closeout; UC-H2-003/005/006/007..012 delivered.
- Seat audit behind this revision: hq repo, brain/zerfoo-audit-2026-08-14.md (key facts restated in Discovery Summary above).
