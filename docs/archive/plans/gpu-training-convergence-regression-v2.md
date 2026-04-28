# GPU Training Convergence Regression v2 — Closeout

Supersedes `docs/plans/gpu-training-convergence-regression.md` (v1). The v1
binary fork (E2 read-path vs E3 write-path) was falsified by Wave 1-3
diagnostics on 2026-04-08. This v2 plan closes out the real bug plus every
adjacent issue surfaced during the investigation.

## Context

### Root cause (empirically confirmed, 2026-04-08 on DGX)

PatchTST GPU training loss is frozen at exactly `0.268357` across epochs on
the 5K x 10ch x 3ep bench. The bug is NOT in the AdamW writeback path (v1
plan's E3) and NOT in a stale forward-pass parameter cache (v1 plan's E2).
It is mechanism beta: **the `gradTs` slice in
`timeseries/patchtst_gpu_train.go` is a stale snapshot**. Its entries point
to different memory arenas than the live `grads.X` tensors that the
backward pass writes into. AdamW reads from `gradTs[i]` and sees
all-zero buffers, applies a zero update, weights never move.

Empirical evidence (T3.1b on branch `debug/gpu-train-grad-check`):

- `grads.patchEmbW` pointer `0x...2000` vs `gradTs[0]` pointer `0x...2800`
  (different arena, offset 0x800).
- `grads.headB` and `gradTs[36]` in entirely different arenas.
- Both Point A (post-backward) and Point B (pre-AdamW) show
  `gradTs[i].Data()[:4] == [0 0 0 0]` on the first step.
- The sentinel pointer check at `~line 1076` passes because it checks a
  subset that happens to alias. It is checking the wrong thing.

The `headB` case is dispositive: it has no encoder scratch-accumulator
indirection and still shows an arena mismatch, so the bug is a general
`gradTs` staleness, not encoder-specific machinery.

### Secondary findings from Waves 1-3

1. **The "GPU" training path uses CPUStorage throughout** on GB10 unified
   memory. `paramTs[i].GetStorage()` is `*tensor.CPUStorage[float32]`. The
   comment near line 609-615 about `Data()` being a "fresh D->H memcpy" is
   false. This misled both T1.6 (ztensor SetData audit) and the original
   `f29c93bd` CPU-mirror fix.
2. `f29c93bd` (the CPU-mirror AdamW writeback fix merged in Wave 0) is
   almost certainly redundant machinery. SetData on a CPUStorage tensor is
   a slice swap; the cpuParams mirror is copying host memory into host
   memory. Must be audited, not reverted blindly, because it may be
   masking a second issue.
3. The scratch-tensor accumulator at `patchtst_gpu_train.go:506-530`
   carries a comment admitting it "reproduces pre-fix behavior
   bit-identically". T3.1b ruled it out as the primary mechanism
   (headB has no scratch indirection) but it remains suspect.
4. The sentinel pointer check at `~line 1076` gives false confidence. Needs
   to be strengthened to compare every tensor via
   `unsafe.Pointer(&tensor.Data()[0])` and fail loudly with a full arena
   dump.
5. Spark HTTP API `/logs` endpoint truncates to ~100 lines. Both T3.1a and
   T3.1b had to fall back to `sudo podman logs <container>` on DGX to
   recover full debug output. Needs an upstream issue at feza-ai/spark.

### Objectives

- Land a minimal, correct fix for the stale `gradTs` snapshot.
- Audit and simplify the redundant CPU-mirror scheme from `f29c93bd` if and
  only if it is safe to remove.
- Strengthen the sentinel check so this class of bug cannot recur silently.
- Validate on DGX with the exact v1 success metric: `scripts/bench-spark.sh
  -samples 5000 -channels 10 -epochs 3` with >=90% loss reduction.
- Ensure T4.1 (`TestGPUSingleStepParity`) and T4.2
  (`TestGPUTinyTrainingConvergence`) actually execute and pass on GPU.
- Full root-cause narrative in `docs/devlog.md`.
- Clean up all investigation debris (debug branches, instrumentation).

### Non-Goals

- No refactor of PatchTST GPU training beyond the surgical fix and
  sentinel strengthening.
- No changes to ztensor or the compute engine.
- No changes to Spark itself; only a bug report issue upstream.
- No investigation of the 10-epoch hang (separate issue).
- No re-architecting of the gradient accumulator. If the scratch-tensor
  accumulator is not on the critical path, leave it alone.

### Constraints and Assumptions

- Spark v1.6.1 is operational on DGX 192.168.86.250. Use
  `scripts/bench-spark.sh` for all GPU benches. DO NOT use interactive
  `ssh dgx bench_train ...`.
- CPU training is green throughout this work. CPU tests must stay green.
- `go test ./timeseries/... -race` must stay green on CPU. Excluding the
  pre-existing flaky `TestAllBackends_CPUTrainingBenchmark` (30s budget
  overrun, unrelated) is acceptable.
- Max 4 agents per wave (per user feedback on agent parallelism limits).
- PR #361 (Wave 1 merged) and PR #362 (T4.1+T4.2 tests, open at plan
  authoring) are baseline context.

### Success Metrics

- 5K x 10ch x 3ep via `scripts/bench-spark.sh` shows strictly decreasing
  loss with final/initial ratio <=0.1 (>=90% reduction).
- `TestGPUSingleStepParity` passes on DGX (weights change after one step).
- `TestGPUTinyTrainingConvergence` passes on DGX (loss[1] < loss[0]*0.9).
- No frozen `0.268357` anywhere in the loss trajectory for default seed.
- Zero SSH session leaks across validation runs.
- Full `go test ./timeseries/... -race` green on CPU (excluding the known
  unrelated flake).

## Discovery Summary

Engineering work. Single affected use case: UC-TS01 (PatchTST training),
currently BROKEN on GPU, WIRED on CPU. Inference paths unaffected. The use
case manifest at `.claude/scratch/usecases-manifest.json` is unchanged from
v1.

**Prior context in scratch notes (all from Waves 1-3 on 2026-04-08):**

- `.claude/scratch/t1.4-audit.md` — extractGPUParams aliasing audit
- `.claude/scratch/t1.5-audit.md` — forward-pass weight read sites (28/28 alias paramTs)
- `.claude/scratch/t1.6-audit.md` — SetData semantics (unconditional HtoD — misled by false comment)
- `.claude/scratch/t3.1a-narrow-result.md` — storage kind + pData diagnostic (exonerates write path)
- `.claude/scratch/t3.1b-grad-check-result.md` — arena mismatch between grads.X and gradTs[i]

**Key file:** `timeseries/patchtst_gpu_train.go` (1203 lines).
**Key regions:**
- `extractGPUParams` at 51-170
- scratch-tensor accumulator at 506-530
- `cpuParams` mirror setup at 565-625
- sentinel pointer check at ~1076
- AdamW loop + SetData writeback at 1113-1180

**Related artifacts:**
- `timeseries/weight_hash_debug.go` — Wave 1 HashParamTensors helper
  (env-gated, FNV-1a). Already on main via PR #361. Keep as a permanent
  diagnostic tool.
- `timeseries/patchtst_gpu_train_parity_test.go` (T4.1) and
  `timeseries/patchtst_gpu_convergence_test.go` (T4.2) on PR #362.
- Investigation branches awaiting cleanup:
  `debug/gpu-train-hash-instrument`, `debug/gpu-train-pdata-check`,
  `debug/gpu-train-grad-check`, `test/gpu-train-single-step-parity`,
  `test/gpu-train-tiny-convergence`, `audit/t1.4-extractgpuparams-aliasing`,
  `wave-1-integration`, `wave-3-tests-integration`,
  `wave-2-diagnostic`.

## Scope and Deliverables

### In Scope

- Surgical fix to re-point `gradTs` at the live `grads.X` tensors per
  batch, or eliminate the snapshot entirely.
- Strengthen the sentinel pointer check.
- Audit the `cpuParams` mirror from `f29c93bd` and remove if redundant.
- Audit the scratch-tensor accumulator at 506-530 and remove if dead.
- Correct the misleading comment at ~line 609-615.
- Ensure T4.1 and T4.2 execute on GPU and pass post-fix.
- DGX validation via Spark at the v1 success-metric bench size.
- Devlog root-cause writeup (original T6.1).
- Spark upstream issue for the log truncation.
- Clean up debug branches and reset instrumentation.

### Out of Scope

- ztensor changes.
- Engine or compute-layer changes.
- Other timeseries models.
- Performance tuning.
- 10-epoch hang investigation.
- Changes to scripts/bench-spark.sh or ADR 083.

### Deliverables Table

| ID | Description | Owner | Acceptance Criteria |
|---|---|---|---|
| D1 | gradTs staleness fixed | TBD | `grads.X` and `gradTs[i]` share the same backing array at AdamW read time for every index. Verified by strengthened sentinel check. |
| D2 | Sentinel check strengthened | TBD | Check compares `unsafe.Pointer(&tensor.Data()[0])` for every param. Fails loudly with a full arena dump on mismatch. |
| D3 | CPU-mirror scheme audited | TBD | Written verdict: keep / simplify / remove, with reasoning. If remove: removal commit with CPU tests green. |
| D4 | Scratch accumulator audited | TBD | Written verdict: live-code / dead-code / suspect-but-harmless. If dead: removal commit. |
| D5 | Misleading comment corrected | TBD | Line 609-615 comment reflects CPUStorage reality. |
| D6 | DGX convergence validated | TBD | 5K x 10ch x 3ep >=90% loss reduction. Zero SSH leak. |
| D7 | T4.1 + T4.2 pass on GPU | TBD | Both tests executed on DGX (via bench pod or direct `go test`), both PASS. |
| D8 | Devlog writeup | TBD | New entry in `docs/devlog.md` covering the v1 false fork, the beta mechanism, the five-hypothesis postmortem, the fix, before/after bench snippet. |
| D9 | Spark upstream issue | TBD | Issue filed at feza-ai/spark with reproducer for /logs truncation. |
| D10 | Investigation debris cleaned | TBD | All debug and audit branches deleted locally and on origin. Worktrees removed. `bench_train` binary added to .gitignore if still untracked. |

## Checkable Work Breakdown

### Epic E1: Surgical fix

- [ ] T1.1 Locate gradTs population site  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Grep for `gradTs[` assignments in `timeseries/patchtst_gpu_train.go`.
  Document file:line of every write and every read. Identify the snapshot
  point that captures stale pointers.
  Acceptance: short note at `.claude/scratch/e1-gradts-sites.md` listing
  all gradTs sites with file:line and role (populate / read / snapshot).

- [ ] T1.2 Apply the minimal fix  Owner: TBD  Est: 1.5h  verifies: [UC-TS01]
  Deps: T1.1
  Acceptance: Either (a) rebuild `gradTs` from `grads.allParamTensors()`
  each batch immediately before the AdamW loop, or (b) eliminate `gradTs`
  and iterate over `grads.X` directly in the AdamW loop. Choose whichever
  is the smaller diff. Comment the fix with a reference to this plan and
  T3.1b findings. No unrelated refactors.

- [ ] T1.3 Lint + CPU test gate  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T1.2
  Acceptance: `go build ./...`, `go vet ./...`,
  `go test ./timeseries/... -race -count=1` all green on CPU (excluding
  the unrelated flaky `TestAllBackends_CPUTrainingBenchmark`).

### Epic E2: Sentinel strengthening + comment correction

- [ ] T2.1 Strengthen the sentinel pointer check  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: The check at `~line 1076` is replaced with a loop that
  verifies for every `i` that
  `unsafe.Pointer(&grads.allParamTensors()[i].Data()[0]) ==
  unsafe.Pointer(&gradTs[i].Data()[0])`. On mismatch, panic with a full
  dump: index, tensor name (if available), both pointers, both
  `Data()[:4]` samples.

- [ ] T2.2 Correct the misleading CPUStorage comment  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Acceptance: Replace the false "Data() is a fresh D->H memcpy" comment
  near line 609-615 with an accurate note: on GB10 unified memory the
  training path uses CPUStorage and `Data()` is a direct slice reference,
  not a memcpy.

- [ ] T2.3 Add a unit test for the sentinel  Owner: TBD  Est: 0.75h  verifies: [UC-TS01]
  Deps: T2.1
  Acceptance: New test in `timeseries/` that constructs a `gpuParams` /
  `gpuGrads` pair where one `gradTs[i]` deliberately points at a different
  slice, invokes the sentinel, and asserts it panics or returns error.
  Catches future regressions of this class.

### Epic E3: Audit + cleanup

- [ ] T3.1 Audit the cpuParams mirror scheme  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: Written verdict at `.claude/scratch/e3-cpumirror-audit.md`.
  Read the cpuParams setup at 565-625 and every use site. Determine
  whether it is (a) redundant given CPUStorage reality, (b) accidentally
  load-bearing, or (c) actively harmful. If (a), include a removal diff
  in the same commit as T3.3. If (b) or (c), document and leave alone
  pending a separate plan.

- [ ] T3.2 Audit the scratch-tensor accumulator  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: Written verdict at `.claude/scratch/e3-accumulator-audit.md`.
  Read lines 506-530 and every call site. Determine live-code vs dead-code.
  If dead-code, include removal in T3.3. If live-code, correct the
  comment that admits it "reproduces pre-fix behavior bit-identically".

- [ ] T3.3 Apply removals from T3.1 and T3.2 if approved  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Deps: T3.1, T3.2, T1.2 (fix must land first)
  Acceptance: Atomic commit removing only the code verdicts marked for
  removal. `go test ./timeseries/... -race` green. Commit message cites
  both audit files.

- [ ] T3.4 Lint and vet  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T3.3 (or T1.3 if T3.3 is skipped)
  Acceptance: `go vet ./...` clean, `golangci-lint run ./timeseries/...`
  clean.

### Epic E4: Regression tests activated on GPU

- [ ] T4.1 Ensure PR #362 is merged  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Acceptance: PR #362 (T4.1 TestGPUSingleStepParity + T4.2
  TestGPUTinyTrainingConvergence) merged to main. If not merged, hold E5
  until it is.

- [ ] T4.2 Run T4.1 (parity test) on DGX  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: E1 complete, T4.1
  Acceptance: On DGX, `cd ~/zerfoo && git checkout main && git pull &&
  /usr/local/go/bin/go test ./timeseries/... -race -count=1 -run
  TestGPUSingleStepParity`. Must PASS (not skip). Capture output to
  devlog.

- [ ] T4.3 Run T4.2 (tiny convergence test) on DGX  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T4.2
  Acceptance: Same as T4.2 but for `TestGPUTinyTrainingConvergence`.
  Must PASS. Capture output to devlog.

### Epic E5: DGX convergence validation

- [ ] T5.1 Rebuild bench_train on DGX from fix branch  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T1.2 committed and pushed
  Acceptance: `ssh ndungu@192.168.86.250 'cd ~/zerfoo && git fetch &&
  git checkout <fix-branch> && /usr/local/go/bin/go build -o
  /tmp/bench_train_new ./cmd/bench_train && sudo install
  /tmp/bench_train_new /var/lib/zerfoo/bin/bench_train && md5sum
  /var/lib/zerfoo/bin/bench_train'`. Record md5.

- [ ] T5.2 Smoke bench 1K x 5 x 2  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T5.1
  Acceptance: `scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 2
  -cleanup` shows decreasing loss. No `0.268357` anywhere.

- [ ] T5.3 Regression bench 5K x 10 x 3  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T5.2
  Acceptance: `scripts/bench-spark.sh -samples 5000 -channels 10 -epochs
  3 -cleanup` strictly decreasing loss with final/initial ratio <=0.1
  (>=90% reduction). Compare against CPU baseline
  (`0.115235 -> 0.000863`).

- [ ] T5.4 SSH session leak check  Owner: TBD  Est: 0.1h  verifies: [infrastructure]
  Deps: T5.3
  Acceptance: `ssh ndungu@192.168.86.250 'who | wc -l'` before T5.1 and
  after T5.3 must be equal.

### Epic E6: Documentation + cleanup

- [ ] T6.1 Devlog root-cause writeup  Owner: TBD  Est: 1h  delivers: [devlog entry with full postmortem]
  Deps: T5.3, T4.3
  Acceptance: New entry appended to `docs/devlog.md` covering:
  - The symptom (`0.268357` frozen loss).
  - The five v1 hypotheses and why each was wrong.
  - The Wave 1 audits that all pointed to E3 on paper.
  - The Wave 2 diagnostic that claimed E3.
  - The Wave 3 narrowing that revealed the real mechanism (beta).
  - The fix diff.
  - Before/after bench snippet.
  - Lessons learned: code comments were the primary source of
    misdirection (the "GPU" path is CPUStorage, Data() does not memcpy),
    and the sentinel check was checking the wrong thing.

- [ ] T6.2 File Spark /logs truncation issue upstream  Owner: TBD  Est: 0.5h  delivers: [github issue at feza-ai/spark]
  Acceptance: Issue filed at feza-ai/spark with:
  - Reproducer: a pod that emits >100 lines to stdout.
  - Observed: HTTP `/api/v1/pods/<name>/logs` returns truncated.
  - Expected: full logs, or a documented pagination mechanism.
  - Workaround: `sudo podman logs <container>` on the host.
  - Link this plan's T3.1a/T3.1b findings as context.

- [ ] T6.3 Delete investigation debris  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T6.1 (devlog preserves findings before branches are deleted)
  Acceptance: For each of:
    `debug/gpu-train-hash-instrument`,
    `debug/gpu-train-pdata-check`,
    `debug/gpu-train-grad-check`,
    `audit/t1.4-extractgpuparams-aliasing`,
    `wave-1-integration`,
    `wave-3-tests-integration`,
    `test/gpu-train-single-step-parity` (after merge),
    `test/gpu-train-tiny-convergence` (after merge):
  verify the branch has no unique unmerged content not captured elsewhere
  (`git cherry main <branch>` shows only `-` lines), then delete local
  and remote. Remove stale worktrees under `.claude/worktrees/`. Add
  `bench_train` binary to `.gitignore` if still untracked.

- [ ] T6.4 HashParamTensors helper: keep as permanent tool  Owner: TBD  Est: 0.1h  verifies: [infrastructure]
  Acceptance: Decision recorded in T6.1 devlog entry. The helper at
  `timeseries/weight_hash_debug.go` stays on main as a permanent
  diagnostic tool, env-gated on `ZERFOO_DEBUG_WEIGHT_HASH=1`. No code
  change required.

## Parallel Work

### Tracks

| Track | Tasks | Description |
|---|---|---|
| A: Fix | T1.1 -> T1.2 -> T1.3 | Sequential. Critical path. |
| B: Sentinel + comment | T2.1, T2.2, T2.3 | Parallel with A. T2.3 depends on T2.1. |
| C: Audits | T3.1, T3.2 | Parallel with A and B. |
| D: Cleanup apply | T3.3, T3.4 | After A, B, C. |
| E: DGX validation | T5.1 -> T5.2 -> T5.3 -> T5.4 | Sequential, single agent. After A+B+D. |
| F: GPU tests | T4.1, T4.2, T4.3 | After E1. |
| G: Docs + cleanup | T6.1, T6.2, T6.3, T6.4 | After E and F. |

### Waves

#### Wave 1: Fix + parallel audits (4 agents)

- [ ] T1.1 Locate gradTs population site  verifies: [UC-TS01]
- [ ] T2.1 Strengthen sentinel pointer check  verifies: [UC-TS01]
- [ ] T3.1 Audit cpuParams mirror scheme  verifies: [UC-TS01]
- [ ] T3.2 Audit scratch-tensor accumulator  verifies: [UC-TS01]

Note: T1.1 is discovery-only and its output feeds T1.2 in Wave 2. T2.1
can run fully in parallel because it touches a different region.

#### Wave 2: Apply fix + sentinel test + comment + audit removals (4 agents)

- [ ] T1.2 Apply the minimal gradTs fix  verifies: [UC-TS01]
- [ ] T2.2 Correct misleading CPUStorage comment  verifies: [UC-TS01]
- [ ] T2.3 Add sentinel unit test  verifies: [UC-TS01]
- [ ] T3.3 Apply removals from T3.1 and T3.2 if approved  verifies: [UC-TS01]

All four agents run in isolated worktrees; merge protocol per /apply
handles overlap on `patchtst_gpu_train.go`.

#### Wave 3: Local validation (1 agent)

- [ ] T1.3 Lint + CPU test gate
- [ ] T3.4 Lint and vet

Single agent: these are a fast sequential pair that must run on the
merged Wave 2 integration branch.

#### Wave 4: Ensure test PR merged (1 agent, gate)

- [ ] T4.1 Ensure PR #362 merged

Gate step. If PR #362 is unmerged, this wave halts until a human merges
it. Once merged, Wave 5 unblocks.

#### Wave 5: DGX validation (1 agent, sequential internally)

- [ ] T5.1 Rebuild bench_train on DGX
- [ ] T5.2 Smoke 1K x 5 x 2
- [ ] T5.3 Regression 5K x 10 x 3
- [ ] T5.4 SSH leak check
- [ ] T4.2 Run TestGPUSingleStepParity on DGX
- [ ] T4.3 Run TestGPUTinyTrainingConvergence on DGX

Single agent because each step is fast and strictly sequential on the
same DGX host. T4.2 and T4.3 run as `go test` on DGX directly (not via
bench pod) since they are skippable tests keyed on GPU availability.

#### Wave 6: Docs + cleanup (3 agents)

- [ ] T6.1 Devlog writeup
- [ ] T6.2 File Spark upstream issue
- [ ] T6.3 Delete investigation debris

T6.4 (helper keep-decision) is folded into T6.1.

### Dependency-Minimization Notes

- Wave 1 saturates 4 agents by running the fix-locate, sentinel
  strengthening, and both audits in parallel. Only T1.1 feeds T1.2; the
  other three are independent.
- Wave 2 saturates 4 agents by applying all code changes in parallel
  across isolated worktrees. The merge protocol handles file-level
  overlap on `patchtst_gpu_train.go` since all four tasks touch it.
- Waves 3, 4, 5 are necessarily narrow: Wave 3 is a single gate, Wave 4
  is a human-checkpoint, Wave 5 is strictly sequential on one host.
- Wave 6 parallelizes the tail docs+cleanup work.

## Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|---|---|---|---|
| M1 | Fix + sentinel committed | Wave 2 complete | T1.2, T2.1, T2.2, T2.3, T3.3 all merged onto integration branch. CPU tests green. |
| M2 | Local validation green | Wave 3 complete | `go build`, `go vet`, `go test -race` green on CPU. |
| M3 | DGX convergence validated | Wave 5 complete | 5K x 10ch x 3ep >=90% reduction. T4.1 and T4.2 pass on GPU. Zero SSH leak. |
| M4 | Closeout | Wave 6 complete | Devlog entry landed. Spark issue filed. Debug branches deleted. PR merged to main. |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | T1.1 reveals gradTs is populated in more than one place and T1.2's minimal fix is insufficient | Plan loops | Med | T1.1 is explicitly a discovery task; if it finds complexity, escalate to human review before T1.2. |
| R2 | Strengthened sentinel (T2.1) panics on legitimate state the plan did not anticipate | CPU or GPU tests fail in Wave 3 | Med | Sentinel is gated by the same env var as HashParamTensors; fall back to log-only on first pass if needed. |
| R3 | cpuParams audit concludes removal is unsafe but leaves behind load-bearing dead code | Confusing machinery persists | Low | T3.1 verdict is written; if ambiguous, default to keep-with-comment rather than remove. |
| R4 | PR #362 cannot be merged in time for Wave 5 | Wave 5 blocked | Low | Wave 4 is an explicit gate; if PR #362 is stuck, the plan halts at Wave 4 rather than proceeds with no regression tests. |
| R5 | DGX bench shows partial convergence (e.g., 50%) — fix is directionally right but something else is still broken | Fix incomplete | Low | Return to Wave 1 with new instrumentation. Do not merge a partial fix. |
| R6 | Third-party bug in ztensor CPUStorage that only manifests under training workload | Fix needs upstream change | Low | Audit in T3.1 covers this indirectly. If found, file a zerfoo/ztensor issue and pause. |
| R7 | SSH session accumulation on DGX during validation | Host reboot | Low | All bench work via Spark. Only short ssh for git/build/install. T5.4 verifies. |

## Operating Procedure

### Definition of Done (per task)

- Code in small focused commits using Conventional Commits
  (`fix(timeseries):`, `test(timeseries):`, `docs(devlog):`).
- `go test ./timeseries/... -race` green on CPU.
- Lint green on changed packages.
- For the fix PR specifically: DGX bench (D6) passes before merge.

### Review and QA

- Single reviewer on the fix PR. Tensor aliasing is subtle; one extra
  set of eyes is cheap insurance.
- T4.1 and T4.2 are the primary regression gates. If either passes on a
  pre-fix checkout, it is not catching the bug.
- T2.3 (sentinel unit test) is the secondary gate that prevents this
  class of staleness from recurring silently.

### Commit Discipline

- One commit per logical change. No cross-directory commits.
- Instrumentation, fix, sentinel, removals, and docs each in their own
  commits.
- No commit straddles timeseries and ztensor (this plan does not touch
  ztensor anyway).

### Versioning

- No dedicated release. The fix will ride the next routine release-please
  bump of zerfoo.

## Progress Log

### Change Summary 2026-04-08 (v2 plan created)

- v2 plan created to supersede v1 after empirical diagnostics falsified
  the v1 E2/E3 fork.
- Root cause identified: stale `gradTs` snapshot (mechanism beta).
  Evidence from `.claude/scratch/t3.1b-grad-check-result.md`.
- Six epics, 22 tasks across 6 waves, max 4 agents per wave.
- v1 plan remains on disk at
  `docs/plans/gpu-training-convergence-regression.md` as historical
  context. Not deleted.
- No ADR created. The fix is a local bug patch; no architectural
  decisions are at stake. If T3.1 audit concludes the f29c93bd
  CPU-mirror scheme should be removed, an ADR for the removal may be
  appropriate but is deferred until that verdict is in.
- PR #361 already merged (Wave 1 of v1: weight-hash helper + audits).
- PR #362 open at plan authoring (T4.1 + T4.2 regression tests).
- Investigation branches to delete in T6.3 are listed in the Discovery
  Summary.

## Hand-off Notes

- **What you need to know first:** The v1 plan's binary E2/E3 fork was
  wrong. The real bug is mechanism beta — `gradTs` is a stale snapshot
  in `timeseries/patchtst_gpu_train.go` that captures parameter-gradient
  pointers at setup time but is divorced from the live `grads.X` tensors
  that backward actually writes to. AdamW sees zero grads and applies
  zero updates. Weights never move. Loss frozen at `0.268357` on GPU.
  CPU path works fine with the same binary.
- **Start here:** Wave 1 (4 agents: T1.1 gradTs locate, T2.1 sentinel
  strengthen, T3.1 cpuParams audit, T3.2 scratch accumulator audit).
- **DGX access:** `ssh ndungu@192.168.86.250`. Repo at `~/zerfoo`.
  Go at `/usr/local/go/bin/go`. Bench binary at
  `/var/lib/zerfoo/bin/bench_train`. Passwordless sudo.
- **Bench runner:** `scripts/bench-spark.sh -samples N -channels C
  -epochs E`. Spark v1.6.1. DO NOT use interactive
  `ssh dgx bench_train`.
- **Key file regions:** `timeseries/patchtst_gpu_train.go`:
  - `extractGPUParams` at 51-170
  - scratch-tensor accumulator at 506-530
  - `cpuParams` mirror setup at 565-625
  - sentinel pointer check at ~1076
  - AdamW loop + SetData writeback at 1113-1180
- **Scratch notes from Waves 1-3:** `.claude/scratch/t1.4-audit.md`,
  `t1.5-audit.md`, `t1.6-audit.md`, `t3.1a-narrow-result.md`,
  `t3.1b-grad-check-result.md`.
- **Misleading code to distrust:** Comment near line 609-615 about
  `Data()` being a "fresh D->H memcpy" — false on GB10. The comment at
  506-530 that the scratch accumulator "reproduces pre-fix behavior
  bit-identically" — suspect but not the primary bug.
- **Pre-existing test flake:** `TestAllBackends_CPUTrainingBenchmark`
  fails on main due to a 30s budget overrun. Unrelated to this plan.
  Exclude it from full-suite runs during this work.
- **When you are done:** The bench wave from `docs/plan.md`
  (T50.5.2, T51.5.2, T54.4.1, T63.2.1, T61.3.2) is fully unblocked, same
  as v1's hand-off notes.

## Appendix

### Expected convergence signature (CPU, reference)

```
epoch  1: loss=0.115235 ok
epoch  2: loss=0.001384 ok
epoch  3: loss=0.000863 ok
convergence: OK (0.115235 -> 0.000863, 99.3% reduction)
```

### Current broken signature (GPU, pre-fix)

```
epoch  1: loss=0.268357 ok
epoch  2: loss=0.268357 ok
epoch  3: loss=0.268357 ok
convergence: FAILED (loss did not decrease)
```

### Useful one-liners

Rebuild and restage bench_train on DGX from a fix branch:
```
ssh ndungu@192.168.86.250 'cd ~/zerfoo && git fetch && git checkout <branch> && /usr/local/go/bin/go build -o /tmp/bench_train_new ./cmd/bench_train && sudo install /tmp/bench_train_new /var/lib/zerfoo/bin/bench_train && md5sum /var/lib/zerfoo/bin/bench_train'
```

Submit the regression bench via Spark:
```
ssh ndungu@192.168.86.250 'cd ~/bench-spark && SPARK_HOST=localhost:8080 ./scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3 -cleanup'
```

Recover full logs when Spark /logs truncates:
```
ssh ndungu@192.168.86.250 'sudo podman logs <container-name>'
```

### References

- v1 plan: `docs/plans/gpu-training-convergence-regression.md`
- Spark ADR: `docs/adr/083-spark-bench-runner.md`
- Devlog: `docs/devlog.md` (Wave 1 audit entries already landed)
- PR #361 (Wave 1 merged): weight-hash helper + audits
- PR #362 (open at authoring): T4.1 + T4.2 regression tests
- claude-mem breadcrumbs: #1776, #1789, #1791, #1793, #1812, #1813
