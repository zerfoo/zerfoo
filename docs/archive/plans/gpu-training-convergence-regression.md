# GPU Training Convergence Regression Investigation

## Context

On 2026-04-08 we commissioned Spark as the DGX bench runner
(`docs/plans/spark-bench-runner.md`, ADR 083) and empirically validated the
"fix" on branch `fix/gpu-train-cpu-mirror` (commit `0750c440`, merged to main
as `f29c93bd`) only to discover it is incomplete.

**Symptom.** On PatchTST training with 5000 samples x 10 channels x 3 epochs
submitted via `scripts/bench-spark.sh`, GPU loss is frozen at exactly
`0.268357` across all three epochs, byte-for-byte identical to the pre-fix
regression value recorded in memory #1776. The identical binary on CPU
converges normally: `0.115235 -> 0.000863` (99.3% reduction in 3 epochs).

**Key diagnostic clue.** The frozen value `0.268357` is exactly reproducible
across runs. This suggests the weights are STATIC FROM INITIALIZATION on the
GPU path, not merely updating too slowly or diverging. Something on the GPU
side is preventing updates from taking effect on the forward pass, even
after the AdamW `SetData` writeback lands.

**What the current fix does.** `timeseries/patchtst_gpu_train.go` around
line 609 maintains persistent CPU mirrors `cpuParams[i]` and
`cpuGrads[i]` per parameter tensor. After each batch's AdamW step on the
CPU mirror, line 1152 calls `paramTs[i].SetData(pData)` to push the updated
weights back to the device tensor. This fix is necessary (without it
`paramTs[i].Data()` returned a throwaway device->host copy on every call)
but apparently not sufficient.

**Where the second hole probably lives.** Candidates to investigate, in
decreasing order of suspicion:

1. **`extractGPUParams` returns copies, not aliases.** If the `gpuParams`
   struct at line 13 holds fresh GPU tensors that are *copies* of model
   parameters at the start of training, then `SetData(pData)` updates
   those copies but the forward pass inside the training loop might be
   reading from a different set of tensors entirely. Inspect whether
   `paramTs` aliases the tensors actually consumed by the matMul ops in
   `encoderForward*` / `decoderForward*`.

2. **Forward pass has its own parameter cache.** The encoder may stash
   parameters in `gpuBatchForwardCache` at setup time and read from that
   cache every batch. If the cache is populated once before the training
   loop and never refreshed, updates will not reach the forward pass.

3. **`SetData` is a no-op on GPU tensors under certain conditions.** If
   the GPU tensor backing store is lazily-allocated, read-only, or
   pinned, `SetData` might silently skip the device copy. Audit
   `ztensor/tensor/*.go` for the GPU `SetData` implementation and check
   whether it guards on tensor kind, shape, or residency.

4. **A second parameter path outside the AdamW loop.** There may be a
   parameter-like tensor (LayerNorm scale, patch embedding bias, output
   head) that is not in `gpuParams.allParamTensors()` and therefore is
   never updated at all. If most of the model is frozen, the loss would
   look static.

5. **Gradient tensor pointer disconnection (memory #1793), second hole.**
   The writeback fix addressed the optimizer's params; a symmetric hole
   may exist on the gradient side where backward writes to a tensor the
   next batch does not read from.

### Objectives
- Identify the exact code path that prevents GPU parameter updates from
  reaching the forward pass.
- Fix the root cause with the smallest correct change.
- Produce a regression test in the Go test suite that reproduces the
  original symptom on a minimal deterministic synthetic input and would
  have caught this bug before it reached main.
- Validate the fix on DGX via `scripts/bench-spark.sh` at 5K x 10ch x 3ep,
  expecting convergence comparable to the CPU path (>= 90% loss reduction).

### Non-Goals
- No refactor of the PatchTST GPU training path beyond the minimal fix.
- No changes to the Spark bench runner or the ADR 083 infrastructure.
- No reverting `f29c93bd` (the CPU-mirror writeback fix). That fix is
  correct as far as it goes; this plan adds the missing piece.
- No investigation of training hangs at 10+ epoch runs (separate issue).

### Constraints and Assumptions
- Spark v1.6.1 is already installed on DGX 192.168.86.250 and the bench
  pipeline is fully operational per the 2026-04-08 devlog entry. Use
  `scripts/bench-spark.sh -samples N -channels C -epochs E` for all GPU
  validation.
- DO NOT use interactive `ssh dgx bench_train ...` for benches. That
  caused the 2026-04-07 outage.
- Short ssh commands for debugging (grep, file inspection, spark pod
  queries) remain fine in foreground.
- CPU training is known good and must remain green throughout this work.
- The bench binary at `/var/lib/zerfoo/bin/bench_train` on DGX is built
  from local main. Rebuild and restage after every change.

### Success Metrics
- 5K x 10ch x 3ep GPU run via `scripts/bench-spark.sh` shows a
  strictly decreasing loss with final/initial ratio <= 0.1 (>= 90%
  reduction).
- A new Go test under `timeseries/` reproduces the static-weights bug on
  a minimal deterministic input AND fails cleanly on a checkout of the
  pre-fix state, demonstrating the test would have caught the regression.
- Go test suite remains green: `go test ./... -race -timeout 300s`.
- Zero SSH session leaks on DGX across the validation runs.

## Discovery Summary

**Work type:** engineering (investigation + bug fix).

**Use cases affected:** UC-TS01 (PatchTST training) is currently BROKEN on
GPU. CPU path is WIRED and functional. Inference paths (Gemma/DeepSeek/
Llama) are UNAFFECTED.

**Files in scope:**
- `timeseries/patchtst_gpu_train.go` (1203 lines) — the GPU training loop,
  `extractGPUParams`, the AdamW step with CPU mirrors (line 1127), the
  forward/backward glue, the writeback at line 1170.
- `timeseries/patchtst.go` — model struct, inference forward pass (for
  comparison).
- `timeseries/trainable_test.go`, `timeseries/training_ops_test.go` —
  existing training tests, likely CPU-only.
- `ztensor/tensor/` — SetData, Data, GPU storage semantics.
- `ztensor/compute/` — Engine.MatMul, Engine.Zero on the GPU path.

**Prior breadcrumbs (from claude-mem):**
- #1789: "missing engine.Zero calls before MatMul in encoder buffer
  pre-allocation" — tentatively resolved in main.
- #1791: "encoder uses matMulInto wrapper with Zero calls but GPU
  regression persists" — confirms the issue is not in the zero-then-
  matmul pattern alone.
- #1793: "gradient tensor pointer disconnection prevents weight updates"
  — partial fix landed; this plan assumes at least one more hole.
- #1812: "implemented CPU-GPU parameter synchronization in AdamW
  optimizer loop" — the fix that landed on main in `f29c93bd`.

**Reference:** `docs/devlog.md` 2026-04-08 entry has the full
reproduction recipe, the exact bench output (including the byte-frozen
`0.268357`), and the comparison to CPU convergence.

## Scope and Deliverables

### In Scope
- Diagnostic instrumentation: weight hash dumps before/after the first
  optimizer step on GPU, to confirm/deny the "weights are static"
  hypothesis.
- Code audit: `extractGPUParams`, the forward/backward paths, `SetData`
  semantics on GPU tensors, any parameter caches.
- Root-cause fix.
- Deterministic GPU/CPU parity test at single-step granularity.
- Long-form regression test on a tiny synthetic config that runs in
  both CI and `bench_train`.
- Bench validation on DGX via Spark.
- Devlog writeup of the final root cause and fix.

### Out of Scope
- `bench_batch`, `bench_tps`, any inference benches.
- Changes to other timeseries models (TSPulse, etc.) unless the same
  bug affects them and the fix is trivially the same.
- Performance optimization of the GPU training path.
- The 10-epoch hang investigation.

### Deliverables Table
| ID | Description | Owner | Acceptance Criteria |
|---|---|---|---|
| D1 | Root cause identified and documented | TBD | A specific file:line or small set of lines in `timeseries/patchtst_gpu_train.go` or a dependency is named as the hole, with evidence from instrumentation. |
| D2 | Fix committed | TBD | Smallest change that makes 5K x 10ch x 3ep converge on GPU. No unrelated refactors. |
| D3 | Unit-level regression test | TBD | New test under `timeseries/` that asserts GPU param tensors change after a single optimizer step on a 2-sample 1-channel 1-epoch deterministic input. Fails on the pre-fix commit. |
| D4 | Integration regression test | TBD | New test under `timeseries/` that runs a tiny end-to-end training loop (e.g. 256 samples x 2 channels x 2 epochs) and asserts loss[1] < loss[0] * 0.9. Skippable when no GPU engine is present. |
| D5 | DGX validation via Spark | TBD | `scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3` shows decreasing loss with >= 90% reduction. Zero SSH session leak. |
| D6 | Devlog writeup | TBD | New entry in `docs/devlog.md` with the root cause, the fix, and a before/after bench snippet. |

## Checkable Work Breakdown

### Epic E1: Diagnose

- [ ] T1.1 Add a weight-hash logging helper  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Acceptance: A helper (can be dev-only behind a build tag or env var)
  that walks `paramTs` and logs an FNV-1a or SHA-256 hash per tensor
  after snapshotting via `.Data()`. Output format: one line per tensor
  with tensor name and hex hash. No test dependency.

- [ ] T1.2 Instrument training loop with pre/post-step hashes  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T1.1
  Acceptance: Before the first optimizer step and after it, call the
  hash helper from T1.1. Commit this as a debug commit that will be
  reverted or gated behind a flag before the final fix lands.

- [ ] T1.3 Run instrumented bench on DGX via Spark  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T1.2
  Acceptance: Rebuild `bench_train` on DGX, restage to
  `/var/lib/zerfoo/bin/`, run
  `scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 1 -cleanup`
  and capture the pre/post hashes. If any hash differs, the weights DO
  update on device and the bug is in the forward-pass read path
  (investigate E2). If all hashes match, the weights do NOT update and
  the bug is on the write path (investigate E3).

- [ ] T1.4 Audit `extractGPUParams` and `paramTs` aliasing  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: Read `timeseries/patchtst_gpu_train.go` lines 51-170 and
  adjacent code. Produce a short note (can go in devlog or a scratch
  file) describing whether `paramTs[i]` is the live GPU tensor that the
  forward pass reads from, or a separate copy. Attach
  file:line references.

- [ ] T1.5 Audit forward-pass parameter access  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: Read the encoder and decoder forward functions invoked
  from the training loop. Identify every read of a trainable weight.
  Confirm each read goes through the same tensor that AdamW writes to.
  Document any discrepancies.

- [ ] T1.6 Audit `SetData` semantics on GPU tensors  Owner: TBD  Est: 1h  verifies: [UC-TS01]
  Acceptance: Read the `SetData` implementation in ztensor
  (`ztensor/tensor/*.go`). Confirm that on a GPU-backed tensor it
  performs an unconditional host->device copy. Note any guards,
  lazy-alloc behavior, or early returns that could cause a silent skip.

### Epic E2: Fix on the read path (if E1 concludes weights DO update)

- [ ] T2.1 Refactor forward pass to read from `paramTs`  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Deps: T1.3, T1.4, T1.5
  Acceptance: Any forward-pass read that currently hits a stale cache
  is rerouted to read from the live `paramTs` tensors. Minimal change;
  no wholesale refactor.

- [ ] T2.2 Rebuild and validate locally on CPU  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T2.1
  Acceptance: `go test ./timeseries/... -race` stays green. CPU bench
  still converges.

### Epic E3: Fix on the write path (if E1 concludes weights do NOT update)

- [ ] T3.1 Patch `SetData` or replace it with an engine copy  Owner: TBD  Est: 1.5h  verifies: [UC-TS01]
  Deps: T1.3, T1.6
  Acceptance: The AdamW writeback path at
  `timeseries/patchtst_gpu_train.go:1152` reliably updates device
  memory. If `SetData` has a silent-skip path, fix it; otherwise
  replace the call with an explicit `engine.Copy(ctx, hostTensor,
  deviceTensor)` or equivalent.

- [ ] T3.2 Rebuild and validate locally on CPU  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T3.1
  Acceptance: CPU tests and CPU bench remain green.

### Epic E4: Regression tests

- [ ] T4.1 Single-step GPU/CPU parity test  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Acceptance: New test in `timeseries/`: builds a PatchTST with fixed
  seeds, constructs 2 samples x 1 channel deterministic input, runs one
  forward + backward + AdamW step on the GPU engine, asserts
  `paramTs[0].Data()` differs from a snapshot taken before the step by
  more than a small epsilon. Skippable with `t.Skip` when the GPU
  engine is unavailable. Must fail on a pre-fix checkout.

- [ ] T4.2 End-to-end tiny training regression test  Owner: TBD  Est: 2h  verifies: [UC-TS01]
  Deps: T4.1
  Acceptance: New test that trains on 256 samples x 2 channels for 2
  epochs and asserts `loss[1] < loss[0] * 0.9`. Runs in <= 30s on GPU,
  skippable on CPU-only CI. Must fail on a pre-fix checkout.

- [ ] T4.3 Wire both tests into CI skip guards  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T4.1, T4.2
  Acceptance: Tests use the existing GPU-availability skip pattern and
  do not break CPU CI. `go test ./timeseries/...` passes locally on CPU.

- [ ] T4.4 Lint and vet  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T4.1, T4.2, T4.3, (T2.1 or T3.1)
  Acceptance: `go vet ./...` clean, `golangci-lint run ./timeseries/...`
  clean.

### Epic E5: Validate on DGX via Spark

- [ ] T5.1 Rebuild bench_train on DGX from fix branch  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T2.1 or T3.1
  Acceptance: `cd ~/zerfoo && git fetch && git checkout <fix-branch> &&
  /usr/local/go/bin/go build -o /tmp/bench_train_new ./cmd/bench_train`
  then `sudo install /tmp/bench_train_new /var/lib/zerfoo/bin/bench_train`.
  md5sum recorded in the bench.

- [ ] T5.2 Smoke validation (1K x 5ch x 2ep)  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T5.1
  Acceptance: `scripts/bench-spark.sh -samples 1000 -channels 5 -epochs 2 -cleanup`
  shows `engine: GPU (CUDA)` and a decreasing loss across epochs.

- [ ] T5.3 Regression validation (5K x 10ch x 3ep)  Owner: TBD  Est: 0.5h  verifies: [UC-TS01]
  Deps: T5.2
  Acceptance: `scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3 -cleanup`
  shows a strictly decreasing loss with final/initial ratio <= 0.1.
  Loss trajectory must not include the frozen value `0.268357` anywhere
  if weights are initialized with the default seed.

- [ ] T5.4 SSH session leak check  Owner: TBD  Est: 0.25h  verifies: [infrastructure]
  Deps: T5.3
  Acceptance: `ssh ndungu@192.168.86.250 'who | wc -l'` before and after
  T5.1..T5.3 must be unchanged.

### Epic E6: Documentation

- [ ] T6.1 Devlog writeup  Owner: TBD  Est: 0.5h  delivers: [devlog entry with root cause + fix + bench delta]
  Deps: T5.3
  Acceptance: New entry appended to `docs/devlog.md` covering the
  investigation, the diagnostic that pinned the hole (the hash
  delta from T1.3), the fix, the regression test, and a
  before/after bench snippet.

- [ ] T6.2 Remove the instrumentation commit  Owner: TBD  Est: 0.25h  verifies: [UC-TS01]
  Deps: T6.1
  Acceptance: The hash logging from T1.1/T1.2 is reverted or gated
  behind a debug flag so it does not run in normal training.

## Parallel Work

### Tracks
| Track | Tasks | Description |
|---|---|---|
| A: Diagnose | T1.1 -> T1.3 (sequential) | Instrumentation + bench run. Gates E2 vs E3 choice. |
| B: Code audit | T1.4, T1.5, T1.6 | Can run fully in parallel with Track A. |
| C: Tests | T4.1, T4.2, T4.3 | Can begin in parallel once E1 has narrowed the hypothesis. |
| D: Fix | E2 or E3 (exclusive) | Chosen based on T1.3 outcome. |
| E: Validate | T5.1 -> T5.4 (sequential) | Depends on fix and tests. |
| F: Docs | T6.1, T6.2 | After E5 completes. |

### Sync Points
- T1.3 is the fork point: its result decides whether E2 or E3 executes.
- T4.4 (lint) must wait for the fix (T2.1 or T3.1) AND the tests.
- T5.1 must wait for the fix to be committed on a branch.

### Waves

#### Wave 1: Instrument + audit (4 agents)
- [ ] T1.1 Weight-hash helper
- [ ] T1.4 Audit extractGPUParams aliasing
- [ ] T1.5 Audit forward-pass param access
- [ ] T1.6 Audit SetData semantics

#### Wave 2: Run diagnostic (1 agent, gated)
- [ ] T1.2 Wire hashes into training loop
- [ ] T1.3 Run instrumented bench on DGX (depends on T1.2)

Only one agent here because T1.2 and T1.3 are a short sequential pair
and the outcome branches the plan. Treat this as a single wave with two
sequential steps.

#### Wave 3: Fix + parity test (3 agents)
- [ ] T2.1 or T3.1 (exactly one, chosen from Wave 2 outcome)
- [ ] T4.1 Single-step GPU/CPU parity test
- [ ] T4.2 End-to-end tiny training regression test

#### Wave 4: Local validation + lint (3 agents)
- [ ] T2.2 or T3.2 (whichever branch E2/E3 was taken)
- [ ] T4.3 Wire tests into CI skip guards
- [ ] T4.4 Lint and vet

#### Wave 5: DGX validation (1 agent, sequential internally)
- [ ] T5.1 Rebuild bench_train on DGX
- [ ] T5.2 Smoke 1K x 5 x 2
- [ ] T5.3 Regression 5K x 10 x 3
- [ ] T5.4 SSH session leak check

All four T5.* steps run as a single sequential chain inside one agent
because each step is fast (<1 min) and depends strictly on the previous.

#### Wave 6: Docs (2 agents)
- [ ] T6.1 Devlog writeup
- [ ] T6.2 Remove instrumentation

### Dependency-Minimization Notes
- T1.4, T1.5, T1.6 all run in parallel with T1.1 in Wave 1 because the
  audits do not need the instrumentation to land.
- T4.1 and T4.2 can begin writing immediately in Wave 3 even before
  the fix is finalized, because the test's structure is independent of
  which hole the bug is in.
- E2 and E3 are mutually exclusive. The plan reserves only one wave slot
  for whichever turns out to be needed.

## Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|---|---|---|---|
| M1 | Root cause identified | Wave 2 complete | T1.3 pre/post hashes analyzed; the hole is named in a devlog draft. |
| M2 | Fix committed on a branch | Wave 3 complete (code side) | Either T2.1 or T3.1 committed; CPU tests still green. |
| M3 | Regression tests merged | Wave 4 complete | T4.1, T4.2, T4.3 in place; `go test ./timeseries/...` green on CPU. |
| M4 | GPU convergence validated on DGX | Wave 5 complete | 5K x 10ch x 3ep bench shows >= 90% loss reduction; SSH sessions unchanged. |
| M5 | Investigation closed | Wave 6 complete | Devlog entry landed; instrumentation removed/gated; PR merged to main. |

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | T1.3 reveals hashes ARE different, meaning write path is fine but forward pass reads stale copies AND our mental model is wrong | Plan assumes binary fork E2/E3; actual root cause may be subtler | Med | T1.4 and T1.5 audits run in parallel with T1.3 so we have code-level evidence on both paths by the time Wave 2 finishes. |
| R2 | The fix needs an ztensor change (e.g. SetData semantics), which crosses repo boundaries | Scope creep | Med | Scoped to ztensor/tensor only. Commit in ztensor, tag, update zerfoo go.mod. Budgeted in T3.1. |
| R3 | Regression test is flaky because of GPU nondeterminism | CI instability | Low | Use a fixed seed, deterministic kernels where possible, tolerance bounds on the loss ratio. |
| R4 | bench_train on DGX gets rebuilt from main by another session, masking the fix | Confusing bench results | Low | T5.1 records the md5sum of the deployed binary before T5.2/T5.3 run. |
| R5 | 5K x 10ch x 3ep still does not converge even after the fix because there is a THIRD hole | Plan loops | Low | If T5.3 fails, return to Wave 1 with new instrumentation rather than patching blindly. |
| R6 | SSH session accumulation from debugging commands | DGX reboot | Low | Use Spark API for anything loopy; ssh only for short foreground commands. |

## Operating Procedure

### Definition of Done (per task)
- Code committed in small focused commits, Conventional Commits format.
- `go test ./timeseries/... -race` green on CPU.
- Lint green on changed packages.
- Bench validation (D5) passes on DGX via Spark before the fix PR is merged.

### Review and QA
- One reviewer on the fix PR (code path is subtle; another set of eyes
  on tensor aliasing is cheap insurance).
- The regression tests in E4 are the primary QA gate. If either test
  passes on a pre-fix checkout, it is not actually catching the bug and
  must be strengthened.

### Always Add Tests / Lint
- T4.1 and T4.2 are the regression tests. Do not skip them.
- T4.4 is the lint task. `go vet ./...` + `golangci-lint run
  ./timeseries/...`.

### Commit Discipline
- Instrumentation, fix, tests, and devlog each in their own commits.
- Use `fix(timeseries):`, `test(timeseries):`, `docs(devlog):` prefixes.
- No commit straddles directories.

### Versioning
- This plan does not produce a tagged release on its own. When merged to
  main, the next routine release-please bump will include it.

## Progress Log

### Change Summary 2026-04-08
- Plan created after commissioning Spark empirically showed that
  commit `f29c93bd` (merged from `fix/gpu-train-cpu-mirror`) is
  necessary but insufficient to resolve the GPU training convergence
  regression.
- Zero completed tasks. All T1.x..T6.x are new.
- Five hypotheses listed in Context, ranked by suspicion, with a
  binary diagnostic (T1.3 weight-hash delta) that forks the plan into
  E2 or E3.
- Six epics, 22 tasks across 6 waves, max 4 agents per wave per
  user feedback on agent parallelism limits.
- No ADR created. If the fix turns out to require changing `SetData`
  semantics in ztensor, an ADR there may be appropriate but that
  decision is deferred until Wave 2 results are in.
- Linked context: `docs/devlog.md` 2026-04-08 entry,
  `docs/plans/spark-bench-runner.md`, `docs/adr/083-spark-bench-runner.md`.

## Hand-off Notes

- **What you need to know:** `fix/gpu-train-cpu-mirror` fixes a real
  AdamW writeback bug but is not the whole story. On GPU,
  PatchTST loss is frozen at exactly `0.268357` for
  5K x 10ch x 3ep, matching the pre-fix reproduction byte-for-byte.
  CPU works fine with the same binary. The plan's entry point is
  T1.1 (hash helper) + T1.3 (run on DGX) to tell us whether weights
  update on device at all.
- **Bench runner:** `scripts/bench-spark.sh -samples N -channels C
  -epochs E`. Spark v1.6.1 on DGX 192.168.86.250. Do NOT use
  interactive `ssh dgx bench_train`.
- **DGX access:** `ssh ndungu@192.168.86.250`. Repo at `~/zerfoo`.
  Go at `/usr/local/go/bin/go`. Bench staged at
  `/var/lib/zerfoo/bin/bench_train`.
- **Key file:** `timeseries/patchtst_gpu_train.go`. Read lines
  51-170 (`extractGPUParams`), 565-625 (CPU mirror setup), and
  1113-1170 (AdamW step + writeback) before starting.
- **Prior context in memory:** #1776, #1789, #1791, #1793, #1812,
  #1813, and the 2026-04-08 devlog entry. Treat #1813's "verified
  GPU parameter synchronization fix" as premature; the verification
  did not exercise the path that is actually broken.
- **When you are done:** The bench wave from `docs/plan.md`
  (T50.5.2, T51.5.2, T54.4.1, T63.2.1, T61.3.2) is fully
  unblocked.

## Appendix

### Useful one-liners

Rebuild and restage the bench binary on DGX after committing a fix:
```
ssh ndungu@192.168.86.250 'cd ~/zerfoo && git fetch && git checkout <branch> && /usr/local/go/bin/go build -o /tmp/bench_train_new ./cmd/bench_train && sudo install /tmp/bench_train_new /var/lib/zerfoo/bin/bench_train && md5sum /var/lib/zerfoo/bin/bench_train'
```

Submit the regression bench:
```
ssh ndungu@192.168.86.250 'cd ~/bench-spark && SPARK_HOST=localhost:8080 ./scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3 -cleanup'
```

Inspect a stuck pod:
```
ssh ndungu@192.168.86.250 'curl -sf http://localhost:8080/api/v1/pods/<name> | python3 -m json.tool'
```

### Expected convergence signature (from CPU)

```
epoch  1: loss=0.115235 ok
epoch  2: loss=0.001384 ok
epoch  3: loss=0.000863 ok
convergence: OK (0.115235 -> 0.000863, 99.3% reduction)
```

### Current broken signature (GPU, pre-fix of this plan)

```
epoch  1: loss=0.268357 ok
epoch  2: loss=0.268357 ok
epoch  3: loss=0.268357 ok
convergence: FAILED (loss did not decrease: 0.268357 -> 0.268357)
```
