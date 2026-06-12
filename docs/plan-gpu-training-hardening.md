# Zerfoo/ztensor: GPU Training-Stack Hardening (Porting PyTorch's Lessons)

> **Status:** ACTIVE
> **Created:** 2026 06 10
> **Repos:** github.com/zerfoo/zerfoo (this repo) and github.com/zerfoo/ztensor.
> Most implementation lands in ztensor (graph, arena, kernels); zerfoo owns the
> layer migrations and consumes the harnesses. The end-to-end validation target
> is the Wolf CrossAsset model (github.com/feza-ai/wolf).
> **Decision rationale:** docs/adr/091-gradcheck-pytorch-oracle-verification.md
> (verification strategy) and ztensor docs/adr/006-save-for-backward-arena-pinning.md
> (memory/autograd contract).

---

## Context

### Problem statement

GPU f32 training on the DGX GB10 diverges (NaN or runaway loss) for the Wolf
CrossAsset model while the identical model, data, and seed train cleanly on
CPU (accuracy 0.6765). One investigation (Wolf devlog 2026 06 09/10, Wolf ADR
072) found and fixed eight real bugs -- arm64 NEON softmax NaN, fast-math tanh
overflow, LayerNorm cached-variance arena corruption, AdamW f32 second moment,
gradient-zeroing arena corruption, non-stationary input features, attention
conditioning, and a QK-norm cached-intermediate bug -- and GPU f32 training
STILL diverges. Every bug was discovered as a production NaN on the DGX.

The conclusion is framework-level, not bug-level: Zerfoo/ztensor's GPU
training path is immature in exactly the two areas PyTorch spent a decade
hardening, and PyTorch (open source, BSD-3) documents the solutions:

1. **Memory/autograd lifetime.** ztensor's arena reuses buffers within a step;
   nodes cache forward intermediates in struct fields and read them in
   Backward; the arena overwrites them first. PyTorch's save_for_backward +
   caching-allocator design makes this class impossible.
2. **Kernel numerics.** Hand-rolled kernels compile with global
   --use_fast_math (internal/cuda/kernels/Makefile line 7), reductions
   accumulate in arbitrary order, and there is no external numerical
   reference. PyTorch never uses global fast-math, does max-subtraction
   softmax, accumulates in fp32+ with pairwise ordering, and validates every
   op's backward with finite-difference gradcheck.

PyTorch f32 trains this exact architecture to ~0.66 on the same GB10, so the
hardware and precision are sufficient -- the gap is the framework.

### Objectives

1. Per-op verification harnesses (gradcheck, GPU-vs-CPU parity under arena
   stress, PyTorch-as-oracle) so numerics and lifetime bugs fail as named
   tests, not as DGX NaNs.
2. A save-for-backward contract with arena pinning that structurally
   eliminates the live-tensor corruption class (zerfoo#842, zerfoo#845, the
   Wolf QK-norm bug).
3. Kernels free of global fast-math, with fp32 fixed-order reduction
   accumulation, each gated by the oracle harness.
4. A deterministic-reductions debug mode for reproducible GPU runs.
5. END GOAL: the Wolf CrossAsset model trains a GB10 f32 fold with no NaN and
   fold-0 accuracy within 1pp of the CPU baseline (0.6765), unblocking Wolf's
   speed-parity plan.

### Non-goals

- New model features, new ops, or speed optimization (Wolf's
  plan-pytorch-speed-parity.md owns speed; this plan unblocks it).
- Multi-GPU; bf16/tf32 mixed precision (separate plan; and per Wolf ADR 072
  bf16 does not address overflow-range issues).
- Porting PyTorch code verbatim. We port the architectural patterns
  (BSD-3 makes studying and reimplementing them unambiguously fine).
- Replacing cuBLAS matmul (already the same NVIDIA library PyTorch uses).

### Constraints and assumptions

- Single GB10 at the DGX; ALL GPU jobs run via Spark
  (http://192.168.86.250:8080) and MUST serialize -- concurrent GPU pods
  corrupt each other. CI has no GPU, so GPU harness runs are Spark pods, not
  CI jobs. Memory limit on every pod.
- PyTorch oracle runs inside nvcr.io/nvidia/pytorch:26.02-py3 (arm64, sm_121)
  on the GB10 -- the same image as Wolf's reference benchmark. Pin the tag.
- Branch -> PR -> CI green -> rebase-merge in both repos; release-please tags;
  Wolf consumes via go get after tags ship.
- Existing precedent: zerfoo timeseries/*_backward_test.go contains ad-hoc
  finite-difference checks -- generalize, do not duplicate.
- Key existing APIs: ztensor graph.Node.Backward already receives the live
  `inputs ...` (graph/node.go:22); arena entry points are
  compute.GPUEngine.ResetPool (gpu_engine.go:324) and MarkStepBoundary
  (step_scope.go:75); ArenaPool.Alloc (internal/cuda/arena.go:195).

### Success metrics

| Metric | Target | How measured |
|---|---|---|
| Harness catches known bug classes | each harness has a red/green regression fixture from a historically-fixed bug | fixture tests in CI / Spark harness run |
| gradcheck coverage | every graph.Node op in ztensor + every zerfoo layer Backward registered | OpInfo registry count vs op inventory |
| Lifetime safety | zero pinned-buffer reuse; poison-mode full-suite run clean | ZTENSOR_ARENA_POISON=1 parity-harness run on GB10 |
| Kernel numerics | all kernels pass oracle gate without global fast-math | oracle harness report, per-op tolerances |
| Determinism | two seeded GB10 epochs bitwise-identical losses | ZTENSOR_DETERMINISTIC=1 double-run diff |
| END GOAL | Wolf GB10 f32 fold: 0 NaN, acc within 1pp of CPU 0.6765, twice | Wolf results JSON, two consecutive runs |

---

## Discovery Summary

Engineering work. 7 new use cases (UC-GH-001..007, all P0/P1, all PLANNED)
appended to .claude/scratch/usecases-manifest.json alongside the 3 legacy
UC-EXT entries. Key findings:

- No general gradcheck exists; ad-hoc finite-difference code in
  timeseries/timemixer_backward_test.go and siblings proves the approach in
  this codebase and should be generalized.
- --use_fast_math is global (kernels Makefile NVCC_FLAGS) and three .cu files
  reference it (elementwise.cu, fused_encoder_fwd.cu, fused_encoder_bwd.cu).
  The tanh overflow (ztensor#125) was a direct consequence.
- The arena has no pin/unpin facility; ResetPool/MarkStepBoundary reuse
  everything unconditionally. At least three shipped bugs (zerfoo#842,
  zerfoo#845, Wolf QK-norm backward) are the same use-after-reuse class.
- graph.Node.Backward already receives live `inputs ...`, so recompute-based
  migration is possible today; the save-for-backward contract (ztensor ADR
  006) is the structural fix for intermediates that are expensive to
  recompute.
- A cached-for-backward audit must be field-based, not comment-based: grep
  comments found only 2 zerfoo layers, but the pattern (struct fields written
  in Forward, read in Backward) is known to be wider (softmax, swiglu,
  dropout masks, attention weights).

Reference: .claude/scratch/usecases-manifest.json.

---

## Scope and Deliverables

### In scope
- The three harnesses + OpInfo registry (ztensor core, zerfoo consumers).
- Save-for-backward contract, arena pin/unpin, poison-on-reset mode, and
  migration of all node/layer Backward implementations in both repos.
- Kernel numerics audit of internal/cuda/kernels (elementwise, softmax,
  reductions, fused encoder) against PyTorch conventions.
- Deterministic-reductions debug mode.
- End-to-end Wolf CrossAsset GB10 f32 validation + releases + Wolf bump.

### Out of scope
- Speed work, new ops, mixed precision, multi-GPU, PJRT/StableHLO paths.

### Deliverables

| ID | Description | Owner | Acceptance |
|---|---|---|---|
| D1 | gradcheck package + OpInfo registry (ztensor) | TBD | every registered op's Backward validated vs f64 central differences in CI; a planted wrong-Jacobian fixture fails red |
| D2 | GPU-vs-CPU parity harness with arena-stress schedules | TBD | runs as a Spark pod on GB10; reproduces the cached-intermediate class (pre-fix LayerNorm-style fixture) red, current code green |
| D3 | PyTorch oracle harness (exchange format + NGC runner) | TBD | per-op fwd/bwd diffs vs torch on GB10 within tolerances; pre-fix fast-math tanh fixture fails red |
| D4 | Arena poison-on-reset mode (ZTENSOR_ARENA_POISON) | TBD | use-after-reset explodes at the corruption site in a demo test |
| D5 | Save-for-backward contract + arena pin/unpin | TBD | pinned buffers never reused; contract documented; ztensor ADR 006 honored |
| D6 | All Backward impls migrated (ztensor nodes + zerfoo layers) | TBD | field-based audit complete; each impl uses SaveForBackward or live-inputs recompute; poison-mode suite green |
| D7 | Kernels without global fast-math + fp32 tree reductions | TBD | oracle gate green for every kernel; perf delta measured and recorded |
| D8 | ZTENSOR_DETERMINISTIC mode | TBD | two seeded GB10 epochs bitwise-identical |
| D9 | Wolf GB10 f32 clean fold on hardened stack | TBD | 0 NaN, acc within 1pp of 0.6765, two consecutive runs; devlog entry |
| D10 | Releases + Wolf bump | TBD | ztensor + zerfoo tags shipped; Wolf go.mod bumped; Wolf speed-parity plan unblocked |

---

## Checkable Work Breakdown

### E1: Verification harnesses (the foundation)

**Component:** testing

Acceptance: all three harnesses operational; each encodes at least one
historically-fixed bug as a red/green regression fixture; gradcheck runs in CI,
GPU harnesses run as Spark pods.

- [x] T1.1 gradcheck core + OpInfo registry (ztensor)  2026 06 10  (DONE ztensor#129: 26 ops, wrong-Jacobian red-proof)
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-001]  kind: agent  blocked-by: []
  - New ztensor package (testing/gradcheck or graph/gradcheck): for a
    graph.Node under test, compute analytic Backward and compare against
    float64 central differences on CPU, per input and per parameter.
    OpInfo-style registry: op constructor, representative shapes, input
    domains (e.g. positive-only for sqrt/log), tolerances. Register all
    existing ztensor graph nodes. Plant a deliberately-wrong-Jacobian fixture
    and assert gradcheck fails it (the red proof).
  - Generalize, then retire, the ad-hoc finite-difference code in zerfoo
    timeseries tests (retirement is T1.6).
  - Acceptance: `go test ./.../gradcheck/...` green in CI; wrong-Jacobian
    fixture red; every ztensor graph node registered.
  - Decision rationale: docs/adr/091-gradcheck-pytorch-oracle-verification.md.

- [x] S1.1.1 Unit tests + lint for gradcheck core  2026 06 10  (DONE in #129)
       Owner: TBD  Est: 2h  verifies: [UC-GH-001]  kind: agent  blocked-by: [T1.1]
  - Table-driven tests for the checker itself (perturbation sizing, tolerance
    handling, multi-input ops). gofmt/go vet/build clean.

- [x] T1.2 GPU-vs-CPU parity harness with arena-stress schedules (ztensor)  2026 06 10  (DONE ztensor#133: schedules a/b, StressEngine CI variant, red-proof red; GB10 runs DGX-deferred)
       Owner: TBD  Est: 2d  verifies: [UC-GH-002]  kind: agent  blocked-by: [T1.1]
  - Harness that runs the same op set CPU-f32 vs GPU-f32, forward AND
    backward, comparing within per-op tolerances. Critically, it must run
    INTERLEAVED schedules (opA.Forward, opB.Forward, ..., opA.Backward) with a
    deliberately small arena to force buffer reuse between an op's forward and
    its backward -- the schedule shape that exposes cached-intermediate
    corruption. Packaged to run as a Spark pod on the GB10 (no GPU in CI).
    Include a regression fixture node that caches an intermediate the way
    pre-fix LayerNorm did, and assert the harness (with poison mode, T1.4)
    catches it red.
  - Acceptance: Spark-pod run green on current code; cached-intermediate
    fixture red; results emitted as JSON for the devlog.

- [x] S1.2.1 Tests + lint for the parity harness  2026 06 10  (DONE in #133)
       Owner: TBD  Est: 2h  verifies: [UC-GH-002]  kind: agent  blocked-by: [T1.2]

- [x] T1.3 PyTorch-oracle harness: exchange format + NGC runner (cross-repo)  2026 06 10  (DONE ztensor#131: bundle format, NGC runner, Spark pod, tanh red-proof; GB10 runs DGX-deferred)
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-003]  kind: agent  blocked-by: []
  - Define a tensor exchange format (npz or raw+JSON manifest: op name,
    shapes, dtypes, seeds, inputs, upstream gradients). Go side dumps
    inputs/outputs/gradients for registered ops; a small Python runner
    (test-infrastructure only) executes the same ops in torch inside
    nvcr.io/nvidia/pytorch:26.02-py3 on the GB10 via a Spark pod and emits
    diffs. Per-op tolerance table. Fixture: the pre-fix fast-math tanh kernel
    behavior must fail red against torch.tanh.
  - Acceptance: oracle run on GB10 produces a per-op diff report; tanh fixture
    red against the unfixed kernel, green against the fixed one.

- [x] S1.3.1 Tests + lint for the exchange format (round-trip)  2026 06 10  (DONE in #131)
       Owner: TBD  Est: 1h  verifies: [UC-GH-003]  kind: agent  blocked-by: [T1.3]

- [x] T1.4 Arena poison-on-reset debug mode (ztensor)  2026 06 10  (DONE ztensor#130: ZTENSOR_ARENA_POISON, Reset/FreeArena hooks, capture-safe)
       Owner: TBD  Est: 4h  verifies: [UC-GH-002, UC-GH-004]  kind: agent  blocked-by: []
  - ZTENSOR_ARENA_POISON=1: ResetPool/MarkStepBoundary and intra-step reuse
    fill freed regions with NaN sentinels (f32 0x7FC00000 pattern) before
    handing them out, so any read of a stale cached buffer produces an
    immediate, deterministic NaN at the corruption site. Off by default; zero
    cost when unset. Demo test: a node that reads a cached tensor after
    ResetPool explodes under poison, passes without.
  - Acceptance: demo test red under poison / green without; documented in
    ztensor docs; referenced by ADR 006.

- [x] S1.4.1 Tests + lint for poison mode  2026 06 10  (DONE in #130)
       Owner: TBD  Est: 1h  verifies: [UC-GH-004]  kind: agent  blocked-by: [T1.4]

- [x] T1.6 Migrate zerfoo ad-hoc finite-difference tests to gradcheck  2026 06 11  (DONE #856: 6 OpInfo entries via flatParamNode adapter; bespoke timeseries FD checkers retired)
       Owner: TBD  Est: 4h  verifies: [UC-GH-001]  kind: agent  blocked-by: [T1.1]
  - Register the timeseries ops (timemixer, patchtst, itransformer backward
    tests) in the OpInfo registry; replace bespoke checkers with gradcheck
    calls; keep test intent and tolerances.
  - Acceptance: zerfoo suite green; no remaining bespoke finite-difference
    implementations.

### E2: Save-for-backward contract + arena pinning

**Component:** memory

Acceptance: the contract is implemented per ztensor ADR 006, every Backward in
both repos is migrated, and the Wolf-style per-sample-reset stress test passes
under poison mode.

- [x] T2.1 SaveForBackward API + graph lifetime ownership (ztensor)  2026 06 10  (DONE ztensor#132: per-node Saver via Builder, SaverAware)
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-004]  kind: agent  blocked-by: []
  - Add the contract to the ztensor graph: nodes call SaveForBackward(t...)
    during Forward (via a context or a graph-provided saver); the graph
    records the saved set per node; saved tensors are pinned (T2.2) at
    registration and unpinned when that node's Backward returns (or at
    MarkStepBoundary for forward-only passes). Keep `inputs ...` recompute as
    the documented alternative for cheap intermediates.
  - Acceptance: API documented; graph unit tests cover save -> backward ->
    unpin lifecycle, including multi-consumer and error paths.
  - Decision rationale: ztensor docs/adr/006-save-for-backward-arena-pinning.md.

- [x] S2.1.1 Unit tests + lint for the contract lifecycle  2026 06 10  (DONE in #132)
       Owner: TBD  Est: 2h  verifies: [UC-GH-004]  kind: agent  blocked-by: [T2.1]

- [x] T2.2 Arena Pin/Unpin (refcounted) honored by ResetPool and reuse (ztensor)  2026 06 10  (DONE ztensor#132: raise-the-floor Reset, deferred frees, PinnedBytes)
       Owner: TBD  Est: 1d  verifies: [UC-GH-004]  kind: agent  blocked-by: []
  - ArenaPool gains Pin(ptr)/Unpin(ptr) refcounts; ResetPool, MarkStepBoundary,
    and free-list reuse skip pinned buffers; poison mode must not poison
    pinned regions. Track and expose the pinned high-water mark for memory
    monitoring.
  - Acceptance: arena unit tests: pinned buffer survives ResetPool; unpinned
    buffer is reused/poisoned; refcount over/underflow guarded.

- [x] S2.2.1 Tests + lint for pin/unpin (incl. poison interplay)  2026 06 10  (DONE in #132)
       Owner: TBD  Est: 2h  verifies: [UC-GH-004]  kind: agent  blocked-by: [T2.2, T1.4]

- [x] T2.3 Field-based audit + migration of all Backward impls (zerfoo + ztensor)  2026 06 10  (DONE zerfoo#848: 45 types audited, 26 migrated, 18 follow-up under #847; found+fixed incomplete #842 f64-path fix and PolynomialExpansion stale-cache bug)
       Owner: TBD  Est: 2d  verifies: [UC-GH-004]  kind: agent  blocked-by: [T2.1, T2.2]
  - Audit EVERY graph.Node / layer Backward in both repos for struct fields
    written in Forward and read in Backward (do not rely on comments; known
    minimum: zerfoo layers/activations/softmax.go, swiglu.go, dropout masks,
    attention weights, normalization stats). For each: migrate to
    SaveForBackward, or recompute from the live `inputs ...`. Produce the
    audit table (op, cached fields, resolution) in the PR description.
  - Acceptance: audit table complete; both repos' suites green; parity harness
    (T1.2) under ZTENSOR_ARENA_POISON=1 green across the registry.

- [x] S2.3.1 Poison-mode full-suite run on GB10 (Spark, serial)  2026 06 11  (DONE ztensor#140: Spark pod ztensor-parity-4b4d759c, ZTENSOR_ARENA_POISON=1, 64MiB arena -- parity schedules 26/26 + 26/26 green, GPU red-proof red, Wolf-pattern training loop finite + CPU-parity; ztensor devlog 2026-06-11)
       Owner: TBD  Est: 2h  verifies: [UC-GH-004]  kind: agent  blocked-by: [T2.3]
  - The proof: parity harness + a small training loop with per-sample
    ResetPool (the Wolf gr-12 pattern: accumulate gradients across a batch,
    reset arena every sample) runs clean under poison.

- [x] T2.4 Wolf-pattern integration stress test in ztensor  2026 06 12  (DONE ztensor#141: attention graph Q@K^T->softmax->@V + residual/RMS-norm on SaveForBackward, reset-between-fwd-bwd + per-sample ResetPool, persistent Parameters; StressCI green, GB10 pod ztensor-parity-853a7fa1 green under poison, red-proof verified)
       Owner: TBD  Est: 4h  verifies: [UC-GH-004, UC-GH-007]  kind: agent  blocked-by: [T2.1, T2.2]
  - A ztensor integration test reproducing the exact hazard schedule that bit
    Wolf: multi-sample gradient accumulation into persistent Parameters with
    ResetPool between samples, optimizer step per batch, on a small synthetic
    graph with attention-like nodes. Runs CPU in CI; GPU variant in the Spark
    harness.
  - Acceptance: test exists, green on CPU in CI and GPU on GB10 under poison.

### E3: Kernel numerics audit

**Component:** kernels

Acceptance: kernels build without global --use_fast_math, reductions accumulate
fp32 fixed-order, every kernel passes the oracle gate, perf delta recorded.

- [ ] T3.1 Remove global --use_fast_math; selective per-kernel intrinsics only
       Owner: TBD  Est: 1d  verifies: [UC-GH-005]  kind: agent  blocked-by: [T1.3]
  - Drop --use_fast_math from NVCC_FLAGS (internal/cuda/kernels/Makefile:7).
    Re-enable specific fast intrinsics per kernel ONLY where the oracle gate
    proves equivalence within tolerance (e.g. __expf in softmax after
    max-subtraction may pass; tanh famously does not). Rebuild sm_121
    libkernels.so; run the oracle suite; measure per-kernel and end-to-end
    perf delta on the GB10 and record both in the devlog.
  - Acceptance: oracle green for all kernels; perf delta documented; the tanh
    saturation clamp (ztensor#125) retained as defense-in-depth.

- [ ] S3.1.1 Oracle gate run + devlog entry for the fast-math change
       Owner: TBD  Est: 2h  verifies: [UC-GH-005]  kind: agent  blocked-by: [T3.1]

- [ ] T3.2 Reduction-accumulation audit: fp32 fixed-order trees
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-005, UC-GH-006]  kind: agent  blocked-by: [T1.3]
  - Audit every reduction (ReduceSum/Sum, softmax denominator, LayerNorm/RMS
    stats, norms in optimizer clipping) for accumulation order and dtype.
    Convert to fixed-order pairwise/tree accumulation in fp32 (acc-type
    pattern). This both improves accuracy and shrinks run-to-run variance --
    the prerequisite for T4.1 determinism.
  - Acceptance: oracle diffs for reduction ops tighten measurably; documented
    accumulation policy in ztensor docs/design.md.

- [ ] S3.2.1 Tests + lint for reduction changes
       Owner: TBD  Est: 2h  verifies: [UC-GH-005]  kind: agent  blocked-by: [T3.2]

- [ ] T3.3 Oracle-gate every remaining kernel; fix divergences
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-005]  kind: agent  blocked-by: [T1.3, T3.1]
  - Sweep internal/cuda/kernels (elementwise, argmax, dequant paths used in
    training, fill, scaled softmax) through the oracle; fix any out-of-
    tolerance op (eps placement, max-subtraction, saturation). Record the
    per-op tolerance table as the standing gate.
  - Acceptance: oracle suite green across the kernel inventory; tolerance
    table committed.

- [ ] T3.4 Fused encoder fwd/bwd kernels: same treatment
       Owner: TBD  Est: 1d  verifies: [UC-GH-005]  kind: agent  blocked-by: [T3.1]
  - fused_encoder_fwd.cu / fused_encoder_bwd.cu reference fast-math; audit
    against the unfused chain AND torch within 1e-6 fp32 (the existing
    equivalence bar), under the new flags.
  - Acceptance: fused-vs-unfused and fused-vs-torch equivalence green.

### E4: Deterministic-reductions debug mode

**Component:** kernels

- [ ] T4.1 ZTENSOR_DETERMINISTIC=1 mode
       Owner: TBD  Est: 1.5d  verifies: [UC-GH-006]  kind: agent  blocked-by: [T3.2]
  - Env-gated mode selecting fixed-order reduction kernels everywhere,
    disabling any atomics-based accumulation paths, and setting deterministic
    cuBLAS configuration where available. Document scope honestly (what is and
    is not covered). Debug tool, off by default.
  - Acceptance: two identical seeded small training runs on the GB10 produce
    bitwise-identical per-epoch losses; documented in ztensor docs.

- [ ] S4.1.1 Tests + lint + double-run proof on GB10
       Owner: TBD  Est: 2h  verifies: [UC-GH-006]  kind: agent  blocked-by: [T4.1]

### E5: End-to-end validation, release, and Wolf unblock

**Component:** validation

Acceptance: the prize -- Wolf trains clean on GB10 f32 -- plus shipped tags.

- [x] T5.1 Wolf CrossAsset GB10 f32 clean fold on the hardened stack  2026 06 11  (DONE verify9 + verify9b on image 0066d970: two consecutive clean runs, zero NaN, epoch loss 0.778373, fold-0 acc 0.6760 vs CPU 0.6765 -- inside the 1pp gate; Bug 11 fixes ztensor#137/#138 + zerfoo#855/#857; Wolf devlog + ADR 072 resolution in wolf#200. NOTE: the formal blocked-by T3.1/T3.2/T3.3 kernel audit was overtaken by events -- the residual was allocator lifetime (host-access ordering + stale cross-epoch frees), not kernel numerics; T3.1/T3.2/T3.3 remain open as hardening.)
       Owner: TBD  Est: 1d  verifies: [UC-GH-007]  kind: agent  blocked-by: [T2.3, T3.1, T3.2, T3.3]
  - Bump a Wolf branch to the hardened ztensor/zerfoo, rebuild the image, run
    folds=2 epochs=1 seed=42 -precision=f32 -gpu (QK-norm on) on the GB10 via
    Spark. Require: 0 NaN and fold-0 acc within 1pp of the CPU baseline
    (0.6765), TWO consecutive runs (the historical failure was
    timing-sensitive). If it still diverges, run under
    ZTENSOR_ARENA_POISON=1 and ZTENSOR_DETERMINISTIC=1 to localize, fix, and
    repeat -- the harnesses must make the residual nameable.
  - Acceptance: two clean runs; Wolf devlog entry with numbers; Wolf ADR 072
    amended with the resolution.

- [x] T5.2 Release ztensor + zerfoo; bump Wolf  2026 06 11  (DONE: ztensor v1.11.0 (release PR #126; includes #134/#137/#138) and zerfoo v1.49.0 (release PR #804; includes #851/#852/#855/#857 + ztensor bump PR #858) shipped in dependency order; zerfoo#858 also fixed release-please to read the repo config -- the inline release-type ignored the v1.49.0 pin and proposed an unresolvable 4.0.0. Wolf bumped to both released tags in wolf#201 with the speed-parity plan T14.x marked unblocked (two clean GB10 folds, acc 0.6760, ~390 samples/s).)
       Owner: TBD  Est: 4h  verifies: [UC-GH-007]  kind: agent  blocked-by: [T5.1]
  - Ship release-please tags in dependency order (ztensor, then zerfoo); Wolf
    go.mod bump PR; Wolf speed-parity plan T14.x status updated to unblocked.
  - Acceptance: tags exist; Wolf main builds against them; speed-parity plan
    progress log updated.

- [x] T5.3 Documentation: design.md updates + devlog entries (both repos)  2026 06 11  (DONE ztensor#139: design.md host-access sync contract + reset-epoch frees + pinning + dst-form accumulation policy, devlog entry; zerfoo#859: design.md 7.5 per-op verification gates (gradcheck/parity/oracle, ADR-091) + 21.4 gradient accumulation policy (Graph.Engine()), devlog entry; wolf#200: ADR 072 resolution + Bug 11 devlog)
       Owner: TBD  Est: 3h  verifies: [infrastructure]  kind: agent  blocked-by: [T5.1]
  - ztensor docs/design.md: arena pinning + poison mode + accumulation policy
    (general terms). zerfoo docs/design.md: verification-harness usage for new
    ops. Devlog entries with the measured outcomes.

### E6: Always-on quality gates

**Component:** infrastructure

- [ ] T6.1 build / vet / gofmt / tests green per PR; GPU runs serialized
       Owner: TBD  Est: continuous  verifies: [infrastructure]  kind: agent
  - Both repos: go build ./..., go vet ./..., gofmt, full test suites with
    -race where applicable, green before each rebase-merge. One GPU Spark job
    at a time, ever.

---

## Parallel Work

| Track | Tasks | Concurrency notes |
|---|---|---|
| A: Harnesses | T1.1 -> T1.2, T1.6; T1.3; T1.4 | T1.1, T1.3, T1.4 are mutually independent (Wave 1) |
| B: Memory contract | T2.1; T2.2; then T2.3, T2.4 | T2.1 and T2.2 independent of Track A and of each other |
| C: Kernels | T3.1, T3.2, T3.3, T3.4 | all gated on T1.3 (oracle) for acceptance; T3.1/T3.2 parallel |
| D: Determinism | T4.1 | after T3.2 |
| E: Validation | T5.1 -> T5.2, T5.3 | strictly last; GPU-serial |

Sync points: the oracle (T1.3) gates Track C acceptance; poison mode (T1.4) +
pin/unpin (T2.2) gate the T2.3 migration proof; everything gates T5.1.

ALL GB10 runs (T1.2, S2.3.1, T2.4 GPU variant, T3.x oracle runs, T4.1 proof,
T5.1) serialize on the single GPU -- never fan those out.

### Waves

### Wave 1: Foundations (4 agents)
- [ ] T1.1 gradcheck core + OpInfo registry
- [ ] T1.3 PyTorch-oracle exchange format + NGC runner
- [ ] T1.4 Arena poison-on-reset mode
- [ ] T2.1 SaveForBackward API (design per ADR 006)

### Wave 2: Build-out (4 agents)
- [ ] T1.2 Parity harness with arena-stress schedules (GPU runs serial)
- [ ] T1.6 Migrate zerfoo ad-hoc finite-difference tests
- [ ] T2.2 Arena Pin/Unpin
- [ ] T3.1 Remove global fast-math (oracle-gated)

### Wave 3: Migration + audit (3 agents)
- [ ] T2.3 Backward-impl audit + migration (both repos)
- [ ] T3.2 Reduction-accumulation audit
- [ ] T3.3 Oracle-gate remaining kernels

### Wave 4: Hardening tail (3 agents)
- [ ] T2.4 Wolf-pattern stress test
- [ ] T3.4 Fused encoder kernels
- [ ] T4.1 Deterministic mode

### Wave 5: Validation + ship (1 agent, GPU-serial)
- [x] T5.1 Wolf GB10 f32 clean fold x2  2026 06 11  (DONE verify9/verify9b, image 0066d970, acc 0.6760)
- [x] T5.2 Releases + Wolf bump  2026 06 11  (DONE ztensor v1.11.0, zerfoo v1.49.0, wolf#201)
- [x] T5.3 Documentation  2026 06 11  (DONE ztensor#139, zerfoo#859, wolf#200)

---

## Timeline and Milestones

| ID | Milestone | Member tasks | Exit criteria |
|---|---|---|---|
| M1 | Harnesses operational with red/green fixtures | T1.1, T1.2, T1.3, T1.4, T1.6 | wrong-Jacobian, cached-intermediate, and fast-math-tanh fixtures all fail red; current code green |
| M2 | Lifetime class structurally closed | T2.1, T2.2, T2.3, T2.4 | poison-mode full-suite + Wolf-pattern stress green on GB10 |
| M3 | Kernels numerically gated | T3.1, T3.2, T3.3, T3.4 | oracle suite green, no global fast-math, perf delta recorded |
| M4 | Deterministic mode | T4.1 | bitwise-identical seeded double-run |
| M5 | Wolf trains on GB10 f32 (END GOAL) | T5.1, T5.2, T5.3 | 2x clean folds within 1pp of 0.6765; tags shipped; Wolf bumped |

Estimated wall-clock: M1 ~1 week, M2 ~1 week (overlapping M1 tail), M3 ~1
week, M4 2-3 days, M5 2-3 days. Total roughly 3-4 weeks, matching the original
scoping. Dates omitted; the GB10 single-GPU serialization is the main
scheduling constraint.

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Harness false confidence: single-op tests pass while sequences fail | High | Med | T1.2 mandates interleaved arena-stress schedules; fixtures reproduce the real bug classes, not toy cases |
| R2 | Removing fast-math regresses kernel performance materially | Med | Med | T3.1 measures per-kernel + end-to-end delta; selective per-kernel intrinsics re-enabled only under oracle proof; Wolf parity plan owns any remaining speed gap |
| R3 | Pinning raises arena high-water mark / overflow | Med | Low | GB10 has 128 GiB unified memory; pinned watermark exposed (T2.2); existing overflow fallback path applies |
| R4 | T5.1 still diverges after M2+M3 (a ninth, unknown class) | High | Med | poison + deterministic modes exist precisely to make any residual nameable; treat as a new named bug with harness fixture, not a replan |
| R5 | Cross-repo version skew (ztensor/zerfoo/Wolf) during the work | Med | Med | dependency-ordered releases (T5.2); Wolf pins commit SHAs on branches until tags ship |
| R6 | Single GB10 contention with Wolf's own work | Med | High | one GPU Spark job at a time, coordinated; CPU-side work dominates Waves 1-3 anyway |
| R7 | Oracle container drift (NGC image updates) | Low | Low | pin nvcr.io/nvidia/pytorch:26.02-py3; record image digest in the harness manifest |

---

## Operating Procedure

1. Definition of done per task: tests written and green (gradcheck/parity/
   oracle as applicable); PR merged to main via rebase-merge with CI green in
   the owning repo; GPU-dependent acceptance verified by an actual Spark run
   on the GB10 with the result recorded in the devlog; reported honestly
   (observed, not expected).
2. All DGX workloads via Spark only; memory limit on every pod; ONE GPU job at
   a time across this plan and Wolf.
3. Small focused commits; never bundle files from different top-level
   directories in one commit (pre-commit hook enforces).
4. New ops and kernels MUST ship with an OpInfo registration and pass
   gradcheck + parity (+ oracle if a CUDA kernel) before merge -- this is the
   standing gate this plan exists to create.
5. release-please tags after merge where releases are due (T5.2); Wolf bumps
   only via go get of tagged versions on main (commit pins allowed on
   branches).
6. Per-measurement results (oracle diffs, perf deltas, poison runs, the T5.1
   folds) go to the owning repo's docs/devlog.md, newest first.

---

## Progress Log

### 2026 06 10 -- Plan created

**Change Summary:**
- Created this plan from the Wolf GPU f32 investigation conclusion (eight real
  bugs fixed; training still diverges; root cause framework-level per Wolf ADR
  072 and devlog 2026 06 10). Defined E1 (gradcheck + parity-under-arena-stress
  + PyTorch-oracle harnesses, each with a historically-grounded red/green
  fixture), E2 (save-for-backward contract + arena pin/unpin + poison-on-reset
  + full Backward migration), E3 (kernel numerics audit: no global fast-math,
  fp32 fixed-order reductions, oracle-gated), E4 (deterministic mode), E5
  (Wolf GB10 f32 end-to-end validation + releases), E6 (standing gates).
  Milestones M1-M5; waves sized for up to 4 parallel agents with all GPU runs
  serialized on the single GB10.
- ADRs created: zerfoo docs/adr/091-gradcheck-pytorch-oracle-verification.md
  (the three-harness verification strategy) and ztensor
  docs/adr/006-save-for-backward-arena-pinning.md (the memory/autograd
  contract + poison mode).
- Use-case manifest extended with UC-GH-001..007 (gpu-hardening scope),
  preserving the legacy UC-EXT entries.

**Why:** porting PyTorch's architectural lessons (save-for-backward lifetime
ownership, allocator discipline, gradcheck/OpInfo verification, numerics
conventions) is the structural fix for the bug class that has consumed weeks
of DGX debugging, and it is the prerequisite for Wolf's GPU speed-parity goal.

---

## Hand-Off Notes

1. Read first: zerfoo docs/adr/091 and ztensor docs/adr/006 (the two design
   decisions); Wolf docs/adr/072 and Wolf docs/devlog.md 2026 06 09/10 (the
   eight-bug history and why this plan exists); ztensor#125, zerfoo#842,
   zerfoo#845 (the canonical bug instances the fixtures must reproduce).
2. The order matters: harnesses first (M1) -- they are what makes every later
   change verifiable offline instead of via DGX NaN hunts. Do not start kernel
   or contract work without the corresponding harness gate.
3. Key code locations: ztensor graph/node.go:22 (Backward already receives
   live inputs), compute/gpu_engine.go:324 ResetPool +
   compute/step_scope.go:75 MarkStepBoundary, internal/cuda/arena.go:195
   ArenaPool.Alloc, internal/cuda/kernels/Makefile:7 (the fast-math flag),
   zerfoo timeseries/*_backward_test.go (finite-difference precedent).
4. Hardware: single GB10 via Spark at http://192.168.86.250:8080; PyTorch
   oracle inside nvcr.io/nvidia/pytorch:26.02-py3; tanh-fixed sm_121 kernels
   currently mounted from /home/ndungu/ztensor-kernels-build on the DGX.
5. The end-goal acceptance is owned by Wolf: GB10 f32 fold, 0 NaN, fold-0 acc
   within 1pp of the CPU baseline 0.6765, twice in a row. Anything less is
   not done.

---

## Appendix

### The PyTorch patterns being ported (source references for implementers)
- save_for_backward / SavedVariable: torch autograd's explicit saved-tensor
  registry with engine-owned lifetime (torch/csrc/autograd/saved_variable.*).
- CUDACachingAllocator: stream/liveness-aware block reuse
  (c10/cuda/CUDACachingAllocator.cpp) -- the pin/unpin refcount in ADR 006 is
  the minimal Go-shaped equivalent for a single-stream arena.
- gradcheck + OpInfo: finite-difference validation of every op
  (torch/autograd/gradcheck.py, torch/testing/_internal/opinfo/).
- Numerics conventions: no global fast-math; max-subtraction softmax;
  fp32 accumulation types for reductions (aten AccumulateType); deterministic
  algorithms toggle (torch.use_deterministic_algorithms).

### Related documents
- Wolf docs/plan-pytorch-speed-parity.md -- the consumer of this plan's
  outcome (its T14.x/E9 chain unblocks at M5).
- Wolf docs/adr/072 -- why the failure is framework-level.
- ztensor docs/adr/003 (bulk-upload chunking), 004/005 (capture-aware arena)
  -- prior arena decisions this plan builds on.
