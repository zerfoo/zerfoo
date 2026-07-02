# Deep Architectural Review: Design Alignment, DRYness, and Composition

**Date:** 2026-04-27
**Reviewer:** Architectural deep-review
**Scope:** Alignment of `github.com/zerfoo/zerfoo` with the original design at
`docs/design.md@d18e20d9` (2025-08-02). Focus: package layout, composition,
DRYness, and root-directory sprawl.
**Method:** Read of design.md@d18e20d9 (1818 lines), design.md@HEAD, ADRs
082/084 + the eight package-creation ADRs (004, 031, 036, 051, 056, 057, 059,
062), `docs/dirty-architecture.md` (1087 lines, team-authored), graph
extraction (`graphify-out/graph.json` — 16,781 nodes, 21,980 edges, 1,704
communities), targeted file reads in `layers/`, `inference/`, `timeseries/`,
`tests/parity/`, plus `git log` archaeology. Three independent investigation
agents cross-validated findings.

---

## Executive Summary

The codebase is **substantially out of alignment with the original design's
package structure**, but **the core inference path still honors the design's
composition principles**. The user's instinct is correct: the root directory
has bloated from the design's 8 mandated top-level packages
(`tensor`, `compute`, `graph`, `layers`, `training`, `model`, `device`,
`distributed`) to ~47 today. Of the new packages, only ~9 were sanctioned by
ADRs; the rest were introduced as unsanctioned drift. About **80 percent of
the sprawl was added in a single 48-hour window on 2026-03-17/18**, which
matches the in-memory note about the 10-wave parallel-agent execution where
package skeletons were created faster than architectural review could vet
placement.

The top three architectural risks are: (1) **`timeseries/` is a parallel ML
framework** (17,570 LOC) that operates on raw `[]float64` slices, bypassing
`Engine[T]` and `layers/` — this directly contradicts the design's "Engine[T]
is law" rule and is the largest DRY violation in the repo; (2) **two parallel
APIs for the same activation math** (`layers/activations/` Node form vs
`layers/functional/` function form) with no delegation between them, so a fix
in one path does not propagate; (3) **at least 18 of the ~47 root-level
directories should arguably live under existing packages** (`rl/`, `meta/`,
`gp/`, `federated/`, `monitor/`, `recover/` belong under `training/`;
`gnn/`, `synth/`, `shared/` belong under `layers/`; `health/`, `shutdown/`,
`support/`, `security/`, `cloud/`, `marketplace/` belong under `serve/`;
`integration/`, `mobile/`, `testing/` belong under `tests/`).

The single most impactful recommendation: **adopt and execute ADR-082's
composition-remediation epic on a quarterly milestone, paired with a CI lint
that fails any new top-level package without a referenced ADR**. The team
already has the audit, the migration recipe (5 packages were migrated in a
single day on 2026-04-03), and a precedent for reversing scope (ADR-084 moved
`crossasset/` to `wolf`). What is missing is a structural gate to stop the
next 48-hour wave from undoing the cleanup. Note also: `testing/` shadows
the Go stdlib package and should be renamed regardless of the larger
remediation.

---

## Codebase Maturity Assessment

| Dimension                  | Level | Evidence |
|----------------------------|-------|----------|
| Architecture (alignment)   | 2/5   | 47 top-level dirs vs 8 designed; ~18 unsanctioned (`autoopt/`, `causal/`, `rl/`, `gp/`, `gnn/`, `synth/`, `meta/`, `shared/`, `modeldsl/`, `federated/`, `regime/`, `recover/`, `monitor/`, `provenance/`, `support/`, `health/`, `modelcache/`, `marketplace/`); `timeseries/` runs as a parallel framework outside Engine[T]. |
| Composition (core path)    | 4/5   | `layers/core/dense.go:15-21` correctly composes `Linear` + `Bias`; `inference/arch_common.go:104` `buildTransformerGraph` is shared by `arch_qwen.go` (50 lines) and `arch_mistral.go` (49 lines) — design intent honored on the inference critical path. |
| Composition (off-path)     | 1/5   | `timeseries/itransformer.go:190-246` keeps `linearForwardVec` (raw triple loop) AND `linearBatchEngine` AND `linearBatchCPU` all for the same op; `dirty-architecture.md` quantifies ~9,800 redundant lines, 805 unjustified `.Data()` calls, 17 reimplemented LayerNorm, 11 reimplemented Linear, 4 reimplemented GELU. |
| DRY                        | 2/5   | Two activation API surfaces (`layers/activations/gelu.go:27` Node vs `layers/functional/activations.go:15` function) for the same math, no delegation; ~1,155 lines of builder boilerplate duplicated across 25-30 `arch_*.go` files (per dirty-architecture.md). |
| Self-awareness             | 5/5   | `docs/dirty-architecture.md` (rev 3, 1087 lines) and ADR-082 already document and prioritize these issues; the team has a migration recipe and a precedent (ADR-084). |
| Code quality (core)        | 4/5   | Inference path is clean; `arch_common.go` is a textbook composition example; activation registry exists at `layers/activations/registry.go`. |
| Test coverage              | 3/5   | `tests/parity/` exists with 48 files and 19 sub-dirs of golden tests; helpers are file-private inside `layer_parity_test.go` rather than in a shared `testutil` subpackage, limiting reuse. |
| Naming hygiene             | 1/5   | `testing/` shadows the Go stdlib `testing` package; `integration/` and `integrations/` coexist with totally different purposes (smoke tests vs LangChain/Weaviate adapters); `support/` is a multi-tenant SaaS feature (customer-support handlers), not "support utilities." |

**Overall maturity: 2.6/5 (Level 2 — ad-hoc, with strong self-awareness).** The
team is performing at Level 4 inside the inference path and Level 1 at the
root; the variance reflects who built what and when, not the team's
capability. The gap between `dirty-architecture.md` and the code is the
remediation work-in-flight.

---

## Original Design vs Current Layout

### What the design mandated (design.md@d18e20d9)

The design specified exactly 8 top-level packages, listed under "Modular
Package Structure":

```
tensor/        compute/       graph/         layers/
training/      model/         device/        distributed/
```

Three constraints framed everything else:

1. **Engine[T] is the hardware abstraction.** "All computations are
   parameterized by a generic type `T` ... All methods on the `Engine`
   interface ..." The design.md@d18e20d9 wording: *"Each operation in Zerfoo
   knows how to compute its gradient ... The Engine is often used to perform
   these computations"* (line 198).
2. **Composition over inheritance.** "Complex layers that can be built by
   composing simpler layers must not re-implement the code in simpler layers"
   (design.md@d18e20d9 line 219). A Dense layer = Linear + Bias, not a
   monolith.
3. **Each operation in one place.** "If two different layer types both need a
   certain transform, we prefer to implement that as either a utility
   function or its own small Node" (line 217).

### What the codebase looks like now

47 top-level directories. Categorized by sanctioning mechanism:

| Category | Count | Examples |
|---|---|---|
| Original 8 (still here) | 4 | `layers/`, `training/`, `model/`, `distributed/` |
| Original 8 (extracted to ztensor by ADR-036) | 4 | (`tensor/`, `compute/`, `graph/`, `device/` are external) |
| Standard Go layout | 6 | `cmd/`, `internal/`, `tests/`, `docs/`, `examples/`, `scripts/` |
| Sanctioned by ADR | 9 | `inference/` (ADR-004), `generate/` (ADR-004), `serve/` (ADR-031), `tabular/` (ADR-062), `mobile/` (ADR-059), `cloud/` (ADR-056 as `serve/cloud/`, later flattened), plus `benchmarks/`, `bin/`, `deploy/`, `infra/`, `config/` (operational) |
| **Unsanctioned drift** | **~18** | **`autoopt/`, `causal/`, `features/`, `federated/`, `gnn/`, `gp/`, `health/`, `meta/`, `modelcache/`, `modeldsl/`, `monitor/`, `provenance/`, `recover/`, `regime/`, `registry/`, `rl/`, `security/`, `shared/`, `shutdown/`, `support/`, `synth/`, `testing/`, `compliance/`, `marketplace/`, `integration/`, `integrations/`** |

(Some categories overlap; counts approximate.)

### Timeline of drift

```
2025-08-02  design.md committed (d18e20d9)        — 8 packages
2025-08-25  features/ added                         — first drift
2026-03-02  health/ + ADR-004 wave (inference, generate)
2026-03-17  BIG-BANG SPRAWL: ~21 top-level packages added in 24-48h
2026-03-18  cont'd. Coincides with 5-year roadmap / 10-wave Claude Code army.
2026-03-19+ Enterprise tier: marketplace/, security/, support/, compliance/
2026-03-30  timeseries/ extracts internal "shared" math — parallel framework crystallizes
2026-04-02  ADR-082 acknowledges drift, proposes 5-phase remediation (E61-E65)
2026-04-03  5 packages migrated in a single day — recipe proven
2026-04-12  ADR-084: crossasset/ extracted to feza-ai/wolf — first reversal
```

This timeline appears in `git log` traces; the in-memory note about
"Max 3-4 agents per wave; 10 agents consistently freeze" describes the same
incident that produced the sprawl.

---

## High-Impact Architectural Findings

### A1. timeseries/ is a parallel ML framework that bypasses Engine[T]

**Impact: High.** This is the single largest violation of the original
design.

**Evidence:**

- `timeseries/itransformer.go:190-199` defines `linearForwardVec` — a raw
  triple-nested loop computing `x * W + b` directly on `[]float64`.
- `timeseries/itransformer.go:204-246` defines `linearBatchF64` which calls
  `cpuEngine64.MatMul`, but on error falls back to the hand loop at line 190.
  The same op exists in two implementations in the same file.
- `timeseries/itransformer_engine.go:16,64` defines `linearBatchEngine` AND
  `linearBatchCPU` for the same op.
- `timeseries/itransformer_backward.go:363,426,518` — `linearBackwardF64`,
  `layerNormBackwardFunctional`, `multiHeadAttentionBackwardF64` are all
  hand-rolled instead of using `Engine[T]` + autodiff.
- `timeseries/dlinear.go:52,207,234,244,261,347,424` — `normalizeWindows`,
  `movingAverage`, twin `cpuDecompose`/`engineDecompose`, twin
  `cpuLinearProject`/`engineLinearProject`. Five duplicated paths.
- `timeseries/frets_engine.go:312,329` — `fretsEngineMatMul` and
  `fretsScalarMatMul`. Same op, two paths.

`docs/dirty-architecture.md` (rev 3) confirms with hard numbers: 17,570
lines, 254 `.Data()` calls in this package alone, 17 reimplemented LayerNorm,
11 reimplemented Linear, 4 reimplemented GELU, 6 reimplemented attention.
Estimated ~9,800 redundant lines across the 3 worst packages
(`timeseries/`, `tabular/`, `modeldsl/`).

**Why the design is broken here:** the *engine path* was bolted on without
retiring the *cpu path*. Bug fixes and GPU acceleration in `Engine[T]` never
reach `timeseries/` users. The package is effectively a separate framework
that happens to live in the same repo.

**Recommendation:** apply ADR-082 phase E62 (Time Series Migration). Build
`timeseries/` models as `graph.Node[T]` graphs the way `inference/arch_*.go`
do. Delete every `*F64` / `cpu*` / `*Vec` helper after the migration. The
2026-04-03 migration of 5 packages in a day shows the recipe scales.

### A2. Activations have two parallel API surfaces

**Impact: High.** Same math, two implementations, no delegation.

**Evidence:**

- `layers/activations/gelu.go:27` defines `NewGelu` — the canonical Node
  implementation.
- `layers/activations/fast_gelu.go` — separate `FastGelu` Node.
- `layers/functional/activations.go:15` — functional `GELU(...)` that
  reimplements the math instead of delegating to `NewGelu`.
- `layers/functional/gelu_backward.go:20` — separate functional backward.
- An activation registry exists at `layers/activations/registry.go` but
  `layers/functional/` bypasses it entirely.
- At least 8 callers wire activations differently: `layers/core/ffn.go`,
  `layers/vision/clip_encoder.go`, `layers/audio/whisper_encoder.go`,
  several `tabular/*.go` files.

**Recommendation:** make `layers/functional/activations.go` thin wrappers
around the activation Node registry — one source of arithmetic per
activation. This is an afternoon's work and removes a recurring source of
correctness drift.

### A3. Root sprawl: ~18 unsanctioned top-level packages

**Impact: High.** Each new package widens the cognitive surface, multiplies
import paths, and invites the next "let's also create a top-level dir for
this" shortcut.

The following dirs should arguably live elsewhere (proposed home in
parentheses, severity from the investigation agent):

| Move under `training/` | `rl/`, `meta/`, `gp/`, `federated/`, `monitor/`, `recover/`, `provenance/` |
| Move under `layers/` | `gnn/`, `synth/`, `shared/` |
| Move under `serve/` | `health/`, `shutdown/`, `support/`, `security/`, `cloud/`, `marketplace/` |
| Move under `inference/timeseries/` | `causal/`, `features/`, `regime/` |
| Move under `model/` | `modelcache/`, `modeldsl/`, `registry/` |
| Move under `tests/` | `integration/`, `mobile/`, `testing/` |
| Move under `internal/` or `ztensor/` | `autoopt/` (kernel codegen) |
| Rename | `integrations/` → `sdk/integrations/` (it's LangChain + Weaviate adapters, not test integration) |

Cross-import signals already make consolidation mechanical. Examples:
`recover/retrain.go` already imports `monitor`; `cmd/cli/{serve,worker,signal}.go`
already import `health`/`shutdown`/`security` together. Moving the latter
three into `serve/` is a `git mv` plus an import-path rewrite.

### A4. Naming hazards

**Impact: Medium.**

- `testing/` shadows the Go stdlib `testing` package. This is harmful
  regardless of any larger remediation. Rename to `testutil/` or move under
  `tests/testutil/`.
- `integration/` (2 files: production smoke tests) and `integrations/`
  (4 files: LangChain + Weaviate adapters) coexist with confusingly similar
  names and unrelated purposes. Consensus rename: `integration/` →
  `tests/integration/`, `integrations/` → `sdk/integrations/`.
- `support/` (8 files) implements customer-support webhook handlers — not
  "support utilities." Rename or move under `serve/support/`.
- `shared/` is the canonical "anti-pattern bucket" name; it currently holds 3
  files (cross-model latent space). Move under `layers/` with a more specific
  name, e.g., `layers/shared_latent/`.

### A5. Test-helper concentration without a shared package

**Impact: Medium.** Graphify identified `makeTensor`, `setup`, `loadGolden`,
`getFloat32s`, `assertClose` as the most-connected nodes in the entire
codebase (each 76-88 edges). They live in
`tests/parity/layer_parity_test.go` — file-private, single-package. The
helpers can't be reused by GPU parity files or other parity-style tests in
adjacent suites without copy-paste.

**Recommendation:** move them to `tests/parity/testutil/` (or a non-`_test.go`
package importable by other test packages) so the parity helpers are the
single source of truth for golden-file loading and tolerance assertions
across all parity-style suites.

---

## What the codebase does well (calibrating trust in the findings)

- **`inference/arch_common.go:104`** — `buildTransformerGraph` is a textbook
  composition example. `arch_qwen.go` (50 lines) and `arch_mistral.go` (49
  lines) are thin config adapters. Adding a new vanilla decoder-only arch
  costs ~50 lines.
- **`layers/core/dense.go:15-21`** — `Dense` is correctly composed of
  `Linear` + `Bias` + optional activation, exactly as design.md@d18e20d9
  prescribed at line 219.
- **The team has already done this audit.** `docs/dirty-architecture.md`
  (rev 3, 2026-04-03, 1087 lines) is one of the most thorough and honest
  internal architectural-debt documents I have read in any codebase. It
  enumerates violations with file:line refs and categorizes by severity.
  ADR-082 prescribes a 5-phase remediation (E61-E65). ADR-084 set the
  precedent for reversing scope by extracting `crossasset/` to `wolf`.
- **5 packages were migrated in a single day on 2026-04-03**
  (`crossasset/`, `rl/`, `synth/`, `meta/`, `shared/` per the rev-3 doc) —
  the migration recipe is proven and fast.

The variance between the inference critical path (Level 4) and the off-path
domain packages (Level 1) is the variance between "code reviewed by a
careful human" and "skeleton dropped by a parallel-agent wave." The fix is
not to lower the ceiling; it is to raise the floor.

---

## Prioritized Remediation Roadmap

### Fix this week (low cost, high signal)

1. **Rename `testing/`.** Single most dangerous name in the tree. Probably
   ~30 minutes of rewrites.
2. **Resolve `integration/` vs `integrations/`.** Move former to
   `tests/integration/`, latter to `sdk/integrations/`.
3. **Add a CI lint that fails any new top-level Go package not listed in an
   allowlist.** The allowlist is the design's 8 + ADR-sanctioned additions.
   New top-level requires a referenced ADR. This stops the next 48-hour
   wave from re-creating the cleanup target. (See the in-memory note about
   the team's existing wave-runner setup — this is the single highest-ROI
   guardrail.)

### Fix this sprint (architectural cleanup)

4. **Collapse `layers/functional/activations.go` onto the activation Node
   registry** (A2). Half a day. Eliminates a recurring correctness-drift
   vector.
5. **Move `health/`, `shutdown/`, `support/`, `security/` under `serve/`**
   (A3). Cross-imports already exist; this is `git mv` plus rewrites. Half a
   day.
6. **Move `rl/`, `meta/`, `gp/`, `monitor/`, `recover/`, `provenance/`
   under `training/`** (A3). One day. Several already cross-import.
7. **Move `gnn/`, `synth/`, `shared/` under `layers/`** (A3). Half a day.
8. **Move parity test helpers to `tests/parity/testutil/`** (A5). Two
   hours.

### Fix this quarter (the big one)

9. **Execute ADR-082's E62 (timeseries) phase.** Migrate `timeseries/` to
   `Engine[T]` + `graph.Node[T]`. This is the largest single source of
   redundant code in the repo (~9,800 lines across `timeseries/` +
   `tabular/` + `modeldsl/`). The 2026-04-03 single-day migration of 5
   packages shows it can be done; this one is bigger but bounded.
10. **Decide whether `cloud/`, `marketplace/`, `compliance/` belong in this
    repo or in a sibling `zerfoo-enterprise/` repo** per ADR-057's
    open-core-licensing direction. ADR-084 set the precedent for splitting
    out scope; the same reasoning applies here.

### Track as tech debt

11. Refresh `docs/design.md` to either mandate the post-extraction layout
    (4 original + sanctioned additions) or explicitly mark which packages
    are aspirational. Right now it documents the drift rather than gating
    it.
12. Continue ADR-082 phases E63-E65 (`tabular/`, `modeldsl/`, generate/serve
    structural cleanup).

---

## Statistics

- Source files in codebase (Go, non-vendor): ~1,637 (graphify code count)
- Top-level Go directories: 47 (vs 8 in original design)
- Original-design top-level packages still present in this repo: 4
  (`layers/`, `training/`, `model/`, `distributed/`)
- Original-design packages legitimately extracted: 4 (`tensor/`, `compute/`,
  `graph/`, `device/` → ztensor per ADR-036)
- ADR-sanctioned additions: ~9
- Unsanctioned top-level dirs: ~18
- Findings: High [3], Medium [2], Info-positive [3]
- Test files in `tests/parity/`: 48 across 19 sub-dirs
- Graph stats from `graphify-out/graph.json`: 16,781 nodes, 21,980 edges,
  1,704 communities; cohesion of top communities 0.01-0.11 (low — each
  community is internally weakly connected)
- Lines of acknowledged duplicated code (per `dirty-architecture.md` rev 3):
  ~9,800 across `timeseries/` + `tabular/` + `modeldsl/`
- Unjustified `.Data()` bypass calls (per `dirty-architecture.md` rev 3):
  ~805 of ~2,599 total
- ADRs reviewed: 004, 031, 036, 051, 056, 057, 059, 062, 082, 084
- Days from design.md to first drift: ~23
- Days from design.md to big-bang sprawl: ~227
- Days from big-bang sprawl to acknowledgement (ADR-082): ~16

---

## Closing Note

The user's instinct prompted the right question. The repo is healthier than
the directory listing suggests — the inference critical path is exemplary —
but the root has accumulated ~18 packages that the original design would
have placed under existing parents, and one of them (`timeseries/`) has
metastasized into a parallel framework. The team already knows this
(`dirty-architecture.md` is unusually candid) and has a recipe. The missing
piece is a CI guardrail that prevents the next parallel-agent wave from
undoing the next round of cleanup. Add that, finish ADR-082's E62, rename
`testing/`, and the framework will be substantially back in line with the
2025-08 vision.
