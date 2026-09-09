# Zerfoo model-creation refinement plan

Date: 2026-09-09. Status: R01–R04 and R07 implemented; R05/R09 in progress (Codex).
Phase R release remains unverified; recovery, HTTP serving and independent GPU/oracle gates remain open.
Scope: Phase R of the [model-creation roadmap](model-creation-roadmap.md).
Origin: the user's request to prepare Zerfoo for conversational model creation.

## Outcome and boundaries

A real numeric CSV dataset can pass through inspection, reproducible splitting,
training, held-out evaluation, export/reload, and HTTP prediction using Zerfoo.
The same recipe works on CPU and the reference GPU worker, with bounded
execution and recoverable training. Correctness verification is mandatory:
establish a baseline in R01, verify each correction during R02–R09, and verify
the complete lifecycle against independent references in R10 before release.
Each claim has recorded execution evidence.

Support binary and multiclass classification with user-defined labels. Start
with a majority-class baseline, a trainable linear classifier, and a small MLP.
Support finite numeric inputs, an explicit target column, and optional group or
time columns for splitting. Reject unsupported feature types and ambiguous
schemas with actionable errors. Missing numeric values use a documented policy
(training-only imputation or explicit rejection); no implicit row dropping.

This phase does not add a chat UI, MCP, a research corpus, novel operators,
LLM fine-tuning, distributed training, new accelerator backends, or a cloud
scheduler. Those capabilities are not needed to establish this lifecycle.

The [current Trust tracker](plan.md) retains its unfinished work. R01 reconciles
dependencies against current source and issue evidence; historical unchecked
boxes are not proof that defects remain. A defect affecting this recipe blocks
its release; unrelated model-family work stays in its existing lane. Existing
inference support claims remain governed by [verified-models.md](verified-models.md).

## Evidence found during planning

Source inspection only; none of the following is a new runtime test verdict.

| Observation | Source | Refinement |
|---|---|---|
| General CLI training creates synthetic inputs and gradients | [cmd/cli/train.go](../cmd/cli/train.go) | R04 replaces or rejects the unsupported production path |
| The real tabular AutoML worker evaluates all input rows, including training data | [cmd/cli/automl.go](../cmd/cli/automl.go) | R03/R05 establish isolated evaluation and persisted winners |
| Model, training, and persistence assume three trading classes | [tabular/model.go](../tabular/model.go), [train.go](../tabular/train.go), [save.go](../tabular/save.go) | R02 generalizes the supported classifier contract |
| Training uses an internal background context and global random generation | [tabular/train.go](../tabular/train.go) | R04/R06 provide cancellation and restorable random state |
| Tabular persistence is ZTAB despite the documented GGUF-only direction | [tabular/save.go](../tabular/save.go), [CLAUDE.md](../CLAUDE.md) | R01/R07 prove a GGUF export/load path and retain legacy compatibility |
| Existing classification HTTP input is text sentiment | [serve/classify.go](../serve/classify.go) | R08 adds a tabular prediction contract |
| GPU training test can skip; default CI is CPU and uses short mode | [tabular/train_gpu_test.go](../tabular/train_gpu_test.go), [ci.yml](../.github/workflows/ci.yml) | R09 requires a separate mandatory GPU evidence run |

## Contracts to settle before widening implementation

Use existing packages where their contracts fit. New top-level packages need
the repository's ADR process. R01 records the detailed design before APIs land.

| Object | Required content |
|---|---|
| Dataset manifest | Content hash, schema, target/feature order, label mapping, stable row IDs, split IDs and assignments, preprocessing fit provenance |
| Recipe | Version, supported task/device, architecture bounds, losses, optimizer, tunable parameters, train/export/serve capability evidence |
| Training request | Dataset/split references, recipe/config, seed, device, resource limits, output location, cancellation context |
| Training checkpoint | Weights, optimizer/scheduler state, random state, epoch/batch cursor, preprocessing identity, dataset/config/runtime hashes |
| Evaluation report | Artifact and split hashes, metric definition and direction, sample counts, baseline, confusion matrix, class slices, uncertainty method, threshold decision |
| Deployment bundle | GGUF weights, versioned JSON manifest, preprocessing, labels, recipe/runtime compatibility, evaluation reference, hashes for every file |

**Artifact choice:** plan for GGUF deployment weights and a manifest. This
extends the existing policy to the selected tabular recipe; it does not assume
the current GGUF loader can already execute it. Training checkpoints are
versioned resumable state, not deployment artifacts. R01 must prove tiny-model
export/reload before committing to the implementation sequence. If that fails,
record the exact missing capability and revise the design and estimate; do not
quietly introduce another default weight format.

Preserve existing three-class callers through explicit legacy wrappers/defaults
and a tested ZTAB import/migration path. New generic APIs return class IDs and
probabilities with a persisted user label map. Avoid a breaking signature change
unless a versioned migration is deliberately scheduled. Audit all tabular
callers, including ensembles and LoRA; this phase only certifies its recipes.

## Work breakdown

R01–R04/R07 completed by Codex; R05/R09 assigned to Codex. Remaining tasks
are unassigned until pickup. IDs R01–R12 are local to this document and do not reuse existing T/E IDs.
Each task includes its regression evidence and documentation, rather than a
separate cleanup phase.

### R01 — Audit the lifecycle and freeze the reference contract

- [x] R01. Owner: Codex. Evidence: [baseline audit](audits/model-creation-correctness.md),
  [ADR-096](adr/096-model-creation-reference-contract.md), and pinned fixture/contracts.
  CPU baseline: 1,325 passing named test events (parents + subtests), no failures/skips;
  two numerical/format probes passed and three real-worker defect witnesses reproduced.
  GPU/autodiff/release remain unverified; package lint has recorded pre-existing failures.
  Dependencies: none. Deliver a bounded audit, proposed/accepted ADR
  as appropriate, fixture manifest, and revised estimates before larger work.
  Produce a baseline correctness report tracing the actual
  CLI-to-training-to-artifact-to-serving call paths. Corroborate
  the evaluation leakage through the production worker. Run existing relevant
  suites and a tiny GGUF classifier export/load spike. Inventory cancellation,
  seed, checkpoint, optimizer, and engine boundaries. Inspect current dependencies
  behind E147/E148/E149/E152; reuse existing fixes and tasks where applicable.
  Select one redistributable real-data fixture with attribution and hash plus
  small deterministic contract fixtures. Freeze metric definitions, splits,
  baseline comparisons, prediction tolerances, and CPU/GPU convergence bounds
  before implementation results are used to judge success.
  Inventory the operators and shared runtime dependencies used by the supported
  recipes and apply the correctness protocol below. Record confirmed failures,
  existing evidence and unverified contracts before modifying their behavior.
  Acceptance: every lifecycle boundary has a caller and evidence or a named gap;
  all mandatory acceptance thresholds are numeric and nonempty. No claim that
  an existing test passes based on its presence alone.

### R02 — Generalize the supported classifier API

- [x] R02. Owner: Codex. Generic float32/float64 classifier, 2/3/5 classes,
  labels, linear/ReLU MLP, finite/shape validation and legacy compatibility tested. Dependencies: R01. Implement explicit class count, label mapping,
  shape/range validation, and generic prediction output for the linear and MLP
  recipes. Keep tensor math under `compute.Engine[T]`. Preserve the legacy
  three-direction interface intentionally and audit dependent call sites.
  Acceptance: binary, three-class, and five-class cases produce correctly
  shaped finite probabilities; user labels survive prediction; invalid shapes
  and out-of-range targets fail; legacy callers retain their documented behavior.

### R03 — Make datasets and evaluation splits reproducible

- [x] R03. Owner: Codex. CSV inspection, isolated splits, training-only preprocessing,
  manifest persistence/replay and changed-data rejection tested. Fixture v2 isolates duplicates
  before learning without changing frozen thresholds. Dependencies: R01. Implement CSV schema validation, stable row
  identity, stratified/group/temporal splitting as explicitly selected, and
  training-only preprocessing. Persist assignments, feature order and label
  mapping. Detect duplicate/group crossings, target leakage, insufficient class
  coverage, and invalid/nonfinite data. Reject cases where a valid split cannot
  be formed instead of borrowing test rows.
  Acceptance: no row/group overlap, correct temporal order, identical split on
  replay, unseen/malformed labels rejected, and a holdout-only outlier cannot
  change training preprocessing. Dataset changes invalidate existing run inputs.

### R04 — Wire real, cancellable training through the CLI

- [x] R04. Owner: Codex. Real CLI and library training, structured events, cancellation,
  label-sensitive learning and actual GGUF artifact output implemented. Six CPU recipe/seed
  runs met frozen quality thresholds; CLI MLP artifact reloaded at 28/30. Dependencies: R02, R03. Route the supported dataset recipe through
  library and CLI entry points with context propagation and local seeded random
  state. Record architecture/config, actual loss/steps and output artifacts.
  Move synthetic demonstrations to explicit examples/test fixtures. Unsupported
  general/FSDP model paths return clear errors; do not promise broad FSDP training
  as part of this fix. Machine output uses a versioned result/error schema and
  progress events; diagnostics do not corrupt it.
  Acceptance: the public command changes weights on the supplied dataset,
  changing labels changes learning, and the fixed fixture meets R01's convergence
  threshold. Cancellation is distinguishable from success and terminates work.

### R05 — Correct AutoML scoring and retain its winning model

- [x] R05. Owner: Codex. Isolated scoring, majority/linear/MLP comparison,
  per-trial artifacts/failures/steps/timing/allocation records and frozen final evaluation
  implemented. Baseline win, metric direction, artifact reload, cancellation and
  explicit prior-exposure lineage tested. Lineage is local and caller-declared;
  process allocation is not peak memory or a hard cap. Dependencies: R04. Restrict search to parameters actually consumed
  by each recipe; reject unsupported metrics and honor minimize/maximize.
  Compare majority, linear and MLP candidates on the same validation rows.
  Persist every trial and its artifact, including failures and resource usage.
  Prediction errors fail evaluation instead of being silently skipped.
  Keep the final test split inaccessible to tuning. Evaluate a frozen candidate
  once per evaluation protocol version; retuning creates a new experiment and
  explicitly records test exposure.
  Acceptance: a memorizing candidate cannot win by training-set accuracy;
  maximize/minimize select the correct known candidate; the reported winner
  reloads to the exact scored artifact; a baseline win is an honest result.

### R06 — Add training recovery and hard execution limits

- [ ] R06. Dependencies: R04. Save complete checkpoints atomically and validate
  compatibility on resume. Handle interrupted writes and keep the last complete
  checkpoint. Add wall-clock/step limits, graceful cancellation and a bounded
  forced-stop fallback at the execution boundary. Respect the reference worker's
  memory and concurrency controls; do not equate memory estimates with a cap.
  Acceptance: stop/restart matches uninterrupted CPU execution within the frozen
  tolerance, including optimizer and random state; changed data/config is
  rejected; cancellation and limit exhaustion preserve truthful terminal states.

### R07 — Establish the deployment artifact contract

- [x] R07. Owner: Codex. Atomic GGUF bundle publication, validated bounded load,
  fresh-process prediction equality, seven corruption cases and explicit ZTAB migration tested. Dependencies: R02, R03, R04 and R01's format proof. Implement GGUF
  classifier export/load, bundle schema validation and deterministic file hashes.
  Atomically publish complete bundles. Bound dimensions/allocations and reject
  truncation, unsupported versions, inconsistent shapes, and hash mismatches.
  Include legacy ZTAB reading/migration coverage without silently reinterpreting
  old three-class weights.
  Acceptance: fresh-process reload reproduces logits/probabilities within R01's
  tolerance and labels exactly; preprocessing travels with the model; corrupt
  and incompatible artifacts fail before becoming available to inference.

### R08 — Serve the same model and preprocessing

- [ ] R08. Dependencies: R05, R07. Add a generic tabular prediction endpoint
  and CLI route using the bundle loader; document named features, batch limits,
  output probabilities/labels, model version, and errors. Wire existing auth and
  request limits through the actual serving entry point. Health includes loaded
  artifact identity and a prediction check, not just an open TCP port.
  Acceptance: offline and HTTP predictions agree for raw held-out records in a
  fresh server process; missing/reordered/extra fields follow the schema policy;
  invalid inputs fail clearly; server restart preserves the selected artifact.

### R09 — Verify the reference GPU path and CPU deployment

- [ ] R09. Owner: Codex (GPU training gate prepared; complete recovery/serving gates pending). Dependencies: R05, R06, R07, R08. Train the identical recipe/data
  contract on the reference GB10 through Spark, export, and serve on a separate
  CPU target. Prove device dispatch and record peak memory and runtime. Start
  with eager fp32; captured training and mixed precision are separate claims.
  Acceptance: finite gradients/weights, R01's CPU/GPU numerical and convergence
  thresholds met, resume exercised, and CPU serving matches the exported GPU
  artifact. Missing GPU or skipped cases means UNVERIFIED, never PASS. If the
  GPU path fails, CPU work may proceed but the GPU release claim remains blocked.

### R10 — Verify correctness end to end and gate release

- [ ] R10. Dependencies: R05, R06, R08, R09. Verify correctness against the
  frozen contracts and independent references below, using the real CLI,
  training engine, checkpoint loader, artifact loader and HTTP server.
  Revisit every R01 finding and correction; verify actual behavior, not just
  the presence of regression tests. Produce the final correctness report and
  close or explicitly disposition every finding. Add explicit CPU lifecycle and
  reference-device jobs alongside existing CI. Require execution of all mandatory
  cases below; do not rely on `go test -short ./...` to discover integration work.
  Record current commit/dependency versions, suite/case names, executed/pass/fail/
  skip counts, fixture/artifact hashes, device identity, and raw log/report paths.
  Acceptance: every mandatory numerical and lifecycle contract is verified;
  no unresolved correctness defect remains on the released path. Missing fixture,
  wrong model identity and zero-test selection each
  fail the gate; the gate passes on the implemented path at the same revision.
  Publish a recipe/device training-and-deployment matrix distinct from inference,
  linked to the correctness report. Failures reopen their owning tasks; rerun
  affected checks and the full lifecycle after corrections before release.

### R11 — Package a reproducible refinement release

- [ ] R11. Dependencies: R10. Deliver one documented command sequence from CSV
  to evaluation to serving with pinned prerequisites, an example dataset, actual
  resource measurements and a recovery walkthrough. Build the release artifact
  and rerun the lifecycle in a clean environment using that artifact. Review
  existing inference/legacy regression results appropriate to touched packages.
  Acceptance: another session can reproduce the report and endpoint from the
  instructions; supported capabilities match recorded evidence; no hidden local
  workspace or synthetic production dependency.

### R12 — Plan the agent-product phase from measured results

- [ ] R12. Dependencies: R11. Reconcile the Phase A tasks in the roadmap with
  delivered APIs, measurements, residual limitations and current Trust work.
  Produce the single active Phase A tracker, assign its first task, and revise
  estimates. Carry unmet gates explicitly; do not mark the original Trust phase
  closed solely because this lifecycle works. This is the final refinement task.

## Correctness verification protocol

Scope: the supported creation/training/evaluation/deployment path, including
the shared tensor, autograd, optimizer and runtime contracts it exercises.
Shared-component changes also require representative existing inference and
legacy-training regression coverage. The report names excluded architectures
and backends; it cannot claim repository-wide correctness from one recipe.

Verify before remediation (R01), with each correction (R02–R09), and after
integration (R10), then repeat the lifecycle with the release artifact (R11).
A falling loss, changed weights, successful HTTP status, or CPU/GPU agreement
alone is insufficient proof of correct learning.

| Contract | Independent verification required |
|---|---|
| Forward computation | Compare linear/MLP logits, losses and probabilities with a pinned independent reference using identical explicit weights/inputs; cover batch size 1, non-square shapes, multiclass and extreme finite logits |
| Gradients and updates | Finite-difference checks on small smooth fixtures plus an independent autodiff oracle; compare per-parameter gradients and optimizer state/weights after one and multiple updates, including loss reduction and weight decay |
| CPU/GPU runtime | Match both paths to the independent oracle as well as each other; prove GPU dispatch and verify buffer reuse, gradient accumulation/reset and eager fp32 training lifetimes |
| Data and metrics | Independently compute known confusion matrices and metric scores; prove split/preprocessing isolation with leakage traps; reject invalid feature/label mappings and unsupported metrics |
| Recovery and persistence | Compare interrupted/resumed and uninterrupted runs at equal optimizer-step counts, including weights, optimizer/scheduler/RNG state and next-batch identity; verify fresh-process artifact loading |
| Serving and integration | Drive released CLI and HTTP entry points with identical raw records; compare independent expected outputs with offline/served predictions; exercise invalid input, cancellation, failed writes and incompatible artifacts |

Use the existing [gradient/oracle policy](adr/091-gradcheck-pytorch-oracle-verification.md)
and [kernel tolerance guidance](kernel-tolerances.md). A pinned PyTorch/reference
implementation is validation tooling, not a production dependency. Oracle
results must not come from calling Zerfoo itself. Use explicit weights/data
and disable stochastic layers for numerical comparisons; separately verify
training-time stochastic behavior with controlled random state. Finite
differences must avoid nondifferentiable points.

R01 freezes absolute/relative tolerances per dtype/operator and the comparison
rule, normally `abs(actual - expected) <= atol + rtol * abs(expected)`, plus
convergence bounds and sample counts. Require finite values and exact shape,
label and identity checks. Justify tolerances from numerical behavior and
reference precision; do not choose them after observing a failed result.

Every confirmed bug requires a production-path reproduction and a regression
that fails before its fix and passes afterward. Deliberately breaking a
contract must make its new release gate fail. Check that failures represent
product defects rather than incorrect fixtures or impossible test expectations.

During execution, R01 creates and R10 completes
`docs/audits/model-creation-correctness.md`. It records the contract-to-code-to-suite
map, baseline/final revisions, oracle versions, fixture hashes, expected/observed
results, maximum errors/tolerances, executed/pass/fail/skip counts, raw evidence,
and a finding ledger with owner, correction and re-verification result. Until
executed, checks remain PLANNED or UNVERIFIED. Empty, skipped or stale results
cannot become PASS. An excluded capability must lose its release claim and
fail explicitly; exclusion cannot conceal a required-path correctness defect.

R10 is a separate verification activity from implementing fixes: rebuild from
a clean checkout of the candidate revision, inspect the production wiring,
and regenerate evidence against the frozen reference. The same engineer may
perform it; independence comes from the oracle and procedure. No external
review or executed verification may be claimed unless it occurred.

## Mandatory acceptance cases

Names below are proposed acceptance case identifiers, not claims of existing
tests. R01 maps them to real suites and freezes tolerances. R10 must require each
case, with a nonzero execution count and zero skips for mandatory cases.

| Case | Required observation |
|---|---|
| REF-NUMERICS | Forward values, losses, gradients and optimizer updates agree with independent references within frozen tolerances |
| REF-CORRECTNESS | Baseline findings have production reproductions and verified corrections; final clean-build report covers every mandatory contract |
| REF-LABELS | Binary and five-class models retain arbitrary labels; legacy three-class behavior covered |
| REF-DATA | Row/group/time isolation and training-only preprocessing demonstrated |
| REF-LEARN | Real CLI reads pinned data, changes weights and meets frozen convergence threshold |
| REF-SELECT | Leakage trap rejected; correct metric direction and winning artifact identity |
| REF-RESUME | Process interruption restores weights, optimizer, RNG and batch position correctly |
| REF-LIMIT | Cancellation/time/step/resource exhaustion stop execution with truthful status |
| REF-BUNDLE | Fresh-process round trip, legacy migration, malformed artifact rejection |
| REF-SERVE | Raw-input offline/HTTP prediction agreement from released bundle |
| REF-GPU | Real GB10 learning/resume and GPU-trained artifact served on CPU |
| REF-GATE | Missing data, wrong identity, and zero executed tests cannot produce a release PASS |

Learning success on the fixed fixture must exceed the frozen baseline threshold;
arbitrary user data is allowed to yield no improvement. Never lower a threshold
after seeing a failure merely to finish the phase. Synthetic fixtures are useful
for contract failures; the real-data lifecycle is independently required.

## Sequence, effort and operating constraints

Critical path: R01 → R02/R03 → R04 → R05/R06/R07 → R08 → R09 → R10 → R11 → R12.
R05/R06/R07 can be sequenced independently once contracts stabilize; this is a
dependency observation, not a request to spawn parallel agents.

Initial effort allowance: **25–40 engineering days**, low confidence until R01.
This includes baseline correctness audit, implementation, per-fix regression
proof, final independent correctness verification and packaging;
it excludes waiting for hardware and unrelated Trust closeout. Budget R01 at
2–3 days, then replace this allowance with task-level estimates from its findings.
Unexpected format/backend work must revise the scope or estimate explicitly.

Before a multi-package heavy build on the mini: check `uptime`; hold above
1-minute load 10; acquire the shared R-build-lease with the canonical claim
script and the prescribed shared remote, verify WON, and release using its SHA
immediately afterward. Lease TTL is 30 minutes. At most two heavy lanes per
project and one repository-wide race suite at a time. GPU validation uses Spark,
one GPU pod at a time, with no interactive-SSH benchmark workloads. Use the
resource claim when replacing shared tracking files.
