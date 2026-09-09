# Describe the model you need

Date: 2026-09-08. Status: implementation roadmap prepared at the user's request;
all delivery tasks are planned, not completed. Architecture choices below are
planning defaults to formalize during implementation, not accepted ADRs.

**Describe the model you need. Create, train, and deploy it on hardware you control.**

The [product proposal](agent-model-creation-proposal.md) explains the rationale.
The [refinement plan](plan-model-creation-refinement.md) is the first executable
work package. This roadmap defines the next product phase and expansion gates.
Only one phase becomes the active execution tracker at a time.

Correctness verification is mandatory before advancing from refinement: R01
establishes the baseline, every fix carries regression proof, and R10 verifies
the integrated result against independent numerical references and real lifecycle
behavior. R11 repeats the lifecycle with the release artifact. Phase A depends
on that correctness report; passing unit tests or observing decreasing loss alone
does not satisfy the gate.

## Product scope and sequence

| Phase | User outcome | Exit gate |
|---|---|---|
| R — Refine Zerfoo | A documented real-data training/evaluation/deployment lifecycle | Baseline and final correctness verification, independent numerical references, CPU/GPU evidence, artifact compatibility, recovery, matching served predictions |
| A — Agent model creation | Complete that lifecycle by talking to a coding agent | Research-cited plan, bounded real experiments on user hardware, durable supervision, deploy/rollback, independent user completion |
| B — Broaden creation | Forecasting and verified adaptation of an existing small model | Each recipe independently passes data/train/evaluate/export/serve gates |
| C — Experimental architectures | Research proposes new trainable compositions | Supported-operator and gradient checks, baseline comparisons, bounded experiments, complete artifact/runtime compatibility |

Phase A includes actual research retrieval and model construction: a planner
chooses a linear classifier or bounded MLP configuration and trains new weights.
It does not market architecture invention or foundation-model pretraining.
For the first users, support numeric tabular classification, one training worker,
one separate CPU deployment worker, and CPU/GB10 capabilities proven in Phase R.

## Relationship to the existing strategy

The user's direction expands the audience beyond Go developers and makes
conversational creation the proposed next product experience. Preserve the
trust-before-claims requirement of [ADR-093](adr/093-h2-2026-trust-then-traction-strategy.md).
Do not silently replace the unfinished tasks in [plan.md](plan.md) or mark the
older launch plan delivered. R01 maps applicable dependencies; R12 and A01 record
the revised priority/packaging decisions and reconcile the active tracker.
Public inference claims still require their original evidence. This roadmap
authorizes planning, not publication, hardware provisioning or a release.

Keep generic tensor, training, persistence and serving mechanisms in Zerfoo.
Place project orchestration, provider adapters, MCP and the research catalog in
a separately packaged application depending on released Zerfoo APIs. The exact
module/repository name is chosen in A01; no branding or repository creation is
needed to complete this planning task. This respects the embeddable core and
the existing [packaging boundary](adr/090-zerfoo-oss-scope-cloud-marketplace-compliance.md).

## The first conversation

1. The user supplies a labeled dataset and a task such as detecting defective
   parts from inspection measurements. The agent identifies the target and asks
   only for material missing inputs: error costs, grouping/time relationships,
   acceptance targets, and hardware/resource limits.
2. Zerfoo inspects data and devices and searches applicable research. The agent
   proposes a baseline and bounded candidates, explains why the recipes apply,
   and shows estimated resource use and uncertainty.
3. The user establishes the run scope, including deployment criteria. The
   controller executes within it without repeated permission for every trial.
   New spending, data access or deployment scope requires an explicit change.
4. The agent reports measured validation results. A frozen candidate receives
   final evaluation and target-device measurement. If no candidate meets the
   criteria, it reports that result and suggests a concrete next experiment or
   data improvement; it does not quietly relax the criteria.
5. Zerfoo deploys the exact qualified bundle within the authorized scope,
   verifies a prediction, and returns an endpoint and example request. The user
   can inspect lineage, cancel training, roll back, or resume from another agent.

The user's data remains on configured storage/workers. External reasoning sees
schema and aggregate diagnostics by default; sending samples follows the
project's explicit data-sharing scope. Hardware and API credentials stay outside
LLM context, accessed through scoped references.

## Phase A architecture contracts

Expose one application service through an MCP adapter and a CLI with JSON output.
MCP is a transport for discoverable tools, not the durable job engine; see the
[official tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools).
Validate actual host/protocol compatibility during A09.

| Object | Persistent fields and behavior |
|---|---|
| Project | Objective, dataset reference, metric/thresholds, data-sharing scope, resource limits, train/deploy targets, authorization scope/version |
| Plan | Immutable version/hash, cited recipe/evidence IDs, candidate bounds, validation protocol, cost/resource estimate, eligibility reasons |
| Run | Project/plan hash, idempotency key, worker attempt and fencing token, durable state/events, checkpoint/artifact references, usage and errors |
| Evidence | Source/version/section, extracted claim, assumptions, implemented recipe version, reproduction record, confidence/status |
| Deployment | Artifact hash, schema/runtime version, target, active/previous version, health and smoke results, authorization/evaluation reference |

Run states: queued → running → succeeded, failed, cancelled, or budget_exhausted.
A lost worker enters recovering before a fenced retry or terminal failure.
Training success does not imply evaluation passed or deployment occurred; those
are separate records. Persist state before acknowledging mutating requests.
Idempotency keys bind request content; reusing a key with different content fails.
Worker fencing prevents a stale attempt from publishing a winner after retry.

Default storage: a single-controller bbolt database (already a dependency in
Zerfoo) plus content-addressed artifact files. Keep an append-only event history
for recovery and inspection. No highly available control plane or multi-tenant
SaaS in Phase A. No ordinary library call requires the controller to be running.

The existing coding agent is the default planner. Also provide an optional
specialist API planner for hosts that want to delegate model design: one real
provider adapter, a structured Plan result, time/token limits and explicit error
handling. Exactly one planner owns a plan revision. The controller validates
all proposals and never lets provider output bypass recipe or resource limits.
Planning may stop when the API is unavailable; running jobs remain manageable.

Tools: capabilities, project_create/get, dataset_inspect, research_search,
experiment_plan, run_start/status/cancel, model_evaluate, deployment_apply/status/
rollback. Each operation has a JSON schema, typed errors and authorization
checks in the common service. Long tasks return IDs promptly; no dependency on
holding an MCP request open for the duration of training.

## Phase A work breakdown

All tasks are unassigned and planned. Dependencies on R refer to the separate
refinement plan. A01–A10 are document-local IDs, not existing issue assignments.

| Task | Dependencies | Deliverable and acceptance |
|---|---|---|
| A01 — Product contracts and package | R12 | Record audience/strategy and packaging ADRs, create the application package when execution begins, version schemas, and pin the verified Zerfoo release. Installation works without modifying core source or requiring a particular agent host. |
| A02 — Durable project/run service | A01 | Persistent plans, runs, events, idempotency, cancellation, usage ledger and recovery. Process restart retains state; identical retries create one run; conflicting retries fail; stale worker attempts cannot publish artifacts. |
| A03 — User hardware workers | A02 | Register scoped worker identities, probe capabilities, dispatch to the existing Spark reference route and a CPU worker, transfer hash-checked artifacts, and enforce limits locally. A real remote run survives chat closure; worker loss/retry is fenced; cancellation and exhausted budgets stop work. |
| A04 — Research catalog and retrieval | A01 | Curate an initial 10–20 relevant sources into structured evidence cards; combine metadata filtering, keyword retrieval and embeddings for RAG. Pin source versions and extraction provenance. A fixed query set checks citation validity, applicability, unknown/no-match behavior, and misleading or instruction-bearing retrieved text. Corpus size alone is not acceptance. |
| A05 — Planning and provider connection | A02, A04 | Produce schema-valid plans from task/data/device constraints using retrieved evidence and eligible recipes; implement one optional real API planner. Exercise a live provider request and malformed output, timeout and rate-limit handling. Impossible resource/task constraints return a reason, not fabricated feasibility; API and compute usage both count against configured limits. |
| A06 — Research-guided experiments | A03, A05 | Wire Phase R inspection, majority/linear/MLP baselines, bounded candidate construction, real training and isolated evaluation into the persistent controller. Inspect weights/recipe to confirm requested topology; use identical split contracts across trials; winning bundle matches the report. No-improvement and failed trials are first-class results. |
| A07 — Deployment and rollback | A03, A06 | Stage a qualified bundle on the CPU target, check compatibility and authenticated prediction, atomically activate it and retain the previous version. Invalid/stale/unqualified artifacts cannot activate; failed health/smoke retains the old version; retry does not create duplicate deployments; rollback demonstrably restores prior predictions. |
| A08 — CLI and MCP adapters | A02; full workflow depends on A06/A07 | Thin adapters share the service's validation and authority. All lifecycle operations return structured results and job IDs. A fresh host/session recovers the same project; disconnect does not stop workers; tool/schema errors are actionable. |
| A09 — End-to-end compatibility and packaging | A04–A08 | Exercise the released package through real Codex, Claude Code and Cursor hosts using pinned versions and record the transcript/tool/run/artifact IDs. Each advertised host must complete a small real-data CPU workflow; the reference host also completes GPU-to-CPU deployment. Build clean install/configuration guides and record negative/recovery cases below. |
| A10 — Independent user pilot and next plan | A09 | Run the bounded pilot below, fix blockers, publish an internal evidence-backed release verdict and recipe/host/device matrix, and create the single active plan for Phase B from observed demand. Public release/publishing is a distinct action with its own scope. |

## Research content and executable recipes

Every evidence card contains paper ID/version/URL, supporting section, task,
data assumptions, architecture/loss/optimizer, reported compute, baselines,
limitations and implementation/license references. Retain source passages where
permitted and links back to the original so summaries can be checked. Separate
paper-reported results from reproduced Zerfoo results and user-run measurements.

Every executable recipe declares supported devices/precision, operator needs,
parameter bounds, export/load support and verification artifacts. Retrieval
cannot promote a literature-only idea into an executable capability. Rank
applicability and reproduction evidence, expose conflicting findings, and
return no supported recommendation when appropriate. The original
[RAG paper](https://arxiv.org/abs/2005.11401) motivates retrieved context; the
recipe/evaluation system proposed here supplies task-specific execution evidence.

Candidate selection first filters task, data, deployment memory/latency and
training budget, then considers research and validation results. Resource
estimates start as ranges and update after a bounded profiling trial. Monetary
caps require configured prices and budget reservation before dispatch; without
prices enforce compute/token caps and label monetary cost unknown. Enforce all
limits across retries and all trials, not separately per attempt. Record failed
experiments locally; sharing private results into a common corpus is opt-in.

## Release evidence and pilot

Mandatory Phase A acceptance cases (planned names, to map to executed suites):

| Case | Required behavior |
|---|---|
| AGENT-HOSTS | Each advertised host completes a real CPU lifecycle against the same released service |
| AGENT-REMOTE | Reference host trains on user GPU hardware and deploys on the separate CPU worker |
| AGENT-RECOVER | Host disconnect, controller restart and worker loss each preserve truthful recoverable state |
| AGENT-RETRY | Duplicate run/deploy requests and stale workers cannot duplicate work or replace qualified artifacts |
| AGENT-BUDGET | Trial, compute and API limits remain enforced across attempts; cancellation terminates worker execution |
| AGENT-RESEARCH | Supported/unsupported query cases, exact source provenance, invalid citation rejection and untrusted-text isolation |
| AGENT-QUALITY | Held-out selection, no-improvement outcome, label/topology correctness and matching artifact identities |
| AGENT-DEPLOY | Failed health check retains previous version; successful rollback restores its actual predictions |
| AGENT-BOUNDARY | Wrong worker/project credential or out-of-scope data/deployment request is rejected by the service |

Record commit and release hashes, package/host/device versions, named suites,
executed/pass/fail/skip counts, exact fixtures and artifact IDs, and raw evidence.
Require every mandatory case; skipped/inaccessible devices and empty test
selection are UNVERIFIED. Existing tests do not establish host integration.
Mocks cover error contracts but cannot substitute for the live provider, host,
training or deployment acceptance runs.

Pilot target: five consenting people who understand their dataset but are not
ML engineers. At least four independently reach an evaluated endpoint on the
supported workflow without editing training code or an engineer operating for
them. Record all five attempts and assistance, failures, elapsed time, training
resource use, API use and quality versus baseline. This is a proposed usability
gate, not evidence of market fit or a guarantee for arbitrary user datasets.
Also record whether users return with a second task within 14 days. Missed
completion targets trigger fixes to this workflow before expanding model types.
Recruitment and contacting participants are future tasks, not authorized sends
from this planning session.

## Scheduling, risks and expansion

Critical path: R → A01 → A02 → A03 → A06 → A07 → A08/A09 → A10.
A04 and A05 must also complete before A06; retrieval should not be postponed
until after the product is described as research-guided. A08 can begin with A02
but is complete only when it controls the real training/deployment workflow.

Initial planning allowances: R **25–40 engineering days**; A **25–40 engineering
days**, plus the pilot observation window. Roughly **10–16 engineering weeks**
for one full-time implementer before scheduling/hardware/pilot delays. These are
low-confidence scope allowances, not delivery commitments or assumptions about
agent speed. R01 re-estimates refinement; R12 re-estimates Phase A from measured
capabilities. No dated schedule is credible before those checks.

| Risk | Response |
|---|---|
| GPU or GGUF support is less complete than source suggests | Prove both early; add explicit remediation and revise estimates; CPU-only evidence cannot qualify a GPU claim |
| Scope expands into an entire ML platform | Keep one task family, two trainable recipes, single controller and two workers through the pilot |
| Research summaries overstate applicability | Preserve source conditions, require executable eligibility and user-data evaluation |
| Data is inadequate or misleading | Diagnose labels/leakage/class coverage before spending; permit baseline/no-model outcomes |
| Agent or network failures waste compute | Durable state, request idempotency, worker fencing, checkpoints and worker-enforced limits |
| Legacy users break during generalization | Compatibility wrappers, artifact migration fixtures, release notes and existing inference regression checks |
| The product works only for its developers | Fresh installs and independently observed pilot completion before adding model classes |

Phase B is demand-driven: choose forecasting or small-model adaptation first
based on pilot tasks. Freeze data/evaluation/compute requirements and prove
the complete artifact/serving path for each. Model inference availability does
not imply fine-tuning availability. Phase C adds constrained DSL compositions
only after operator/gradient validation, budgeted baseline experiments and
deployability checks. Arbitrary new kernels, general autonomous research and
frontier-scale pretraining require separate plans and evidence.
