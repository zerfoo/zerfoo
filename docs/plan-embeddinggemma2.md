# EmbeddingGemma 2 native inference delivery plan

## Context and authority

Created 2026-10-14 at the owner's request to scope support and merge this plan.
This is an ordinary repository delivery plan, not an enrolled runtime lifecycle.
Plan identity: `zerfoo:embeddinggemma2`; authority identity: `zerfoo:repository`.
The authoritative task source is this file. Related roadmaps link here without
including or duplicating its task rows.

Outcome: Go applications load an exact EmbeddingGemma 2 GGUF and produce
reference-qualified text/code embeddings through library, retrieval and HTTP
interfaces. Full multimodal support remains in scope as a dependency-triggered
outline, expanded only after text qualification. Planning and merging this
artifact do not execute its tasks or authorize model downloads, GPU jobs,
publication, releases, deployment or production operations.

This supplies an additional encoder candidate for R3 in
[the embedded retrieval roadmap](plan-embedded-retrieval.md), under
[ADR-099](adr/099-embedded-capability-retrieval.md). Preserve its existing Qwen3
reference and training work. Reconcile [PR #1013](https://github.com/zerfoo/zerfoo/pull/1013)
before implementation; reuse the contextual loader, role adapter and parity
harness if their reviewed implementation lands. Do not create a competing API
or claim that open PR is already delivered. This feature plan does not replace
[the Phase 1 plan](plan.md), clear Trust gates, or supersede ADR-093's expansion
policy. EG1 preflight must record scheduling relative to Trust closeout and the
existing expansion front before any application work starts.

## Discovery and scope

Source inspected: main `bd895978` (2026-10-14); initial assessment used
`f9e1a42a`, which is now reachable from main. PR #1013 was open at
`f46a2bb0efba481991593f2ee1213c793213226e`. These are dated observations, not
future readiness claims. Scoped discovery traced public Embed, HTTP handlers,
retrieval roles, Gemma builders, GQA and GGUF configuration. The available code
graph reported zero changed functions/flows; literal source inspection supplies
the relevant model-specific evidence.

| Evidence | Implication |
|---|---|
| `inference/inference.go: Model.Embed` averages embedding-table rows; `api.go` and `serve/handlers.go` call it | Add contextual inference without silently substituting this baseline |
| `inference/arch_gemma4.go` has per-layer head/RoPE handling; `inference/load_gguf.go` marks Gemma 4 experimental | Reuse mechanisms only after parity; no inherited correctness claim |
| `layers/attention/grouped_query_attention.go` has bidirectional attention and a separate causal window branch | Qualify combined bidirectional window and padding masks in every eligible execution path |
| `retrieval/index.go` separates EmbedQuery and EmbedDocuments | Use the general role contract, with no consumer-specific code |
| PR #1013 adds `EmbeddingModel`, `ContextualEmbedder` and a parity command for Qwen3 | Reconcile its lifecycle and generalize its accepted contracts instead of duplicating them |

Scoped use cases (all planned, not supported by this plan alone):

| ID | Priority | Observable outcome |
|---|---|---|
| UC-EG-01 | P0 | Load a pinned text-only GGUF and embed text/code locally in Go |
| UC-EG-02 | P0 | Search documents/code with distinct query/document task formatting through retrieval.Embedder |
| UC-EG-03 | P0 | Obtain contextual embeddings through POST /v1/embeddings with validated dimensions and bounded batches |
| UC-EG-04 | P1 | Run the same contract on CPU and Spark CUDA with measured quality, memory and latency |
| UC-EG-05 | P2 | Embed images, audio, video and interleaved content into the same vector space with optional encoders |

Text milestone includes GGUF metadata/tensor mapping; tokenizer/special tokens;
24-layer local/global bidirectional encoder; projection-only PLE; exact norm,
scaling and GELU conventions; trained output projection; mask-aware mean pooling;
768/512/256/128 dimensions and L2 normalization after truncation; task formatting;
resource cleanup, cancellation, batch isolation and concurrency behavior;
CPU and GPU qualification; examples and honest verified-model documentation.
Runtime arithmetic stays in Engine[T]; reusable primitives retain generic types.
GGUF remains the runtime format. Python is reference-test infrastructure only.

Non-goals: training/fine-tuning, a new vector database, consumer integrations,
new device backends, mobile optimization, a performance superiority claim,
Gemma 4 generative-model repair, releases and deployment. Lower-bit quantization
is a separately reported qualification candidate: failure cannot be hidden by
full-precision parity, and an unqualified format must be rejected or explicitly
unsupported. BF16 weights with FP32 computation are the initial path; forbid
unsafe FP16 activation paths for this model.

## Reference contract and open questions

Primary references inspected 2026-10-14:

Reference: [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/): modular 270M text / 740M total model, advertised 8K context.
Reference: [Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2): task prefixes, mean pooling, 512-to-768 projection, dimension truncation and precision guidance.
Reference: [Checkpoint configuration](https://huggingface.co/google/embeddinggemma-2/blob/main/config.json): per-layer head dimensions/KV heads, local/global RoPE, layer pattern and encoder configs.
Reference: [Reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py): bidirectional masks and projection-only PLE; do not substitute Gemma 4 Edge token-lookup PLE.
Reference: [GGUF artifacts](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF): published format availability, not evidence of Zerfoo compatibility.

These moving URLs are discovery references. T-EG1.1 must freeze immutable revisions,
licenses, hashes and exact artifact identities before any numerical claim.
Resolve window semantics (model card 1024 versus config 512), supported input
limit (advertised 8192 versus larger positional configuration), GGUF architecture
identifier/tensor names, pooling inclusion of prompts/special tokens/padding,
projection ordering, and tokenizer parity. Do not infer any of these from the
model name. Freeze a layer-wise oracle and measured tolerances before evaluating
Zerfoo output. Pin processor and reference runtime versions together.

API freeze must specify load options, task/role/title formatting, double-prefix
avoidance, dimension validation, overlength policy, model selection, usage-token
accounting, cancellation and concurrency ownership. Preserve the documented
legacy Model.Embed behavior or explicitly version its replacement; no silent
behavior change. Extend the accepted contextual interface from PR #1013 where
possible. HTTP can accept already-formatted text; any task extension must be
explicitly documented rather than claimed as an OpenAI-standard field.

## Verification contract

T-EG1.1 produces versioned fixtures and a runnable reference/parity recipe, including
fixture provenance, exact files/hashes, per-layer tensors and final vectors.
This recipe is currently a gap; the existing decoder parity suite alone is
insufficient. Acceptance thresholds must be justified by reference repeatability
and precision, then frozen before comparing candidate output. Record cosine and
maximum component error, not shape or finiteness alone. Retrieval tolerances and
sample/split definitions must likewise be fixed before running the candidate.

Required cases: multilingual text and code; task/title prefixes; special tokens;
empty inputs under the frozen policy; padding and mixed lengths; singleton versus
batch equality; repeated and concurrent requests; cancellation and cleanup;
invalid dimensions/types/model IDs; unsupported media; local-window boundaries;
8192-budget boundaries including prefixes; NaN/Inf rejection; all four dimensions.
Long-context cases must detect vacuous skips and wrong-model fallback. At least
one non-consumer-specific retrieval dataset and a preserved Qwen3/other supported
path are required; no single consumer can establish framework correctness.

CPU verification uses focused changed-package tests and required `go test ./...`,
changed-code formatting/lint and repository CI. API checks exercise real HTTP
requests and assert status/body/usage, including invalid requests. New or changed
ops require ADR-091 gradcheck/OpInfo and CPU/GPU parity; CUDA kernels additionally
require the PyTorch oracle. GPU acceptance uses `scripts/dgx-validate.sh` and
bounded Spark manifests on the GB10, with one GPU pod at a time; never interactive
SSH benchmarks or CUDA tests. Extend the gate when embedding cases are absent.
A GPU-dependent change cannot merge on CPU-only evidence, even in the CPU-first
batch. Shared build leases/resource limits apply when tests are executed.

Report exact artifact and source revisions, backend, dtype, case counts, skipped
cases, raw outputs, retrieval Recall@k/NDCG@10, p50/p95 latency and peak memory.
Google's device-memory claims are not Zerfoo targets. Quantized variants receive
separate numerical/ranking results. No support/throughput claim precedes passing
evidence. Store events/results in devlog and reproducible fixtures/reports in
repository-standard test/bench locations; stable contracts belong in design.md
and material accepted decisions in ADRs.

## Checkable work breakdown

Owner roles below are assignments of responsibility, not running agents:
Delivery-agent authors and verifies; Independent-reviewer is a separate inherited
session and never the author of the covered revision. Estimates include checks
and gates in the batch envelope, not additive promises for each row.

### E-EG1 -- Frozen artifacts and reusable interface contract

fidelity: executable

Acceptance: a pinned, reproducible encoder contract resolves artifact ambiguity
and overlapping Qwen3 work before implementation. Batch envelope: 2–3 engineer-days.

- [ ] T-EG1.0 Reconcile scope and readiness  Owner: Delivery-agent  Est: within batch  kind: agent  stage: preflight  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  acc: [Record current main and PR 1013 disposition, Trust/expansion scheduling decision, available model access and compute authority; identify exact reusable contracts and preserve unrelated work; unresolved policy or artifact access blocks dependent work]
- [ ] T-EG1.1 Freeze model and API reference contract  Owner: Delivery-agent  Est: 2–3d including batch gates  kind: agent  stage: implement  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG1.0]  acc: [Pin checkpoint, tokenizer, processor, GGUF converter/artifact and reference revisions and hashes; resolve window/context, PLE, projection and pooling semantics; freeze public and HTTP contracts, oracle fixtures, tolerances, held-out ranking cases and executable recipes; reconcile PR 1013 without duplicating its API; record material decisions in an ADR if needed]
- [ ] T-EG1.2 Verify contract and fixture reproducibility  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG1.1]  acc: [Independent fixture regeneration matches declared reference tolerances; artifact hashes and licenses checked; task/HTTP contracts have testable examples; Wazi gate and documentation/link checks pass for the exact candidate]
- [ ] T-EG1.3 Independently review T-EG1.1 candidate  Owner: Independent-reviewer  Est: within batch  kind: agent  stage: review  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG1.2]  acc: [Record independent reviewer, PR, base/head SHAs, findings and disposition covering the entire contract/fixture candidate; all accepted blockers resolved and re-reviewed]
- [ ] T-EG1.4 Rebase merge reviewed contract  Owner: Delivery-agent  Est: within batch  kind: agent  stage: merge  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG1.3]  acc: [Required checks and review pass on exact guarded head; current target checked immediately before GitHub rebase merge; record landed SHA]
- [ ] T-EG1.5 Verify landed contract  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify-landed  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG1.4]  acc: [Landed SHA reachable from main and reviewed contract/fixtures preserved; relevant checks and plan conformance pass on landed bytes; no application support claim made]

### E-EG2 -- Contextual text/code embeddings through Go and HTTP

fidelity: executable

Acceptance: CPU text embeddings and public interfaces satisfy the frozen oracle
without changing legacy behavior silently. Batch envelope: 9–15 engineer-days.

- [ ] T-EG2.0 Preflight encoder candidate  Owner: Delivery-agent  Est: within batch  kind: agent  stage: preflight  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG1.5]  acc: [Revalidate landed contract, PR 1013 reconciliation and dependency versions; reserve isolated worktree and required build capacity; ensure GPU gate is runnable before any GPU-dependent changes]
- [ ] T-EG2.1 Load and execute the text encoder  Owner: Delivery-agent  Est: 6–10d including local checks  kind: agent  stage: implement  verifies: [UC-EG-01]  blocked-by: [T-EG2.0]  acc: [Exact GGUF mapping and tokenizer load; model-specific graph implements bidirectional local/global attention, projection-only PLE, norms/scales/GELU and trained output projection in Engine operations; intermediate/final FP32 output matches frozen oracle; invalid artifacts and unsafe FP16 paths fail explicitly; no LM head or table-average fallback]
- [ ] T-EG2.2 Integrate Go retrieval and HTTP contracts  Owner: Delivery-agent  Est: 3–5d including local checks  kind: agent  stage: implement  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG2.1]  acc: [Reuse accepted contextual API; mask-aware pooling and normalized 768/512/256/128 outputs work with query/document/title formatting; HTTP routes to encoder with validated model/dimensions, accurate usage, bounded batches and documented task behavior; cancellation, Close and concurrency are safe; legacy APIs remain covered; runnable text/code retrieval example included]
- [ ] T-EG2.3 Verify complete text candidate  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG2.2]  acc: [Cover T-EG2.1 and T-EG2.2 with frozen layer/vector and retrieval oracle cases, all verification-contract edges, real HTTP status/body tests and regressions for another supported architecture; format/lint, CPU suite, required CI and applicable ADR-091 GPU/op/kernel gates pass on exact candidate; plan conformance passes; record skips as blockers where required]
- [ ] T-EG2.4 Independently review T-EG2.1 and T-EG2.2 candidate  Owner: Independent-reviewer  Est: within batch  kind: agent  stage: review  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG2.3]  acc: [Record independent reviewer, PR and exact base/head SHAs, implementation/test/plan coverage, findings and disposition; accepted blockers fixed with affected verification and re-review]
- [ ] T-EG2.5 Rebase merge reviewed text candidate  Owner: Delivery-agent  Est: within batch  kind: agent  stage: merge  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG2.4]  acc: [Required checks and independent review satisfied; compare current base and exact reviewed head before guarded GitHub rebase merge; record landed SHA]
- [ ] T-EG2.6 Verify landed text behavior  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify-landed  verifies: [UC-EG-01, UC-EG-02, UC-EG-03]  blocked-by: [T-EG2.5]  acc: [Verify main reachability and reviewed-tree preservation; run frozen Go/HTTP embedding smoke on landed revision, refresh invalidated checks and Wazi conformance; record CPU-qualified versus still-unqualified GPU status accurately]

### E-EG3 -- GPU, retrieval and support qualification

fidelity: executable

Acceptance: advertised backend/precision support is backed by reproducible
numerical, ranking and resource results. Batch envelope: 4–7 engineer-days.

- [ ] T-EG3.0 Preflight GPU qualification  Owner: Delivery-agent  Est: within batch  kind: agent  stage: preflight  verifies: [UC-EG-04]  blocked-by: [T-EG2.6]  acc: [Verify exact landed encoder and artifact hashes, current Spark authority/capacity, single-GPU serialization and bounded run manifests; no interactive SSH compute; missing resources leave qualification blocked]
- [ ] T-EG3.1 Qualify supported backends and document limits  Owner: Delivery-agent  Est: 4–7d including batch gates  kind: agent  stage: implement  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG3.0]  acc: [Wire embedding cases into dgx-validate and reproducible benchmarks; run frozen CPU/CUDA and reference parity, ranking, long-context and all-dimension cases; qualify or explicitly exclude selected quantized artifacts; measure p50/p95 and peak memory; publish exact model/backend/dtype evidence, examples and limitations in verified-models/design/devlog without unsupported claims]
- [ ] T-EG3.2 Verify qualification candidate  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG3.1]  acc: [Verify T-EG3.1 raw reports match exact source/model identities, expected nonzero case counts and frozen thresholds; required CPU/CI and ADR-091 gates pass, real HTTP smoke reaches qualified encoder, public claims match reports, and Wazi conformance passes]
- [ ] T-EG3.3 Independently review T-EG3.1 candidate  Owner: Independent-reviewer  Est: within batch  kind: agent  stage: review  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG3.2]  acc: [Independent reviewer records PR, exact base/head SHAs, report provenance and code/test/documentation coverage; accepted findings resolved with affected verification and re-review]
- [ ] T-EG3.4 Rebase merge qualification candidate  Owner: Delivery-agent  Est: within batch  kind: agent  stage: merge  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG3.3]  acc: [Required checks and review pass on exact guarded head; current target checked before GitHub rebase merge; landed SHA recorded]
- [ ] T-EG3.5 Verify landed qualified support  Owner: Delivery-agent  Est: within batch  kind: agent  stage: verify-landed  verifies: [UC-EG-01, UC-EG-02, UC-EG-03, UC-EG-04]  blocked-by: [T-EG3.4]  acc: [Landed SHA reachable from main; qualified fixtures, Go/HTTP smoke and Wazi gate pass on landed bytes; reuse GPU evidence only when source/inputs remain equivalent or rerun affected Spark checks; record completed text milestone and remaining multimodal scope without implying release]

### E-EG4 -- Full multimodal embeddings

fidelity: outline

Acceptance: images, audio, video and interleaved inputs reach the same qualified
embedding space, with selective encoder loading and explicit modality limits.
Preliminary additional estimate: 20–35 engineer-days, to be revised after T-EG3.5.

- [ ] T-EG4.1 Expand multimodal delivery plan after text qualification  Owner: Delivery-agent  Est: re-estimate at gate  kind: agent  stage: author  verifies: [UC-EG-05]  blocked-by: [T-EG3.5]  acc: [Freeze vision/audio companion GGUF availability, encoders/projectors, image processing, audio sampling/features/chunk masks, video frames/timestamps and interleaving/token budget; specify optional loading and Go/HTTP media custody/resource limits; add stable implementation, cross-modal oracle, review, merge and landed tasks with owners and measured estimates; validate, independently review and merge expanded plan before multimodal implementation]

## Batches and execution route

Execution style: auto, requested member ceiling 3. Intended executor: current;
capability label: attended ordinary repository delivery. Inherit the active
session's model, reasoning, permissions and authentication route unchanged,
including the independent reviewer. Exact runtime selection is recorded at
future dispatch; no Kazi/Foundry readiness or provider admission is implied.

| Batch | Outcome / members | External prerequisite | Owner | Verification | Review / merge / landed | Selection | Requested / actual members | Split condition |
|---|---|---|---|---|---|---|---|---|
| B-EG1 | Frozen contract: T-EG1.1 | T-EG1.0 preflight | Delivery-agent | T-EG1.2 | T-EG1.3 / T-EG1.4 / T-EG1.5 | inherited current | 3 / 1 | Artifact access or PR 1013 ownership unresolved |
| B-EG2 | Native text/code vertical slice: T-EG2.1, T-EG2.2 | T-EG1.5 and T-EG2.0 preflight | Delivery-agent | T-EG2.3 covers both members | T-EG2.4 / T-EG2.5 / T-EG2.6 | inherited current | 3 / 2 | Cross-repo primitive change, incompatible owner, or review size requires separate candidate |
| B-EG3 | Qualified support: T-EG3.1 | T-EG2.6 and T-EG3.0 preflight | Delivery-agent | T-EG3.2 | T-EG3.3 / T-EG3.4 / T-EG3.5 | inherited current | 3 / 1 | Backend/quantization fault needs independent remediation |

These batches depend on predecessor landed evidence and therefore serialize;
no speculative parallel implementation is needed. Within B-EG2 integrate in
member dependency order. Independent review remains a separate session. A split
requires preserving IDs, adding full candidate gates and revalidating this plan
before dispatch. ztensor/ztoken changes, if discovered, require explicit owning
repository tasks/review/landing and a pinned dependency before downstream merge;
no unreviewed cross-repo patch is hidden inside these estimates.

Accepted review findings require appended stable-ID fix, affected-verification
and re-review rows; fixes depend on implementation, not the unsatisfied review.
Repoint merge dependencies to the passing re-review and retain original evidence.
Never mark a gate complete merely to unblock scheduling.

## Risks, estimates and handoff

The 15–25 engineer-day text/code estimate is an initial range, not a calendar
commitment; the batch envelopes sum to that range. Full multimodal is roughly
35–60 total days. T-EG1.1 must re-estimate after exact artifacts and overlap are
resolved. The principal risks are reference/config drift, incomplete GGUF
conversion, experimental Gemma reuse, decoder-only optimization assumptions,
FP16 activation overflow, non-vacuous long-context tests, and shared GPU capacity.
Time-box correctness investigations and retain experimental status if gates fail.
Do not broaden into fixing all Gemma 4 generative behavior.

Tools qualified for planning: Git and gh; local Python/Node; installed Wazi repair
CLI, source reader and matching offline semantic verifier (contract 0.0.1).
Capability resolver selected engineering guidance with requirements_met=true;
this is guidance availability, not runtime execution qualification. Existing
MCP/desktop tools were unchanged. No additional tools/plugins were installed.
CPU/GPU execution, tokenizer changes and model fixtures remain future preflight
requirements. Wazi source/reader/verifier/contract digests and independent-review
base/head are recorded in the planning PR receipt, then checked on landed bytes.

2026-10-14: created scoped plan; all application tasks remain unchecked. Related
roadmaps retain their existing status and decisions. Next runnable stage after
separate execution authorization is T-EG1.0. Release, deployment and operations
have no tasks because they are outside this inference-support delivery scope.
