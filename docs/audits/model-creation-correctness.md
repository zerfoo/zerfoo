# Model-creation correctness baseline — R01

R01 audit complete; Phase R implementation and release remain **UNVERIFIED**.
Owner: Codex. Recorded 2026-09-08 local date (raw logs include timezone).
Baseline: `60c8742cccc18ccb06d54ba3ad115e82bc13718e`, Go 1.27.1,
Darwin arm64, `GOWORK=off`, released ztensor v1.20.0. The existing working tree
contained planning changes; this audit makes no production behavior changes.
This is not a clean-checkout R10 release verdict or a Go 1.26 CI verdict.

## Deliverables and decision

- [Proposed reference ADR](../adr/096-model-creation-reference-contract.md).
- [Real-data manifest](../../tabular/testdata/model_creation/manifest.json),
  original data, normalized CSV and [attribution](../../tabular/testdata/model_creation/README.md).
- [Frozen numeric acceptance contract](../../tabular/testdata/model_creation/contracts.json).
- [Raw evidence and input hashes](model-creation-evidence/summary.json).
- Explicitly tagged audit probes in `tabular/model_creation_audit_test.go` and
  `cmd/cli/model_creation_audit_test.go`; run with `-tags r01audit`.

GGUF tensor export/reload is feasible with existing released primitives. The
spike reconstructs a legacy classifier inside a test and verifies its actual
Predict output; there is no production tabular GGUF loader yet. Keep the
planned R02–R12 order with R07 implementing that adapter and bundle validation.
No alternate default weight format, repository, or top-level package is needed.

## Actual lifecycle boundaries

Paths are relative to repository root; symbols identify the audited callers.

| Boundary | Actual production call path | Evidence / missing contract | Owner |
|---|---|---|---|
| General training CLI | `cmd/cli/train.go`: TrainCommand.Run → runLocal/runWorker → trainLoop → newTrainModel → ShardedModule | trainLoop manufactures inputs, loss and gradients; it never reads user training data. Existing CLI tests exercise this example, not real learning. | R04 |
| Real tabular CLI entry | `cmd/cli/automl.go`: AutoMLCommand.Run → newTabularWorker → readTabularCSV → tabularWorker.RunTrial | Reads CSV with final integer-label column; no target selection, schema manifest, finite-value check, preprocessing or split identity. Unknown metric accepted by worker. | R03/R05 |
| Model construction | `tabular/train.go`: Train → NewModel in model.go | Fixed 3-class head; HiddenDims must be nonempty. No linear recipe or persisted user labels. Baseline Model/Train suites pass for legacy contract. | R02 |
| Split and seed | Train → splitData; model.go newMLPLayer | Global rand/v2.Perm and NormFloat64. CLI seed controls search proposals, not training. Random validation split is private to Train. | R03/R04/R06 |
| Forward/loss | Train → forwardPass → linearForward → functional.Linear; applyActivation; crossEntropyLoss → loss.CrossEntropyLoss | Real Engine matmul/activation/softmax path. Independent fixed-weight prediction and 15-parameter ReLU finite differences pass. Full oracle matrix unverified. | R02/R10 |
| Gradient/update | Train → backwardPass → buildParams/AdamW.Step | Manual backprop; persistent graph.Parameter wrappers. Engine matmul/reduction and CPU mixed-moment AdamW. Existing numeric reference suites pass; complete state export absent. | R02/R06/R10 |
| Selection | RunTrial → model.Predict for every w.data row → accuracy; Coordinator.Run maximizes Score | Production witness proves in-sample contamination and ignored loss/unknown metric. Train's internal validation loss is not the worker score. | R05 |
| Winning artifact | AutoMLCommand.Run → bestConfigOutput JSON | Writes params/score only; local trained model is discarded. No winner hash, model export or report identity. | R05/R07 |
| Persistence | tabular.Save/Load → ZTAB; general CLI → fsdp.SaveCheckpoint | ZTAB has config and weights only. FSDP GGUF flattens parameters and does not preserve full resumable state. Neither is the proposed bundle contract. | R06/R07 |
| GGUF bridge | ztensor/gguf.Writer.AddTensorF32/Write → model/gguf.Parse/LoadTensors → test-only Model construction → Predict | Four tensors round-trip bit exactly, including [2,3] weights. Six predictions match independent expected labels/confidences. Public loader/manifest/fresh-process bridge absent. | R07 |
| Serving | serve/classify.go: WithClassifier → handleClassify → sentiment Classifier.Classify | Text input only; no tabular artifact-to-HTTP caller exists. Name the gap rather than treating HTTP 200 as evidence. | R08 |
| GPU/recovery | tabular/train_gpu_test.go: TestTrain_GPU → NewGPUEngine → Train | cuda build tag and unavailable-GPU skip guard. Not built in this CPU run. No complete train/resume/CPU-serve lifecycle. | R09 |

## Runtime and compatibility inventory

Supported recipes require transpose, matmul, broadcast bias addition, ReLU,
softmax, sum reductions, multiplication, subtraction, scalar operations and
parameter gradient reset. `functional.Linear` and ReLU are shared entry points;
`training/loss/cross_entropy_loss.go` owns stable softmax and mean loss.
`tabular.backwardPass` currently constructs softmax-minus-one-hot on host slices;
R02 must account for this engine-boundary violation, not copy it into new APIs.
Cross-entropy scalar accumulation and argmax also access host data: distinguish
reporting from tensor math during the R02 boundary cleanup.

`training/optimizer/adamw.go` uses mixed-precision moment sidecars for fp32,
parameter-pointer keys, an internal step counter, and backend-specific update
paths. Reuse its existing numerical fixes. Add validated, stable-name state
export/import for R06; weights alone cannot resume it. Audit destination aliasing,
parameter/gradient pointer stability, accumulation/reset and GPU eager-buffer
lifetimes with 100 reuse iterations against the independent oracle before R10.
No scheduler is active in the reference recipe; any future scheduler must add
its state to checkpoint identity. Training and Predict create background
contexts. AutoML's Worker/Coordinator interfaces have no context argument.

Legacy callers: ensemble.go stores []*Model, uses three probabilities and a
three-class meta-learner; lora.go/pretrain.go access Model layers/head and have
separate training loops. Preserve these adapters and their current tests.
TabNet, SAINT, ResNet and FTTransformer have their own paths and are excluded
from certification here; their CPU baseline tests do not generalize the new
classifier contract. DropoutRate is accepted but not used in Model/Train.
The initial recipes freeze dropout=0 and ReLU; GELU and adapter learning require
separate oracle coverage before a support claim.

## Executed baseline

Command (under the shared build lease, initial 1-minute load below 2):

```sh
GOWORK=off go test -json -count=1 -short -timeout 180s ./tabular ./cmd/cli ./model/gguf ./training/automl ./training/loss ./training/optimizer ./layers/functional
```

[Raw JSON stream](model-creation-evidence/baseline.jsonl.gz) is gzip-compressed
without altering its bytes; [summary](model-creation-evidence/summary.json)
records its SHA-256 and the audit probe hashes.

Counts below are terminal named Go test events, **including parent tests and
subtests**, not independent scenarios. All seven package verdicts PASS.

| Package suite | Pass | Fail | Skip |
|---|---:|---:|---:|
| tabular | 170 | 0 | 0 |
| cmd/cli | 454 | 0 | 0 |
| model/gguf | 328 | 0 | 0 |
| training/automl | 37 | 0 | 0 |
| training/loss | 140 | 0 | 0 |
| training/optimizer | 82 | 0 | 0 |
| layers/functional | 114 | 0 | 0 |

Includes TestTrain_Convergence, TestRoundTrip, TestTabularWorker_RunTrial,
TestAutoMLCommand_TabularModel, TestAdamW_MixedV_MatchesReference and
TestAdamW_MixedV_TracksFullF64Reference. Existing assertions retain their own
historical tolerances; their PASS is not a pass against every new frozen bound.
`.github/workflows/ci.yml` runs Go 1.26 short CPU tests with race; this local run
uses 1.27.1 without race. The new r01audit probes are deliberately opt-in and
not part of default CI. Build-excluded GPU tests are **not executed**, even
though the CPU stream has zero skips. No PyTorch or GPU result was collected.

```sh
go test -json -tags r01audit -run '^TestR01' -count=1 -timeout 90s ./tabular
go test -json -tags r01audit -run '^TestR01' -count=1 -timeout 90s ./cmd/cli
```

- TestR01GGUFClassifierSpike: PASS, 1 test, 4 exact tensor round trips and
  6 prediction comparisons, max confidence absolute error 2.7966647e-8;
  atol=1e-6, rtol=1e-5 fixed before execution.
- TestR01ReLUGradientFiniteDifference: PASS, 1 test, 15 parameter elements,
  independent scalar float64 forward/mean-cross-entropy central differences;
  max absolute gradient error 4.3091444e-8; atol=2e-5, rtol=2e-4, step=1e-3
  fixed before execution. This small fixture does not replace autodiff coverage.
- TestR01ProductionWorkerEvaluationLeakage: defect reproduced, 1 parent plus
  3 passing witness subtests (accuracy/loss/unsupported), zero skips/failures.
  Six identical feature vectors with two labels per class always produce one
  predicted class and whole-data accuracy 2/6. Train reserves floor(6×0.2)=1
  validation row, whose accuracy must be 0 or 1. Actual worker score is 1/3
  for all three metric names. No fake worker or patched production seam is used.
  A passing witness establishes the defect, not healthy selection behavior.

## Findings ledger

| ID | Baseline finding / status | Correction owner | Re-verification |
|---|---|---|---|
| MC01 | Synthetic general train CLI: source-confirmed missing real-data route | R04 | Real CLI pinned-data learning + missing-data rejection required |
| MC02 | Evaluation leakage: production-reproduced | R03/R05 | Replace tagged witness with isolated-split regression; demonstrate red before fix |
| MC03 | Metric names ignored: production-reproduced; coordinator always maximizes by source | R05 | Known metric fixture, minimize/maximize ranking and unknown-metric rejection |
| MC04 | Winner not retained: source-confirmed | R05/R07 | Winner hash must equal reloaded/evaluated bundle hash |
| MC05 | Class count, linear recipe and label map absent: source-confirmed | R02 | Binary, legacy 3-class and 5-class cases |
| MC06 | Context, deterministic training seed, complete checkpoint absent | R04/R06 | Interrupted and continuous state comparison, exact IDs and frozen tolerances |
| MC07 | Generic deployment loader/HTTP route absent; GGUF primitive feasibility proven | R07/R08 | Fresh-process export/load/migration and 30 raw HTTP records |
| MC08 | ZTAB Load trusts config dimensions and lengths; malformed-config safety unverified | R07 | At least 5 malformed artifact cases before migration support |
| MC09 | CPU/GPU/oracle convention, lifetime and convergence matrix unverified | R09/R10 | Real Spark run and independent reference; no CPU fallback verdict |
| MC10 | Engine boundary violations and unused dropout setting in existing tabular training | R02/R04 | Engine-based gradient construction; reject unsupported settings |

Owner refers to the accountable subsequent task; those fixes have not been
performed by this baseline audit. Existing numerical suite results are retained
rather than reported as unverified solely because their task checkboxes are old.

## Dependency reconciliation (source + GitHub API on this audit date)

- E147: ztensor#178 merged 2026-08-24; go.mod already pins v1.20.0. #179 is
  still open/unmerged. Do not claim ZTENSOR_DETERMINISTIC is shipped; per-run RNG
  and same-runtime CPU determinism need their own implementation. GPU exact
  determinism remains unverified. zmf#14 remains open, but zmf is absent from
  go.mod and not on this recipe's dependency path. Do not block R02 on it.
- E148: ztensor#180 is closed; current GQA source restores the optional-symbol
  guarded fused path (local commits 616845d7/a6c9dcbd). Reuse, do not reimplement.
  Linear/ReLU MLP does not execute GQA or RepeatInterleave.
- E149: #981 already has the recorded multi-head cache fix. #982 Gather and
  #983 PatchTST convergence remain open. This recipe does not use KV caches,
  Gather, or PatchTST; shared allocator/optimizer behavior still needs R09 proof.
  A reproduced shared-runtime defect becomes blocking regardless of its label.
- E152: #994 is closed as of 2026-08-24; RoPE fix #993 is already recorded in
  source/history. Eager tabular training does not execute the traced text
  inference path. T152.3 parity honesty remains useful policy, not evidence that
  this new lifecycle passed. Keep existing inference claims under the Trust plan.

No old task was marked done merely from issue closure, and no GPU fix was
re-verified by this CPU-only audit.

## Frozen acceptance matrix and remaining evidence

The machine-readable contract is authoritative for numeric bounds; no learning
run on Iris was used to select them. Small fixed fp32 comparisons use rounding
allowances; the larger GPU bound follows existing kernel guidance. If hardware
fails a bound, record the failure and diagnose before any explicit revision.
Iris is a lifecycle fixture, not a product quality or market benchmark.

| Mandatory case | Frozen requirement | Existing evidence / next suite |
|---|---|---|
| REF-NUMERICS | batches 1/2/7, classes 2/3/5; logits -1000/0/1000; finite combined fp32/64 tolerances; AdamW steps 1/5 | Small forward/gradient spike + existing optimizer references; full oracle matrix R10 |
| REF-CORRECTNESS | all 12 cases, ≥1 execution each, 0 failures/skips; every confirmed corrected defect red-to-green | This baseline + MC ledger; clean candidate verification R10 |
| REF-LABELS | 2/3/5 classes; shape/label identity exact; probability sum error ≤1e-5 | Legacy 3-class tests; generic API R02 |
| REF-DATA | 90/30/30 rows, 0 row/group overlap, 0 future training rows, 0 nontraining fit rows | Fixed manifest; pipeline/leakage traps R03 |
| REF-LEARN | CPU and GPU, each seed 17/42/91, linear and MLP; accuracy ≥26/30, macro-F1 ≥0.85, mean CE ≤0.6; ≤1200 steps/120s/2 GiB | Real CLI recipe R04/R09, presently unverified |
| REF-SELECT | metric fixture accuracy .625 and macro-F1 .6111111111111112 ±1e-12; accuracy gain ≥.5 over majority; exact winning artifact identity | Contamination witness is a confirmed failure; fixed selection R05 |
| REF-RESUME | interrupt at steps 1 and 17; compare at 25; CPU weights bit exact, GPU combined 1e-5/1e-4; 0 state-identity mismatches | Checkpoint API R06 |
| REF-LIMIT | cancellation ≤2s; deadline checks ≤1s; 0 step/trial overshoot | Context/limits R04/R06 |
| REF-BUNDLE | 0 weight/hash mismatches; ≥1 fresh-process and legacy case, ≥5 malformed cases | In-process primitive spike only; R07 |
| REF-SERVE | ≥30 raw records; 0 label mismatches, probability atol=1e-6/rtol=1e-5; ≥5 invalid cases | Route absent; R08 |
| REF-GPU | both recipes ×3 seeds; same quality bounds; CPU/GPU accuracy delta ≤2/30; 100 runtime reuse iterations | Spark GPU learning, resume and CPU deployment R09 |
| REF-GATE | all 12 cases required, ≥1 execution each, 0 failures/skips; ≥3 negative gates (missing data, identity, empty selection) | Release gate R10 |

Independent reference tooling: scalar float64 reference here is hash-pinned with
the probe source. The PyTorch oracle container reference is
`nvcr.io/nvidia/pytorch:26.02-py3` from ADR-091; resolve/pin its digest and record
actual torch version when running it. No executed autodiff evidence is claimed.
Use central differences away from ReLU kinks, exact explicit weights, dropout=0,
and parameter state comparison, not just decreasing loss or CPU/GPU agreement.

## Revised effort and execution order

Engineering-day allowances, not delivery promises; hardware waiting excluded.
The GGUF primitive exists, reducing format uncertainty. Context/state and
selection still need real API work. The larger allowance includes explicit
legacy malformed-load handling and shared engine-boundary verification.

| Task | Revised allowance | Main work |
|---|---:|---|
| R01 | audit delivered | Reference contract, fixture, baseline and spike |
| R02 | 3–4 | Generic classifier/linear path, legacy adapters, engine contract |
| R03 | 2–3 | Data inspection, immutable splits and preprocessing |
| R04 | 3–4 | Real CLI training and cancellation |
| R05 | 2–3 | Correct metrics, selection and winner retention |
| R06 | 4–6 | Optimizer/RNG checkpoints, recovery and hard limits |
| R07 | 3–4 | GGUF bundle, validated loader and ZTAB migration |
| R08 | 2–3 | Prediction endpoint with persisted preprocessing |
| R09 | 2–4 | GPU learning/recovery and CPU deployment |
| R10 | 4–6 | Independent oracle, lifetime stress and clean release gate |
| R11 | 1–2 | Reproducible release package and guide |
| R12 | 1 | Reconcile next active plan with measured capabilities |

Remaining R02–R12 allowance: **27–40 engineering days**, plus actual GPU/access
waiting and unforeseen numerical fixes. R02 and R03 can start from this
contract. R04 follows both; R05/R06/R07 then converge on R08/R09/R10. Phase A
remains deferred until R12 reconciles the Trust/Traction plan from evidence.

## Lint baseline

`golangci-lint run --build-tags r01audit ./tabular` reports 16 existing issues;
`./cmd/cli` reports 87. No diagnostic names either new audit test file.
Logs: [tabular](model-creation-evidence/tabular-lint.txt) and
[CLI](model-creation-evidence/cli-lint.txt). These commands failed; no clean
package-lint verdict is claimed. Fix affected legacy paths during their planned
refinement tasks instead of changing production behavior during R01 baseline
capture. `gofmt` and `git diff --check` are clean for the audit additions.


## Implementation follow-up (2026-09-09)

The sections above are the immutable R01 baseline, not the current defect state.
R02/R03 now implement generic classifiers and inspected/replayable datasets.
R04 connects real bounded training to `train tabular`; synthetic general/FSDP
training is now test-only and the production route rejects unsupported requests.
GGUF bundle save/load, fresh-process roundtrip, corruption rejection and explicit
ZTAB migration are implemented. Complete checkpoint recovery and HTTP serving
are still outstanding, so this is not a Phase R release verdict.

MC02/MC03 are corrected in the real AutoML worker: train-only updates,
validation-only scoring, unknown-metric rejection and proper loss direction.
Its exact winning model is persisted when an output is requested. See
`automl-isolation-red.txt` and `automl-isolation-green.txt` in the evidence folder.
The old tagged worker witness is archived as `r01-worker-witness.go.txt`;
production regression coverage lives in cmd/cli/automl_isolation_test.go.
The baseline witness hash in summary.json identifies that archived source.

Before the first Iris learning run, R03 found duplicate-record crossings in the
initial fixture assignments. Manifest v2 groups exact duplicates, still uses
90/30/30 rows, and retains all frozen numeric thresholds. The old manifest hash
in the baseline summary is historical; current input hashes are recorded in the
implementation evidence. No learning result informed this correction.

All six CPU recipe/seed runs met frozen quality thresholds: linear 29/30 for
seeds 17/42/91; MLP 29/30, 28/30 and 28/30 respectively. Max observed mean CE
was .252161, and minimum macro-F1 was .933333. The real CLI's trained artifact
reloaded at 28/30 for MLP seed 42. These are CPU-only results, not GPU, recovery,
or independent PyTorch/autodiff qualification.

The initial root architecture run found the pre-existing ignored tmp directory.
All four architecture checks passed in an isolated source snapshot without that
unrelated directory. It was not removed and the allowlist was not weakened.
