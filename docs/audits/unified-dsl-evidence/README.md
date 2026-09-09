# ADR 097 implementation checks

Environment: Go 1.27.1, Darwin arm64, CPU. Baseline: `a58e8f47`.
`summary.json` records final source-file hashes and event counts (including
subtests, not a count of independent fixtures). The initial broad suites precede
later schema/capability checks; the final targeted run covers those edits.

- Broad CPU: `go test -p=2 -json -count=1 -short -timeout 180s` across `model`,
  `model/dsl`, `layers/core`, `layers/registry`, `inference`, `tabular`,
  `cmd/zerfoo-create`, `training/loss`, `training/optimizer`, `tests/architecture`.
  **2,303 passing test events, six skipped, zero failures.**
- Same packages with `-race`: **2,303 passing events, six skipped, one failure**:
  `TestPreTrain_TransferBenefit` measured fine-tuned accuracy 0.62 versus scratch
  0.80 (tolerance 0.15). No data-race warning was reported. This is not a green
  broad race verdict. Ten focused repetitions passed on both this branch and
  the unchanged baseline; the intermittent accuracy failure remains unresolved.
- Final `-race -count=1 -short` runs for `model`, `model/dsl`, and
  `cmd/zerfoo-create` pass; exact count is in `summary.json`.
- The real binary/subprocess lifecycle trains an explicit shared residual DSL
  graph for 120 steps, survives MCP exit, and predicts from a fresh CLI process.
  Validation: 30 rows, accuracy 0.9667, macro F1 0.9666. This fixture is not a
  scientific reproduction or a general quality guarantee.
- Bias accumulation regression failed on the old implementation (saved red
  output), then passed after the fix. The new DSL also has independent scalar
  forward expectations and a float64 central-difference gradient reference.
- Changed-package lint reports 58 issues in existing code, including old DSL
  duplication. Diff-scoped lint (`--new-from-rev=origin/main`) reports zero issues.
- GPU-required fixtures were skipped; no GPU, mixed-precision or external
  PyTorch qualification is claimed for this new composition route.

Commands used `GOWORK=off GOMAXPROCS=4` and the shared heavy-build lease.
Raw test logs are compressed without removing failed or skipped events.
