# Model-creation reference fixtures

`iris.data` is the unchanged Iris dataset distributed by the UCI Machine
Learning Repository. Attribution: Fisher, R. (1936). Iris [Dataset].
[DOI:10.24432/C56C76](https://doi.org/10.24432/C56C76).
The [source page](https://archive.ics.uci.edu/dataset/53/iris) licenses it under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Downloaded from the source ZIP recorded in `manifest.json` on 2026-09-08.

`iris.csv` adds a descriptive header and normalizes line endings. Measurements,
labels, and original row order are preserved, including historical source
measurement discrepancies; this is not the alternate `bezdekIris.data` file.
`manifest.json` pins both hashes and every split assignment. `contracts.json`
freezes acceptance thresholds before learning results are collected. The split
hash algorithm does not depend on a language's random-number implementation.

Fit standardization only on the 90 training rows. The 30 validation rows select
candidates; the 30 test rows evaluate the frozen winner. Never select a seed or
relax a threshold using test results. The fixture demonstrates a bounded
lifecycle, not quality on arbitrary user data.

Deterministic numerical inputs/weights are in
`../../model_creation_audit_test.go`. The original six-row constant-feature leakage witness is archived in
`../../../../docs/audits/model-creation-evidence/r01-worker-witness.go.txt`;
its replacement rejection and validation-score regressions run in default CLI
tests. The numerical spike uses the explicit `r01audit` build tag. They require no data
network access. Numerical spike tolerances were set before their first run.
The leakage witness intentionally characterizes an existing defect and must
be replaced with desired-behavior regression coverage when R05 fixes it.
