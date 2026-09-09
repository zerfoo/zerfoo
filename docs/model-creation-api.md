# Numeric classifier and dataset APIs

R02/R03 provide inspected datasets and generic classifier construction,
prediction, and loss. R04 adds real CLI training; GGUF bundle save/load and
explicit legacy migration are implemented. AutoML now isolates validation and
retains the selected model when --output is supplied. Baseline-aware search,
complete recovery, and HTTP prediction remain in progress.
The API is alpha; deployment/learning qualification is not yet claimed.

```go
file, err := os.Open("measurements.csv")
if err != nil { return err }
defer file.Close()

dataset, err := tabular.InspectCSV(ctx, file, tabular.DatasetOptions{
    Target: "species",
    Split: "stratified",
    Seed: 42,
})
if err != nil { return err }
manifest := dataset.Manifest()
classifier, err := tabular.NewClassifier(tabular.ClassifierConfig{
    InputDim: len(manifest.Options.Features),
    ClassCount: len(manifest.Labels),
    Labels: manifest.Labels,
    HiddenDims: []int{16}, // omit for a linear classifier
    Seed: 42,
}, compute.NewCPUEngine(numeric.Float32Ops{}))
if err != nil { return err }
rows, _, err := dataset.Partition("validation")
if err != nil { return err }
predictions, err := classifier.PredictBatch(ctx, rows)
// This classifier is initialized, not trained. Predictions prove the API
// contract, not useful model quality. Training is R04.
```

`Classifier[T]` supports float32 and float64; all linear/activation/softmax
operations use the supplied compute engine and existing functional layers.
`Loss` validates integer targets and computes mean stable logit cross entropy.
The legacy `Model`, `Direction`, `Train`, `Save`, ensembles and adapters retain
their previous interfaces and three-class semantics.

Classifier configuration owns copies of labels and dimensions. Labels must be
unique, nonempty UTF-8; their supplied order defines class IDs. Dataset labels
are sorted lexically. Input dimensions, hidden dimensions and class count are
validated before allocation. Models and intermediate batches are bounded to
16,777,216 parameter/activation elements. Features and logits must be finite;
float32 input overflow is rejected. Prediction ties select the lowest class ID.
No dropout or unverified activation setting is exposed in these recipes.

CSV inspection requires a header with unique column names and a named target.
Numeric features use header order unless explicitly selected. Target/group/time
columns cannot be features. Missing, nonnumeric, NaN, infinite and
float32-overflowing values fail. Input is capped at 64 MiB. This implementation
supports three explicit split policies, with a nominal 60/20/20 allocation:

- `stratified`: order duplicate-record groups by seed-derived SHA-256, and place
  whole groups into per-class train/validation/test capacities. If these cannot
  fit or any split lacks a class, inspection fails.
- `group`: requires a group column; seeded group order partitions whole groups
  by group count. Row counts can differ from 60/20/20. Missing class coverage or
  identical records crossing groups/splits fails; the caller must choose a
  valid explicit assignment rather than retry until a desirable score appears.
- `temporal`: requires an RFC3339Nano time column. Sort chronologically, cut at
  60% and 80%, and require strict timestamp separation and class coverage.
  Equal timestamps at a boundary fail; use valid explicit assignments. Combined
  group/time policies are currently rejected.

Explicit `Assignments` maps train/validation/test to 1-based original row IDs.
All rows must appear exactly once. Duplicate records and groups cannot cross
splits. No omitted rows, borrowed test rows, or implicit fallback split is used.
Unknown labels that remove class coverage from a split fail inspection.

Standardization uses training-only means and population standard deviations;
constant features have scale 1. `Partition` returns copies of standardized rows.
`Standardization.Transform` applies saved moments to raw numeric features.
Neither method refits preprocessing on held-out inputs.

`Manifest` returns copies of source hash, schema, labels, assignments and
preprocessing. `WriteManifest` writes its JSON representation; the caller owns
file close/atomic publication. `ReplayCSV` takes a decoded manifest and validates
all of it against the source before returning a dataset. Changed source bytes,
labels, preprocessing or assignments require a new run. `Hash` identifies the
canonical manifest and includes the source byte hash.

The [reference fixture](../tabular/testdata/model_creation/README.md) pins
90/30/30 Iris rows. Version 2 corrected duplicate-record crossings before any
learning run; no quality threshold changed. See the
[baseline audit](audits/model-creation-correctness.md) for remaining release gates.

## Real training command

```sh
zerfoo train tabular --data iris.csv --target species --recipe mlp --output iris.bundle
```

Use `--dataset-manifest` with JSON produced by `Dataset.WriteManifest` to pin
existing split assignments and preprocessing. The training defaults are 200
epochs, batch size 15, learning rate .01, seed 42 and a two-minute time limit.
`--max-steps` sets an optional optimizer-step cap. JSONL events have version 1
and types progress/result/error; terminal states are succeeded, canceled,
budget_exhausted, or failed. Errors do not produce success-shaped results.
The old general GGUF/FSDP train command is explicitly unsupported. Its synthetic
loop exists only in test fixtures.

Each successful command saves GGUF F32 weights, manifest.json and bundle.sha256
in a new immutable directory. Load through `LoadClassifierBundle`, pinning the
returned bundle ID. `PredictRaw` applies the saved preprocessing. Bundle readers
validate format/version, identity, shapes, names, types and finite values.
`MigrateLegacyClassifier` accepts bounded ZTAB v1 ReLU models and an explicit
three-label map in Long/Short/Flat ID order; incompatible label order or GELU
migration is rejected instead of silently changing predictions.

For `automl --model tabular`, the final CSV column is the target. Supported
metrics are accuracy, macro_f1, loss and cross_entropy. The fixed split is shared
by all trials. Loss is negated only as an internal maximization utility; reported
loss remains positive. `--output best.json` additionally stores the exact winner
at best.json.bundle and records its ID. Complete baseline-aware search, all-trial
artifact persistence and frozen final-evaluation scheduling are still R05 work.

GPU validation is opt-in through Spark with `ZERFOO_R_MODEL_GPU=1`; unavailable
GPU dispatch fails that gate. CPU tests and source support are not a GPU verdict.
