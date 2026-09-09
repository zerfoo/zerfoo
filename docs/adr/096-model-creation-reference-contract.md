# ADR 096: Bounded tabular model-creation lifecycle

Status: Proposed implementation contract (R01). No new package or public API
has been introduced by this ADR.

Date: 2026-09-08

## Context

The [R01 audit](../audits/model-creation-correctness.md) finds working legacy
three-class training and tensor persistence primitives, but no complete generic
CSV-to-deployment contract. AutoML evaluates training rows and discards the
winning model; the general train CLI runs a synthetic FSDP example. Exposing
these paths to agents now would misrepresent their results.

## Decision

Implement Phase R in existing `tabular`, `cmd/cli`, `training/automl`,
`training/optimizer`, and `serve` packages. No top-level package is needed.
Keep legacy Model/Direction/Train/Save/Load behavior behind explicit adapters;
new classifier APIs accept context, explicit class labels, and a recipe.
Legacy ensembles and adapters retain their three-class contract until
separately generalized. Linear means zero hidden layers; the first MLP is
ReLU with one 16-unit hidden layer. Reject nonzero dropout and unsupported
activations in the certified recipes until their training behavior is proven.

A dataset manifest pins SHA-256 content, ordered numeric features, target,
lexically ordered UTF-8 labels, 1-based row IDs, split assignments, and
training-only mean/population-standard-deviation preprocessing. Reject missing,
NaN, infinite, or float32-overflowing values and ambiguous/duplicate columns.
Explicit group splits keep groups whole; time splits train strictly before
validation and test. Unsupported combinations fail; no silent random fallback.
The fixed real-data fixture keeps identical records in the same split and uses the assignments in
`tabular/testdata/model_creation/manifest.json`.

Training requests name the dataset/split hash, recipe, seed, device, limits,
and output location. Use per-run restorable RNG state for initialization,
splits and batch order. Check context at batch boundaries and propagate
cancellation as cancellation. No synthetic production training fallback.
Checkpoints atomically persist weights, named optimizer moments/timestep,
RNG state, batch cursor, preprocessing, dataset/config/runtime identity.
A weights-only file is not a resumable checkpoint. Reject mismatches before
changing live state. CPU exact resume is within the same pinned runtime;
GPU numeric bounds do not excuse incorrect state identities.

Metrics are explicit: maximize accuracy/macro-F1, minimize mean cross entropy.
Rows of the confusion matrix are true class; columns are predicted class.
Macro-F1 averages all configured classes, zero denominator contributes zero.
Cross entropy is the mean of `-log(max(p_true,1e-7))`. Reject unknown metrics.
Validation chooses candidates; test evaluates the frozen winner once. Persist
the exact winning model and link its content hash to the report. A majority
baseline breaks class-count ties by lowest persisted class ID. A no-improvement
result is allowed and cannot receive a qualification flag.

Deployment uses GGUF v3 F32 weights plus a versioned JSON manifest, not a new
weight format. Metadata identifies `zerfoo.tabular.mlp.v1` or the linear recipe;
use stable names `layerN.weights`, `layerN.biases`, `head.weights`, `head.biases`.
Runtime tensors use [input, output] weights and [1, output] biases. The shared
GGUF writer reverses dimensions on disk, and LoadTensors reverses them back.
The R01 spike proves this boundary, including a nonsquare output tensor.

R07 adds a public tabular-specific loader around these primitives; the text
inference loader is not the classifier executor. Validate architecture,
version, shapes, tensor names/types, file hashes, feature/label counts, and
finite preprocessing before constructing a model. Write bundles by staging and
atomic activation; retain ZTAB import fixtures and migration. Prediction serves
raw numeric records through `/v1/tabular/predict`, applying the saved feature
order and preprocessing and returning class IDs, labels, and probabilities.
Text `/v1/classify` keeps its existing contract.

## Evidence and consequences

The audit spike reconstitutes a classifier manually inside test code; it is
not a released loader, fresh-process deployment test, or GPU qualification.
The format is feasible, but R07 still needs manifest validation and migration.

[Numeric and lifecycle thresholds](../../tabular/testdata/model_creation/contracts.json)
are frozen for the reference fixture. Small fp32 forward tolerances allow a
few rounding units; gradient tolerances also accommodate finite differences.
GPU bounds follow the existing combined tolerance rule and remain a required
unverified gate. Failure prompts diagnosis and an explicit contract revision,
never retrospective threshold tuning. Independent PyTorch/autodiff evidence,
engine lifetime stress, and GPU convergence are still required before release.

This scope postpones agent orchestration, arbitrary architectures, mixed
precision, distributed training and new backends. It preserves embeddability
and the GGUF-only deployment direction while providing a testable reference
path for the subsequent agent product.
