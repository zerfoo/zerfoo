# ADR 062: Tabular Model Package

## Status
Accepted

## Date
2026-03-18

## Context
Zerfoo's training infrastructure (Trainer[T], AdamW, CrossEntropy, GPU kernels)
is designed for transformer/LLM workloads. An internal consumer uses LLM inference
as a workaround for tabular prediction -- formatting numeric feature vectors as text
prompts and parsing classification labels from generated text. This is orders of
magnitude slower than a direct tabular model and wastes GPU compute.

The internal consumer also hand-rolled pure Go implementations of CNN and TabNet
rather than using Zerfoo's Trainer[T], because no high-level tabular model API
exists. These hand-rolled models are CPU-only and cannot benefit from Zerfoo's
GPU acceleration.

Zerfoo's metee library provides LightGBM/XGBoost gradient boosted tree bindings,
but there is no way to combine tree ensemble outputs with neural network predictions
in a single pipeline.

## Decision
Create a `tabular` package in zerfoo that provides:

1. **tabular.Model** -- Configurable MLP built on ztensor compute graph with
   `Predict([]float64) -> (direction, confidence)` API.
2. **tabular.Train** -- Wire existing Trainer[T] + AdamW + CrossEntropy to train
   tabular models from `(data [][]float64, labels []int)` input.
3. **tabular.Save / tabular.Load** -- Binary serialization of model weights and
   configuration for deployment.
4. **tabular.Ensemble** -- Combine metee tree ensemble outputs with MLP outputs
   via stacking (meta-learner).

Follow-on phases will add advanced architectures (FTTransformer, TabNet, SAINT,
ResNet) and time-series architectures (TFT, N-BEATS, PatchTST) as additional
model types within the same package interface.

Design constraints:
- All tensor operations through Engine[T] for CPU/GPU transparency.
- No CGo; GPU via existing purego path.
- Model serialization must be self-contained (weights + config in one file).
- Ensemble must work with metee v1.0.1 tree models without changes to metee.

## Consequences

**Positive:**
- Internal consumer replaces LLM-as-tabular-classifier with native tabular
  inference (1000x+ speedup for numeric feature classification).
- Internal consumer replaces hand-rolled CPU models with GPU-accelerated training.
- Opens Zerfoo to tabular ML use cases beyond LLM inference.
- Reuses existing Trainer[T], optimizer, and loss function infrastructure.

**Negative:**
- Expands Zerfoo's scope from "LLM inference framework" to "ML framework."
- tabular package becomes a new surface area to maintain and document.
- Ensemble coupling with metee creates cross-repo dependency for testing.
