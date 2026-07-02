# ADR 084: Extract crossasset package from zerfoo to wolf

## Status
Accepted

## Date
2026-04-12

## Context
The `crossasset/` package in zerfoo implements a domain-specific multi-source
attention model for financial market direction prediction (long/short/flat).
It is not a generic ML building block -- it is an application-level model that
belongs in the system that uses it (Wolf, the autonomous trading system at
feza-ai/wolf).

Zerfoo's mission is to be a generic, embeddable ML framework. Shipping a
proprietary trading model inside the public framework violates that boundary.
Wolf already imports `github.com/zerfoo/zerfoo/crossasset` in two places:
`cmd/train-crossasset/main.go` and `internal/model/crossasset.go`.

Additionally, zerfoo's `timeseries/crossasset_engine.go` (274 lines) provides
a timeseries-training adapter that wraps `crossasset.Model`. This adapter also
belongs in wolf since it exists solely to bridge crossasset into Wolf's
training pipeline.

## Decision
Move the entire `crossasset/` package (2,672 lines across 10 files) and the
`timeseries/crossasset_engine.go` adapter (274 lines) from zerfoo to
feza-ai/wolf. Wolf will own the crossasset model as a first-party package.
Zerfoo will have no knowledge of crossasset after the extraction.

This is a breaking change for any consumer of `github.com/zerfoo/zerfoo/crossasset`.
Wolf is the only known consumer.

## Consequences
**Positive:**
- Zerfoo stays focused on generic ML infrastructure.
- CrossAsset model code is co-located with the system that uses it.
- Removes ~3,000 lines of domain-specific code from the public framework.
- Wolf gains full ownership of its model architecture.

**Negative:**
- Breaking change requires a major version bump (v3.0.0) in zerfoo.
- Wolf must update imports and may need to adjust its go.mod replace directives.
- GPU parity tests for crossasset move to wolf, requiring wolf CI to have
  GPU test infrastructure (or skip on CPU-only CI like zerfoo does).
