# ADR 058: API Stability v1.0 Contract

## Status
Accepted

## Date
2026-03-18

## Context
Zerfoo needs a v1.0 release with a clear backwards-compatibility guarantee to drive
enterprise adoption. Go ecosystem conventions (stdlib compatibility promise, import
path versioning) provide established patterns. The compute.Engine[T] interface is the
most critical API surface -- adding methods breaks all third-party implementations.

Alternatives considered:
1. Freeze all exported APIs indefinitely (Go stdlib model) -- too restrictive.
2. Semantic versioning with no guarantee window -- too weak for enterprise trust.
3. 2-year guarantee window with extension interfaces -- balanced approach.

## Decision
Zerfoo v1.0 will guarantee backwards compatibility for 2 years (through v1.x):

1. compute.Engine[T] interface is frozen. New capabilities added via optional
   extension interfaces checked with type assertions (e.g., EngineWithFP8,
   EngineWithPagedKV). This follows the http.Hijacker, io.ReaderFrom pattern.

2. Sub-packages are labeled by maturity:
   - Stable: inference/, generate/, serve/, model/, layers/ -- full v1 guarantee.
   - Beta: training/, distributed/ -- schema preserved, behavior may change.
   - Alpha: training/nas/, training/automl/ -- may be restructured.

3. Deprecation protocol: // Deprecated: doc comment, 2 minor releases of
   coexistence, removal only in v2.0.

4. Import path stays at github.com/zerfoo/zerfoo (implicit v1). v2 uses
   github.com/zerfoo/zerfoo/v2.

## Consequences
Positive:
- Enterprise teams can depend on Zerfoo v1.0 with confidence.
- Extension interface pattern allows capability growth without breaking changes.
- Maturity labels set correct expectations for less stable packages.

Negative:
- Engine[T] freeze may require creative workarounds for new GPU features.
- 2-year window means living with early design mistakes until v2.0.
