# ADR 074: Satellite Libraries v1 Release Policy

## Status
Accepted

## Date
2026-03-29

## Context
The Zerfoo ecosystem comprises six active libraries in addition to the main framework:
float16 (v0.2.1), float8 (v0.3.1), ztensor (v0.14.1), ztoken (v0.3.4), zonnx (v0.9.0),
and zerfoo itself (v1.36.0, already at v1+). The main framework has an established v1 API
stability contract (ADR-058) but the satellite libraries have not made equivalent stability
commitments. Downstream users importing these libraries have no guarantee that the public API
will not change incompatibly between minor versions.

## Decision

### ztensor stability scope (narrowed)

ztensor is promoted to v1.0.0 but with a narrow stable surface. Every research epic in E34-E44
added new kernel primitives to ztensor, and future epics will do the same. Freezing the full
exported API would create a maintenance burden that slows research iteration.

Stable v1 surface for ztensor (full Go module compatibility guarantee):
- compute.Engine[T] interface and all its methods
- tensor.Tensor[T] type and its exported methods
- tensor.Numeric constraint
- device.Device interface
- numeric.* arithmetic functions

Not covered by the v1 stability guarantee (documented with "// This API is not covered by
the v1 stability guarantee." in doc comments):
- graph/ package (compilation pipeline still evolving)
- Any kernel-level or backend-level exported types outside the five stable packages above
- internal/ packages (already protected by Go convention)

zerfoo may import unstable ztensor symbols as long as it pins a specific ztensor version in
go.mod. The narrowed surface allows zerfoo to add new ztensor primitives in minor versions
without bumping ztensor to v2.

### General promotion criteria

Each satellite library will be promoted to v1.0.0 when it meets the following criteria:

1. **API completeness:** All planned public types and functions for the library's stated scope
   are implemented and tested.
2. **Documentation:** A design.md exists describing the architecture, and an ADR documents the
   API stability contract for that library.
3. **Test coverage:** All exported functions have at least one test. Coverage of core
   arithmetic/compute paths must be >= 95%.
4. **No known correctness bugs:** All reported correctness issues are resolved or explicitly
   scoped out of v1 with a documented rationale.
5. **Go version:** All libraries must target Go 1.26+.

Once a library ships v1.0.0, the standard Go module compatibility guarantee applies:
no breaking changes to exported symbols without a major version bump to v2.

Priority order for v1 promotion:
1. zonnx (v0.9.0 -> v1.0.0): All features shipped, closest to v1. Only needs API review and
   an ADR.
2. ztensor (v0.14.1 -> v1.0.0): Extensive test coverage (192 files). Needs design.md and a
   missing docs/QUALITY.md file referenced in CI.
3. ztoken (v0.3.4 -> v1.0.0): Simple scope. Needs expanded tests and design.md.
4. float8 (v0.3.1 -> v1.0.0): Functional core. Needs documentation and test depth.
5. float16 (v0.2.1 -> v1.0.0): Float16 is stable; BFloat16 requires Phases 2-5 completions
   before the overall library can be declared production-ready.

## Consequences

Positive:
- Downstream consumers get clear stability guarantees per library.
- Forces documentation and test gaps to be resolved before claiming production readiness.
- Ecosystem coherence: all active libraries under a unified v1+ commitment.

Negative:
- float16 v1 is blocked on BFloat16 completion (Phases 2-5), which may take weeks.
- ztensor has a large surface area; breaking-change avoidance adds maintenance burden.
