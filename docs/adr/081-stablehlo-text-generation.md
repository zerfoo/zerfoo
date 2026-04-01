# ADR 081: StableHLO Text Generation over HLO Protobuf

## Status

Accepted

## Date

2026-04-01

## Context

To compile computation graphs for PJRT-supported accelerators, Zerfoo must
produce programs in a format that `PJRT_Client_Compile()` accepts. Two
formats are supported:

1. **HLO protobuf:** The XLA HLO intermediate representation serialized as
   a Protocol Buffer. This is the native XLA format. Using it from Go
   requires either importing the protobuf definitions (adding a proto
   dependency) or manually serializing the binary protobuf format.

2. **StableHLO MLIR text:** StableHLO is a text-based MLIR dialect that is
   the standardized input format for PJRT. Programs are plain UTF-8 strings
   containing MLIR operations like `stablehlo.add`, `stablehlo.dot_general`,
   `stablehlo.reduce`. The PJRT compiler parses this text and compiles it.

GoMLX (`github.com/gomlx/go-xla`) uses StableHLO text generation in pure Go.
Their `stablehlo` package constructs MLIR text programmatically with shape
inference, passes it to `PJRT_Client_Compile` as a string, and the PJRT
plugin JIT-compiles it. No external dependencies are needed for the text
generation itself.

## Decision

Generate StableHLO MLIR text strings in pure Go. The emitter package
(`ztensor/internal/stablehlo/`) walks the `TracedOp[]` sequence from
`CompileTraced()` and produces MLIR text with SSA-style value naming.

Rationale:

- **Zero dependencies:** No protobuf compiler, no proto imports, no generated
  code. Text is built with `strings.Builder` and `fmt.Sprintf`.
- **Debuggability:** StableHLO text is human-readable. Engineers can inspect
  the generated program, diff it, and verify correctness visually.
- **Proven approach:** GoMLX demonstrates that pure Go StableHLO text
  generation works end-to-end with PJRT compilation on CPU, CUDA, and TPU.
- **Maintainability:** StableHLO is a stable specification maintained by
  the OpenXLA project. It evolves slowly and backwards-compatibly.

The emitter must implement shape inference for each supported operation to
produce correctly typed MLIR. The GoMLX stablehlo package provides a
reference implementation of shape inference rules in Go.

## Consequences

### Positive

- Zero external dependencies. Pure Go string manipulation.
- Human-readable output for debugging and testing.
- No protobuf toolchain needed in the build.
- Test cases can assert on exact MLIR text output.

### Negative

- Shape inference must be implemented in Go for every supported Engine[T]
  operation (~50 ops). This is mechanical but non-trivial work.
- Text parsing by the PJRT compiler is slightly slower than binary protobuf
  deserialization. This cost is negligible compared to compilation time.
- If StableHLO adds new required syntax or deprecates existing ops, the
  emitter must be updated. The risk is low given OpenXLA's stability policy.
