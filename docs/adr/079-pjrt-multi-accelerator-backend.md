# ADR 079: PJRT Multi-Accelerator Backend via purego

## Status

Accepted

## Date

2026-04-01

## Context

Zerfoo currently supports NVIDIA CUDA (custom kernels via purego), AMD ROCm
(HIP via purego), and OpenCL (CLBlast via purego) through the GPU Runtime
Abstraction Layer (GRAL) in ztensor's `internal/gpuapi/`. Adding support for
new accelerators (AWS Trainium/Inferentia, Google TPU, Apple Metal, Intel GPU)
requires writing vendor-specific GRAL adapters for each -- a significant
engineering investment per backend.

PJRT (Pretty much Just another RunTime) is the OpenXLA project's hardware
abstraction layer. A PJRT plugin is a single `.so` file that exports one C
symbol (`GetPjrtApi`) returning a struct of ~100 function pointers. Each
hardware vendor ships their own plugin: CUDA (`pjrt-plugin-cuda.so`), TPU
(`libtpu.so`), Neuron/Trainium (`libneuronpjrt.so`), ROCm
(`pjrt-plugin-rocm.so`), Metal (`jax-metal` plugin). The C API is pre-1.0
(v0.103) but practically stable for core operations, with a `struct_size`
field on every argument struct enabling forward/backward compatibility.

Two integration paths were evaluated:

1. **Direct vendor runtime (e.g., libnrt.so for Neuron):** Lower-level,
   vendor-specific, requires per-backend implementation of GRAL's 140+
   `KernelRunner` methods. Only benefits one hardware target.

2. **PJRT plugin:** Higher-level, graph-oriented. One integration covers all
   PJRT-supported accelerators. Trades per-kernel control for breadth.

GoMLX (`github.com/gomlx/go-xla`) demonstrates the PJRT-from-Go approach
works but uses CGo for dlopen. ZML (`github.com/zml/zml`) provides a Zig
reference implementation.

## Decision

Add PJRT as a new compilation backend in ztensor, loaded via purego (zero CGo).
The PJRT path consumes `CompileTraced()` output (flat `TracedOp[]` sequences),
translates them to StableHLO MLIR text, and hands the program to
`PJRT_Client_Compile()` for JIT compilation on the target accelerator.

PJRT is a **parallel path**, not a replacement for existing backends. Custom
CUDA/ROCm/OpenCL kernels remain the performance-optimal path for hardware where
Zerfoo has hand-tuned kernels. PJRT is the "reach play" for hardware without
custom kernels.

Key design choices:

- **StableHLO text generation (not HLO protobuf):** StableHLO is a text-based
  MLIR dialect. Generating MLIR text strings in Go is simpler than importing
  protobuf definitions and requires zero external dependencies. GoMLX proves
  this approach works with pure Go StableHLO generation including shape
  inference.

- **CompileTraced() as the entry point:** Zerfoo's existing `CompileTraced()`
  decomposes composite nodes (RMSNorm, GQA, etc.) into primitive `Engine[T]`
  operations (Add, MatMul, Softmax) via EngineProxy tracing. These primitives
  map 1:1 to StableHLO ops, making the translation straightforward.

- **purego, not CGo:** The PJRT C API is flat C functions with opaque handles.
  This is the same pattern used for CUDA bindings in
  `ztensor/internal/cuda/purego.go`. No C++ types, templates, or exceptions.

## Consequences

### Positive

- One integration covers AWS Trainium, Google TPU, Apple Metal, Intel GPU,
  and AMD ROCm -- any hardware with a PJRT plugin.
- Zero CGo maintained. `go build ./...` continues to work everywhere.
- Backward compatible: no changes to Engine[T] interface or existing backends.
- Opens partnership opportunities with hardware vendors (Annapurna Labs/AWS).
- CompileTraced() already exists -- no new graph compilation infrastructure.

### Negative

- PJRT plugins include the XLA compiler, which is a large binary (~200MB+).
  This is a runtime dependency, not a build dependency.
- PJRT operates at graph granularity, not kernel granularity. Fused operations
  (megakernels, paged attention) are not available through this path -- the
  XLA compiler must discover its own fusion opportunities.
- The PJRT C API is pre-1.0 (v0.103). The `PJRT_Api` struct layout must be
  kept in sync with the header. The `struct_size` compatibility mechanism
  mitigates breaking changes.
- StableHLO text generation requires implementing shape inference in Go.
  GoMLX has a reference implementation that can guide this work.
- Performance will not match custom CUDA kernels on NVIDIA GPUs. PJRT is
  optimized for correctness and breadth, not per-kernel performance.
