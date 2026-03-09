# ADR 021: Graph Compilation and Persistent Worker Pool

## Status

Accepted

## Date

2026-03-06

## Context

Phase 29 profiling on DGX Spark GB10 (Gemma 3 2B Q4_0) revealed that compute
kernels (SGEMM + Q4 GEMV) take only ~24ms/token wall time, but total per-token
latency is ~154ms. The ~130ms gap is framework overhead:

1. **Graph traversal:** ~780 node executions per token (30+ nodes x 26 layers).
   Each node: interface method dispatch, shape validation, pool acquire/release.
2. **Goroutine scheduling:** ~130 MatMul calls per token, each spawning and
   joining ~20 goroutines (runtime.NumCPU). That is ~2600 goroutine create/join
   cycles per token.
3. **Memory management:** TensorPool reduces heap allocations but the pool
   itself has mutex contention, and Go GC still scans pooled memory.

To reach 15 tok/s (67ms/token), overhead must drop from ~130ms to ~43ms (3x
reduction). Incremental kernel optimization cannot achieve this.

## Decision

Introduce two architectural changes:

### 1. Graph Compiler

Pre-compile the computation graph into a flat instruction sequence before the
autoregressive decode loop. The compiled representation eliminates:

- Per-node interface dispatch (replaced by direct function pointers)
- Per-node shape validation (shapes are fixed after the first forward pass)
- Per-node pool acquire/release (pre-allocate all intermediate buffers)

The compiler runs once after the first forward pass (prefill), when all tensor
shapes are known. It produces an `ExecutionPlan` containing:

- A flat slice of `Instruction` structs (opcode + pre-resolved buffer indices)
- A pre-allocated buffer arena (all intermediate tensors)
- Direct function pointers to kernel implementations

During decode, `ExecutionPlan.Run()` iterates the flat instruction slice with
no interface dispatch, no shape checks, and no pool operations.

### 2. Persistent Worker Pool

Replace per-MatMul goroutine creation with a fixed-size worker pool initialized
at engine creation time. Workers block on a channel waiting for work items.
MatMul and GEMV operations submit work to the pool and wait for completion
via a sync primitive (WaitGroup or similar).

This eliminates ~2600 goroutine create/join cycles per token, replacing them
with channel send/receive operations on pre-existing goroutines.

## Consequences

### Positive

- Expected 3-5x reduction in per-token overhead (130ms -> 26-43ms)
- Worker pool eliminates goroutine scheduling overhead entirely
- Graph compilation enables future optimizations (instruction fusion, buffer
  aliasing, memory planning)
- Pre-allocated buffer arena eliminates remaining GC pressure

### Negative

- Graph compiler adds complexity: two execution paths (interpreted for prefill,
  compiled for decode)
- Compiled plan must be invalidated if input shape changes (e.g., batch size)
- Worker pool requires careful sizing and shutdown coordination
- Debugging is harder with compiled execution (no per-node error context)
- Increased binary size from the compiler infrastructure

### Mitigations

- Keep the interpreted path as fallback; compiled path is opt-in via
  `graph.Compile()` after first forward pass
- Worker pool size defaults to runtime.NumCPU() and is configurable
- Add instruction-level tracing behind a debug flag for troubleshooting
