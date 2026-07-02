# ADR 043: Arena Free-List with Tensor Lifetime Analysis

## Status
Accepted

## Date
2026-03-17

## Context

The inference arena allocator uses a bump-pointer strategy: O(1) allocation,
O(1) reset between decode steps. This is optimal for the common case but means
all intermediate tensors live until the next ResetPool call, even if they are
no longer needed mid-pass.

For a single decode step across a 26-layer transformer, intermediate tensors
accumulate in the arena throughout the full forward pass. With tensor lifetime
analysis, intermediates that are no longer referenced after their last use can
be freed and their memory reused within the same pass, reducing peak arena
pressure.

llama.cpp study confirmed that tensor lifetime analysis combined with intra-pass
reuse is a key strategy for memory efficiency in GPU inference.

## Decision

Add a free-list overlay to the arena allocator (internal/cuda/arena.go):
freed blocks are tracked in a best-fit free list with block splitting and
coalescing. The bump pointer is used when no suitable free block exists.

Add tensor lifetime analysis to graph/compile.go: during compilation, identify
the last use of each intermediate tensor in the instruction sequence. The
executor (cuda_graph.go) frees intermediates immediately after their last use,
returning memory to the arena free list for reuse within the same pass.

The bump-pointer fast path is preserved; the free list only activates when a
block is explicitly freed via the lifetime analysis path.

## Consequences

+ Peak arena pressure is reduced by reusing intermediate buffers within a pass.
+ Best-fit + coalescing ensures fragmentation stays bounded.
+ The bump-pointer fast path is unchanged for allocations without a suitable
  free block, preserving O(1) typical-case performance.
+ Architecture is more resilient to longer-context models where KV cache growth
  already consumes significant arena space.
- On GB10 unified memory, nvidia-smi does not report per-process GPU memory,
  making the reduction in arena pressure unmeasurable via standard tooling.
- Throughput impact at batch=1 is within measurement noise (memory-bandwidth
  bound workload does not benefit from reduced allocation pressure).
