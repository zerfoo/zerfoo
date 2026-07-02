# ADR 080: KV Cache as Explicit I/O for PJRT Execution Path

## Status

Accepted

## Date

2026-04-01

## Context

Zerfoo's KV cache is stateful. During autoregressive decoding, the KV cache
mutates in place between decode steps via two mechanisms:

1. **StatefulInputNode / AddKVPair:** The `Graph[T]` struct maintains a
   `kvPairs` list linking stateful input nodes to output nodes. After each
   `Forward()`, output tensors are fed back into the corresponding input nodes
   via `SetStored()` (graph.go lines 253-258).

2. **KVCache[T] struct:** The `generate/kvcache.go` `KVCache[T]` holds
   pre-allocated buffers per layer with a cursor that advances on each `Update`
   call. Variants (Q4, Q3, FP8, FP16, compressed, paged, tiered) all follow
   the same pattern of in-place mutation.

PJRT executables are pure functions: they take input buffers and produce
output buffers with no side effects. A compiled StableHLO program cannot
reference or mutate external state between invocations. This creates a
fundamental mismatch with Zerfoo's stateful KV cache design.

Three approaches were evaluated:

**Option A: KV cache as explicit I/O (pure functional).** Pass KV cache
tensors as additional inputs to the compiled program. The program returns
updated KV cache tensors as additional outputs alongside logits. Each decode
step: `execute(token, kv_in) -> (logits, kv_out)`. PJRT's buffer donation
mechanism allows the runtime to reuse input buffer memory for outputs,
avoiding actual copies on hardware that supports it (TPU, Trainium).

**Option B: Compile separate prefill and decode programs.** Prefill program
takes all tokens, produces initial KV cache. Decode program takes one token
plus existing KV cache, produces updated KV cache. Two compilations, but
each is a pure function.

**Option C: Host-side KV cache with per-step device transfer.** Keep KV cache
on host (CPU), transfer relevant portions to device before each step, transfer
updated portions back. Compatible with existing KVCache[T] design but adds
transfer latency.

## Decision

Use Option A (KV cache as explicit I/O) with PJRT buffer donation. This is
the same approach used by JAX models on TPU and Trainium -- it is proven at
scale and aligns with the PJRT execution model.

Implementation details:

1. **PJRTPlan wrapper:** A new `PJRTPlan[T]` type in `graph/` wraps a
   `PJRT_LoadedExecutable` and manages the buffer lifecycle. It maintains
   device-resident PJRT buffers for the KV cache between decode steps.

2. **KV state rewriting in CompilePJRT():** The `CompilePJRT()` method
   analyzes the traced ops to identify KV cache update patterns (nodes that
   implement `StatefulInputNode[T]`). It rewrites them as explicit function
   parameters and return values in the emitted StableHLO program.

3. **Buffer donation:** When executing, KV cache input buffers are marked
   with `PJRT_Buffer_MemoryLayout` donation flags. The PJRT runtime reuses
   the device memory for the output KV buffers, making the "copy" effectively
   free on hardware that supports donation (TPU, Trainium, CUDA).

4. **Existing paths untouched:** The `Graph.Forward()` method and all
   existing `KVCache[T]` implementations are unchanged. The PJRT path is
   entirely contained in `CompilePJRT()` and `PJRTPlan[T]`.

5. **Prefill vs decode:** `CompilePJRT()` compiles two StableHLO programs:
   one for prefill (variable-length input, produces initial KV cache) and
   one for decode (single token + KV cache, produces updated KV cache).
   The `PJRTPlan[T]` switches between them automatically.

## Consequences

### Positive

- Pure functional execution model matches PJRT's design exactly.
- Buffer donation avoids actual memory copies for KV cache between steps.
- Two-program design (prefill + decode) allows the PJRT compiler to optimize
  each program independently with appropriate shapes.
- Existing KV cache implementations (Q4, Q3, paged, tiered, compressed) are
  completely unaffected.
- Matches the proven JAX/TPU pattern used at scale by Google.

### Negative

- KV cache shape must be known at compile time. Dynamic maximum sequence
  length requires recompilation or padding to a fixed maximum.
- Two compiled programs (prefill + decode) means two compilation cycles.
  First-run latency increases but is amortized over the generation session.
- Buffer donation is hardware-dependent. On hardware without donation support,
  KV cache copies happen every step, adding latency proportional to cache
  size.
- The KV state rewriting logic in `CompilePJRT()` must correctly identify
  all stateful nodes in the graph. Incorrect identification would produce
  wrong results silently.
