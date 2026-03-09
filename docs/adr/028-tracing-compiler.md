# ADR 028: Tracing Compiler for Automatic Primitive Op Decomposition

## Status
Accepted

## Date
2026-03-07

## Context
The megakernel code generator (ADR 026) assumes the compiled ExecutionPlan
instruction tape contains primitive Engine ops (Add, MatMul, Softmax, etc.).
However, graph.Compile() records each graph Node as a single instruction using
node.OpType(). For the Gemma 3 model built by inference/arch_common.go, the
graph contains composite nodes:

| Node Type | OpType() | In Emitter Table? |
|-----------|----------|-------------------|
| embeddingLookupNode | "EmbeddingLookup" | No |
| normalization.RMSNorm | "RMSNorm" | Yes |
| attention.GroupedQueryAttention | "GroupedQueryAttention" | No |
| core.Add | "Add" | Yes |
| core.FFN | "FFN" | No |
| lmHeadNode | "LMHead" | No |

4 of 6 op types are unsupported. CheckSupport() returns them all, and
tryCompileMegakernel silently falls back to the per-instruction path. The
megakernel never fires.

Even though composite nodes internally use Engine methods (MatMul, Reshape,
Softmax, etc.), the compiler only sees the top-level node. The internal Engine
calls are invisible to the instruction tape.

Three approaches were evaluated:

A) Graph-level decomposition: Rewrite arch_common.go to build graphs with
   primitive nodes instead of composite ones. Requires every model builder to
   think in primitives. Makes composite layer types (GQA, FFN) dead code.
   Invasive to model construction.

B) Extended InstructionMeta + composite emitters: Add attributes and weight
   slot references to InstructionMeta. Write composite CUDA emitters for GQA,
   FFN, etc. Makes the megakernel model-specific. Weights are internal to
   composite nodes and not exposed as graph-level slots.

C) Tracing compiler: Create an EngineProxy that records each Engine method call
   during compilation. Forward() calls proceed normally through all composite
   layers, but the proxy captures every primitive Engine call (MatMul, Softmax,
   etc.) as a separate instruction. The traced primitive ops become the
   instruction tape.

## Decision
Implement approach C: the tracing compiler.

Key components:

1. **EngineProxy[T]**: Wraps Engine[T]. In normal mode, delegates directly
   to the underlying engine. In tracing mode, delegates to the engine AND
   records each call as a TracedOp (op name, input tensor IDs, output tensor
   ID, output shape).

2. **Tracer[T]**: Records TracedOps. Tracks tensor identity via pointer to
   assign slot indices. Frozen tensors (model weights) are identified by their
   presence in the graph's parameter set. New tensors from engine calls get
   new slot indices.

3. **TracingCacheProvider[T]**: Wraps the real KV CacheProvider. Records
   cache.Update() and cache.Get() as special instructions ("KVCacheUpdate",
   "KVCacheGet") so the megakernel can manage KV data on GPU.

4. **CompileTraced()**: New graph.Compile variant. Activates tracing on the
   EngineProxy, runs Forward() on each node in topological order, deactivates
   tracing, and builds the ExecutionPlan from the traced primitive ops instead
   of from graph-level nodes.

5. **GPU KV Cache**: Persistent GPU memory for KV data that survives between
   megakernel launches. The megakernel reads/writes KV cache via device
   pointers, eliminating Go-managed cache for the decode path.

The EngineProxy must be passed at graph construction time (instead of the
real engine) so all layers internally call through the proxy. The Graph stores
a reference to the proxy and calls proxy.StartTracing()/proxy.StopTracing()
during CompileTraced().

## Consequences
Positive:
- No changes to model construction code (arch_common.go unchanged).
- No changes to layer implementations (GQA, FFN, RMSNorm all unchanged).
- Composite layers remain reusable, testable, high-level abstractions.
- Decomposition happens automatically for any model, any architecture.
- Single point of truth for decomposition (the compiler, not scattered across
  model builders).
- Proven pattern (JAX tracing, PyTorch FX, TensorFlow tf.function).
- The op table in codegen/optable.go covers all primitive ops regardless of
  which composite layers use them.

Negative:
- EngineProxy adds one interface dispatch per Engine call (negligible cost).
- Tracing must handle edge cases: UnaryOp with opaque closures falls back to
  non-traceable; control flow inside Forward() (e.g., KV cache existence
  check) means the trace is valid only for the traced execution path (decode
  mode with seqLen=1).
- GPU KV cache management is additional infrastructure.
- EngineProxy must implement the full Engine interface (~25 methods).
- The traced instruction count is much larger (600+ for Gemma 3 vs 160 with
  composite nodes), though this only affects the code generator, not runtime.
