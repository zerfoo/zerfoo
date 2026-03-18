// Package generate implements autoregressive text generation for transformer
// models loaded by the inference package. It provides the core decode loop,
// KV caching, token sampling, streaming output, batch generation, and
// speculative decoding. (Stability: stable)
//
// # Generator
//
// [Generator] is the primary entry point. It takes a compiled computation
// graph, a tokenizer, a compute engine, and a [ModelConfig], then drives
// the prefill-and-decode loop:
//
//	gen := generate.NewGenerator[float32](graph, tok, engine, cfg)
//	text, err := gen.Generate(ctx, "Once upon a time", generate.DefaultSamplingConfig())
//
// The Generator compiles the graph into an [graph.ExecutionPlan] after the
// first decode step, optionally capturing a CUDA graph for near-zero kernel
// launch overhead. A megakernel code generator may further fuse the plan's
// instructions into a single GPU kernel.
//
// Generator options include [WithPagedKV] (block-allocated KV cache) and
// [WithGeneratorKVDtype] (FP16 KV cache storage for reduced memory bandwidth).
//
// # Sampling
//
// [SamplingConfig] controls token selection: temperature scaling, top-K
// filtering, nucleus (top-P) sampling, repetition penalty, stop tokens,
// stop strings, and grammar-constrained decoding. [DefaultSamplingConfig]
// returns sensible defaults (temperature 1.0, no filtering, 256 max tokens).
// When Temperature is zero, greedy argmax is used with an optimized GPU
// fast path that copies only 4 bytes instead of the full vocabulary logits.
//
// # KV Cache Variants
//
// The package provides three KV cache implementations behind the
// [CacheProvider] interface:
//
//   - [KVCache] pre-allocates flat CPU buffers sized to maxSeqLen on first
//     use. Zero-copy views are returned for batch=1. Suitable for simple
//     CPU inference.
//
//   - [TensorCache] is the default for GPU-accelerated inference. It
//     pre-allocates GPU-resident buffers and uses direct D2D memcpy for
//     KV appends. It supports FP16 storage mode (halving memory bandwidth),
//     GPU-resident position counters for CUDA graph capture compatibility,
//     and the [FullBufferProvider] interface for flash attention decode.
//
//   - [PagedKVCache] allocates fixed-size blocks on demand from a shared
//     [BlockPool], reducing memory waste for concurrent sequences of varying
//     length. Blocks are shared across layers and recycled via Alloc/Free.
//
//   - [GPUKVCache] manages raw GPU device pointers for megakernel inference.
//     It uses offset_memcpy and increment_counter CUDA kernels to write KV
//     data at GPU-counter-derived offsets, making the entire append path
//     capturable in CUDA graphs.
//
// Caches are attached to a context via [WithCache] and retrieved by attention
// layers via [GetCache].
//
// # Streaming
//
// [GenerateStream] delivers tokens incrementally through a [TokenStream]
// callback as they are decoded. [TokenStreamFunc] adapts a plain function
// to the interface. Stop strings are checked incrementally and any text
// preceding a match is emitted before the done signal.
//
// # Batch Generation
//
// [Generator.BatchGenerate] and [Generator.BatchGenerateStream] accept
// multiple prompts and run them sequentially (request-level parallelism).
// True batched tensor operations (batch dimension > 1) require native
// batch support in the model graph, which is planned but not yet implemented.
//
// # Speculative Decoding
//
// [SpeculativeGenerator] pairs a small draft model with a large target model.
// The draft proposes N tokens greedily, the target verifies all N in a single
// batched forward pass, and accepted tokens are emitted. On mismatch the
// target's token is used. An adaptive draft length tracker adjusts N based
// on rolling acceptance rate (increasing when acceptance > 80%, decreasing
// when < 40%).
//
// # Constrained Decoding
//
// When [SamplingConfig.GrammarState] is set, a token mask is computed from
// the grammar at each step, restricting sampling to tokens that produce
// valid continuations. The grammar state advances through the bytes of each
// sampled token. Generation stops early when the grammar reaches a complete
// state.
//
// # Tracing
//
// [TracingCacheProvider] wraps a real CacheProvider and records KV cache
// operations into a compute.Tracer during compilation tracing passes,
// capturing the full attention dataflow including cache reads and writes.
// Stability: stable
package generate
