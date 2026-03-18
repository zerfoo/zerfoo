# Package generate

```
import "github.com/zerfoo/zerfoo/generate"
```

Package generate implements autoregressive text generation for transformer models loaded by the [inference](inference.md) package. It provides the core decode loop, KV caching, token sampling, streaming output, batch generation, and speculative decoding.

Most users should use the high-level [inference.Model](inference.md#type-model) API. Use this package directly when you need fine-grained control over the generation loop, KV cache management, or custom sampling strategies.

## Quick Start

```go
gen := generate.NewGenerator[float32](graph, tok, engine, cfg)
text, err := gen.Generate(ctx, "Once upon a time", generate.DefaultSamplingConfig())
```

---

## Generator

### type Generator

```go
type Generator[T tensor.Numeric] struct {
    // unexported fields
}
```

Generator produces text autoregressively using a loaded model graph. It takes a compiled computation graph, a tokenizer, a compute engine, and a ModelConfig, then drives the prefill-and-decode loop.

The Generator compiles the graph into a `graph.ExecutionPlan` after the first decode step, optionally capturing a CUDA graph for near-zero kernel launch overhead. A megakernel code generator may further fuse the plan's instructions into a single GPU kernel.

### func NewGenerator

```go
func NewGenerator[T tensor.Numeric](
    g *graph.Graph[T],
    tok tokenizer.Tokenizer,
    eng compute.Engine[T],
    cfg ModelConfig,
    opts ...GeneratorOption,
) *Generator[T]
```

Creates a Generator from a model graph, tokenizer, engine, and config.

```go
cfg := generate.ModelConfig{
    VocabSize:  32000,
    MaxSeqLen:  4096,
    EOSTokenID: 2,
    BOSTokenID: 1,
    NumLayers:  32,
}
gen := generate.NewGenerator[float32](graph, tok, engine, cfg)
```

### func (*Generator) Generate

```go
func (gen *Generator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error)
```

Produces text from a prompt using the given sampling configuration. It tokenizes the prompt, runs the autoregressive loop with KV caching, and returns the generated text (excluding the prompt).

When `WithSpeculativeDraft` is configured, Generate uses speculative decoding: the draft model proposes K tokens, the target model verifies them in one forward pass. If the rolling acceptance rate drops below 0.4, generation falls back to standard autoregressive decoding.

```go
sc := generate.DefaultSamplingConfig()
sc.Temperature = 0.7
sc.MaxNewTokens = 256

text, err := gen.Generate(ctx, "The meaning of life is", sc)
```

### func (*Generator) GenerateStream

```go
func (gen *Generator[T]) GenerateStream(ctx context.Context, prompt string, sc SamplingConfig, stream TokenStream) error
```

Produces text from a prompt, delivering each token to the stream as it is generated. The final output matches what `Generate` would return.

```go
err := gen.GenerateStream(ctx, "Tell me about Go.",
    generate.DefaultSamplingConfig(),
    generate.TokenStreamFunc(func(token string, done bool) error {
        if !done {
            fmt.Print(token)
        }
        return nil
    }),
)
```

### func (*Generator) BatchGenerate

```go
func (gen *Generator[T]) BatchGenerate(ctx context.Context, requests []BatchRequest) []BatchResult
```

Runs multiple generation requests concurrently. Each request gets its own KV cache and sampling state. Provides request-level parallelism.

```go
requests := []generate.BatchRequest{
    {Prompt: "Hello", Sampling: generate.DefaultSamplingConfig()},
    {Prompt: "World", Sampling: generate.DefaultSamplingConfig()},
}
results := gen.BatchGenerate(ctx, requests)
for _, r := range results {
    if r.Err != nil {
        log.Printf("error: %v", r.Err)
        continue
    }
    fmt.Println(r.Text)
}
```

### func (*Generator) BatchGenerateStream

```go
func (gen *Generator[T]) BatchGenerateStream(ctx context.Context, requests []BatchRequest, streams []TokenStream) []error
```

Runs multiple streaming generation requests concurrently. Each request gets its own KV cache, sampling state, and token stream.

### func (*Generator) NewSession

```go
func (gen *Generator[T]) NewSession() *InferenceSession[T]
```

Creates a new InferenceSession with its own KV cache. The session shares the Generator's graph, tokenizer, and engine but maintains independent KV cache state for isolation.

### func (*Generator) Config

```go
func (gen *Generator[T]) Config() ModelConfig
```

Returns the model configuration.

### func (*Generator) Engine

```go
func (gen *Generator[T]) Engine() compute.Engine[T]
```

Returns the compute engine.

### func (*Generator) Graph

```go
func (gen *Generator[T]) Graph() *graph.Graph[T]
```

Returns the underlying computation graph.

### func (*Generator) Tokenizer

```go
func (gen *Generator[T]) Tokenizer() tokenizer.Tokenizer
```

Returns the tokenizer.

### func (*Generator) GetPrefixCache

```go
func (gen *Generator[T]) GetPrefixCache() *PrefixCache[T]
```

Returns the prefix cache, or nil if prefix caching is disabled.

---

## Generator Options

### type GeneratorOption

```go
type GeneratorOption func(*generatorOptions)
```

Configures a Generator at construction time.

### func WithPagedKV

```go
func WithPagedKV(maxMemoryMB, headDim int) GeneratorOption
```

Enables paged KV caching with the given memory budget in MB. When enabled, blocks are allocated from a shared `BlockPool` instead of pre-allocating the full maxSeqLen per sequence. For GQA models, pass `numKVHeads * actualHeadDim` as `headDim`.

```go
gen := generate.NewGenerator[float32](graph, tok, engine, cfg,
    generate.WithPagedKV(512, 128), // 512 MB budget, 128-dim heads
)
```

### func WithGeneratorKVDtype

```go
func WithGeneratorKVDtype(dtype string) GeneratorOption
```

Sets the KV cache storage dtype. Supported: `"fp32"` (default), `"fp16"`. FP16 halves KV cache memory bandwidth.

### func WithMetrics

```go
func WithMetrics(c runtime.Collector) GeneratorOption
```

Attaches a metrics collector. When speculative decoding is active, updates a `"speculative_acceptance_rate"` gauge after each verify step.

### func WithPrefixCache

```go
func WithPrefixCache(capacityBlocks int) GeneratorOption
```

Enables prefix caching with the given capacity in blocks. When enabled and paged KV is active, sessions that share the same system prompt prefix reuse cached KV blocks instead of re-running prefill.

### func WithSpeculativeDraft

```go
func WithSpeculativeDraft[T tensor.Numeric](draftGraph *graph.Graph[T], draftCfg ModelConfig, draftLen int) GeneratorOption
```

Enables speculative decoding using a separate draft model graph. The draft model proposes `draftLen` tokens greedily per step, then the target model verifies them in a single batched forward pass.

---

## Sampling

### type SamplingConfig

```go
type SamplingConfig struct {
    Temperature       float64          // Divide logits by this value; 0 = greedy
    TopK              int              // Keep only top K tokens; 0 = disabled
    TopP              float64          // Cumulative prob threshold; 1.0 = disabled
    RepetitionPenalty float64          // Penalize repeated tokens; 1.0 = disabled
    MaxNewTokens      int              // Maximum tokens to generate
    StopTokenIDs      []int            // Stop on these token IDs
    StopStrings       []string         // Stop on these strings
    GrammarState      *grammar.Grammar // Optional grammar constraint
}
```

SamplingConfig controls how tokens are selected during generation. When Temperature is zero, greedy argmax is used with an optimized GPU fast path that copies only 4 bytes instead of the full vocabulary logits.

### func DefaultSamplingConfig

```go
func DefaultSamplingConfig() SamplingConfig
```

Returns sensible defaults: temperature 1.0, no filtering, 256 max tokens.

```go
sc := generate.DefaultSamplingConfig()
sc.Temperature = 0.8
sc.TopP = 0.95
sc.MaxNewTokens = 512
```

---

## Inference Sessions

### type InferenceSession

```go
type InferenceSession[T tensor.Numeric] struct {
    // unexported fields
}
```

Holds per-session state for independent, concurrent inference. Each session owns its own KV cache and position tracking, allowing multiple sessions to generate simultaneously without data races.

### func (*InferenceSession) Generate

```go
func (s *InferenceSession[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error)
```

Produces text using the session's own KV cache. Multiple sessions can Generate concurrently.

```go
session := gen.NewSession()
text, err := session.Generate(ctx, "Hello", generate.DefaultSamplingConfig())
```

### func (*InferenceSession) GenerateStream

```go
func (s *InferenceSession[T]) GenerateStream(ctx context.Context, prompt string, sc SamplingConfig, stream TokenStream) error
```

Produces text using the session's own KV cache, delivering tokens via a stream callback.

### func (*InferenceSession) Cache

```go
func (s *InferenceSession[T]) Cache() CacheProvider[T]
```

Returns the session's KV cache provider.

---

## Streaming

### type TokenStream

```go
type TokenStream interface {
    OnToken(token string, done bool) error
}
```

Receives tokens as they are generated. When `done` is true, generation is complete (token may be empty). Returning a non-nil error stops generation.

### type TokenStreamFunc

```go
type TokenStreamFunc func(token string, done bool) error
```

Adapts a function to the TokenStream interface.

```go
stream := generate.TokenStreamFunc(func(token string, done bool) error {
    if !done {
        fmt.Print(token)
    }
    return nil
})
```

---

## Batch Types

### type BatchRequest

```go
type BatchRequest struct {
    Prompt   string
    Sampling SamplingConfig
}
```

A single generation request in a batch.

### type BatchResult

```go
type BatchResult struct {
    Text string
    Err  error
}
```

The output for a single request in a batch.

---

## KV Cache

### type CacheProvider

```go
type CacheProvider[T tensor.Numeric] interface {
    Update(layer int, newK, newV *tensor.TensorNumeric[T]) error
    Get(layer int) (*LayerKV[T], bool)
    SeqLen() int
    Reset()
    Truncate(newSeqLen int)
}
```

The interface implemented by all KV cache variants. Attention layers use this interface to store and retrieve cached key-value tensors during generation.

### func WithCache

```go
func WithCache[T tensor.Numeric](ctx context.Context, cache CacheProvider[T]) context.Context
```

Returns a new context that carries the given CacheProvider.

### func GetCache

```go
func GetCache[T tensor.Numeric](ctx context.Context) (CacheProvider[T], bool)
```

Extracts the CacheProvider from the context, if present.

### type LayerKV

```go
type LayerKV[T tensor.Numeric] struct {
    Key   *tensor.TensorNumeric[T]
    Value *tensor.TensorNumeric[T]
}
```

Holds the cached key and value tensors for a single attention layer.

### type FullBufferProvider

```go
type FullBufferProvider[T tensor.Numeric] interface {
    GetFullBuffer(layer int) (k, v *tensor.TensorNumeric[T])
    MaxSeqLen() int
    KVSeqLenPtr() unsafe.Pointer
}
```

Optional interface for caches that support fixed-size (maxSeqLen) KV buffer access. Enables CUDA graph capture for the decode attention loop.

---

## KVCache (CPU Pre-allocated)

### type KVCache

```go
type KVCache[T tensor.Numeric] struct {
    // unexported fields
}
```

Stores key-value tensors for all attention layers during autoregressive generation. Buffers are pre-allocated to maxSeqLen on first `Update`, and subsequent Updates copy data at the cursor position with zero allocation. Suitable for simple CPU inference.

### func NewKVCache

```go
func NewKVCache[T tensor.Numeric](numLayers, maxSeqLen int) *KVCache[T]
```

Creates a KVCache for the specified number of layers and maximum sequence length. Backing buffers are lazily allocated on the first Update call.

```go
cache := generate.NewKVCache[float32](32, 4096)
```

### func (*KVCache) Update

```go
func (c *KVCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error
```

Appends new key and value tensors to the cache. Tensors must have shape `[batch, seq_len, dim]`.

### func (*KVCache) Get

```go
func (c *KVCache[T]) Get(layer int) (*LayerKV[T], bool)
```

Returns the cached KV pair for the given layer as tensors covering `[0:cursor]`. For batch=1, zero-copy views are returned.

### func (*KVCache) SeqLen / Reset / Truncate / NumLayers

```go
func (c *KVCache[T]) SeqLen() int
func (c *KVCache[T]) Reset()
func (c *KVCache[T]) Truncate(newSeqLen int)
func (c *KVCache[T]) NumLayers() int
```

---

## KVCacheFP16

### type KVCacheFP16

```go
type KVCacheFP16 struct {
    // unexported fields
}
```

Drop-in replacement for `KVCache[float32]` with 2x bandwidth reduction. On `Update`, float32 values are converted to FP16; on `Get`, FP16 values are converted back to float32.

### func NewKVCacheFP16

```go
func NewKVCacheFP16(numLayers, maxSeqLen int) *KVCacheFP16
```

Creates a KVCacheFP16. FP16 backing buffers are lazily allocated.

```go
cache := generate.NewKVCacheFP16(32, 4096)
```

### func (*KVCacheFP16) Update / Get / SeqLen / Reset / Truncate / NumLayers

Same interface as KVCache. Update converts float32 to FP16; Get decodes FP16 back to float32.

---

## TensorCache (GPU Pre-allocated)

### type TensorCache

```go
type TensorCache[T tensor.Numeric] struct {
    // unexported fields
}
```

The default KV cache for GPU-accelerated inference. Pre-allocates GPU-resident buffers and uses direct D2D memcpy for KV appends. Supports FP16 storage mode, GPU-resident position counters for CUDA graph capture compatibility, and the `FullBufferProvider` interface for flash attention decode.

### func NewTensorCache

```go
func NewTensorCache[T tensor.Numeric](engine compute.Engine[T], numLayers, maxSeqLen int, opts ...TensorCacheOption) *TensorCache[T]
```

Creates a TensorCache backed by the given engine. If the engine implements `GPUStreamAccessor`, async memcpy is used for CUDA graph capture compatibility.

```go
cache := generate.NewTensorCache[float32](engine, 32, 4096,
    generate.WithKVDtype("fp16"),
)
```

### type TensorCacheOption

```go
type TensorCacheOption func(*tensorCacheOptions)
```

### func WithKVDtype

```go
func WithKVDtype(dtype string) TensorCacheOption
```

Sets KV storage dtype. Supported: `"fp32"` (default), `"fp16"`. FP16 halves memory bandwidth but requires GPU and CUDA conversion kernels.

### func (*TensorCache) Update / Get / SeqLen / Reset / Truncate / Free

Standard CacheProvider interface, plus:

### func (*TensorCache) GetFullBuffer

```go
func (c *TensorCache[T]) GetFullBuffer(layer int) (k, v *tensor.TensorNumeric[T])
```

Returns GPU-backed KV tensors spanning the full pre-allocated buffer (`maxSeqLen` capacity). Used by flash_attention_decode which reads KV length from a GPU-resident counter.

### func (*TensorCache) MaxSeqLen

```go
func (c *TensorCache[T]) MaxSeqLen() int
```

Returns the maximum sequence length (buffer capacity).

### func (*TensorCache) GPUCounterPtr / KVSeqLenPtr

```go
func (c *TensorCache[T]) GPUCounterPtr() unsafe.Pointer
func (c *TensorCache[T]) KVSeqLenPtr() unsafe.Pointer
```

Return device pointers to GPU-resident int32 counters for CUDA graph capture compatibility.

### func (*TensorCache) SyncCounterFromGPU

```go
func (c *TensorCache[T]) SyncCounterFromGPU() error
```

Performs a D2H copy of the GPU counter to update the CPU seqLen. Call this after the decode loop completes.

---

## PagedKVCache (Block-allocated)

### type PagedKVCache

```go
type PagedKVCache[T tensor.Numeric] struct {
    // unexported fields
}
```

Stores key-value tensors using block-level allocation from a `BlockPool`. Blocks of `blockSize` tokens are allocated on demand, reducing memory waste for concurrent sequences of varying length.

### func NewPagedKVCache

```go
func NewPagedKVCache[T tensor.Numeric](pool *BlockPool[T], numLayers int) *PagedKVCache[T]
```

Creates a paged KV cache backed by the given block pool.

```go
pool, _ := generate.NewBlockPool[float32](32, 64, 128, 512) // 512 MB
cache := generate.NewPagedKVCache[float32](pool, 32)
```

### func (*PagedKVCache) Update / Get / SeqLen / Reset / Truncate / Free

Standard CacheProvider interface, plus `Free()` which returns all allocated blocks to the pool.

### func (*PagedKVCache) BlockTable

```go
func (c *PagedKVCache[T]) BlockTable() []*Block[T]
```

Returns the current block table. Used by the prefix cache to snapshot blocks after prefill.

### func (*PagedKVCache) InjectBlocks

```go
func (c *PagedKVCache[T]) InjectBlocks(blocks []*Block[T], seqLen int)
```

Sets the cache's block table to pre-populated blocks and advances cursors. Used by the prefix cache to inject cached KV data without running a forward pass.

---

## BlockPool

### type BlockPool

```go
type BlockPool[T tensor.Numeric] struct {
    // unexported fields
}
```

Manages a fixed-size pool of pre-allocated KV cache blocks. All methods are safe for concurrent use.

### func NewBlockPool

```go
func NewBlockPool[T tensor.Numeric](numLayers, blockSize, headDim, maxMemoryMB int) (*BlockPool[T], error)
```

Creates a pool of blocks sized to fit within `maxMemoryMB`. Each block holds K and V data for `blockSize` token positions across `numLayers`, with `headDim` elements per position per layer.

```go
pool, err := generate.NewBlockPool[float32](32, 64, 128, 1024) // 1 GB
```

### func (*BlockPool) Alloc / Free

```go
func (p *BlockPool[T]) Alloc() (*Block[T], error)
func (p *BlockPool[T]) Free(b *Block[T])
```

Alloc returns a free block (error if exhausted). Free returns a block to the pool.

### func (*BlockPool) Available / Cap / BlockSize / FragmentationRatio

```go
func (p *BlockPool[T]) Available() int
func (p *BlockPool[T]) Cap() int
func (p *BlockPool[T]) BlockSize() int
func (p *BlockPool[T]) FragmentationRatio() float64
```

### type Block

```go
type Block[T tensor.Numeric] struct {
    K    []T
    V    []T
    Used int // number of token positions written (0..blockSize)
}
```

Holds pre-allocated key and value data for a fixed number of token positions across all layers.

---

## GPUKVCache (Megakernel)

### type GPUKVCache

```go
type GPUKVCache struct {
    // unexported fields
}
```

Manages raw GPU device pointers for megakernel inference. Uses `offset_memcpy` and `increment_counter` CUDA kernels to write KV data at GPU-counter-derived offsets, making the entire append path capturable in CUDA graphs.

### func NewGPUKVCache

```go
func NewGPUKVCache(alloc GPUAllocator, numLayers, maxSeqLen, numHeads, headDim int) (*GPUKVCache, error)
```

Allocates GPU buffers for all attention layers.

### func (*GPUKVCache) Append

```go
func (c *GPUKVCache) Append(layerIdx int, k, v []float32, seqPos int) error
```

Copies host K/V data to the correct GPU position.

### func (*GPUKVCache) AppendGPU

```go
func (c *GPUKVCache) AppendGPU(layerIdx int, kSrc, vSrc unsafe.Pointer, stream unsafe.Pointer) error
```

Copies GPU-resident K/V data using the `offset_memcpy` kernel. Reads the GPU counter to compute write offsets, eliminating D2H copies per token.

### func (*GPUKVCache) Pointers / DevicePointerArrays / GPUCounterPtr

```go
func (c *GPUKVCache) Pointers(layerIdx int) (kPtr, vPtr unsafe.Pointer, seqLen int)
func (c *GPUKVCache) DevicePointerArrays() (kPtrs, vPtrs unsafe.Pointer, err error)
func (c *GPUKVCache) GPUCounterPtr() unsafe.Pointer
```

### func (*GPUKVCache) SeqLen / Reset / Close / SyncCounterFromGPU

```go
func (c *GPUKVCache) SeqLen() int
func (c *GPUKVCache) Reset()
func (c *GPUKVCache) Close() error
func (c *GPUKVCache) SyncCounterFromGPU() error
```

### type GPUAllocator

```go
type GPUAllocator interface {
    Alloc(size int) (unsafe.Pointer, error)
    Free(ptr unsafe.Pointer) error
    Memcpy(dst, src unsafe.Pointer, size int, kind int) error
}
```

Abstracts GPU memory operations. Production code uses a `gpuapi.Runtime` wrapper; tests supply a mock.

---

## Prefix Cache

### type PrefixCache

```go
type PrefixCache[T tensor.Numeric] struct {
    // unexported fields
}
```

Caches KV blocks for shared prompt prefixes using a radix tree. When multiple sessions share the same system prompt, the second session skips prefill by copying cached block data. Safe for concurrent use.

### func NewPrefixCache

```go
func NewPrefixCache[T tensor.Numeric](capacity int, pool *BlockPool[T]) *PrefixCache[T]
```

Creates a prefix cache storing up to `capacity` KV blocks.

### func (*PrefixCache) Insert

```go
func (pc *PrefixCache[T]) Insert(tokenIDs []int32, blocks []*Block[T])
```

Stores KV blocks for a token prefix. Block data is copied into the cache.

### func (*PrefixCache) Match

```go
func (pc *PrefixCache[T]) Match(prefix []int32) ([]*Block[T], int)
```

Returns cached blocks for the longest matching prefix and the number of tokens matched. Returns `nil, 0` on miss.

### func (*PrefixCache) Size

```go
func (pc *PrefixCache[T]) Size() int
```

Returns the number of blocks currently cached.

---

## SSM State

### type SSMState

```go
type SSMState[T tensor.Numeric] struct {
    States    []*tensor.TensorNumeric[T] // one per layer: [1, d_inner, d_state]
    NumLayers int
    DInner    int
    DState    int
}
```

Holds the recurrent hidden state h_t for each MambaBlock layer. Unlike KV cache which grows with sequence length O(seq_len * d_model), SSM state is O(d_state * d_inner) per layer -- constant regardless of sequence length.

### func NewSSMState

```go
func NewSSMState[T tensor.Numeric](numLayers, dInner, dState int) *SSMState[T]
```

Creates an SSMState with each layer's hidden state initialized to zeros.

### func (*SSMState) GetLayer / SetLayer

```go
func (s *SSMState[T]) GetLayer(i int) (*tensor.TensorNumeric[T], error)
func (s *SSMState[T]) SetLayer(i int, h *tensor.TensorNumeric[T]) error
```

### func (*SSMState) Reset

```go
func (s *SSMState[T]) Reset()
```

Clears all layer states to zero, retaining allocated tensors.

### func (*SSMState) MemoryBytes

```go
func (s *SSMState[T]) MemoryBytes() int64
```

Returns total memory used by all layer states in bytes.

---

## Speculative Decoding

### type SpeculativeGenerator

```go
type SpeculativeGenerator[T tensor.Numeric] struct {
    // unexported fields
}
```

Pairs a small draft model with a large target model. The draft proposes N tokens greedily, the target verifies all N in a single batched forward pass. Accepted tokens are emitted; on first mismatch the target's token is used.

An adaptive draft length tracker adjusts N based on rolling acceptance rate (increasing when acceptance > 80%, decreasing when < 40%).

### func NewSpeculativeGenerator

```go
func NewSpeculativeGenerator[T tensor.Numeric](
    draftGraph, targetGraph *graph.Graph[T],
    tok tokenizer.Tokenizer,
    engine compute.Engine[T],
    draftCfg, targetCfg ModelConfig,
    draftLen int,
) *SpeculativeGenerator[T]
```

Creates a speculative generator. `draftLen` controls tokens proposed per verification step (typically 2-8).

```go
specGen := generate.NewSpeculativeGenerator[float32](
    draftGraph, targetGraph, tok, engine,
    draftCfg, targetCfg, 4,
)
text, err := specGen.Generate(ctx, "Hello world", generate.DefaultSamplingConfig())
```

### func (*SpeculativeGenerator) Generate

```go
func (sg *SpeculativeGenerator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error)
```

Produces text using speculative decoding with greedy sampling.

### func (*SpeculativeGenerator) WithAdaptive

```go
func (sg *SpeculativeGenerator[T]) WithAdaptive(enabled bool) *SpeculativeGenerator[T]
```

Enables or disables adaptive draft length adjustment (enabled by default).

---

## Tracing

### type TracingCacheProvider

```go
type TracingCacheProvider[T tensor.Numeric] struct {
    // unexported fields
}
```

Wraps a real CacheProvider and records KV cache operations into a `compute.Tracer` during compilation tracing passes. Captures the full attention dataflow including cache reads and writes.

### func NewTracingCacheProvider

```go
func NewTracingCacheProvider[T tensor.Numeric](real CacheProvider[T], tracer *compute.Tracer[T]) *TracingCacheProvider[T]
```

Creates a tracing wrapper around a real cache.

---

## Model Config

### type ModelConfig

```go
type ModelConfig struct {
    VocabSize  int // Total tokens in vocabulary
    MaxSeqLen  int // Maximum sequence length
    EOSTokenID int // End-of-sequence token ID
    BOSTokenID int // Beginning-of-sequence token ID
    NumLayers  int // Number of transformer layers (for KV cache sizing)
}
```

Holds model architecture parameters needed for generation.

---

## Context Helpers (Deprecated)

### func WithKVCache

```go
func WithKVCache[T tensor.Numeric](ctx context.Context, cache *KVCache[T]) context.Context
```

Deprecated: Use `WithCache` for CacheProvider-based caching.

### func GetKVCache

```go
func GetKVCache[T tensor.Numeric](ctx context.Context) (*KVCache[T], bool)
```

Deprecated: Use `GetCache` for CacheProvider-based caching.
