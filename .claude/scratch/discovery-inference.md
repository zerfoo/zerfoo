# Inference Pipeline Discovery -- End-to-End Audit

Generated: 2026-03-21

---

## 1. Entry Points

### 1.1 `inference.Load()` -- inference/inference.go:184

High-level loader: resolves alias, checks registry, pulls if needed, delegates to `LoadFile`.

**Flow:** `Load` -> `ResolveAlias` -> `registry.Get/Pull` -> `findGGUF` -> `LoadFile`

### 1.2 `inference.LoadFile()` -- inference/load_gguf.go:15

Primary GGUF loader. Steps:
1. `LoadGGUF(path)` -- parse GGUF, load tensors, map names, split merged QKV/GateUp
2. `gguf.ExtractTokenizer(gm.File)` -- build BPE tokenizer from GGUF metadata
3. `createEngine(device)` -- CPU or GPU engine
4. `applyDType(eng, dtype)` -- optional FP16/FP8 precision
5. `gguf.QuantizeToFP8E4M3(tensors)` -- optional FP8 weight quantization
6. `buildArchGraph(arch, tensors, cfg, eng)` -- dispatch to architecture builder
7. Upload weights to GPU if `compute.WeightUploader`
8. `generate.NewGenerator(g, tok, eng, cfg)` -- wire up generator
9. Pre-warm session pool (capacity=8) for CUDA graph address reuse

**Error handling:** Wrapped `fmt.Errorf` at each step. No panics in loading path.

---

## 2. GGUF Parsing -- `model/gguf/`

### 2.1 `parser.go` -- GGUF v2/v3 Binary Parser

- **`Parse(r io.ReadSeeker)`** (line 73): Reads header, validates magic (`0x46554747`), version (2-3), reads metadata KV pairs, tensor info entries
- Supports 13 metadata value types (uint8 through float64, string, array, bool)
- Tensor alignment: 32-byte boundary after header
- Sanity checks: max string length 1MB, max array length 1M entries
- **No unsafe operations** -- pure `binary.Read` + `io.ReadFull`

### 2.2 `arch.go` -- Model Config Extraction

- **`ExtractModelConfig(f *File)`** (line 60): Reads `general.architecture`, then arch-prefixed keys
- Maps GGUF metadata keys to `ModelConfig` fields (hidden size, layers, heads, RoPE, etc.)
- **`MapTensorName(arch, ggufName)`** (line 293): Converts GGUF names (e.g., `blk.0.attn_q.weight`) to canonical names (e.g., `model.layers.0.self_attn.q_proj.weight`)
- Architecture-specific name maps: default, gemma3/gemma3n (4 norms per layer), BERT

### 2.3 `loader.go` -- Tensor Data Loading

- **`LoadTensors(f *File, r io.ReadSeeker)`** (line 18): Reads tensor binary data, decodes per GGML type
- Supported types: F32, F16, BF16, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0
- **Performance trade-off:** Q4_K, Q5_K, Q6_K, Q5_0 are all re-quantized to Q4_0 at load time for fast GEMV decode. Comment says "TODO: optimize Q4_K GEMV to match Q4_0 speed, then use native Q4_K" (line 157)
- F16 stored as `Float16Storage` (compact, 2 bytes/element)
- BF16 stored as `BFloat16Storage`
- **`QuantizeToFP8E4M3(tensors)`** (line 287): In-place FP8 conversion, skips embed/lm_head/norm/bias

### 2.4 `tokenizer.go` -- Tokenizer Extraction

- **`ExtractTokenizer(f *File)`** (line 12): Builds `BPETokenizer` from `tokenizer.ggml.*` metadata
- Extracts vocab, merges, special tokens (BOS/EOS/UNK/PAD)
- Detects GPT-2 byte-level BPE vs SentencePiece
- Extracts control tokens (type 3) for exact matching

### 2.5 `split.go` -- Tensor Splitting

- **`SplitMergedQKV(tensors, cfg)`** (line 17): Handles Phi-style merged QKV weights
- **`SplitMergedGateUp(tensors, cfg)`** (line 134): Handles Phi-style merged gate+up MLP weights
- Both weight and bias variants handled

---

## 3. Architecture Graph Builders -- `inference/`

### 3.1 Dispatch -- `load_gguf.go:141`

`buildArchGraph` dispatches on `cfg.Architecture`:

| Arch String | Builder Function | Special Features |
|---|---|---|
| `llama` | `buildLlamaGraph` | Detects Mistral via SlidingWindow>0 |
| `gemma`, `gemma3` | `buildGemmaGraph` | Tied LM head, sqrt(d) embed scaling |
| `qwen2` | `buildQwenGraph` | Attention bias |
| `mistral` | `buildMistralGraph` | Sliding window attention |
| `phi`, `phi3` | `buildPhiGraph` | Partial rotary factor |
| `deepseek_v3`, `deepseek2` | `buildDeepSeekGraph` | MLA + MoE |
| `mamba` | `buildMambaGraph` | SSM (non-transformer) |
| `mamba3` | `buildMamba3Graph` | SSM v3 |
| `jamba` | `buildJambaGraph` | Hybrid attention+SSM |
| `whisper` | `buildWhisperGraph` | Encoder-decoder |
| `bert` | `buildBertGraph` | Encoder-only |
| `llama4` | `buildLlama4Graph` | Llama + MoE |
| `llava` | `buildLLaVAGraph` | Vision-language |
| `qwen_vl` | `buildQwenVLGraph` | Vision-language |
| fallback | Dynamic registry (`GetArchitecture`) | |

### 3.2 Shared Transformer Builder -- `arch_common.go:101`

`buildTransformerGraph` is the workhorse, used by Llama, Gemma, Mistral, Qwen, Phi.

**Per-layer structure:**
```
Input -> RMSNorm -> GQA -> [PostAttnNorm?] -> FusedAdd+RMSNorm -> FFN(SwiGLU) -> [PostFFNNorm?] -> ResidualAdd
```

**Key implementation details:**

- **Weight transposition** (`transposeWeight`, line 131): Complex logic for Q4/Q8/FP16/FP8/BF16 storage types. Q4 uses "virtual transpose" (shape swap only). Q8 is dequantized to F32 then transposed. FP16/FP8 are dequantized, transposed, and re-encoded to preserve native storage.
- **Merged QKV GEMV** (line 391): For Q4 weights, merges Q/K/V into single tensor for fused decode GEMV
- **Merged Gate+Up GEMV** (line 510): Same for MLP gate+up projections
- **LM head Q8->Q4 conversion** (line 560): Converts Q8 lm_head weights to Q4 for fast GEMV
- **Fused ops:** `fusedAddRMSNormNode`, `fusedNormAddNode` save kernel launches

**Architecture-specific `transformerGraphOpts`:**
- `embedScale` -- Gemma: sqrt(hidden_size)
- `postNorm` -- Gemma 3: post-attention and post-FFN norms
- `qkNorm` -- Gemma 3: RMSNorm on Q/K after projection
- `logitSoftcap` -- Gemma 3: `cap * tanh(logit/cap)` using rational tanh approximation
- `slidingWindowSize` -- Mistral: causal sliding window mask
- `attnBias` -- Qwen 2: bias on Q/K/V projections
- `partialRotaryFactor` -- Phi: RoPE on fraction of head dims

### 3.3 DeepSeek Builder -- `arch_deepseek.go:30`

Unique architecture with:
- **Multi-head Latent Attention (MLA):** Compresses KV into low-rank latent space via `attention.NewMultiHeadLatentAttention`
- **MoE (Mixture of Experts):** Router-gated expert selection with stacked expert weight slicing (`extractExpertSlice`, line 469)
- **Shared experts:** Optional parallel expert FFN
- Custom reshape nodes for 3D<->2D conversion for MoE

### 3.4 Architecture Registry -- `registry.go` + `registry_init.go`

- Thread-safe registry (`sync.RWMutex`) for `ArchBuilder` functions
- `init()` registers 14 built-in architectures
- `RegisterArchitecture` panics on empty name, nil builder, or duplicate registration

### 3.5 AutoBuild -- `auto_builder.go:74`

Automatic graph construction from GGUF metadata without per-model builder. Detects features from `ModelConfig` fields. Falls back to registered builder for non-transformer architectures (Mamba, Whisper, RWKV).

### 3.6 Config Parsing -- `arch_config.go`

`ArchConfigRegistry` with per-model-type parsers for config.json files. Fallback parser for unknown types extracts common HuggingFace field names.

---

## 4. Model Abstraction -- `model/`

### 4.1 `model/model.go`

Simple `Model[T]` struct with `Embedding` + `Graph`. Forward: embedding lookup then graph forward.

### 4.2 `model/interfaces.go`

Rich interface hierarchy (not all implemented):
- `ModelProvider[T]`, `ModelInstance[T]`, `ModelSerializer[T]`, `ModelLoader[T]`, `ModelExporter[T]`, `ModelValidator[T]`, `ModelOptimizer[T]`
- Mostly forward-looking design; inference pipeline uses `inference.Model` directly

### 4.3 `model/registry.go`

Layer builder registry mapping `op_type` strings to `LayerBuilder[T]` functions.

---

## 5. Text Generation -- `generate/`

### 5.1 Generator -- `generator.go`

**`Generator[T]`** (line 133): Core generation engine.

**Key fields:**
- `graph`, `tokenizer`, `engine`, `config`
- `plan atomic.Pointer[graph.ExecutionPlan[T]]` -- compiled CUDA graph
- `blockPool *BlockPool[T]` -- paged KV allocation
- `prefixCache *PrefixCache[T]` -- radix-tree prefix KV reuse
- `specDraft *specDraftConfig` -- speculative decoding config

**`Generate()`** (line 294): Main generation loop.
1. Encode prompt, prepend BOS
2. Select cache: PagedKV > TensorCache (GPU) > KVCache (CPU)
3. Reset stateful nodes
4. Prefill: full prompt forward
5. Decode loop: per-token forward with compile-on-first-decode (`compileGraph`)
6. CUDA graph capture after first decode forward
7. Arena pool reset between tokens (`compute.PoolResetter`)
8. Grammar masking per step if grammar-constrained
9. Stop on EOS, stop token IDs, stop strings, grammar completion

**`compileGraph()`** (line 220):
- `sync.Once` guarded
- Tries `CompileTraced` first, falls back to `Compile`
- Validates traced plan with test run
- CUDA graph capture via `graph.NewCUDAGraphExecutor` with arena floor protection
- Async megakernel compilation (`tryCompileMegakernel`)

### 5.2 InferenceSession -- `session.go`

Per-session state for concurrent inference:
- Owns its own KV cache
- Shares graph, tokenizer, engine with Generator
- `graphMu *sync.Mutex` -- shared mutex (graph is not concurrent-safe)
- Sessions are pooled in `inference.Model.sessionPool` (capacity 8)
- Prefix cache integration: match cached KV blocks, skip redundant prefill

**Generate flow:** Lock session, lock graph, encode, prefill, decode loop (same as Generator)

### 5.3 Sampling -- `sampling.go`

- `applyTemperature`, `applyTopK`, `applyTopP`, `applyRepetitionPenalty`
- `sampleFromDistribution` -- weighted random from softmax probs
- `softmax` with max-value numerical stability
- GPU argmax fast path: `compute.GPUArgmaxer` copies 4 bytes vs ~1MB of logits

### 5.4 Streaming -- `stream.go`

- `TokenStream` interface: `OnToken(token string, done bool) error`
- `TokenStreamFunc` adapter for function callbacks
- Incremental delta emission: decode full sequence, diff against previous output
- Stop string detection mid-stream

### 5.5 KV Cache Variants

**`KVCache[T]`** (kvcache.go:28): CPU pre-allocated cache
- Lazy allocation on first Update per layer
- Zero-copy views for batch=1
- Buffer retained across Reset for reuse

**`PagedKVCache[T]`** (paged_kv.go:18): Block-based allocation from shared `BlockPool`
- On-demand block allocation (blockSize tokens per block)
- Multi-layer blocks: `[numLayers][blockSize][headDim]` layout
- Supports `InjectBlocks` for prefix cache integration

**`TensorCache[T]`** (tensor_cache.go): GPU-backed cache
- Uses `unsafe.Pointer` for direct GPU memory operations
- FP16 KV cache support (`WithKVDtype("fp16")`)
- GPU-resident sequence length counter for CUDA graph capture (`KVSeqLenPtr()`)
- `GPUCounterPtr()` returns `unsafe.Pointer` to device int32

**`GPUKVCache`** (gpu_kv_cache.go): Low-level GPU KV cache
- Direct CUDA memory allocation via `GPUAllocator` interface
- `Alloc(size int) (unsafe.Pointer, error)`, `Free(ptr unsafe.Pointer)`, `Memcpy(dst, src unsafe.Pointer, ...)`
- `AppendGPU` for device-to-device copy with stream
- `DevicePointerArrays` allocates GPU arrays of per-layer K/V pointers

### 5.6 Context -- `context.go`

- `CacheProvider[T]` interface: `Update`, `Get`, `SeqLen`, `Reset`, `Truncate`
- `FullBufferProvider[T]`: GPU-backed full-buffer access for CUDA graph capture
  - `KVSeqLenPtr() unsafe.Pointer` -- GPU-resident counter
- Cache stored in `context.Context` via `kvCacheKey{}`

---

## 6. Speculative Decoding

### 6.1 `generate/speculative.go` -- Integrated Speculative Generator

`SpeculativeGenerator[T]` with separate draft/target graphs.

**Flow:**
1. Prefill both models
2. Draft phase: K greedy tokens from draft model
3. Verify phase: target model processes all K in single forward
4. Accept/reject: compare target argmax with draft tokens
5. Bonus token from target at last position
6. Cache rollback on rejection (Truncate)
7. Adaptive draft length adjustment

### 6.2 `generate/speculative/` -- Advanced Strategies

**`SelfDraft[T]`** (self_draft.go:25): Same model for draft+verify
- Draft uses first N/2 layers (early exit)
- Verify uses all N layers
- No separate draft model needed

**`ExternalDraft[T]`** (external_draft.go:17): Separate smaller model
- Returns tokens + log probabilities
- Shares compute engine and block pool

**`AcceptTokens`** (sampler.go:30): Rejection sampling per Leviathan et al. 2023
- Accept with prob `min(1, p(x)/q(x))`
- Rejection: sample from `max(0, p-q)` renormalized
- All accepted: bonus token from target distribution

### 6.3 Generator-integrated speculative -- `generator.go:656`

`generateSpeculative` within Generator:
- Alpha-based fallback: if rolling acceptance < 0.4 after 8 steps, switches to standard autoregressive
- Prometheus gauge: `speculative_acceptance_rate`
- Adaptive draft length via `adaptiveDraftLen`

---

## 7. Agent / Tool-Calling -- `generate/agent/`

### 7.1 `tools.go` -- Tool Registry

- `ToolRegistry`: thread-safe (RWMutex) map of tool name -> handler function
- `ToolCall`: `{ID, Name, Arguments}`
- `ToolResult`: `{CallID, Output, IsError}`
- `Call`: executes handler with panic recovery (`defer func() { if rec := recover() {...} }()`)

### 7.2 `function_call.go` -- Parser

- `FunctionCallParser.Parse(text)`: Scans model output for JSON tool-call objects
- Recognizes `{"name":..., "arguments":...}` and `{"tool":..., "parameters":...}`
- Balanced brace extraction with string escape handling
- Extracted JSON removed from text

### 7.3 `supervisor.go` -- Agentic Loop

- `Supervisor.RunLoop`: generate -> parse -> execute tools -> repeat
- Stop conditions: no tool calls, max steps, tool error, max tokens
- History: tool results appended as JSON for next generation

---

## 8. Grammar-Constrained Decoding -- `generate/grammar/`

### 8.1 `grammar.go` -- State Machine

- `Grammar` wraps immutable `node` interface
- `Advance(b byte) -> (next, ok)` -- consumes one byte
- `ValidBytes() []byte` -- all valid next bytes
- `IsComplete() bool` -- accepting state (complete JSON value)

### 8.2 `mask.go` -- Token Masking

- `TokenMask(g, vocab) []bool` -- per-token validity mask
- For each vocab token, tests all bytes via `Grammar.Advance`

### 8.3 `grammar_mask.go` (in generate/) -- Integration

- `applyTokenMask(logits, mask)` -- sets invalid tokens to -Inf
- `advanceGrammar(g, tokenID, vocab)` -- advances state through sampled token's bytes

---

## 9. Multimodal Inference -- `inference/multimodal/`

### 9.1 `preprocess.go` -- Image Preprocessing

- `PreprocessImage(data, format, cfg)`: Decode (JPEG/PNG) -> bilinear resize -> normalize per channel -> extract patches
- `PatchConfig`: patch size, image size, channel-wise mean/std

### 9.2 `vision_encoder.go` -- Vision Encoder

- `VisionEncoder[T]` interface: `Encode(patches, cfg) -> []T`
- `SigLIPEncoder[T]`: Linear projection from patch embeddings to hidden dim via `engine.MatMul`

### 9.3 `connector.go` -- Vision-to-Text Projection

- `ProjectionConnector[T]`: Linear projection from vision dim to text dim
- Weight key defaults to `mm.projector.weight`

### 9.4 `merge.go` -- Embedding Merge

- `MergeEmbeddings`: Replaces image-token positions in text embeddings with vision embeddings
- Supports `MaxImageTokens` limit

### 9.5 `audio.go`, `audio_session.go` -- Audio Processing

Audio preprocessing and session management for speech models.

### 9.6 `gguf_loader.go` -- Multimodal GGUF Loading

Loads vision encoder weights from GGUF metadata (CLIP-style fields: `clip.vision.*`).

---

## 10. Parallel Inference -- `inference/parallel/`

### 10.1 `pipeline_parallel.go` -- Pipeline Parallelism

- `PipelineParallelConfig`: NumStages, NumLayers, MicroBatchSize
- `AssignLayers`: Even distribution of layers across stages
- `PipelineExecutor`: Multi-stage forward with activation transfer between stages
- Uses goroutines + channels for inter-stage communication

### 10.2 `tensor_parallel.go` -- Tensor Parallelism

- `TensorParallelConfig`: NumGPUs, DeviceIDs
- `AllReducer[T]` interface: `AllReduceSum` across ranks
- `ShardedWeight[T]`: Per-device weight slice with `ColumnSplit` or `RowSplit` modes

---

## 11. Time-Series Inference -- `inference/timeseries/`

### 11.1 `gguf_loader.go` -- TS Config Loading

- `TimeSeriesSignalConfig`: patch_len, stride, input_features, hidden_dim, num_heads, num_layers, horizon_len
- Loaded from `ts.signal.*` GGUF metadata keys

### 11.2 Architecture Builders

- `arch_patchtst.go` -- PatchTST architecture
- `arch_tft.go` -- Temporal Fusion Transformer
- `arch_regime.go` -- Regime detection model

---

## 12. Sentiment Analysis -- `inference/sentiment/`

### 12.1 `finetune.go` -- Fine-tuning

- `TrainingConfig`: epochs, LR, batch size, validation split, LoRA rank, labels
- `TrainableModel` interface: `Forward(inputIDs) -> logits`, `UpdateParams(grad, lr)`
- Data loading from CSV/JSONL files

---

## 13. Fused Operations -- `inference/`

### 13.1 `fused_add_rmsnorm_node.go`

- **`fusedAddRMSNormNode[T]`** (line 19): Fuses `Add + RMSNorm` into single kernel
  - GPU path: `compute.FusedAddRMSNormProvider.GPUFusedAddRMSNorm`
  - CPU fallback: separate `engine.Add` + `compute.FusedRMSNorm`
  - Stores residual internally for `residualAddNode` retrieval
- **`residualAddNode[T]`** (line 89): Adds ffnOut + stored residual
- **`residualRefNode[T]`** (line 119): Retrieves stored residual as graph input

### 13.2 `fused_norm_add_node.go`

Fused `RMSNorm + Add` for post-FFN normalization in Gemma 3.

---

## 14. Unsafe Operations Inventory

### 14.1 `generate/gpu_kv_cache.go` -- **Heavy unsafe usage**

- `unsafe.Pointer` for GPU memory pointers (K/V buffers, counter)
- `unsafe.Add` for pointer arithmetic on GPU buffers (line 154, 159)
- `unsafe.Pointer(&data[0])` for host->device copy source
- `GPUAllocator` interface exposes raw `unsafe.Pointer`

### 14.2 `generate/tensor_cache.go`

- `unsafe.Add(dst.Ptr(), offset*2)` -- FP16 offset arithmetic (line 372)
- `unsafe.Pointer(nil)` for stream pointers

### 14.3 `generate/cuda_allocator.go`

- Wraps CUDA alloc/free/memcpy behind `unsafe.Pointer` interface

### 14.4 `generate/ssm_state.go`

- `unsafe.Sizeof(float32(0))` for element size calculation (line 70)

### 14.5 `generate/context.go`

- `KVSeqLenPtr() unsafe.Pointer` in `FullBufferProvider` interface (line 36)

### 14.6 `inference/tensorrt_convert.go` + `tensorrt_pipeline.go`

- `unsafe.Pointer(&data[0])` for TensorRT weight/bias data and tensor addresses
- Direct device memory operations

---

## 15. Error Handling Patterns

1. **Wrapped errors:** Consistent use of `fmt.Errorf("context: %w", err)` throughout
2. **Missing tensor errors:** All builders check for required tensors with `lookup()` returning descriptive errors
3. **No panics in hot path:** Only `RegisterArchitecture` panics (init-time, not runtime)
4. **GPU fallbacks:** Fused operations fall through to CPU on GPU error
5. **Context cancellation:** All generation loops check `ctx.Err()` per iteration
6. **Session mutex:** Both session-level and graph-level mutexes prevent concurrent graph access
7. **Panic recovery:** `ToolRegistry.Call` uses `defer/recover` for tool handler panics

---

## 16. Data Flow Summary

```
GGUF File
  |
  v
gguf.Parse() -- binary header, metadata, tensor info
  |
  v
gguf.LoadTensors() -- read+decode F32/F16/BF16/Q4/Q5/Q6/Q8 -> tensor.TensorNumeric[float32]
  |
  v
gguf.MapTensorName() -- GGUF names -> canonical names
gguf.SplitMergedQKV() / SplitMergedGateUp() -- handle merged tensors
  |
  v
gguf.ExtractModelConfig() -- metadata -> ModelConfig
gguf.ExtractTokenizer() -- metadata -> BPETokenizer
  |
  v
buildArchGraph() -- dispatch by architecture
  |
  v
buildTransformerGraph() -- shared builder:
  - embeddingLookupNode (embed + optional scale)
  - per-layer: RMSNorm -> GQA (with RoPE, optional bias/QK norms/sliding window) -> FusedAddRMSNorm -> FFN(SwiGLU) -> ResidualAdd
  - final RMSNorm -> lmHeadNode (with optional softcap)
  |
  v
graph.Graph[float32] + embedWeight
  |
  v
generate.NewGenerator() -- wraps graph + tokenizer + engine + cache config
  |
  v
inference.Model -- public API:
  - Generate(ctx, prompt) -> string
  - GenerateStream(ctx, prompt, handler) -> error
  - GenerateBatch(ctx, prompts) -> []string
  - Chat(ctx, messages) -> Response
  - Embed(text) -> []float32
  - SpeculativeGenerate(ctx, draft, prompt) -> string
  |
  v
Generation loop (in Generator or InferenceSession):
  1. Encode prompt -> token IDs
  2. Prepend BOS
  3. Select cache (PagedKV > TensorCache > KVCache)
  4. Prefill: graph.Forward(promptTensor)
  5. Sample first token from logits
  6. Decode loop:
     a. Reset arena pool
     b. graph.Forward([1,1] token tensor) OR plan.Run (CUDA graph)
     c. Sample from logits (greedy / temperature / top-k / top-p / grammar)
     d. Check stop conditions
  7. Decode token IDs -> text
```
