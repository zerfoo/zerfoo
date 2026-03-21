# Discovery: Layers, Training, and Internal GPU Subsystems

Audit date: 2026-03-21
Scope: `layers/`, `training/`, `internal/cuda/`, `internal/gpuapi/`, `internal/xblas/`, `internal/codegen/`

---

## 1. Layers Subsystem

### 1.1 Attention (`layers/attention/`)

**Files**: `grouped_query_attention.go`, `scaled_dot_product_attention.go`, `multi_head_latent_attention.go`, `flash.go`, `flash_rocm.go`, `global_attention.go`, `local_attention.go`, `attention_head.go`, `qk_norm.go`, `block_table_reader.go`, `interface.go`, registries

**Implementations**:
- **GroupedQueryAttention (GQA)**: Full GQA with Q/K/V/O projections, RoPE, KV cache, optional QK norms, merged QKV decode optimization, fused QK norm+RoPE path, sliding window, bidirectional mode, block table reader
- **ScaledDotProductAttention (SDPA)**: Core Q*K^T scaling, causal masking, softmax, V multiplication. Supports fused flash forward, fused scaled softmax, MatMulTransposeB
- **MultiHeadLatentAttention (MLA)**: DeepSeek V3 MLA with KV low-rank compression, partial RoPE
- **AttentionHead**: Single-head wrapper with Q/K/V Dense projections
- **GlobalAttention**: Thin wrapper over GQA
- **LocalAttention**: Sliding-window wrapper over GQA with explicit mask construction
- **Flash Attention**: CUDA flash attention forward kernel for head_dim <= 128, GPU-only

**Findings**:

1. **[MEMORY SAFETY] Causal masking does CPU .Data() on potentially GPU tensors** (`scaled_dot_product_attention.go:195-208`). During prefill with `causal=true` and no explicit mask, the code calls `scaledAttentionScores.Data()` and directly indexes into the returned slice to apply causal masking. If the tensor is GPU-resident, `.Data()` triggers a D2H copy, modifies the CPU copy, but the GPU copy remains unmasked. The GQA decode path (seqLen=1) skips this, but prefill on GPU hits this path.

2. **[RESOURCE LEAK] Flash attention creates and destroys a stream per call** (`flash.go:58-63`). `tryFlashForward` creates a new CUDA stream, launches the kernel, syncs, and destroys it every invocation. For prefill this is acceptable, but if called repeatedly it adds per-call overhead. The stream could be obtained from the engine.

3. **[CORRECTNESS] negInfValue returns zero for non-float types** (`scaled_dot_product_attention.go:14-25`). For integer types or custom numeric types, `negInfValue` returns the zero value of T, meaning causal masking would not apply any masking at all.

4. **[CORRECTNESS] unsafe.Pointer used in GQA fused paths** (`grouped_query_attention.go:422-433`). The `gpuCounterProvider` interface uses `unsafe.Pointer` for GPU-resident counters. Type assertions are performed on storage types to extract raw GPU pointers for zero-copy views. This is inherently unsafe but follows the pattern used throughout the codebase for GPU interop.

5. **[DESIGN] GQA caches many forward intermediates** (`grouped_query_attention.go:69-76`). Seven fields cache tensors for backward pass. These hold references to GPU memory that persists until the next forward call, preventing GC from reclaiming intermediates between training steps. This is the standard pattern in ML frameworks.

6. **[CORRECTNESS] FlashAttentionDecode path is disabled** (`grouped_query_attention.go:655`). The `if false &&` guard disables the flash decode path. The comment explains SDPA with arena is 15% faster (170 vs 148 tok/s). Dead code is harmless but should be removed or gated behind a flag.

7. **[CORRECTNESS] MLA splitLastDim uses .Data() directly** (`multi_head_latent_attention.go:260-269`). Creates new tensors from raw data copies. On GPU tensors, `.Data()` would trigger D2H copy, and the new tensors would be CPU-resident. MLA partial RoPE would silently fall back to CPU.

### 1.2 Normalization (`layers/normalization/`)

**Files**: `rmsnorm.go`, `layer_normalization.go`, `batch_norm.go`, `simplified_layer_normalization.go`, `skip_simplified_layer_normalization.go`

**Findings**:

8. **[CORRECTNESS] RMSNorm has dual fused paths with any() type assertions** (`rmsnorm.go:121-151`). Both GPU and CPU fused paths use `any(input).(*tensor.TensorNumeric[float32])` type assertions. If T is not float32, both paths silently fall through to the generic multi-step path. This is correct but the fallback path is much slower.

9. **[CORRECTNESS] RMSNorm backward ReduceSum iterates all dims except last** (`rmsnorm.go:218-224`). Uses a loop `for dim := range ndim - 1` which processes dims 0, 1, ..., ndim-2. Each ReduceSum with keepDims=true changes the shape, so subsequent dims are still valid. This is correct.

10. **[CORRECTNESS] BatchNormalization backward returns nil** (`batch_norm.go:125-127`). Inference-only -- no gradient computation. Training would require a separate training-mode batch norm implementation.

### 1.3 Transformer (`layers/transformer/`)

**File**: `block.go`

Standard pre-norm transformer block: norm1 -> attention -> residual -> normPostAttention -> norm2 -> FFN -> residual. Three RMSNorm layers per block.

**Findings**:

11. **[CORRECTNESS] Backward gradient aliasing** (`block.go:189-190`). `dA := dR1[0]` and `dX := dR1[0]` point to the same tensor. At residual 1, the gradient splits into two branches. Since both dA and dX reference the same tensor, this is correct for addition's gradient (both branches get the same upstream gradient). However, if either branch modifies the tensor in-place, the other would be corrupted. The Add/norm/attention operations create new tensors, so this is safe in practice.

### 1.4 State Space Models (`layers/ssm/`)

**Files**: `mamba_block.go`, `s4.go`, `complex_state.go`, `mimo_ssm.go`, `bc_norm.go`

**Implementations**:
- **MambaBlock**: Mamba-1 selective scan with Conv1D, SiLU, SSM (ZOH and ExpTrap discretization)
- **S4**: Diagonal S4D with HiPPO initialization, BPTT backward
- **ComplexSSMState**: Mamba 3 with RoPE on B/C matrices, BCNorm stabilization
- **MIMOMambaBlock**: Multi-head MIMO SSM with cross-head mixing

**Findings**:

12. **[PERFORMANCE] S4 Forward is sequential per timestep** (`s4.go:212-275`). The scan loop iterates over seqLen timesteps, extracting data with copy() and creating new tensors per step. For long sequences this is O(seqLen * dim * stateDim) with high constant overhead from tensor allocations.

13. **[CORRECTNESS] S4 Backward assumes pre-allocated gradients** (`s4.go:402-409`). Directly accesses `s.aLog.Gradient.Data()` without nil check. If gradients are not pre-allocated (first backward call), this will panic. The parameter gradients should be lazily initialized.

14. **[CORRECTNESS] MambaBlock Conv1D and SSM are CPU-only** (`mamba_block.go:289-308`). The causal depthwise Conv1D operates on raw .Data() slices using element-wise ops.Add/ops.Mul. This forces D2H copies for GPU tensors and produces CPU-resident outputs. The entire selective scan would then proceed on CPU.

15. **[MEMORY SAFETY] MambaBlock large intermediate allocations** (`mamba_block.go:267-284`). Creates xData, zData, xConvData etc. as Go slices. For large models (dInner=8192, seqLen=4096, batch=32) these become multi-GB allocations on the Go heap.

### 1.5 Activations (`layers/activations/`)

**Files**: `softmax.go`, `swiglu.go`, `gelu.go`, `fast_gelu.go`, `sigmoid.go`, `relu.go`, `leaky_relu.go`, `tanh.go`, `erf.go`

Standard activations, all implemented via Engine primitives. GELU uses tanh approximation. SwiGLU splits input along last dim, applies SiLU gate.

**Findings**:

16. **[CORRECTNESS] Softmax Backward returns nil** (`softmax.go:36-38`). Inference-only. Training through softmax would fail silently.

17. **[CORRECTNESS] SwiGLU caches lastInput for backward** (`swiglu.go:79`). The entire unsplit input is cached, doubling memory during training. This is standard practice but notable for large models.

### 1.6 Core Layers (`layers/core/`)

Large package with: dense.go, linear.go, ffn.go, conv1d.go, conv2d.go, matmul.go, matmul_nbits.go, gemm.go, rotary_embedding.go, lm_head.go, add.go, mul.go, concat.go, reshape.go, slice.go, pad.go, bias.go, cast.go, polynomial.go, spectral_fingerprint.go, film.go, and many more. Not read in detail due to volume.

### 1.7 Embeddings (`layers/embeddings/`)

**Files**: `token_embedding.go`, `rotary_positional_embedding.go`

RoPE implementation with position offset, scaling, GPU-accelerated angle computation.

### 1.8 Vision (`layers/vision/`)

**File**: `clip_encoder.go` -- CLIP vision encoder stub.

### 1.9 Audio (`layers/audio/`)

**File**: `whisper_encoder.go` -- Whisper audio encoder.

---

## 2. Training Subsystem

### 2.1 Core Training (`training/`)

**Files**: `trainer.go`, `default_trainer.go`, `interfaces.go`, `gradient_strategy.go`, `strategy_backprop.go`, `strategy_one_step.go`, `strategy_common.go`, `batch.go`, `adapter.go`, `early_stop.go`, `windowed.go`, `chunked_iterator.go`

**Architecture**: DefaultTrainer delegates gradient computation to a GradientStrategy (backprop or one-step), then calls optimizer.Step().

**Findings**:

18. **[DESIGN] Trainer interface is minimal** (`trainer.go:13-24`). Single TrainStep method. DefaultTrainer adds strategy pattern. Clean separation of concerns.

### 2.2 Optimizers (`training/optimizer/`)

**Files**: `optimizer.go`, `ema.go`, `swa.go`, `adamw8bit.go`

**Implementations**:
- **AdamW8bit**: Block-wise INT8 quantization of m/v states (4x memory reduction). Dequantize per step, compute update, re-quantize.
- **EMA**: Exponential moving average weight wrapper
- **SWA**: Stochastic weight averaging at epoch boundaries

**Findings**:

19. **[CORRECTNESS] AdamW8bit quantization precision loss** (`adamw8bit.go:24-71`). Block size of 256 elements with INT8 (-127 to 127) gives ~0.8% relative error per element. Over many training steps, the repeated quantize-dequantize cycle accumulates rounding errors in the moment estimates. This is by design (bitsandbytes paper), but users should be aware.

20. **[CORRECTNESS] AdamW8bit clears gradients by zeroing data slice** (`adamw8bit.go:177-179`). Directly writes zeros to gradData[i]. For GPU-resident gradient tensors, .Data() returns a CPU copy, so the zeros would not propagate back to GPU. This optimizer assumes CPU tensors.

21. **[CORRECTNESS] AdamW8bit modifies param.Value.Data() in-place** (`adamw8bit.go:167`). paramData[i] = T(...) directly modifies the underlying slice. For CPU tensors this works. For GPU tensors, param.Value.Data() returns a CPU copy, and in-place modifications would be lost.

22. **[MEMORY] EMA shadow weights double parameter memory** (`ema.go:52-63`). Each parameter gets a full copy in the shadow map. Expected for EMA but significant for large models.

### 2.3 Loss Functions (`training/loss/`)

**Files**: `loss.go`, `cross_entropy_loss.go`, `mse.go`, `corr.go`, `quantile.go`

**Findings**:

23. **[CORRECTNESS] MSE Forward computes mean on CPU** (`mse.go:50-55`). Iterates squared.Data() with a scalar accumulator. For GPU tensors, this D2H copy plus serial reduction is very slow. Should use engine's ReduceSum.

24. **[CORRECTNESS] CrossEntropyLoss uses Softmax then Log separately** (`cross_entropy_loss.go:81-89`). Numerically, log(softmax(x)) should be computed as log_softmax(x) in a single fused operation to avoid underflow when softmax outputs near-zero probabilities. The separate Log of near-zero softmax values can produce -inf.

### 2.4 LoRA (`training/lora/`)

**Files**: `linear.go`, `inject.go`, `qlora.go`, `checkpoint.go`

**Findings**:

25. **[CORRECTNESS] LoRA backward gradient accumulation may fail on first call** (`linear.go:200-201`). `l.B.Gradient, err = l.engine.Add(ctx, l.B.Gradient, dB)` -- if l.B.Gradient is nil on first backward, Add(nil, dB) will likely error. Same issue at line 220 for A.

26. **[CORRECTNESS] LoRA inject assumes weight shape [dIn, dOut]** (`inject.go:83`). `dIn, dOut := weightShape[0], weightShape[1]`. Some layers store weights transposed as [dOut, dIn]. A mismatch would create LoRA matrices with swapped dimensions.

27. **[CORRECTNESS] LoRA checkpoint GGUF writing uses binary.Write per float** (`checkpoint.go:161-165`). Writing each float32 with binary.Write individually is extremely slow for large adapters. Should batch-write the entire float32 slice.

28. **[DESIGN] QLoRATrainer implements its own training loop** (`qlora.go:84-165`). Has forward/backward/optimizer step inline rather than delegating to the trainer framework. This duplicates logic and misses features like gradient clipping, LR scheduling, etc.

### 2.5 FP8 Training (`training/fp8/`)

**Files**: `linear.go`, `loss_scaler.go`, `master_weights.go`, `nvfp4.go`

**Findings**:

29. **[CORRECTNESS] FP8Linear quantizes input on every forward** (`fp8/linear.go:118-123`). Creates new Go slices and tensors per call for the quantized input. For GPU tensors, .Data() forces D2H, and tensor.New creates CPU tensors. The FP8 GEMM dispatch (mentioned in comments) would never trigger since both operands are CPU.

30. **[CORRECTNESS] LossScaler has no upper bound on scale** (`fp8/loss_scaler.go:59-61`). Scale doubles every GrowInterval steps without overflow. After ~60 grow events, scale exceeds float64 max. Should cap at a reasonable maximum (e.g., 2^24).

31. **[CORRECTNESS] MasterWeightStore SyncToFP8 modifies tensor data in-place** (`fp8/master_weights.go:72-73`). masterData[i] = T(v) directly modifies the underlying slice. Same CPU-only assumption as AdamW8bit.

---

## 3. Internal GPU Subsystem

### 3.1 CUDA Runtime (`internal/cuda/`)

**Files**: `purego.go`, `purego_darwin.go`, `purego_linux_arm64.go`, `purego_linux_arm64_cgo.go`, `purego_other.go`, `runtime_purego.go`, `arena.go`, `mempool.go`

**Architecture**: Zero-CGo CUDA bindings via dlopen/dlsym + platform-specific ccall. CUDALib holds resolved function pointers. Global singleton with sync.Once initialization.

**Findings**:

32. **[MEMORY SAFETY] ptrFromUintptr bypasses vet checks** (`runtime_purego.go:37-38`). `return *(*unsafe.Pointer)(unsafe.Pointer(&p))` is a deliberate circumvention of go vet's unsafeptr analyzer. Required for purego FFI but inherently unsafe. A corrupted uintptr return from ccall would become a valid-looking pointer.

33. **[MEMORY SAFETY] goStringFromPtr walks memory unbounded** (`runtime_purego.go:67-73`). Searches for null terminator without a bounds check. If the C string is not null-terminated (corrupt GPU error string), this will walk into unmapped memory and crash.

34. **[RESOURCE LEAK] ArenaPool.Drain does not drain managed memory cache** (`arena.go:171-192`). The Drain() method frees the arena base and drains the fallback pool, but does not free individual fallbackPtrs. After Drain, any outstanding fallback allocations are orphaned. However, fallback.Drain() handles the pool side.

35. **[CORRECTNESS] MemPool bucketing changes allocation size** (`mempool.go:49-63`). Alloc requests > 256 bytes are rounded to next power of 2. A 257-byte request becomes 512 bytes. The caller doesn't know the actual allocation size, but since GPU memory is accessed by kernel-determined bounds, the extra bytes are unused padding.

36. **[CORRECTNESS] ArenaPool managed memory requires env var** (`arena.go:65`). `ManagedMemorySupported(deviceID) && os.Getenv("ZERFOO_ENABLE_MANAGED_MEM") != ""` -- managed memory on DGX Spark requires opt-in via environment variable. Sensible default (disabled) since managed memory has different performance characteristics.

37. **[CORRECTNESS] CUDA graph capture uses relaxed mode** (`runtime_purego.go:313-314`). cudaStreamCaptureModeRelaxed only captures ops on the capturing stream. Other streams can execute normally. This is the correct choice for the arena-based inference path where only one stream is used.

38. **[CORRECTNESS] DeviceComputeCapability uses hardcoded struct offsets** (`runtime_purego.go:27-31`). sizeofCudaDeviceProp = 1008, offsetDevicePropMajor = 360. These are CUDA 13.0 arm64-specific. A CUDA version change would shift offsets and silently return wrong compute capability values.

### 3.2 GPU Abstraction Layer (`internal/gpuapi/`)

**Files**: `runtime.go`, `mempool.go`, `blas.go`, `dnn.go`, `kernels.go`, `factory.go`, `cuda_*.go`, `rocm_*.go`, `opencl_*.go`

**Architecture**: Interfaces for Runtime, MemPool, BLAS, DNN, KernelRunner. Vendor implementations in separate files. BLAS supports SGEMM, BF16, FP16, mixed-precision, batched, transpose-B variants. DNN provides conv, batchnorm, activation, pooling, softmax. KernelRunner has 50+ kernel operations.

**Findings**:

39. **[DESIGN] KernelRunner interface is monolithic** (`kernels.go`). Single interface with 50+ methods. Adding a new kernel requires all vendor implementations to be updated. Could benefit from interface segregation (e.g., separate ElementwiseKernels, ReductionKernels, etc.).

40. **[CORRECTNESS] BLAS assumes row-major layout** (`blas.go:11-16`). Comment says "The implementation handles the row-major to column-major conversion internally." This is critical -- cuBLAS is column-major, so the adapter must transpose. Bugs here would produce silently wrong GEMM results.

### 3.3 SIMD Assembly (`internal/xblas/`)

**Files**: `gemm.go`, `gemm_simd_arm64.go`, `gemm_simd_amd64.go`, `gemm_simd_generic.go`, `gemm_quant.go`, `q4dot.go`, `q4dot_arm64.go`, `rmsnorm_arm64.go`, `rope_arm64.go`, `silu_arm64.go`, `softmax_arm64.go`, `scalar_arm64.go`, `exp_arm64.go`, `elementwise_arm64.go`, `pool.go`

**Architecture**: SIMD-accelerated CPU kernels for ARM NEON and x86 AVX2. GemmF32 dispatches to SgemmSimd. GemmF16/F8 convert through float32 SGEMM.

**Findings**:

41. **[PERFORMANCE] GemmF16 and GemmF8 convert to float32 and back** (`gemm.go:29-70`). Two full-precision conversion passes (input + output) plus the SGEMM itself. For large matrices, the conversion allocations dominate. A native FP16 SIMD GEMM would be significantly faster on ARM NEON (which has FP16 FMLA).

42. **[RESOURCE LEAK] Pool init is not idempotent but guarded** (`pool.go:9-13`). InitPool checks defaultPool != nil before creating. ShutdownPool sets defaultPool = nil after closing. Re-initialization after shutdown works correctly.

### 3.4 Megakernel Codegen (`internal/codegen/`)

**Files**: `compile.go`, `emit.go`, `optable.go`, `runner.go`, `check.go`

**Architecture**: Compiles computation graph instruction tape into a single CUDA megakernel. CachedCompile hashes source for cache validation. EmitMegakernel generates .cu with workspace layout. MegakernelRunner manages compiled .so via dlopen.

**Findings**:

43. **[CORRECTNESS] Megakernel uses tid for all ops** (`optable.go:119-123`). All elementwise ops index as slot_X[tid]. For operations where input and output have different sizes (e.g., broadcast, reduce), this indexing would be incorrect. The reduce and broadcast ops use device functions that handle indexing internally, but the elementwise ops assume element counts match.

44. **[CORRECTNESS] CachedCompile uses hardcoded GPU arch** (`compile.go:36-40`). sm_121 for arm64 (Blackwell), sm_80 for everything else. If the actual GPU has a different compute capability, the compiled kernel may not run or may miss optimizations.

45. **[CORRECTNESS] MegakernelRunner.Close ignores Free errors** (`runner.go:166-171`). All cuda.Free errors are discarded with _ = cuda.Free(buf). If a free fails (e.g., device lost), the error is silently swallowed.

46. **[SECURITY] CachedCompile writes .cu source to disk** (`compile.go:83-84`). The source is written with 0o600 permissions (owner read/write only). The cache directory could be user-controlled, but file permissions are appropriate.

47. **[CORRECTNESS] launch_megakernel calls cudaDeviceSynchronize** (`emit.go:218`). The generated host wrapper synchronizes the entire device after launch. This prevents overlapping compute with memory transfers. For production, the launch should be async with stream-level sync.

---

## 4. Cross-Cutting Issues

### 4.1 CPU/GPU Boundary Violations

The most systemic issue is **CPU-only operations on potentially GPU-resident tensors**. Multiple subsystems call .Data() which triggers D2H copies:
- SDPA causal masking (finding 1)
- MLA splitLastDim (finding 7)
- SSM sequential scan (findings 12, 14)
- MSE loss reduction (finding 23)
- AdamW8bit optimizer (findings 20, 21)
- FP8 linear (finding 29)

These are not bugs on CPU-only paths but will silently produce incorrect results or poor performance when GPU tensors are involved.

### 4.2 Backward Pass Gradient Initialization

Several layers assume gradients are pre-allocated before accumulation:
- S4 backward (finding 13) -- s.aLog.Gradient.Data() without nil check
- LoRA backward (finding 25) -- Add(nil, dB) on first call

### 4.3 Numerical Stability

- CrossEntropyLoss applies Softmax then Log separately (finding 24) -- should use LogSoftmax
- FP8 LossScaler has no upper bound on scale growth (finding 30)

### 4.4 Memory Management

- Training caches forward intermediates across layers (standard but memory-intensive)
- EMA doubles parameter memory
- SSM layers allocate large Go heap slices for intermediate data

---

## 5. Summary Statistics

| Subsystem | Files Read | Findings |
|-----------|-----------|----------|
| Attention | 8 | 7 |
| Normalization | 3 | 3 |
| Transformer | 1 | 1 |
| SSM | 4 | 4 |
| Activations | 3 | 2 |
| Training core | 2 | 1 |
| Optimizers | 3 | 4 |
| Loss | 3 | 2 |
| LoRA | 4 | 4 |
| FP8 | 3 | 3 |
| CUDA runtime | 4 | 7 |
| GPU API | 4 | 2 |
| SIMD/xblas | 2 | 2 |
| Codegen | 4 | 5 |
| **Total** | **48** | **47** |

### Priority Findings

**High** (correctness/data integrity):
- Finding 1: SDPA causal masking on GPU tensors modifies CPU copy only
- Finding 13: S4 backward null gradient panic
- Finding 24: CrossEntropy log(softmax) numerical instability
- Finding 25: LoRA gradient accumulation nil panic on first backward
- Finding 38: Hardcoded cudaDeviceProp struct offsets

**Medium** (performance/design):
- Finding 14: MambaBlock SSM is CPU-only (forces D2H)
- Finding 20-21: AdamW8bit assumes CPU tensors
- Finding 29: FP8Linear quantization forces D2H
- Finding 23: MSE loss CPU reduction
- Finding 47: Megakernel uses device-wide sync

**Low** (style/maintenance):
- Finding 6: Dead FlashAttentionDecode code
- Finding 27: Per-float binary.Write in checkpoint
- Finding 39: Monolithic KernelRunner interface
- Finding 45: Ignored cuda.Free errors
