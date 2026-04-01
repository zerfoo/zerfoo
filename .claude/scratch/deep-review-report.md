# DRY Composition Audit Report

**Repository:** `github.com/zerfoo/zerfoo`
**Date:** 2026-03-30
**Scope:** `timeseries/`, `layers/`, `inference/`, `training/`, `generate/`, `serve/`

---

## Executive Summary

The zerfoo codebase has **strong DRY composition in its core layers and inference packages**, where neural network operations are defined once in `layers/` and composed via `inference/arch_common.go`. However, the **`timeseries/` package has severe DRY violations**, with the same fundamental operations (GELU, layer norm, softmax, linear projection, AdamW, gradient clipping) reimplemented up to 11 times across different backends and training paths.

**By the numbers:**
- `timeseries/` non-test source: **19,150 lines**
- `layers/` non-test source: **28,532 lines** (defines 56+ ops ONCE each)
- Estimated duplicated lines in `timeseries/`: **~4,800 lines** (25% of source)
- Estimated reduction with DRY refactoring: **~3,200 lines** (17% of source)

---

## Phase 1: Architecture Map

### Where DRY Composition Works Well

**A. `layers/` package (28,532 non-test lines)**
Each neural network operation is defined once with generics:
- `layers/normalization/layer_normalization.go` (346 lines) -- single LayerNormalization[T] with Forward/Backward
- `layers/normalization/rmsnorm.go` (307 lines) -- single RMSNorm[T]
- `layers/activations/gelu.go` (240 lines) -- single Gelu[T] composed from Engine primitives
- `layers/attention/scaled_dot_product_attention.go` (336 lines) -- single SDPA with causal masking
- `layers/attention/grouped_query_attention.go` (1,212 lines) -- single GQA
- `layers/core/ffn.go` (368 lines) -- single FFN using Dense layers + SwiGLU

This follows the design principle perfectly: each op is defined once, parameterized via generics, and composed from Engine primitives.

**B. `inference/` architecture builders (30,061 non-test lines)**
`inference/arch_common.go` defines `buildTransformerGraph()` which is shared by **17 architectures** (Llama, Gemma, Mistral, Qwen, Phi, etc.) through `transformerGraphOpts`. Architecture-specific files provide only the delta:
- `arch_llama.go` configures RoPE theta
- `arch_gemma.go` adds embedding scaling, QK norms, logit softcap
- `arch_qwen.go` enables attention bias
- `arch_phi.go` sets partial rotary factor

Only truly different architectures (BERT, DeepSeek MLA, Mamba, RWKV) have their own graph builders. This is excellent DRY composition.

**C. `training/` package (3,919 lines)**
Well-decomposed abstractions:
- `interfaces.go` -- `Model[T]`, `Trainer[T]`, `Optimizer[T]`, `LossFunction[T]`
- `windowed.go` -- generic windowed training loop
- `strategy_backprop.go`, `strategy_one_step.go` -- composable gradient strategies
- `adapter.go` -- bridges timeseries models to training framework

**D. `generate/` package**
KV cache hierarchy is well-layered: `kvcache.go` -> `kvcache_fp16.go` / `kvcache_fp8.go` / `kvcache_q4.go` / `paged_kv.go` with clear separation of concerns.

---

## Phase 2: DRY Violations

### Category A: Duplicated Primitive Operations

#### A1. GELU Activation -- 7 implementations

| Implementation | File | Type | Lines |
|---|---|---|---|
| `geluScalar` | `patchtst.go:696` | float32 | 7 |
| `geluF64` | `patchtst_backward.go:678` | float64 | 10 |
| `geluDerivF64` | `patchtst_backward.go:655` | float64 derivative | 10 |
| `geluDerivF32` | `patchtst_gpu_train.go:1416` | float32 derivative | 8 |
| `gelu` | `itransformer.go:353` | float64 | 7 |
| `geluGrad` | `itransformer_backward.go:6` | float64 derivative | 8 |
| `geluMatrix` | `ttm.go:1436` | float64 matrix | 11 |

**Total duplicated: ~61 lines**
**Classification: UNJUSTIFIED** -- These are identical formulas. A single generic `geluScalar[T]` + `geluDeriv[T]` would suffice. The `layers/activations/gelu.go` already exists but operates on tensors via Engine, so a shared package-level scalar function is needed.

#### A2. Layer Normalization -- 11 implementations

| Implementation | File | Type | Lines |
|---|---|---|---|
| `layerNormF64WithCache` | `patchtst_backward.go:556` | f64 fwd+cache | 36 |
| `layerNormBackwardF64` | `patchtst_backward.go:593` | f64 backward | 37 |
| `layerNormF64` | `patchtst_backward.go:993` | f64 fwd only | 25 |
| `layerNormF32WithCache` | `patchtst_gpu_train.go:336` | f32 fwd+cache | 30 |
| `layerNormBackwardF32` | `patchtst_gpu_train.go:369` | f32 backward | 27 |
| `PatchTST.layerNorm` | `patchtst.go:509` | tensor Engine | 30 |
| `layerNormCached` | `itransformer_backward.go:290` | f64 vec+cache | 25 |
| `layerNormBackward` | `itransformer_backward.go:539` | f64 backward | 38 |
| `layerNorm` | `itransformer.go:329` | f64 vec only | 22 |
| `TTM.layerNormF32` | `ttm.go:627` | tensor Engine | 30 |
| `TFT.layerNorm` | `tft.go:535` | tensor Engine | ~30 |

**Total duplicated: ~330 lines**
**Classification: PARTIALLY JUSTIFIED**
- The f32/f64 split is inherent to Go (no numeric generics for plain floats)
- The per-model Engine method variants (`PatchTST.layerNorm`, `TTM.layerNormF32`) are identical and UNJUSTIFIED -- should be a shared free function
- The fwd-with-cache vs fwd-only vs backward split is structurally necessary for training
- iTransformer uses a different signature (1D vector vs 2D matrix) -- should be unified to 2D

#### A3. MatMul Engine Wrapper -- 3 copies (2 are exact duplicates)

| Implementation | File | Lines |
|---|---|---|
| `PatchTST.matMulEngine` | `patchtst_engine.go:21` | 70 |
| `PatchTST.matMulEngineWithBufs` | `patchtst_engine.go:25` | 67 |
| `TTM.matMulEngine` | `ttm_train_engine.go:15` | 52 |

**`TTM.matMulEngine` is character-for-character identical to `PatchTST.matMulEngine` minus buffer reuse.**
**Total duplicated: ~52 lines**
**Classification: UNJUSTIFIED** -- Should be a shared free function.

#### A4. Linear F64 Engine -- 2 exact copies

| Implementation | File | Lines |
|---|---|---|
| `PatchTST.linearF64Engine` | `patchtst_engine.go:98` | 22 |
| `TTM.linearF64Engine` | `ttm_train_engine.go:72` | 22 |

**Character-for-character identical except the receiver type.**
**Classification: UNJUSTIFIED** -- Should be a shared free function.

#### A5. Softmax -- 4 implementations

Three inline implementations in PatchTST files (patchtst_backward.go, patchtst_engine.go: both ~15 lines each, identical max-subtract-exp-normalize pattern) plus `softmax` in itransformer.go (12 lines).
**Total duplicated: ~42 lines**
**Classification: PARTIALLY JUSTIFIED** -- Inline in hot loops may avoid function call overhead, but the f64 standalone function should be shared.

### Category B: Duplicated Training Infrastructure

#### B1. AdamW Optimizer -- 4 implementations

| Implementation | File | Type | Lines |
|---|---|---|---|
| `adamWUpdate` | `training_ops.go:41` | shared f64, `[]*float64` | 12 |
| `NHiTS.adamUpdate` | `nhits.go:652` | f32, `[]float32` | 12 |
| Inline AdamW | `patchtst_gpu_train.go:1366` | f32, tensor data | 12 |
| Inline AdamW | `patchtst_engine.go:706` | f64, `[]*float64` | 6 |

**Classification: PARTIALLY JUSTIFIED**
- The f64 `[]*float64` vs f32 `[]float32` split is a real type boundary
- The shared `adamWUpdate` is already used by 4 backends via `TrainLoop`
- NHiTS engine and PatchTST GPU duplicate the f32 variant UNJUSTIFIABLY

#### B2. Gradient Clipping -- 3 implementations

| Implementation | File | Type | Lines |
|---|---|---|---|
| `clipGradients` | `training_ops.go:8` | shared f64 | 14 |
| `NHiTS.clipGradients` | `nhits.go:634` | f32 method | 14 |
| Inline | `patchtst_gpu_train.go:1332` | f32 tensor | 18 |

**Classification: UNJUSTIFIED** -- The f32 variant is a trivial type change. Should have a shared `clipGradientsF32`.

#### B3. adamState struct -- 4 definitions

| Definition | File | Type |
|---|---|---|
| `adamWState` | `training_ops.go:26` | f64 |
| `adamState` | `nhits.go:276` | f32 |
| `adamState` (inline) | `frets_engine.go:113` | f32 |
| `adamState` (inline) | `dlinear_engine.go:52` | f32 |

**Classification: UNJUSTIFIED** -- Should be a single `adamStateF32` in a shared file.

#### B4. Training Loops -- 5+ backends with their own loops

Backends using shared `TrainLoop` (good): DLinear, FreTS, NHiTS (CPU), CfC
Backends with bespoke loops: PatchTST (3 loops!), ITransformer, TTM, NHiTS (engine), TimeMixer

The bespoke loops all implement the same outer structure:
1. Batch iteration
2. Forward pass
3. MSE loss + gradient computation
4. Gradient clipping
5. LR warmup
6. AdamW update
7. Loss history tracking

**Classification: HISTORICALLY JUSTIFIED, now UNJUSTIFIED for most**
- PatchTST GPU (`trainWindowedGPU`) fuses forward+backward into a single pass for performance -- **PERFORMANCE-JUSTIFIED**
- All others could use a generalized training loop with a callback pattern

#### B5. TimeMixer Has a Different TrainConfig Signature

`TimeMixer.TrainWindowed` takes `(windows, labels, epochs int)` instead of `(windows, labels, TrainConfig)`. This means it cannot use the shared `TrainLoop` infrastructure.

**Classification: UNJUSTIFIED** -- API inconsistency that prevents composition.

### Category C: Duplicated Data Processing

#### C1. `copyMatrix` vs `deepCopy2D` -- 2 identical functions

| Function | File | Purpose |
|---|---|---|
| `copyMatrix` | `patchtst_backward.go:668` | Deep-copy `[][]float64` |
| `deepCopy2D` | `itransformer_backward.go:163` | Deep-copy `[][]float64` |

**Classification: UNJUSTIFIED** -- Identical logic, different names.

#### C2. `normalizeWindows` and `applyNormalization` -- well shared

These are defined once in `dlinear.go` and used across backends. **Good DRY.**

### Category D: Duplicated Type Definitions

#### D1. PatchTST has 3 parallel type hierarchies

1. **Inference types**: `encoderLayer`, `linearLayer` (float32 tensors)
2. **CPU training types**: `patchTSTParamsF64`, `encoderLayerParamsF64`, `encoderLayerF64Grad` (float64 slices)
3. **GPU training types**: `gpuParams`, `gpuEncoderLayer`, `gpuGrads`, `gpuAdamState` (float32 tensors)
4. **Cache types**: `patchTSTCacheF64`, `encoderLayerCache` (float64 for CPU backward)
5. **GPU cache types**: `gpuBatchLayerCache`, `gpuBatchForwardCache` (float32 tensors)

Each set mirrors the same structure (patchEmb, posEmb, N layers with qkvo+ffn+norm, head). This structural duplication accounts for ~300 lines of pure boilerplate.

**Classification: PARTIALLY JUSTIFIED** -- Different types are needed for different numeric precisions and storage formats. But the boilerplate of `allParamTensors()`, `allocGrads()`, and `writeBack*` methods could be generated or abstracted.

#### D2. iTransformer has its own separate type hierarchy

`iTransformerLayer`, `iTransformerCache`, `iTransformerLayerCache`, `iTransformerGrads`, `iTransformerLayerGrads` -- all structurally similar to PatchTST's types but for a different model. This is expected (different architectures), but the cache/grad patterns are repetitive.

### Category E: The PatchTST Mega-Duplication

PatchTST is the worst DRY offender in the entire codebase. Here is what its 6,196 lines break down to:

| File | Lines | What it implements |
|---|---|---|
| `patchtst.go` | 958 | Config, constructor, tensor Forward, tensor layerNorm, helper fns |
| `patchtst_backward.go` | 1,104 | CPU f64: forwardWithCache, backwardF64, linearF64, layerNorm variants, GELU, helpers |
| `patchtst_engine.go` | 825 | Engine f64: forwardWithCacheEngine, forwardBatchEngine, trainWindowedEngine, trainWindowedCPU, matMulEngine, linearF64Engine |
| `patchtst_backward_engine.go` | 412 | Engine f64: backwardBatchEngine (linearBackward via engine MatMul) |
| `patchtst_gpu_train.go` | 1,452 | GPU f32: FULL reimplementation of forward+backward in float32 with tensor ops, layerNormF32WithCache, layerNormBackwardF32, geluDerivF32, AdamW, grad clip, writeBack |

The forward pass of the PatchTST transformer encoder is implemented **5 times**:
1. `patchtst.go:Forward` -- tensor-based inference
2. `patchtst_backward.go:forwardF64WithCache` -- CPU f64 training
3. `patchtst_engine.go:forwardF64WithCacheEngine` -- engine f64 single-sample
4. `patchtst_engine.go:forwardBatchF64WithCacheEngine` -- engine f64 batched
5. `patchtst_gpu_train.go:trainWindowedGPU` (forward section) -- GPU f32 batched

The backward pass is implemented **3 times**:
1. `patchtst_backward.go:backwardF64` -- CPU f64
2. `patchtst_backward_engine.go:backwardBatchF64Engine` -- engine f64 batched
3. `patchtst_gpu_train.go:trainWindowedGPU` (backward section) -- GPU f32 batched

---

## Phase 3: Justification Analysis

### Performance-Justified Divergences

1. **PatchTST GPU training (`patchtst_gpu_train.go`)** -- The fused forward+backward pass in float32 keeps all data as tensors, avoids f64<->f32 conversion per batch, and uses engine ops (MatMul, Softmax, Transpose, Sum) end-to-end. The only CPU fallback is for layer norm and GELU on small tensors where kernel launch overhead exceeds compute. This is a legitimate performance optimization that would be hard to achieve through composition.

2. **Batched engine forward** (`forwardBatchF64WithCacheEngine`) -- Concatenates samples for single-MatMul-per-projection is a batching optimization that requires a different code path than single-sample forward.

3. **Inline softmax in attention loops** -- For tiny sequence lengths (6 patches), calling a function adds measurable overhead vs inline code.

### Historically-Justified (Technical Debt)

1. **iTransformer using different signatures** -- iTransformer was likely written before the `layerNormF64WithCache` and `linearF64` patterns were established in PatchTST. The 1D vector vs 2D matrix interface difference is accidental.

2. **TTM copying PatchTST's engine wrappers** -- `ttm_train_engine.go` was clearly copy-pasted from `patchtst_engine.go` (identical function bodies, different receivers).

3. **NHiTS bespoke engine training loop** -- Written before `TrainLoop` existed or before it supported engine-backed models.

### Unjustified Duplication

1. **7 GELU implementations** -- No performance reason. A `geluScalar[T constraints.Float](x T) T` and `geluDeriv[T constraints.Float](x T) T` in a shared file solves this.

2. **`copyMatrix` vs `deepCopy2D`** -- Identical functions, different names.

3. **`TTM.matMulEngine` = `PatchTST.matMulEngine`** -- Character-for-character copy.

4. **`TTM.linearF64Engine` = `PatchTST.linearF64Engine`** -- Character-for-character copy.

5. **3 adamState struct definitions in engine files** -- Should be shared.

6. **3 clipGradients implementations** -- Should have f32 and f64 variants in training_ops.go.

7. **TimeMixer's incompatible TrainConfig signature** -- Prevents using shared infra.

---

## Phase 4: Quantification

### Lines by Category

| Category | Duplicated Lines | Reducible Lines | Difficulty |
|---|---|---|---|
| GELU implementations (7) | 61 | 45 | Easy |
| Layer norm implementations (11) | 330 | 180 | Medium |
| MatMul engine wrapper (3) | 174 | 52 | Easy |
| Linear F64 engine (2) | 44 | 22 | Easy |
| Softmax (4) | 42 | 20 | Easy |
| AdamW optimizer (4) | 42 | 18 | Easy |
| Gradient clipping (3) | 46 | 28 | Easy |
| adamState struct (4) | 20 | 12 | Easy |
| copyMatrix/deepCopy2D | 20 | 10 | Easy |
| PatchTST forward x5 | ~2,400 | ~1,200 | Hard |
| PatchTST backward x3 | ~1,200 | ~600 | Hard |
| Type boilerplate (allParamTensors, allocGrads, writeBack) | ~350 | ~200 | Medium |
| Independent training loops | ~600 | ~300 | Medium |
| **Total** | **~5,329** | **~2,687** | |

### Duplication Density by File

| File | Lines | Estimated Duplicated | Density |
|---|---|---|---|
| `patchtst_gpu_train.go` | 1,452 | ~600 | 41% |
| `patchtst_engine.go` | 825 | ~350 | 42% |
| `patchtst_backward_engine.go` | 412 | ~150 | 36% |
| `patchtst_backward.go` | 1,104 | ~300 | 27% |
| `itransformer_backward.go` | 662 | ~120 | 18% |
| `ttm_train_engine.go` | 518 | ~200 | 39% |
| `nhits.go` | 1,157 | ~80 | 7% |

---

## Phase 5: Recommendations

### Tier 1: Easy Wins (< 1 hour each, no performance impact)

**1. Create `timeseries/math_ops.go` for shared scalar operations**
- Consolidate all GELU, GELU derivative, softmax, copyMatrix into one file
- Use Go generics: `func geluScalar[T ~float32 | ~float64](x T) T`
- Replace 7 GELU + 2 GELU-deriv + 2 matrix-copy functions
- **Lines saved: ~100**

**2. Extract shared `matMulEngine` and `linearF64Engine` free functions**
- Move from PatchTST receiver to package-level functions taking `compute.Engine[float32]`
- Delete TTM's copies entirely
- **Lines saved: ~74**

**3. Create `timeseries/adamw_f32.go` for f32 optimizer ops**
- Shared `adamStateF32`, `clipGradientsF32`, `adamWUpdateF32`
- Replace 3 adamState definitions + 2 clipGradients copies + 2 inline AdamW copies
- **Lines saved: ~80**

**4. Fix TimeMixer's TrainConfig signature**
- Change from `(windows, labels, epochs int)` to `(windows, labels, TrainConfig)`
- Allow TimeMixer to use shared training infrastructure
- **Lines saved: ~20 (enables future consolidation)**

### Tier 2: Medium Effort (2-4 hours each, requires testing)

**5. Consolidate layer norm into `timeseries/layernorm_ops.go`**
- Define `layerNormF64(x [][]float64, scale, bias []float64, dModel int) [][]float64`
- Define `layerNormF64WithCache(...)` and `layerNormBackwardF64(...)`
- Define `layerNormF32WithCache(...)` and `layerNormBackwardF32(...)`
- Replace 11 implementations with 5 canonical functions
- The per-model `Engine`-based `layerNorm` methods stay (they use different tensor APIs), but refactor to call a shared helper
- **Lines saved: ~180**

**6. Generalize engine training loop**
- Extend `TrainLoop` or create `EngineTrainLoop` that accepts a `BatchForwardBackward` callback
- Move batch iteration, MSE loss, grad clip, LR warmup, AdamW into the shared loop
- NHiTS engine, TTM, iTransformer can adopt it
- **Lines saved: ~300**

**7. Unify iTransformer's layer norm/attention signatures**
- Change iTransformer from 1D `layerNorm(x, scale, bias []float64)` to 2D `layerNormF64`
- Reuse shared softmax, GELU, copyMatrix
- **Lines saved: ~120**

### Tier 3: Hard (1-2 days, requires careful performance validation)

**8. Collapse PatchTST CPU forward into Engine forward**
- `forwardF64WithCache` and `forwardF64WithCacheEngine` differ only in linear layer calls
- Introduce a `linearFunc` callback: CPU path passes `linearF64`, engine path passes `linearF64Engine`
- **Lines saved: ~400**
- **Risk: Must verify no numerical drift between paths**

**9. Collapse PatchTST f64 backward + engine backward**
- `backwardF64` and `backwardBatchF64Engine` share the same gradient math
- The engine version batches MatMul calls; the CPU version loops over samples
- A unified implementation could accept a strategy (CPU vs engine) for the MatMul calls
- **Lines saved: ~400**
- **Risk: Engine path needs careful buffer management to avoid regressions**

**10. Do NOT touch `patchtst_gpu_train.go`**
- This file is a legitimate performance optimization that fuses forward+backward in float32
- It avoids f64<->f32 conversion and uses engine ops end-to-end
- Extracting shared functions would break the fusion and hurt throughput
- The layer norm / GELU on CPU within it IS justified (small tensors, kernel launch overhead)
- **Recommendation: Leave as-is, mark with `// perf: fused GPU training path` comment**

---

## Summary Scorecard

| Area | DRY Score | Notes |
|---|---|---|
| `layers/` | **A** | Excellent. Each op defined once with generics. |
| `inference/` | **A-** | Shared `buildTransformerGraph` + per-arch opts. Minor: `transposeWeight` has storage-type branching that could be a method. |
| `training/` | **A** | Clean interfaces, composable strategies. |
| `generate/` | **B+** | KV cache hierarchy is clean. Some duplication in quantized variants. |
| `serve/` | **A** | No DRY issues found. |
| `timeseries/` shared infra | **B** | `TrainLoop`, `TrainConfig`, `normalizeWindows` are good. Some backends don't use them. |
| `timeseries/` PatchTST | **D** | 5 forward passes, 3 backward passes, massive duplication. |
| `timeseries/` scalar ops | **F** | 7 GELUs, 11 layer norms, 3 matMul wrappers. |
| `timeseries/` optimizer infra | **C** | Shared f64 loop exists but f32 engine backends reinvent it. |

**Overall timeseries DRY score: C-**
**Overall codebase DRY score: B+** (layers/inference/training carry the grade; timeseries drags it down)

---

## Appendix: File-by-File Reference

### timeseries/ files with highest duplication density

1. `patchtst_engine.go` (825 lines) -- 42% duplicated. Contains 3 training loops (engine, CPU, batch).
2. `patchtst_gpu_train.go` (1,452 lines) -- 41% duplicated BUT performance-justified. layerNormF32WithCache/Backward, geluDerivF32, inline AdamW are the unjustified parts.
3. `ttm_train_engine.go` (518 lines) -- 39% duplicated. matMulEngine and linearF64Engine are exact PatchTST copies.
4. `patchtst_backward_engine.go` (412 lines) -- 36% duplicated. linearBackwardF64EngineAccum reimplements what could be composed from shared backward ops.
5. `patchtst_backward.go` (1,104 lines) -- 27% duplicated. Contains canonical f64 implementations that SHOULD be the shared source, but layerNormF64, geluF64, linearF64 need to be extracted to a shared file.

### layers/ files -- exemplary composition

- `layers/core/ffn.go` composes from `Dense` + `SwiGLU`
- `layers/attention/grouped_query_attention.go` composes from `ScaledDotProductAttention` + `Linear`
- `layers/transformer/block.go` composes from `attention` + `normalization` + `core`
- `layers/normalization/registry.go` provides a factory pattern for norm type selection

These demonstrate the design principle at work: components built from smaller components.
