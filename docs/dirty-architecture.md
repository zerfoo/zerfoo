# Dirty Architecture: Composability Violations Audit

*Updated: 2026-04-03 (revision 3)*
*Previous: 2026-04-03 (revision 2), 2026-04-02 (revision 1)*

## Design Principle Under Review

From `docs/design.md` line 285:

> Complex components must be built by composing smaller components. The `layers/`
> package provides 56+ neural network operations. The `compute.Engine[T]` interface
> provides type-safe tensor arithmetic. Models should compose these building blocks
> rather than reimplementing low-level math.

The exemplar is `inference/arch_common.go`, which composes
`layers/attention.GroupedQueryAttention`, `layers/normalization.RMSNorm`,
`layers/embeddings.RotaryPositionalEmbedding`, and `layers/core.Linear` into a
computation graph that enables CUDA graph capture, fusion passes, and megakernel
codegen.

This audit identifies every place in the codebase where this principle is violated,
categorizes each violation as justified or unjustified, and quantifies the scope.

---

## Summary

| Metric | Rev 1 (Apr 2) | Rev 2 (Apr 3) | Rev 3 (Apr 3) | Delta (2->3) |
|--------|---------------|---------------|---------------|--------------|
| Source files reviewed (non-test .go) | 860 | 10,373 | 10,373 | -- |
| Total lines of code | 170,860 | 77,720 | 77,720 | recount scope |
| Packages with composability violations | 14 | 8 | 8 | -- |
| Additional packages with structural violations | -- | -- | 3 | +3 (NEW) |
| Reimplemented LayerNorm instances | 17 | 10 | 10 | -- |
| Reimplemented Linear/MatMul/MLP instances | 18 | 11 | 11 | -- |
| Reimplemented softmax instances | 6 | 4 | 4 | -- |
| Reimplemented GELU instances | 5 | 4 | 4 | -- |
| Reimplemented AdamW instances | 5 | 1 | 1 | -- |
| Reimplemented SGD instances | 5+ | 0 | 0 | -- |
| Reimplemented attention instances | 8 | 6 | 6 | -- |
| Intra-layers/ violations | 10 | 14 | 14 | -- |
| Total .Data() calls (non-test) | -- | -- | ~2,599 | NEW metric |
| Unjustified .Data() calls | -- | -- | ~805 | NEW metric |
| Monolithic functions (>150 lines) | -- | -- | 8 | NEW metric |
| Duplicated code patterns (cross-file) | -- | -- | 12 | NEW metric |
| Estimated redundant lines | ~14,000 | ~8,200 | ~9,800 | +1,600 (broader scope) |

**What changed since revision 2:**

- No new code migrations. Revision 3 is a **deeper audit**, not a code change.
- **Broader scope:** Now covers structural composition violations beyond
  operation reimplementation. Three packages (generate/, serve/, cmd/) have
  significant internal duplication and monolithic functions that violate
  composition at the architectural level.
- **New metrics:** .Data() bypass inventory (2,599 total, 805 unjustified),
  monolithic function count, cross-file duplication patterns.
- **Backward pass analysis:** Documented the two-tier backward architecture
  (functional ops vs monolithic model backward) and the 2,180 lines of
  monolithic backward code in timeseries/.
- **generate/ deep dive:** Identified 4 duplicated autoregressive loops and
  monolithic Generate/sampleFromLogits functions.
- **serve/ deep dive:** Identified duplicated handler logic and monolithic
  request processing.
- **Inference builder boilerplate:** 6 patterns duplicated across 25-30 files.

**Bottom line:** The inference path follows the composability principle well. Five
of the eight S1 violators from revision 1 have been migrated. The remaining
operation-level violations are concentrated in three packages: **timeseries/**
(17,570 lines), **tabular/** (4,010 lines), and **modeldsl/** (1,520 lines).
Beyond operation reimplementation, **generate/** and **serve/** have significant
structural composition violations (monolithic functions, duplicated loops).
The `layers/` package itself has 14 internal violations where sub-packages
bypass `Engine[T]` or duplicate each other.

---

## Severity Tiers

- **S1 (Critical):** Package operates entirely on raw `[]float64` or `[]float32`
  slices, reimplementing attention, normalization, linear layers, optimizers, and
  loss functions from scratch. No use of `Engine[T]` or `layers/`. These packages
  cannot benefit from GPU acceleration, CUDA graph capture, megakernel codegen, or
  any future engine backend.

- **S2 (Major):** Package uses `Engine[T]` for some operations but reimplements
  others inline (e.g., manual `tensor.Data()` access for LayerNorm while using
  `engine.MatMul` for projections). Partial bypass of the composition system.

- **S2.5 (Structural):** Package does not reimplement operations, but violates
  composition through monolithic functions, duplicated code patterns, or missing
  abstractions. These are not Engine bypass violations but architectural violations
  of the "build complex from simple" principle.

- **S3 (Minor):** Package composes from `layers/` but has isolated custom nodes for
  architecture-specific operations. These are often justified by unique structural
  requirements (e.g., Falcon's parallel attention+FFN, RWKV's linear attention).

---

## S1: Critical Violations (Complete Bypass)

### 1. timeseries/ -- Parallel ML Framework (17,570 lines)

**Files:** 35 non-test source files including `math_ops.go`, `layernorm_ops.go`,
`training_ops.go`, `adamw_f32.go`, plus model files and `*_backward.go` files.

The top-level `timeseries/` package is effectively a separate ML framework that
shares minimal code with `layers/`. It reimplements:

| Component | timeseries/ file:line | Canonical |
|-----------|----------------------|-----------|
| LayerNorm (2D+cache) | layernorm_ops.go:46-92 | `layers/normalization.LayerNorm` |
| LayerNorm (engine reimpl) | patchtst_encoder.go:21-91 | `layers/normalization.LayerNorm` |
| LayerNorm backward | patchtst_encoder.go:94-195 | `layers/normalization.LayerNorm.Backward` |
| LayerNorm (1D cached) | itransformer_backward.go:233-242 | `layers/normalization.LayerNorm` |
| GELU | itransformer_backward.go:250-254 | `layers/activations.Gelu` |
| GELU (engine reimpl) | patchtst_encoder.go:411-448 | `layers/activations.Gelu` |
| Softmax (raw f64) | itransformer_backward.go:638-655 | `layers/activations.Softmax` |
| AdamW state | training_ops.go:26 | `training/optimizer.AdamW[T]` |
| AdamW update | training_ops.go:41 | `training/optimizer.AdamW[T].Step` |
| AdamW (float32) | adamw_f32.go:30 | `training/optimizer.AdamW[T]` |
| MSE loss | training_ops.go:57 | `training/loss.MSE[T]` |
| Gradient clipping | training_ops.go:8 | (should be in training/) |
| Multi-head attention (f64) | itransformer_backward.go:185-212 | `layers/attention.GroupedQueryAttention` |
| Attention backward (f64) | itransformer_backward.go:1037-1093 | `layers/attention.GroupedQueryAttention` |
| Linear backward (f64) | patchtst_backward.go:405-430 | `layers/core.Linear` |
| Linear backward (f64) | itransformer_backward.go:345-361 | `layers/core.Linear` |
| MLP forward (raw) | timemixer_backward.go:196-223 | `layers/core.Linear` + `layers/core.FFN` |
| Encoder backward (raw) | patchtst_encoder.go:938-1120 | `graph.Backward()` |
| Mixing backward (raw) | timemixer_backward.go:278-400+ | `graph.Backward()` |

**Composition status per model:**

| Model | Imports layers/? | Uses Engine[T]? | Forward | Backward |
|-------|-----------------|-----------------|---------|----------|
| PatchTST | No | Partial (encoder) | Mixed | Raw f64 loops |
| iTransformer | `functional` (partial) | Declared, unused in backward | Mixed | Raw f64 loops |
| N-HiTS | `functional.Linear` | Partial | Composed | Custom backward |
| N-BEATS | `functional.Linear` | Partial | Composed | Raw loops |
| TTM | `functional` (partial) | Partial | Mixed | Raw f64 loops |
| DLinear | No | Optional, unused | Raw f64 | Raw f64 |
| TimeMixer | No | Optional, unused | Raw f64 | Raw f64 |
| CfC | No | Optional, unused | Raw f64 | Raw f64 |
| FreTS | No | Optional, unused | Raw f64 | Raw f64 |
| Mamba | `layers/ssm`, `layers/core` | Yes | Composed | Composed |

**Backward pass analysis (NEW in rev 3):**

The framework provides composable functional backward operations in
`layers/functional/`: `LinearBackward`, `GELUBackward`, `LayerNormBackward`,
`SoftmaxBackward`, `MultiHeadAttentionBackward`, `MLPBackward`. These compose
from engine operations (MatMul, Transpose, ReduceSum, Mul, Sub, etc.).

The timeseries backward passes ignore all of these:

| File | Lines | Approach | Uses functional backward? |
|------|-------|----------|--------------------------|
| patchtst_backward.go | 825 | Monolithic raw f64 loops | No |
| itransformer_backward.go | 723 | Monolithic with heavy caching | No |
| timemixer_backward.go | 632 | Monolithic with manual gradient accumulation | No |
| **Total** | **2,180** | | |

Each backward file maintains its own cache struct, recomputes forward statistics
inline, and implements gradient formulas from scratch. The `layers/functional`
backward ops exist specifically to replace this pattern.

**.Data() violation count:** 254 calls across 23 files.

**Impact:** 17,570 lines (23% of non-test codebase) exist in a parallel universe.
10 model architectures each carry their own math. Bug fixes to canonical layers
do not propagate here. GPU acceleration requires duplicate effort for each model.
4 models (DLinear, TimeMixer, CfC, FreTS) declare optional engine fields that are
never used for training.

**Justified?** No. Mamba proves that full composition is feasible. The `_engine.go`
files prove incremental migration works but was never completed.

**Estimated redundant lines:** ~4,500


### 2. tabular/ -- Another Parallel Framework (4,010 lines)

**Files:** `ft_transformer.go`, `saint.go`, `tabnet.go`, `resnet.go`, `model.go`,
`train.go`, `lora.go`, `pretrain.go`, `save.go`, `ensemble.go`

**Status:** No changes since revision 1. Zero imports from `layers/functional`.

Reimplements:

| Component | tabular/ file:line | Canonical |
|-----------|-------------------|-----------|
| LayerNorm | ft_transformer.go:457-493 | `layers/normalization.LayerNorm` |
| LayerNorm | resnet.go:232-280 | `layers/normalization.LayerNorm` |
| LayerNorm | saint.go:653-691 | `layers/normalization.LayerNorm` |
| Multi-head attention | ft_transformer.go:340-399 | `layers/attention.GroupedQueryAttention` |
| Multi-head attention | saint.go (self-attn + intersample) | `layers/attention` |
| GELU | model.go:212-216 (`geluScalar`) | `layers/activations.Gelu` |
| Linear forward | ft_transformer.go:496-502 | `layers/core.Linear` |
| Linear forward | saint.go:745-751 | `layers/core.Linear` |
| Linear forward | tabnet.go:318-324 | `layers/core.Linear` |
| Linear forward | resnet.go:282-288 | `layers/core.Linear` |
| Linear forward | model.go:191-197 | `layers/core.Linear` |
| Custom loss | train.go:255 (`crossEntropyLoss`) | `training/loss.CrossEntropyLoss` |

**Mixed pattern:** tabular/ uses `compute.Engine[T]` for MatMul and some ops,
but reimplements LayerNorm and GELU inline with `tensor.Data()` access.

**.Data() violation count:** 51 calls.

**Impact:** 5 model architectures (FT-Transformer, SAINT, TabNet, TabResNet,
base Model) each carry duplicate linearForward/layerNorm methods. The
`linearForward` function appears 5 times with nearly identical code.

**Justified?** No. These are standard transformer and MLP architectures.
The `engine.MatMul` usage proves engine integration is possible.

**Estimated redundant lines:** ~1,600


### 3. modeldsl/ -- Custom Layer Type System (1,520 lines)

**Files:** `dsl.go`, `model.go`, `graph.go`, `optimize.go`, `train.go`

Reimplements:

| Component | modeldsl/ file:line | Canonical |
|-----------|---------------------|-----------|
| Linear layer | model.go:108 (`linearLayer`) | `layers/core.Linear` |
| RMSNorm | model.go (via `rmsnormLayerT`) | `layers/normalization.RMSNorm` |
| SiLU | model.go (via `siluLayerT`) | `layers/activations.SwiGLU` |
| Softmax | model.go (via `softmaxLayerT`) | `layers/activations.Softmax` |
| Attention | model.go:166-210 (`attentionLayer`) | `layers/attention` |
| Xavier init | model.go:117 | `layers/components.WeightInitializer` |

The DSL defines `LayerType` constants (`LayerLinear`, `LayerRMSNorm`, `LayerSiLU`,
`LayerSoftmax`, `LayerAttention`) that duplicate the layer registry in
`layers/registry/`.

**.Data() violation count:** 0 (uses its own abstraction layer).

**Justified?** Partially. The DSL is a model definition language that compiles
user-provided layer definitions. The custom implementations are encapsulated within
modeldsl/ and not imported by other packages. However, the DSL should ideally
produce graphs that compose from existing `layers/` implementations rather than
reimplementing them from scratch on raw `[]float64` slices.

**Estimated redundant lines:** ~600

---

## S1 Packages Migrated Since Revision 1

These packages were S1 violations in revision 1 and have been migrated:

| Package | Lines | Migration | Date | Task |
|---------|-------|-----------|------|------|
| crossasset/ | 1,407 | `layers/functional` + `optimizer.AdamW[float64]` | Apr 3 | T68.1.1-T68.1.3 |
| rl/ | 1,293 | `layers/functional` + `optimizer.SGD` | Apr 3 | T71.1.1 |
| synth/ | 615 | `layers/functional` + `optimizer.SGD` | Apr 3 | T71.1.2 |
| meta/ | 467 | `layers/functional` | Apr 3 | T71.1.3 |
| shared/ | 293 | `engine.MatMul` | Apr 3 | T71.1.4 |

**Note on crossasset/:** The cpu forward path now uses `functional.LayerNorm` at
crossasset.go:603. The backward path uses graph-based differentiation via
`layers/core.Linear` and `layers/attention.ScaledDotProductAttention`. The
canonical `optimizer.AdamW[float64]` replaces the custom `adamWUpdateAll`. 1,357
lines of dead GPU training infrastructure were deleted.

**Note on gnn/:** The GNN package uses custom matrix algebra (`gcn.go:239
matMul`, `gcn.go:278 reluMatrix`, `gcn.go:300 softmaxMatrix`) for graph
convolution operations. These are graph-structure-specific operations that cannot
trivially compose from standard layers. Reclassified from S1 to S3.

---

## S1.5: Violations Within layers/ Itself

The `layers/` package, which is meant to be the canonical source of composable
primitives, has internal violations where sub-packages bypass each other or
bypass `Engine[T]`.

### vision/clip_encoder.go -- Raw-Loop Multi-Head Attention

**File:** `layers/vision/clip_encoder.go`

| Lines | Violation | Should use |
|-------|-----------|-----------|
| 232-251 | 6-level nested raw loop for patch extraction via `.Data()` | engine.Reshape or Conv2d |
| 274-287 | Manual class token concatenation via `.Data()` | engine.Concat |
| 290-295 | Position embedding addition via raw for loop + `.Data()` | engine.Add with broadcasting |
| 362-410 | QK^T scores, softmax, weighted V sum on raw data (~50 lines) | `attention.ScaledDotProductAttention` |
| 255-270 | Linear projection as raw triple-nested loop | `engine.MatMul` |
| 503-536 | Inline QuickGELU via raw ops | `layers/activations.Gelu` |

This is the single worst violator within `layers/` itself: ~200 lines of raw CPU
loops implementing operations that exist as composable engine ops.

### timeseries/mlstm.go -- All-CPU Forward Pass with .Data() Access

**File:** `layers/timeseries/mlstm.go`

| Lines | Violation | Should use |
|-------|-----------|-----------|
| 248, 261, 274 | Scalar bias via `.Data()[0]` extraction | engine.AddScalar or tensor wrapping |
| 322, 326, 342, 375, 386 | Tensor slicing via `.Data()` indexing | engine.Slice |
| 338-339 | Scalar gate extraction via `.Data()` | engine.Index |
| 320-402 | Sequential batch-element loop (~80 lines of manual tensor manipulation) | Vectorized engine ops |

### timeseries/slstm.go -- Custom Gate Computation

**File:** `layers/timeseries/slstm.go:313-316`

Inline clamp + exp via UnaryOp. While using engine.UnaryOp (correct), this is
hand-rolled clamping + exponential that could be a shared primitive.

### timeseries/ssm.go -- All-CPU Discretization + Sequential Loop

**File:** `layers/timeseries/ssm.go`

| Lines | Violation | Should use |
|-------|-----------|-----------|
| 188-217 | Raw CPU loops for discretization (exp, mul) | engine.Exp, engine.Mul |
| 245-302 | Sequential batch/time scan creating new tensors per timestep | Vectorized scan |
| 300 | Output assembly via `copy(outputData[...], y.Data())` | engine.Concat |

### timeseries/vsn.go -- Manual Weight Extraction

**File:** `layers/timeseries/vsn.go`

| Lines | Violation | Should use |
|-------|-----------|-----------|
| 342-346, 417-420 | Weight column extraction via `.Data()` indexing | engine.Slice |
| 364-373 | Manual importance weight mean via raw loops | engine.ReduceMean |

### core/gemm.go -- Hand-Rolled GEMM

**File:** `layers/core/gemm.go:66-86`

Triple-nested `for i/j/k` loop over raw `aData[]/bData[]` slices for matrix
multiply. Never hits GPU, SIMD, CUDA graph, or any engine backend.

Should use: `engine.MatMul` + `engine.MulScalar` + `engine.Add`

### core/variable_selection.go -- Inline GELU

**File:** `layers/core/variable_selection.go:145-146`

```go
return v.ops.Mul(x, v.ops.Sigmoid(v.ops.Mul(x, v.ops.FromFloat64(1.702))))
```

Computes `GELU(x) ~ x * sigmoid(1.702 * x)` inline. Duplicates the FastGELU
approximation from `activations/fast_gelu.go`.

Should use: `layers/activations.Gelu` or `layers/activations.FastGelu`

### core/temporal_conv_encoder.go -- Manual Pool Gradient

**File:** `layers/core/temporal_conv_encoder.go:159-168`

Manual backward pass for global average pool with raw `.Data()` loops instead of
engine broadcast/repeat operations.

### normalization/simplified_layer_normalization.go -- Shares rmsNormalize

**Status:** RESOLVED. This file correctly delegates to the shared `rmsNormalize`
helper. Not a violation. (Corrected from revision 1.)

### activations/fast_gelu.go -- Properly Delegates

**Status:** RESOLVED. Line 23 shows proper delegation to Gelu. Not a duplication.
(Corrected from revision 1.)

### residual/block_attn_res.go -- Clean

**Status:** RESOLVED. Properly uses engine operations throughout. No `.Data()`
access. (Corrected from revision 1.)

---

## S2: Major Violations (Partial Bypass)

### 10. inference/ Architecture Builders -- Custom Nodes

The architecture builders (30+ files, `arch_*.go`) compose from `layers/` for
the common pattern (see `arch_common.go`) but create custom `graph.Node[T]`
implementations for architecture-specific operations:

**Unjustified custom nodes (should compose from layers/):**

| Architecture | Node | Lines | Issue |
|-------------|------|-------|-------|
| Falcon | `falconGeluFFN` | ~40 | Standard FFN with GELU; use `layers/core.FFN` |
| BERT | `bertResidualLayerNormNode` | ~40 | Add + LayerNorm; compose from existing layers |
| BERT | `bertFFNNode` | ~40 | Standard FFN; use `layers/core.FFN` |
| BERT | `bertEmbeddingNode` | ~80 | Token+pos+segment embedding with raw `.Data()` at lines 324, 332-334 |
| GPT-2 | `gpt2ResidualAddNode` | ~40 | Simple residual add; use `layers/core.Add` |
| GPT-2 | `gpt2EmbeddingNode` | ~60 | Token+pos embedding with `.Data()` at lines 411, 423-424 |
| Command R | `commandRResidualAddNode` | ~40 | Simple residual add |
| LLaVA | `llamaAttnNode` (arch_vision_helpers.go:17-132) | ~115 | Complete custom GQA with nested `.Data()` loops |
| LLaVA | `llamaFFNNode` (arch_vision_helpers.go:135-189) | ~55 | Custom SwiGLU with inline sigmoid via `.Data()` |
| LLaVA | `rmsNormWrapNode` (arch_vision_helpers.go:223-260) | ~40 | Custom RMSNorm with raw `.Data()` normalization |

**Justified custom nodes (unique architecture requirements):**

| Architecture | Node | Justification |
|-------------|------|---------------|
| Fused ops | `fusedAddRMSNormNode`, `fusedNormAddNode` | Performance-justified fusions (single kernel) |
| RWKV | `rwkvTimeMixNode`, `rwkvChannelMixNode` | Fundamentally different recurrence (linear attention) |
| Kimi | `kimiLinearAttentionNode` | Linear attention with ELU+1 feature map |
| Mamba | `mambaResidualAddNode` | SSM-specific residual pattern |
| DeepSeek | `deepSeekReshapeNode`, `deepSeekConstNode` | MLA-specific tensor manipulation |

**Total custom nodes:** ~50 types, ~1,660 lines
**Unjustified:** ~550 lines across 10 node types
**Justified:** ~1,110 lines across ~40 node types

The worst offender is `arch_vision_helpers.go` which contains 3 custom nodes
(llamaAttnNode, llamaFFNNode, rmsNormWrapNode) totaling ~210 lines of raw
`.Data()` loops that completely bypass the engine.

**Inference builder boilerplate duplication (NEW in rev 3):**

Six patterns are duplicated across 25-30 architecture builder files:

| Pattern | Appears in | Lines each | Total duplication |
|---------|-----------|------------|-------------------|
| Tensor lookup wrapper `func(name string)` | ~30 files | ~7 | ~210 |
| Param wrapper `func(name, tensor) *Parameter` | ~30 files | ~4 | ~120 |
| Weight transpose logic (6 storage type branches) | ~6 files | ~100 | ~600 |
| Embedding lookup node instantiation | ~25 files | ~3 | ~75 |
| LM head node instantiation | ~25 files | ~3 | ~75 |
| Residual add setup (fusedAddRMSNorm + residualAdd) | ~15 files | ~5 | ~75 |

**Total boilerplate:** ~1,155 lines of identical patterns that should be
extracted to factory functions or a builder helper module.


### 11. inference/timeseries/ -- Custom Nodes with .Data() Access

**Files:** 20 files, `arch_*.go` pattern

Mixed composition: some builders compose from `layers/` (arch_ttm.go,
arch_flowstate.go, arch_patchtst.go), others create custom nodes with extensive
`.Data()` access:

| File | Lines with .Data() | Issue |
|------|-------------------|-------|
| arch_timemixer.go | 335, 337, 389, 396-399, 446, 464, 482 | 10+ .Data() calls; inline softmax at line 507 |
| arch_tft.go | 139, 314, 494, 519 | Custom temporal fusion with inline LSTM |
| arch_chronos.go | 871 | Tokenization node |
| arch_ttm.go | 522, 572, 591, 607, 727 | Custom normalization and output assembly |
| arch_tirex.go | 376, 580 | Custom projection |
| arch_tspulse.go | 271, 350, 401 | Probability/semantic extraction |
| arch_flowstate.go | 360, 382 | Feature extraction |
| arch_regime.go | 423, 436 | Regime detection |

**Estimated .Data() bypass lines:** ~2,000

**Monolithic node architecture:** Time-series inference builders construct a
single large Node that implements Forward() directly (e.g., `chronosNode` with
full T5 encoder-decoder, `moiraiNode` with frequency embeddings). These nodes
contain internal layer objects (RMSNorm, Linear) but don't expose them as
graph nodes, preventing CUDA graph capture and fusion.


### 12. generate/ -- Monolithic Functions and Duplicated Loops (NEW in rev 3)

**Files:** 56 non-test files, ~12,000 lines

The generate package has good foundational abstractions (CacheProvider[T]
interface with 5+ implementations, options pattern, sampling primitives) but
violates composition through monolithic functions and duplicated code.

**Monolithic functions:**

| Function | File:lines | Lines | Issue |
|----------|-----------|-------|-------|
| `Generate()` | generator.go:366-585 | 220 | Cache selection (8 branches), autoregressive loop, grammar advancement, GPU counter sync all in one function |
| `sampleFromLogits()` | generator.go:589-718 | 130 | GPU argmax fast path, logit conversion, temperature/TopK/TopP, grammar masking, repetition penalty all inline |
| `GenerateStream()` | stream.go:36-195 | 160 | Duplicates Generate() loop with streaming |

**Duplicated autoregressive loops (4 implementations):**

| File | Function | Overlap with Generate() |
|------|----------|------------------------|
| generator.go | `Generate()` | -- (baseline) |
| stream.go | `GenerateStream()` | ~80% identical |
| speculative.go | `generateDraft()` + `generateWithVerification()` | ~60% identical |
| eagle_speculative.go | (variant) | ~60% identical |

Each reimplements the decode loop with ~80% identical logic: cache setup,
token feeding, sampling, stop condition checking, grammar advancement.

**Duplicated cache selection:**

The 8-branch if-else chain for cache provider selection appears twice:
- `generator.go:404-430`
- `stream.go:59-83`

Should be: `selectCacheProvider(cfg, blockPool, engine) CacheProvider[T]`

**Engine bypass (tensor .Data() access):**

| File:line | Violation |
|-----------|-----------|
| generator.go:635 | `logits.Data()` bypasses Engine for CPU sampling |
| generator.go:631 | `gs.CopyTo(data)` assumes GPUStorage implementation |
| generator.go:605 | `logits.GetStorage().(*tensor.GPUStorage[T])` type assertion |

**KV cache structural duplication:**

Five KV cache implementations (`kvcache.go`, `kvcache_fp16.go`, `kvcache_fp8.go`,
`kvcache_q3.go`, `kvcache_q4.go`, 1,235 lines total) with the same interface
(`Get`, `Set`, `Len`, `MaxSeqLen`) but different quantization. Structural code
(layer indexing, sequence management, resize logic) is duplicated across all five.

**Justified?** Partially for KV caches (quantization math differs). Not justified
for the duplicated autoregressive loops or monolithic functions.

**Estimated redundant lines:** ~800 (autoregressive loops ~400, cache selection ~50,
KV cache structural ~350)


### 13. training/loss/ -- Engine Fields Unused

| Loss | File | Lines | Issue |
|------|------|-------|-------|
| `BCELoss` | training/loss/bce.go | 156 | Has `engine` field (line 18), uses only raw `.Data()` for log/mul/add |
| `RoutingContrastive` | training/loss/routing_contrastive.go | 224 | Triple-nested raw loops for cosine similarity; `.Data()` at lines 65, 91, 108, 118, 152, 188, 211, 280-281 |
| `QuantileLoss` | training/loss/quantile.go | 92 | Hardcoded `float32` casts via `any(...).(float32)` despite generic `[T]`; panics for non-float32 |
| `SharpeLoss` | training/loss/quantile.go:83-152 | ~70 | Raw `.Data()` iteration with manual softmax + portfolio computation |

`BCELoss` is the clearest violation: it stores `engine` but its `Forward` and
`Backward` methods use only raw ops. The computation (log, mul, add, sub, div)
maps directly to engine ops.

`QuantileLoss` breaks generics entirely with `any(targetData[i]).(float32)` --
this panics at runtime for any type other than `float32`.


### 14. training/optimizer/ -- Raw-Slice Gradient Operations

| Method | File:line | Issue |
|--------|-----------|-------|
| `AdamW8bit.Step` | training/optimizer/adamw8bit.go:140-178 | Entire 8-bit Adam update on raw `.Data()` slices; element-by-element loop at lines 155-168 |
| `guardAndClipGradients` | training/optimizer/adamw.go:239-300 | Uses engine ops (ReduceSum, Mul) with minimal `.Data()` for final scalars -- ACCEPTABLE after re-review |

The `AdamW8bit` optimizer is the remaining violation: the full parameter update
loop at lines 155-168 iterates element-by-element instead of using vectorized
engine ops, completely defeating GPU acceleration.

---

## S2.5: Structural Composition Violations (NEW in rev 3)

These packages don't reimplement neural network operations but violate the
composition principle through monolithic design and code duplication.

### 15. serve/ -- Monolithic Handlers (NEW)

**Files:** 33 non-test files

**Good patterns:** Options pattern for Server (WithBatchScheduler, WithDraftModel,
WithAPIKey), pluggable optional features (Transcriber, Classifier, GuardEvaluator),
clean BatchScheduler abstraction. No direct tensor access.

**Violations:**

| Function | File:lines | Lines | Issue |
|----------|-----------|-------|-------|
| `handleChatCompletions()` | handlers.go:18-218 | 200 | HTTP validation, sampling param validation, grammar conversion, image fetching, batch/direct decision, tool call detection, response formatting -- all in one function |
| `handleCompletions()` | handlers.go:220-324 | 105 | Duplicates ~60% of handleChatCompletions: sampling validation, adapter parsing, option building, batch/direct path |

Both handlers duplicate:
- Sampling parameter validation
- Adapter field parsing from model name
- `inference.GenerateOption` slice construction
- Batch vs direct generation branching

Should extract: `buildGenerationOptions(request) ([]inference.GenerateOption, error)`

Grammar/tool integration is tightly coupled into the handler body:
- handlers.go:96-110: Grammar JSON schema conversion (framework-specific, not pluggable)
- handlers.go:170-203: Tool call detection (custom business logic, not pluggable)

**Estimated redundant lines:** ~200


### 16. cmd/ -- No Shared Bootstrap (NEW)

**Structure:** 20+ command subdirectories: `cli/`, `zerfoo/`, `zerfoo-edge/`,
`zerfoo-predict/`, `finetune/`, `train_distributed/`, `ts_train/`, plus 7
benchmark variants (`bench/`, `bench_batch/`, `bench_disagg/`, `bench_mamba/`,
`bench_prefix/`, `bench_spec/`, `bench_tps/`).

Each command directory likely duplicates:
- Model loading boilerplate
- Configuration parsing
- Generator creation
- Server initialization
- Logger setup

No shared bootstrap library exists. Should extract: `cmd/bootstrap/` with
`LoadModel()`, `CreateGenerator()`, `CreateServer()`, `ParseConfig()`.

**Estimated redundant lines:** ~400 (across 20+ directories)


### 17. distributed/ -- Interface Leakage (NEW)

**Files:** 20 non-test files. Good patterns overall (Strategy interface,
two swappable implementations, separated concerns).

**Violations:**

| Issue | Location | Impact |
|-------|----------|--------|
| Protobuf types in public API | grpc_strategy.go, NetworkManager | `pb.DistributedServiceClient` leaks into public interface |
| WorkerNode does too much | worker_node.go:56-94 | Creates strategy, registers, manages server, registers health -- should use builder pattern |
| AllReduceGradients hardcoded to sum | interfaces.go:19 | No way to compose different reduction operations (max, min, avg) |

---

## S3: Minor Violations (Isolated Custom Ops)

### Justified Performance Fusions

These bypass `layers/` for documented performance reasons (see ADR-027):

| Component | Location | Justification |
|-----------|----------|---------------|
| Fused dequant+GEMV Q4_K | internal/cuda/kernels/ | Single kernel vs 2 launches |
| Fused QK RMSNorm+RoPE | internal/cuda/kernels/ | Single kernel vs 3 launches |
| Fused SwiGLU | internal/cuda/kernels/ | Single kernel vs 2 launches |
| Fused Scale+Softmax | internal/cuda/kernels/ | Single kernel vs 2 launches |
| ARM NEON SIMD assembly | internal/xblas/ | Platform-specific optimization |
| Megakernel codegen | internal/codegen/ | Whole-graph kernel fusion |

These are the correct pattern: performance-critical hot paths where composition
adds measurable overhead.

### GNN Package (Reclassified from S1 to S3)

**Files:** `gcn.go`, `gat.go` (639 lines)

Uses custom matrix algebra (`matMul`, `reluMatrix`, `softmaxMatrix`,
`xavierMatrix`) for graph convolution operations. Graph neighborhood aggregation
and adjacency normalization are operations that cannot trivially compose from
standard neural network layers. A declared `var cpuEngine` at gcn.go:15 is
unused.

**Justified?** Partially. The graph-specific operations (adjacency normalization,
sparse neighborhood aggregation) are legitimately novel. The `matMul` and
`reluMatrix` helpers could still delegate to engine ops.

### Legitimate Standalone Packages

These packages do not use `Engine[T]` or `layers/` but are not violations
because their functionality is outside the neural network domain:

| Package | Domain | Why no layers/ needed |
|---------|--------|---------------------|
| `causal/` | Statistical causal inference (PC algorithm) | Graph algorithms, not NN ops |
| `gp/` | Genetic programming (expression trees) | Symbolic computation |
| `federated/` | Federated learning coordination | Weight averaging, no model ops |
| `features/` | Time-series feature engineering (lag, rolling, FFT) | Preprocessing |
| `data/` | Dataset container | Data loading |
| `config/` | JSON config loading | Infrastructure |
| `health/` | HTTP health checks | Infrastructure |
| `shutdown/` | Graceful shutdown | Infrastructure |
| `compliance/` | Audit logging | Infrastructure |
| `security/` | API keys, encryption | Infrastructure |
| `provenance/` | Model lineage tracking | Metadata |
| `monitor/` | Drift detection | Statistical tests |
| `recover/` | Retraining triggers | Orchestration |
| `regime/` | HMM regime detection | Statistical model |
| `serve/` | HTTP API server | Composes from inference/generate |
| `distributed/` | gRPC gradient exchange | Network protocol |
| `registry/` | Model registry | File management |

---

## .Data() Bypass Inventory (NEW in rev 3)

Total `.Data()` calls across production code (non-test): **~2,599**

| Package | Calls | Status |
|---------|-------|--------|
| layers/core/ | ~1,376 | **Mostly justified** (kernel implementations) |
| inference/ | ~272 | VIOLATIONS (custom nodes, vision helpers) |
| timeseries/ | ~254 | VIOLATIONS (raw f64 training loops) |
| training/ | ~205 | VIOLATIONS (loss, optimizer bypass) |
| generate/ | ~70 | VIOLATIONS (sampling, KV cache) |
| tabular/ | ~51 | VIOLATIONS (layernorm, linear inline) |
| rl/ | ~26 | VIOLATIONS (policy gradient, parameter copy) |
| distributed/ | ~20 | VIOLATIONS (batch handling) |
| model/ | ~15 | Minor violations |
| crossasset/ | ~12 | Moderate (backward gradient accumulation) |
| Other | ~17 | Minor |
| **Unjustified total** | **~805** | Blocks GPU acceleration and CUDA graph capture |

**Critical .Data() patterns to eliminate:**

```go
// Pattern 1: Parameter copying (rl/, crossasset/)
// OLD:
ls := make([]float64, len(p.Value.Data()))
copy(ls, p.Value.Data())
// NEW: Use engine.Copy() or tensor.Clone()

// Pattern 2: Returning raw slices (rl/, generate/)
// OLD:
return out.Data(), nil
// NEW: Return tensor reference, let caller decide

// Pattern 3: Scalar extraction (training/loss/, training/optimizer/)
// OLD:
val := tensor.Data()[i]  // Array indexing
// ACCEPTABLE ONLY FOR: val := reductionResult.Data()[0]  // Single scalar from ReduceSum

// Pattern 4: Data layout reshaping (inference/, timeseries/)
// OLD:
qData := q.Data()
// [manual reshape with slice indexing]
// NEW: engine.Reshape(ctx, q, newShape)
```

---

## Backward Pass Composition Analysis (NEW in rev 3)

### Two-Tier Architecture

The framework has a two-tier backward pass architecture:

**Tier 1: Functional Backward Operations** (`layers/functional/`)

Composable, engine-based gradient computations:

| Operation | File | Composes from |
|-----------|------|---------------|
| `LinearBackward[T]` | linear_backward.go | engine.MatMul, engine.Transpose, engine.ReduceSum |
| `GELUBackward[T]` | gelu_backward.go | engine.Mul, engine.Add, engine.MulScalar (~10 ops) |
| `LayerNormBackward[T]` | layernorm_backward.go | engine.ReduceSum, engine.Mul, engine.Sub, engine.Div |
| `SoftmaxBackward[T]` | softmax_backward.go | engine.Mul, engine.ReduceSum, engine.Sub |
| `MultiHeadAttentionBackward[T]` | attention_backward.go | SoftmaxBackward, engine.MatMul, engine.Transpose |
| `MLPBackward[T]` | mlp_backward.go | LinearBackward (x2), GELUBackward |

`MLPBackward` demonstrates ideal composition: it calls `LinearBackward` twice
and `GELUBackward` once, composing complex gradient computation from simple
building blocks.

**Tier 2: Model-Specific Backward Passes** (timeseries/, crossasset/)

Monolithic implementations that ignore Tier 1:

| File | Lines | Composition | Uses Tier 1? |
|------|-------|-------------|--------------|
| timeseries/patchtst_backward.go | 825 | Monolithic, custom cache | No |
| timeseries/itransformer_backward.go | 723 | Monolithic, heavy caching | No |
| timeseries/timemixer_backward.go | 632 | Monolithic, manual gradient accumulation | No |
| crossasset/backward.go | 528 | Hybrid -- calls layer .Backward() methods | Partially |

### Duplication Between Tiers

| Gradient Computation | Tier 1 (functional) | timeseries/ (monolithic) |
|---------------------|---------------------|--------------------------|
| LayerNorm forward stats recompute | layernorm_backward.go:33-69 | itransformer_backward.go:233-242, patchtst_encoder.go:94-195 |
| Transpose + MatMul for dWeight | linear_backward.go | patchtst_backward.go:405-430, itransformer_backward.go:345-361 |
| Reshape for attention heads | attention_backward.go (splitHeads) | crossasset/backward.go (reshapeForHeads) |
| Residual gradient accumulation | Via tensor returns | crossasset/backward.go:269-275 (manual addition) |

### Layer-Level Backward Methods

All 45+ layer types in `layers/core/` implement `Backward()` methods.
These are well-composed, delegating gradient computation to engine operations.
The issue is that timeseries/ model backward passes reimplement what these
layer methods already provide.

---

## Quantified Impact

### Lines of Redundant Code by Component

| Reimplemented Component | Instances | Est. Redundant Lines | Change (rev 2->3) |
|------------------------|-----------|---------------------|-------------------|
| Linear/MatMul/MLP layer | 11 | ~1,400 | -- |
| LayerNorm (all variants) | 10 | ~1,200 | -- |
| Multi-head attention | 6 | ~1,500 | -- |
| GELU activation | 4 | ~120 | -- |
| Softmax | 4 | ~150 | -- |
| AdamW optimizer | 1 | ~80 | -- |
| MSE loss | 1 | ~20 | -- |
| BCE/Quantile/Contrastive loss (raw) | 3 | ~470 | -- |
| Gradient clipping (raw .Data()) | 1 | ~30 | -- |
| Xavier initialization | 1 | ~20 | -- |
| ReLU activation | 1 | ~10 | -- |
| Full backward passes (raw f64) | 3 | ~2,500 | -- |
| Intra-layers/ duplication | 14 | ~600 | -- |
| Autoregressive loop duplication (NEW) | 4 | ~400 | +400 |
| Inference builder boilerplate (NEW) | 6 patterns | ~1,155 | +1,155 |
| Handler duplication (NEW) | 2 | ~200 | +200 |
| **Total** | **72+** | **~9,855** | **+1,755** |

### Packages That Follow the Principle

| Package | Pattern | Evidence |
|---------|---------|---------|
| `inference/arch_common.go` | Exemplar | Composes attention, normalization, core, embeddings |
| `inference/arch_llama.go` | Good | Via arch_common |
| `inference/arch_gemma.go` | Good | Via arch_common |
| `inference/arch_mistral.go` | Good | Via arch_common |
| `inference/arch_qwen.go` | Good | Via arch_common |
| `inference/arch_phi.go` | Good | Via arch_common |
| `inference/arch_deepseek.go` | Good | Via arch_common + 2 justified custom nodes |
| `layers/transformer/` | Good | Composes attention, normalization, core |
| `layers/vision/` | Partial | Composes core, normalization; clip_encoder.go violates |
| `layers/audio/` | Good | Composes core |
| `layers/hrm/` | Good | Composes transformer |
| `layers/ssm/` | Good | Composes core |
| `layers/timeseries/` | Partial | Composes core, normalization; mlstm/ssm/vsn violate |
| `layers/functional/` | Good | Composes engine ops into backward operations |
| `model/` | Excellent | Clean interfaces, adapter pattern, registry |
| `model/hrm/` | Good | Composes layers/core, layers/hrm |
| `crossasset/` | Good (migrated) | Uses `layers/functional` + canonical optimizer |
| `rl/` | Good (migrated) | Uses `layers/functional` + canonical optimizer |
| `synth/` | Good (migrated) | Uses `layers/functional` + canonical optimizer |
| `meta/` | Good (migrated) | Uses `layers/functional` |
| `shared/` | Good (migrated) | Uses `engine.MatMul` |

### Packages That Still Violate the Principle

| Package | Severity | Lines | Primary Violation |
|---------|----------|-------|-------------------|
| `timeseries/` | S1 | 17,570 | Complete parallel framework, 254 .Data() calls |
| `tabular/` | S1 | 4,010 | 5x linearForward, 3x layerNorm, 51 .Data() calls |
| `modeldsl/` | S1 | 1,520 | Custom layer type system reimplementing layers/ |
| `gnn/` | S3 | 639 | Raw float64 matrix ops, no tensor integration |
| `training/loss/` | S2 | ~470 | Engine fields unused, raw .Data() loops |
| `training/optimizer/` | S2 | ~140 | AdamW8bit element-by-element on raw slices |
| `inference/` (custom nodes) | S2 | ~1,660 | 10 unjustified custom nodes, ~550 lines |
| `inference/` (builder boilerplate) | S2.5 | ~1,155 | 6 patterns duplicated across 30 files |
| `inference/timeseries/` | S2 | ~2,000 | Custom nodes with .Data(), monolithic Forward() |
| `generate/` (loops + cache) | S2.5 | ~1,235 | 4 duplicated decode loops, 5 KV cache variants |
| `serve/` (handlers) | S2.5 | ~200 | Monolithic handlers, duplicated option building |
| `layers/` (internal) | S1.5 | ~600 | Self-bypass of Engine and cross-sub-package duplication |

---

## Root Cause Analysis

### Why did this happen?

1. **Historical precedent.** The `timeseries/` and `crossasset/` packages predate
   the `layers/` package. They were built when the framework only had `Engine[T]`
   (and in many cases, not even that). The raw-slice approach was the only option
   at the time.

2. **Training vs inference divergence.** The `layers/` system was designed
   primarily for the GGUF inference path. Training requires backward passes and
   optimizer integration that the early `layers/` API did not support well.
   Packages needing training chose to reimplement rather than extend `layers/`.

3. **Type constraint friction.** The `layers/` system uses `[T tensor.Numeric]`
   generics and `compute.Engine[T]`. Packages that only need `float64` (which is
   most training code) find it simpler to use raw slices than to instantiate
   `compute.NewCPUEngine[float64]()` and wrap data in tensors. The ergonomic
   overhead of the generic system pushed developers toward raw math.

4. **Copy-paste across packages.** When a new domain package was created (tabular,
   GNN, RL, synth, meta), the author copied the linear/layerNorm/attention pattern
   from whichever existing package was closest, rather than importing from `layers/`.

5. **No enforcement mechanism.** There is no lint rule, CI check, or architecture
   test that prevents packages from reimplementing `Engine[T]` operations with raw
   slice math. ADR-027 documented the principle but did not enforce it. (Note: an
   architecture enforcement test was added to CI on Apr 3 -- see obs #577 -- but
   timeseries/ is currently on the allowlist.)

6. **Missing abstractions in hot paths.** The generate/ and serve/ packages grew
   organically. New generation modes (streaming, speculative, EAGLE speculative)
   were added by copying the existing loop rather than extracting a shared
   abstraction. Each new mode added ~150-200 lines of near-duplicate code.

7. **Inference builder template copying.** Each new architecture builder was
   created by copying an existing one and modifying the model-specific parts.
   The shared boilerplate (tensor lookup, param wrapper, weight transpose) was
   never extracted because each copy-paste was faster than creating a shared
   utility.

### What is working?

The migration pattern used for crossasset/, rl/, synth/, meta/, and shared/ is
proven and repeatable:

1. Import `layers/functional` for forward ops (Linear, LayerNorm, GELU, etc.)
2. Import `training/optimizer` for canonical optimizer implementations
3. Delete reimplemented helpers
4. Run existing tests to verify behavioral equivalence

This pattern reduced 6 packages from S1 violations to clean composition in a
single day.

---

## Remediation Strategy

### Priority 1: Migrate tabular/ (highest ROI, smallest effort)

4,010 lines, 5 identical `linearForward` methods, 3 identical `layerNorm`
methods, 1 shared `geluScalar`. The migration from crossasset/ provides a
direct template.

Steps:
1. Add `import "github.com/zerfoo/zerfoo/layers/functional"` to each file
2. Replace `linearForward(ctx, engine, x, w, b)` with `functional.Linear(ctx, engine, x, w, b)`
3. Replace `layerNorm(ctx, engine, x, gamma, beta, eps)` with `functional.LayerNorm(ctx, engine, x, gamma, beta, eps)`
4. Replace `geluScalar()` calls with `functional.GELU`
5. Replace manual multi-head attention with `functional.MultiHeadAttention`
6. Delete orphaned helper methods
7. Run `go test ./tabular/...`

**Estimated reduction:** ~600 lines

### Priority 2: Migrate timeseries/ forward paths (biggest bang for buck)

The `timeseries/` package has 17,570 lines across 35 files. Migration order:

1. Replace `layernorm_ops.go` calls with `functional.LayerNorm`
2. Replace `math_ops.go` (GELU, softmax) with `functional` equivalents
3. Replace `training_ops.go` (AdamW, MSE) with `training/optimizer` and `training/loss`
4. Replace inline attention in iTransformer and PatchTST with `functional.MultiHeadAttention`
5. Replace `linearForward` / `mlpLayer` with `functional.Linear`
6. Delete `layernorm_ops.go`, `math_ops.go`, `training_ops.go`, `adamw_f32.go`

The backward passes are harder -- they require either `graph.Backward()` support
or maintaining custom backward functions that call `functional` variants.

**Estimated reduction:** ~2,000 lines (forward path only)

### Priority 3: Migrate timeseries/ backward passes to functional ops

Replace the 2,180 lines of monolithic backward code with compositions of
`layers/functional` backward operations:

1. Replace inline linear backward in patchtst_backward.go and itransformer_backward.go
   with `functional.LinearBackward`
2. Replace inline GELU backward with `functional.GELUBackward`
3. Replace inline LayerNorm backward with `functional.LayerNormBackward`
4. Replace inline attention backward with `functional.MultiHeadAttentionBackward`
5. Compose full encoder backward from `functional.MLPBackward` +
   `functional.MultiHeadAttentionBackward` + `functional.LayerNormBackward`
6. Delete custom cache structs that only exist to support monolithic backward

**Estimated reduction:** ~1,500 lines

### Priority 4: Fix layers/ internal violations

The 14 violations within `layers/` itself undermine the credibility of the
composition principle. Priority order:

1. `vision/clip_encoder.go` -- Rewrite patch extraction, class token concat,
   position embedding, and attention to use engine ops (~200 lines)
2. `timeseries/mlstm.go` -- Replace `.Data()` access with engine.Slice and
   vectorized ops (~80 lines)
3. `timeseries/ssm.go` -- Replace raw discretization loops with engine ops (~60 lines)
4. `timeseries/vsn.go` -- Replace weight extraction with engine.Slice (~30 lines)
5. `core/gemm.go` -- Replace hand-rolled GEMM with engine.MatMul (~20 lines)
6. `core/variable_selection.go` -- Replace inline GELU with activations.Gelu (~2 lines)
7. `core/temporal_conv_encoder.go` -- Replace raw pool gradient with engine ops (~10 lines)

### Priority 5: Extract generate/ shared abstractions

1. Extract `selectCacheProvider()` from the duplicated 8-branch if-else
2. Extract `CoreAutoregressiveDecoder[T]` struct with `runDecodeStep()` method
   shared by Generate(), GenerateStream(), speculative, and EAGLE speculative
3. Extract `sampleFromLogits()` sub-functions: `tryGPUArgmax()`,
   `applyAllSamplingModifiers()`, `convertLogitsToFloat64()`, `selectToken()`
4. Replace `.Data()` calls in sampling with engine ops where possible

**Estimated reduction:** ~400 lines of duplication, improved testability

### Priority 6: Extract inference/ builder utilities

1. Create `inference/builder_helpers.go` with shared factory functions:
   - `newTensorLookup(tensors map[string]*tensor.TensorNumeric[T])`
   - `newParamWrapper()`
   - `transposeWeight(tensor, storageType)` (consolidate 6 storage branches)
   - `newEmbeddingNode(engine, weight)` (factory)
   - `newLMHeadNode(engine, weight)` (factory)
2. Migrate 25-30 architecture builders to use shared utilities
3. Replace unjustified custom nodes (BERT, GPT-2, Falcon, Command R, LLaVA)

**Estimated reduction:** ~1,000 lines of boilerplate

### Priority 7: Fix inference/ custom nodes

Replace unjustified custom nodes in architecture builders:

1. `arch_vision_helpers.go` -- llamaAttnNode, llamaFFNNode, rmsNormWrapNode (~210 lines)
2. `arch_bert.go` -- bertFFNNode, bertResidualLayerNormNode, bertEmbeddingNode (~140 lines)
3. `arch_gpt2.go` -- gpt2ResidualAddNode, gpt2EmbeddingNode (~100 lines)
4. `arch_falcon.go` -- falconGeluFFN (~40 lines)
5. `arch_commandr.go` -- commandRResidualAddNode (~40 lines)

### Priority 8: Fix training/loss/ engine bypass

1. `bce.go` -- Replace raw ops with engine.Log, engine.Mul, engine.Add, engine.Sub
2. `routing_contrastive.go` -- Replace triple-nested loops with engine.MatMul for
   batch pairwise cosine similarity
3. `quantile.go` -- Fix broken generics (panics for non-float32) and use engine ops

### Priority 9: Extract serve/ handler helpers

1. Extract `buildGenerationOptions(request) ([]inference.GenerateOption, error)`
2. Extract `parseAndApplyGrammar(schema) (grammar.Grammar, error)`
3. Extract `detectAndFormatToolCalls(response, tools) *ToolCall`
4. Shared between handleChatCompletions and handleCompletions

**Estimated reduction:** ~200 lines

### Priority 10: Add Architecture Test Enforcement

Remove `timeseries/` from the architecture test allowlist as migration completes.
Add `tabular/` to the test if not already covered.

---

## Positive Observations

1. **The inference path is exemplary.** `arch_common.go` and the standard
   architecture builders (Llama, Gemma, Mistral, Qwen, Phi) cleanly compose
   from `layers/`. This is the pattern to follow.

2. **layers/ itself is well-designed.** 56+ operations across 20 sub-packages
   in a clean tier hierarchy: atomic ops (transpose, reducesum, gather) ->
   foundational (activations, normalization, embeddings) -> core (56 ops) ->
   functional (backward ops) -> specialized (attention, ssm, timeseries) ->
   orchestration (transformer). Higher tiers compose from lower tiers.

3. **The migration pattern is proven.** Five packages were migrated from S1 to
   clean composition in a single day (Apr 3). The `layers/functional` API
   provides the ergonomic bridge that was missing when the violating packages
   were originally written.

4. **Migration preserves behavioral equivalence.** All migrated packages pass
   their existing test suites. The functional API is a drop-in replacement for
   the reimplemented helpers.

5. **Performance-justified exceptions are documented.** Fused CUDA kernels,
   SIMD assembly, and megakernel codegen are correctly treated as optimized
   implementations, not composability violations.

6. **Mamba in timeseries/ is the reference.** `timeseries/mamba.go` imports
   `layers/ssm` and `layers/core`, proving that full composition is feasible
   even for complex time-series models.

7. **Architecture enforcement is now in CI.** The composition test (obs #577)
   prevents new violations from being introduced, even though legacy violations
   are currently allowlisted.

8. **model/ package is gold standard.** 8 clean interfaces (ModelProvider,
   ModelSerializer, ModelLoader, ModelExporter, ModelValidator, ModelOptimizer),
   adapter pattern for LoRA, registry for discovery -- demonstrates
   dependency inversion and composition principles.

9. **generate/ has good foundations.** The CacheProvider[T] interface with 5+
   implementations (KVCache, PagedKVCache, CompressedKVCache, TensorCache,
   TieredKVStore via adapter) is textbook composable design. The structural
   violations are in the orchestration layer, not the abstractions.

10. **Functional backward API is well-composed.** `MLPBackward` calling
    `LinearBackward` twice and `GELUBackward` once is exactly the composition
    pattern the framework advocates. The problem is that model-level backward
    passes don't use it yet.

---

## Revision History

| Date | Rev | Changes |
|------|-----|---------|
| 2026-04-02 | 1 | Initial audit: 14 violating packages, ~14,000 redundant lines |
| 2026-04-03 | 2 | 5 packages migrated, deeper layers/ review, +4 internal violations found. 8 violating packages remain, ~8,200 redundant lines |
| 2026-04-03 | 3 | Deeper audit: added .Data() inventory (2,599 total, 805 unjustified), backward pass composition analysis, generate/ structural violations (4 duplicated loops, monolithic functions), serve/ handler duplication, inference builder boilerplate (6 patterns x 30 files), cmd/ bootstrap gap. Total estimated redundant: ~9,800 lines across 72+ violation instances. Added S2.5 severity tier for structural composition violations. |
