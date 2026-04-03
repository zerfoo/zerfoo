# Dirty Architecture: Composability Violations Audit

*Updated: 2026-04-03 (revision 2)*
*Previous: 2026-04-02 (revision 1)*

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

| Metric | Rev 1 (Apr 2) | Rev 2 (Apr 3) | Delta |
|--------|---------------|---------------|-------|
| Source files reviewed (non-test .go) | 860 | 10,373 | +9,513 |
| Total lines of code | 170,860 | 77,720 | recount (non-test only) |
| Packages with composability violations | 14 | 8 | -6 (migrated) |
| Reimplemented LayerNorm instances | 17 | 10 | -7 |
| Reimplemented Linear/MatMul/MLP instances | 18 | 11 | -7 |
| Reimplemented softmax instances | 6 | 4 | -2 |
| Reimplemented GELU instances | 5 | 4 | -1 |
| Reimplemented AdamW instances | 5 | 1 | -4 |
| Reimplemented SGD instances | 5+ | 0 | -5+ |
| Reimplemented attention instances | 8 | 6 | -2 |
| Intra-layers/ violations | 10 | 14 | +4 (more found) |
| Estimated redundant lines | ~14,000 | ~8,200 | -5,800 |

**What changed since revision 1:**

- **crossasset/** migrated to `layers/functional` + canonical `optimizer.AdamW[float64]`
  (T68.1.1-T68.1.3, Apr 3). 1,357 lines of dead GPU code deleted.
- **rl/** migrated to `layers/functional` + `optimizer.SGD` (T71.1.1, Apr 3).
- **synth/** migrated to `layers/functional` + `optimizer.SGD` (T71.1.2, Apr 3).
- **meta/** migrated to `layers/functional` (T71.1.3, Apr 3).
- **shared/** migrated `matVecMul` to `engine.MatMul` (T71.1.4, Apr 3).
- Deeper review of `layers/` internals found 4 additional violations not caught
  in revision 1.

**Bottom line:** The inference path follows the composability principle well. Five
of the eight S1 violators from revision 1 have been migrated. The remaining
violations are concentrated in three packages: **timeseries/** (17,570 lines),
**tabular/** (4,010 lines), and **modeldsl/** (1,520 lines). The `layers/`
package itself has 14 internal violations where sub-packages bypass `Engine[T]`
or duplicate each other.

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

**Mixed pattern:** tabular/ uses `compute.Engine[T]` for MatMul and some ops,
but reimplements LayerNorm and GELU inline with `tensor.Data()` access.

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


### 12. generate/ -- KV Cache Duplication

**Files:** `kvcache.go`, `kvcache_fp16.go`, `kvcache_fp8.go`, `kvcache_q3.go`,
`kvcache_q4.go` (1,235 lines total)

Five KV cache implementations with the same interface (`Get`, `Set`, `Len`,
`MaxSeqLen`) but different quantization. Structural code (layer indexing,
sequence management, resize logic) is duplicated across all five.

**Justified?** Partially. The quantization math genuinely differs between
formats. However, a base struct with quantization strategy injection would
eliminate ~400 lines of structural duplication.


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

## Quantified Impact

### Lines of Redundant Code by Component

| Reimplemented Component | Instances | Est. Redundant Lines | Change |
|------------------------|-----------|---------------------|--------|
| Linear/MatMul/MLP layer | 11 | ~1,400 | -1,000 |
| LayerNorm (all variants) | 10 | ~1,200 | -600 |
| Multi-head attention | 6 | ~1,500 | -500 |
| GELU activation | 4 | ~120 | (same) |
| Softmax | 4 | ~150 | -50 |
| AdamW optimizer | 1 | ~80 | -320 |
| MSE loss | 1 | ~20 | -20 |
| BCE/Quantile/Contrastive loss (raw) | 3 | ~470 | (same) |
| Gradient clipping (raw .Data()) | 1 | ~30 | -60 |
| Xavier initialization | 1 | ~20 | -80 |
| ReLU activation | 1 | ~10 | -40 |
| Full backward passes (raw f64) | 3 | ~2,500 | -500 |
| SGD optimizer | 0 | 0 | -150 |
| Intra-layers/ duplication | 14 | ~600 | +200 |
| **Total** | **60+** | **~8,200** | **-5,800** |

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
| `model/hrm/` | Good | Composes layers/core, layers/hrm |
| `crossasset/` | Good (NEW) | Uses `layers/functional` + canonical optimizer |
| `rl/` | Good (NEW) | Uses `layers/functional` + canonical optimizer |
| `synth/` | Good (NEW) | Uses `layers/functional` + canonical optimizer |
| `meta/` | Good (NEW) | Uses `layers/functional` |
| `shared/` | Good (NEW) | Uses `engine.MatMul` |

### Packages That Still Violate the Principle

| Package | Severity | Lines | Uses Engine[T]? | Uses layers/? |
|---------|----------|-------|-----------------|---------------|
| `timeseries/` | S1 | 17,570 | Partial (_engine.go files) | 3 of 35 files |
| `tabular/` | S1 | 4,010 | Partial (MatMul only) | No |
| `modeldsl/` | S1 | 1,520 | No | No |
| `gnn/` | S3 | 639 | No (declared, unused) | No |
| `training/loss/` | S2 | ~470 | Has field, unused | N/A |
| `training/optimizer/` | S2 | ~140 | Partial (8bit bypasses) | N/A |
| `inference/` (custom nodes) | S2 | ~1,660 | Yes | Partial |
| `inference/timeseries/` | S2 | ~2,000 | Yes | Partial |
| `generate/` (KV cache) | S2 | ~1,235 | N/A | N/A |
| `layers/` (internal) | S1.5 | ~600 | Partial | Self-bypass |

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

### Priority 3: Fix layers/ internal violations

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

### Priority 4: Fix inference/ custom nodes

Replace unjustified custom nodes in architecture builders:

1. `arch_vision_helpers.go` -- llamaAttnNode, llamaFFNNode, rmsNormWrapNode (~210 lines)
2. `arch_bert.go` -- bertFFNNode, bertResidualLayerNormNode, bertEmbeddingNode (~140 lines)
3. `arch_gpt2.go` -- gpt2ResidualAddNode, gpt2EmbeddingNode (~100 lines)
4. `arch_falcon.go` -- falconGeluFFN (~40 lines)
5. `arch_commandr.go` -- commandRResidualAddNode (~40 lines)

### Priority 5: Fix training/loss/ engine bypass

1. `bce.go` -- Replace raw ops with engine.Log, engine.Mul, engine.Add, engine.Sub
2. `routing_contrastive.go` -- Replace triple-nested loops with engine.MatMul for
   batch pairwise cosine similarity
3. `quantile.go` -- Fix broken generics (panics for non-float32) and use engine ops

### Priority 6: Add Architecture Test Enforcement

Remove `timeseries/` from the architecture test allowlist as migration completes.
Add `tabular/` to the test if not already covered.

---

## Positive Observations

1. **The inference path is exemplary.** `arch_common.go` and the standard
   architecture builders (Llama, Gemma, Mistral, Qwen, Phi) cleanly compose
   from `layers/`. This is the pattern to follow.

2. **layers/ itself is well-designed.** 56+ operations across 23 sub-packages,
   with clear separation of concerns and consistent interfaces. The composition
   principle works when it is followed.

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

---

## Revision History

| Date | Rev | Changes |
|------|-----|---------|
| 2026-04-02 | 1 | Initial audit: 14 violating packages, ~14,000 redundant lines |
| 2026-04-03 | 2 | 5 packages migrated, deeper layers/ review, +4 internal violations found. 8 violating packages remain, ~8,200 redundant lines |
