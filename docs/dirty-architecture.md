# Dirty Architecture: Composability Violations Audit

*Date: 2026-04-02*

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

| Metric | Count |
|--------|-------|
| Source files reviewed | 860 |
| Total lines of code | 170,860 |
| Packages with composability violations | 14 |
| Reimplemented LayerNorm instances | 17 |
| Reimplemented Linear/MatMul/MLP instances | 18 |
| Reimplemented softmax instances | 6 (non-kernel) |
| Reimplemented GELU instances | 5 |
| Reimplemented AdamW instances | 5 |
| Reimplemented SGD instances | 5+ |
| Reimplemented attention instances | 8 |
| Intra-layers/ violations | 10 |
| Estimated redundant lines | ~14,000 |

**Bottom line:** The inference path (GGUF model loading via `arch_common.go` and
the architecture builders that compose from `layers/`) follows the composability
principle well. The training backends (`timeseries/`, `crossasset/`, `tabular/`)
and experimental packages (`modeldsl/`, `gnn/`, `rl/`, `synth/`, `meta/`,
`shared/`) are a parallel universe of raw-slice math that bypasses both `Engine[T]`
and `layers/` entirely.

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

### 1. crossasset/ -- Raw-Slice Transformer (918 lines)

**Files:** `crossasset.go`, `backward.go`, `adamw.go`, `gpu_params.go`, `gpu_train.go`

The cross-asset attention model operates entirely on raw `[]float64` slices.
It reimplements:

| Component | crossasset/ implementation | Canonical implementation |
|-----------|---------------------------|--------------------------|
| Linear/MatVec | `matVecMul()` at crossasset.go:518 | `layers/core.Linear` |
| Multi-head attention | `forwardLayerCached()` at backward.go:30 | `layers/attention.GroupedQueryAttention` |
| LayerNorm | `layerNorm()` at crossasset.go:558 | `layers/normalization.LayerNorm` |
| GELU | `gelu()` at crossasset.go:584 | `layers/activations.Gelu` |
| Softmax | `softmax()` at crossasset.go:538 | `layers/activations.Softmax` |
| AdamW | `adamWUpdateAll()` at adamw.go:55 | `training/optimizer.AdamW[T]` |
| Full backward pass | backward.go (entire file) | `graph.Backward()` |

The `gpu_train.go` file (851 lines) contains a separate GPU training path that
also reimplements LayerNorm (`cpuLayerNorm` at line 851), softmax backward
(`cpuSoftmaxBackward` at line 910), and AdamW (`adamWUpdate` at line 959) --
this time on `[]float32` tensors, but still outside the `layers/` system.

**Impact:** Cannot benefit from CUDA graph capture, megakernel codegen, or any
future engine optimization. The CPU and GPU paths have diverged: `crossasset.go`
uses `[]float64`, `gpu_train.go` uses `tensor.TensorNumeric[float32]` with
manual `.Data()` access. Two independent implementations of the same model.

**Justified?** No. The cross-asset model is a standard transformer encoder.
`inference/arch_common.go` demonstrates that transformers compose cleanly from
`layers/`. The only difference is that crossasset trains with backpropagation,
which the `graph.Backward()` system already supports.


### 2. timeseries/ -- Parallel ML Framework (18,197 lines)

**Files:** 34 source files including `math_ops.go`, `layernorm_ops.go`,
`training_ops.go`, `adamw_f32.go`, plus model files.

The top-level `timeseries/` package is effectively a separate ML framework that
shares zero code with `layers/`. It reimplements:

| Component | timeseries/ file:line | Canonical |
|-----------|----------------------|-----------|
| LayerNorm (2D) | layernorm_ops.go:7 | `layers/normalization.LayerNorm` |
| LayerNorm (2D+cache) | layernorm_ops.go:35 | `layers/normalization.LayerNorm` |
| LayerNorm backward | layernorm_ops.go:72 | `layers/normalization.LayerNorm.Backward` |
| LayerNorm (1D) | layernorm_ops.go:106 | `layers/normalization.LayerNorm` |
| LayerNorm (1D+cache) | layernorm_ops.go:130 | `layers/normalization.LayerNorm` |
| GELU | math_ops.go:7 | `layers/activations.Gelu` |
| GELU derivative | math_ops.go:14 | `layers/activations.Gelu.Backward` |
| Softmax | math_ops.go:34 | `layers/activations.Softmax` |
| AdamW state | training_ops.go:26 | `training/optimizer.AdamW[T]` |
| AdamW update | training_ops.go:41 | `training/optimizer.AdamW[T].Step` |
| AdamW (float32) | adamw_f32.go:30 | `training/optimizer.AdamW[T]` |
| MSE loss | training_ops.go:57 | `training/loss.MSE[T]` |
| Gradient clipping | training_ops.go:8 | (should be in training/) |
| Multi-head attention | patchtst.go:315 | `layers/attention.GroupedQueryAttention` |
| Multi-head attn (f64) | patchtst_backward.go:881 | `layers/attention.GroupedQueryAttention` |
| Linear forward | nbeats.go:426, nhits.go:260 | `layers/core.Linear` |
| MLP layer type | nbeats.go:70, doc.go:21 | `layers/core.Linear` + `layers/core.FFN` |

Several timeseries models have _engine.go companion files (patchtst_engine.go,
itransformer_engine.go, frets_engine.go, etc.) that partially migrate operations
to `compute.Engine[T]`. These represent incomplete migration: some ops use the
engine, but LayerNorm and attention are still reimplemented manually even in the
engine files (e.g., `patchtst_encoder.go:21` has `layerNormForwardWithEngine`
which manually computes mean/variance using engine ops rather than using
`layers/normalization.LayerNorm`).

**Impact:** 18,197 lines (11% of the codebase) exist in a parallel universe.
10 model architectures (PatchTST, iTransformer, DLinear, NHiTS, N-BEATS, TFT,
TimeMixer, CfC, TTM, Mamba-TS) each carry their own math. Bug fixes to
`layers/normalization.LayerNorm` do not propagate here. GPU acceleration
requires duplicate effort for each model.

**Justified?** No. The models are standard neural networks (transformer
encoders, MLPs, recurrent networks). The `_engine.go` files prove that engine
migration is feasible -- it just was never completed.

**Partially justified exceptions within timeseries/:**
- `mamba.go` imports `layers/ssm` and `layers/core` -- this file follows the
  composability principle.
- `ttm_engine.go` imports `layers/core` and `layers/timeseries` -- partial
  composition.


### 3. tabular/ -- Another Parallel Framework (4,010 lines)

**Files:** `ft_transformer.go`, `saint.go`, `tabnet.go`, `resnet.go`, `model.go`,
`train.go`, `lora.go`, `pretrain.go`, `save.go`, `ensemble.go`

Reimplements:

| Component | tabular/ file:line | Canonical |
|-----------|-------------------|-----------|
| LayerNorm | ft_transformer.go:457 | `layers/normalization.LayerNorm` |
| LayerNorm | resnet.go:232 | `layers/normalization.LayerNorm` |
| LayerNorm | saint.go:653 | `layers/normalization.LayerNorm` |
| Multi-head attention | ft_transformer.go:313 | `layers/attention.GroupedQueryAttention` |
| Multi-head attention | saint.go (self-attn + intersample) | `layers/attention` |
| GELU | model.go:212 | `layers/activations.Gelu` |
| Linear forward | ft_transformer.go:496 | `layers/core.Linear` |
| Linear forward | saint.go:745 | `layers/core.Linear` |
| Linear forward | tabnet.go:318 | `layers/core.Linear` |
| Linear forward | resnet.go:282 | `layers/core.Linear` |
| Linear forward | model.go:191 | `layers/core.Linear` |
| MLP layer type | model.go:59 | `layers/core.Linear` + `layers/core.FFN` |

**Mixed pattern:** tabular/ uses `compute.Engine[T]` for MatMul and some ops,
but reimplements LayerNorm and GELU inline with `tensor.Data()` access. This is
the S2 pattern in an S1-sized package.

**Impact:** 5 model architectures (FT-Transformer, SAINT, TabNet, TabResNet,
base Model) each carry duplicate linearForward/layerNorm methods. The
`linearForward` function appears 5 times with nearly identical code.

**Justified?** No. These are standard transformer and MLP architectures.
The `engine.MatMul` usage proves engine integration is possible.


### 4. modeldsl/ -- Third Linear Layer (696 lines)

**Files:** `dsl.go`, `model.go`, `graph.go`, `optimize.go`, `train.go`

Reimplements:

| Component | modeldsl/ file:line | Canonical |
|-----------|---------------------|-----------|
| Linear layer | model.go:108 (`linearLayer`) | `layers/core.Linear` |
| RMSNorm | model.go (via `rmsnormLayerT`) | `layers/normalization.RMSNorm` |
| SiLU | model.go (via `siluLayerT`) | `layers/activations.SwiGLU` |
| Softmax | model.go (via `softmaxLayerT`) | `layers/activations.Softmax` |
| Attention | model.go:153 (`attentionLayer`) | `layers/attention` |
| Xavier init | model.go:117 | `layers/components.WeightInitializer` |

The file contains a telling comment at model.go:141:
> NOTE: Element-wise layers (rmsnorm, silu, softmax) operate on raw []float64
> slices rather than tensors. The layers/activations/ package provides
> tensor-based equivalents via compute.Engine[T], but the DSL intentionally
> uses slice-based ops because its entire pipeline (forward, backward,
> parameter updates) operates on []float64.

This is honest about the violation but does not justify it. The DSL defines
`LayerType` constants (`LayerLinear`, `LayerRMSNorm`, `LayerSiLU`, `LayerSoftmax`,
`LayerAttention`) that duplicate the layer registry in `layers/registry/`.

**Justified?** No. The DSL should produce graphs that compose from existing
`layers/` implementations rather than creating a parallel type system.


### 5. gnn/ -- Fourth Linear Layer (416 lines)

**Files:** `gcn.go`, `gat.go`

Reimplements:

| Component | gnn/ file:line | Canonical |
|-----------|---------------|-----------|
| Linear (matmul) | gcn.go:239 (`matMul`) | `layers/core.Linear` |
| ReLU | gcn.go:278 (`reluMatrix`) | `layers/activations.ReLU` |
| Softmax | gcn.go:300 (`softmaxMatrix`) | `layers/activations.Softmax` |
| Xavier init | gcn.go:223 (`xavierMatrix`) | `layers/components.WeightInitializer` |
| Multi-head attention | gat.go:195 | `layers/attention` |
| Add bias | gcn.go:269 (`addBias`) | `layers/core.Add` |

**Impact:** Uses raw `[][]float64`. No `Engine[T]`, no tensors. A private
`var cpuEngine` is declared at gcn.go:15 but is not used by any method.

**Justified?** No. GCN and GAT are standard neural network architectures.
The graph-specific operation (adjacency normalization and neighborhood
aggregation) is the only part that couldn't compose from existing layers.


### 6. rl/ -- Fifth Linear Layer (787 lines)

**Files:** `ppo.go`, `sac.go`, `replay.go`, `rl.go`

Reimplements:

| Component | rl/ file:line | Canonical |
|-----------|-------------|-----------|
| MLP layer | ppo.go:38 (`mlpLayer`) | `layers/core.Linear` |
| Linear forward | ppo.go:57 | `layers/core.Linear` |
| Linear backward | ppo.go:71 | `layers/core.Linear.Backward` |
| ReLU | ppo.go (inline) | `layers/activations.ReLU` |
| Tanh | ppo.go (inline) | `layers/activations.Tanh` |
| SGD update | ppo.go (inline) | `training/optimizer.SGD[T]` |

**Justified?** No. PPO and SAC are standard policy gradient methods that use
MLP function approximators. The MLP could compose from `layers/core.Linear`.


### 7. synth/ -- Sixth Linear Layer (676 lines)

**Files:** `vae.go`, `crash.go`

Reimplements:

| Component | synth/ file:line | Canonical |
|-----------|-----------------|-----------|
| Linear forward | vae.go:389 (`linearForward`) | `layers/core.Linear` |
| ReLU | vae.go (inline) | `layers/activations.ReLU` |
| Xavier init | vae.go (inline) | `layers/components.WeightInitializer` |
| SGD update | vae.go (inline) | `training/optimizer.SGD[T]` |

**Justified?** No. VAE is a standard generative model.


### 8. meta/ -- Seventh Linear Layer (307 lines)

**Files:** `meta.go`

Reimplements:

| Component | meta/ file:line | Canonical |
|-----------|----------------|-----------|
| Linear forward | meta.go:248 (`linearForward`) | `layers/core.Linear` |
| ReLU | meta.go (inline) | `layers/activations.ReLU` |
| Xavier init | meta.go (inline) | `layers/components.WeightInitializer` |
| SGD update | meta.go (inline) | `training/optimizer.SGD[T]` |

MAML meta-learning with raw `[]float64` MLP.

**Justified?** Partially. MAML requires computing gradients through the
inner-loop optimizer, which is awkward with the current `graph.Backward()`
system. However, the linear forward pass itself could still compose from
`layers/core`.


### 9. shared/ -- Eighth Linear Layer (259 lines)

**Files:** `latent.go`

Reimplements:

| Component | shared/ file:line | Canonical |
|-----------|------------------|-----------|
| MatVecMul | latent.go:241 | `layers/core.Linear` or `engine.MatMul` |
| SGD update | latent.go (inline) | `training/optimizer.SGD[T]` |

Latent space projections with raw matrix math.

**Justified?** No.

---

## S1.5: Violations Within layers/ Itself

The `layers/` package, which is meant to be the canonical source of composable
primitives, has internal violations where sub-packages bypass each other or
bypass `Engine[T]`.

### core/gemm.go -- Hand-Rolled GEMM

**File:** `layers/core/gemm.go:66-86`

Triple-nested `for i/j/k` loop over raw `aData[]/bData[]` slices for matrix
multiply. Never hits GPU, SIMD, CUDA graph, or any engine backend.

Should use: `engine.MatMul` + `engine.MulScalar` + `engine.Add`

### vision/clip_encoder.go -- Raw-Loop Multi-Head Attention

**File:** `layers/vision/clip_encoder.go:362-410`

6-nested loops implement QK^T scores, softmax, and weighted V sum on raw data
slices (~50 lines). Duplicates `attention.ScaledDotProductAttention`.

Also at lines 255-270: linear projection as raw triple-nested loop instead of
`engine.MatMul`. And at lines 492-503: in-place QuickGELU via raw ops.

Should use: `layers/attention.SDPA`, `layers/core.Linear`, `layers/activations.Gelu`

### timeseries/mlstm.go, slstm.go, ssm.go -- All-CPU Forward Passes

**Files:** `layers/timeseries/mlstm.go:237-317`, `slstm.go:257-316`, `ssm.go:230-264`

Entire forward passes as raw CPU loops. Key/value/query projections, gates
(exp/sigmoid), matrix cell state updates, all bypassing Engine.

Should use: `layers/core.Linear` for projections, engine ops for activations.

### timeseries/vsn.go -- Local LayerNorm Reimplementation

**File:** `layers/timeseries/vsn.go:180-224`

Reimplements LayerNorm using `engine.UnaryOp` for epsilon (breaks tracing).

Should use: `layers/normalization.LayerNormalization`

### normalization/simplified_layer_normalization.go -- Duplicates RMSNorm

**File:** `layers/normalization/simplified_layer_normalization.go:100-141`

Algorithm is identical to `rmsnorm.go:153-190` (~80 identical lines, including
fused fast paths). Both compute RMS normalization with the same structure.

Should share: common internal RMS computation or delegate to RMSNorm.

### activations/fast_gelu.go -- Near-Copy of gelu.go

**File:** `layers/activations/fast_gelu.go:26-91`

Forward is structurally identical to `gelu.go:47-111` (~65 duplicated lines).
Same formula, same Engine calls.

Should: consolidate into one implementation or compose from shared internals.

### core/variable_selection.go -- Inline GELU

**File:** `layers/core/variable_selection.go:147-154`

GELU via `ops.Mul/ops.Sigmoid` on raw data, bypassing Engine.

Should use: `layers/activations.Gelu`

### core/temporal_conv_encoder.go -- Inline ReLU

**File:** `layers/core/temporal_conv_encoder.go:107-113, 121-127`

ReLU via `.Data()` mutation instead of Engine op.

Should use: `layers/activations.Relu`

### residual/block_attn_res.go -- rmsNormLite

**File:** `layers/residual/block_attn_res.go:34-64`

Reimplements RMSNorm without gain (~20 lines).

Should use: `layers/normalization.RMSNorm` with gain fixed to ones.

---

## S2: Major Violations (Partial Bypass)

### 10. inference/ architecture builders -- Custom Nodes

The architecture builders (30 files, `arch_*.go`) compose from `layers/` for
the common pattern (see `arch_common.go`) but create custom `graph.Node[T]`
implementations for architecture-specific operations:

| Architecture | Custom nodes | Lines |
|-------------|-------------|-------|
| Falcon | `falconResidualLayerNorm`, `falconGeluFFN`, `falconParallelAddNode` | ~100 |
| BERT | `bertEmbeddingNode`, `bertResidualLayerNormNode`, `bertFFNNode`, `bertPoolerNode`, `bertClassifierNode` | ~220 |
| GPT-2 | `gpt2EmbeddingNode`, `gpt2ResidualAddNode` | ~120 |
| Command R | `commandRResidualAddNode` | ~40 |
| DeepSeek | `deepSeekReshapeNode`, `deepSeekConstNode` | ~60 |
| RWKV | `rwkvTimeMixNode`, `rwkvChannelMixNode` | ~300 |
| Kimi | `kimiLinearAttentionNode` | ~200 |
| Mamba | `mambaResidualAddNode` | ~40 |
| Voxtral | `voxtralAdapterNode` | ~80 |
| LLaVA | `mmProjectorNode`, `llamaAttnNode`, `llamaFFNNode` | ~200 |
| QwenVL | `qwenVLAttnNode` | ~150 |
| Fused ops | `fusedAddRMSNormNode`, `fusedNormAddNode`, `residualAddNode`, `residualRefNode` | ~150 |

**Total custom nodes:** ~50 types, ~1,660 lines

**Justified instances:**
- `fusedAddRMSNormNode`, `fusedNormAddNode` -- Performance-justified fusions
  that combine multiple engine ops into a single kernel launch.
- `rwkvTimeMixNode`, `rwkvChannelMixNode` -- RWKV has fundamentally different
  attention (linear recurrence) that cannot compose from standard attention layers.
- `kimiLinearAttentionNode` -- Linear attention is a distinct mechanism.
- `mambaResidualAddNode` -- SSM-specific residual pattern.
- `deepSeekReshapeNode`, `deepSeekConstNode` -- MLA-specific tensor manipulation.

**Unjustified instances:**
- `falconGeluFFN` -- This is a standard FFN with GELU. Should compose from
  `layers/core.FFN`.
- `bertFFNNode` -- Standard FFN. Should compose from `layers/core.FFN`.
- `bertResidualLayerNormNode` -- Residual + LayerNorm. Could compose from
  `layers/normalization.LayerNorm` + `layers/core.Add`.
- `gpt2ResidualAddNode` -- Simple residual add. Should use `layers/core.Add`.
- `commandRResidualAddNode` -- Same as gpt2, simple residual add.
- `gpt2EmbeddingNode` -- Token + position embedding. Could compose from
  `layers/embeddings.TokenEmbedding`.
- `bertEmbeddingNode` -- Token + position + segment embedding. Most of this
  could compose from existing embeddings.

### 11. generate/ -- KV Cache Duplication

**Files:** `kvcache.go`, `kvcache_fp16.go`, `kvcache_fp8.go`, `kvcache_q3.go`,
`kvcache_q4.go` (1,235 lines total)

Five KV cache implementations with the same interface (`Get`, `Set`, `Len`,
`MaxSeqLen`) but different quantization:

| Type | File | Lines | Quantization |
|------|------|-------|-------------|
| `KVCache[T]` | kvcache.go | 172 | None (generic) |
| `KVCacheFP16` | kvcache_fp16.go | 188 | float32 -> float16 |
| `KVCacheFP8` | kvcache_fp8.go | 183 | float32 -> float8 |
| `KVCacheQ4` | kvcache_q4.go | 300 | float32 -> Q4_K |
| `KVCacheQ3` | kvcache_q3.go | 392 | float32 -> Q3_K |

**Justified?** Partially. The quantization math genuinely differs between
formats, and the Q3/Q4 block quantization has no generic expression. However,
the structural code (layer indexing, sequence management, resize logic) is
duplicated across all five. A base struct with quantization strategy injection
would eliminate ~400 lines of duplication.


### 12. inference/timeseries/ -- Partial Composition

**Files:** 20 files, `arch_*.go` pattern

The `inference/timeseries/` sub-package (distinct from the top-level
`timeseries/`) follows a mixed pattern. Some architecture builders compose from
`layers/`:

- `arch_ttm.go` -- imports `layers/core` and `layers/timeseries`
- `arch_flowstate.go` -- imports `layers/core` and `layers/timeseries`
- `arch_patchtst.go` -- imports `layers/core` and `layers/timeseries`

Others create custom nodes:
- `arch_timemixer.go:96` -- `timeMixerNode` with 500+ lines including
  inline softmax (`softmaxKernel` at line 507)
- `arch_chronos.go:135` -- `chronosNode` with encoder/decoder blocks
- `arch_tirex.go:107` -- `tiRexNode`
- `arch_tft.go:325` -- `tftNode` with inline LSTM

**Justified?** Mixed. TimeMixer's seasonality-trend decomposition and TFT's
variable selection networks have unique structure. But the inline softmax and
LSTM could compose from existing layers.


### 13. training/loss/ -- Engine Fields Unused

Several loss functions accept `compute.Engine[T]` but never delegate to it,
instead iterating over `.Data()` with element-wise `ops` calls:

| Loss | File | Lines | Issue |
|------|------|-------|-------|
| `BCELoss` | training/loss/bce.go | 156 | Has `engine` field, iterates `.Data()` for log/mul/add |
| `RoutingContrastive` | training/loss/routing_contrastive.go | 224 | Triple-nested raw loops for cosine similarity |
| `QuantileLoss` | training/loss/quantile.go | 92 | Hardcoded `float32` casts via `any(...).(float32)` despite generic `[T]` |

`BCELoss` is the clearest violation: it stores `engine` but its `Forward` and
`Backward` methods use only `ops` (element-wise arithmetic). The computation
(log, mul, add, sub, div) maps directly to engine ops (`engine.Log`,
`engine.Mul`, `engine.Add`, `engine.Sub`, `engine.Div`). Using engine ops would
enable GPU-accelerated loss computation.

`RoutingContrastive` has triple-nested `for b/i/j` loops computing pairwise
cosine similarity on raw slices. Could use `engine.MatMul` for batch pairwise
dot products and `engine.ReduceSum` for norms.

`QuantileLoss` breaks generics entirely with `any(targetData[i]).(float32)` --
this panics at runtime for any type other than `float32`.


### 14. training/optimizer/ -- Raw-Slice Gradient Operations

The canonical `AdamW[T]` optimizer properly uses `engine.MulScalar`,
`engine.Add`, etc. for the core parameter update. However:

| Method | File:line | Issue |
|--------|-----------|-------|
| `guardAndClipGradients` | training/optimizer/adamw.go:236 | Iterates every element of every gradient via `.Data()` -- forces D2H transfer on GPU, defeats CUDA graph capture |
| `AdamW8bit.Step` | training/optimizer/adamw8bit.go | Entire 8-bit Adam update on raw slices, zero Engine usage |
| `SGD.Step` | training/optimizer/sgd.go | Allocates O(N) tensor to broadcast learning rate instead of using `engine.MulScalar` |

`guardAndClipGradients` is called on every `Step()`. For GPU tensors, each
`.Data()` call triggers a D2H copy. With 100+ parameters, this means 100+
synchronous D2H copies per optimization step. The global norm computation
should use `engine.Mul` + `engine.ReduceSum` and the scaling should use
`engine.MulScalar`.

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
adds measurable overhead. They are not composability violations -- they are
optimized implementations of composed operations.

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

| Reimplemented Component | Instances | Est. Redundant Lines |
|------------------------|-----------|---------------------|
| Linear/MatMul/MLP layer | 18 | ~2,400 |
| LayerNorm (all variants) | 17 | ~1,800 |
| Multi-head attention | 8 | ~2,000 |
| GELU activation | 4 | ~120 |
| Softmax | 6 | ~200 |
| AdamW optimizer | 5 | ~400 |
| MSE loss | 2 | ~40 |
| BCE/Quantile/Contrastive loss (raw) | 3 | ~470 |
| Gradient clipping (raw .Data()) | 3 | ~90 |
| Xavier initialization | 5 | ~100 |
| ReLU activation | 5 | ~50 |
| Full backward passes | 3 | ~3,000 |
| SGD optimizer | 5+ | ~150 |
| Intra-layers/ duplication | 10 | ~400 |
| **Total** | **90+** | **~11,900** |

### Packages That Follow the Principle

| Package | Pattern | Imports from layers/ |
|---------|---------|---------------------|
| `inference/arch_common.go` | Exemplar | attention, normalization, core, embeddings |
| `inference/arch_llama.go` | Good | Via arch_common |
| `inference/arch_gemma.go` | Good | Via arch_common |
| `inference/arch_mistral.go` | Good | Via arch_common |
| `inference/arch_qwen.go` | Good | Via arch_common |
| `inference/arch_phi.go` | Good | Via arch_common |
| `inference/arch_deepseek.go` | Good | Via arch_common + 2 custom |
| `layers/transformer/` | Good | Composes attention, normalization, core |
| `layers/vision/` | Good | Composes core, normalization |
| `layers/audio/` | Good | Composes core |
| `layers/hrm/` | Good | Composes transformer |
| `layers/ssm/` | Good | Composes core |
| `layers/timeseries/` | Good | Composes core, normalization |
| `model/hrm/` | Good | Composes layers/core, layers/hrm |

### Packages That Violate the Principle

| Package | Severity | Lines | Uses Engine[T]? | Uses layers/? |
|---------|----------|-------|-----------------|---------------|
| `timeseries/` | S1 | 18,197 | Partial (_engine.go files) | 2 of 34 files |
| `tabular/` | S1 | 4,010 | Partial (MatMul only) | No |
| `crossasset/` | S1 | 918+851 | Partial (GPU path) | No |
| `modeldsl/` | S1 | 696 | No | No |
| `rl/` | S1 | 787 | No | No |
| `gnn/` | S1 | 416 | No | No |
| `synth/` | S1 | 676 | No | No |
| `meta/` | S1 | 307 | No | No |
| `shared/` | S1 | 259 | No | No |
| `training/loss/` | S2 | ~470 | Has field, unused | N/A |
| `training/optimizer/` | S2 | ~190 | Partial (D2H in clip) | N/A |
| `inference/` (custom nodes) | S2 | ~1,660 | Yes | Partial |
| `generate/` (KV cache) | S2 | ~1,235 | N/A | N/A |
| `inference/timeseries/` | S2 | ~2,000 | Yes | Partial |

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
   The `mlpLayer` struct appears in 5 separate packages with nearly identical code.

5. **No enforcement mechanism.** There is no lint rule, CI check, or architecture
   test that prevents packages from reimplementing `Engine[T]` operations with raw
   slice math. ADR-027 documented the principle but did not enforce it.

---

## Remediation Strategy

### Priority 1: Extract Shared Training Primitives

The most impactful change is to make `layers/` ergonomic for training workloads.
Currently, training code avoids `layers/` because:
- Creating a `LayerNorm` requires constructing parameters and registering with a
  graph
- Calling forward requires wrapping data in `tensor.TensorNumeric[T]`
- The backward pass requires the graph system

A lightweight functional API on top of `layers/` would eliminate the friction:

```go
// Proposed: layers/functional package
func LayerNorm[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
    x *tensor.TensorNumeric[T], scale, bias *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
```

### Priority 2: Migrate timeseries/ (biggest bang for buck)

The `timeseries/` package has 18,197 lines and 10 model architectures.
Migration order:
1. Replace `layernorm_ops.go` calls with `layers/normalization.LayerNorm`
2. Replace `math_ops.go` (GELU, softmax) with `layers/activations`
3. Replace `training_ops.go` (AdamW, MSE) with `training/optimizer` and
   `training/loss`
4. Replace inline attention in each model with `layers/attention`
5. Replace `linearForward` / `mlpLayer` with `layers/core.Linear`

### Priority 3: Consolidate crossasset/

Rewrite `crossasset/crossasset.go` to compose from `layers/`. The GPU training
path (`gpu_train.go`) should use the same graph-based approach as the inference
pipeline.

### Priority 4: Consolidate tabular/

5 identical `linearForward` methods -> 1 shared function using `layers/core.Linear`.
3 identical `layerNorm` methods -> `layers/normalization.LayerNorm`.

### Priority 5: Add Architecture Test

```go
// go test -run TestNoRawSliceMath ./...
func TestNoRawSliceMath(t *testing.T) {
    // Scan non-internal, non-test .go files for patterns that indicate
    // raw slice math where Engine[T] should be used:
    // - tensor.Data() in non-test files outside internal/
    // - manual for loops over tensor elements
    // - import "math" without import layers/
}
```

### Priority 6: Delete or migrate experimental packages

`modeldsl/`, `gnn/`, `rl/`, `synth/`, `meta/`, `shared/` are all small (<800
lines each). Options:
- **Migrate:** Rewrite to compose from `layers/` and `training/`
- **Delete:** If the package is unused or experimental, remove it
- **Flag:** Mark as `// Deprecated: does not compose from layers/` until migrated

---

## Positive Observations

1. **The inference path is exemplary.** `arch_common.go` and the standard
   architecture builders (Llama, Gemma, Mistral, Qwen, Phi) cleanly compose
   from `layers/`. This is the pattern to follow.

2. **layers/ itself is well-designed.** 56+ operations across 19 sub-packages,
   with clear separation of concerns and consistent interfaces. The composition
   principle works when it is followed.

3. **The _engine.go migration pattern works.** Files like `patchtst_engine.go`
   and `itransformer_engine.go` show that migrating raw-slice code to `Engine[T]`
   is feasible and incremental. The pattern just needs to be completed.

4. **ADR-027 correctly identified the problem.** The 12 composition violations
   in `layers/` were documented and a remediation plan was created. The same
   rigor needs to be applied to the 12 packages identified in this audit.

5. **Performance-justified exceptions are documented.** Fused CUDA kernels,
   SIMD assembly, and megakernel codegen are correctly treated as optimized
   implementations, not composability violations.
