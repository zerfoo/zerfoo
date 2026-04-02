# Dirty Architecture Report

## Date: 2026-04-02

## Design Principle Under Review

Zerfoo's stated design goal is to **build complex components by composing smaller
components**. The `layers/` package provides 56+ neural network operations across 18
sub-packages (attention, normalization, activations, core, embeddings, vision, audio,
ssm). The `compute.Engine[T]` interface provides type-safe tensor arithmetic. Complex
models should compose these building blocks rather than reimplementing low-level math.

This report identifies every place where the codebase violates this principle.

---

## Executive Summary

The composition principle is well-followed in the **inference** path (GGUF model
builders compose layers/ extensively -- 70 imports from inference/). It is **severely
violated** in the **training** path, where 5 major packages (timeseries, crossasset,
tabular, gnn, modeldsl) each reimplement fundamental math operations from scratch
instead of composing from layers/ or engine ops.

The result is:

| Operation | Distinct Reimplementations | Should Be |
|-----------|--------------------------|-----------|
| softmax | 15 | 1 (engine.Softmax or layers/activations) |
| GELU | 13 | 1 (layers/activations.GELU or engine op) |
| sigmoid | 7 | 1 (layers/activations) |
| layerNorm | 67 function references | 1 (layers/normalization.LayerNorm) |
| matmul/matvecmul | 33 | 1 (engine.MatMul) |
| AdamW optimizer | 5 distinct implementations | 1 (training/optimizer.AdamW) |
| backward pass logic | 219 backward functions | Centralized in layers/ |

---

## Finding 1: Training Backends Bypass layers/ Entirely

**Severity: HIGH -- Unjustified**

Of the 9 timeseries backends, **8 out of 9** do not import from `layers/` at all.
Only `mamba.go` composes layers (importing `layers/core` and `layers/ssm`).

| Backend | Lines | Imports layers/ | Verdict |
|---------|-------|----------------|---------|
| patchtst.go | 951 | NO | VIOLATION |
| itransformer.go | 646 | NO | VIOLATION |
| nhits.go | 1121 | NO | VIOLATION |
| dlinear.go | 532 | NO | VIOLATION |
| timemixer.go | 577 | NO | VIOLATION |
| frets.go | 925 | NO | VIOLATION |
| cfc.go | 842 | NO | VIOLATION |
| ttm.go | 1537 | NO | VIOLATION |
| mamba.go | 701 | YES | GOOD |

Each of these backends reimplements its own:
- Matrix multiplication (raw slice loops)
- Layer normalization (layerNormF64, layerNorm1D, etc.)
- GELU activation (geluScalar, geluMatrix)
- Softmax (softmaxF64)
- Attention (manual Q/K/V dot products)
- Backward passes (manual gradient computation)

**Why this is unjustified:** The `layers/` package already provides all of these
operations with proper GPU support via `compute.Engine[T]`. The inference path
uses them successfully for 12+ model architectures. The training backends predate
the layers/ package and were never migrated.

**Impact:** Every bug fix or optimization to layers/ (e.g., fused kernel support,
FP16 paths, CUDA graph compatibility) must be manually duplicated across 8
backends. The GPU training path (`patchtst_gpu_train.go`) bypasses layers/ and
reimplements the encoder forward/backward inline, which is why E53 (unified
training forward/backward) was needed as a separate epic.

---

## Finding 2: crossasset Package Reimplements All Math

**Severity: HIGH -- Unjustified**

`crossasset/crossasset.go` (539 lines) implements its own:

| Function | Line | What It Does | Should Use |
|----------|------|-------------|------------|
| `matVecMul` | 471 | W @ x manual loop | engine.MatMul |
| `vecAdd` | 484 | dst += src loop | engine.Add |
| `softmax` | 491 | exp/sum manual | engine.Softmax |
| `layerNorm` | 511 | mean/var/scale manual | layers/normalization.LayerNorm |
| `gelu` | 537 | tanh approximation | layers/activations.GELU |

`crossasset/gpu_train.go` (981 lines) then reimplements ALL of these AGAIN as
GPU-compatible versions:

| Function | Line | What It Does |
|----------|------|-------------|
| `cpuLayerNorm` | 848 | LayerNorm on CPU slices |
| `cpuGELU` | 869 | GELU on CPU slices |
| `cpuGELUBackward` | 880 | GELU backward on CPU |
| `cpuSoftmaxBackward` | 898 | Softmax backward on CPU |
| `adamWUpdate` | 947 | AdamW (copied from timeseries/) |
| `clipGrads` | 935 | Gradient clipping (copied from timeseries/) |

The crossasset package has **zero imports from layers/**. It reimplements a complete
cross-attention transformer from raw float64 slice operations, then reimplements
the same thing again in float32 for GPU training.

**Why this is unjustified:** The layers/attention/GroupedQueryAttention already
implements multi-head cross-attention with engine ops, GPU support, and CUDA
graph compatibility.

---

## Finding 3: tabular Package Reimplements Math Ops

**Severity: MEDIUM -- Partially Justified**

`tabular/` has zero imports from `layers/`. Files affected:

| File | Function | Line |
|------|----------|------|
| tabular/tabnet.go | `sigmoid` | 350 |
| tabular/model.go | `geluScalar` | 212 |
| tabular/train.go | `geluGradScalar` | 437 |

The tabular package implements TabNet, SAINT, and other tabular models using raw
float32 slice operations. The models are simpler (no attention, no normalization
layers), so the violation is less severe, but `geluScalar` and `sigmoid` are
still unnecessary reimplementations.

**Partial justification:** Tabular models use sparse attention (TabNet's
attentive transformer) which differs from the dense attention in layers/.
However, the activation functions are identical and should be shared.

---

## Finding 4: gnn Package Reimplements Matrix Operations

**Severity: MEDIUM -- Unjustified**

`gnn/gcn.go` (line 206+) reimplements:
- `matMul` (line 206) -- manual 2D matrix multiply
- `matMulTransposeA` (line 226) -- A^T @ B
- `softmaxMatrix` (line 281) -- 2D softmax

These should use `engine.MatMul` and `engine.Softmax`. The GNN package operates
on float64 2D slices (`[][]float64`) instead of tensors, which prevents engine
composition. This is a data representation mismatch, not a performance choice.

---

## Finding 5: modeldsl Package Reimplements Training Pipeline

**Severity: MEDIUM -- Unjustified**

`modeldsl/` implements its own forward/backward passes with private softmax
and training loop:

| File | Function | Line |
|------|----------|------|
| modeldsl/model.go | `softmaxLayer.forward` | 176 |
| modeldsl/train.go | `softmaxLayerT.forward` | 341 |
| modeldsl/train.go | `softmaxLayerT.backward` | 365 |

The modeldsl package provides a DSL for building custom models. It should compose
from layers/ rather than reimplementing layers as private types.

---

## Finding 6: God Objects in ztensor

**Severity: MEDIUM -- Performance Justified, But Excessive**

| File | Lines | Methods | Description |
|------|-------|---------|-------------|
| compute/gpu_engine.go | 4318 | 94 | GPU engine implementation |
| compute/cpu_engine.go | 2465 | 59 | CPU engine implementation |
| compute/gpu_kernels.go | 1414 | -- | GPU kernel wrappers |
| internal/gpuapi/kernels.go | -- | 71 | KernelRunner interface |

The GPUEngine has **94 methods**. The KernelRunner interface has **71 methods**.
These are god objects by any measure.

**Partial justification:** GPU engines need direct kernel dispatch for performance.
You can't compose softmax from smaller kernels without losing fusion opportunities.
The kernel runner interface is wide because each kernel is a distinct GPU program.

**Unjustified aspect:** The `matMulQ4K`, `matMulQ5_0`, `matMulQ6K`, `matMulQ4KBWeight`,
`matMulQ5_0BWeight`, `matMulQ6KBWeight` methods are copy-paste variants that differ
only in the storage type and block size. These could be refactored into a single
generic quantized matmul dispatcher.

---

## Finding 7: Duplicate AdamW Implementations

**Severity: HIGH -- Unjustified**

There are **5 distinct AdamW implementations**:

| Location | Type | Used By |
|----------|------|---------|
| `training/optimizer/adamw.go` | Generic `[T tensor.Numeric]` | training.Trainer |
| `training/optimizer/adamw8bit.go` | 8-bit quantized | -- |
| `timeseries/adamw_f32.go` | float32 CPU | patchtst_gpu_train |
| `timeseries/training_ops.go` | float64 CPU | timeseries.Trainer |
| `crossasset/gpu_train.go` | float32 CPU (copy) | crossasset.TrainGPU |
| `distributed/fsdp/optimizer_shard.go` | Sharded float32 | FSDP training |

The canonical implementation is `training/optimizer/adamw.go` which uses
`compute.Engine[T]` and works on GPU. The timeseries and crossasset packages
ignore it and implement their own CPU-only versions.

**Why this is unjustified:** `training/optimizer.AdamW[T]` already handles
float32 and float64. The CPU-only copies exist because the timeseries
training loop manages gradients as raw slices instead of graph parameters.

---

## Finding 8: Inference Builders Follow Composition (POSITIVE)

The inference path is the **exemplar** of the composition principle:

- `inference/arch_common.go` composes:
  - `layers/attention.GroupedQueryAttention`
  - `layers/normalization.RMSNorm`
  - `layers/embeddings.RotaryPositionalEmbedding`
  - `layers/core.Linear`
  - `layers/activations.SiLU`, `GELU`, etc.

- 70 imports from `layers/` in the inference package
- Each architecture builder (Llama, Gemma, Mistral, etc.) creates a graph of
  composable nodes, each wrapping a layer from `layers/`
- The graph system enables CUDA graph capture, fusion passes, and megakernel codegen

This is exactly how the composition principle should work.

---

## Finding 9: Duplicate layerNorm Implementations

**Severity: HIGH -- Unjustified**

Layer normalization has the most reimplementations of any operation:

| Location | Variant |
|----------|---------|
| `layers/normalization/layer_norm.go` | Canonical, engine-based |
| `layers/normalization/rms_norm.go` | RMSNorm variant, engine-based |
| `timeseries/layernorm_ops.go` | 5 functions: layerNormF64, WithCache, Backward, 1D, 1DCached |
| `timeseries/patchtst_encoder.go` | layerNormForwardWithEngine, layerNormBackwardWithEngine |
| `timeseries/itransformer_backward.go` | layerNormBackward (line 496) |
| `crossasset/crossasset.go` | layerNorm (line 511) |
| `crossasset/gpu_train.go` | cpuLayerNorm (line 848) |
| `inference/arch_falcon.go` | falconLayerNormNode (line 384) |

That's **8+ separate implementations** of the same mathematical operation. The
canonical versions in `layers/normalization/` are GPU-compatible and well-tested.

---

## Finding 10: Separate CPU and GPU Forward Paths

**Severity: MEDIUM -- Historical Debt**

The PatchTST backend has TWO complete forward pass implementations:

1. `timeseries/patchtst.go` -- CPU forward pass using raw float64 slices
2. `timeseries/patchtst_gpu_train.go` -- GPU forward pass using engine ops

These implement the same mathematical operations but in completely different
code paths. When a bug is fixed in one, the other may not get the fix. The
E53 epic (unified training forward/backward) was created specifically to
address this, but even after E53, both paths persist.

**The same pattern exists in crossasset:**
1. `crossasset/crossasset.go` -- CPU forward (float64, raw slices)
2. `crossasset/gpu_train.go` -- GPU forward (float32, engine ops)

---

## Justified Performance Exceptions

These reimplementations are **justified** because they exist for performance
reasons that composition cannot achieve:

| Component | Why Custom | Performance Gain |
|-----------|-----------|-----------------|
| CUDA GEMV kernels (Q4_0, Q4_K, Q5_0, etc.) | Fused dequant+multiply avoids intermediate F32 buffer | 10-100x vs dequant then GEMV |
| Fused RoPE+QKNorm kernel | Single kernel launch vs 4 separate launches | ~4x fewer kernel launches |
| Fused softmax+V multiply | Avoids materializing attention weight tensor | ~2x memory, 1 fewer kernel |
| RepeatInterleave kernel | Avoids reshape+repeat+reshape chain | 3 fewer kernel launches |
| ARM NEON SIMD assembly | Hand-tuned SIMD for GEMM hot loop | 2-4x vs Go scalar |
| Separated Q4 GPU layout | Enables 128-bit coalesced loads | ~1.5x memory throughput |

These are **not** violations of the composition principle. They are optimized
implementations of operations that happen to be performance-critical.

---

## Root Cause Analysis

The violations share a common root cause: **the training backends were written
before the layers/ package existed** (or before it was mature enough). Each
timeseries backend was initially a standalone prototype with its own math
utilities. As the framework matured, layers/ was built for the inference path,
but the training backends were never migrated to use it.

The composition gap is:

```
layers/ (well-composed)
  ^
  | used by
  |
inference/ (composes layers/)    timeseries/ (reimplements everything)
                                 crossasset/ (reimplements everything)
                                 tabular/ (reimplements basics)
                                 gnn/ (reimplements matrix ops)
                                 modeldsl/ (reimplements training)
```

---

## Remediation Roadmap

### Phase 1: Unify Math Primitives (1-2 days)

Extract the duplicated scalar/slice operations into a shared package that
both CPU training and engine-based training can use:

1. Create `internal/mathops/` with:
   - `SoftmaxF64(x []float64) []float64`
   - `SoftmaxF32(x []float32) []float32`
   - `GELUF64(x float64) float64`
   - `GELUF32(x float32) float32`
   - `GELUDerivF64/F32`
   - `SigmoidF64/F32`
   - `LayerNormF64(x, gamma, beta []float64) []float64`

   Note: `timeseries/math_ops.go` and `timeseries/layernorm_ops.go` already
   contain shared versions. Move them to `internal/mathops/` and have all
   packages import from there.

2. Delete the private reimplementations in crossasset, tabular, gnn, modeldsl.

### Phase 2: Migrate Training to Engine Ops (1 week)

For each training backend:
1. Replace raw slice math with `compute.Engine[T]` calls
2. Remove the dual CPU/GPU forward pass -- use engine ops which work on both
3. Compose layers from `layers/` where possible (Linear, LayerNorm, GELU, Attention)

Priority order by impact:
1. crossasset (smallest, just shipped GPU training -- easiest to refactor)
2. patchtst (most complex, but has the encoder extraction started in E53)
3. itransformer
4. nhits, dlinear, frets, cfc, timemixer, ttm

### Phase 3: Consolidate AdamW (1 day)

Export `training/optimizer.AdamW[T]` as the single AdamW implementation.
Make the timeseries trainer use it. Delete the 4 copies.

### Phase 4: Decompose God Objects (1 week)

1. Split `gpu_engine.go` (4318 lines, 94 methods):
   - `gpu_engine_matmul.go` -- all quantized matmul variants
   - `gpu_engine_elementwise.go` -- add/sub/mul/div/scalar ops
   - `gpu_engine_reduction.go` -- softmax/sum/argmax
   - `gpu_engine_memory.go` -- upload/gather/copy
   - `gpu_engine.go` -- core struct + lifecycle

2. Refactor quantized matmul variants into a generic dispatcher:
   ```go
   func (e *GPUEngine[T]) matMulQuantizedBWeight(ctx, a, storage quantStorage, b, dst) {
       // dispatch by storage type to upload + GEMV/dequant+GEMM
   }
   ```

---

## Statistics

| Metric | Value |
|--------|-------|
| Packages violating composition | 5 (timeseries, crossasset, tabular, gnn, modeldsl) |
| Packages following composition | 2 (inference, layers) |
| Reimplemented softmax functions | 15 |
| Reimplemented GELU functions | 13 |
| Reimplemented sigmoid functions | 7 |
| Reimplemented layerNorm variants | 67 references |
| Reimplemented matmul variants | 33 |
| Distinct AdamW implementations | 5 |
| Total backward pass functions | 219 |
| God files (>800 lines, non-test) | 26 |
| GPUEngine methods | 94 |
| KernelRunner interface methods | 71 |
| Files read for this review | 200+ |
