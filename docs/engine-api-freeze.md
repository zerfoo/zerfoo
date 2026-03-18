# Engine[T] API Freeze for v1.0

**Status:** Frozen
**Date:** 2026-03-18
**ADR:** Supplements ADR-037 (GGUF sole format) and ADR-044+ (v1.0 roadmap)

## Overview

`compute.Engine[T tensor.Numeric]` is the central compute abstraction in
`github.com/zerfoo/ztensor/compute`. Every tensor arithmetic operation in the
Zerfoo ecosystem flows through this interface, enabling transparent CPU/GPU
switching and CUDA graph capture. This document catalogs the frozen v1.0
surface and defines extension interfaces for specialized hardware capabilities.

## Core Interface — `Engine[T tensor.Numeric]`

All implementations **must** provide every method below. The interface is
defined in `compute/engine.go`.

### Type Metadata

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 1 | `Ops` | `() numeric.Arithmetic[T]` | Returns type-specific arithmetic operations |

### Element-wise Arithmetic

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 2 | `Add` | `(ctx, a, b, dst...) (*Tensor, error)` | Element-wise addition with broadcasting |
| 3 | `Sub` | `(ctx, a, b, dst...) (*Tensor, error)` | Element-wise subtraction with broadcasting |
| 4 | `Mul` | `(ctx, a, b, dst...) (*Tensor, error)` | Element-wise multiplication with broadcasting |
| 5 | `Div` | `(ctx, a, b, dst...) (*Tensor, error)` | Element-wise division with broadcasting |
| 6 | `Pow` | `(ctx, base, exp, dst...) (*Tensor, error)` | Element-wise power |
| 7 | `MulScalar` | `(ctx, a, scalar, dst...) (*Tensor, error)` | Multiply tensor by scalar |
| 8 | `DivScalar` | `(ctx, a, scalar, dst...) (*Tensor, error)` | Divide tensor by scalar |
| 9 | `AddScalar` | `(ctx, a, scalar, dst...) (*Tensor, error)` | Add scalar to tensor |

### Unary Math

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 10 | `UnaryOp` | `(ctx, a, func(T)T, dst...) (*Tensor, error)` | Apply arbitrary unary function |
| 11 | `Exp` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise exponential |
| 12 | `Log` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise natural logarithm |
| 13 | `Sin` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise sine |
| 14 | `Cos` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise cosine |
| 15 | `Tanh` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise hyperbolic tangent |
| 16 | `TanhPrime` | `(ctx, a, upstream, dst...) (*Tensor, error)` | Tanh gradient for backprop |
| 17 | `Sqrt` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise square root |
| 18 | `Rsqrt` | `(ctx, a, dst...) (*Tensor, error)` | Element-wise reciprocal square root |

### Linear Algebra

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 19 | `MatMul` | `(ctx, a, b, dst...) (*Tensor, error)` | Matrix multiplication (2D+) |

### Reduction

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 20 | `Sum` | `(ctx, a, axis, keepDims, dst...) (*Tensor, error)` | Sum along axis |
| 21 | `ReduceSum` | `(ctx, a, axis, keepDims, dst...) (*Tensor, error)` | Reduction sum (optimized path) |
| 22 | `ReduceMean` | `(ctx, a, axis, keepDims, dst...) (*Tensor, error)` | Mean along axis |
| 23 | `Softmax` | `(ctx, a, axis, dst...) (*Tensor, error)` | Softmax along axis |

### Shape Manipulation

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 24 | `Transpose` | `(ctx, a, axes, dst...) (*Tensor, error)` | Transpose along given axes |
| 25 | `Reshape` | `(ctx, a, shape, dst...) (*Tensor, error)` | Reshape without copying data |
| 26 | `Split` | `(ctx, a, numSplits, axis) ([]*Tensor, error)` | Split tensor along axis |
| 27 | `Concat` | `(ctx, tensors, axis, dst...) (*Tensor, error)` | Concatenate tensors along axis |
| 28 | `Repeat` | `(ctx, a, axis, reps, dst...) (*Tensor, error)` | Repeat tensor along axis |

### Embedding / Scatter

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 29 | `Gather` | `(ctx, params, indices, output) error` | Embedding-style gather (2D params, 1D/2D indices) |
| 30 | `ScatterAdd` | `(ctx, dEmbedTable, indices, dOut) error` | Scatter-add for embedding gradients |

### Initialization / Memory

| # | Method | Signature | Purpose |
|---|--------|-----------|---------|
| 31 | `Zero` | `(ctx, a) error` | Set all elements to zero |
| 32 | `Zeros` | `(ctx, a, shape) error` | Fill with zeros, optionally reallocate shape |
| 33 | `Copy` | `(ctx, dst, src) error` | Copy data between tensors |
| 34 | `Fill` | `(ctx, t, value) error` | Fill tensor with scalar |
| 35 | `RandomUniform` | `(ctx, t, min, max) error` | Fill with uniform random values |
| 36 | `OneHot` | `(ctx, input, depth, dst...) (*Tensor, error)` | One-hot encoding |

**Total: 36 methods** in the core interface.

## Extension Interfaces

Extension interfaces are **optional**. Callers use Go type assertions
(`eng.(FusedRoPEProvider[T])`) to check availability at runtime. All extension
interfaces are defined in the `compute/` package alongside `Engine[T]`.

### Fused Kernel Providers

These eliminate kernel-launch overhead by combining multiple operations.

| Interface | Method | File | Purpose |
|-----------|--------|------|---------|
| `FusedRMSNormer` | `FusedRMSNormGPU(input, weight, eps)` | `engine.go` | Fused RMSNorm on GPU (float32 only) |
| `FusedAddRMSNormProvider[T]` | `GPUFusedAddRMSNorm(input, residual, weight, eps)` | `fused_add_rmsnorm.go` | Fused residual-add + RMSNorm |
| `FusedNormAddProvider[T]` | `GPUFusedNormAdd(input, weight, residual, eps)` | `fused_norm_add.go` | Fused RMSNorm + element-wise add |
| `FusedScaledSoftmaxProvider[T]` | `GPUScaledSoftmax(input, scale, axis)` | `fused_scaled_softmax.go` | Fused scale + softmax |
| `FusedRoPEProvider[T]` | `GPUFusedRoPE(input, cos, sin, rotaryDim)` | `fused_rope.go` | Fused rotary position embedding |
| `FusedSwiGLUProvider[T]` | `GPUFusedSwiGLU(w1, w3)` | `fused_swiglu.go` | Fused SwiGLU activation |
| `FusedQKNormRoPEProvider[T]` | `GPUFusedQKNormRoPE(input, wQ, wK, cos, sin, eps, ...)` | `fused_qk_norm_rope.go` | Fused QK-norm + RoPE (4 kernels -> 1) |

### Mixed-Precision / Quantized

| Interface | Method | File | Purpose |
|-----------|--------|------|---------|
| `FP16ToF32Converter` | `ConvertFP16ToF32(t)` | `engine.go` | Convert Float16Storage tensor to float32 GPU tensor |
| `TransposeBMatMuler[T]` | `MatMulTransposeB(ctx, a, b, dst...)` | `engine.go` | C = A * B^T without explicit transpose allocation |
| `W4A16MatMuler[T]` | `MatMulW4A16(ctx, a, b, dst...)` | `w4a16.go` | Mixed 4-bit weight / FP16 activation MatMul |

### GPU Infrastructure

| Interface | Method(s) | File | Purpose |
|-----------|-----------|------|---------|
| `StreamProvider` | `Stream() unsafe.Pointer` | `engine.go` | Expose GPU stream for CUDA graph capture |
| `GPUStreamAccessor` | `GPUStream() gpuapi.Stream` | `engine.go` | Typed GPU stream for async memory ops |
| `PoolResetter` | `ResetPool()` | `engine.go` | O(1) arena reset between forward passes |
| `WeightUploader` | `UploadWeights([]*Tensor) error` | `engine.go` | Pre-upload weights to device memory |
| `GPUArgmaxer` | `GPUArgmax(t) (int, error)` | `engine.go` | GPU-side argmax (avoids ~1MB D2H copy per token) |

### Paged Attention

| Interface | Method(s) | File | Purpose |
|-----------|-----------|------|---------|
| `PagedGQAer` | `PagedGQA(Q, blockPtrsK, blockPtrsV, blockIndices, ...)` | `engine.go` | Paged grouped-query attention via block-table indirection |
| | `IsPagedGQASupported() bool` | | Runtime availability check |

## Proposed Extension Interfaces for v1.0

The following extension interfaces are **recommended** for formalization before
the v1.0 freeze. They capture patterns already present in the codebase as
unexported methods or ad-hoc type assertions.

### `EngineWithFP8`

FP8 E4M3 support is currently handled inside `GPUEngine.MatMul` via storage
type inspection. Formalizing it as an extension interface makes FP8 capability
discoverable and testable.

```go
// EngineWithFP8 is an optional interface for engines that support
// FP8 E4M3FN quantized matrix multiplication.
type EngineWithFP8 interface {
    // MatMulFP8 performs C = dequant(A_fp8) * B_f32 or C = A_f32 * dequant(B_fp8).
    // The FP8 operand is identified by its FP8E4M3Storage.
    MatMulFP8(ctx context.Context, a, b *tensor.TensorNumeric[float32],
        dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)

    // IsFP8Supported returns true if the hardware supports FP8 operations.
    IsFP8Supported() bool
}
```

### `EngineWithPagedKV`

Paged KV cache management is tightly coupled to the attention kernel. An
extension interface makes the capability explicit for serving-layer code.

```go
// EngineWithPagedKV extends PagedGQAer with KV cache block management.
type EngineWithPagedKV interface {
    PagedGQAer

    // AllocKVBlocks allocates N blocks of [blockSize, headDim] on device.
    AllocKVBlocks(n, blockSize, headDim, numKVHeads int) (blockPtrs unsafe.Pointer, err error)

    // FreeKVBlocks releases previously allocated KV blocks.
    FreeKVBlocks(blockPtrs unsafe.Pointer, n int) error

    // AppendKVToken appends a single token's K and V vectors to the paged cache.
    AppendKVToken(blockPtrs unsafe.Pointer, blockIndices unsafe.Pointer,
        k, v *tensor.TensorNumeric[float32], seqPos, blockSize, headDim int) error
}
```

### `EngineWithBF16`

BFloat16 is used in some model weights. A formal extension makes BF16
capability discoverable.

```go
// EngineWithBF16 is an optional interface for engines that support
// BFloat16 arithmetic or mixed BF16/F32 operations.
type EngineWithBF16 interface {
    // MatMulBF16 performs matrix multiplication with BF16 inputs.
    MatMulBF16(ctx context.Context, a, b *tensor.TensorNumeric[float32],
        dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)

    // IsBF16Supported returns true if the hardware supports BF16 operations.
    IsBF16Supported() bool
}
```

## Implementation Matrix

| Capability | CPUEngine | GPUEngine | ROCmEngine | OpenCLEngine |
|-----------|-----------|-----------|------------|-------------|
| **Core Engine[T]** | Full | Full | Partial | Partial |
| FusedRMSNormer | -- | Yes | -- | -- |
| FusedAddRMSNormProvider | -- | Yes | -- | -- |
| FusedNormAddProvider | -- | Yes | -- | -- |
| FusedScaledSoftmaxProvider | -- | Yes | -- | -- |
| FusedRoPEProvider | -- | Yes | -- | -- |
| FusedSwiGLUProvider | -- | Yes | -- | -- |
| FusedQKNormRoPEProvider | -- | Yes | -- | -- |
| FP16ToF32Converter | -- | Yes | -- | -- |
| TransposeBMatMuler | -- | Yes | -- | -- |
| W4A16MatMuler | -- | Yes (planned) | -- | -- |
| StreamProvider | -- | Yes | -- | -- |
| GPUStreamAccessor | -- | Yes | -- | -- |
| PoolResetter | -- | Yes | -- | -- |
| WeightUploader | -- | Yes | -- | -- |
| GPUArgmaxer | -- | Yes | -- | -- |
| PagedGQAer | -- | Yes | -- | -- |
| EngineWithFP8 (proposed) | -- | Implicit | -- | -- |
| EngineWithPagedKV (proposed) | -- | Partial | -- | -- |
| EngineWithBF16 (proposed) | -- | Implicit | -- | -- |

## Recommendations Before v1.0 Freeze

### 1. Consolidate `Sum` and `ReduceSum`

`Sum` and `ReduceSum` have identical signatures and overlapping semantics. The
doc comment on `ReduceSum` says "potentially with different internal handling."
For v1.0 clarity, either:

- **Option A:** Remove `ReduceSum`, keep `Sum` (fewer methods, simpler).
- **Option B:** Differentiate clearly in docs — e.g., `Sum` for graph-traced
  reductions, `ReduceSum` for eager-mode optimized reductions.

**Recommendation:** Option A. One method, one name.

### 2. Make `FusedRMSNormer` Generic

`FusedRMSNormer` is the only fused interface that uses concrete `float32`
instead of the generic `[T tensor.Numeric]` pattern used by all other fused
providers. This inconsistency should be resolved:

```go
// Before (inconsistent):
type FusedRMSNormer interface {
    FusedRMSNormGPU(input, weight *tensor.TensorNumeric[float32], eps float32) (...)
}

// After (consistent with other providers):
type FusedRMSNormer[T tensor.Numeric] interface {
    FusedRMSNormGPU(input, weight *tensor.TensorNumeric[T], eps float32) (...)
}
```

### 3. Formalize FP8 as an Extension Interface

FP8 MatMul dispatch currently happens via storage-type inspection deep inside
`GPUEngine.MatMul`. Extracting it into `EngineWithFP8` (see above) would:

- Make FP8 capability explicitly discoverable
- Enable testing with mock FP8 engines
- Prepare for multi-backend FP8 (ROCm HIP has FP8 support)

### 4. Add `Close() error` to Engine[T]

GPU engines hold device memory (arena, scratchpads, BLAS handles). There is no
standard way to release these resources. Adding `Close() error` to the core
interface would enable deterministic cleanup:

```go
type Engine[T tensor.Numeric] interface {
    // ... existing methods ...
    Close() error
}
```

This is a breaking addition that should be evaluated before freeze.

### 5. Standardize `dst ...` Optional Output Pattern

All 21 methods that return `(*Tensor, error)` accept a variadic `dst` parameter
for in-place output. This pattern is good — it enables buffer reuse during
CUDA graph replay. The convention should be documented:

- If `dst[0]` is non-nil, the result is written into it (shape must match).
- If `dst` is empty or `dst[0]` is nil, a new tensor is allocated.
- Implementations must not read from `dst[0]` (it may contain stale data).

## Compatibility Contract

Once frozen at v1.0:

1. **No method removals** from `Engine[T]`.
2. **No signature changes** to existing methods.
3. **New methods** require a new extension interface (not added to core).
4. **Extension interfaces** may be added freely in minor versions.
5. **Behavioral changes** (e.g., broadcasting rules) require an ADR.

## EngineProxy Compatibility

`EngineProxy[T]` (`compute/engine_proxy.go`) wraps `Engine[T]` and forwards
all 36 core methods. It also exposes proxy methods for key extension interfaces:

- `FusedRMSNormGPU` — delegates to `FusedRMSNormer`
- `GPUFusedAddRMSNorm` — delegates to `FusedAddRMSNormProvider[T]`
- `MatMulTransposeB` — delegates to `TransposeBMatMuler[T]` (with fallback)
- `ResetPool` — delegates to `PoolResetter`
- `ArenaUsedBytes` / `SetArenaResetFloor` — delegates via ad-hoc assertions

The proxy also implements `TraceRecorder[T]` integration for computation graph
tracing (used by `graph.CompileTraced`).

Any new extension interface added post-freeze should include an `EngineProxy`
delegation method if it needs to be visible through traced execution.
