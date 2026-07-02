# ADR 027: Enforce Layer Composition Before Megakernel

## Status
Accepted

## Date
2026-03-07

## Context
Phase 34 Track B (megakernel) generates a single CUDA kernel by walking the
ExecutionPlan instruction tape -- a flat list of primitive Engine ops (Add,
MatMul, RMSNorm, Softmax, etc.). This only works if ALL layers decompose into
these primitives. If a layer does complex math inline (direct tensor.Data()
access, manual loops, external library calls), those computations are invisible
to the instruction tape and will not appear in the megakernel.

A 5-agent parallel audit of the entire layers/ directory found 12 composition
violations across 11 files:

Critical (block Gemma 3 inference path):
1. MatMulNBits -- inline Q4 dequantization with manual unpacking
2. QKNorm -- manual RMS normalization with tensor.Data() loops

Moderate (affect other models or training):
3. BatchNormalization -- manual per-channel normalization loops
4. Conv2d -- 6-nested-loop convolution with direct data access
5. MoEGate -- manual expert routing with tensor.Data() loops
6. MixtureOfExperts -- manual token extraction and weighted sum
7. PolynomialExpansion -- manual power computation with tensor.Data()
8. SpectralFingerprint (core) -- inline DFT with manual cos/sin
9. S4 -- 4-nested-loop diagonal SSM scan with tensor.Data()
10. SpectralFeature (features) -- external Gonum FFT library call
11. Gelu -- math.Tanh inline instead of engine.Tanh
12. LocalAttention mask -- manual mask fill with tensor.Data()

Additionally, Dropout and FeatureDropout generate random masks inline instead
of using engine.RandomUniform(), though this is less critical for inference
(dropout is disabled during inference).

## Decision
Before starting Track A or Track B, complete a prerequisite "Track 0" (E96)
that refactors all 12 violated layers to compose from Engine primitives. Priority
order is based on the Gemma 3 inference path:

1. MatMulNBits, QKNorm -- used in every Gemma 3 forward pass
2. Gelu, BatchNorm, LocalAttention mask -- used in some model variants
3. Conv2d, MoE, Polynomial, Spectral, S4 -- used in non-Gemma models

Each refactoring replaces inline math with engine method calls, ensuring the
operation appears in the instruction tape. Existing tests verify correctness
(output parity within tolerance).

## Consequences
Positive:
- Every layer's computation is visible in the ExecutionPlan instruction tape
- Megakernel code generator covers 100% of ops for any model
- Uniform composition pattern across the codebase
- Easier to add new Engine backends (each layer automatically works)

Negative:
- Some refactored layers may be slightly slower on CPU due to extra engine
  dispatch overhead (e.g., S4 scanning loop becoming per-step engine calls)
- MatMulNBits dequant refactoring may require a new engine.DequantQ4 primitive
- Conv2d im2col+MatMul decomposition uses more memory than fused convolution
- Increased number of engine methods if new primitives are needed
