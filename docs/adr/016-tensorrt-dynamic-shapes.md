# ADR-016: TensorRT Dynamic Shape Support

**Status:** Accepted
**Date:** 2026-03-03
**Phase:** 19 (TensorRT Dynamic Shapes)

## Context

TensorRT engines are compiled with fixed tensor dimensions by default. This
means a new engine must be built for every distinct input shape (e.g., different
sequence lengths, batch sizes). For transformer models with variable-length
inputs, this is impractical. TensorRT optimization profiles allow specifying
min/opt/max dimensions per input tensor, enabling a single engine to handle a
range of shapes.

## Decision

### 1. C Shim Extensions

Five new functions added to `internal/tensorrt/cshim/trt_capi.h/cpp`:

| Function | Purpose |
|----------|---------|
| `trt_create_optimization_profile` | Creates a new profile from the builder |
| `trt_profile_set_dimensions` | Sets min/opt/max dims for a named input |
| `trt_config_add_optimization_profile` | Attaches profile to builder config |
| `trt_context_set_input_shape` | Sets actual dims before enqueue |
| `trt_context_set_optimization_profile` | Selects the active profile on a context |

### 2. Go Bindings

New types and methods in `internal/tensorrt/tensorrt.go`:

- `OptimizationProfile` type with `SetDimensions` and `AddToConfig`
- `ExecutionContext.SetInputShape` and `SetOptimizationProfile`

### 3. Converter Integration

`DynamicShapeConfig` and `ShapeRange` types in `inference/tensorrt_convert.go`
allow callers to specify per-input shape ranges. `ConvertGraphToTRT` accepts
an optional `*DynamicShapeConfig`. When non-nil:

1. An optimization profile is created after the graph walk
2. `SetDimensions` is called for each input with min/opt/max dims
3. The profile is added to the builder config
4. The profile index is returned in `trtConversionResult`

### 4. Pipeline Integration

`TRTInferenceEngine` in `inference/tensorrt_pipeline.go` tracks whether
dynamic shapes are active. When `dynamic` is true:

- The optimization profile is set on the context at build time
- `Forward` calls `SetInputShape` for each input before `EnqueueV3`
- Actual input tensor shapes are read at inference time

### 5. Cache Key

The cache key incorporates dynamic shape ranges (min/opt/max per input) so
engines built with different shape ranges are cached separately.

## Consequences

**Positive:**
- Single TensorRT engine handles variable batch sizes and sequence lengths
- No re-compilation for different input shapes within the configured range
- Backward-compatible: passing nil DynamicShapeConfig preserves static behavior
- Cache correctly differentiates static vs dynamic engines

**Negative:**
- Dynamic shape engines may be slightly less optimized than fixed-shape engines
- Cannot be hardware-tested without CUDA GPU

**Risks:**
- Profile with very wide min-max range may reduce kernel selection quality
- TensorRT may fall back to slower kernels for shapes far from the opt dims
