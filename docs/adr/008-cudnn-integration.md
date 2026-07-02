# ADR-008: cuDNN Integration

**Phase:** 11
**Status:** Accepted
**Date:** 2026-03-03

## Context

The GPU backend uses cuBLAS for MatMul and 15 custom CUDA kernels for
elementwise/reduction operations. Several operations fall back to CPU because
no GPU implementation exists: Conv2d, BatchNorm (training mode), activation
functions (Sigmoid, ReLU, LeakyReLU, GELU), and pooling (MaxPool, AvgPool,
GlobalAvgPool). These CPU fallbacks require device-to-host transfers that
dominate inference time for vision models and vision-language models (e.g.,
SigLIP encoder in Gemma 3).

cuDNN provides GPU-optimized implementations with automatic algorithm
selection and workspace management. Integration follows the same CGo pattern
as cuBLAS (`compute/gpu_engine.go`) and NCCL (`internal/nccl/`).

## Decision

### CGo Bindings (`internal/cudnn/`)

New package `internal/cudnn/` behind `//go:build cuda` with `-lcudnn` linking.
Types wrap cuDNN C handles:

- **Handle** wraps `cudnnHandle_t`. Created per GPUEngine on the same CUDA
  stream as the cuBLAS handle. Destroyed in `GPUEngine.Close()`.
- **TensorDescriptor** wraps `cudnnTensorDescriptor_t`. Describes NCHW layout.
- **FilterDescriptor** wraps `cudnnFilterDescriptor_t`. Describes conv weights.
- **ConvolutionDescriptor** wraps `cudnnConvolutionDescriptor_t`. Describes
  stride, padding, dilation.
- **ActivationDescriptor** wraps `cudnnActivationDescriptor_t`. Modes: RELU,
  SIGMOID, TANH, (GELU if cuDNN >= 8.1).
- **PoolingDescriptor** wraps `cudnnPoolingDescriptor_t`. Modes: MAX,
  AVERAGE_COUNT_INCLUDE_PADDING.

A `doc.go` without build tag provides package identity for the linter,
following the pattern established in `internal/nccl/doc.go`.

### Forward Operation Bindings

- `cudnnConvolutionForward` with algorithm selected via
  `cudnnGetConvolutionForwardAlgorithm_v7` (heuristic mode).
- `cudnnGetConvolutionForwardWorkspaceSize` to query workspace requirements.
  Workspace allocated from the existing per-device `MemPool`.
- `cudnnBatchNormalizationForwardInference` for inference-mode batch norm.
- `cudnnActivationForward` for ReLU, Sigmoid, Tanh.
- `cudnnPoolingForward` for MaxPool and AvgPool.
- `cudnnSoftmaxForward` for softmax (may replace existing custom kernel if
  performance is better).

### GPUEngine Integration

- `GPUEngine` gains a `cudnnHandle *cudnn.Handle` field alongside the existing
  `cublasHandle`.
- `NewGPUEngine` creates the cuDNN handle after the cuBLAS handle, bound to
  the same CUDA stream.
- `Close()` destroys both handles.
- For each cuDNN-accelerated operation, the GPUEngine method:
  1. Creates descriptors from tensor shapes.
  2. Selects algorithm (for convolution).
  3. Allocates workspace from MemPool.
  4. Calls the cuDNN forward function.
  5. Destroys descriptors.
  6. Falls back to the existing implementation if the cuDNN call fails.

### NCHW Layout

cuDNN natively uses NCHW tensor layout. The existing tensor system uses
row-major layout, which matches NCHW when the shape is (N, C, H, W).
Descriptor setup maps shape dimensions directly without transpose.

### Algorithm Selection

For convolution, cuDNN offers multiple algorithms (IMPLICIT_GEMM, FFT,
WINOGRAD, etc.). Use the heuristic API (`cudnnGetConvolutionForwardAlgorithm_v7`)
for a recommendation. No algorithm caching in the first implementation;
add if profiling shows descriptor + algorithm selection overhead is significant.

## Consequences

### Positive

- Conv2d, BatchNorm, activations, and pooling run on GPU without CPU fallback.
- Eliminates device-to-host transfers for vision model inference.
- cuDNN's auto-tuned kernels are faster than hand-written CUDA kernels for
  standard operations.
- Backwards compatible: cuDNN handle creation is optional (non-CUDA builds
  unaffected). Fallback to existing implementation on cuDNN failure.

### Negative

- Adds libcudnn as a build-time and runtime dependency (gated by build tag).
- Descriptor create/set/destroy overhead for each operation call. Mitigated by
  cuDNN's efficient descriptor management.
- cuDNN version compatibility: minimum cuDNN 8.0 required. Different cuDNN
  versions may produce slightly different numerical results.

### Files Added

- `internal/cudnn/doc.go` -- package identity (no build tag)
- `internal/cudnn/cudnn.go` -- CGo bindings (~540 lines)
- `internal/cudnn/cudnn_test.go` -- descriptor and forward operation tests
- `compute/gpu_cudnn.go` -- cuDNN-accelerated GPUEngine methods (~555 lines)

### Files Modified

- `compute/gpu_engine.go` -- add cudnnHandle field, create/destroy in
  constructor/Close
