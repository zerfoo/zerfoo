# ADR-014: cuDNN Backward Pass

**Status:** Accepted
**Date:** 2026-03-03
**Phase:** 17 (cuDNN Backward Pass)

## Context

The framework had cuDNN-accelerated forward operations (convolution, batch
normalization, activation, pooling, softmax) but training backward passes fell
back to CPU. GPU-accelerated backward operations are essential for training
performance, as gradients must flow through the same operations as the forward
pass without costly GPU-to-CPU-to-GPU round trips.

## Decision

### 1. cuDNN Backward CGo Bindings

The `internal/cudnn/` package was extended with 8 backward methods on Handle:

| Method | cuDNN Function |
|--------|---------------|
| `ConvolutionBackwardData` | `cudnnConvolutionBackwardData` |
| `GetConvolutionBackwardDataWorkspaceSize` | `cudnnGetConvolutionBackwardDataWorkspaceSize` |
| `ConvolutionBackwardFilter` | `cudnnConvolutionBackwardFilter` |
| `GetConvolutionBackwardFilterWorkspaceSize` | `cudnnGetConvolutionBackwardFilterWorkspaceSize` |
| `BatchNormalizationForwardTraining` | `cudnnBatchNormalizationForwardTraining` |
| `BatchNormalizationBackward` | `cudnnBatchNormalizationBackward` |
| `ActivationBackward` | `cudnnActivationBackward` |
| `PoolingBackward` | `cudnnPoolingBackward` |

Two algorithm types were added: `ConvBwdDataAlgo` (4 variants) and
`ConvBwdFilterAlgo` (5 variants). Both default to Algo0 for broad compatibility.

### 2. CUDA DNN Adapter (GRAL Layer)

The `internal/gpuapi/cuda_dnn.go` adapter's backward stubs were replaced with
real implementations. Each method:

1. Creates cuDNN descriptors (tensor, filter, convolution, activation, pooling)
2. Queries workspace size for convolution backward ops
3. Allocates temporary workspace via `cudaMalloc`
4. Calls the cuDNN backward function
5. Frees workspace and destroys descriptors

### 3. GPUEngine Backward Methods

Six methods were added to GPUEngine in `compute/gpu_cudnn.go`:

| Method | Inputs | Outputs |
|--------|--------|---------|
| `Conv2dBackwardData` | w, dy, dxShape, strides, pads, dilations, groups | dx |
| `Conv2dBackwardFilter` | x, dy, dwShape, strides, pads, dilations, groups | dw |
| `BatchNormForwardTraining` | x, scale, bias, runningMean, runningVar, epsilon, expAvgFactor | y, saveMean, saveInvVar |
| `CudnnBatchNormBackward` | x, dy, scale, saveMean, saveInvVar | dx, dScale, dBias |
| `CudnnActivationBackward` | x, y, dy, mode | dx |
| `CudnnPoolingBackward` | x, y, dy, mode, window/pad/stride params | dx |

All methods follow the established forward-pass pattern: get device pointers,
allocate output from memory pool, call DNN adapter, synchronize stream, return
GPU-backed tensor result.

### 4. DNN Interface Coverage

The GRAL DNN interface (`internal/gpuapi/dnn.go`) already defined all backward
method signatures. The CUDA adapter now fully implements them. ROCm and OpenCL
adapters retain their existing behavior (MIOpen stubs and ErrNotSupported
respectively).

### 5. Design Choices

- **Symmetric padding only**: Backward convolution methods enforce symmetric
  padding (same as forward), returning errors for asymmetric pads.
- **Algo0 default**: Convolution backward algorithms default to Algo0 for broad
  hardware compatibility. Future work could add algorithm auto-tuning.
- **BatchNorm epsilon**: The DNN interface's `BatchNormBackward` does not expose
  epsilon; the CUDA adapter uses a hardcoded 1e-5, matching cuDNN's default.
- **Shape-flexible activation backward**: Uses the same 4D packing strategy as
  the forward path, supporting 1D through 4D inputs.

## Consequences

**Positive:**
- Full GPU-accelerated training pipeline for Conv2d, BatchNorm, ReLU/Sigmoid/Tanh, MaxPool/AvgPool
- No CPU round-trip for gradient computation
- Memory pool reuse for workspace and gradient tensors
- Consistent API pattern with forward methods

**Negative:**
- Cannot be hardware-tested without CUDA GPU (CI relies on CPU-only builds)
- Hardcoded epsilon in BatchNormBackward limits flexibility
- No algorithm auto-tuning for convolution backward ops

**Risks:**
- cuDNN version < 8.0 may lack some backward algorithm variants
- Workspace allocation for large convolutions may trigger OOM on small GPUs
