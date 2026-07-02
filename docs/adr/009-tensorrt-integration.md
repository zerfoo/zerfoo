# ADR-009: TensorRT Integration

**Phase:** 12
**Status:** Accepted
**Date:** 2026-03-03

## Context

Individual GPU operations are accelerated via cuBLAS (MatMul) and cuDNN
(Conv2d, BatchNorm, activations, pooling -- Phase 11). However, operation-level
optimization misses graph-level opportunities: kernel fusion (combining
adjacent operations into a single kernel launch), layer merging (folding
BatchNorm into Conv2d weights), precision reduction (FP16/INT8 with minimal
accuracy loss), and per-GPU auto-tuning.

TensorRT is NVIDIA's inference optimizer that takes a model graph and produces
a highly optimized execution engine. It uses cuDNN internally for many
operations, making Phase 11 a prerequisite.

The Zerfoo `graph.Graph[T]` represents computation as a DAG of `Node[T]`
values. TensorRT integration requires converting this graph representation
to TensorRT's `INetworkDefinition` and building an optimized `ICudaEngine`.

## Decision

### CGo Bindings via C Shim (`internal/tensorrt/`)

TensorRT exposes a C++ API (no C API). A thin C shim (`trt_capi.h` and
`trt_capi.cpp`) wraps the essential C++ classes with C function signatures
that CGo can call. This follows industry practice for Go-TensorRT integration.

Types:
- **Logger** wraps `ILogger`. Reports build progress and warnings.
- **Builder** wraps `IBuilder`. Creates networks and builds engines.
- **BuilderConfig** wraps `IBuilderConfig`. Sets workspace limits, precision
  flags (FP16, INT8), and optimization profiles.
- **NetworkDefinition** wraps `INetworkDefinition`. Represents the model graph
  as TensorRT layers.
- **Runtime** wraps `IRuntime`. Deserializes cached engines.
- **Engine** wraps `ICudaEngine`. The optimized inference artifact.
- **ExecutionContext** wraps `IExecutionContext`. Runs inference on an engine.

Build tag: `//go:build cuda`. Linker: `-lnvinfer -lnvinfer_plugin`.

### Subgraph Conversion Strategy

Rather than converting the entire Zerfoo graph to TensorRT (which would
require TensorRT plugins for every unsupported operation like MoE and
MatMulNBits), the converter identifies the largest contiguous subgraphs of
TRT-supported operations and converts those independently. Unsupported nodes
at subgraph boundaries execute via the existing GPUEngine.

Supported operation mappings:
- MatMul -> `addMatrixMultiply`
- Add/Sub/Mul/Div -> `addElementWise`
- ReLU/Sigmoid/Tanh -> `addActivation`
- Softmax -> `addSoftMax`
- Conv2d -> `addConvolutionNd`
- Reshape -> `addShuffle`
- ReduceSum/ReduceMean -> `addReduce`
- Constant -> `addConstant`

### Engine Caching

The TensorRT optimization step is expensive (30 seconds to 5 minutes depending
on model size and target precision). Serialized engines are cached at:
`~/.cache/zerfoo/tensorrt/<model_id>_<precision>_<gpu_arch>.engine`

Cache key components:
- `model_id`: Registry model identifier
- `precision`: "fp32" or "fp16"
- `gpu_arch`: Compute capability (e.g., "7.5" for T4, "8.0" for A100)

On load, the cache is checked first. The engine is rebuilt only on cache miss
or GPU architecture change. The GPU architecture is detected via
`cudaGetDeviceProperties`.

### Inference Pipeline Integration

A new `WithBackend("tensorrt")` option on `inference.Load()` triggers the
TensorRT path:
1. Load model weights and build the Zerfoo graph (existing path).
2. Check engine cache for the (model, precision, GPU arch) tuple.
3. On cache hit: deserialize the engine via `Runtime.DeserializeCudaEngine`.
4. On cache miss: convert graph to TRT network, build engine, serialize to
   cache.
5. Create an `ExecutionContext` and wrap it in a TRT-backed engine struct
   that satisfies the existing inference contract.

### Precision

- FP32: Default. No calibration needed.
- FP16: Enabled via `WithPrecision("fp16")`. Set the FP16 flag on
  `BuilderConfig`. No calibration needed; TensorRT handles mixed precision
  automatically.
- INT8: Deferred. Requires a calibration dataset and calibration API
  integration. Can be added in a future phase.

## Consequences

### Positive

- 2-5x inference speedup via kernel fusion, layer merging, and precision
  reduction.
- Engine caching amortizes the expensive build step across runs.
- FP16 inference with minimal accuracy loss (automatic mixed precision).
- Subgraph approach handles unsupported ops gracefully without requiring
  plugins for every custom operation.

### Negative

- C++ API requires a C shim layer, adding build complexity.
- TensorRT build step is slow (30s-5min) on first run for each model/GPU
  combination.
- Serialized engines are GPU-architecture-specific and not portable.
- Subgraph boundaries introduce data transfer overhead between TRT engine
  and GPUEngine execution.
- libnvinfer is a large dependency (~500MB with plugins).

### Files Added

- `internal/tensorrt/doc.go` -- package identity
- `internal/tensorrt/tensorrt.go` -- CGo bindings
- `internal/tensorrt/trt_capi.h` -- C shim header
- `internal/tensorrt/trt_capi.cpp` -- C shim implementation
- `inference/tensorrt_convert.go` -- graph-to-TRT converter
- `inference/tensorrt_cache.go` -- engine serialization/caching

### Files Modified

- `inference/inference.go` -- add WithBackend and WithPrecision options
- `inference/engine_cuda.go` -- add TRT engine creation path
