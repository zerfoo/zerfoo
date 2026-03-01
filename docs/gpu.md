# Zerfoo GPU Compute Engine

## Overview

The GPU engine is a CUDA-accelerated drop-in replacement for CPUEngine. Both implement the `compute.Engine[T]` interface. All GPU code is behind the `//go:build cuda` build tag -- non-CUDA builds work without any GPU dependencies.

## Usage

```go
import (
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/numeric"
)

ops := numeric.Float32Ops{}

// GPU engine (requires -tags cuda build)
gpuEngine, err := compute.NewGPUEngine(ops)
if err != nil {
    log.Fatal(err)
}
defer gpuEngine.Close()  // Must close to release cuBLAS handle

// Use identically to CPUEngine -- same Engine[T] interface
result, err := gpuEngine.MatMul(ctx, tensorA, tensorB)
```

## Build Requirements

- CUDA Toolkit (libcudart, development headers)
- cuBLAS library (libcublas)
- NVIDIA GPU with Compute Capability >= 7.0 (Volta/Turing or newer)
- GCC/G++ (for CGO linking)

### Compile CUDA Kernels

```bash
cd internal/cuda/kernels/
make
```

This produces `libkernels.a` from `elementwise.cu` using `nvcc -O2 -arch=sm_70`.

### Build with GPU Support

```bash
go build -tags cuda ./...
go test -tags cuda ./...
```

## Architecture

### GPU-Accelerated Operations (float32 only)

| Category | Operations | Backend |
|----------|-----------|---------|
| Matrix | MatMul (2D and batched) | cuBLAS Sgemm |
| Element-wise | Add, Sub, Mul, Div, Pow | Custom CUDA kernels |
| Scalar | AddScalar, MulScalar, DivScalar | Custom CUDA kernels |
| Activation | Tanh, TanhPrime | Custom CUDA kernels |
| Math | Exp, Log, Sqrt, Rsqrt | Custom CUDA kernels |
| Reduction | Sum, ReduceSum, ReduceMean | Custom CUDA kernels (shared memory) |
| Other | Softmax, Fill | Custom CUDA kernels |

### CPU Fallback Operations

These operations delegate to CPUEngine by design (not compute-bound or require Go runtime):

- UnaryOp (Go function pointers)
- Transpose (metadata-only)
- Zero, Zeros, Copy
- Reshape, Split, Concat, Repeat
- Gather, ScatterAdd (integer indexing)
- OneHot, RandomUniform

### Type Support

- **float32:** Full GPU kernel support
- **All other types:** Automatic CPU fallback (transparent to callers)

## Memory Model

### Device-Resident Pipeline (Phase 3)

GPU operations produce tensors with `GPUStorage`, keeping data on-device between chained operations. Only the first input (if CPU-backed) does an H2D copy, and only the final result (when `.Data()` is called) does a D2H copy.

```
CPU Input -> H2D (via pool) -> Kernel -> GPUStorage output
                                            |
                               GPUStorage input (zero-copy) -> Kernel -> GPUStorage output
                                                                            |
                                                               .Data() -> D2H copy -> CPU slice
```

Key components:
- **`getDevicePtr`**: Checks tensor storage type. GPUStorage returns the device pointer directly (zero-copy). CPUStorage allocates from the memory pool and copies H2D.
- **`makeGPUResult`**: Creates output tensors with GPUStorage wrapping the device pointer.
- **CUDA MemPool** (`internal/cuda/mempool.go`): Size-bucketed free-list allocator. Reuses previously freed device memory, avoiding per-operation `cudaMalloc`/`cudaFree`.
- **CUDA Stream** (`internal/cuda/runtime.go`): Dedicated non-default stream for async kernel execution. All kernels and cuBLAS operations execute on this stream.
- **OOM Fallback**: When `cudaMalloc` fails (OOM), GPU operations fall back to `CPUEngine` transparently. An atomic counter (`OOMFallbackCount()`) tracks fallback frequency.

### Tensor Transfer Helpers

```go
gpuTensor, err := tensor.ToGPU(cpuTensor)  // Host -> Device
cpuTensor := tensor.ToCPU(gpuTensor)        // Device -> Host
```

### Storage Abstraction

`tensor.Storage[T]` interface abstracts CPU vs GPU memory:
- `CPUStorage[T]`: Direct slice access, zero-copy
- `GPUStorage[T]`: Device pointer, copies on `Slice()`/`Set()` calls

## File Layout

```
compute/
  engine.go              # Engine[T] interface definition
  cpu_engine.go          # CPUEngine implementation
  gpu_engine.go          # GPUEngine (pool, stream, cuBLAS) (//go:build cuda)
  gpu_kernels.go         # getDevicePtr, makeGPUResult, kernel dispatch (//go:build cuda)
  gpu_engine_test.go     # Unit tests (//go:build cuda)
  gpu_integration_test.go # Integration + chained-ops tests (//go:build cuda)

tensor/
  storage.go             # Storage[T] interface, CPUStorage[T], NewWithStorage
  gpu_storage.go         # GPUStorage[T], TrySlice/TrySet (//go:build cuda)
  transfer.go            # ToGPU/ToCPU helpers (//go:build cuda)

device/
  cuda_device.go         # CUDA device abstraction (//go:build cuda)
  cuda_allocator.go      # CUDA memory allocator (//go:build cuda)

internal/cuda/
  runtime.go             # CUDA runtime + Stream bindings (//go:build cuda)
  mempool.go             # Size-bucketed device memory pool (//go:build cuda)
  kernels/
    elementwise.cu       # CUDA kernel source (17 kernels, stream-aware)
    elementwise.go       # CGO bindings for kernels (//go:build cuda)
    Makefile             # nvcc compilation

internal/cublas/
  cublas.go              # cuBLAS + SetStream bindings (//go:build cuda)
```

## Testing on Real Hardware

### GCP VM Setup

Use GCP deep learning images with CUDA pre-installed:

```bash
# Image family with CUDA 12.8 + NVIDIA driver 570
gcloud compute instances create gpu-test \
    --machine-type=n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=80GB
```

Compatible GPUs (us-central1-a):
- **nvidia-tesla-t4** (sm_75, 16 GB) -- cheapest, good for testing
- **nvidia-l4** (sm_89, 24 GB) -- newer, faster
- **nvidia-tesla-v100** (sm_70, 16 GB) -- high performance
- **nvidia-tesla-a100** (sm_80, 40/80 GB) -- highest performance

### Run Tests

```bash
# 1. Compile CUDA kernels
cd zerfoo/internal/cuda/kernels && make

# 2. Run GPU test suite
cd zerfoo
go test -tags cuda -count=1 -v \
    ./compute/ \
    ./tensor/ \
    ./internal/cuda/... \
    ./internal/cublas/... \
    ./device/

# 3. Run parity tests (GPU vs CPU)
go test -tags cuda -run Parity -v ./compute/
```

## Known Limitations

1. **float32 only** -- other types fall back to CPU transparently
2. **No broadcasting** in GPU kernels -- broadcast cases fall back to CPU
3. **Single GPU** -- no multi-GPU or distributed support
4. **No cuDNN** -- all kernels are custom CUDA
5. **No mixed precision** -- full float32 throughout
6. **Default device** -- always uses cuda:0, no device selection API
7. **Hardware validation pending** -- E15/E20 blocked on GCP GPU quota (request submitted, pending approval)

## CGO Linker Flags

```
internal/cuda/runtime.go:     -lcudart
internal/cublas/cublas.go:    -lcublas
internal/cuda/kernels/*.go:   -L${SRCDIR} -lkernels -lcudart -lstdc++
```

## Parity Tolerances

- MatMul: 1e-5 relative error
- Element-wise ops: 1e-6 relative error
- Reductions (Sum, Mean): 1e-5 relative error
