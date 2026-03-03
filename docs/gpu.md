# Multi-GPU and Distributed GPU Support

This document describes what is needed to add multi-GPU and distributed GPU
support to Zerfoo. It serves as a roadmap for future implementation.

## Current State

Zerfoo has a single-GPU CUDA backend (float32 only) behind `//go:build cuda`.
The runtime bindings for `cuda.SetDevice()` and `cuda.GetDeviceCount()` exist
in `internal/cuda/runtime.go` but are never called in production code.
Everything implicitly uses `cuda:0`. See
[ADR-006](adr/006-gpu-engine-architecture.md) for the current GPU architecture.

Key limitations:
- `GPUEngine` has no `deviceID` field (`compute/gpu_engine.go:27`).
- `GPUStorage` has no device affinity (`tensor/gpu_storage.go:17`).
- `MemPool` is keyed by byte size only, not per-device (`internal/cuda/mempool.go:13`).
- `inference.Load()` hardcodes `NewCPUEngine` and ignores the `WithDevice("cuda")` option (`inference/inference.go:149`).
- Distributed gradient exchange copies GPU tensors to CPU before serialization (`distributed/grpc_strategy.go:371`).

## Layer 1: CUDA Device Affinity (Foundation)

Thread a device ID through the CUDA stack so each GPU engine, tensor, and
memory pool is explicitly bound to a specific device.

### GPUEngine (`compute/gpu_engine.go`)

Add `deviceID int` field to the struct. `NewGPUEngine()` accepts a device ID
parameter and calls `cuda.SetDevice()` before creating the cuBLAS handle,
stream, and memory pool. All operations call `SetDevice()` before dispatching
kernels.

### GPUStorage (`tensor/gpu_storage.go`)

Add `deviceID int` field. Every allocation path (`NewGPUStorage`,
`NewGPUStorageFromSlice`, `NewGPUStorageFromPtr`) calls `SetDevice()` before
`cuda.Malloc()`. `TrySlice()` and `TrySet()` set device before D2H/H2D copies.
`DeviceType()` or a new `DeviceID()` method exposes which GPU the tensor
resides on.

### MemPool (`internal/cuda/mempool.go`)

Make per-device. Current cache is `map[int][]unsafe.Pointer` keyed by byte
size. Change to per-device-per-size so a pointer allocated on GPU 0 is never
reused on GPU 1. `Alloc()` and `Free()` take a `deviceID` parameter. `Drain()`
iterates all devices.

### CudaAllocator (`device/cuda_allocator.go`)

`cuda.Malloc()` must be preceded by `SetDevice()`. The allocator needs the
device ID to bind allocations to the correct GPU.

### Estimated scope

~400 lines across 6 files. No interface changes; purely threading a device ID
through constructors and adding `SetDevice()` guards.

## Layer 2: Multi-GPU Inference

### Fix inference.Load (`inference/inference.go`)

Currently hardcoded to `compute.NewCPUEngine`. The `WithDevice("cuda")` option
is accepted but ignored. Fix: parse `"cuda:0"`, `"cuda:1"` etc., create the
corresponding `GPUEngine`. Add `Model.Close()` for device cleanup.

### Tensor transfer helpers (`tensor/transfer.go`)

Add `ToGPU(deviceID)` variant for targeting a specific GPU. Add peer-to-peer
D2D copy (`cudaMemcpyPeer`) for cross-GPU tensor movement. Current `ToGPU()`
and `ToCPU()` helpers only handle single-device transfers.

### Graph execution (`graph/graph.go`)

Currently one `Engine[T]` per `Graph`. Simplest multi-GPU approach: one Graph
per device, manual data marshalling between them. Model parallelism (splitting
layers across GPUs) would require device-aware node routing — significantly
harder.

### Parallelism strategies

| Strategy | Complexity | Use Case |
|----------|-----------|----------|
| Data parallelism (same model on each GPU, split batches) | Medium | Training, batch inference |
| Pipeline parallelism (different layers on different GPUs) | High | Large models exceeding single GPU memory |

Data parallelism is the natural next step since the distributed package already
does gradient averaging via `InternalStrategy[T]`.

### Estimated scope

~200 lines for basic multi-GPU inference. ~500 additional lines for data
parallelism with batch splitting and gradient sync.

## Layer 3: Distributed GPU (NCCL)

The current distributed gradient exchange path:

```
GPU tensor -> .Data() (D2H copy) -> protobuf float32 -> gRPC -> deserialize -> CPU tensor
```

This works but is slow for multi-GPU on the same node. NCCL enables direct
GPU-to-GPU collective operations without CPU involvement.

### NCCL bindings (`internal/nccl/`)

New CGo package wrapping `libnccl`:
- `ncclAllReduce` -- sum/average gradients across GPUs
- `ncclBroadcast` -- send tensor from one GPU to all others
- `ncclBarrier` -- synchronize GPUs
- `ncclCommInitRank` -- initialize communicator

Requires NCCL library (libnccl2) as a build dependency, gated behind
`//go:build cuda` like existing CUDA code.

### NcclStrategy (`distributed/`)

New `NcclStrategy[T]` implementing `InternalStrategy[T]` using NCCL for
intra-node GPU-GPU collective ops. Tensors stay on-device — no D2H/H2D copies.
The existing `AllReduceStrategy[T]` already supports local + cross-node
strategy composition, so `NcclStrategy` slots in as the local strategy.

### Hybrid topology

NCCL for intra-node, gRPC for inter-node. Detect topology via
`cudaDeviceCanAccessPeer()`. Within a node, gradients are reduced directly
on GPU. Across nodes, a single GPU per node acts as the gateway, copies to
CPU for gRPC transport, then distributes back via NCCL.

### Estimated scope

~800 lines for NCCL bindings and NcclStrategy.

## Implementation Order

| Layer | Scope | Depends On | Rough Size |
|-------|-------|-----------|-----------|
| 1. Device affinity | Thread device ID through CUDA stack | Nothing | ~400 lines, 6 files |
| 2a. Multi-GPU inference | Fix inference.Load, add D2D transfer | Layer 1 | ~200 lines |
| 2b. Data parallelism | Batch splitting + gradient sync on GPU | Layer 1 | ~500 lines |
| 3. NCCL bindings + strategy | New CGo package + NcclStrategy | Layer 1 | ~800 lines |

Layer 1 is prerequisite for everything. Layer 2a gives multi-GPU inference.
Layer 2b gives multi-GPU training with CPU-based gradient exchange. Layer 3
gives fast GPU-native gradient exchange.

## Architecture Advantages

The existing codebase anticipated this extension:
- `device/` package already has a device registry and per-device allocator concept.
- `InternalStrategy[T]` interface cleanly separates gradient exchange from training logic.
- `AllReduceStrategy[T]` already composes local + cross-node strategies.
- `Engine[T]` interface is device-agnostic; multiple GPU engines can coexist.
- `//go:build cuda` tag keeps GPU code cleanly separated.
