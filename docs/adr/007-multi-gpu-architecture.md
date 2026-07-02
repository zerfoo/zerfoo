# ADR-007: Multi-GPU Architecture

**Phase:** 10
**Status:** Accepted
**Date:** 2026-03-03

## Context

Zerfoo's CUDA GPU engine (Phase 2-3) assumed a single GPU. All CUDA operations
targeted device 0 implicitly. To support multi-GPU inference and NCCL-based
distributed training, every CUDA-interacting component needed explicit device
affinity so that operations target the correct GPU.

## Decision

### Device Affinity Threading

Every CUDA-aware component now carries a `deviceID int` field and calls
`cuda.SetDevice(deviceID)` before any CUDA operation:

- **MemPool** (`internal/cuda/mempool.go`): Cache key changed from `byteSize` to
  `(deviceID, byteSize)` using `map[int]map[int][]unsafe.Pointer`. Prevents
  cross-device pointer reuse.

- **GPUEngine** (`compute/gpu_engine.go`): `NewGPUEngine(ops, ...int)` accepts an
  optional device ID (default 0). A `setDevice()` guard is called at the top of
  every method that dispatches CUDA kernels or cuBLAS calls.

- **GPUStorage** (`tensor/gpu_storage.go`): Constructors accept optional `deviceID`.
  `TrySlice()` and `TrySet()` call `SetDevice` before memcpy operations. Exposes
  `DeviceID() int` for consumers.

- **cudaAllocator** (`device/cuda_allocator.go`): `SetDevice` before Malloc/Free.

### Cross-Device Transfer

`ToGPUDevice[T](tensor, deviceID)` in `tensor/transfer.go` handles three cases:
- CPU to GPU: `NewGPUStorageFromSlice(data, deviceID)`
- GPU to same GPU: `cuda.Memcpy` with `MemcpyDeviceToDevice`
- GPU to different GPU: `cuda.MemcpyPeer` for peer-to-peer D2D copy

### Inference Device Selection

`inference.Load()` uses the existing `WithDevice("cuda:N")` option. Build-tag-gated
files (`engine_cuda.go`, `engine_nocuda.go`) handle conditional compilation.
`Model.Close()` releases GPU resources via `io.Closer`.

### NCCL Bindings

`internal/nccl/` provides CGo bindings for NCCL: `GetUniqueID`, `InitRank`,
`Destroy`, `AllReduce`, `Broadcast`, `GroupStart`, `GroupEnd`, `GetAsyncError`.
All behind `//go:build cuda` with `-lnccl` linking.

### NcclStrategy

`distributed/nccl_strategy.go` implements `InternalStrategy[T]` for GPU-native
gradient exchange. Key design choices:

- **No CPU round-trip**: Extracts device pointers directly from `GPUStorage.Ptr()`.
- **Grouped operations**: Uses `ncclGroupStart/GroupEnd` to batch all gradient
  all-reduces into a single launch.
- **Barrier via AllReduce**: Uses a 1-element dummy all-reduce since NCCL has no
  native barrier API.
- **InitWithUID**: Separate from `Init()` for single-process multi-GPU where a
  coordinator can distribute the `UniqueID` directly.

## Consequences

### Positive
- Any model can run on any GPU via `inference.Load(id, WithDevice("cuda:N"))`.
- Multiple GPUEngines can coexist in the same process on different devices.
- NCCL gradient exchange avoids host memory staging (significant bandwidth win).
- Backwards compatible: all device ID parameters are optional, defaulting to 0.

### Negative
- Every CUDA method has a `SetDevice` call overhead (negligible vs kernel launch).
- NCCL requires the `-lnccl` library at link time for distributed features.
- Multi-GPU tests require actual hardware; CI without GPUs exercises only the
  non-CUDA paths.

### Files Added
- `inference/engine_cuda.go`, `inference/engine_nocuda.go`
- `internal/nccl/doc.go`, `internal/nccl/nccl.go`, `internal/nccl/nccl_test.go`
- `distributed/nccl_strategy.go`, `distributed/nccl_strategy_test.go`
- `tests/parity/multigpu_test.go`

### Files Modified
- `internal/cuda/mempool.go`, `internal/cuda/runtime.go`
- `compute/gpu_engine.go`, `compute/gpu_kernels.go`
- `tensor/gpu_storage.go`, `tensor/transfer.go`
- `device/cuda_allocator.go`, `device/cuda_device.go`
- `inference/inference.go`
