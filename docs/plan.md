# Zerfoo Enterprise Production Readiness Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).

Phases 1-9 brought the framework to production grade: observability, security,
reliability, configuration, CI/CD enforcement, open-weights model import (6
model families), embeddable inference library with BPE tokenizer, KV cache,
generation loop, streaming, model registry, high-level API, CLI commands, and
OpenAI-compatible HTTP server.

The current GPU backend is limited to a single GPU (cuda:0). The runtime
bindings for `cuda.SetDevice()` and `cuda.GetDeviceCount()` exist in
`internal/cuda/runtime.go` but are never called in production code. Every
allocation and kernel dispatch implicitly uses the default device.

Phase 10 adds multi-GPU support in three layers: (1) CUDA device affinity --
threading a device ID through the GPU engine, storage, and memory pool;
(2) multi-GPU inference -- fixing `inference.Load()` to use the device option
and adding cross-device tensor transfer; (3) NCCL-based distributed GPU
gradient exchange for fast intra-node training.

Architecture, design, GPU details, operations, and troubleshooting are
documented in docs/design.md (the single reference document). Stable design
decisions are extracted into docs/adr/ (see [ADR index](design.md#13-architectural-decision-records)).
The multi-GPU research and roadmap is in [docs/gpu.md](gpu.md).

### Objectives

- O12: Thread device ID through the CUDA stack so each GPUEngine, GPUStorage,
  and MemPool is explicitly bound to a specific GPU device.
- O13: Fix `inference.Load()` to create a GPUEngine when `WithDevice("cuda")`
  or `WithDevice("cuda:N")` is specified.
- O14: Add cross-device tensor transfer helpers (peer-to-peer D2D copy).
- O15: Add NCCL CGo bindings for GPU-native collective operations.
- O16: Implement NcclStrategy[T] for intra-node GPU-GPU gradient exchange.

### Non-Goals

- cuDNN, TensorRT, or other NVIDIA library integration.
- AMD ROCm or OpenCL backends.
- Mixed precision training.
- Breaking changes to the Engine[T] or Node[T] interfaces.
- Replacing gRPC with a different RPC framework.
- Adding third-party test frameworks (testify, etc.).
- SSM/Mamba architectures (Falcon Mamba, RWKV, Jamba).
- Pipeline parallelism (splitting layers across GPUs).
- Multi-GPU KV cache partitioning.
- Tensor parallelism within a single operation.

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- NCCL code behind `//go:build cuda` (requires libnccl2).
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- Table-driven tests using the standard testing package.
- No breaking changes to the Engine[T] interface. NewGPUEngine gains an
  optional deviceID parameter via a variadic int or option pattern; existing
  callers that pass zero arguments get device 0 (backwards compatible).
- NCCL requires NVIDIA GPU with Compute Capability >= 7.0 and libnccl2.
- Multi-GPU tests require at least 2 GPUs; tests skip on single-GPU systems.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Device affinity | All GPU allocations specify device | Grep for cuda.Malloc without SetDevice guard = 0 in production code |
| Multi-GPU engine | GPUEngine works on any device | Test creates engines on device 0 and 1, runs MatMul on each |
| Inference device | inference.Load("cuda:1") works | Integration test loads model on specified GPU |
| Cross-device transfer | D2D copy works | Test copies tensor from GPU 0 to GPU 1, verifies data |
| NCCL AllReduce | GPU gradients averaged without CPU | 2-GPU NCCL AllReduce produces correct result |
| No regression | Existing tests pass | go test ./... -race and go test -tags cuda ./... both green |

---

## 2. Scope and Deliverables

### In Scope

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D11 | Device-affine GPUEngine | GPUEngine stores deviceID; calls SetDevice before all CUDA ops |
| D12 | Device-affine GPUStorage | GPUStorage stores deviceID; all constructors set device before malloc |
| D13 | Per-device memory pool | MemPool keyed by (deviceID, byteSize); no cross-device pointer reuse |
| D14 | Device-affine allocator | cudaAllocator calls SetDevice before cuda.Malloc |
| D15 | Multi-GPU inference | inference.Load("cuda:0") creates GPUEngine on specified device |
| D16 | Cross-device transfer | ToGPUDevice(t, deviceID) and D2D peer copy via cudaMemcpyPeer |
| D17 | NCCL bindings | CGo bindings for ncclAllReduce, ncclBroadcast, ncclCommInitRank |
| D18 | NcclStrategy | InternalStrategy[T] using NCCL for intra-node GPU collective ops |

### Out of Scope

- Pipeline parallelism (different layers on different GPUs).
- Multi-GPU KV cache partitioning for inference.
- Tensor parallelism within a single MatMul.
- Automatic device placement or load balancing.
- Web UI or dashboard for GPU monitoring.

### Prior Phase Deliverables (Complete)

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D1 | Structured logging | Logger interface with Debug/Info/Warn/Error levels; JSON output mode; all packages instrumented |
| D2 | Metrics interface | Counters, gauges, histograms; default in-memory impl; export-ready |
| D3 | gRPC TLS | TLS config struct; mTLS support; integration test with TLS |
| D4 | Config management | JSON loader; env var overrides; validation errors |
| D5 | Graceful shutdown | Context-based cancellation; cleanup ordering; integration test |
| D6 | Health checks | HTTP /healthz and /readyz endpoints; configurable checks |
| D7 | CI hardening | Blocking parity/numerics; coverage gate; benchmark gate |
| D8 | Resource limits | Memory cap on Engine; per-operation timeout; GPU memory limit |
| D9 | Production docs | Deployment runbook; troubleshooting guide; performance tuning |
| D10 | GPU validation | Tests pass on real T4; benchmark results documented |

---

## 3. Work Breakdown

### Completed Phases (1-9)

Phase 1 (Test Coverage), Phase 2 (GPU Engine), Phase 3 (GPU Production
Readiness), Phase 4 (Enterprise Production Readiness), Phase 5 (Distributed
Training Protocol), Phase 6 (Open Weights Model Import), Phase 7 (Architecture
Cleanup), Phase 8 (Embeddable Inference Library), Phase 9 (Multi-Architecture
Support) are all complete. See docs/adr/ for design decisions.

### Blocked Items (Prior Phases)

#### E29: GPU Hardware Validation

- [ ] T29.1 Create GCP T4 spot VM and validate GPU tests  **BLOCKED:** GCP GPU quota = 0.
  - Quota increase request pending (preference ID: zerfoo-gpu-test, project: numerai-488804).
  - Unblock: `gcloud beta quotas preferences describe zerfoo-gpu-test --project=numerai-488804`
  - Alternative: try a different GCP project or cloud provider.
  - Steps: create n1-standard-4 spot VM with T4, install CUDA 12.x + Go 1.25,
    `go test -tags cuda ./...`, capture benchmarks, delete VM immediately.
- [ ] T29.2 Run optimized benchmarks on T4  **BLOCKED:** Depends on T29.1.
  - Benchmark MatMul (128/512/1024), Softmax, chained attention ops.
  - Document results in docs/design.md.

---

### Phase 10: Multi-GPU and Distributed GPU Support

#### Phase 10 Context

The GPU backend works correctly on a single device but has no explicit device
binding. `cuda.SetDevice()` exists in `internal/cuda/runtime.go:74-81` but is
never called from production code. `GPUEngine` (`compute/gpu_engine.go:27-36`)
has no `deviceID` field. `GPUStorage` (`tensor/gpu_storage.go:17-21`) has no
device affinity. `MemPool` (`internal/cuda/mempool.go:13-16`) caches pointers
by byte size only, not per device. `cudaAllocator` (`device/cuda_allocator.go:12-13`)
calls `cuda.Malloc()` without preceding `SetDevice()`. `inference.Load()`
(`inference/inference.go:149`) hardcodes `NewCPUEngine` and ignores the
`WithDevice("cuda")` option. Distributed gradient exchange in
`distributed/grpc_strategy.go:367-385` copies GPU tensors to CPU before
serialization via `.Data()`.

The `device/cuda_device.go:12-16` `cudaDevice` struct already stores a
`deviceID int` field and the init function (lines 39-48) registers one device
per GPU, but the stored deviceID is never used to call `SetDevice()`.

#### Phase 10 Design Decisions

**Backwards-compatible constructor:** `NewGPUEngine` gains a variadic
`...int` parameter for the device ID. Zero arguments means device 0 (current
behavior). This avoids breaking existing callers. The engine calls
`cuda.SetDevice(deviceID)` before creating the cuBLAS handle, CUDA stream,
and memory pool.

**Device guard pattern:** Every GPUEngine method that dispatches a CUDA kernel
or cuBLAS call must call `cuda.SetDevice(e.deviceID)` at the top. This is a
cheap no-op when only one GPU exists and correct when multiple engines target
different devices from different goroutines.

**Per-device memory pool:** The `MemPool` cache key changes from `byteSize` to
`(deviceID, byteSize)`. The simplest implementation: nested map
`map[int]map[int][]unsafe.Pointer` where outer key is deviceID. Alloc and Free
both take a deviceID parameter and call SetDevice before cuda.Malloc.

**GPUStorage device tracking:** Each GPUStorage stores its deviceID. This
enables the runtime to detect cross-device operations (e.g., trying to use a
tensor from GPU 0 in an operation on GPU 1) and either error or trigger a D2D
copy.

**Inference device selection:** `inference.Load()` parses the device string
("cpu", "cuda", "cuda:0", "cuda:1") and creates the appropriate engine. For
"cuda" without a device number, default to device 0.

**D2D transfer:** A new `ToGPUDevice[T](t, deviceID)` function uses
`cudaMemcpyPeer()` for cross-device copy. The CUDA runtime handles NVLink or
PCIe routing automatically.

**NCCL strategy:** `NcclStrategy[T]` implements `InternalStrategy[T]` using
NCCL for intra-node collective operations. It slots into the existing
`AllReduceStrategy[T]` as the local strategy (replacing `GrpcStrategy` for
same-node workers). Tensors stay on-device throughout the all-reduce.

---

#### E70: Per-Device Memory Pool

Make the CUDA memory pool device-aware so pointers allocated on one GPU are
never reused on another.

- [ ] T70.1 Add deviceID parameter to MemPool.Alloc and MemPool.Free  Owner: TBD  Est: 1h
  - Dependencies: None
  - Files: internal/cuda/mempool.go
  - Acceptance: MemPool.Alloc(deviceID int, byteSize int) calls cuda.SetDevice(deviceID)
    before cuda.Malloc when cache misses. MemPool.Free(deviceID int, ptr, byteSize) stores
    the pointer under (deviceID, byteSize). Cache key is (deviceID, byteSize). Drain()
    iterates all devices, calling SetDevice before Free for each. No cross-device pointer
    reuse possible. Existing behavior preserved when all calls use deviceID=0.
  - [ ] S70.1.1 Change cache type from map[int][]unsafe.Pointer to map[int]map[int][]unsafe.Pointer  Est: 15m
  - [ ] S70.1.2 Update Alloc to accept deviceID, call SetDevice before Malloc  Est: 15m
  - [ ] S70.1.3 Update Free to accept deviceID, store under (deviceID, byteSize)  Est: 10m
  - [ ] S70.1.4 Update Drain to iterate per-device, call SetDevice before Free  Est: 10m
  - [ ] S70.1.5 Write unit tests: alloc/free on device 0, alloc/free on device 1, no cross-reuse  Est: 20m
  - [ ] S70.1.6 Run golangci-lint and go test -cover  Est: 5m

#### E71: Device-Affine GPU Engine

Add device ID tracking to GPUEngine so each engine is explicitly bound to a
specific GPU and calls SetDevice before all CUDA operations.

- [ ] T71.1 Add deviceID field to GPUEngine and update constructor  Owner: TBD  Est: 1.5h
  - Dependencies: E70
  - Files: compute/gpu_engine.go
  - Acceptance: GPUEngine gains a `deviceID int` field. NewGPUEngine(ops, ...int)
    accepts an optional device ID (default 0). Constructor calls cuda.SetDevice(deviceID)
    before creating cuBLAS handle, stream, and pool. A DeviceID() int method exposes
    the device. Existing callers (zero args) get device 0. Static assertion: single-arg
    call still compiles.
  - [ ] S71.1.1 Add deviceID int field to GPUEngine struct (line 27)  Est: 5m
  - [ ] S71.1.2 Update NewGPUEngine signature to accept variadic ...int  Est: 10m
  - [ ] S71.1.3 Call cuda.SetDevice(deviceID) before cublas.CreateHandle  Est: 10m
  - [ ] S71.1.4 Pass deviceID to cuda.NewMemPool or store on engine for pool calls  Est: 10m
  - [ ] S71.1.5 Add DeviceID() int method  Est: 5m
  - [ ] S71.1.6 Write tests: create engine on device 0, verify DeviceID(); create on device 1 if available  Est: 20m
  - [ ] S71.1.7 Run golangci-lint and go test -cover  Est: 5m

- [ ] T71.2 Add SetDevice guard to all GPUEngine methods  Owner: TBD  Est: 1.5h
  - Dependencies: T71.1
  - Files: compute/gpu_engine.go, compute/gpu_kernels.go
  - Acceptance: Every method on GPUEngine that dispatches a CUDA kernel or cuBLAS
    call begins with cuda.SetDevice(e.deviceID). This includes: MatMul, Add, Sub,
    Mul, Div, Pow, AddScalar, MulScalar, DivScalar, Tanh, TanhPrime, Exp, Log,
    Sqrt, Rsqrt, Sum, ReduceSum, ReduceMean, Softmax, Fill. Methods that delegate
    to CPUEngine (Transpose, Reshape, etc.) do NOT need the guard. Test: two
    GPUEngines on different devices can run MatMul concurrently without races or
    wrong-device errors.
  - Risk: Must not break the OOM fallback path (gpu_engine.go). When cudaMalloc
    fails and falls back to CPUEngine, the SetDevice call is harmless but must
    not interfere with the fallback logic.
  - [ ] S71.2.1 Add e.setDevice() helper that calls cuda.SetDevice(e.deviceID)  Est: 10m
  - [ ] S71.2.2 Insert e.setDevice() at the top of all GPU-dispatching methods  Est: 30m
  - [ ] S71.2.3 Verify OOM fallback path still works with SetDevice  Est: 15m
  - [ ] S71.2.4 Write concurrent test: 2 engines on 2 devices, parallel MatMul (skip if < 2 GPUs)  Est: 20m
  - [ ] S71.2.5 Run golangci-lint and go test -cover -race  Est: 5m

- [ ] T71.3 Update all GPUEngine pool calls to pass deviceID  Owner: TBD  Est: 45m
  - Dependencies: T71.1, E70
  - Files: compute/gpu_engine.go, compute/gpu_kernels.go
  - Acceptance: Every call to e.pool.Alloc() and e.pool.Free() passes e.deviceID.
    getDevicePtr and makeGPUResult in gpu_kernels.go pass deviceID through.
    No pool call without a deviceID argument remains.
  - [ ] S71.3.1 Update getDevicePtr to use e.pool.Alloc(e.deviceID, ...)  Est: 15m
  - [ ] S71.3.2 Update makeGPUResult to use e.pool.Alloc(e.deviceID, ...)  Est: 15m
  - [ ] S71.3.3 Grep for remaining pool.Alloc/pool.Free calls without deviceID; fix any  Est: 10m
  - [ ] S71.3.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T71.4 Run linters and verify coverage for E71  Owner: TBD  Est: 15m
  - Dependencies: T71.3
  - Acceptance: golangci-lint 0 issues on compute/. go test -tags cuda -cover -race
    passes. Coverage >= 95%.
  - [ ] S71.4.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S71.4.2 Fix any remaining issues  Est: 5m

#### E72: Device-Affine GPU Storage

Add device ID tracking to GPUStorage so each tensor knows which GPU it resides
on.

- [ ] T72.1 Add deviceID field to GPUStorage and update constructors  Owner: TBD  Est: 1.5h
  - Dependencies: None (can be done in parallel with E71 after E70)
  - Files: tensor/gpu_storage.go
  - Acceptance: GPUStorage gains a `deviceID int` field and a `DeviceID() int`
    method. NewGPUStorage(length, deviceID) calls cuda.SetDevice(deviceID) before
    cuda.Malloc. NewGPUStorageFromSlice(data, deviceID) calls SetDevice before
    Memcpy. NewGPUStorageFromPtr(ptr, length, deviceID) stores the deviceID.
    TrySlice() calls SetDevice(s.deviceID) before D2H copy. TrySet() calls
    SetDevice(s.deviceID) before H2D copy. Existing behavior preserved when
    deviceID=0.
  - Risk: All callers of NewGPUStorage must be updated to pass deviceID. This
    includes gpu_engine.go, transfer.go, and any test files.
  - [ ] S72.1.1 Add deviceID int field to GPUStorage struct (line 17)  Est: 5m
  - [ ] S72.1.2 Add DeviceID() int method  Est: 5m
  - [ ] S72.1.3 Update NewGPUStorage to accept deviceID, call SetDevice  Est: 15m
  - [ ] S72.1.4 Update NewGPUStorageFromSlice to accept deviceID, call SetDevice  Est: 15m
  - [ ] S72.1.5 Update NewGPUStorageFromPtr to accept deviceID  Est: 10m
  - [ ] S72.1.6 Add SetDevice call in TrySlice and TrySet  Est: 10m
  - [ ] S72.1.7 Update all callers (grep for NewGPUStorage, NewGPUStorageFromSlice, NewGPUStorageFromPtr)  Est: 15m
  - [ ] S72.1.8 Write tests: create storage on device 0 and device 1, verify DeviceID  Est: 15m
  - [ ] S72.1.9 Run golangci-lint and go test -cover  Est: 5m

- [ ] T72.2 Update device/cuda_allocator.go with device affinity  Owner: TBD  Est: 30m
  - Dependencies: None
  - Files: device/cuda_allocator.go
  - Acceptance: cudaAllocator gains a deviceID int field. NewCUDAAllocator(deviceID)
    stores it. Allocate() calls cuda.SetDevice(a.deviceID) before cuda.Malloc().
    Free() calls cuda.SetDevice(a.deviceID) before cuda.Free(). Update
    cuda_device.go newCUDADevice to pass deviceID to NewCUDAAllocator.
  - [ ] S72.2.1 Add deviceID field to cudaAllocator  Est: 5m
  - [ ] S72.2.2 Update NewCUDAAllocator to accept deviceID  Est: 5m
  - [ ] S72.2.3 Add SetDevice calls in Allocate and Free  Est: 10m
  - [ ] S72.2.4 Update newCUDADevice to pass deviceID  Est: 5m
  - [ ] S72.2.5 Write tests for device-affine allocation  Est: 10m
  - [ ] S72.2.6 Run golangci-lint and go test -cover  Est: 5m

- [ ] T72.3 Add cross-device tensor transfer  Owner: TBD  Est: 1h
  - Dependencies: T72.1
  - Files: tensor/transfer.go, internal/cuda/runtime.go
  - Acceptance: New function ToGPUDevice[T](t *TensorNumeric[T], deviceID int)
    creates a copy of the tensor on the specified GPU. If the source tensor is
    on a different GPU, uses cudaMemcpyPeer for D2D copy. If the source is CPU,
    uses cudaMemcpyHostToDevice with SetDevice. New CGo binding
    cuda.MemcpyPeer(dst, dstDevice, src, srcDevice, size) wraps cudaMemcpyPeer.
    Update existing ToGPU to default to device 0 for backwards compatibility.
  - [ ] S72.3.1 Add MemcpyPeer binding to internal/cuda/runtime.go  Est: 15m
  - [ ] S72.3.2 Implement ToGPUDevice[T] in tensor/transfer.go  Est: 20m
  - [ ] S72.3.3 Update existing ToGPU to call ToGPUDevice with device 0  Est: 5m
  - [ ] S72.3.4 Write tests: CPU to GPU:0, CPU to GPU:1, GPU:0 to GPU:1 (skip if < 2 GPUs)  Est: 20m
  - [ ] S72.3.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T72.4 Run linters and verify coverage for E72  Owner: TBD  Est: 15m
  - Dependencies: T72.3
  - Acceptance: golangci-lint 0 issues on tensor/, device/. go test -tags cuda
    -cover -race passes. Coverage >= 95%.
  - [ ] S72.4.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S72.4.2 Fix any remaining issues  Est: 5m

#### E73: Multi-GPU Inference

Fix inference.Load() to create a GPUEngine when the device option specifies
CUDA, and add Model.Close() for resource cleanup.

- [ ] T73.1 Implement device selection in inference.Load  Owner: TBD  Est: 1.5h
  - Dependencies: E71, E72
  - Files: inference/inference.go
  - Acceptance: inference.Load(modelID, WithDevice("cuda")) creates a GPUEngine[float32]
    on device 0. inference.Load(modelID, WithDevice("cuda:1")) creates a GPUEngine on
    device 1. inference.Load(modelID, WithDevice("cpu")) creates a CPUEngine (current
    behavior). The device string is parsed: "cpu" -> CPUEngine; "cuda" -> GPUEngine(0);
    "cuda:N" -> GPUEngine(N). If GPU creation fails (no CUDA, invalid device), return
    a clear error. Test: mock-based test verifying device string parsing and engine
    creation dispatch.
  - [ ] S73.1.1 Add device string parsing: extractDeviceType and extractDeviceID  Est: 15m
  - [ ] S73.1.2 Replace hardcoded NewCPUEngine (line 149) with device switch  Est: 20m
  - [ ] S73.1.3 Add Model.Close() method that calls GPUEngine.Close() if applicable  Est: 15m
  - [ ] S73.1.4 Write tests for device parsing: "cpu", "cuda", "cuda:0", "cuda:1", "invalid"  Est: 20m
  - [ ] S73.1.5 Write integration test: load model on GPU (skip if no CUDA)  Est: 15m
  - [ ] S73.1.6 Run golangci-lint and go test -cover  Est: 5m

- [ ] T73.2 Add multi-GPU inference integration test  Owner: TBD  Est: 1h
  - Dependencies: T73.1
  - Files: tests/parity/multigpu_test.go (new)
  - Acceptance: Test loads the same model on device 0 and device 1 (skip if < 2 GPUs).
    Both models generate the same output for the same prompt (greedy decode). Verifies
    device affinity: tensors from model 0 are on device 0, tensors from model 1 are on
    device 1.
  - [ ] S73.2.1 Create tests/parity/multigpu_test.go with device count check  Est: 20m
  - [ ] S73.2.2 Test dual-device model loading and generation  Est: 25m
  - [ ] S73.2.3 Run golangci-lint and go test -tags cuda  Est: 5m

- [ ] T73.3 Run linters and verify coverage for E73  Owner: TBD  Est: 15m
  - Dependencies: T73.2
  - Acceptance: golangci-lint 0 issues on inference/. go test -cover -race passes.
  - [ ] S73.3.1 Run golangci-lint, go vet, go test -cover -race  Est: 10m
  - [ ] S73.3.2 Fix any remaining issues  Est: 5m

#### E74: NCCL Bindings

Add CGo bindings for NCCL (NVIDIA Collective Communications Library) to enable
GPU-native collective operations.

- [ ] T74.1 Create internal/nccl/ package with CGo bindings  Owner: TBD  Est: 2h
  - Dependencies: None (can start in parallel with E70-E72)
  - Files: internal/nccl/nccl.go (new)
  - Acceptance: Package internal/nccl provides Go bindings for: ncclGetUniqueId,
    ncclCommInitRank, ncclCommDestroy, ncclAllReduce (sum, avg), ncclBroadcast,
    ncclGroupStart, ncclGroupEnd, ncclCommGetAsyncError. All behind //go:build cuda.
    CGo links against -lnccl. NcclComm type wraps ncclComm_t. NcclUniqueID type
    wraps ncclUniqueId. DataType mapping: float32 -> ncclFloat32. ReduceOp mapping:
    Sum -> ncclSum. All functions return Go errors wrapping ncclResult_t.
  - [ ] S74.1.1 Create internal/nccl/nccl.go with CGo preamble and linker flags  Est: 15m
  - [ ] S74.1.2 Bind ncclGetUniqueId and NcclUniqueID type  Est: 15m
  - [ ] S74.1.3 Bind ncclCommInitRank and NcclComm type  Est: 15m
  - [ ] S74.1.4 Bind ncclAllReduce with stream parameter  Est: 20m
  - [ ] S74.1.5 Bind ncclBroadcast with stream parameter  Est: 15m
  - [ ] S74.1.6 Bind ncclCommDestroy, ncclGroupStart, ncclGroupEnd  Est: 10m
  - [ ] S74.1.7 Bind ncclCommGetAsyncError for error checking  Est: 10m
  - [ ] S74.1.8 Write unit tests: init/destroy comm on single GPU, AllReduce with 1 rank  Est: 20m
  - [ ] S74.1.9 Run golangci-lint and go test -tags cuda -cover  Est: 5m

- [ ] T74.2 Add multi-GPU NCCL integration test  Owner: TBD  Est: 1.5h
  - Dependencies: T74.1
  - Files: internal/nccl/nccl_test.go
  - Acceptance: Test initializes NCCL communicator across 2 GPUs (skip if < 2 GPUs).
    Each GPU has a different float32 buffer. ncclAllReduce(Sum) produces correct
    element-wise sum on both GPUs. ncclBroadcast from rank 0 sends data to rank 1.
    Uses goroutines (one per GPU) to simulate multi-rank within a process.
  - [ ] S74.2.1 Write 2-GPU AllReduce test with goroutines  Est: 30m
  - [ ] S74.2.2 Write 2-GPU Broadcast test  Est: 20m
  - [ ] S74.2.3 Write error handling test (invalid comm)  Est: 15m
  - [ ] S74.2.4 Run golangci-lint and go test -tags cuda -cover -race  Est: 5m

- [ ] T74.3 Run linters and verify coverage for E74  Owner: TBD  Est: 15m
  - Dependencies: T74.2
  - Acceptance: golangci-lint 0 issues. go test -tags cuda -cover -race passes.
    Coverage >= 95% on internal/nccl/.
  - [ ] S74.3.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S74.3.2 Fix any remaining issues  Est: 5m

#### E75: NCCL Strategy

Implement NcclStrategy[T] that performs gradient exchange directly on GPU memory
using NCCL, avoiding CPU round-trips.

- [ ] T75.1 Create NcclStrategy[T] struct  Owner: TBD  Est: 1.5h
  - Dependencies: E71, E74
  - Files: distributed/nccl_strategy.go (new)
  - Acceptance: NcclStrategy[T] implements InternalStrategy[T]. Fields: rank int,
    size int, deviceID int, comm *nccl.NcclComm, engine *compute.GPUEngine[T],
    stream *cuda.Stream, logger log.Logger. Static interface assertion
    var _ InternalStrategy[float32] = (*NcclStrategy[float32])(nil) compiles.
    All behind //go:build cuda.
  - [ ] S75.1.1 Create distributed/nccl_strategy.go with struct definition  Est: 15m
  - [ ] S75.1.2 Implement NewNcclStrategy constructor  Est: 15m
  - [ ] S75.1.3 Implement Init: create NCCL communicator with rank and size  Est: 20m
  - [ ] S75.1.4 Implement Rank(), Size() methods  Est: 5m
  - [ ] S75.1.5 Write constructor tests  Est: 15m
  - [ ] S75.1.6 Run golangci-lint and go test -cover  Est: 5m

- [ ] T75.2 Implement AllReduceGradients using NCCL  Owner: TBD  Est: 2h
  - Dependencies: T75.1
  - Files: distributed/nccl_strategy.go
  - Acceptance: NcclStrategy.AllReduceGradients(gradients) iterates gradient tensors,
    calls nccl.AllReduce on each tensor's device pointer directly (no D2H copy),
    using ncclAvg or ncclSum+divide. GPU stream synchronization after all reductions.
    Gradient tensors are updated in-place on device. Metrics: nccl_allreduce_count,
    nccl_allreduce_duration_seconds. Test: 2 GPUs, each with different gradient
    tensors, after AllReduce both have the average.
  - [ ] S75.2.1 Implement AllReduceGradients: iterate tensors, call ncclAllReduce  Est: 30m
  - [ ] S75.2.2 Add stream synchronization after all reductions  Est: 15m
  - [ ] S75.2.3 Handle GPU tensors: extract device pointer without D2H copy  Est: 15m
  - [ ] S75.2.4 Add metrics instrumentation  Est: 10m
  - [ ] S75.2.5 Write 2-GPU test: different gradients, verify average (skip if < 2 GPUs)  Est: 25m
  - [ ] S75.2.6 Run golangci-lint and go test -tags cuda -cover  Est: 5m

- [ ] T75.3 Implement Barrier and BroadcastTensor using NCCL  Owner: TBD  Est: 1h
  - Dependencies: T75.1
  - Files: distributed/nccl_strategy.go
  - Acceptance: NcclStrategy.Barrier() uses ncclAllReduce on a dummy 1-element buffer
    as a synchronization primitive (NCCL has no native barrier). BroadcastTensor(t,
    rootRank) uses ncclBroadcast to send a tensor from rootRank to all other ranks.
    Stream synchronized after each operation.
  - [ ] S75.3.1 Implement Barrier via dummy AllReduce  Est: 15m
  - [ ] S75.3.2 Implement BroadcastTensor via ncclBroadcast  Est: 20m
  - [ ] S75.3.3 Write tests for Barrier and Broadcast (skip if < 2 GPUs)  Est: 20m
  - [ ] S75.3.4 Run golangci-lint and go test -tags cuda -cover  Est: 5m

- [ ] T75.4 Implement Shutdown  Owner: TBD  Est: 30m
  - Dependencies: T75.1
  - Files: distributed/nccl_strategy.go
  - Acceptance: NcclStrategy.Shutdown() destroys the NCCL communicator via
    ncclCommDestroy. Idempotent via sync.Once. No panic on double call.
  - [ ] S75.4.1 Implement Shutdown with sync.Once  Est: 10m
  - [ ] S75.4.2 Write test: single shutdown, double shutdown  Est: 10m
  - [ ] S75.4.3 Run golangci-lint and go test -cover  Est: 5m

- [ ] T75.5 Run linters and verify coverage for E75  Owner: TBD  Est: 15m
  - Dependencies: T75.4
  - Acceptance: golangci-lint 0 issues on distributed/. go test -tags cuda -cover
    -race passes. Coverage >= 95% on nccl_strategy.go.
  - [ ] S75.5.1 Run golangci-lint, go vet, go test -tags cuda -cover -race  Est: 10m
  - [ ] S75.5.2 Fix any remaining issues  Est: 5m

#### E76: Phase 10 Final Verification

Run the full quality gate suite after all Phase 10 work is complete.

- [ ] T76.1 Run full test suite  Owner: TBD  Est: 30m
  - Dependencies: E70, E71, E72, E73, E74, E75
  - Acceptance: go test ./... -cover -race passes (CPU tests). go test -tags cuda
    ./... -cover -race passes (GPU tests). No regressions in existing packages.
    All multi-GPU tests skip gracefully on single-GPU or no-GPU systems.
  - [ ] S76.1.1 Run go test ./... -cover -race (CPU)  Est: 10m
  - [ ] S76.1.2 Run go test -tags cuda ./... -cover -race (GPU)  Est: 10m
  - [ ] S76.1.3 Verify multi-GPU tests skip gracefully  Est: 5m
  - [ ] S76.1.4 Fix any regressions  Est: 5m

- [ ] T76.2 Run linters  Owner: TBD  Est: 15m
  - Dependencies: T76.1
  - Acceptance: golangci-lint run ./... reports 0 issues. go vet ./... clean.
  - [ ] S76.2.1 Run golangci-lint run ./...  Est: 5m
  - [ ] S76.2.2 Run go vet ./...  Est: 5m
  - [ ] S76.2.3 Fix any remaining issues  Est: 5m

- [ ] T76.3 Update documentation  Owner: TBD  Est: 45m
  - Dependencies: T76.2
  - Files: docs/plan.md, docs/design.md, docs/gpu.md, docs/adr/ (new ADR)
  - Acceptance: docs/plan.md Phase 10 tasks marked complete. docs/design.md updated
    with multi-GPU section. docs/gpu.md updated with completion status. New ADR for
    multi-GPU architecture decisions.
  - [ ] S76.3.1 Update docs/plan.md  Est: 10m
  - [ ] S76.3.2 Update docs/design.md with multi-GPU section  Est: 15m
  - [ ] S76.3.3 Create docs/adr/007-multi-gpu-architecture.md  Est: 15m
  - [ ] S76.3.4 Update docs/gpu.md with completion status  Est: 5m

---

## 4. Timeline and Milestones

### Phase 10 Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M55 | Per-device memory pool | E70 | Pool keyed by (deviceID, byteSize); tests pass |
| M56 | Device-affine GPU engine | E71 | GPUEngine bound to specific device; SetDevice guard on all ops |
| M57 | Device-affine storage | E72 | GPUStorage tracks deviceID; cross-device transfer works |
| M58 | Multi-GPU inference | E73 | inference.Load("cuda:1") creates engine on device 1 |
| M59 | NCCL bindings | E74 | AllReduce and Broadcast work across 2 GPUs |
| M60 | NCCL strategy | E75 | NcclStrategy implements InternalStrategy with GPU-native ops |
| M61 | Phase 10 complete | E76 | Full suite green; docs updated |

### Recommended Sequence

1. **E70** (Per-device memory pool) -- Foundation; no dependencies.
2. **E71** (Device-affine engine) -- Depends on E70.
3. **E72** (Device-affine storage) -- Can partially parallel E71 after E70.
4. **E73** (Multi-GPU inference) -- Depends on E71 + E72.
5. **E74** (NCCL bindings) -- Independent of E70-E73; can start in parallel.
6. **E75** (NCCL strategy) -- Depends on E71 + E74.
7. **E76** (Final verification) -- After all epics.

Parallelism: E74 (NCCL bindings) can run in parallel with E70-E73.
E71 and E72 can partially overlap after E70 completes.

### Prior Phase Timeline

All 9 phases complete (2026-02-24 through 2026-03-03). 69 epics (E1-E69),
~200 tasks. Only E29 (GPU hardware validation) remains blocked on external
GCP GPU quota.

---

## 5. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | NCCL not available on target system | NCCL strategy unusable | Medium | NCCL is optional; falls back to gRPC-based exchange. Build tag gates NCCL code. |
| R2 | Multi-GPU tests cannot run in CI (no multi-GPU runner) | Reduced test coverage | High | Tests skip gracefully on < 2 GPUs. Validate manually on multi-GPU hardware. |
| R3 | SetDevice overhead in tight loops | Performance regression | Low | SetDevice is a no-op when current device matches requested. Benchmark before/after. |
| R4 | Cross-device D2D copy slower than expected on PCIe | Transfer bottleneck | Medium | Document NVLink vs PCIe expectations. Profile with nvidia-smi. |
| R5 | GCP GPU quota still blocked for E29 | Cannot validate on real hardware | High | Try alternative cloud provider or local hardware. E29 is independent of Phase 10. |
| R6 | Breaking existing single-GPU callers | Regression in Phases 2-3 | Medium | Variadic constructor defaults to device 0. Run all existing GPU tests after changes. |

---

## 6. Operating Procedure

### Definition of Done

A task is done when:
1. Implementation matches the acceptance criteria.
2. All existing tests pass (`go test ./... -count=1`).
3. New code has unit tests with >= 95% coverage.
4. `golangci-lint run ./package/` reports 0 issues.
5. `go vet ./package/` reports no issues.
6. Tests pass with `-race` flag.
7. Non-CUDA build (`go build ./...` without cuda tag) compiles.
8. CUDA build (`go build -tags cuda ./...`) compiles.
9. Changes are committed in a small commit touching one directory only.

### Review and QA Steps

1. Read existing implementation before writing code.
2. Write tests first or alongside implementation. Use table-driven tests.
3. After implementation, run `go test -cover ./package/` to verify coverage.
4. Run `golangci-lint run --fix ./package/` to fix lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. Run `go test ./... -count=1` to verify no regressions.
7. Run `go build ./...` (without cuda tag) to verify non-CUDA build.
8. Run `go build -tags cuda ./...` to verify CUDA build.
9. Multi-GPU tests must skip gracefully when fewer than 2 GPUs are available.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(cuda): add per-device memory pool`, `feat(compute): add device affinity to GPUEngine`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run linters and formatters before committing.

---

## 7. Progress Log

| Date | Phase | Summary |
|------|-------|---------|
| 2026-03-03 | 10 | Phase 10 planned: multi-GPU and distributed GPU support. 7 epics (E70-E76), 16 tasks. Three layers: device affinity (E70-E72), multi-GPU inference (E73), NCCL bindings + strategy (E74-E75), verification (E76). |
| 2026-03-03 | 9 | Multi-architecture support complete (6 model families) |
| 2026-03-03 | -- | ADRs extracted, plan.md trimmed from 3058 to 272 lines |
| 2026-03-02 | 8 | Embeddable inference library complete |
| 2026-03-02 | 7 | Architecture cleanup complete |
| 2026-03-02 | 6 | Open weights model import complete (13 new operators) |
| 2026-03-02 | 5 | Distributed protocol complete (96% coverage) |
| 2026-03-01 | 4 | Enterprise readiness complete (except E29 blocked) |
| 2026-03-01 | 2-3 | GPU engine + production readiness complete |
| 2026-02-25 | 1 | Test coverage complete (30/33 packages >= 95%) |
| 2026-02-24 | 1 | Initial plan created |

---

## 8. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout,
  GPU architecture, operations, and troubleshooting. It is the single reference
  document. Design decisions are in docs/adr/. Multi-GPU roadmap is in docs/gpu.md.
- **Phase 1-9:** Complete. See section 3 summaries and ADR files.
- **Phase 10:** Multi-GPU and distributed GPU support. In progress.
- **GPU hardware validation:** Blocked on GCP GPU quota (E29). Independent of Phase 10.
- **Key files for Phase 10:**
  - compute/gpu_engine.go -- GPUEngine struct (lines 27-36), constructor (lines 40-70)
  - tensor/gpu_storage.go -- GPUStorage struct (lines 17-21), constructors
  - internal/cuda/mempool.go -- MemPool (lines 13-16), Alloc (lines 28-40)
  - internal/cuda/runtime.go -- SetDevice (lines 74-81), Malloc (lines 30-39)
  - device/cuda_allocator.go -- cudaAllocator (lines 12-13)
  - device/cuda_device.go -- cudaDevice (lines 12-16), init (lines 39-48)
  - tensor/transfer.go -- ToGPU (lines 8-26)
  - inference/inference.go -- Load, hardcoded CPUEngine (line 149)
  - distributed/grpc_strategy.go -- tensorToProto (lines 367-385)
- **How to run tests:** `go test ./... -cover` for full suite. `go test -tags cuda ./...` for GPU.
- **How to build:** `go build ./...` (CPU). `go build -tags cuda ./...` (GPU).
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.
- **Multi-GPU tests:** Require >= 2 NVIDIA GPUs. Tests skip gracefully on single-GPU.

### External Dependencies

- GCP GPU quota increase for hardware validation (preference ID: zerfoo-gpu-test,
  project: numerai-488804).
- NCCL library (libnccl2) required for E74-E75. Available via CUDA Toolkit or
  apt-get install libnccl2 libnccl-dev.

---

## 9. Appendix

### Production Readiness Scorecard (After Phase 9)

| Category | Score | How Achieved |
|----------|-------|-------------|
| Architecture | 10/10 | Multi-architecture config parsing (E57); MLA attention variant (E66) |
| Core Functionality | 10/10 | 6 model families supported: Gemma, Llama, Mistral, Qwen, Phi, DeepSeek |
| Testing | 10/10 | Parity tests for all supported architectures (E59, E62, E65, E68) |
| Error Handling | 9/10 | Structured logging, RPC validation, context deadlines |
| Security | 8/10 | TLS/mTLS for gRPC; HF_TOKEN for gated models |
| Observability | 8/10 | Logging, metrics, pprof endpoints |
| Configuration | 10/10 | Architecture-aware config parsing with HuggingFace field mapping (E57) |
| Operations | 10/10 | CLI pull/run/serve, OpenAI-compatible HTTP API |
| Documentation | 10/10 | Consolidated design.md + ADRs; supported architectures table |
| CI/CD | 9/10 | Blocking tests, coverage gate, benchmark gate |
| Model Coverage | 10/10 | Covers >90% of open-weight model downloads on HuggingFace |

### New Packages and Files (Phases 1-9)

| Package / File | Purpose | Phase |
|---------|---------|-------|
| log/ | Structured logging with levels | 4 |
| metrics/runtime/ | Runtime metrics collection | 4 |
| config/ | File-based configuration loading | 4 |
| shutdown/ | Graceful shutdown coordinator | 4 |
| health/ | HTTP health check server | 4 |
| cmd/coverage-gate/ | CI coverage enforcement script | 4 |
| cmd/bench-compare/ | CI benchmark regression detection | 4 |
| distributed/worker_service.go | DistributedServiceServer (AllReduce, Barrier, Broadcast) | 5 |
| distributed/grpc_strategy.go | GrpcStrategy[T] over gRPC | 5 |
| distributed/integration_test.go | Multi-worker integration tests | 5 |
| distributed/worker_node.go | WorkerNode lifecycle management | 5 |
| cmd/cli/worker.go | Worker CLI subcommand | 5 |
| layers/activations/{softmax,erf}.go | Softmax, Erf layer nodes | 6 |
| layers/normalization/batch_norm.go | BatchNormalization inference mode | 6 |
| layers/core/{slice,pad,topk,conv2d,global_avg_pool,resize,moe,constant}.go | Core operators | 6 |
| tests/parity/{gemma3,siglip}_test.go | Model parity tests | 6 |
| pkg/tokenizer/{bpe,loader}.go | Production BPE tokenizer | 8 |
| generate/{kvcache,context,generator,sampling,stream}.go | Generation pipeline | 8 |
| registry/{registry,pull}.go | Model registry + HuggingFace download | 8 |
| inference/{inference,chat,embed}.go | High-level API | 8 |
| serve/server.go | OpenAI-compatible HTTP server | 8 |
| cmd/cli/{pull,run,serve}.go | CLI commands | 8 |
| inference/arch_config.go | Multi-architecture config parsing | 9 |
| model/param_resolver.go | Architecture-specific param resolution | 9 |
| layers/attention/{multi_head_latent_attention,mla_registry}.go | MLA for DeepSeek | 9 |
| tests/parity/{llama3,mistral,qwen,phi4,deepseek}_test.go | Parity tests | 9 |

### New Packages and Files (Phase 10 -- Planned)

| Package / File | Purpose | Epic |
|---------|---------|------|
| internal/nccl/nccl.go | NCCL CGo bindings (AllReduce, Broadcast, Comm) | E74 |
| distributed/nccl_strategy.go | NcclStrategy[T] for GPU-native gradient exchange | E75 |
| tests/parity/multigpu_test.go | Multi-GPU inference integration test | E73 |
