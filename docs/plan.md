# Zerfoo Development Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages. This plan covers two major efforts:

**Phase 1 (Completed): Test Coverage Improvement.** Raise every testable package to at least 95% statement coverage. This phase is complete. 30 of 33 testable packages are at or above 95%. Three packages (layers/gather 93.1%, layers/embeddings 93.5%, layers/features 93.8%) remain below 95% due to unreachable tensor.New error paths, documented as acceptable exceptions.

**Phase 2 (Active): GPU Engine Implementation.** Implement `compute.GPUEngine[T]` that satisfies the existing `compute.Engine[T]` interface. All existing layer code must remain untouched. GPU support must work for both training and inference without breaking the current API. The approach is incremental: start with tensor storage abstraction, then cuBLAS MatMul, then add GPU kernels one-by-one.

### Architecture Overview

The existing compute architecture centers on these types:

- `compute.Engine[T tensor.Numeric]` -- interface with 34 methods covering arithmetic, matrix ops, activations, reductions, and tensor manipulation. Every layer delegates all computation to this interface.
- `compute.CPUEngine[T]` -- the only existing Engine implementation. Uses `numeric.Arithmetic[T]` for element-wise ops, `internal/xblas` (gonum BLAS) for MatMul, and `parallelFor()` for multi-core CPU parallelism.
- `tensor.TensorNumeric[T]` -- core tensor type with fields: `shape []int`, `strides []int`, `data []T`, `isView bool`. The `data []T` field is a Go slice stored in row-major order.
- `device.Device` -- interface with `ID() string`, `GetAllocator() Allocator`, `Type() Type`. Only CPU is implemented. `device.CUDA` type constant exists as a placeholder.
- `device.Allocator` -- interface with `Allocate(size int) (any, error)` and `Free(ptr any) error`. Only CPU allocator exists (backed by `make([]byte, size)`).

Layers (e.g., `layers/core/linear.go`, `layers/activations/base_activation.go`) accept an `Engine[T]` at construction time and call engine methods like `MatMul`, `Add`, `UnaryOp`, etc. They never directly access tensor data for computation. This means swapping `CPUEngine` for `GPUEngine` requires zero layer code changes.

The critical challenge: `TensorNumeric[T].data` is `[]T` (a Go slice in CPU RAM). For GPU computation, data must reside in GPU device memory. The `data` field must become a device-aware storage interface so that:
- CPU path: zero overhead, `Data()` returns the underlying slice directly
- GPU path: data lives in CUDA device memory, `Data()` copies to host on demand

### Objectives

- O1 (Done): Every testable package at >= 95% statement coverage.
- O2: Implement `compute.GPUEngine[T]` satisfying `compute.Engine[T]` with no changes to any layer code.
- O3: Start with `Storage[T]` abstraction (critical path) replacing `TensorNumeric[T].data []T`.
- O4: Implement cuBLAS-backed `MatMul` via CGO as the first GPU operation.
- O5: Incrementally add GPU kernels: elementwise ops, Softmax, reductions.
- O6: All GPU code behind `//go:build cuda` build tag so non-CUDA builds are unaffected.

### Non-Goals

- Modifying any existing layer code to support GPU.
- Multi-GPU support or distributed GPU training in this phase.
- GPU support for non-float32 types initially (float32 first, others later).
- cuDNN integration (custom kernels only for now).
- AMD ROCm or OpenCL backends.
- Testing generated protobuf code (distributed/pb/).
- Unit-testing main() functions in cmd packages.

### Constraints and Assumptions

- Use Go standard library only for non-CUDA code. No third-party test frameworks.
- CUDA code uses CGO with `#cgo LDFLAGS: -lcublas -lcudart`.
- All CUDA-dependent code must be gated behind `//go:build cuda` build tags.
- The pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag (CPU tests; GPU tests are inherently single-threaded on device).
- CUDA Toolkit >= 11.0 required for GPU builds.
- Minimum GPU: NVIDIA with Compute Capability >= 7.0 (Volta or newer).

### Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Per-package coverage | >= 95% for all testable packages | `go test ./... -cover` |
| Engine interface compliance | GPUEngine passes static type assertion | `var _ Engine[float32] = (*GPUEngine[float32])(nil)` |
| MatMul parity | GPU MatMul matches CPU MatMul within 1e-5 relative error | Parity test comparing GPU vs CPU output |
| MatMul speedup | >= 10x for 1024x1024 matrices vs CPU | Benchmark with `go test -bench` |
| Non-CUDA build | Project builds and all tests pass without CUDA | `go test ./... -cover` without cuda build tag |
| Lint | Zero issues | `golangci-lint run ./...` |

---

## 2. Scope and Deliverables

### In Scope

- Storage[T] interface and CPUStorage[T] implementation in tensor package.
- GPUStorage[T] implementation wrapping CUDA device memory.
- Refactoring TensorNumeric[T] to use Storage[T] internally.
- CUDA allocator implementing device.Allocator.
- cuBLAS CGO bindings in internal/cublas package.
- GPUEngine[T] struct implementing all 34 Engine[T] methods.
- GPU kernels for: MatMul (cuBLAS), Add, Sub, Mul, Div, AddScalar, MulScalar, DivScalar, Pow, Exp, Log, Tanh, TanhPrime, Sqrt, Rsqrt, Softmax, Sum, ReduceSum, ReduceMean.
- GPU implementations for: Zero, Zeros, Fill, Copy, Transpose, Reshape, Split, Concat, Repeat, Gather, ScatterAdd, OneHot, RandomUniform, UnaryOp.
- Parity tests comparing GPU output vs CPU output for every method.
- Benchmarks for MatMul and key operations.
- Remaining test coverage tasks from Phase 1.

### Out of Scope

- Multi-GPU or distributed GPU.
- cuDNN, TensorRT, or other NVIDIA library integration.
- AMD ROCm or OpenCL backends.
- Automatic device placement or tensor migration policies.
- Mixed precision training.

### Deliverables

| ID | Description | Owner | Acceptance Criteria |
|----|-------------|-------|---------------------|
| D1 | Phase 1 test coverage (done) | TBD | 30/33 packages >= 95%, 3 documented exceptions |
| D7 | Storage[T] interface + CPUStorage[T] | TBD | TensorNumeric uses Storage[T]; all existing tests pass unchanged |
| D8 | CUDA allocator + GPUStorage[T] | TBD | Allocate/free GPU memory; host-device transfers work correctly |
| D9 | cuBLAS MatMul | TBD | GPUEngine.MatMul produces results matching CPUEngine within 1e-5 |
| D10 | GPU elementwise ops | TBD | Add, Sub, Mul, Div, scalar variants, Pow all pass parity tests |
| D11 | GPU activation + math ops | TBD | Exp, Log, Tanh, TanhPrime, Sqrt, Rsqrt, Softmax pass parity tests |
| D12 | GPU reductions + tensor ops | TBD | Sum, ReduceSum, ReduceMean, Transpose, Reshape, etc. pass parity |
| D13 | Full Engine compliance | TBD | `var _ Engine[float32] = (*GPUEngine[float32])(nil)` compiles |

---

## 3. Checkable Work Breakdown

### Phase 1: Test Coverage (Completed)

#### E1: Zero-Coverage Packages -- COMPLETED

- [x] T1.1 Add tests for pkg/tokenizer  Completed: 2026 02 24  Result: 100%
- [x] T1.2 Add tests for layers/gather  Completed: 2026 02 25  Result: 93.1% (tensor.New gaps)
- [x] T1.3 Add tests for layers/reducesum  Completed: 2026 02 24  Result: 95.9%
- [x] T1.4 Add tests for layers/registry  Completed: 2026 02 24  Result: 100%
- [x] T1.5 Add tests for internal/xblas  Completed: 2026 02 24  Result: 100%

#### E2: Sub-50% Coverage Packages -- COMPLETED

- [x] T2.1 Raise training/ to >= 95%  Completed: 2026 02 24  Result: 95.7%
- [x] T2.2 Raise cmd/cli/ to >= 95%  Completed: 2026 02 24  Result: 96.5%
- [x] T2.3 Raise layers/normalization/ to >= 95%  Completed: 2026 02 24  Result: 96.6%
- [x] T2.4 Raise model/ to >= 95%  Completed: 2026 02 24  Result: 95.4%

#### E3: 50-69% Coverage Packages -- COMPLETED

- [x] T3.1 Raise layers/attention/ to >= 95%  Completed: 2026 02 24  Result: 95.1%
- [x] T3.2 Raise layers/core/ to >= 95%  Completed: 2026 02 24  Result: 96.0%
- [x] T3.3 Raise tensor/ to >= 95%  Completed: 2026 02 24  Result: 98.9%
- [x] T3.4 Raise layers/activations/ to >= 95%  Completed: 2026 02 24  Result: 97.1%
- [x] T3.5 Raise layers/recurrent/ to >= 95%  Completed: 2026 02 24  Result: 96.7%
- [x] T3.6 Raise graph/ to >= 95%  Completed: 2026 02 24  Result: 97.0%
- [x] T3.7 Raise training/loss/ to >= 95%  Completed: 2026 02 24  Result: 96.3%
- [x] T3.8 Raise layers/tokenizers/ to >= 95%  Completed: 2026 02 24  Result: 100%
- [x] T3.9 Raise layers/transpose/ to >= 95%  Completed: 2026 02 24  Result: 97.2%
- [x] T3.10 Raise numeric/ to >= 95%  Completed: 2026 02 24  Result: 98.5%

#### E4: 70-89% Coverage Packages -- COMPLETED

- [x] T4.1 Raise layers/embeddings/ to >= 95%  Completed: 2026 02 25  Result: 93.5% (tensor.New gaps)
- [x] T4.2 Raise layers/transformer/ to >= 95%  Completed: 2026 02 24  Result: 96.4%
- [x] T4.3 Raise distributed/ to >= 95%  Completed: 2026 02 24  Result: 96.5%
- [x] T4.4 Raise model/hrm/ to >= 95%  Completed: 2026 02 24  Result: 98.1%
- [x] T4.5 Raise training/optimizer/ to >= 95%  Completed: 2026 02 24  Result: 97.4%
- [x] T4.6 Raise layers/features/ to >= 95%  Completed: 2026 02 25  Result: 93.8% (tensor.New gaps)
- [x] T4.7 Raise layers/components/ to >= 95%  Completed: 2026 02 24  Result: 100%
- [x] T4.8 Raise layers/hrm/ to >= 95%  Completed: 2026 02 25  Result: 95.5%
- [x] T4.9 Raise compute/ to >= 95%  Completed: 2026 02 25  Result: 96.2%
- [x] T4.10 Raise distributed/coordinator/ to >= 95%  Completed: 2026 02 25  Result: 99.1%

#### E5: Near-Target Packages -- COMPLETED

- [x] T5.1 Raise data/ to >= 95%  Completed: 2026 02 25  Result: 100%
- [x] T5.2 Raise features/ to >= 95%  Completed: 2026 02 24  Result: 99.0%

#### E6: Test Utility Validation (Best Effort)

- [ ] T6.1 Add targeted tests for tests/internal/testutil  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: Math helpers (MeanRelativeError, TopKAgreement, RelError) tested for correctness.
  - [ ] S6.1.1 Write tests for MeanRelativeError with known inputs  Est: 15m
  - [ ] S6.1.2 Write tests for TopKAgreement with known overlap  Est: 15m
  - [ ] S6.1.3 Write tests for RelError edge cases (zero denominator)  Est: 10m
  - [ ] S6.1.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T6.2 Add targeted tests for testing/testutils mock correctness  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: MockEngine key methods tested.
  - [ ] S6.2.1 Write tests for assertion helpers (AssertEqual, AssertError, etc.)  Est: 20m
  - [ ] S6.2.2 Write tests for MockEngine interface compliance  Est: 20m
  - [ ] S6.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E7: Final Verification (Phase 1)

- [ ] T7.1 Run full test suite with coverage  Owner: TBD  Est: 30m
  - Dependencies: E1, E2, E3, E4, E5
  - Acceptance: Every testable package shows >= 95% in `go test ./... -cover` (with documented exceptions)
  - [ ] S7.1.1 Run go test ./... -cover and capture output  Est: 10m
  - [ ] S7.1.2 Verify each package meets target; list any exceptions with justification  Est: 10m
  - [ ] S7.1.3 Run go test ./... -race and verify zero races  Est: 10m

- [ ] T7.2 Run linters and formatters  Owner: TBD  Est: 15m
  - Dependencies: T7.1
  - Acceptance: golangci-lint 0 issues, go vet clean, gofmt clean
  - [ ] S7.2.1 Run golangci-lint run ./...  Est: 5m
  - [ ] S7.2.2 Run go vet ./...  Est: 5m
  - [ ] S7.2.3 Run gofmt -l . and verify no files  Est: 5m

#### Documented Coverage Exceptions

Three packages are below the 95% target. In all three cases, the remaining uncovered
code consists exclusively of `tensor.New[T]()` error handling that cannot be triggered
with valid inputs. These are defensive checks against memory allocation failures.

| Package | Coverage | Uncovered Lines | Justification |
|---------|----------|-----------------|---------------|
| layers/gather | 93.1% | gather.go:71-72, 94-95, 127-128, 145-147; registry.go:60-62 | All tensor.New error paths; tensor.New with valid shape never fails |
| layers/embeddings | 93.5% | token_embedding.go:64-66, 84-86, 149-151, 156-158, 165-167, 191-193, 224-226; rotary_positional_embedding.go:92-94, 96-98 | All tensor.New/NewParameter error paths |
| layers/features | 93.8% | spectral.go:63-65, 81-83, 99-101 | All tensor.New error paths in Forward/Backward |

---

### Phase 2: GPU Engine Implementation (Active)

#### E8: Tensor Storage Abstraction (Critical Path)

This epic replaces `TensorNumeric[T].data []T` with a `Storage[T]` interface. This is the foundation for all GPU work. The change must be fully backward-compatible: all existing code that calls `Data()`, `SetData()`, or constructs tensors via `tensor.New()` must work identically without modification.

**Storage[T] Interface Design (in tensor package):**

```go
// Storage[T] abstracts over CPU and GPU tensor data storage.
type Storage[T Numeric] interface {
    // Len returns the number of elements.
    Len() int
    // Slice returns a CPU-accessible []T. For CPU storage this is the
    // underlying slice directly (zero copy). For GPU storage this copies
    // device memory to a new host slice.
    Slice() []T
    // Set replaces the storage contents from a CPU slice. For GPU storage
    // this copies host data to device memory.
    Set(data []T)
    // DeviceType returns the device type this storage resides on.
    DeviceType() device.Type
}
```

**CPUStorage[T] Implementation:**

```go
type CPUStorage[T Numeric] struct {
    data []T
}

func (s *CPUStorage[T]) Len() int         { return len(s.data) }
func (s *CPUStorage[T]) Slice() []T       { return s.data }
func (s *CPUStorage[T]) Set(data []T)     { s.data = data }
func (s *CPUStorage[T]) DeviceType() device.Type { return device.CPU }
```

**TensorNumeric[T] Refactoring:**

```go
type TensorNumeric[T Numeric] struct {
    shape   []int
    strides []int
    storage Storage[T]   // was: data []T
    isView  bool
}

// Data() delegates to storage -- backward compatible
func (t *TensorNumeric[T]) Data() []T    { return t.storage.Slice() }
func (t *TensorNumeric[T]) SetData(d []T) { t.storage.Set(d) }

// New accessor for engine-level code
func (t *TensorNumeric[T]) GetStorage() Storage[T] { return t.storage }
func (t *TensorNumeric[T]) SetStorage(s Storage[T]) { t.storage = s }
```

- [x] T8.1 Define Storage[T] interface in tensor/storage.go  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Interface compiles. Has Len(), Slice(), Set(), DeviceType() methods.
  - Risk: Interface must be minimal to avoid constraining future GPU implementations.
  - [ ] S8.1.1 Create tensor/storage.go with Storage[T] interface definition  Est: 15m
  - [ ] S8.1.2 Run golangci-lint on tensor package  Est: 5m
  - [ ] S8.1.3 Write unit tests for the interface contract (compile-time checks)  Est: 10m

- [x] T8.2 Implement CPUStorage[T] in tensor/storage.go  Completed: 2026 03 01
  - Dependencies: T8.1
  - Acceptance: CPUStorage[T] satisfies Storage[T]. Slice() returns underlying slice with zero copy. Len() and Set() work correctly.
  - [ ] S8.2.1 Implement CPUStorage[T] struct and all interface methods  Est: 15m
  - [ ] S8.2.2 Add NewCPUStorage[T](data []T) constructor  Est: 10m
  - [ ] S8.2.3 Write unit tests: Len, Slice identity (same pointer), Set, DeviceType  Est: 15m
  - [ ] S8.2.4 Run golangci-lint and go test -cover on tensor package  Est: 5m

- [x] T8.3 Refactor TensorNumeric[T] to use Storage[T]  Completed: 2026 03 01
  - Dependencies: T8.2
  - Acceptance: `data []T` field replaced by `storage Storage[T]`. Data() returns storage.Slice(). SetData() calls storage.Set(). New GetStorage()/SetStorage() accessors added. All existing tensor tests pass without modification.
  - Risk: This touches the core tensor type used by every package. Must not change Data() return semantics.
  - [ ] S8.3.1 Replace `data []T` field with `storage Storage[T]` in TensorNumeric struct  Est: 15m
  - [ ] S8.3.2 Update Data() to return storage.Slice()  Est: 5m
  - [ ] S8.3.3 Update SetData() to call storage.Set()  Est: 5m
  - [ ] S8.3.4 Add GetStorage() and SetStorage() methods  Est: 5m
  - [ ] S8.3.5 Update tensor.New[T]() to create CPUStorage internally  Est: 10m
  - [ ] S8.3.6 Update Copy(), Each(), Bytes(), String() to use storage  Est: 15m
  - [ ] S8.3.7 Update NewFromBytes, NewFromType to use storage  Est: 10m
  - [ ] S8.3.8 Run full test suite: go test ./... to confirm zero regressions  Est: 15m
  - [ ] S8.3.9 Run golangci-lint run ./... to confirm no lint issues  Est: 10m

- [x] T8.4 Verify CPUEngine works with Storage[T] changes  Completed: 2026 03 01
  - Dependencies: T8.3
  - Acceptance: CPUEngine still passes all existing tests. No behavioral changes.
  - [ ] S8.4.1 Verify CPUEngine code still works (it calls Data() which now delegates to storage)  Est: 10m
  - [ ] S8.4.2 Run compute package tests: go test ./compute/ -cover  Est: 10m
  - [ ] S8.4.3 Run golangci-lint on compute package  Est: 5m
  - [ ] S8.4.4 Run full test suite to confirm no regressions  Est: 5m

#### E9: CUDA Device and Memory Management

Implement CUDA device memory allocation and GPUStorage[T] using CGO. All CUDA code must be behind `//go:build cuda` build tags.

**GPUStorage[T] Design:**

```go
// In tensor/gpu_storage.go (//go:build cuda)
type GPUStorage[T Numeric] struct {
    devicePtr unsafe.Pointer  // CUDA device pointer from cudaMalloc
    length    int             // number of elements
    byteSize  int             // total bytes = length * sizeof(T)
}

func (s *GPUStorage[T]) Len() int   { return s.length }
func (s *GPUStorage[T]) Slice() []T {
    // Copy from GPU to a new CPU slice
    hostData := make([]T, s.length)
    cudaMemcpyDtoH(hostData, s.devicePtr, s.byteSize)
    return hostData
}
func (s *GPUStorage[T]) Set(data []T) {
    cudaMemcpyHtoD(s.devicePtr, data, s.byteSize)
}
func (s *GPUStorage[T]) DeviceType() device.Type { return device.CUDA }
// GPU-specific accessor
func (s *GPUStorage[T]) Ptr() unsafe.Pointer { return s.devicePtr }
```

- [x] T9.1 Create internal/cuda/runtime.go with CGO bindings for CUDA runtime  Completed: 2026 03 01
  - Dependencies: T8.1
  - Acceptance: cudaMalloc, cudaFree, cudaMemcpy (H2D, D2H, D2D), cudaGetDeviceCount, cudaSetDevice wrapped and callable from Go. Compiles with `go build -tags cuda`.
  - [ ] S9.1.1 Create internal/cuda/ package directory  Est: 5m
  - [ ] S9.1.2 Write runtime.go with CGO includes for cuda_runtime.h  Est: 20m
  - [ ] S9.1.3 Implement CudaMalloc(size int) (unsafe.Pointer, error)  Est: 10m
  - [ ] S9.1.4 Implement CudaFree(ptr unsafe.Pointer) error  Est: 5m
  - [ ] S9.1.5 Implement CudaMemcpy variants (HtoD, DtoH, DtoD)  Est: 15m
  - [ ] S9.1.6 Run golangci-lint on internal/cuda package  Est: 5m

- [x] T9.2 Implement CUDA allocator in device/cuda_allocator.go  Completed: 2026 03 01
  - Dependencies: T9.1
  - Acceptance: CUDAAllocator implements device.Allocator. Allocate returns a CUDA device pointer. Free calls cudaFree.
  - [ ] S9.2.1 Create device/cuda_allocator.go (build tag: cuda)  Est: 15m
  - [ ] S9.2.2 Implement Allocate() using internal/cuda.CudaMalloc  Est: 10m
  - [ ] S9.2.3 Implement Free() using internal/cuda.CudaFree  Est: 5m
  - [ ] S9.2.4 Write unit tests with actual GPU allocation (build tag: cuda)  Est: 10m
  - [ ] S9.2.5 Run golangci-lint on device package  Est: 5m

- [x] T9.3 Register CUDA device in device registry  Completed: 2026 03 01
  - Dependencies: T9.2
  - Acceptance: `device.Get("cuda:0")` returns a valid CUDA device on machines with NVIDIA GPU. Device reports Type() == device.CUDA.
  - [ ] S9.3.1 Create device/cuda_device.go with cudaDevice struct (build tag: cuda)  Est: 10m
  - [ ] S9.3.2 Register CUDA device in init() using cudaGetDeviceCount  Est: 10m
  - [ ] S9.3.3 Write unit test verifying device registration  Est: 5m
  - [ ] S9.3.4 Run golangci-lint on device package  Est: 5m

- [x] T9.4 Implement GPUStorage[T] in tensor/gpu_storage.go  Completed: 2026 03 01
  - Dependencies: T8.1, T9.1
  - Acceptance: GPUStorage[T] satisfies Storage[T]. Slice() copies from GPU to CPU host slice. Set() copies from CPU to GPU. Ptr() returns CUDA device pointer. DeviceType() returns device.CUDA.
  - [ ] S9.4.1 Create tensor/gpu_storage.go (build tag: cuda) with struct definition  Est: 10m
  - [ ] S9.4.2 Implement NewGPUStorage[T](length int) constructor using cudaMalloc  Est: 10m
  - [ ] S9.4.3 Implement NewGPUStorageFromSlice[T](data []T) that allocates and copies H2D  Est: 10m
  - [ ] S9.4.4 Implement Len(), Slice(), Set(), DeviceType(), Ptr()  Est: 10m
  - [ ] S9.4.5 Implement Free() for explicit deallocation  Est: 5m
  - [ ] S9.4.6 Write unit tests: round-trip H2D then D2H matches original data  Est: 10m
  - [ ] S9.4.7 Run golangci-lint on tensor package  Est: 5m

- [x] T9.5 Add tensor helper: ToGPU / ToCPU transfer functions  Completed: 2026 03 01
  - Dependencies: T8.3, T9.4
  - Acceptance: `ToGPU(t)` creates a new tensor with GPUStorage containing the same data. `ToCPU(t)` creates a new tensor with CPUStorage. Shape and strides are preserved.
  - [ ] S9.5.1 Implement ToGPU[T](t *TensorNumeric[T]) in tensor/transfer.go (build tag: cuda)  Est: 15m
  - [ ] S9.5.2 Implement ToCPU[T](t *TensorNumeric[T]) in tensor/transfer.go (build tag: cuda)  Est: 10m
  - [ ] S9.5.3 Write round-trip test: create CPU tensor, transfer to GPU, transfer back, compare  Est: 10m
  - [ ] S9.5.4 Write test for shape/strides preservation  Est: 5m
  - [ ] S9.5.5 Run golangci-lint on tensor package  Est: 5m

#### E10: cuBLAS MatMul via CGO

Wrap cuBLAS sgemm for float32 matrix multiplication. This gives the single biggest speedup with the least implementation work.

**cuBLAS Binding Design:**

```go
// internal/cublas/cublas.go (//go:build cuda)
// #cgo LDFLAGS: -lcublas
// #include <cublas_v2.h>
import "C"

func Sgemm(handle C.cublasHandle_t, m, n, k int, alpha float32,
    a unsafe.Pointer, lda int, b unsafe.Pointer, ldb int,
    beta float32, c unsafe.Pointer, ldc int) error
```

- [x] T10.1 Create internal/cublas package with CGO bindings  Completed: 2026 03 01
  - Dependencies: T9.1
  - Acceptance: cublasSgemm is callable from Go. cuBLAS handle creation and destruction work. Build tag: cuda.
  - [ ] S10.1.1 Create internal/cublas/ package directory  Est: 5m
  - [ ] S10.1.2 Write cublas.go with CGO includes for cublas_v2.h  Est: 10m
  - [ ] S10.1.3 Implement handle management: CreateHandle(), DestroyHandle()  Est: 10m
  - [ ] S10.1.4 Implement Sgemm wrapping cublasSgemm  Est: 15m
  - [ ] S10.1.5 Note: cuBLAS uses column-major; handle row-major to col-major conversion by computing B^T * A^T = (AB)^T  Est: 10m
  - [ ] S10.1.6 Run golangci-lint on internal/cublas  Est: 5m
  - [ ] S10.1.7 Write unit test: multiply two known matrices, verify result  Est: 5m

- [x] T10.2 Create GPUEngine[T] struct skeleton  Completed: 2026 03 01
  - Dependencies: T9.4, T10.1
  - Acceptance: GPUEngine[T] struct exists with cuBLAS handle field. Constructor creates handle. Ops() returns numeric.Arithmetic[T]. Static type assertion compiles (all methods stubbed with CPU fallback).
  - [ ] S10.2.1 Create compute/gpu_engine.go (build tag: cuda) with struct definition  Est: 10m
  - [ ] S10.2.2 Add cublasHandle, ops fields and NewGPUEngine constructor  Est: 10m
  - [ ] S10.2.3 Implement Ops() method  Est: 5m
  - [ ] S10.2.4 Add ensureGPU helper that checks if tensor storage is GPUStorage  Est: 10m
  - [ ] S10.2.5 Add Close() method to destroy cuBLAS handle  Est: 5m
  - [ ] S10.2.6 Run golangci-lint on compute package  Est: 5m

- [x] T10.3 Implement GPUEngine.MatMul using cuBLAS  Completed: 2026 03 01
  - Dependencies: T10.2
  - Acceptance: GPUEngine.MatMul produces correct results for 2D and batched float32 matrix multiplications. Parity test with CPUEngine: max relative error < 1e-5 for random 128x128 matrices.
  - Risk: cuBLAS column-major vs row-major conversion must be correct.
  - [ ] S10.3.1 Implement MatMul for 2D case using cuBLAS Sgemm  Est: 30m
  - [ ] S10.3.2 Implement batched MatMul (loop over batch dims calling Sgemm per batch)  Est: 20m
  - [ ] S10.3.3 Handle broadcasting case (a is [batch..., m, k], b is [k, n])  Est: 15m
  - [ ] S10.3.4 Write parity test: random matrices, compare GPU vs CPU output  Est: 15m
  - [ ] S10.3.5 Write benchmark: 1024x1024 MatMul GPU vs CPU  Est: 5m
  - [ ] S10.3.6 Run golangci-lint on compute package  Est: 5m

- [x] T10.4 Stub remaining 33 Engine methods with CPU fallback  Completed: 2026 03 01
  - Dependencies: T10.2, T9.5
  - Acceptance: Every Engine[T] method is implemented. Non-MatMul methods transfer tensor to CPU, delegate to CPUEngine, then transfer result back to GPU. Static type assertion: `var _ Engine[float32] = (*GPUEngine[float32])(nil)` compiles.
  - [ ] S10.4.1 Implement fallbackToCPU helper: copies input tensors to CPU, calls CPUEngine method, copies result to GPU  Est: 20m
  - [ ] S10.4.2 Stub all unary ops (UnaryOp, Tanh, TanhPrime, Exp, Log, Sqrt, Rsqrt, Softmax) using fallback  Est: 10m
  - [ ] S10.4.3 Stub all binary ops (Add, Sub, Mul, Div, Pow) using fallback  Est: 10m
  - [ ] S10.4.4 Stub all scalar ops (AddScalar, MulScalar, DivScalar) using fallback  Est: 5m
  - [ ] S10.4.5 Stub tensor manipulation (Transpose, Reshape, Split, Concat, Repeat, Copy, Zero, Zeros, Fill) using fallback  Est: 10m
  - [ ] S10.4.6 Stub remaining (Sum, ReduceSum, ReduceMean, RandomUniform, Gather, ScatterAdd, OneHot) using fallback  Est: 10m
  - [ ] S10.4.7 Add static type assertion  Est: 5m
  - [ ] S10.4.8 Write integration test: Linear layer forward pass on GPUEngine  Est: 10m
  - [ ] S10.4.9 Run golangci-lint on compute package  Est: 5m

#### E11: GPU Elementwise Operations (CUDA Kernels)

Replace CPU fallbacks with native CUDA kernels for elementwise operations. Each kernel is a .cu file compiled via CGO or a Go-based PTX approach.

- [x] T11.1 Create CUDA kernel infrastructure  Completed: 2026 03 01
  - Dependencies: T10.4
  - Acceptance: A pattern for writing, compiling, and calling CUDA kernels from Go via CGO is established. One example kernel (vector add) works end-to-end.
  - [ ] S11.1.1 Create internal/cuda/kernels/ directory with build infrastructure  Est: 15m
  - [ ] S11.1.2 Write elementwise_ops.cu with vector_add kernel  Est: 15m
  - [ ] S11.1.3 Create Go wrapper in internal/cuda/kernels/elementwise.go  Est: 15m
  - [ ] S11.1.4 Write test verifying vector_add kernel produces correct output  Est: 10m
  - [ ] S11.1.5 Run golangci-lint  Est: 5m

- [x] T11.2 Implement GPU Add, Sub, Mul, Div kernels  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: GPU Add, Sub, Mul, Div produce results matching CPUEngine within 1e-6. Broadcasting is supported.
  - [ ] S11.2.1 Write CUDA kernels for add, sub, mul, div with broadcasting  Est: 20m
  - [ ] S11.2.2 Write Go wrappers calling kernels  Est: 10m
  - [ ] S11.2.3 Wire into GPUEngine replacing CPU fallback for Add, Sub, Mul, Div  Est: 10m
  - [ ] S11.2.4 Write parity tests: GPU vs CPU for each operation  Est: 15m
  - [ ] S11.2.5 Run golangci-lint on compute package  Est: 5m

- [x] T11.3 Implement GPU scalar ops and Pow kernel  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: AddScalar, MulScalar, DivScalar, Pow produce correct results on GPU.
  - [ ] S11.3.1 Write CUDA kernels for scalar add, mul, div, and pow  Est: 15m
  - [ ] S11.3.2 Wire into GPUEngine  Est: 10m
  - [ ] S11.3.3 Write parity tests  Est: 15m
  - [ ] S11.3.4 Run golangci-lint  Est: 5m

#### E12: GPU Activation and Math Function Kernels

- [x] T12.1 Implement GPU Exp, Log, Sqrt, Rsqrt kernels  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: Each operation matches CPU output within 1e-6 relative error.
  - [ ] S12.1.1 Write CUDA kernels for exp, log, sqrt, rsqrt (unary elementwise)  Est: 15m
  - [ ] S12.1.2 Wire into GPUEngine  Est: 10m
  - [ ] S12.1.3 Write parity tests for each operation  Est: 15m
  - [ ] S12.1.4 Run golangci-lint  Est: 5m

- [x] T12.2 Implement GPU Tanh, TanhPrime kernels  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: Tanh and TanhPrime match CPU output within 1e-6.
  - [ ] S12.2.1 Write CUDA kernels for tanh and tanh_prime (tanh_prime = (1-tanh^2) * upstream)  Est: 10m
  - [ ] S12.2.2 Wire into GPUEngine  Est: 5m
  - [ ] S12.2.3 Write parity tests  Est: 10m
  - [ ] S12.2.4 Run golangci-lint  Est: 5m

- [x] T12.3 Implement GPU Softmax kernel  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: Softmax along any valid axis matches CPU output within 1e-5. Numerically stable (max subtraction before exp).
  - Risk: Reduction within softmax requires shared memory or multi-pass kernel.
  - [ ] S12.3.1 Write CUDA kernel for softmax with max-subtraction stability  Est: 25m
  - [ ] S12.3.2 Handle arbitrary axis by computing outer/inner/axisSize strides  Est: 10m
  - [ ] S12.3.3 Wire into GPUEngine  Est: 5m
  - [ ] S12.3.4 Write parity tests: 1D, 2D (axis=0, axis=1), 3D cases  Est: 15m
  - [ ] S12.3.5 Run golangci-lint  Est: 5m

#### E13: GPU Reduction and Tensor Manipulation Kernels

- [x] T13.1 Implement GPU Sum, ReduceSum, ReduceMean kernels  Completed: 2026 03 01
  - Dependencies: T11.1
  - Acceptance: Sum/ReduceSum/ReduceMean match CPU output within 1e-5 for all valid axes and keepDims settings.
  - Risk: Parallel reduction requires careful shared-memory design.
  - [ ] S13.1.1 Write CUDA reduction kernel for sum along axis  Est: 20m
  - [ ] S13.1.2 Implement ReduceSum (delegates to Sum implementation)  Est: 5m
  - [ ] S13.1.3 Implement ReduceMean (Sum then DivScalar)  Est: 10m
  - [ ] S13.1.4 Wire into GPUEngine  Est: 5m
  - [ ] S13.1.5 Write parity tests: multiple axes, keepDims=true/false  Est: 15m
  - [ ] S13.1.6 Run golangci-lint  Est: 5m

- [ ] T13.2 Implement GPU Transpose kernel  Owner: TBD  Est: 45m
  - Dependencies: T11.1
  - Acceptance: Transpose with arbitrary axes permutation matches CPU output exactly.
  - [ ] S13.2.1 Write CUDA kernel for general N-D transpose  Est: 20m
  - [ ] S13.2.2 Wire into GPUEngine  Est: 5m
  - [ ] S13.2.3 Write parity tests: 2D default, 3D with various axes  Est: 15m
  - [ ] S13.2.4 Run golangci-lint  Est: 5m

- [ ] T13.3 Implement GPU Zero, Zeros, Fill, Copy  Owner: TBD  Est: 30m
  - Dependencies: T9.1
  - Acceptance: Zero fills GPU memory with zeros via cudaMemset. Fill sets all elements to a value. Copy performs D2D copy.
  - [ ] S13.3.1 Implement Zero using cudaMemset  Est: 5m
  - [ ] S13.3.2 Implement Zeros (allocate if shape provided, then Zero)  Est: 5m
  - [ ] S13.3.3 Implement Fill with a simple CUDA kernel  Est: 10m
  - [ ] S13.3.4 Implement Copy using cudaMemcpy D2D  Est: 5m
  - [ ] S13.3.5 Run golangci-lint  Est: 5m

- [ ] T13.4 Implement GPU Reshape, Split, Concat, Repeat  Owner: TBD  Est: 60m
  - Dependencies: T11.1
  - Acceptance: Each operation matches CPU output exactly. Reshape is metadata-only (no kernel needed).
  - [ ] S13.4.1 Implement Reshape (change shape/strides, copy data if needed)  Est: 10m
  - [ ] S13.4.2 Write CUDA kernel for Split  Est: 15m
  - [ ] S13.4.3 Write CUDA kernel for Concat  Est: 15m
  - [ ] S13.4.4 Write CUDA kernel for Repeat  Est: 10m
  - [ ] S13.4.5 Write parity tests for each operation  Est: 15m
  - [ ] S13.4.6 Run golangci-lint  Est: 5m

- [ ] T13.5 Implement GPU UnaryOp, Gather, ScatterAdd, OneHot, RandomUniform  Owner: TBD  Est: 60m
  - Dependencies: T11.1
  - Acceptance: All operations match CPU output. UnaryOp falls back to CPU (arbitrary Go func cannot run on GPU). RandomUniform uses cuRAND or CPU fallback.
  - [ ] S13.5.1 Implement UnaryOp via CPU fallback (Go functions cannot execute on GPU)  Est: 5m
  - [ ] S13.5.2 Write CUDA kernel for Gather (embedding lookup)  Est: 15m
  - [ ] S13.5.3 Write CUDA kernel for ScatterAdd  Est: 15m
  - [ ] S13.5.4 Write CUDA kernel for OneHot  Est: 10m
  - [ ] S13.5.5 Implement RandomUniform using cuRAND or CPU fallback  Est: 10m
  - [ ] S13.5.6 Write parity tests  Est: 10m
  - [ ] S13.5.7 Run golangci-lint  Est: 5m

#### E14: Integration Testing and Benchmarks

- [x] T14.1 End-to-end Linear layer test on GPU  Completed: 2026 03 01
  - Dependencies: T10.3, T10.4
  - Acceptance: Create Linear layer with GPUEngine, run forward and backward pass, verify output matches CPUEngine within 1e-5.
  - [ ] S14.1.1 Write test: construct Linear with GPUEngine, forward pass  Est: 15m
  - [ ] S14.1.2 Write test: backward pass, verify gradients  Est: 15m
  - [ ] S14.1.3 Compare GPU gradients with CPU gradients  Est: 10m
  - [ ] S14.1.4 Run golangci-lint  Est: 5m

- [ ] T14.2 End-to-end Transformer layer test on GPU  Owner: TBD  Est: 60m
  - Dependencies: E11, E12, E13
  - Acceptance: Full transformer forward pass on GPUEngine produces output matching CPUEngine within 1e-4.
  - [ ] S14.2.1 Write test: construct Transformer with GPUEngine  Est: 15m
  - [ ] S14.2.2 Run forward pass, compare with CPUEngine output  Est: 20m
  - [ ] S14.2.3 Run backward pass, compare gradients  Est: 20m
  - [ ] S14.2.4 Run golangci-lint  Est: 5m

- [ ] T14.3 End-to-end training loop test on GPU  Owner: TBD  Est: 60m
  - Dependencies: T14.2
  - Acceptance: Training loop (forward, loss, backward, optimizer step) runs on GPUEngine. Loss decreases over 10 steps.
  - [ ] S14.3.1 Write test: simple 2-layer network, MSE loss, SGD optimizer  Est: 20m
  - [ ] S14.3.2 Run 10 training steps on GPU, verify loss decreases  Est: 15m
  - [ ] S14.3.3 Compare final weights with CPU training run  Est: 15m
  - [ ] S14.3.4 Run golangci-lint  Est: 5m

- [ ] T14.4 Performance benchmarks  Owner: TBD  Est: 45m
  - Dependencies: E11, E12, E13
  - Acceptance: Benchmark results documented. MatMul 1024x1024 shows >= 10x speedup over CPU.
  - [ ] S14.4.1 Write benchmark: MatMul at sizes 128, 256, 512, 1024, 2048  Est: 15m
  - [ ] S14.4.2 Write benchmark: Softmax on [64, 128, 512] tensor  Est: 10m
  - [ ] S14.4.3 Write benchmark: full forward pass for small transformer  Est: 15m
  - [ ] S14.4.4 Run golangci-lint  Est: 5m

- [x] T14.5 Verify non-CUDA build still works  Completed: 2026 03 01
  - Dependencies: E10
  - Acceptance: `go test ./... -cover` passes without cuda build tag. `go build ./...` succeeds. No CUDA imports leak into non-tagged files.
  - [ ] S14.5.1 Run go test ./... without cuda tag  Est: 10m
  - [ ] S14.5.2 Run go build ./... without cuda tag  Est: 5m
  - [ ] S14.5.3 Verify no unconditional imports of internal/cuda or internal/cublas  Est: 10m
  - [ ] S14.5.4 Run golangci-lint  Est: 5m

---

## 4. Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M1 | Zero-coverage packages tested | E1 | COMPLETED 2026 02 24 |
| M2 | Sub-50% packages at target | E2 | COMPLETED 2026 02 24 |
| M3 | Medium-coverage packages at target | E3 | COMPLETED 2026 02 24 |
| M4 | All remaining packages at target | E4, E5 | COMPLETED 2026 02 25 |
| M5 | Final verification (Phase 1) | E7 | Full suite green, lint clean |
| M6 | Storage abstraction complete | E8 | TensorNumeric uses Storage[T], all tests pass, zero regressions |
| M7 | CUDA infrastructure ready | E9 | GPUStorage allocates/frees, H2D/D2H transfers verified |
| M8 | cuBLAS MatMul working | E10 | GPUEngine.MatMul parity test passes, all 34 methods stubbed |
| M9 | Native GPU kernels complete | E11, E12, E13 | All Engine methods have native GPU implementations (except UnaryOp) |
| M10 | GPU integration validated | E14 | End-to-end tests pass, benchmarks show >= 10x MatMul speedup |

### Recommended Sequence

1. **E8** -- Storage abstraction (critical path, blocks everything)
2. **E9** -- CUDA device + memory (can partially overlap with E8.3-E8.4)
3. **E10** -- cuBLAS MatMul + GPUEngine skeleton (first working GPU operation)
4. **E11** -- GPU elementwise ops (highest frequency operations)
5. **E12** -- GPU activations + Softmax (needed for transformer layers)
6. **E13** -- GPU reductions + tensor manipulation (completes the Engine)
7. **E14** -- Integration tests + benchmarks (validates everything end-to-end)
8. **E6, E7** -- Phase 1 remaining tasks (independent, low priority)

Within E11, E12, E13, tasks are independent of each other and can be done in any order or in parallel.

---

## 5. Operating Procedure

### Definition of Done

A task is done when:
1. Implementation matches the acceptance criteria for that task.
2. All existing tests pass (`go test ./... -count=1`).
3. New code has unit tests with >= 95% coverage for non-CUDA code.
4. GPU parity tests compare output with CPUEngine (tolerance: 1e-5 for reductions, 1e-6 for elementwise).
5. `golangci-lint run ./package/` reports 0 issues.
6. `go vet ./package/` reports no issues.
7. Tests pass with `-race` flag (CPU tests).
8. Non-CUDA build (`go build ./...` without cuda tag) still compiles and tests pass.
9. Changes are committed in a small, logical commit touching one directory only.

### Review and QA Steps

1. Before writing code, read the existing implementation in the relevant file(s).
2. Write tests first or alongside implementation. Use table-driven tests with the standard testing package.
3. After implementation, run `go test -cover ./package/` to verify coverage.
4. Run `golangci-lint run --fix ./package/` to fix lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. For GPU code: run parity test comparing GPU output with CPU baseline.
7. Run `go test ./... -count=1` to verify no regressions across the full suite.
8. Run `go build ./...` (without cuda tag) to verify non-CUDA build still works.

### Build Tag Strategy

All CUDA-dependent code must use the build tag: `//go:build cuda`

Files that need the cuda build tag:
- `tensor/gpu_storage.go`
- `tensor/transfer.go`
- `device/cuda_allocator.go`
- `device/cuda_device.go`
- `compute/gpu_engine.go`
- `internal/cuda/*.go`
- `internal/cublas/*.go`
- `internal/cuda/kernels/*.go`

Each of these files must also have a `_nocuda.go` stub if any exported types or functions are referenced from non-tagged code. If no cross-boundary references exist, no stub is needed.

### Commit Discipline

- Never commit files from different directories in the same commit. The pre-commit hook rejects it.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(tensor): add Storage[T] interface`, `feat(compute): implement GPUEngine MatMul`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run relevant linters and formatters before committing.

---

## 6. Progress Log

- **2026 03 01 (update 2):** Change Summary: Completed E11 (GPU Elementwise Ops), E12 (GPU Activation/Math Kernels), and partial E13 (T13.1 Sum/ReduceSum/ReduceMean). All 15 elementwise CUDA kernels (add, sub, mul, div, pow, add_scalar, mul_scalar, div_scalar, exp, log, sqrt, rsqrt, tanh, tanh_prime, fill) are wired into GPUEngine for float32 with CPU fallback for other types. Softmax kernel uses shared-memory reduction with numerical stability (max subtraction). SumAxis reduction kernel added for Sum/ReduceSum/ReduceMean. T14.1 (Linear layer integration test) and T14.5 (non-CUDA build verification) also completed. Remaining: T13.2-T13.5 (tensor manipulation), T14.2-T14.4 (integration tests/benchmarks).

- **2026 03 01:** Change Summary: Completed E8 (Tensor Storage Abstraction), E9 (CUDA Device and Memory Management), and E10 (cuBLAS MatMul). E8: Created Storage[T] interface and CPUStorage[T] (T8.1-T8.2), refactored TensorNumeric[T].data to use Storage[T] (T8.3), verified CPUEngine compatibility (T8.4). E9: Created internal/cuda CGO runtime bindings (T9.1), CUDA allocator and device registration (T9.2-T9.3), GPUStorage[T] (T9.4), ToGPU/ToCPU transfer functions (T9.5). E10: Created internal/cublas Sgemm bindings (T10.1), implemented GPUEngine[T] with cuBLAS MatMul and CPU fallback for all 33 remaining methods (T10.2-T10.4). All code behind //go:build cuda. All existing tests pass unchanged. Commits: b922d98, 32bf3a8, b3c54b3, f9c20a2, 615a08d, 54e3717, 3ed95b8, cc80304.

- **2026 02 28:** Change Summary: Added Phase 2 GPU Engine Implementation plan (E8-E14, T8.1-T14.5). New epics cover: E8 tensor storage abstraction (critical path), E9 CUDA device/memory management, E10 cuBLAS MatMul, E11 GPU elementwise ops, E12 GPU activations/math, E13 GPU reductions/tensor manipulation, E14 integration testing and benchmarks. Added milestones M6-M10. Updated Context section with architecture overview and GPU design rationale. Added build tag strategy to Operating Procedure. Preserved all existing Phase 1 tasks and status unchanged.

- **2026 02 25:** Change Summary: All epics E1-E5 completed. 30 of 33 testable packages now at >= 95% coverage. Three packages (layers/gather 93.1%, layers/embeddings 93.5%, layers/features 93.8%) remain below 95% due to unreachable tensor.New error paths. These gaps are documented as acceptable exceptions. Key commits this session: compute (85.5% -> 96.2%), data (93.5% -> 100%), layers/gather (91.7% -> 93.1%), layers/embeddings (92.5% -> 93.5%). Prior sessions raised all other packages to >= 95%.

- **2026 02 24:** Change Summary: Plan created. Defined 7 epics covering 35+ packages, prioritized by coverage tier from 0% to 93.5%. Excluded generated code (distributed/pb), main entrypoints (cmd/zerfoo*), trivial packages (types, pkg/prelude), and test utilities (best-effort only). Target: >= 95% statement coverage for all testable packages.

---

## 7. Hand-off Notes

### For a New Contributor

- **Phase 1 status:** Test coverage work is complete. See Documented Coverage Exceptions for the 3 packages below 95%.
- **Phase 2 status:** E8-E12 complete, E13 partially complete (T13.1 done). GPUEngine[T] has native GPU implementations for 20 of 34 Engine methods: MatMul (cuBLAS), Add, Sub, Mul, Div, Pow, AddScalar, MulScalar, DivScalar, Exp, Log, Sqrt, Rsqrt, Tanh, TanhPrime, Softmax, Sum, ReduceSum, ReduceMean, Fill. Remaining 14 methods use CPU fallback: UnaryOp, Transpose, Zero, Zeros, Copy, Gather, ScatterAdd, RandomUniform, Split, Concat, Repeat, OneHot, Reshape. Next: T13.2-T13.5 (tensor manipulation kernels).
- **Key files to understand first:**
  - `tensor/tensor.go` -- TensorNumeric[T] struct, the core data type
  - `compute/engine.go` -- Engine[T] interface (34 methods)
  - `compute/cpu_engine.go` -- CPUEngine[T] implementation (reference for GPUEngine)
  - `internal/xblas/gemm.go` -- existing BLAS wrappers (pattern for cuBLAS wrappers)
  - `device/device.go` -- Device interface and registry
  - `device/allocator.go` -- Allocator interface
- **How to run tests:** `go test ./... -cover` for full suite. For GPU tests: `go test -tags cuda ./... -cover`.
- **How to build with CUDA:** `go build -tags cuda ./...`
- **Key constraint:** The pre-commit hook runs `golangci-lint` and `go test ./...`. It rejects commits that touch multiple directories.
- **No credentials required.** All work is local. CUDA Toolkit must be installed for GPU development.
- **Design principle:** GPUEngine must be a drop-in replacement for CPUEngine. Layers pass tensors to the engine and never access data directly for computation. The Storage[T] interface is the only change to the tensor package API.

---

## 8. Appendix

### Packages Excluded from 95% Coverage Target

| Package | Reason |
|---------|--------|
| distributed/pb/ | Generated protobuf code (1518 LOC, all auto-generated) |
| cmd/zerfoo/ | Main entrypoint (33 LOC), no testable logic beyond CLI wiring |
| cmd/zerfoo-predict/ | Main entrypoint; testable logic is in cmd/cli/ |
| cmd/zerfoo-tokenize/ | Main entrypoint; testable logic is in pkg/tokenizer/ |
| pkg/prelude/ | 1 line, no statements to cover |
| types/ | 12 lines, type definitions only |
| testing/testutils/ | Test utility (997 LOC), exercised transitively, best-effort testing in E6 |
| tests/internal/testutil/ | Test utility (116 LOC), best-effort testing in E6 |
| tests/helpers/ | Test helper, no source files with statements |

### Engine[T] Interface Method Summary (34 methods)

| Category | Methods |
|----------|---------|
| Arithmetic | Ops |
| Unary | UnaryOp |
| Binary elementwise | Add, Sub, Mul, Div, Pow |
| Scalar | AddScalar, MulScalar, DivScalar |
| Matrix | MatMul, Transpose |
| Activations | Tanh, TanhPrime, Softmax, Exp, Log |
| Reductions | Sum, ReduceSum, ReduceMean |
| Math | Sqrt, Rsqrt |
| Tensor manipulation | Reshape, Split, Concat, Repeat |
| Initialization | Zero, Zeros, Fill, Copy, RandomUniform |
| Embedding | Gather, ScatterAdd |
| Encoding | OneHot |

### cuBLAS Row-Major Strategy

cuBLAS operates in column-major order. To compute C = A * B in row-major:
- Observe that for row-major matrices, A_row = A_col^T
- So C_row = A_row * B_row = (A_col^T) * (B_col^T)
- In column-major: C_col^T = (A_col^T) * (B_col^T)
- Equivalently: C_col = B_col * A_col
- Therefore: call cublasSgemm with B as first argument, A as second, swapping m and n

This avoids explicit transposition and works for any matrix dimensions.

### References

- Go coverage tool: `go help testflag`
- Coverage visualization: `go tool cover -html=cover.out`
- Race detector: `go test -race ./...`
- CUDA Toolkit documentation: developer.nvidia.com/cuda-toolkit
- cuBLAS documentation: docs.nvidia.com/cuda/cublas
- CGO documentation: pkg.go.dev/cmd/cgo
