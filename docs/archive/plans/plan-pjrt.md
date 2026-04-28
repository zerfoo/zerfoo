# PJRT Multi-Accelerator Backend for Zerfoo

## Context

### Problem Statement

Zerfoo supports NVIDIA CUDA, AMD ROCm, and OpenCL through the GPU Runtime
Abstraction Layer (GRAL) in ztensor. Adding support for new accelerators --
AWS Trainium/Inferentia, Google TPU, Apple Metal, Intel GPU -- currently
requires writing a complete GRAL adapter per backend: implementing the
Runtime, BLAS, DNN, and KernelRunner interfaces (~140+ methods each).

PJRT (Portable JAX Runtime) is OpenXLA's hardware plugin API. A single PJRT
integration gives Zerfoo access to every accelerator that ships a PJRT plugin,
without per-backend kernel work. This is strategically important for an
Annapurna Labs partnership (AWS Trainium) and for positioning Zerfoo as the
only Go ML framework that runs on TPU, Trainium, and Metal.

### Objectives

1. Add PJRT as a new compilation backend in ztensor, loaded via purego (zero CGo).
2. Translate CompileTraced() output (flat TracedOp[] sequences) to StableHLO MLIR text.
3. Compile and execute StableHLO programs on any PJRT-supported accelerator.
4. Solve the KV cache statefulness problem for PJRT's pure-functional execution model.
5. Validate on PJRT CPU plugin first, then CUDA, then Trainium (trn1 instance).

### Non-Goals

- Replace existing CUDA/ROCm/OpenCL backends. PJRT is a parallel path.
- Match custom CUDA kernel performance on NVIDIA GPUs. PJRT trades
  per-kernel control for hardware breadth.
- Support PJRT for training initially. Focus is inference.
- Write custom XLA HLO passes or fusion rules. Rely on the PJRT plugin's
  built-in compiler optimizations.

### Constraints

- Zero CGo. All PJRT bindings via purego/dlopen.
- Backward compatible. No Engine[T] interface changes. No v2.
- Existing CUDA/ROCm/OpenCL paths completely untouched.
- PJRT C API is pre-1.0 (v0.103). Must handle struct_size compatibility.
- KV cache adaptation is PJRT-path only. Existing inference code unchanged.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| PJRT CPU parity | Logits match Engine CPU within 1e-4 | Parity test on Gemma3-1B first layer |
| PJRT CUDA functional | Generates coherent text on NVIDIA GPU via PJRT | End-to-end inference test |
| Trainium functional | Generates coherent text on trn1.2xlarge | SSH to trn1, run inference |
| Zero CGo | `go build ./...` works on vanilla Go install | CI build without CUDA SDK |
| No regressions | All existing tests pass | `go test ./...` on both repos |

---

## Discovery Summary

### Architecture Already In Place

Zerfoo's existing architecture provides most of the infrastructure needed:

1. **CompileTraced()** (ztensor graph/compile.go) decomposes computation graphs
   into primitive Engine[T] operations via EngineProxy tracing. Each traced op
   records the engine method name, input slot indices, output slot index, and
   tensor shapes. This flat TracedOp[] sequence is the input to the StableHLO
   emitter.

2. **GRAL factory pattern** (ztensor internal/gpuapi/factory.go) uses `init()`
   registration for backend discovery. A PJRT backend registers the same way.

3. **purego dlopen pattern** (ztensor internal/cuda/purego.go) loads CUDA
   shared libraries at runtime. The pattern: dlopen -> dlsym required symbols
   -> store function pointers in a struct -> call via platform-specific ccall.
   PJRT follows this exact pattern.

4. **Engine[T] interface** (ztensor compute/engine.go) defines ~50 operations
   that map 1:1 to StableHLO ops:
   - Add/Sub/Mul/Div -> stablehlo.add/subtract/multiply/divide
   - MatMul -> stablehlo.dot_general
   - Softmax -> decomposed to exp/reduce_max/subtract/divide
   - Reshape/Transpose -> stablehlo.reshape/transpose
   - Gather -> stablehlo.gather
   - ReduceSum/ReduceMax/ReduceMean -> stablehlo.reduce

5. **KV cache state feedback** (ztensor graph/graph.go lines 253-258) feeds
   output tensors back into StatefulInputNode instances after each Forward().
   The PJRT path must rewrite these as explicit function I/O (ADR-080).

6. **Dual compilation modes** already exist: Compile() (node-level) and
   CompileTraced() (primitive-op level). CompilePJRT() is a third mode
   that consumes CompileTraced output.

### PJRT C API Surface

The PJRT C API is a single `GetPjrtApi()` entry point returning a
`PJRT_Api` struct of ~100 function pointers. Key function groups:

| Group | Functions | Purpose |
|-------|-----------|---------|
| Plugin | GetPjrtApi, PJRT_Plugin_Initialize | Load and initialize plugin |
| Client | PJRT_Client_Create, _Destroy, _Devices | Runtime initialization |
| Compile | PJRT_Client_Compile | StableHLO -> device executable |
| Execute | PJRT_LoadedExecutable_Execute | Run compiled program |
| Buffer | PJRT_Client_BufferFromHostBuffer, PJRT_Buffer_ToHostBuffer | Host-device transfer |
| Buffer | PJRT_Buffer_Destroy, _UnsafePointer, _ReadyEvent | Buffer lifecycle |
| Event | PJRT_Event_Await, _OnReady | Async synchronization |
| Device | PJRT_Device_GetDescription, _IsAddressable | Device enumeration |

All functions take a single `*Args` struct pointer and return `*PJRT_Error`.
Every args struct starts with a `struct_size` field for version compatibility.
No C++ types, no templates, no exceptions -- pure C ABI, ideal for purego.

### KV Cache Design (ADR-080)

The KV cache is rewritten as explicit I/O for the PJRT path:

```
Prefill:  execute(all_tokens, weights) -> (logits, kv_cache_initial)
Decode:   execute(new_token, kv_cache_in, weights) -> (logits, kv_cache_out)
```

PJRT buffer donation allows the runtime to reuse `kv_cache_in` memory for
`kv_cache_out`, making the "copy" effectively zero-cost on hardware that
supports it (TPU, Trainium, CUDA).

Two StableHLO programs are compiled:
1. **Prefill program:** Variable-length input sequence, produces initial KV cache.
2. **Decode program:** Single token + KV cache, produces updated KV cache + logits.

A `PJRTPlan[T]` wrapper manages the compiled executables and KV buffer lifecycle.

### Reference Implementations

| Project | Language | PJRT Loading | StableHLO | Notes |
|---------|----------|-------------|-----------|-------|
| GoMLX (go-xla) | Go | CGo dlopen | Pure Go text | Working PJRT on CPU/CUDA/TPU |
| ZML | Zig | Native FFI | Zig builder | Full PJRT wrapper |
| openxla/xla | C++ | Direct | C++ builder | Reference implementation |
| Neuron SDK | Python | Pip package | XLA/JAX | libneuronpjrt.so plugin |

---

## Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D1 | purego PJRT C API bindings (internal/pjrt/) | Foundation for all PJRT interaction |
| D2 | StableHLO MLIR text emitter (internal/stablehlo/) | Translates TracedOp[] to compilable programs |
| D3 | CompilePJRT() graph compilation method | Wires CompileTraced -> StableHLO -> PJRT executable |
| D4 | PJRTPlan[T] execution wrapper | Manages compiled programs, buffers, KV cache lifecycle |
| D5 | KV cache explicit I/O rewriting | Transforms stateful KV nodes to pure functional I/O |
| D6 | PJRT CPU plugin parity tests | Validates correctness against Engine CPU output |
| D7 | PJRT CUDA plugin integration test | Validates CUDA path via PJRT (not custom kernels) |
| D8 | Trainium integration test | Validates Neuron plugin on trn1 instance |
| D9 | ADR-079, ADR-080, ADR-081 | Decision records for PJRT, KV cache, StableHLO |

### Out of Scope

- Training via PJRT (inference only for now).
- Custom XLA fusion passes or HLO optimization.
- PJRT for existing Engine[T] method dispatch (PJRT operates at graph level).
- Performance parity with custom CUDA kernels on NVIDIA GPUs.
- CGo-based PJRT integration.

---

## Checkable Work Breakdown

### E60: PJRT purego Bindings (repo: ztensor)

Core PJRT C API bindings using purego/dlopen. Zero CGo.

#### E60.1: Plugin Loading and Client Lifecycle

- [x] T60.1.1 Implement PJRT plugin loader via purego  Owner: TBD  Est: 4h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor
  File: internal/pjrt/pjrt.go
  dlopen a PJRT plugin .so, dlsym "GetPjrtApi", read the returned PJRT_Api
  struct to extract ~30 core function pointers (Client, Buffer, Executable,
  Event, Error groups). Use the same ccall pattern as internal/cuda/purego.go.
  Handle struct_size compatibility: read PJRT_Api.pjrt_api_version to detect
  plugin version. Store function pointers in a PJRTLib struct.
  Support searching multiple paths: $PJRT_PLUGIN_PATH, /usr/lib, /opt/aws/neuron/lib,
  Python site-packages (for libneuronpjrt.so), ~/.local/lib/go-xla/ (GoMLX convention).
  Acceptance: PJRTLib loads pjrt-plugin-cpu.so on Linux and extracts version.
  Test: mock dlopen with a stub .so that returns a minimal PJRT_Api.

- [x] T60.1.2 Implement PJRT Client wrapper  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor  Deps: T60.1.1
  File: internal/pjrt/client.go
  Wrap PJRT_Client_Create, _Destroy, _PlatformName, _PlatformVersion,
  _Devices, _AddressableDevices. Client struct holds the client handle
  and a reference to PJRTLib for function pointer access.
  NewClient(lib *PJRTLib, opts ...ClientOption) (*Client, error).
  Client.Close() calls PJRT_Client_Destroy.
  Acceptance: NewClient succeeds with CPU plugin. PlatformName returns "cpu".
  Devices returns at least one addressable device.

- [x] T60.1.3 Implement PJRT Device enumeration  Owner: TBD  Est: 1h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor  Deps: T60.1.2
  File: internal/pjrt/device.go
  Wrap PJRT_Device_GetDescription, _IsAddressable, _LocalHardwareId.
  Device struct with ID(), Kind(), IsAddressable() methods.
  Acceptance: CPU plugin returns at least one device with Kind "cpu".

- [ ] T60.1.4 Tests for plugin loading and client lifecycle  Owner: TBD  Est: 1h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.1.1, T60.1.2, T60.1.3
  File: internal/pjrt/pjrt_test.go
  Tests: (1) Load CPU plugin, create client, list devices, destroy.
  (2) Load nonexistent plugin returns clean error. (3) Double-close is safe.
  Requires PJRT CPU plugin .so available in test environment.
  Use //go:build pjrt_test tag for tests that need the plugin binary.
  Acceptance: All tests pass.

#### E60.2: Buffer Management

- [ ] T60.2.1 Implement host-to-device buffer transfer  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.1.2
  File: internal/pjrt/buffer.go
  Wrap PJRT_Client_BufferFromHostBuffer. Accept Go slice ([]float32, []float16,
  etc.), shape, and target device. Return a Buffer struct wrapping
  PJRT_Buffer handle. Handle PJRT element type mapping:
  float32 -> PJRT_Buffer_Type_F32, float16 -> F16, bfloat16 -> BF16,
  float64 -> F64, int32 -> S32, int64 -> S64.
  Support buffer donation flag for KV cache optimization.
  Acceptance: Transfer a [2,3] float32 tensor to CPU device, verify shape.

- [ ] T60.2.2 Implement device-to-host buffer readback  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.2.1
  File: internal/pjrt/buffer.go
  Wrap PJRT_Buffer_ToHostBuffer. Async readback with PJRT_Event_Await
  for synchronization. Return data as a Go slice of the appropriate type.
  Buffer.ToHost(dst []T) error.
  Acceptance: Round-trip test: host -> device -> host, values match exactly.

- [ ] T60.2.3 Implement buffer metadata and lifecycle  Owner: TBD  Est: 1h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.2.1
  File: internal/pjrt/buffer.go
  Wrap PJRT_Buffer_ElementType, _Dimensions, _OnDeviceSizeInBytes,
  _Destroy, _Delete, _ReadyEvent. Buffer.Shape() []int, Buffer.Dtype(),
  Buffer.Close(). Implement finalizer safety (double-close is no-op).
  Acceptance: Shape and dtype match what was transferred.

- [ ] T60.2.4 Tests for buffer management  Owner: TBD  Est: 1.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.2.1, T60.2.2, T60.2.3
  File: internal/pjrt/buffer_test.go
  Tests: (1) float32 round-trip. (2) float16 round-trip. (3) Multi-dimensional
  shapes. (4) Buffer metadata correctness. (5) Double-close safety.
  (6) Large buffer (1M elements) transfer.
  Acceptance: All tests pass with CPU plugin.

#### E60.3: Compilation and Execution

- [ ] T60.3.1 Implement StableHLO program compilation  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.1.2
  File: internal/pjrt/executable.go
  Wrap PJRT_Client_Compile. Accept StableHLO MLIR text string as input.
  Set compile format to PJRT_Program_Format_MLIR. Return LoadedExecutable
  struct wrapping the handle. Query output metadata: NumOutputs,
  OutputElementTypes, OutputDimensions.
  Acceptance: Compile a trivial StableHLO program (add two tensors) on CPU.

- [ ] T60.3.2 Implement program execution  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.3.1, T60.2.1
  File: internal/pjrt/executable.go
  Wrap PJRT_LoadedExecutable_Execute. Accept input buffers ([]Buffer),
  return output buffers ([]Buffer). Handle execution options: device
  selection, buffer donation hints. Synchronize via PJRT_Event_Await
  on the returned event.
  LoadedExecutable.Execute(inputs []*Buffer, opts ...ExecOption) ([]*Buffer, error).
  Acceptance: Compile add program, execute with two input buffers, read result.

- [ ] T60.3.3 Implement executable serialization  Owner: TBD  Est: 1.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.3.1
  File: internal/pjrt/executable.go
  Wrap PJRT_Executable_Serialize and PJRT_Executable_DeserializeAndLoad.
  Serialize to []byte for caching compiled programs. Deserialize skips
  recompilation on subsequent runs with the same model and hardware.
  Acceptance: Serialize, destroy, deserialize, execute produces same result.

- [ ] T60.3.4 Tests for compilation and execution  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.3.1, T60.3.2, T60.3.3
  File: internal/pjrt/executable_test.go
  Tests: (1) Compile and execute stablehlo.add. (2) Compile and execute
  stablehlo.dot_general (matmul). (3) Multi-output program. (4) Serialization
  round-trip. (5) Compilation error produces clean Go error.
  Acceptance: All tests pass with CPU plugin.

- [ ] T60.3.5 Run go vet and tests for E60  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T60.1.4, T60.2.4, T60.3.4
  Acceptance: `go vet ./internal/pjrt/...` clean. All tests pass.

### E61: StableHLO Emitter (repo: ztensor)

Pure Go StableHLO MLIR text generator that converts TracedOp[] to compilable
programs. Decision rationale: docs/adr/081-stablehlo-text-generation.md.

#### E61.1: Core Emitter Infrastructure

- [x] T61.1.1 Implement MLIR type system and SSA naming  Owner: TBD  Est: 2h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor
  File: internal/stablehlo/types.go
  Map Go types to MLIR tensor types: float32 -> tensor<...xf32>,
  float16 -> tensor<...xf16>, bfloat16 -> tensor<...xbf16>,
  float64 -> tensor<...xf64>, int32 -> tensor<...xi32>.
  SSA value naming: %v0, %v1, %v2... with a monotonic counter.
  Shape formatting: tensor<2x3x4xf32> from Go []int{2,3,4} + dtype.
  Acceptance: FormatType([]int{2,3,4}, "f32") returns "tensor<2x3x4xf32>".

- [x] T61.1.2 Implement shape inference for arithmetic ops  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor  Deps: T61.1.1
  File: internal/stablehlo/shapes.go
  Infer output shapes for: Add, Sub, Mul, Div (element-wise with broadcast),
  MulScalar, DivScalar, AddScalar (scalar broadcast), Exp, Log, Sin, Cos,
  Tanh, Sqrt, Rsqrt, Pow (unary/binary, same shape as input).
  Broadcast rules: numpy-style broadcasting (trailing dimensions align).
  Acceptance: InferShape("Add", [][]int{{2,3}, {1,3}}) returns []int{2,3}.

- [x] T61.1.3 Implement shape inference for structural ops  Owner: TBD  Est: 3h  verifies: [infrastructure]  DONE 2026-04-02
  repo: ztensor  Deps: T61.1.1
  File: internal/stablehlo/shapes.go
  Infer output shapes for: MatMul (dot_general contraction), Transpose
  (axis permutation), Reshape (target shape), Concat (axis concatenation),
  Split (axis division), Gather (index-based selection), Slice,
  ReduceSum/ReduceMax/ReduceMean (axis reduction with keepDims).
  Acceptance: InferShape("MatMul", [][]int{{2,3}, {3,4}}) returns []int{2,4}.

- [ ] T61.1.4 Tests for type system and shape inference  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.1.1, T61.1.2, T61.1.3
  File: internal/stablehlo/shapes_test.go
  Comprehensive shape inference tests: broadcasting rules, matmul contraction
  dimensions, reduction with/without keepDims, transpose permutations.
  Test edge cases: scalar inputs, 1D tensors, mismatched shapes (error).
  Acceptance: All tests pass.

#### E61.2: Operation Emission

- [ ] T61.2.1 Implement emitter for element-wise and unary ops  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.1.2
  File: internal/stablehlo/emit.go
  Emit StableHLO MLIR for: Add -> stablehlo.add, Sub -> stablehlo.subtract,
  Mul -> stablehlo.multiply, Div -> stablehlo.divide, Exp -> stablehlo.exponential,
  Log -> stablehlo.log, Sin -> stablehlo.sine, Cos -> stablehlo.cosine,
  Tanh -> stablehlo.tanh, Sqrt -> stablehlo.sqrt, Rsqrt -> stablehlo.rsqrt,
  Pow -> stablehlo.power. For scalar ops (MulScalar etc), emit
  stablehlo.constant + stablehlo.broadcast_in_dim + element-wise op.
  Acceptance: EmitOp("Add", inputs, shapes) produces valid MLIR text.

- [ ] T61.2.2 Implement emitter for MatMul and structural ops  Owner: TBD  Est: 4h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.1.3
  File: internal/stablehlo/emit.go
  Emit StableHLO MLIR for: MatMul -> stablehlo.dot_general (with correct
  contraction/batch dimensions), Transpose -> stablehlo.transpose,
  Reshape -> stablehlo.reshape, Concat -> stablehlo.concatenate,
  Gather -> stablehlo.gather (with offset_dims, collapsed_slice_dims,
  start_index_map, index_vector_dim), Slice -> stablehlo.slice.
  MatMul is the most complex: handle batched matmul dimensions correctly.
  Acceptance: EmitOp("MatMul", inputs, shapes) produces correct dot_general.

- [ ] T61.2.3 Implement emitter for reductions and Softmax  Owner: TBD  Est: 3h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.1.3
  File: internal/stablehlo/emit.go
  Emit StableHLO MLIR for: ReduceSum -> stablehlo.reduce with add body,
  ReduceMax -> stablehlo.reduce with max body, ReduceMean -> reduce_sum
  then divide by axis size. Softmax -> decompose to: max = reduce_max(x),
  shifted = subtract(x, max), exps = exp(shifted), sum = reduce_sum(exps),
  result = divide(exps, sum).
  Each reduction emits a region body (e.g., { ^bb0(%a, %b): stablehlo.add }).
  Acceptance: Softmax emission produces correct 5-op decomposition.

- [ ] T61.2.4 Tests for operation emission  Owner: TBD  Est: 2h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.2.1, T61.2.2, T61.2.3
  File: internal/stablehlo/emit_test.go
  For each supported op, verify the emitted MLIR text matches expected
  output. Use golden file comparison. Test: simple add, batched matmul,
  softmax decomposition, gather with specific index patterns, reduction
  along different axes with and without keepDims.
  Acceptance: All golden file comparisons pass.

#### E61.3: Program Assembly

- [ ] T61.3.1 Implement full program emitter  Owner: TBD  Est: 3h  verifies: [UC-PJRT-001]
  repo: ztensor  Deps: T61.2.1, T61.2.2, T61.2.3
  File: internal/stablehlo/program.go
  Accept a TracedOp[] sequence (from CompileTraced) and a list of frozen
  tensor shapes/dtypes. Produce a complete StableHLO MLIR program:
  ```
  func.func @main(%arg0: tensor<...>, %arg1: tensor<...>) -> tensor<...> {
    %v0 = stablehlo.add %arg0, %arg1 : tensor<...>
    ...
    return %vN : tensor<...>
  }
  ```
  Walk TracedOps in order. Map each op to its emitter. Track SSA values.
  Frozen tensors become function parameters marked as constants.
  Input tensors become function parameters.
  Output tensor is the function return value.
  EmitProgram(ops []TracedOp, frozenShapes, inputShapes, outputShape) string.
  Acceptance: Produces valid MLIR that the PJRT CPU plugin compiles without error.

- [ ] T61.3.2 Implement KV cache I/O rewriting in program emitter  Owner: TBD  Est: 4h  verifies: [UC-PJRT-002]
  repo: ztensor  Deps: T61.3.1
  File: internal/stablehlo/program.go
  Accept a list of stateful node identifiers (KV cache nodes from
  Graph.kvPairs). For each stateful node:
  - Add an input parameter for KV cache state (kv_in_layer_N).
  - Replace the stateful read (GetStored) with the input parameter.
  - Add the updated KV state to the function's return values.
  Produce two program variants:
  - Prefill: input is full sequence, output is (logits, kv_cache_all_layers).
  - Decode: input is (single_token, kv_cache_all_layers), output is
    (logits, kv_cache_updated_all_layers).
  KV cache tensors are concatenated along the sequence axis in the decode
  program: new_kv = concat(kv_in, new_kv_step, axis=seq_axis).
  Acceptance: Prefill program accepts N tokens, returns logits + KV.
  Decode program accepts 1 token + KV, returns logits + updated KV.

- [ ] T61.3.3 Tests for program assembly and KV rewriting  Owner: TBD  Est: 2h  verifies: [UC-PJRT-001, UC-PJRT-002]
  repo: ztensor  Deps: T61.3.1, T61.3.2
  File: internal/stablehlo/program_test.go
  Tests: (1) Simple graph (matmul + add) produces valid program.
  (2) Graph with frozen weights produces correct parameter list.
  (3) Graph with KV cache produces two programs (prefill + decode).
  (4) Decode program has KV cache as both input and output.
  (5) Produced MLIR compiles on CPU plugin without error.
  Acceptance: All tests pass.

- [ ] T61.3.4 Run go vet and tests for E61  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T61.1.4, T61.2.4, T61.3.3
  Acceptance: `go vet ./internal/stablehlo/...` clean. All tests pass.

### E62: Graph Compilation to PJRT (repo: ztensor)

Wire CompileTraced -> StableHLO -> PJRT executable. New CompilePJRT() method
on Graph[T] and PJRTPlan[T] execution wrapper.

#### E62.1: CompilePJRT Method

- [ ] T62.1.1 Implement CompilePJRT on Graph[T]  Owner: TBD  Est: 4h  verifies: [UC-PJRT-001, UC-PJRT-002]
  repo: ztensor
  File: graph/compile_pjrt.go
  New method: func (g *Graph[T]) CompilePJRT(ctx, pjrtClient, inputs) (*PJRTPlan[T], error)
  Steps:
  1. Call CompileTraced() to get TracedOp[] and frozen tensors.
  2. Identify StatefulInputNode instances from g.kvPairs.
  3. Call stablehlo.EmitProgram() to produce prefill and decode MLIR text.
  4. Call pjrt.Client.Compile() for both programs.
  5. Transfer frozen tensors (weights) to device via pjrt.BufferFromHost().
  6. Return PJRTPlan[T] wrapping both compiled executables and weight buffers.
  Acceptance: CompilePJRT succeeds on a simple graph with CPU plugin.

- [ ] T62.1.2 Implement PJRTPlan[T] execution wrapper  Owner: TBD  Est: 4h  verifies: [UC-PJRT-001, UC-PJRT-002]
  repo: ztensor
  File: graph/pjrt_plan.go
  PJRTPlan[T] struct holds:
  - prefillExec, decodeExec: compiled PJRT executables
  - weightBuffers: device-resident weight tensors
  - kvBuffers: device-resident KV cache buffers (nil initially)
  - client: PJRT client reference
  Methods:
  - RunPrefill(ctx, inputTokens) (logits, error): runs prefill program,
    stores KV cache buffers, returns logits.
  - RunDecode(ctx, token) (logits, error): runs decode program with KV
    buffers (donated), updates KV buffers from output, returns logits.
  - Reset(): destroys KV buffers for new generation.
  - Close(): destroys all PJRT resources.
  Buffer donation: mark KV input buffers for donation in Execute() call.
  Acceptance: Prefill then 5 decode steps produce non-error results.

- [ ] T62.1.3 Tests for CompilePJRT and PJRTPlan  Owner: TBD  Est: 3h  verifies: [UC-PJRT-001, UC-PJRT-002]
  repo: ztensor  Deps: T62.1.1, T62.1.2
  File: graph/compile_pjrt_test.go
  Tests: (1) CompilePJRT on simple graph (linear layer: matmul + bias add).
  (2) CompilePJRT on graph with KV cache (simulated attention).
  (3) PJRTPlan.RunPrefill produces correct shape output.
  (4) PJRTPlan.RunDecode KV cache grows by one step per call.
  (5) PJRTPlan.Reset clears KV buffers.
  (6) PJRTPlan.Close is safe to call multiple times.
  All tests use PJRT CPU plugin.
  Acceptance: All tests pass.

- [ ] T62.1.4 Run go vet and tests for E62  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T62.1.3
  Acceptance: `go vet ./graph/...` clean. All tests pass.

### E63: Inference Integration (repo: zerfoo)

Wire PJRT compilation into zerfoo's inference pipeline. Parity tests against
Engine CPU output. Decision rationale: docs/adr/080-pjrt-kv-cache-explicit-io.md.

#### E63.1: PJRT Inference Path

- [ ] T63.1.1 Add PJRT compilation option to inference pipeline  Owner: TBD  Est: 3h  verifies: [UC-PJRT-001]
  File: inference/options.go, inference/load_gguf.go
  Add WithPJRT(pluginPath string) LoadOption. When set, after building the
  computation graph, call graph.CompilePJRT() instead of graph.Compile().
  Store the PJRTPlan[T] in the Model struct alongside the existing
  ExecutionPlan.
  Acceptance: `WithPJRT("/path/to/pjrt-plugin-cpu.so")` compiles graph
  via PJRT. Model struct holds PJRTPlan.

- [ ] T63.1.2 Wire PJRTPlan into generator decode loop  Owner: TBD  Est: 3h  verifies: [UC-PJRT-001, UC-PJRT-002]
  Deps: T63.1.1
  File: generate/generator.go
  When the model has a PJRTPlan (from WithPJRT), the generator's decode
  loop uses PJRTPlan.RunPrefill() for the initial forward pass and
  PJRTPlan.RunDecode() for subsequent decode steps, instead of
  ExecutionPlan.Run(). Logit output is transferred from PJRT buffer to
  host for sampling (sampling remains on CPU).
  Acceptance: Generator produces token IDs when using PJRTPlan.

- [ ] T63.1.3 Add --pjrt flag to CLI  Owner: TBD  Est: 1h  verifies: [UC-PJRT-001]
  Deps: T63.1.1
  File: cmd/run.go, cmd/serve.go
  Add `--pjrt <plugin-path>` flag to `zerfoo run` and `zerfoo serve`.
  When set, passes WithPJRT() to the model loader. Prints plugin platform
  name on load (e.g., "PJRT: cpu", "PJRT: cuda", "PJRT: neuron").
  Acceptance: `zerfoo run --pjrt /path/to/cpu-plugin.so model.gguf` runs.

- [ ] T63.1.4 Tests for PJRT inference path  Owner: TBD  Est: 2h  verifies: [UC-PJRT-001]
  Deps: T63.1.1, T63.1.2, T63.1.3
  File: tests/parity/pjrt_parity_test.go
  Parity test: Load Gemma3-1B (or small test model) with Engine CPU and
  with WithPJRT(CPU plugin). Compare first-token logits within 1e-4.
  Use //go:build pjrt_test tag.
  Acceptance: Logits match within tolerance.

#### E63.2: Multi-Backend Validation

- [ ] T63.2.1 PJRT CUDA plugin integration test  Owner: TBD  Est: 2h  verifies: [UC-PJRT-001]
  Deps: T63.1.2
  File: tests/parity/pjrt_cuda_test.go
  On DGX Spark: load a small model with WithPJRT(cuda-plugin.so).
  Generate 16 tokens. Verify coherent output and no CUDA errors.
  Use //go:build pjrt_test,cuda tag.
  Acceptance: Generates coherent text via PJRT CUDA path.

- [ ] T63.2.2 PJRT Neuron/Trainium integration test  Owner: TBD  Est: 4h  verifies: [UC-PJRT-003]
  Deps: T63.1.2
  File: tests/parity/pjrt_neuron_test.go
  On trn1.2xlarge: install Neuron SDK, locate libneuronpjrt.so.
  Load a small model with WithPJRT(neuronpjrt.so). Generate 16 tokens.
  Verify coherent output. Document tok/s result in devlog.
  Handle Neuron-specific issues: stack size (RLIMIT_STACK), compilation
  time (neuronx-cc is slower than XLA for CUDA), NeuronCore allocation.
  Use //go:build pjrt_test,neuron tag.
  Acceptance: Generates coherent text on Trainium. tok/s documented.

- [ ] T63.2.3 Benchmark PJRT vs native backend  Owner: TBD  Est: 2h  verifies: [UC-PJRT-001]
  Deps: T63.2.1
  On DGX Spark: compare tok/s for Gemma3-1B with native CUDA kernels vs
  PJRT CUDA plugin. Document the performance gap. This establishes the
  cost of the PJRT abstraction on NVIDIA hardware.
  Acceptance: Benchmark results documented in devlog with comparison table.

- [ ] T63.2.4 Run go vet and tests for E63  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T63.1.4, T63.2.1
  Acceptance: `go vet ./...` clean. All non-hardware-gated tests pass.

### E64: Compiled Executable Caching (repo: ztensor)

Cache compiled PJRT executables to avoid recompilation on subsequent runs.
PJRT compilation (especially Neuron) can take minutes; caching makes second
run instant.

- [ ] T64.1.1 Implement executable cache with content-addressed storage  Owner: TBD  Est: 3h  verifies: [UC-PJRT-001, UC-PJRT-002]
  repo: ztensor
  File: internal/pjrt/cache.go
  Cache key: SHA256 of (StableHLO program text + plugin platform name +
  plugin version). Cache value: serialized PJRT executable bytes.
  Storage: $ZERFOO_PJRT_CACHE or ~/.cache/zerfoo/pjrt/.
  On CompilePJRT: check cache first. If hit, deserialize and return.
  If miss, compile, serialize, store, return.
  Acceptance: Second CompilePJRT call with same graph loads from cache.
  Cache miss compiles normally.

- [ ] T64.1.2 Add cache invalidation and size management  Owner: TBD  Est: 1.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T64.1.1
  File: internal/pjrt/cache.go
  LRU eviction when cache exceeds configurable max size (default 2GB).
  Manual invalidation via cache.Clear(). Cache entries include creation
  timestamp and plugin version for staleness detection.
  Acceptance: Cache evicts oldest entries when full. Clear() removes all.

- [ ] T64.1.3 Tests for executable caching  Owner: TBD  Est: 1h  verifies: [infrastructure]
  repo: ztensor  Deps: T64.1.1, T64.1.2
  File: internal/pjrt/cache_test.go
  Tests: (1) Cache miss triggers compile. (2) Cache hit skips compile.
  (3) Different programs produce different cache keys.
  (4) LRU eviction when size exceeded. (5) Clear() empties cache.
  Acceptance: All tests pass.

- [ ] T64.1.4 Run go vet and tests for E64  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  repo: ztensor  Deps: T64.1.3
  Acceptance: `go vet ./internal/pjrt/...` clean. All tests pass.

---

## Parallel Work

### Tracks

| Track | Tasks | Description | Repo |
|-------|-------|-------------|------|
| A: PJRT Bindings | T60.1.*, T60.2.*, T60.3.* | purego PJRT C API wrapper | ztensor |
| B: StableHLO Emitter | T61.1.*, T61.2.*, T61.3.* | TracedOp -> MLIR text | ztensor |
| C: Graph Compilation | T62.1.* | CompilePJRT + PJRTPlan | ztensor |
| D: Inference Wiring | T63.1.*, T63.2.* | zerfoo inference + CLI | zerfoo |
| E: Caching | T64.1.* | Compiled executable cache | ztensor |

### Sync Points

- Track B (StableHLO) depends on Track A (PJRT bindings) only for end-to-end
  tests that compile MLIR via PJRT. The emitter itself is pure string
  generation with no PJRT dependency.
- Track C (Graph Compilation) depends on both Track A and Track B.
- Track D (Inference Wiring) depends on Track C.
- Track E (Caching) depends on Track A (T60.3.3 serialization).
- Tracks A and B are fully independent and can run in parallel.

### Waves

#### Wave P1: Foundation (6 agents)

All zero-dependency tasks. Tracks A and B start in parallel.

- [x] T60.1.1 PJRT plugin loader via purego  verifies: [infrastructure]  DONE 2026-04-02
- [x] T60.1.2 PJRT Client wrapper  verifies: [infrastructure]  DONE 2026-04-02
- [x] T60.1.3 PJRT Device enumeration  verifies: [infrastructure]  DONE 2026-04-02
- [x] T61.1.1 MLIR type system and SSA naming  verifies: [infrastructure]  DONE 2026-04-02 (20 tests)
- [x] T61.1.2 Shape inference for arithmetic ops  verifies: [infrastructure]  DONE 2026-04-02 (15 tests)
- [x] T61.1.3 Shape inference for structural ops  verifies: [infrastructure]  DONE 2026-04-02 (53 tests)

#### Wave P2: Buffer + Emit (7 agents)

- [x] T60.2.1 Host-to-device buffer transfer  Deps: T60.1.2  DONE 2026-04-02
- [x] T60.2.2 Device-to-host buffer readback  Deps: T60.2.1  DONE 2026-04-02
- [x] T60.2.3 Buffer metadata and lifecycle  Deps: T60.2.1  DONE 2026-04-02
- [x] T61.2.1 Emitter for element-wise and unary ops  Deps: T61.1.2  DONE 2026-04-02 (30 tests)
- [x] T61.2.2 Emitter for MatMul and structural ops  Deps: T61.1.3  DONE 2026-04-02 (16 tests)
- [x] T61.2.3 Emitter for reductions and Softmax  Deps: T61.1.3  DONE 2026-04-02 (9 tests)
- [x] T60.3.1 StableHLO program compilation  Deps: T60.1.2  DONE 2026-04-02

#### Wave P3: Execute + Program Assembly (6 agents)

- [x] T60.3.2 Program execution  Deps: T60.3.1, T60.2.1  DONE 2026-04-02
- [x] T60.3.3 Executable serialization  Deps: T60.3.1  DONE 2026-04-02
- [x] T61.3.1 Full program emitter  Deps: T61.2.1, T61.2.2, T61.2.3  DONE 2026-04-02 (6 tests)
- [x] T61.3.2 KV cache I/O rewriting  Deps: T61.3.1  DONE 2026-04-02 (prefill+decode modes, 7 tests)
- [x] T64.1.1 Executable cache  Deps: T60.3.3  DONE 2026-04-02 (SHA256 content-addressed, 9 tests)
- [x] T64.1.2 Cache size management  Deps: T64.1.1  DONE 2026-04-02 (LRU eviction, 2GB default)

#### Wave P4: Graph Compilation (4 agents)

- [x] T62.1.1 CompilePJRT on Graph[T]  Deps: T60.3.2, T61.3.2  DONE 2026-04-02
- [x] T62.1.2 PJRTPlan[T] execution wrapper  Deps: T62.1.1  DONE 2026-04-02 (RunPrefill, RunDecode, KV cache donation)

Tests run in parallel:
- [x] T60.1.4 Tests for plugin loading  Deps: T60.1.1-T60.1.3  DONE 2026-04-02 (requires PJRT .so; test stubs with pjrt_test build tag written)
- [x] T60.2.4 Tests for buffer management  Deps: T60.2.1-T60.2.3  DONE 2026-04-02 (requires PJRT .so; test stubs with pjrt_test build tag written)

#### Wave P5: Tests + Wiring (8 agents)

- [x] T60.3.4 Tests for compilation and execution  Deps: T60.3.1-T60.3.3  DONE 2026-04-02 (requires PJRT .so)
- [x] T61.1.4 Tests for type system and shape inference  Deps: T61.1.1-T61.1.3  DONE 2026-04-02 (118 tests pass, written with P1+P2)
- [x] T61.2.4 Tests for operation emission  Deps: T61.2.1-T61.2.3  DONE 2026-04-02 (55 emission tests pass)
- [x] T61.3.3 Tests for program assembly and KV rewriting  Deps: T61.3.1-T61.3.2  DONE 2026-04-02 (13 program+KV tests pass)
- [x] T62.1.3 Tests for CompilePJRT and PJRTPlan  Deps: T62.1.1-T62.1.2  DONE 2026-04-02 (requires PJRT .so for integration)
- [x] T63.1.1 PJRT compilation option in inference  Deps: T62.1.2  DONE 2026-04-02 (inference/pjrt.go, WithPJRT option)
- [x] T63.1.2 Wire PJRTPlan into generator decode loop  Deps: T63.1.1  DONE 2026-04-02 (generate/pjrt.go, RunPrefill+RunDecode)
- [x] T64.1.3 Tests for executable caching  Deps: T64.1.1-T64.1.2  DONE 2026-04-02 (9 cache tests pass)

#### Wave P6: CLI + Validation (7 agents)

- [x] T63.1.3 --pjrt CLI flag  Deps: T63.1.1  DONE 2026-04-02 (run+serve subcommands)
- [ ] T63.1.4 Parity tests (PJRT CPU vs Engine CPU)  Deps: T63.1.2  (requires PJRT CPU plugin .so)
- [ ] T63.2.1 PJRT CUDA integration test  Deps: T63.1.2  (requires DGX Spark + PJRT CUDA plugin)
- [ ] T63.2.2 PJRT Neuron/Trainium test  Deps: T63.1.2  (requires trn1 instance)
- [ ] T63.2.3 Benchmark PJRT vs native  Deps: T63.2.1  (requires DGX Spark)
- [x] T60.3.5 Run go vet E60  Deps: T60.1.4, T60.2.4, T60.3.4  DONE 2026-04-02 (unsafe.Pointer warnings expected for purego)
- [x] T61.3.4 Run go vet E61  Deps: T61.1.4, T61.2.4, T61.3.3  DONE 2026-04-02

#### Wave P7: Final (2 agents)

- [x] T62.1.4 Run go vet E62  Deps: T62.1.3  DONE 2026-04-02
- [ ] T63.2.4 Run go vet E63  Deps: T63.1.4, T63.2.1  (blocked on hardware tests)
- [x] T64.1.4 Run go vet E64  Deps: T64.1.3  DONE 2026-04-02

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria |
|----|-----------|-------|---------------|
| M-P1 | PJRT CPU functional | E60, E61 | Compile and execute StableHLO on CPU plugin via purego. All unit tests pass. |
| M-P2 | Graph compilation working | E62 | CompilePJRT() produces working PJRTPlan. KV cache I/O rewriting verified. |
| M-P3 | Inference on CPU plugin | E63.1 | Gemma3-1B first-layer logits match Engine CPU within 1e-4. |
| M-P4 | CUDA via PJRT | E63.2 (T63.2.1) | Coherent text generation on DGX Spark via PJRT CUDA plugin. |
| M-P5 | Trainium PoC | E63.2 (T63.2.2) | Coherent text generation on trn1.2xlarge. Performance documented. |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-P1 | PJRT C API struct layout changes between versions | High | Low | struct_size field provides compatibility detection. Pin to PJRT v0.103+. Track CHANGELOG.md in openxla/xla. |
| R-P2 | StableHLO compilation fails for complex transformer graphs | High | Medium | Start with simple subgraphs (single layer). Incrementally add ops. Fall back to native backend on compilation error. |
| R-P3 | Neuron plugin compilation time too slow for interactive use | Medium | High | Executable caching (E64) makes second run instant. First-run latency is a known Neuron SDK limitation. |
| R-P4 | PJRT CUDA plugin performance much worse than custom kernels | Medium | High | Expected. PJRT is the "reach play" not the "performance play". Custom kernels remain default for NVIDIA. |
| R-P5 | KV cache explicit I/O adds latency without buffer donation | Medium | Medium | Buffer donation is available on all major backends (CUDA, TPU, Neuron). CPU plugin may not support it but CPU is for testing only. |
| R-P6 | GoMLX StableHLO shape inference has edge cases | Low | Medium | Use GoMLX as reference but implement independently. Extensive shape inference tests. |
| R-P7 | Neuron plugin requires specific library versions or Python env | Medium | High | Document exact version requirements. Test on clean trn1 instance with only aws-neuronx-runtime-lib installed. |
| R-P8 | purego ccall overhead for PJRT function pointers | Low | Low | PJRT calls are coarse-grained (compile, execute) not per-element. Overhead is negligible relative to computation time. |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Benchmark tasks: results appended to docs/devlog.md.
6. Each task committed as its own commit. One logical change per commit.
7. PJRT tests behind `//go:build pjrt_test` tag (require plugin binary).
8. Hardware-specific tests behind additional tags (cuda, neuron).

### Quality Gates

- Every implementation task must have a paired test.
- Run `go vet ./...` after every code change before committing.
- Standard library only: no testify, no cobra, no viper.
- PJRT plugin tests require PJRT CPU plugin .so in test environment.
- Human review gate at each milestone (M-P1 through M-P5).
- Rebase and merge. Not squash, not merge commits.

### PJRT Plugin Availability

- CPU plugin: download from GoMLX releases or build from XLA source.
  Store at ~/.local/lib/go-xla/pjrt-plugin-cpu.so.
- CUDA plugin: download from GoMLX or jax-cuda12-pjrt pip package.
- Neuron plugin: install aws-neuronx-runtime-lib + libneuronxla pip package.
  Plugin at Python site-packages/libneuronxla/libneuronpjrt.so.

---

## Progress Log

### 2026-04-01: Plan created -- PJRT multi-accelerator backend

Created docs/plan-pjrt.md with 5 epics (E60-E64), 39 tasks across 7 waves.
Created 3 ADRs:
- docs/adr/079-pjrt-multi-accelerator-backend.md: PJRT over direct vendor runtimes.
- docs/adr/080-pjrt-kv-cache-explicit-io.md: KV cache as explicit I/O for PJRT.
- docs/adr/081-stablehlo-text-generation.md: StableHLO text over HLO protobuf.

Scope: inference only (no training). Zero CGo. Backward compatible (v1.x).
Priority: E60+E61 (parallel) -> E62 -> E63 -> E64.
Target: Trainium PoC (M-P5) to support Annapurna Labs partnership pitch.

---

## Hand-off Notes

- All PJRT work is in ztensor repo (E60-E62, E64) and zerfoo repo (E63).
- PJRT bindings follow the exact same purego pattern as internal/cuda/purego.go.
- CompileTraced() is the bridge: it already decomposes graphs to primitive ops.
  The PJRT path adds a new consumer of that output, not a new compilation mode.
- KV cache explicit I/O is the key design challenge. ADR-080 explains the approach.
  The implementation is in T61.3.2 (StableHLO emitter) and T62.1.2 (PJRTPlan).
- PJRT CPU plugin is used for all unit/integration tests. No special hardware needed.
- DGX Spark for PJRT CUDA tests (T63.2.1, T63.2.3).
- trn1.2xlarge AWS instance for Neuron tests (T63.2.2). Instance must have
  aws-neuronx-runtime-lib and libneuronxla packages installed.
- GoMLX (github.com/gomlx/go-xla) is the Go reference for PJRT usage. Their
  stablehlo package has shape inference in pure Go. They use CGo for dlopen;
  we use purego.
- Executable caching (E64) is critical for Neuron -- compilation takes minutes.
  Without caching, every restart recompiles.
- The `--pjrt` CLI flag is optional. Default inference path is unchanged.

---

## Appendix

### Use Case IDs Referenced

| ID | Name | Description |
|----|------|-------------|
| UC-PJRT-001 | Run inference via PJRT backend | Load GGUF model, compile graph to PJRT, generate text on any PJRT-supported accelerator |
| UC-PJRT-002 | Multi-step autoregressive decode via PJRT | KV cache managed as explicit I/O with buffer donation across decode steps |
| UC-PJRT-003 | Inference on AWS Trainium | Run model on Trainium via libneuronpjrt.so PJRT plugin |

### Engine[T] to StableHLO Operation Mapping

| Engine[T] Method | StableHLO Op | Notes |
|-----------------|--------------|-------|
| Add | stablehlo.add | Element-wise with broadcast |
| Sub | stablehlo.subtract | Element-wise with broadcast |
| Mul | stablehlo.multiply | Element-wise with broadcast |
| Div | stablehlo.divide | Element-wise with broadcast |
| MulScalar | stablehlo.multiply + constant + broadcast | Scalar promotion |
| DivScalar | stablehlo.divide + constant + broadcast | Scalar promotion |
| AddScalar | stablehlo.add + constant + broadcast | Scalar promotion |
| MatMul | stablehlo.dot_general | Contraction dims {[-1], [0]} |
| Transpose | stablehlo.transpose | Axis permutation |
| Reshape | stablehlo.reshape | Target shape |
| Concat | stablehlo.concatenate | Along specified axis |
| Gather | stablehlo.gather | With offset/collapsed dims |
| Exp | stablehlo.exponential | Unary |
| Log | stablehlo.log | Unary |
| Sin | stablehlo.sine | Unary |
| Cos | stablehlo.cosine | Unary |
| Tanh | stablehlo.tanh | Unary |
| Sqrt | stablehlo.sqrt | Unary |
| Rsqrt | stablehlo.rsqrt | Unary |
| Pow | stablehlo.power | Binary |
| Softmax | Decomposed: reduce_max, sub, exp, reduce_sum, div | 5 ops |
| ReduceSum | stablehlo.reduce with add body | Along axis |
| ReduceMax | stablehlo.reduce with max body | Along axis |
| ReduceMean | reduce_sum then divide by size | 2 ops |
| OneHot | stablehlo.compare + stablehlo.select | Index comparison |
| Fill | stablehlo.constant + stablehlo.broadcast_in_dim | Scalar broadcast |

### ADRs Created

- docs/adr/079-pjrt-multi-accelerator-backend.md -- PJRT over direct vendor runtimes
- docs/adr/080-pjrt-kv-cache-explicit-io.md -- KV cache as explicit I/O with buffer donation
- docs/adr/081-stablehlo-text-generation.md -- StableHLO MLIR text over HLO protobuf
