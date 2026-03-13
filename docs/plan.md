# Zerfoo Development Plan -- Surpass Ollama Inference Performance

## 1. Context

### Problem Statement

Zerfoo inference on DGX Spark GB10 achieves 188.92 tok/s (95.8% of Ollama's
197.21 tok/s). The 95% parity target is met, but the goal is now to surpass
Ollama. The remaining 8.29 tok/s gap (~4.2%) is caused by:

1. Kernel launch overhead: ~338 launches/token at ~7us each = ~2.37ms. CUDA
   graph replay would reduce this to ~15us total.
2. Three D2H (device-to-host) copy sites in the decode path that block CUDA
   graph capture.
3. No fused dequant+GEMV kernel: separate dequantize then multiply doubles
   memory traffic through 273 GB/s LPDDR5x.
4. Unified memory (cudaMallocManaged) not yet exploited on GB10's shared
   LPDDR5x, which eliminates H2D/D2H copies entirely.

Additionally, the OpenAI-compatible inference server in serve/ needs additional
endpoints and an OpenAPI specification to be production-ready. See
docs/adr/031-openai-server-in-zerfoo.md for the server placement decision.

See docs/design.md for full architecture, completed optimizations, and ADR-025
results.

### Objectives

- O1: Surpass Ollama throughput (>197.21 tok/s) on DGX Spark GB10.
- O2: Enable CUDA graph capture and replay for near-zero launch overhead.
- O3: Exploit GB10 unified memory to eliminate H2D/D2H copies.
- O4: Fused dequant+GEMV kernel to halve memory bandwidth for Q4 matmul.
- O5: Production-ready OpenAI-compatible server with OpenAPI spec.
- O6: GPU Transpose, Gather, broadcasting kernels to eliminate CPU fallbacks.
- O7: Convert cuBLAS, cuDNN, TensorRT, CUTLASS, ROCm, OpenCL from CGo to purego.
- O8: Megakernel investigation and fix or documented abandonment.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Ollama baseline: 197.21 tok/s (Gemma 3 1B Q4_K_M, measured 2026-03-12).
- Zerfoo current best: 188.92 tok/s average (3 runs, 2026-03-12).
- Theoretical max for Q4 model on GB10: ~350-400 tok/s (bandwidth ceiling).
- internal/cublas/ and internal/cudnn/ currently require CGo (//go:build cuda).
- ROCm backend ~80% complete. OpenCL ~40%.
- Existing serve/ package has core OpenAI endpoints but lacks embeddings, model
  management, and formal OpenAPI spec.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Inference throughput | > 197.21 tok/s on DGX Spark | bench_tps with Gemma 3 1B Q4, 3-run avg |
| CUDA graph operational | Graph replay for decode tokens 3+ | nsys trace shows single graph launch |
| D2H copies eliminated | Zero D2H memcpy in decode path | nsys trace, grep for Memcpy D2H |
| OpenAPI spec | Valid OpenAPI 3.1 YAML | Swagger validator passes |
| Build simplicity | go build ./... (no CGo, no build tags) | Build on clean machine without CUDA toolkit |
| Test suite | All tests pass | go test ./... -race -timeout 120s |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D20 | D2H copy elimination (3 sites) | Unblocks CUDA graph capture |
| D21 | CUDA graph capture and replay | Eliminates ~2.37ms launch overhead/token |
| D22 | Unified memory exploitation | Eliminates H2D/D2H on GB10 shared LPDDR5x |
| D23 | Fused dequant+GEMV kernel | Halves bandwidth for Q4 matmul |
| D24 | OpenAPI server completion | Production-ready /v1/* endpoints with spec |
| D9 | GPU Transpose, Gather, broadcasting kernels | Eliminate CPU fallbacks |
| D13 | cuBLAS purego conversion | Single-binary, no CGo |
| D14 | cuDNN purego conversion | Single-binary, no CGo |
| D15 | TensorRT purego conversion | Single-binary, no CGo |
| D16 | CUTLASS/flash attention purego conversion | Single-binary, no CGo |
| D17 | ROCm purego conversion | AMD GPU support without CGo |
| D18 | OpenCL purego conversion | Intel/other GPU support without CGo |
| D12 | Megakernel investigation | Fix or documented abandonment |
| D19 | Kernel optimization | Close gap to theoretical max |

### Out of Scope

- Training performance optimization (inference only).
- New model architecture support beyond what exists.
- Distributed inference (multi-GPU parallelism).
- Mobile or embedded deployment.

---

## 3. Checkable Work Breakdown

### E301: D2H Copy Elimination

- [x] T301.1 Eliminate D2H in GPUEngine.Gather (indices int64->int32)  Owner: task-T301.1  Completed: 2026-03-13
  - Currently reads indices.Data() to convert int64 to int32 on CPU before
    uploading to GPU. Replace with a GPU int64-to-int32 conversion kernel or
    keep indices as int32 from the start.
  - File: compute/gpu_engine.go, line ~1242.
  - Acceptance: GPUEngine.Gather executes without any cudaMemcpy D2H during
    decode. Verified by adding a log guard or nsys trace.
  - Dependencies: none.

- [x] T301.2 Eliminate D2H in GQA TrySlice CPU fallback  Owner: task-T301.2  Completed: 2026-03-13
  - GPUStorage.TrySlice in layers/attention/grouped_query_attention.go copies
    entire GPU tensor to host for slicing. Replace with GPU-side slicing using
    pointer arithmetic on the device buffer (offset into GPUStorage).
  - File: tensor/gpu_storage.go (TrySlice), layers/attention/grouped_query_attention.go.
  - Acceptance: GQA forward pass has zero TrySlice calls during decode.
  - Dependencies: none.

- [x] T301.3 Eliminate D2H in tensor_cache appendGPU fallback  Owner: task-T301.3  Completed: 2026-03-13
  - When source tensor is GPU-resident, appendGPU calls src.Data() which
    triggers D2H. Replace with GPU-to-GPU copy (cudaMemcpy D2D) using the
    source's device pointer directly.
  - File: generate/tensor_cache.go, line ~124.
  - Acceptance: appendGPU uses D2D copy when source is GPUStorage.
  - Dependencies: none.

- [ ] S301.3.1 D2H elimination verification test  Owner: TBD  Est: 1h
  - Run bench_tps with instrumented D2H counter. Assert zero D2H copies
    in the decode path (post-prefill).
  - Acceptance: go test passes. No D2H memcpy during decode loop.
  - Dependencies: T301.1, T301.2, T301.3.

- [ ] T301.4 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S301.3.1.

### E302: CUDA Graph Capture and Replay

- [x] T302.1 Enable CUDA graph capture in generator.go  Owner: task-T302.1  Completed: 2026-03-13
  - Uncomment the CUDA graph executor wiring in compileGraph(). The D2H copy
    elimination in E301 makes this safe.
  - File: generate/generator.go, lines 166-175.
  - Acceptance: CUDAGraphExecutor is created and wired into the megakernelFn
    hook when StreamProvider and GraphAvailable() are both true.
  - Dependencies: E301.

- [ ] T302.2 Verify graph capture succeeds on DGX Spark  Owner: TBD  Est: 1h
  - Run bench_tps on DGX Spark. Confirm log message "cuda graph: captured and
    instantiated successfully" appears.
  - If capture fails, diagnose and fix the remaining conflict.
  - Acceptance: Graph capture succeeds. Tokens 3+ use graph replay.
  - Dependencies: T302.1.

- [ ] T302.3 Benchmark CUDA graph replay vs per-op  Owner: TBD  Est: 30m
  - Run bench_tps 3 times with graph, 3 times without. Report tok/s delta.
  - Acceptance: Results documented. Graph replay faster than per-op.
  - Dependencies: T302.2.

- [ ] S302.3.1 CUDA graph correctness test  Owner: TBD  Est: 1h
  - Compare output tokens with graph enabled vs disabled for 10 tokens at
    temperature=0. Tokens must be identical.
  - Acceptance: go test passes on DGX Spark.
  - Dependencies: T302.2.

- [ ] T302.4 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S302.3.1.

### E303: Unified Memory Exploitation

- [x] T303.1 Use cudaMallocManaged for arena allocator on GB10  Owner: task-T303  Completed: 2026-03-13
  - Detect NVLink-C2C / unified memory hardware at startup. When available,
    switch the arena allocator from cudaMalloc to cudaMallocManaged.
  - This eliminates all explicit H2D copies for intermediate tensors because
    the GPU and CPU share the same physical memory on GB10.
  - File: compute/gpu_engine.go (arena allocator), internal/cuda/runtime_purego.go.
  - Acceptance: Arena uses managed memory on GB10. No functional regression.
  - Dependencies: none.

- [x] T303.2 Use cudaMallocManaged for model weights on GB10  Owner: task-T303  Completed: 2026-03-13
  - When loading model weights, allocate with managed memory instead of
    cudaMalloc + cudaMemcpy H2D. Weights become accessible from both CPU
    and GPU without explicit copies.
  - File: compute/gpu_engine.go (weight upload path).
  - Acceptance: Model loading skips explicit H2D when managed memory available.
  - Dependencies: none.

- [ ] T303.3 Benchmark unified memory vs explicit copy  Owner: TBD  Est: 30m
  - Compare tok/s with managed memory vs current explicit copy path.
  - Acceptance: Results documented. No regression.
  - Dependencies: T303.1, T303.2.

- [ ] S303.3.1 Unified memory correctness test  Owner: TBD  Est: 1h
  - Run inference with managed memory, compare output with explicit copy path.
  - Tokens must match at temperature=0.
  - Acceptance: go test passes.
  - Dependencies: T303.3.

- [ ] T303.4 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S303.3.1.

### E304: Fused Dequant+GEMV Kernel

- [x] T304.1 Write CUDA fused dequant+GEMV kernel for Q4_K_M  Owner: task-T304.1  Completed: 2026-03-13
  - Read Q4_K_M blocks, dequantize in registers, multiply by activation
    vector, accumulate in F32. Single-token decode (batch=1).
  - This is the highest-impact single kernel optimization: dequantize is the
    largest kernel by time, and fusing it with GEMV halves memory traffic.
  - Acceptance: Kernel compiles for sm_121. Output matches unfused path
    (max rel error < 1e-4).
  - Dependencies: none.

- [x] T304.2 Wire fused dequant+GEMV into GPUEngine  Owner: task-T304.2  Completed: 2026-03-13
  - Detect Q4_K_M quantized weights in MatMul dispatch. When batch=1 and
    weights are Q4, use fused kernel instead of dequantize + cuBLAS Sgemm.
  - File: compute/gpu_engine.go (MatMul path).
  - Acceptance: bench_tps uses fused kernel for Q4 decode (verify via logging).
  - Dependencies: T304.1.

- [ ] S304.2.1 Fused dequant+GEMV parity test  Owner: TBD  Est: 1h
  - Compare fused kernel output with unfused path for multiple matrix sizes.
  - Max relative error < 1e-4.
  - Acceptance: go test passes with -race.
  - Dependencies: T304.2.

- [ ] T304.3 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S304.2.1.

### E305: OpenAI-Compatible Server Completion

- [x] T305.1 Add POST /v1/embeddings endpoint  Owner: task-E305  Completed: 2026-03-13
  - Implement OpenAI-compatible embeddings endpoint using inference.Model.Embed().
  - Support single and batch input strings.
  - Return embedding vectors in OpenAI response format.
  - File: serve/server.go.
  - Acceptance: curl POST /v1/embeddings returns valid embedding response.
  - Dependencies: none.

- [x] T305.2 Add DELETE /v1/models/:id endpoint  Owner: task-E305  Completed: 2026-03-13
  - Allow unloading a model from memory via API.
  - File: serve/server.go.
  - Acceptance: DELETE returns 200, model is unloaded.
  - Dependencies: none.

- [x] T305.3 Add GET /v1/models/:id endpoint  Owner: task-E305  Completed: 2026-03-13
  - Return detailed model info (id, object, created, owned_by, architecture).
  - File: serve/server.go.
  - Acceptance: curl GET /v1/models/:id returns model detail.
  - Dependencies: none.

- [x] T305.4 Add OpenAPI 3.1 specification YAML  Owner: task-T305.4  Completed: 2026-03-13
  - Write serve/openapi.yaml documenting all /v1/* endpoints with request
    and response schemas. Follow OpenAI API spec structure.
  - Acceptance: Validates with swagger-cli validate.
  - Dependencies: T305.1, T305.2, T305.3.

- [x] T305.5 Add GET /openapi.yaml endpoint to serve spec  Owner: task-T305.5  Completed: 2026-03-13
  - Serve the OpenAPI spec from the server itself for client discovery.
  - Embed the YAML using go:embed.
  - File: serve/server.go.
  - Acceptance: curl GET /openapi.yaml returns valid YAML.
  - Dependencies: T305.4.

- [x] T305.6 Add usage token counting to all response types  Owner: task-E305  Completed: 2026-03-13
  - Count prompt_tokens and completion_tokens in all response types.
  - Currently only TotalTokens is set in some responses.
  - File: serve/server.go.
  - Acceptance: All responses include accurate prompt_tokens and completion_tokens.
  - Dependencies: none.

- [ ] S305.6.1 Server integration test  Owner: TBD  Est: 2h
  - Test all endpoints (chat, completions, embeddings, models, openapi.yaml)
    with httptest.NewServer. Verify request/response formats match OpenAI spec.
  - Test SSE streaming for chat and completions.
  - Acceptance: go test -run TestServer passes with -race.
  - Dependencies: T305.1, T305.2, T305.3, T305.4, T305.5, T305.6.

- [ ] T305.7 Run go vet on serve/  Owner: TBD  Est: 15m
  - Dependencies: S305.6.1.

### E203: GPU Transpose Kernel

- [x] T203.1 Write CUDA transpose kernel for 2D/3D/4D tensors  Owner: task-T203.1  Completed: 2026-03-13
  - Add transpose.cu or extend existing kernel file.
  - Support arbitrary axis permutations for 2D, 3D, and 4D tensors.
  - Use shared memory tiling for coalesced reads and writes.
  - Acceptance: Kernel compiles for sm_75 and sm_121.
  - Dependencies: none.

- [x] T203.2 Wire GPU Transpose into GPUEngine  Owner: task-T203.2  Completed: 2026-03-13
  - Replace CPU fallback in GPUEngine.Transpose with GPU kernel dispatch.
  - Keep CPU fallback for tensors with >4 dimensions.
  - Acceptance: GPUEngine.Transpose returns GPUStorage result for 2D/3D/4D inputs.
  - Dependencies: T203.1.

- [ ] S203.2.1 GPU Transpose parity test  Owner: TBD  Est: 1h
  - Compare GPU Transpose output with CPU Transpose for multiple shapes and
    axis permutations. Max relative error < 1e-6.
  - Acceptance: go test -run TestGPUTranspose passes with -race.
  - Dependencies: T203.2.

- [ ] T203.3 Run go vet on compute/ and internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S203.2.1.

### E204: GPU Gather Kernel

- [x] T204.1 Write CUDA gather kernel for embedding lookups  Owner: task-T204.1  Completed: 2026-03-13
  - Add gather.cu kernel or extend existing. Support int32 and int64 indices.
  - Each thread block handles one index, copies embedding_dim elements.
  - Acceptance: Kernel compiles for sm_75 and sm_121.
  - Dependencies: none.

- [x] T204.2 Wire GPU Gather into GPUEngine  Owner: task-T204.2  Completed: 2026-03-13 (already implemented)
  - Replace CPU fallback in GPUEngine.Gather with GPU kernel dispatch.
  - Acceptance: GPUEngine.Gather returns GPUStorage result.
  - Dependencies: T204.1.

- [ ] S204.2.1 GPU Gather parity test  Owner: TBD  Est: 1h
  - Compare GPU Gather output with CPU Gather for multiple table sizes and
    index patterns. Exact match required (integer indexing, no floating point).
  - Acceptance: go test -run TestGPUGather passes with -race.
  - Dependencies: T204.2.

- [ ] T204.3 Run go vet on compute/ and internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S204.2.1.

### E205: GPU Broadcasting for Element-wise Ops

- [x] T205.1 Extend CUDA element-wise kernels for broadcasting  Owner: task-T205.1  Completed: 2026-03-13
  - Support scalar broadcast, row broadcast, and column broadcast patterns.
  - Compute strides for broadcasting inside the kernel (no pre-expansion).
  - Acceptance: Kernels compile and handle mismatched shapes.
  - Dependencies: none.

- [ ] T205.2 Remove sameShape guard from GPUEngine binary ops  Owner: TBD  Est: 1h
  - Update Add, Sub, Mul, Div, Pow to use broadcasting kernels when shapes differ.
  - Keep CPU fallback for >4D broadcasting edge cases.
  - Acceptance: GPUEngine binary ops produce GPUStorage for broadcastable shapes.
  - Dependencies: T205.1.

- [ ] S205.2.1 GPU broadcasting parity test  Owner: TBD  Est: 1h
  - Test scalar * tensor, [1,N] + [M,N], [M,1] * [M,N] patterns.
  - Max relative error < 1e-6.
  - Acceptance: go test passes with -race.
  - Dependencies: T205.2.

- [ ] T205.3 Run go vet on compute/ and internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S205.2.1.

### E306: Fused Kernel Wiring and Integration

- [ ] T306.1 Wire remaining fused kernels into GPUEngine  Owner: TBD  Est: 2h
  - Fused SwiGLU (T206.1, done) and Scale+Softmax (T206.2, done) are
    implemented but may not be dispatched in all code paths. Verify dispatch
    logic covers ExecutionPlan and direct Forward paths.
  - Acceptance: bench_tps log shows fused kernel dispatch for all fusable ops.
  - Dependencies: none.

- [ ] S306.1.1 Fused kernel integration test  Owner: TBD  Est: 1h
  - End-to-end: run inference with fused kernels, compare output with
    unfused path. Tokens must match.
  - Acceptance: go test passes with -race.
  - Dependencies: T306.1.

- [ ] T306.2 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S306.1.1.

### E207: CUDA Graph Capture and Replay (Infrastructure)

Note: CUDA graph API wrappers (T207.1) are already implemented in
internal/cuda/runtime_purego.go. CUDAGraphExecutor is implemented in
graph/cuda_graph.go. StreamProvider is on GPUEngine. What remains is
enabling it after D2H elimination.

- [ ] T207.2 Pre-allocate fixed buffer layout in ExecutionPlan  Owner: TBD  Est: 3h
  - At compile time, compute shape of every intermediate tensor.
  - Allocate one contiguous GPU buffer with fixed offsets per slot.
  - CUDA graph capture requires fixed memory addresses.
  - Acceptance: ExecutionPlan uses pre-allocated buffers. No per-op alloc/free.
  - Dependencies: E301.

- [ ] S207.2.1 Fixed buffer layout test  Owner: TBD  Est: 1h
  - Verify pre-allocated buffers produce identical results to dynamic allocation.
  - Acceptance: go test passes.
  - Dependencies: T207.2.

- [ ] T207.3 Run go vet on graph/  Owner: TBD  Est: 15m
  - Dependencies: S207.2.1.

### E208: Megakernel Performance Investigation

- [ ] T208.1 Profile megakernel with nsys on DGX Spark  Owner: TBD  Est: 2h
  - Run nsys profile on bench_tps with megakernel enabled.
  - Identify: kernel duration, occupancy, register usage, shared memory usage.
  - Compare with per-op kernels for the same operations.
  - Acceptance: Profile report with root cause for performance gap.
  - Dependencies: none.

- [ ] T208.2 Fix or redesign megakernel based on profile  Owner: TBD  Est: 4h
  - If fixable: optimize the code generator output (tiling, shared memory, etc.).
  - If not fixable: document why and rely on CUDA graph + fused kernels instead.
  - Acceptance: Megakernel >= 50 tok/s, OR documented decision to abandon
    megakernel in favor of CUDA graph approach.
  - Dependencies: T208.1.

- [ ] S208.2.1 Megakernel benchmark comparison  Owner: TBD  Est: 30m
  - Compare megakernel tok/s with per-op tok/s and CUDA graph tok/s.
  - Acceptance: Results documented.
  - Dependencies: T208.2.

- [ ] T208.3 Run go vet on internal/codegen/  Owner: TBD  Est: 15m
  - Dependencies: T208.2.

### E209: Kernel Optimization

- [ ] T209.1 Tune register pressure and occupancy  Owner: TBD  Est: 3h
  - Use --maxrregcount to limit registers per thread.
  - Profile occupancy with nsys for each kernel.
  - Target: >= 50% occupancy for compute-bound kernels.
  - Acceptance: Occupancy report. Adjustments applied where beneficial.
  - Dependencies: none.
  - Note: NVCC -O3 --use_fast_math already applied (commit d1ed26a, negligible gain).

- [ ] T209.2 Optimize shared memory usage in attention and reduction kernels  Owner: TBD  Est: 2h
  - Flash attention: tune BLOCK_SIZE for sm_121 shared memory capacity.
  - Reduction kernels: use warp shuffle instead of shared memory where possible.
  - Acceptance: Kernels pass parity tests. Benchmark improvement documented.
  - Dependencies: none.

- [ ] S209.2.1 Full kernel benchmark suite  Owner: TBD  Est: 1h
  - Benchmark each kernel individually: elementwise, flash_attention, gemm_q4,
    rmsnorm, transpose, gather.
  - Compare with pre-optimization baselines.
  - Acceptance: Per-kernel benchmark results documented.
  - Dependencies: T209.1, T209.2.

- [ ] T209.3 Run go vet on internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S209.2.1.

### E210: cuBLAS Purego Conversion

- [ ] T210.1 Create purego wrappers for cuBLAS API  Owner: TBD  Est: 4h
  - Wrap via purego dlopen of libcublas.so: cublasCreate, cublasDestroy,
    cublasSetStream, cublasSgemm, cublasGemmEx.
  - Match existing internal/cublas/ API surface.
  - Note: cublasSgemm and cublasSgemmStridedBatched are already implemented
    via purego. Remaining: cublasGemmEx for BFloat16 and other types.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T210.2 Replace CGo cublas.go with purego implementation  Owner: TBD  Est: 2h
  - Delete or archive internal/cublas/cublas.go (CGo version).
  - Remove //go:build cuda tag from new purego file.
  - Add cublas.Available() runtime guard.
  - Acceptance: go build ./internal/cublas/... without -tags cuda.
  - Dependencies: T210.1.

- [ ] T210.3 Update gpuapi/cuda_blas.go to use purego cublas  Owner: TBD  Est: 1h
  - Remove //go:build cuda tag.
  - Acceptance: go build ./internal/gpuapi/... without -tags cuda.
  - Dependencies: T210.2.

- [ ] S210.3.1 cuBLAS purego parity test  Owner: TBD  Est: 1h
  - Run MatMul with purego cuBLAS path on DGX Spark.
  - Compare output with CPU MatMul. Max relative error < 1e-5.
  - Benchmark purego vs CGo cuBLAS overhead (should be < 5% difference).
  - Acceptance: Parity test passes. Performance delta documented.
  - Dependencies: T210.3.

- [ ] T210.4 Run go vet on internal/cublas/ and internal/gpuapi/  Owner: TBD  Est: 15m
  - Dependencies: S210.3.1.

### E211: cuDNN Purego Conversion

- [ ] T211.1 Create purego wrappers for cuDNN API  Owner: TBD  Est: 6h
  - Wrap via purego dlopen of libcudnn.so: cudnnCreate, cudnnDestroy,
    cudnnSetStream, cudnnCreateTensorDescriptor, cudnnSetTensor4dDescriptor,
    cudnnConvolutionForward, cudnnBatchNormalizationForwardInference,
    cudnnActivationForward, cudnnPoolingForward, cudnnSoftmaxForward,
    and their backward counterparts.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T211.2 Replace CGo cudnn.go with purego implementation  Owner: TBD  Est: 2h
  - Delete or archive internal/cudnn/cudnn.go (CGo version).
  - Remove //go:build cuda tag.
  - Add cudnn.Available() runtime guard.
  - Acceptance: go build ./internal/cudnn/... without -tags cuda.
  - Dependencies: T211.1.

- [ ] T211.3 Update gpuapi/cuda_dnn.go to use purego cudnn  Owner: TBD  Est: 1h
  - Remove //go:build cuda tag.
  - Acceptance: go build ./internal/gpuapi/... without -tags cuda.
  - Dependencies: T211.2.

- [ ] S211.3.1 cuDNN purego parity test  Owner: TBD  Est: 2h
  - Run each cuDNN operation (conv, batchnorm, activation, pooling, softmax)
    with purego path on DGX Spark.
  - Compare output with CPU reference. Max relative error < 1e-4.
  - Acceptance: All operations pass parity test.
  - Dependencies: T211.3.

- [ ] T211.4 Run go vet on internal/cudnn/ and internal/gpuapi/  Owner: TBD  Est: 15m
  - Dependencies: S211.3.1.

### E212: TensorRT Purego Conversion

- [ ] T212.1 Create purego wrappers for TensorRT C API  Owner: TBD  Est: 6h
  - TensorRT has a C++ API wrapped by internal/tensorrt/cshim/. The C shim
    provides a flat C interface. Create purego wrappers that dlopen libtrt_capi.so.
  - Update Makefile to produce shared library.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T212.2 Replace CGo tensorrt.go with purego implementation  Owner: TBD  Est: 2h
  - Remove //go:build cuda tag. Add tensorrt.Available() runtime guard.
  - Acceptance: go build ./internal/tensorrt/... without -tags cuda.
  - Dependencies: T212.1.

- [ ] T212.3 Update inference/tensorrt_*.go to remove build tags  Owner: TBD  Est: 1h
  - Remove //go:build cuda from tensorrt_cache.go, tensorrt_convert.go,
    tensorrt_pipeline.go. Add runtime Available() guards.
  - Acceptance: go build ./inference/... without -tags cuda.
  - Dependencies: T212.2.

- [ ] S212.3.1 TensorRT purego integration test  Owner: TBD  Est: 2h
  - Run TensorRT inference pipeline on DGX Spark with purego path.
  - Compare output with standard inference path.
  - Acceptance: Tokens match. Test passes.
  - Dependencies: T212.3.

- [ ] T212.4 Run go vet on internal/tensorrt/ and inference/  Owner: TBD  Est: 15m
  - Dependencies: S212.3.1.

### E213: CUTLASS Flash Attention Purego Conversion

- [ ] T213.1 Convert flash_attention.cu dispatch to purego  Owner: TBD  Est: 3h
  - flash_attention.go currently uses CGo (//go:build cuda && cutlass).
  - Convert to purego dlopen of libkernels.so.
  - Remove cutlass build tag requirement.
  - Acceptance: Flash attention dispatches via purego. No CGo.
  - Dependencies: none.

- [ ] T213.2 Update layers/attention/flash_cuda.go  Owner: TBD  Est: 1h
  - Remove //go:build cuda && cutlass tag. Add runtime cuda.Available() guard.
  - Merge flash_cuda.go and flash_nocuda.go into single file.
  - Acceptance: go build ./layers/attention/... without build tags.
  - Dependencies: T213.1.

- [ ] S213.2.1 Flash attention purego parity test  Owner: TBD  Est: 1h
  - Compare purego flash attention output with naive attention path.
  - Max relative error < 1e-4.
  - Acceptance: Test passes on DGX Spark.
  - Dependencies: T213.2.

- [ ] T213.3 Run go vet on layers/attention/ and internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S213.2.1.

### E214: ROCm Purego Conversion

- [ ] T214.1 Create purego wrappers for HIP runtime API  Owner: TBD  Est: 4h
  - Wrap via purego dlopen of libamdhip64.so.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T214.2 Create purego wrappers for rocBLAS API  Owner: TBD  Est: 3h
  - Wrap via purego dlopen of librocblas.so.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T214.3 Replace CGo ROCm files with purego implementations  Owner: TBD  Est: 3h
  - Update internal/hip/, internal/rocblas/, internal/gpuapi/rocm_*.go.
  - Remove //go:build rocm tags. Add hip.Available() runtime guard.
  - Acceptance: go build ./... without -tags rocm.
  - Dependencies: T214.1, T214.2.

- [ ] T214.4 Convert HIP kernels to purego dispatch  Owner: TBD  Est: 2h
  - HIP kernels in internal/hip/kernels/ compiled to .so.
  - Dispatch via purego dlopen instead of CGo.
  - Acceptance: Kernels dispatch via purego.
  - Dependencies: T214.3.

- [ ] S214.4.1 ROCm purego integration test  Owner: TBD  Est: 2h
  - Run basic inference on AMD GPU with purego ROCm path.
  - Acceptance: Inference produces correct output on AMD GPU.
  - Dependencies: T214.4.
  - Risk: Requires access to AMD GPU for testing.

- [ ] T214.5 Run go vet on all ROCm packages  Owner: TBD  Est: 15m
  - Dependencies: S214.4.1.

### E215: OpenCL Purego Conversion

- [ ] T215.1 Create purego wrappers for OpenCL API  Owner: TBD  Est: 4h
  - Wrap via purego dlopen of libOpenCL.so.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T215.2 Replace CGo OpenCL runtime with purego  Owner: TBD  Est: 2h
  - Update internal/opencl/runtime.go. Remove //go:build opencl tag.
  - Add opencl.Available() runtime guard.
  - Acceptance: go build ./internal/opencl/... without -tags opencl.
  - Dependencies: T215.1.

- [ ] T215.3 Update gpuapi/opencl_*.go to use purego  Owner: TBD  Est: 2h
  - Remove build tags from all 5 opencl adapter files.
  - Acceptance: go build ./internal/gpuapi/... without opencl tag.
  - Dependencies: T215.2.

- [ ] S215.3.1 OpenCL purego integration test  Owner: TBD  Est: 2h
  - Run basic inference on OpenCL device with purego path.
  - Acceptance: Inference produces output on OpenCL device.
  - Dependencies: T215.3.
  - Risk: Requires OpenCL-capable device for testing.

- [ ] T215.4 Run go vet on internal/opencl/ and internal/gpuapi/  Owner: TBD  Est: 15m
  - Dependencies: S215.3.1.

### E307: Performance Verification -- Surpass Ollama

- [ ] T307.1 Run bench_tps on DGX Spark with all optimizations  Owner: TBD  Est: 1h
  - Test all paths: per-op, fused kernels, CUDA graph, megakernel.
  - Record tok/s for each. Compare with baselines.
  - Acceptance: At least one path achieves > 197.21 tok/s (surpasses Ollama).
  - Dependencies: E301, E302, E303, E304.

- [ ] T307.2 Compare Zerfoo vs Ollama output quality  Owner: TBD  Est: 1h
  - Same prompt, same model, temperature=0. Compare first 50 tokens.
  - Acceptance: Output is coherent and comparable to Ollama.
  - Dependencies: T307.1.

- [ ] S307.2.1 Performance and correctness report  Owner: TBD  Est: 30m
  - Document all benchmark results in a table.
  - Include: Zerfoo (all paths), Ollama, theoretical max.
  - Acceptance: Report written to docs/QUALITY.md.
  - Dependencies: T307.2.

- [ ] T307.3 Verify go build ./... without any build tags  Owner: TBD  Est: 30m
  - Build on macOS (no GPU) and DGX Spark (GPU).
  - All packages must compile without -tags cuda, -tags rocm, -tags opencl.
  - Acceptance: go build ./... passes on both platforms.
  - Dependencies: E210, E211, E212, E213, E214, E215.

- [ ] T307.4 Run full test suite  Owner: TBD  Est: 1h
  - go test ./... -race -timeout 120s on DGX Spark.
  - Acceptance: All tests pass.
  - Dependencies: T307.3.

- [ ] T307.5 Run go vet on all packages  Owner: TBD  Est: 15m
  - Dependencies: T307.4.

---

## 4. Parallel Work

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: D2H Elimination | E301 | Critical path. Unblocks CUDA graph. 3 independent subtasks (T301.1-T301.3). |
| Track B: CUDA Graph | E302, E207 | Depends on E301 (D2H elimination). |
| Track C: Unified Memory | E303 | Independent. Can run parallel with Track A. |
| Track D: Fused Dequant | E304 | Independent. Can run parallel with Tracks A-C. |
| Track E: OpenAPI Server | E305 | Independent. No GPU dependencies. |
| Track F: GPU Residency | E203, E204, E205 | Independent. Different kernel files. |
| Track G: Kernel Fusion | E306 | Independent. Verifies existing fused kernels. |
| Track H: Megakernel | E208 | Independent. Can run parallel with all tracks. |
| Track I: Kernel Opt | E209 | Independent. Can run parallel with all tracks. |
| Track J: Purego Conversions | E210-E215 | 6 independent epics. All can run parallel. |
| Track K: Verification | E307 | Sync point: depends on Tracks A-D for perf, J for build. |

Sync points:
- After Track A (E301): Track B (CUDA graph) unblocks.
- After Tracks A + B + C + D: Performance verification T307.1.
- After Track J: Build verification T307.3.
- After all: Final verification E307.

Maximum parallelism:
- Wave 1: E301 (D2H elimination, 3 parallel subtasks) + E303 (unified memory) +
  E304 (fused dequant) + E305 (OpenAPI server) + E203-E205 (GPU residency) +
  E208-E209 (megakernel + kernel opt) + E210-E215 (purego conversions)
- Wave 2: E302 + E207 (CUDA graph, depends on E301) + E306 (kernel wiring)
- Wave 3: E307 (verification, depends on all)

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M72: D2H copies eliminated | E301 | Zero D2H memcpy in decode path, verified by instrumentation |
| M73: CUDA graph operational | E302, E207 | Graph replay for tokens 3+, log confirms capture success |
| M74: Surpass Ollama | E301-E304, T307.1 | bench_tps > 197.21 tok/s on DGX Spark |
| M75: OpenAPI server complete | E305 | All endpoints functional, OpenAPI spec validates |
| M76: Single binary | E210-E215 | go build ./... without any build tags or CGo |
| M77: All verified | E307 | Full test suite passes, benchmark report complete |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R301 | D2H elimination introduces subtle correctness bugs | Incorrect output | Medium | Parity tests at each step. Compare output before/after each elimination. |
| R302 | CUDA graph still fails after D2H elimination (unknown D2H site) | Cannot use graph | Low | Instrument all cudaMemcpy calls with direction logging. |
| R303 | Unified memory slower than explicit copy on GB10 | Performance regression | Low | Benchmark before committing. Keep explicit copy path as fallback. |
| R304 | Fused dequant+GEMV register pressure too high for sm_121 | Kernel fails or slow | Medium | Profile with --ptxas-options=-v. Tile to reduce registers. |
| R305 | CUDA graph capture conflicts with arena allocator reset | Graph replay uses stale pointers | Medium | Pre-allocate fixed buffer layout (T207.2) before capture. |
| R203 | purego cuBLAS/cuDNN slower than CGo | Performance regression | Low | purego overhead ~100ns/call. Benchmark before/after. |
| R205 | ROCm/OpenCL testing blocked by hardware access | Cannot verify backends | High | Focus on CUDA first. Test in CI with rented GPU instances. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without build tags.
3. go test for the modified package passes with -race.
4. Commit passes pre-commit hooks.
5. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits. Do not let changes pile up.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- Benchmark: bench_tps on DGX Spark for performance-related changes.

---

## 8. Progress Log

### Change Summary -- 2026-03-12 (Plan Update)

Plan restructured to target surpassing Ollama (>197.21 tok/s) rather than
just matching. Major changes:

- Trimmed completed epics E201 (baseline), E202 (correctness), T206.1 (SwiGLU),
  T206.2 (Scale+Softmax) from work breakdown. Knowledge preserved in
  docs/design.md under "Ollama Performance Parity Achieved" section.
- Added E301: D2H copy elimination (3 specific sites identified).
- Added E302: CUDA graph enablement (depends on E301, infrastructure exists).
- Added E303: Unified memory exploitation on GB10.
- Added E304: Fused dequant+GEMV kernel (was T206.3, promoted to epic).
- Added E305: OpenAI server completion with embeddings, model management,
  OpenAPI spec.
- Added E306: Fused kernel wiring verification.
- Added E307: Performance verification targeting >197.21 tok/s.
- Renumbered T209.1 (NVCC flags already done, task now focuses on occupancy).
- Updated success metrics from >=100 tok/s to >197.21 tok/s.
- Created ADR: docs/adr/031-openai-server-in-zerfoo.md (server stays in Zerfoo).
- Removed obsolete risks R201, R208 (correctness fixed, Ollama measured).
- Updated risk R202 target from 100 tok/s to surpassing Ollama.
- Archived S201.2.1 (nsys profiling) as optional enhancement, not blocking.

### Prior Progress -- 2026-03-12

95% Ollama performance target achieved: 188.92 tok/s avg (3 runs) vs 187.35
target on DGX Spark GB10. Ollama measured at 197.21 tok/s.

Completed:
- E201 (T201.1, T201.2): Baselines measured.
- E202 (T202.1, T202.2): Inference correctness fixed.
- T206.1: Fused SwiGLU kernel.
- T206.2: Fused Scale+Softmax kernel.
- CUDA graph infrastructure built but disabled (3 D2H copy sites block capture).
- NVCC -O3 --use_fast_math applied (negligible gain, bandwidth-bound).
- Arena allocator, KV cache pre-allocation, GQA broadcast, MatMulTransposeB,
  cublasSgemmStridedBatched, fused QK norm+RoPE, zero-copy Q+K view,
  fused post-FFN norm+add all committed and operational.

### Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| Ollama GB10 | 197.21 | Measured 2026-03-12 |
| Zerfoo GB10 (optimized) | 188.92 avg | 3-run avg 2026-03-12 |
| Zerfoo GB10 (initial GPU Q4) | 8.61 | Pre-optimization |
| Theoretical max (Q4 on GB10) | ~350-400 | 273 GB/s bandwidth ceiling |

---

## 9. Hand-off Notes

- **ADR-031:** docs/adr/031-openai-server-in-zerfoo.md -- server stays in Zerfoo serve/ package.
- **ADR-030:** docs/adr/030-ollama-performance-parity.md -- original performance strategy.
- **ADR-024:** docs/adr/024-cuda-graph-fused-kernels.md -- CUDA graph and fused kernel design.
- **ADR-025:** docs/adr/025-purego-cuda-bindings.md -- completed. Results in docs/design.md.
- **Existing serve/ package:** serve/server.go has OpenAI endpoints (chat, completions, models,
  SSE streaming, batch scheduling, speculative decoding). See ADR-031.
- **CUDA Graph code:** graph/cuda_graph.go (CUDAGraphExecutor), internal/cuda/runtime_purego.go
  (API wrappers), compute/engine.go (StreamProvider). Generator wiring is commented out in
  generate/generator.go lines 166-175.
- **D2H copy sites:** (1) compute/gpu_engine.go ~line 1242 (Gather indices.Data()),
  (2) tensor/gpu_storage.go TrySlice (used in GQA), (3) generate/tensor_cache.go ~line 124
  (appendGPU src.Data()).
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
- **Benchmark:** bench_tps -model /home/ndungu/models/gemma3-gguf/model.gguf -tokens 256
  -prompt 'The meaning of life is' -device cuda
- **Pre-commit hook:** rejects multi-directory commits.
- **purego pattern:** See internal/cuda/runtime_purego.go for the established pattern.
- **Zonnx:** Separate repo at github.com/zerfoo/zonnx. ONNX converter only. Must not import Zerfoo.

---

## 10. Archived Tasks

### From Prior Plan -- Completed and Trimmed

- E201 (T201.1, T201.2): Baseline measurements completed 2026-03-12. Preserved in design.md.
- E202 (T202.1, T202.2): Inference correctness fixed 2026-03-12. Preserved in design.md.
- T206.1: Fused SwiGLU kernel completed 2026-03-12. Preserved in design.md.
- T206.2: Fused Scale+Softmax kernel completed 2026-03-12. Preserved in design.md.
- S201.2.1: nsys profiling -- optional, not blocking. Can be done ad hoc.
- S202.2.1: Correctness regression test -- optional, correctness verified via bench_tps.
- T202.3: go vet post-correctness -- subsumed by per-task go vet.
- T207.1: CUDA graph API wrappers -- already implemented in runtime_purego.go.
- T209.1 (old): NVCC flags -- completed (commit d1ed26a).

### From Prior Plan -- Subsumed by Completed Work

- S88.3.1 Full kernel test suite -- subsumed by E102 (purego-only kernels).
- T89.2 Remove build tags from compute/ GPU files -- subsumed by E103.
- S89.3.1 Cross-platform build verification -- subsumed by E107.
- T94.1-T95.4 Track B performance tuning -- incorporated into E208, E209.
