# Zerfoo Development Plan -- Ollama Performance Parity

## 1. Context

### Problem Statement

Zerfoo inference on DGX Spark GB10 produces degenerate output and runs at
0.44-12.84 tok/s depending on execution path. Ollama running open weights on the
same hardware achieves ~100 tok/s. The performance gap is 8-230x. Both CPU and
CUDA inference paths produce nonsensical tokens, indicating a correctness bug
that predates the ADR-025 build tag removal work.

Prior plan (ADR-025 runtime GPU detection) is complete. All build tags removed
from pure Go CUDA files, 16 op emitters added, megakernel verified on DGX Spark.
See docs/design.md for full ADR-025 results.

### Objectives

- O1: Fix inference correctness so Zerfoo produces coherent output matching
  reference implementations.
- O2: Match or exceed Ollama throughput (~100 tok/s) on DGX Spark GB10 for the
  same model.
- O3: Convert cuBLAS from CGo to purego for single-binary deployment.
- O4: Convert cuDNN from CGo to purego for single-binary deployment.
- O5: Convert TensorRT bindings to purego.
- O6: Convert CUTLASS flash attention to purego.
- O7: Convert ROCm backend to purego.
- O8: Convert OpenCL backend to purego.
- O9: Optimize megakernel or replace with CUDA graph + fused kernels approach.
- O10: Eliminate all CPU fallbacks during GPU inference (Transpose, Gather,
  broadcasting).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark available at ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: 273 GB/s LPDDR5x, Blackwell GPU (sm_121), 128GB unified memory.
- Theoretical max for 1.5GB Q4 model: ~182 tok/s.
- Ollama baseline must be measured precisely (model, quantization, prompt) before
  performance comparison is meaningful.
- Correctness must be fixed before performance optimization.
- internal/cublas/ and internal/cudnn/ currently require CGo (//go:build cuda).
- ROCm backend is ~80% feature-complete. OpenCL is ~40%.
- Pre-existing purego ccall SIGSEGV on linux/arm64 affects some packages.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Output correctness | Coherent text matching reference model | Compare 50 tokens with llama.cpp or Ollama output |
| Inference throughput | >= 100 tok/s on DGX Spark | bench_tps with Gemma 3 1B Q4 |
| Build simplicity | go build ./... (no CGo, no build tags) | Build on clean machine without CUDA toolkit |
| Test suite | All tests pass | go test ./... -race -timeout 120s |
| Multi-backend | ROCm and OpenCL functional | Basic inference on AMD and Intel GPUs |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D7 | Inference correctness fix | Degenerate output blocks all performance work |
| D8 | Ollama performance parity (>= 100 tok/s) | Core goal |
| D9 | GPU Transpose, Gather, broadcasting kernels | Eliminate CPU fallbacks (43% overhead) |
| D10 | Fused kernels (SwiGLU, Scale+Softmax, dequant+GEMV) | Reduce memory bandwidth |
| D11 | CUDA graph capture and replay | Eliminate per-op launch overhead |
| D12 | Megakernel investigation and fix | 30x slower than per-op needs diagnosis |
| D13 | cuBLAS purego conversion | Single-binary, no CGo toolchain |
| D14 | cuDNN purego conversion | Single-binary, no CGo toolchain |
| D15 | TensorRT purego conversion | Single-binary, no CGo toolchain |
| D16 | CUTLASS/flash attention purego conversion | Single-binary, no CGo toolchain |
| D17 | ROCm purego conversion | AMD GPU support without CGo |
| D18 | OpenCL purego conversion | Intel/other GPU support without CGo |
| D19 | Kernel optimization (NVCC flags, occupancy, shared memory) | Close gap to theoretical max |

### Out of Scope

- Training performance optimization (inference only).
- New model architecture support beyond what exists.
- Distributed inference (multi-GPU parallelism).
- Mobile or embedded deployment.

---

## 3. Checkable Work Breakdown

### E201: Establish Performance Baseline

- [x] T201.1 Measure Ollama on DGX Spark  Owner: TBD  Est: 1h  Done: 2026-03-12
  - SSH to DGX Spark. Run Ollama with the same model Zerfoo uses (Gemma 3 1B Q4
    or equivalent). Record exact model name, quantization, prompt, and tok/s.
  - Run 3 times, report median.
  - Acceptance: Exact Ollama tok/s documented with model and config details.
  - Dependencies: none.

- [x] T201.2 Measure Zerfoo all paths on DGX Spark  Owner: TBD  Est: 1h  Done: 2026-03-12
  - Run bench_tps with: (a) CPU, (b) CUDA per-op, (c) megakernel.
  - Use same model and prompt as Ollama measurement.
  - Record tok/s for each path. Run 3 times each, report median.
  - Acceptance: Zerfoo tok/s for all 3 paths documented alongside Ollama.
  - Dependencies: T201.1.

- [ ] S201.2.1 Profile Zerfoo CUDA path with nsys  Owner: TBD  Est: 1h
  - Run nsys profile on bench_tps CUDA path.
  - Identify top 5 time consumers (kernel launches, memcpy, synchronization).
  - Acceptance: nsys report saved, top 5 bottlenecks listed with percentages.
  - Dependencies: T201.2.

### E202: Fix Inference Correctness

- [x] T202.1 Diagnose degenerate output root cause  Owner: TBD  Est: 3h  Done: 2026-03-12
  - Compare Zerfoo weight loading with llama.cpp for the same GGUF file.
  - Check: (a) weight tensor shapes and strides, (b) Q4 dequantization formula,
    (c) RoPE frequency computation, (d) attention mask construction,
    (e) logit computation before sampling.
  - Dump intermediate activations at each transformer layer and compare with
    a reference implementation.
  - Acceptance: Root cause identified and documented.
  - Dependencies: T201.2.

- [x] T202.2 Fix the correctness bug  Owner: TBD  Est: 4h  Done: 2026-03-12
  - Implement the fix identified in T202.1.
  - Acceptance: 50 generated tokens match reference output (Ollama or llama.cpp)
    for the same prompt and temperature=0.
  - Dependencies: T202.1.

- [ ] S202.2.1 Correctness regression test  Owner: TBD  Est: 1h
  - Add a Go test that loads a small model (or test fixture), generates tokens
    with temperature=0, and asserts output matches expected string.
  - Acceptance: go test passes. Test is deterministic.
  - Dependencies: T202.2.

- [ ] T202.3 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: T202.2.

### E203: GPU Transpose Kernel

- [ ] T203.1 Write CUDA transpose kernel for 2D/3D/4D tensors  Owner: TBD  Est: 3h
  - Add transpose.cu or extend existing kernel file.
  - Support arbitrary axis permutations for 2D, 3D, and 4D tensors.
  - Use shared memory tiling for coalesced reads and writes.
  - Acceptance: Kernel compiles for sm_75 and sm_121.
  - Dependencies: none.

- [ ] T203.2 Wire GPU Transpose into GPUEngine  Owner: TBD  Est: 2h
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

- [ ] T204.1 Write CUDA gather kernel for embedding lookups  Owner: TBD  Est: 2h
  - Add gather.cu kernel or extend existing. Support int32 and int64 indices.
  - Each thread block handles one index, copies embedding_dim elements.
  - Acceptance: Kernel compiles for sm_75 and sm_121.
  - Dependencies: none.

- [ ] T204.2 Wire GPU Gather into GPUEngine  Owner: TBD  Est: 2h
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

- [ ] T205.1 Extend CUDA element-wise kernels for broadcasting  Owner: TBD  Est: 3h
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

### E206: Fused CUDA Kernels (ADR-024)

- [x] T206.1 Fused SwiGLU kernel  Owner: TBD  Est: 3h  Done: 2026-03-12
  - Single kernel: gate * silu(up). Saves 2 launches and 1 intermediate per FFN.
  - Acceptance: Kernel compiles, parity test passes (max rel error < 1e-5).
  - Dependencies: none.

- [x] T206.2 Fused Scale+Softmax kernel  Owner: TBD  Est: 3h  Done: 2026-03-12
  - Single kernel: scale attention scores by 1/sqrt(d) then softmax.
  - Use shared memory for max/sum reductions.
  - Acceptance: Kernel compiles, parity test passes.
  - Dependencies: none.

- [ ] T206.3 Fused dequant+GEMV kernel for Q4 decode  Owner: TBD  Est: 4h
  - Read Q4 blocks, dequantize in registers, multiply by activation vector,
    accumulate in F32. Single-token decode (batch=1).
  - This is the highest-impact single optimization (dequant is largest kernel).
  - Acceptance: Kernel compiles, output matches unfused path (max rel error < 1e-4).
  - Dependencies: none.

- [ ] T206.4 Wire fused kernels into GPUEngine  Owner: TBD  Est: 2h
  - Add dispatch logic to detect fusable patterns and call fused kernels.
  - Fall back to unfused when patterns do not match.
  - Acceptance: bench_tps uses fused kernels (verify via logging).
  - Dependencies: T206.1, T206.2, T206.3.

- [ ] S206.4.1 Fused kernel integration test  Owner: TBD  Est: 1h
  - End-to-end test: run inference with fused kernels enabled, compare output
    with unfused path. Tokens must match.
  - Acceptance: go test passes with -race.
  - Dependencies: T206.4.

- [ ] T206.5 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S206.4.1.

### E207: CUDA Graph Capture and Replay (ADR-024)

- [ ] T207.1 Add CUDA graph API wrappers to internal/cuda/  Owner: TBD  Est: 2h
  - Wrap cudaStreamBeginCapture, cudaStreamEndCapture, cudaGraphInstantiate,
    cudaGraphLaunch, cudaGraphExecUpdate via purego.
  - Acceptance: Wrappers compile and link on DGX Spark.
  - Dependencies: none.

- [ ] T207.2 Pre-allocate fixed buffer layout in ExecutionPlan  Owner: TBD  Est: 3h
  - At compile time, compute shape of every intermediate tensor.
  - Allocate one contiguous GPU buffer with fixed offsets per slot.
  - CUDA graph capture requires fixed memory addresses.
  - Acceptance: ExecutionPlan uses pre-allocated buffers. No per-op alloc/free.
  - Dependencies: T207.1.

- [ ] T207.3 Implement graph capture on first decode token  Owner: TBD  Est: 3h
  - Record the decode forward pass via cudaStreamBeginCapture on first token.
  - Instantiate graph. Replay for subsequent tokens.
  - Re-capture when sequence length crosses a threshold.
  - Acceptance: Second token onward uses graph replay (verify via nsys).
  - Dependencies: T207.2.

- [ ] S207.3.1 CUDA graph correctness test  Owner: TBD  Est: 1h
  - Compare output with and without CUDA graph capture for 10 tokens.
  - Tokens must be identical.
  - Acceptance: go test passes.
  - Dependencies: T207.3.

- [ ] T207.4 Run go vet on modified packages  Owner: TBD  Est: 15m
  - Dependencies: S207.3.1.

### E208: Megakernel Performance Investigation

- [ ] T208.1 Profile megakernel with nsys on DGX Spark  Owner: TBD  Est: 2h
  - Run nsys on bench_tps with megakernel enabled.
  - Identify: kernel duration, occupancy, register usage, shared memory usage.
  - Compare with per-op kernels for the same operations.
  - Acceptance: Profile report with root cause for 30x gap.
  - Dependencies: E201.

- [ ] T208.2 Fix or redesign megakernel based on profile  Owner: TBD  Est: 4h
  - If fixable: optimize the code generator output (tiling, shared memory, etc.).
  - If not fixable: document why and rely on CUDA graph + fused kernels instead.
  - Acceptance: Megakernel >= 10 tok/s, OR documented decision to abandon megakernel
    in favor of CUDA graph approach.
  - Dependencies: T208.1.

- [ ] S208.2.1 Megakernel benchmark comparison  Owner: TBD  Est: 30m
  - Compare megakernel tok/s with per-op tok/s and CUDA graph tok/s.
  - Acceptance: Results documented.
  - Dependencies: T208.2.

- [ ] T208.3 Run go vet on internal/codegen/  Owner: TBD  Est: 15m
  - Dependencies: T208.2.

### E209: Kernel Optimization

- [ ] T209.1 Optimize NVCC compilation flags  Owner: TBD  Est: 1h
  - Add -O3, --use_fast_math to Makefile for sm_121.
  - Add --ptxas-options=-v to measure register and shared memory usage.
  - Compare tok/s before and after.
  - Acceptance: Makefile updated. Benchmark delta documented.
  - Dependencies: none.

- [ ] T209.2 Tune register pressure and occupancy  Owner: TBD  Est: 3h
  - Use --maxrregcount to limit registers per thread.
  - Profile occupancy with nsys for each kernel.
  - Target: >= 50% occupancy for compute-bound kernels.
  - Acceptance: Occupancy report. Adjustments applied where beneficial.
  - Dependencies: T209.1.

- [ ] T209.3 Optimize shared memory usage in attention and reduction kernels  Owner: TBD  Est: 2h
  - Flash attention: tune BLOCK_SIZE for sm_121 shared memory capacity.
  - Reduction kernels: use warp shuffle instead of shared memory where possible.
  - Acceptance: Kernels pass parity tests. Benchmark improvement documented.
  - Dependencies: T209.1.

- [ ] S209.3.1 Full kernel benchmark suite  Owner: TBD  Est: 1h
  - Benchmark each kernel individually: elementwise, flash_attention, gemm_q4,
    rmsnorm, transpose, gather.
  - Compare with pre-optimization baselines.
  - Acceptance: Per-kernel benchmark results documented.
  - Dependencies: T209.2, T209.3.

- [ ] T209.4 Run go vet on internal/cuda/kernels/  Owner: TBD  Est: 15m
  - Dependencies: S209.3.1.

### E210: cuBLAS Purego Conversion

- [ ] T210.1 Create purego wrappers for cuBLAS API  Owner: TBD  Est: 4h
  - Wrap via purego dlopen of libcublas.so: cublasCreate, cublasDestroy,
    cublasSetStream, cublasSgemm, cublasGemmEx.
  - Match existing internal/cublas/ API surface.
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
  - cuDNN API is larger than cuBLAS. Focus on operations listed in design.md
    section 4.9.
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
    (trt_capi.h/cpp compiled to libtrt_capi.a) provides a flat C interface.
  - Create purego wrappers that dlopen libtrt_capi.so (change from .a to .so).
  - Update Makefile to produce shared library.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T212.2 Replace CGo tensorrt.go with purego implementation  Owner: TBD  Est: 2h
  - Remove //go:build cuda tag.
  - Add tensorrt.Available() runtime guard.
  - Acceptance: go build ./internal/tensorrt/... without -tags cuda.
  - Dependencies: T212.1.

- [ ] T212.3 Update inference/tensorrt_*.go to remove build tags  Owner: TBD  Est: 1h
  - Remove //go:build cuda from tensorrt_cache.go, tensorrt_convert.go,
    tensorrt_pipeline.go.
  - Add runtime Available() guards.
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
  - Convert to purego dlopen of libkernels.so (flash attention is already
    compiled into this shared library).
  - Remove cutlass build tag requirement.
  - Acceptance: Flash attention dispatches via purego. No CGo.
  - Dependencies: none.

- [ ] T213.2 Update layers/attention/flash_cuda.go  Owner: TBD  Est: 1h
  - Remove //go:build cuda && cutlass tag.
  - Add runtime cuda.Available() guard.
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
  - Wrap via purego dlopen of libamdhip64.so: hipMalloc, hipFree, hipMemcpy,
    hipStreamCreate, hipStreamSynchronize, hipModuleLoad, hipModuleLaunchKernel.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T214.2 Create purego wrappers for rocBLAS API  Owner: TBD  Est: 3h
  - Wrap via purego dlopen of librocblas.so: rocblas_create_handle,
    rocblas_destroy_handle, rocblas_sgemm.
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
  - Wrap via purego dlopen of libOpenCL.so: clGetPlatformIDs, clGetDeviceIDs,
    clCreateContext, clCreateCommandQueue, clCreateBuffer, clEnqueueWriteBuffer,
    clEnqueueReadBuffer, clCreateProgramWithSource, clBuildProgram,
    clCreateKernel, clSetKernelArg, clEnqueueNDRangeKernel.
  - Acceptance: Wrappers compile without CGo.
  - Dependencies: none.

- [ ] T215.2 Replace CGo OpenCL runtime with purego  Owner: TBD  Est: 2h
  - Update internal/opencl/runtime.go.
  - Remove //go:build opencl tag. Add opencl.Available() runtime guard.
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

### E216: Performance Verification and Comparison

- [ ] T216.1 Run bench_tps on DGX Spark with all optimizations  Owner: TBD  Est: 1h
  - Test all paths: per-op, fused kernels, CUDA graph, megakernel.
  - Record tok/s for each. Compare with E201 baselines.
  - Acceptance: At least one path achieves >= 100 tok/s.
  - Dependencies: E202, E203, E204, E205, E206, E207, E208, E209.

- [ ] T216.2 Compare Zerfoo vs Ollama output quality  Owner: TBD  Est: 1h
  - Same prompt, same model, temperature=0.
  - Compare first 50 tokens.
  - Acceptance: Output is coherent and comparable to Ollama.
  - Dependencies: T216.1.

- [ ] S216.2.1 Performance and correctness report  Owner: TBD  Est: 30m
  - Document all benchmark results in a table.
  - Include: Zerfoo (all paths), Ollama, llama.cpp (if available).
  - Acceptance: Report written to docs/QUALITY.md.
  - Dependencies: T216.2.

- [ ] T216.3 Verify go build ./... without any build tags  Owner: TBD  Est: 30m
  - Build on macOS (no GPU) and DGX Spark (GPU).
  - All packages must compile without -tags cuda, -tags rocm, -tags opencl.
  - Acceptance: go build ./... passes on both platforms.
  - Dependencies: E210, E211, E212, E213, E214, E215.

- [ ] T216.4 Run full test suite  Owner: TBD  Est: 1h
  - go test ./... -race -timeout 120s on DGX Spark.
  - Acceptance: All tests pass (pre-existing failures documented separately).
  - Dependencies: T216.3.

- [ ] T216.5 Run go vet on all packages  Owner: TBD  Est: 15m
  - Dependencies: T216.4.

---

## 4. Parallel Work

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Correctness | E201, E202 | Sequential. Must complete before performance work is meaningful. |
| Track B: GPU Residency | E203, E204, E205 | Parallel (different packages: compute/, kernels/). Depends on E202. |
| Track C: Kernel Fusion | E206 | Independent. Can start after E202. |
| Track D: CUDA Graph | E207 | Depends on E203-E205 (needs fixed buffer layout). |
| Track E: Megakernel | E208 | Depends on E201 (needs baseline). Can run parallel with Track B/C. |
| Track F: Kernel Optimization | E209 | Independent. Can start immediately. |
| Track G: cuBLAS Purego | E210 | Independent. Can run parallel with Tracks B-F. |
| Track H: cuDNN Purego | E211 | Independent. Can run parallel with Track G. |
| Track I: TensorRT Purego | E212 | Independent. Can run parallel with Tracks G-H. |
| Track J: CUTLASS Purego | E213 | Independent. Can run parallel with Tracks G-I. |
| Track K: ROCm Purego | E214 | Independent. Can run parallel with Tracks G-J. |
| Track L: OpenCL Purego | E215 | Independent. Can run parallel with Tracks G-K. |
| Track M: Verification | E216 | Sync point: depends on all other tracks. |

Sync points:
- After Track A (E202): Tracks B, C, E, F unblock.
- After Tracks B + C + D + E + F: Performance verification T216.1.
- After Tracks G-L: Build verification T216.3.
- After all: Final verification E216.

Maximum parallelism:
- Wave 1: E201 (baseline) + E209 (kernel optimization, no deps) + E210-E215 (purego conversions, independent)
- Wave 2: E202 (correctness, depends on E201)
- Wave 3: E203 + E204 + E205 + E206 + E208 (parallel, depends on E202)
- Wave 4: E207 (CUDA graph, depends on E203-E205)
- Wave 5: E216 (verification, depends on all)

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M65: Baseline measured | E201 | Ollama and Zerfoo tok/s documented with identical test conditions |
| M66: Correct output | E202 | 50 tokens match reference implementation |
| M67: GPU-resident inference | E203, E204, E205 | No CPU fallbacks during decode loop |
| M68: Fused + graph | E206, E207 | CUDA graph replay with fused kernels operational |
| M69: 100 tok/s | E208, E209, T216.1 | bench_tps >= 100 tok/s on DGX Spark |
| M70: Single binary | E210-E215 | go build ./... without any build tags or CGo |
| M71: All verified | E216 | Full test suite passes, benchmark report complete |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R201 | Degenerate output is a fundamental architecture bug, not a simple fix | Blocks all work | Medium | Methodical layer-by-layer activation comparison with reference |
| R202 | 100 tok/s unreachable without rewriting core inference loop | Project goal unmet | Medium | CUDA graph + fused kernels proven approach (llama.cpp uses same). Accept 50+ tok/s as interim target. |
| R203 | purego cuBLAS/cuDNN slower than CGo | Performance regression | Low | purego overhead ~100ns/call. For large GEMM this is < 0.1%. Benchmark before/after. |
| R204 | CUDA graph capture fails with dynamic shapes | Cannot use CUDA graph for variable-length sequences | Medium | Re-capture at sequence length thresholds. Use fixed-size buckets. |
| R205 | ROCm/OpenCL testing blocked by hardware access | Cannot verify backends | High | Focus on CUDA first. ROCm/OpenCL are stretch goals. Test in CI with rented GPU instances. |
| R206 | Megakernel fundamentally limited by register pressure | Cannot fuse entire decode | High | Accept partial fusion. Use CUDA graph for remaining ops. Profile with --ptxas-options=-v. |
| R207 | TensorRT C shim changes break ABI | purego wrappers crash | Medium | Pin TensorRT version. Add version check in Available(). |
| R208 | Ollama tok/s may include prompt caching or batching that inflates number | Unfair comparison | Medium | Measure with identical conditions: single prompt, no cache, batch=1 |
| R92 | Register pressure: hidden_dim=2048 | Must tile, slower | High | Profile with nvcc --ptxas-options=-v |
| R95 | KV cache reads limit bandwidth | Cannot reach max | High | Focus on short contexts (<512) |

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

### Change Summary -- 2026-03-12

95% Ollama performance target achieved: 188.01 tok/s avg vs 187.35 target on
DGX Spark GB10. Ollama measured at 197.21 tok/s.

Tasks completed:
- E201 (T201.1, T201.2): Ollama and Zerfoo baselines measured on DGX Spark.
- E202 (T202.1, T202.2): Inference correctness fixed; output is now coherent.
- T206.1: Fused SwiGLU kernel (commit c3835ad merged gate+up).
- T206.2: Fused Scale+Softmax kernel (part of SDPA path).

Additional performance work completed (not mapped to specific plan tasks):
- Fused QK RMSNorm+RoPE kernel (commit 42f4008).
- Fused post-FFN RMSNorm+residual Add kernel (commit 6b22b47).
- Zero-copy Q+K view avoiding Concat (commit 27bf4d3).
- Arena allocator eliminating cudaMalloc (commit 33b0dee).
- Pre-allocated KV cache buffers (commit 7e80e21).
- GQA KV head broadcast eliminating Repeat (commit e92a04a).
- MatMulTransposeB via cuBLAS SgemmNT (commits 74cac33, bb5e5fd).
- cublasSgemmStridedBatched for batched attention (commit 2bbbeb1).

### Change Summary -- 2026-03-11

New plan created for Ollama performance parity per ADR-030.

Scope expanded to include all former non-goals:
- Track B performance tuning (E206-E209) brought in scope.
- cuBLAS purego conversion (E210) brought in scope.
- cuDNN purego conversion (E211) brought in scope.
- TensorRT purego conversion (E212) brought in scope.
- CUTLASS flash attention purego conversion (E213) brought in scope.
- ROCm purego conversion (E214) brought in scope.
- OpenCL purego conversion (E215) brought in scope.

Completed ADR-025 epics (E101-E108) trimmed from plan and preserved in
docs/design.md under "ADR-025 Implementation Complete" section.

Prior pending Track A tasks (S88.3.1, T89.2, S89.3.1) archived as subsumed
by completed E102, E103, E107 respectively.

Prior Track B tasks (T94.1-T95.4) incorporated into new epics E208 (megakernel
investigation) and E209 (kernel optimization).

Created ADR: docs/adr/030-ollama-performance-parity.md.

### Performance Baselines (from prior plan)

| Config | tok/s | Source |
|--------|-------|--------|
| GPU F32 (non-megakernel) | 12.84 | S100.1.1 DGX Spark (2026-03-11) |
| GPU Q4 (non-megakernel) | 8.61 | S100.1.1 DGX Spark (2026-03-11) |
| CPU ARM64 (post Track D) | 8.15 median | Phase 34 Track D |
| CUDA per-op plan.Run() | 2.22 | T108.2 DGX Spark (2026-03-12) |
| CPU plan.Run() | 5.71 | T108.2 DGX Spark (2026-03-12) |
| Megakernel (falls back) | 0.44 | T108.1 DGX Spark (2026-03-11) |
| Ollama GB10 | 197.21 | E201 measured on DGX Spark (2026-03-12) |
| **Zerfoo GB10 (optimized)** | **188.01 avg** | **95% target achieved (2026-03-12)** |

---

## 9. Hand-off Notes

- **ADR-025:** docs/adr/025-purego-cuda-bindings.md -- completed. Results in docs/design.md.
- **ADR-030:** docs/adr/030-ollama-performance-parity.md -- strategy for this plan.
- **ADR-024:** docs/adr/024-cuda-graph-fused-kernels.md -- CUDA graph and fused kernel design.
- **ADR-026:** docs/adr/026-megakernel-decode.md -- megakernel architecture.
- **ADR-022:** docs/adr/022-gpu-first-inference-pipeline.md -- GPU residency strategy.
- **PR workflow:** PRs go to zerfoo/zerfoo (upstream), not dndungu/zerfoo.
  Use `gh pr create --repo zerfoo/zerfoo --head dndungu:<branch>`.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Pre-commit hook:** rejects multi-directory commits.
- **Ollama on DGX Spark:** Running open weights. Exact model and config must be
  measured in E201 before performance comparison.
- **purego pattern:** See internal/cuda/runtime.go for the established purego
  dlopen pattern. All new purego conversions should follow the same approach:
  dlopen the .so, dlsym each function, wrap in Go function with runtime guard.
- **Build tag removal pattern:** See ADR-025 results in docs/design.md. Remove
  //go:build tag, add Available() runtime check, use gpuapi factory registration.

---

## 10. Archived Tasks

### From Prior Plan -- Subsumed by Completed Work

- S88.3.1 Full kernel test suite -- subsumed by E102 (purego-only kernels).
- T89.2 Remove build tags from compute/ GPU files -- subsumed by E103.
- S89.3.1 Cross-platform build verification -- subsumed by E107.
- T94.1-T95.4 Track B performance tuning -- incorporated into E208, E209, E216.
