# Phase 24: Native GGUF GEMV Kernels -- Beat Ollama With Correct Output

## 1. Context

### Problem Statement

Zerfoo achieves 189 tok/s (Gemma 3 1B Q4_K_M, 512 tokens, CUDA graph, SDPA decode)
on DGX Spark GB10. Ollama achieves 201 tok/s on the same hardware and model. The
6% gap is caused by missing fused dequant-GEMV kernels for three GGUF quant types.

Gemma 3 1B Q4_K_M contains 340 tensors:

| Type | Count | Current Handling | Kernel Path |
|------|-------|-----------------|-------------|
| F32  | 157   | Native float32  | cuBLAS SGEMM (fast) |
| Q5_0 | 117   | Re-quant to Q4_0 | Fused Q4_0 GEMV (lossy) |
| Q4_K | 39    | Re-quant to Q4_0 | Fused Q4_0 GEMV (lossy) |
| Q8_0 | 14    | Native Q8 storage | Fused Q8 GEMM (fast) |
| Q6_K | 13    | Dequant to F32   | cuBLAS SGEMM (2x bandwidth) |

The Q5_0 (117 tensors) and Q4_K (39 tensors) re-quantization to Q4_0 is lossy --
it drops precision bits and causes garbled output text. The Q6_K dequantization to
float32 doubles memory bandwidth. All three types need fused dequant-GEMV kernels
that read quantized data directly, dequantize in registers, and accumulate the dot
product without touching global memory for intermediates.

The prior "234 tok/s beats Ollama" claim (ADR-033) was measured on Q4_0 ZMF format
where ALL weights used the fast Q4_0 GEMV kernel. That benchmark produced garbled
output. This phase delivers correct output AND competitive throughput on GGUF Q4_K_M.

### Objectives

- O1: Write fused dequant-GEMV CUDA kernels for Q6_K, Q5_K, and Q5_0.
- O2: Remove all lossy re-quantization from the GGUF loader.
- O3: Achieve 200+ tok/s on Gemma 3 1B Q4_K_M with coherent output text.
- O4: Beat Ollama (201 tok/s) on the same hardware and model.

### Non-Goals

- Q4_K GEMV optimization (existing kernel works, just 20% slower than Q4_0).
- FP16 KV cache (separate optimization, not needed for throughput parity).
- New model architectures or training features.
- CompileTraced fix (separate issue).

### Constraints

- Pure Go, zero CGo for the default build path.
- CUDA kernels compiled separately via nvcc, loaded at runtime via purego/dlopen.
- Kernels must target sm_121 (Blackwell GB10) with --use_fast_math.
- Changes span two repos: ztensor (GPU engine dispatch, storage types) and
  zerfoo (CUDA kernels, GGUF loader, purego bindings).
- All existing tests must continue to pass.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Gemma 3 1B 50t  | 170 tok/s | 200+ | DGX bench |
| Gemma 3 1B 256t | 187 tok/s | 200+ | DGX bench |
| Gemma 3 1B 512t | 189 tok/s | 200+ | DGX bench |
| vs Ollama | -6% | >= 0% | Side-by-side DGX bench |
| Output quality | Garbled | Coherent | Manual inspection |

---

## 2. Scope and Deliverables

### In Scope

- Three new CUDA kernels: gemv_q6k.cu, gemv_q5k.cu, gemv_q5_0.cu.
- Purego Go bindings for each kernel (purego + CGo build-tag variants).
- KernelRunner interface extension in ztensor gpuapi.
- GPU engine MatMul dispatch for Q5KStorage, Q6KStorage, Q5_0Storage.
- New Q5_0Storage type in ztensor tensor package.
- GGUF loader changes: remove re-quantization, use native storage.
- Parity tests comparing CUDA GEMV output to CPU dequant reference.
- DGX benchmark after each kernel lands.

### Out of Scope

- Q4_K GEMV performance tuning (current kernel is functional).
- cuBLAS fallback path changes (kept as-is for batch > 1).
- FP16 inference path.
- Model architectures beyond Gemma 3.

---

## 3. Checkable Work Breakdown

### E1: Q6_K Fused Dequant-GEMV Kernel (13 tensors in model)

Q6_K super-block: 210 bytes, 256 values. Layout:
  - [0:128] ql: low 4 bits of each 6-bit value
  - [128:192] qh: high 2 bits of each 6-bit value
  - [192:208] sc: int8 scales for 16 sub-blocks of 16 values
  - [208:210] fp16 d: super-block scale

Dequant: val = d * sc[sub_block] * ((ql & 0xF) | ((qh >> shift) & 3) << 4) - 32)

- [ ] T1.1 Write gemv_q6k.cu CUDA kernel  Owner:  Est: 60m
  - File: `ztensor/internal/cuda/kernels/gemv_q6k.cu`
  - Template: copy gemv_q4k.cu structure (shared mem x load, warp-per-row, shuffle reduce).
  - Adapt dequantization to Q6_K layout: 128 ql bytes + 64 qh bytes + 16 sc bytes + 2 d bytes.
  - Each lane processes strided super-blocks. Within each super-block, process two 128-element
    halves. For each half, decode 4 groups of 32 elements using ql low/high nibbles + qh bit pairs.
  - Block config: 4 warps/block (128 threads), shared mem = K * sizeof(float).
  - Acceptance: kernel compiles with nvcc -arch=sm_121 -O3 --use_fast_math.

- [ ] T1.2 Write gemv_q6k.h header  Owner:  Est: 10m
  - File: `ztensor/internal/cuda/kernels/gemv_q6k.h`
  - Declare `extern "C" cudaError_t gemv_q6k_f32(const void* W, const float* x, float* y, int M, int K, cudaStream_t stream)`.
  - Acceptance: header compiles.

- [ ] T1.3 Add purego Go binding for GemvQ6KF32  Owner:  Est: 20m
  - Files: `ztensor/internal/cuda/kernels/gemv_q6k_purego.go` (build tag !cuda),
    `ztensor/internal/cuda/kernels/gemv_q6k.go` (build tag cuda).
  - Add `launchGemvQ6KF32 uintptr` to klib struct in purego.go.
  - Add `{"gemv_q6k_f32", &k.launchGemvQ6KF32}` to symbol table.
  - Acceptance: `go build ./...` passes in ztensor.

- [ ] T1.4 Add GemvQ6KF32 to KernelRunner interface and CUDAKernels  Owner:  Est: 15m
  - Files: `ztensor/internal/gpuapi/cuda_kernels.go`, add method.
  - Add to KernelRunner interface if it exists, or to CUDAKernels struct.
  - Acceptance: `go build ./...` passes.

- [ ] T1.5 Add Q6K MatMul dispatch to GPU engine  Owner:  Est: 30m
  - File: `ztensor/compute/gpu_engine.go`
  - Add `matMulQ6K` and `matMulQ6KBWeight` methods (template from matMulQ4K).
  - In MatMul, check for `*tensor.Q6KStorage` on A or B, dispatch to new methods.
  - GEMV path (n==1): upload raw Q6K bytes to GPU, call GemvQ6KF32.
  - Non-GEMV path: dequant to F32 on GPU, then cuBLAS SGEMM (existing DequantQ4KF32
    pattern but for Q6K -- needs a DequantQ6KF32 kernel too, OR dequant on CPU and
    upload F32 for batch > 1 since it is rare in decode).
  - Acceptance: `go test ./compute/ -race` passes.

- [ ] T1.6 Write parity test for Q6K GEMV  Owner:  Est: 30m
  - File: `ztensor/internal/cuda/kernels/gemv_q6k_test.go`
  - Template: copy gemv_q4k_test.go pattern.
  - Generate random Q6K data, run GemvQ6KF32 on GPU, compare to CPU reference
    (DequantizeQ6K + float32 GEMV). Max abs error < 1e-3.
  - Test with M=256, K=1536 (Gemma dimensions).
  - Acceptance: test passes on DGX.

- [ ] T1.7 Update GGUF loader to use native Q6K storage  Owner:  Est: 15m
  - File: `zerfoo/model/gguf/loader.go`
  - Change `decodeQ6KTensor` to return `tensor.NewWithStorage[float32](shape, q6k)`
    instead of dequantizing to float32.
  - Acceptance: `go test ./model/gguf/ -race` passes.

- [ ] T1.8 Update Makefile and rebuild libkernels.so  Owner:  Est: 10m
  - File: `ztensor/internal/cuda/kernels/Makefile` (and zerfoo copy if separate).
  - Add gemv_q6k.cu to SRCS.
  - Acceptance: `make shared CUDA_ARCH=sm_121` succeeds.

### E2: Q5_K Fused Dequant-GEMV Kernel (0 in Gemma, present in other models)

Q5_K super-block: 176 bytes, 256 values. Layout:
  - [0:2] fp16 d (super-block scale)
  - [2:4] fp16 dmin (super-block min)
  - [4:16] 12 bytes packed 6-bit scales/mins for 8 sub-blocks (same as Q4_K)
  - [16:144] ql: 128 bytes low 4 bits
  - [144:176] qh: 32 bytes high 1 bit per value (256 bits)

Dequant: val = sc * ((ql & 0xF) | (qh_bit << 4)) - mn
Same structure as Q4_K but with an extra high bit from qh.

- [ ] T2.1 Write gemv_q5k.cu CUDA kernel  Owner:  Est: 60m
  - File: `ztensor/internal/cuda/kernels/gemv_q5k.cu`
  - Template: copy gemv_q4k.cu, add qh bit extraction.
  - For each group of 64 elements: read ql (32 bytes) + extract qh bits.
    Low nibbles + qh bit -> first 32 values. High nibbles + qh bit -> next 32.
  - Same block config as Q4K: 4 warps/block, shared mem x.
  - Acceptance: kernel compiles.

- [ ] T2.2 Write gemv_q5k.h header  Owner:  Est: 10m
- [ ] T2.3 Add purego Go binding for GemvQ5KF32  Owner:  Est: 20m
- [ ] T2.4 Add GemvQ5KF32 to KernelRunner and CUDAKernels  Owner:  Est: 15m

- [ ] T2.5 Add Q5K MatMul dispatch to GPU engine  Owner:  Est: 30m
  - File: `ztensor/compute/gpu_engine.go`
  - Add `matMulQ5K` / `matMulQ5KBWeight` (template from Q4K dispatch).
  - Acceptance: `go test ./compute/ -race` passes.

- [ ] T2.6 Write parity test for Q5K GEMV  Owner:  Est: 30m
  - Compare GPU GemvQ5KF32 to CPU DequantizeQ5K + GEMV. Max error < 1e-3.

- [ ] T2.7 Update GGUF loader: Q5_K uses native storage  Owner:  Est: 10m
  - decodeQ5KTensor returns NewWithStorage instead of dequant to F32.

- [ ] T2.8 Update Makefile for gemv_q5k.cu  Owner:  Est: 10m

### E3: Q5_0 Fused Dequant-GEMV Kernel (117 tensors -- BIGGEST IMPACT)

Q5_0 block: 22 bytes, 32 values. Layout:
  - [0:2] fp16 d (block scale)
  - [2:6] 4 bytes qh: high bits (32 bits, one per element)
  - [6:22] 16 bytes qs: packed nibbles (two 4-bit values per byte)

Dequant: val = d * ((qs_nibble | (qh_bit << 4)) - 16)
This is a simple format (no sub-blocks, no min). Simpler than Q4_K.

NOTE: Q5_0 uses 32-element blocks (not 256 super-blocks like K-quants).
The kernel needs a different blocking strategy: more blocks per row.

- [ ] T3.1 Add Q5_0Storage type to ztensor  Owner:  Est: 30m
  - File: `ztensor/tensor/quantized_q5_0.go` (new file)
  - Fields: raw []byte, len int. Block size = 32, block bytes = 22.
  - Methods: NewQ5_0StorageFromRaw, Dequantize, Len, Slice, Set (panic), DeviceType.
  - Dequantize matches the decodeQ5_0Tensor logic in zerfoo/model/gguf/loader.go.
  - Acceptance: `go test ./tensor/ -race` passes.

- [ ] T3.2 Write gemv_q5_0.cu CUDA kernel  Owner:  Est: 60m
  - File: `ztensor/internal/cuda/kernels/gemv_q5_0.cu`
  - Block size = 32 elements, 22 bytes per block.
  - Strategy: same warp-per-row approach but each lane processes more blocks
    (32 elements per block vs 256 for K-quants = 8x more blocks per row).
  - Shared mem x load, warp shuffle reduce.
  - Acceptance: kernel compiles.

- [ ] T3.3 Write gemv_q5_0.h header  Owner:  Est: 10m
- [ ] T3.4 Add purego Go binding for GemvQ5_0F32  Owner:  Est: 20m
- [ ] T3.5 Add GemvQ5_0F32 to KernelRunner and CUDAKernels  Owner:  Est: 15m

- [ ] T3.6 Add Q5_0 MatMul dispatch to GPU engine  Owner:  Est: 30m
  - Check for *tensor.Q5_0Storage on A or B.
  - GEMV path: upload raw bytes, call GemvQ5_0F32.
  - Non-GEMV: dequant to F32 on CPU, upload, cuBLAS SGEMM.
  - Acceptance: `go test ./compute/ -race` passes.

- [ ] T3.7 Write parity test for Q5_0 GEMV  Owner:  Est: 30m
  - Compare GPU GemvQ5_0F32 to CPU reference. Max error < 1e-3.

- [ ] T3.8 Update GGUF loader: Q5_0 uses native storage  Owner:  Est: 15m
  - File: `zerfoo/model/gguf/loader.go`
  - Change decodeQ5_0Tensor to create Q5_0Storage from raw bytes
    instead of dequantizing to float32 and re-quantizing to Q4_0.
  - Acceptance: `go test ./model/gguf/ -race` passes.

- [ ] T3.9 Update Makefile for gemv_q5_0.cu  Owner:  Est: 10m

### E4: Remove Q4_K Re-Quantization

Once native Q4_K GEMV dispatch exists (already in GPU engine), remove the lossy
Q4_K-to-Q4_0 re-quantization in the GGUF loader.

- [ ] T4.1 Update GGUF loader: Q4_K uses native storage  Owner:  Est: 15m
  - File: `zerfoo/model/gguf/loader.go`
  - Change decodeQ4KTensor to return NewWithStorage(shape, q4k) directly.
  - Remove the dequant-to-F32 + QuantizeQ4 roundtrip.
  - Acceptance: `go test ./model/gguf/ -race` passes.

### E5: Integration, Benchmark, and Quality Gate

- [ ] T5.1 Rebuild libkernels.so on DGX with all new kernels  Owner:  Est: 15m
  - Run `make clean && make shared CUDA_ARCH=sm_121` on DGX.
  - Copy libkernels.so to ~/Code/zerfoo/.
  - Acceptance: binary loads and runs without kernel-not-found errors.

- [ ] T5.2 Full test suite pass  Owner:  Est: 30m
  - Deps: all above.
  - Run `go test ./... -race` in both ztensor and zerfoo.
  - Acceptance: zero failures.

- [ ] T5.3 DGX benchmark: throughput  Owner:  Est: 30m
  - Deps: T5.1, T5.2.
  - Run bench_tps with Gemma 3 1B Q4_K_M at 50, 256, 512 tokens.
  - Run Ollama side-by-side for comparison.
  - Acceptance: Zerfoo >= 200 tok/s at 256+ tokens.

- [ ] T5.4 DGX benchmark: output quality  Owner:  Est: 30m
  - Deps: T5.1, T5.2.
  - Run bench_tps with "The capital of France is" and verify coherent output.
  - Compare CPU vs GPU output for consistency.
  - Acceptance: Output is coherent English text, not garbled.

- [ ] T5.5 Update CLAUDE.md benchmark claim  Owner:  Est: 10m
  - Deps: T5.3, T5.4.
  - Update the "234.30 tok/s" claim with the new verified number.
  - Only claim "faster than Ollama" if the benchmark proves it.
  - Acceptance: CLAUDE.md reflects measured, reproducible results.

---

## 4. Parallel Work

### Tracks

| Track | Tasks | Repo | Description |
|-------|-------|------|-------------|
| A: Q6_K kernel | T1.1-T1.8 | ztensor + zerfoo | Q6_K fused GEMV (13 tensors) |
| B: Q5_K kernel | T2.1-T2.8 | ztensor + zerfoo | Q5_K fused GEMV (other models) |
| C: Q5_0 kernel | T3.1-T3.9 | ztensor + zerfoo | Q5_0 fused GEMV (117 tensors) |
| D: Q4_K loader | T4.1 | zerfoo | Remove Q4_K re-quantization |

Tracks A, B, C, D are fully independent. Each agent works in a worktree.

### Maximum Parallelism

**Wave 1** (up to 10 tasks, all independent -- kernel + header + binding work):
T1.1, T1.2, T2.1, T2.2, T3.1, T3.2, T3.3, T4.1

NOTE: T1.1+T1.2, T2.1+T2.2, T3.2+T3.3 are header+kernel pairs that can be done
by the same agent. So effectively 4-5 agents in Wave 1, each handling a full kernel
(CUDA + header + purego binding + KernelRunner + engine dispatch + test + loader).

Recommended agent assignment:
- Agent 1: T1.1, T1.2, T1.3, T1.4, T1.5, T1.6, T1.7, T1.8 (full Q6_K stack)
- Agent 2: T2.1, T2.2, T2.3, T2.4, T2.5, T2.6, T2.7, T2.8 (full Q5_K stack)
- Agent 3: T3.1, T3.2, T3.3, T3.4, T3.5, T3.6, T3.7, T3.8, T3.9 (full Q5_0 stack)
- Agent 4: T4.1 (Q4_K loader fix -- quick, then help with integration)

**Wave 2** (after merge):
T5.1, T5.2

**Wave 3** (after tests pass):
T5.3, T5.4, T5.5

---

## 5. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | Kernel register pressure regresses throughput | Medium | High | Limit maxrregcount, profile occupancy. ADR-033 documents Q4K vectorization failure from register pressure. |
| R2 | Q5_0 small block size (32) causes low occupancy | Medium | Medium | Use more warps per block or multiple blocks per row. Profile occupancy on sm_121. |
| R3 | Native quant storage breaks existing float32 code paths | Low | High | Parity tests catch numerical divergence. Keep dequant fallback for non-GEMV paths. |
| R4 | Output still garbled after removing re-quantization | Low | High | The re-quantization was identified as root cause. If output is still bad, investigate tokenizer/embedding scaling next. |
| R5 | Kernels work but throughput < 200 tok/s | Medium | Medium | Profile to identify new bottleneck. The F32 tensors (157 of 340) still use cuBLAS SGEMM. |

---

## 6. Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in both repos.
2. Tests pass: `go test ./... -race` in both repos.
3. Lint clean: `go vet ./...` in both repos.
4. CUDA kernel compiles: `nvcc -arch=sm_121 -O3 --use_fast_math`.
5. Parity test: GPU GEMV matches CPU dequant + GEMV within 1e-3.
6. DGX benchmark for throughput and quality tasks.

---

## 7. Progress Log

### 2026-03-17: Phase 24 plan created

**Change summary:** Created Phase 24 plan for native GGUF GEMV kernels. Phase 23
completed (Go-side overhead optimizations, CUDA graph capture fix, FlashAttentionDecode
regression bisect and fix). Phase 23 tasks trimmed from plan -- stable knowledge
preserved in docs/devlog.md.

Key findings from Phase 23:
- FlashAttentionDecode was 15% slower than SDPA (bisected, reverted in d0fe532)
- CUDA graph capture works with SDPA path thanks to arena-safe allocation
- Go overhead is <2% of total step time -- kernel efficiency is the bottleneck
- The "234 tok/s beats Ollama" claim was on Q4_0 ZMF with garbled output

---

## 8. Hand-off Notes

- **Repos**: ztensor at `/Users/dndungu/Code/zerfoo/ztensor/`, zerfoo at `../zerfoo/`
- **Key files to modify**:
  - `ztensor/internal/cuda/kernels/` -- new .cu, .h, _purego.go files
  - `ztensor/internal/cuda/kernels/purego.go` -- add launch symbols to klib struct
  - `ztensor/internal/gpuapi/cuda_kernels.go` -- add methods to CUDAKernels
  - `ztensor/compute/gpu_engine.go` -- add MatMul dispatch for new storage types
  - `ztensor/tensor/` -- add Q5_0Storage type
  - `zerfoo/model/gguf/loader.go` -- remove re-quantization
  - `zerfoo/internal/cuda/kernels/Makefile` -- add new .cu to SRCS
- **DGX**: `ssh ndungu@192.168.86.250`, `LD_LIBRARY_PATH=~/Code/zerfoo`
- **Kernel rebuild**: `cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121`
- **Models**: Gemma Q4_K_M at `~/models/gemma3-q4km/model.gguf`
- **Ollama**: `/usr/local/bin/ollama`, model `gemma3:1b-it-q4_K_M`
- **Template kernel**: `ztensor/internal/cuda/kernels/gemv_q4k.cu` -- all new kernels follow this pattern
- **Template binding**: `ztensor/internal/cuda/kernels/gemv_q4k_purego.go`
- **Template dispatch**: `ztensor/compute/gpu_engine.go` matMulQ4K / matMulQ4KBWeight

---

## 9. Appendix

### GGUF Quant Block Formats

```
Q4_K: 144 bytes / 256 values (4-bit + 6-bit sub-block scales)
Q5_K: 176 bytes / 256 values (5-bit + 6-bit sub-block scales, same layout as Q4_K + qh)
Q6_K: 210 bytes / 256 values (6-bit + int8 sub-block scales)
Q5_0:  22 bytes /  32 values (5-bit, simple scale, no sub-blocks)
Q4_0:  18 bytes /  32 values (4-bit, simple scale, no sub-blocks)
```

### Kernel Pattern (from gemv_q4k.cu)

```
1. Load input vector x into shared memory (all threads cooperate)
2. One warp per row (4 warps per block = 128 threads)
3. Each lane processes strided super-blocks within its row
4. Within each super-block: decode scales, dequant in registers, FMA with x from shmem
5. Warp shuffle reduction produces final dot product
6. Lane 0 writes y[row]
```

### Bits Per Weight Comparison

```
Q4_0: 4.50 bits/weight (144 bytes / 256 values * 8)
Q4_K: 4.50 bits/weight (144 bytes / 256 values * 8)
Q5_0: 5.50 bits/weight ( 22 bytes /  32 values * 8)
Q5_K: 5.50 bits/weight (176 bytes / 256 values * 8)
Q6_K: 6.56 bits/weight (210 bytes / 256 values * 8)
F32: 32.00 bits/weight
```

Native quantized GEMV reads only the quantized bytes (4.5-6.6 bits/weight).
Dequant-to-F32 reads 32 bits/weight -- 5-7x more bandwidth.
