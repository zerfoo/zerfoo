# Phase 25: FP16 Weight Dequantization -- Beat Ollama by +18%

## 1. Context

### Problem Statement

Zerfoo achieves 188 tok/s (Gemma 3 1B Q4_K_M, 512 tokens, CUDA graph) on DGX Spark
GB10. Ollama achieves 196 tok/s on the same hardware and model. We have publicly
claimed +18% faster than Ollama, which requires 231+ tok/s.

**Current pipeline:** GGUF Q5_0/Q4_K tensors are dequantized to float32 (4 bytes/weight)
and uploaded to GPU. cuBLAS SGEMM processes float32 weights. On Blackwell GB10 with
273 GB/s bandwidth, float32 weights consume ~2.4 GB per token for the full model.

**Proposed pipeline:** Dequantize to FP16 (2 bytes/weight) instead of float32. The GPU
engine already supports Float16Storage with cuBLAS FP16 GEMM using tensor cores on
Blackwell. FP16 halves memory bandwidth (1.2 GB per token vs 2.4 GB), and tensor
cores provide additional compute throughput.

**Why this works:** Phase 24 proved that custom GEMV kernels are 2-3x slower than cuBLAS
on Blackwell (50-89 tok/s vs 170 tok/s). cuBLAS is the right kernel path. The
optimization is reducing the DATA SIZE that cuBLAS reads, not changing the kernel.

### Key Discovery (Phase 24)

Phase 24 wrote native GEMV kernels for Q6_K, Q5_K, and Q5_0. All kernels compiled and
passed parity tests, but were 2-3x slower than cuBLAS SGEMM on Blackwell due to
cuBLAS's tensor core utilization and optimized memory access patterns. The kernels
remain in ztensor for future optimization. See ADR-040.

Phase 24 also discovered the "garbled output" bug was a corrupted model file (806 MB),
not a code bug. Ollama's model file (815 MB) produces coherent text with our code.

### Objectives

- O1: Dequantize Q5_0, Q4_K, Q5_K, Q6_K weights to FP16 in the GGUF loader.
- O2: Ensure Float16Storage flows through UploadWeights to GPU without conversion to F32.
- O3: Ensure PreUploadFrozenWeights does NOT re-convert FP16 to F32.
- O4: Achieve 231+ tok/s on Gemma 3 1B Q4_K_M (512 tokens) on DGX Spark GB10.
- O5: Verify output quality with the correct model file.

### Non-Goals

- Native quantized GEMV kernels (Phase 24 proved cuBLAS is faster on Blackwell).
- FP16 KV cache (separate optimization).
- New model architectures, training, CompileTraced.

### Constraints

- Pure Go, zero CGo.
- Changes span two repos: ztensor (PreUploadFrozenWeights fix) and zerfoo (GGUF loader).
- Float16Storage, NewFloat16StorageFromF32, and fp16MatMulNative already exist.
- All existing tests must continue to pass.
- The correct Gemma 3 model file is at ~/models/gemma3-q4km/model.gguf (815 MB, from Ollama).

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Gemma 3 1B 50t  | 172 tok/s | 220+ | DGX bench |
| Gemma 3 1B 256t | 186 tok/s | 231+ | DGX bench |
| Gemma 3 1B 512t | 188 tok/s | 231+ | DGX bench |
| vs Ollama (196) | -4% | +18% | Side-by-side DGX bench |
| Output quality | "capital-" (partial) | Coherent | Manual inspection |

### Bandwidth Analysis

```
GB10 LPDDR5x bandwidth: 273 GB/s
Gemma 3 1B weight reads (F32): ~2.4 GB/token -> max 114 tok/s
Gemma 3 1B weight reads (FP16): ~1.2 GB/token -> max 228 tok/s
Current 188 tok/s exceeds F32 limit -> GPU L2 cache + unified memory
FP16 theoretical max 228 tok/s -> +16% over Ollama 196 (close to +18%)
```

---

## 2. Scope and Deliverables

### In Scope

- GGUF loader: convert Q5_0, Q4_K, Q5_K, Q6_K dequant output to Float16Storage.
- ztensor UploadWeights: add FP16 raw bytes upload path for Float16Storage.
- ztensor PreUploadFrozenWeights: skip Float16Storage (do NOT convert to F32).
- DGX benchmark + quality verification.
- Update CLAUDE.md with verified benchmark.

### Out of Scope

- Native GEMV kernel optimization (Phase 24 kernels remain for future use).
- FP16 KV cache, new architectures, training.
- F32 tensor handling changes (157 F32 tensors stay F32).

---

## 3. Checkable Work Breakdown

### E1: GGUF Loader FP16 Dequantization

Convert quantized weight tensors to FP16 instead of F32 in the GGUF loader.
Use `tensor.NewFloat16StorageFromF32` after dequantizing to float32.

- [ ] T1.1 Convert Q5_0 dequant to FP16  Owner:  Est: 15m
  - File: `zerfoo/model/gguf/loader.go`, function `decodeQ5_0Tensor`
  - After dequantizing Q5_0 blocks to float32 data slice, convert:
    `fp16 := tensor.NewFloat16StorageFromF32(data)`
    `return tensor.NewWithStorage[float32](shape, fp16)`
  - Keep the existing Q5_0 block decoding logic (it produces accurate float32).
  - Acceptance: `go build ./...` passes. `go test ./model/gguf/ -race` passes.

- [ ] T1.2 Convert Q4_K dequant to FP16  Owner:  Est: 15m
  - File: `zerfoo/model/gguf/loader.go`, function `decodeQ4KTensor`
  - Current: dequant Q4_K to F32, re-quant to Q4_0. Change to: dequant Q4_K to F32,
    convert to FP16: `fp16 := tensor.NewFloat16StorageFromF32(f32)`
  - Remove the QuantizeQ4 call entirely.
  - Acceptance: `go build ./...` passes. `go test ./model/gguf/ -race` passes.

- [ ] T1.3 Convert Q5_K dequant to FP16  Owner:  Est: 10m
  - File: `zerfoo/model/gguf/loader.go`, function `decodeQ5KTensor`
  - Current: dequant to F32. Change to: dequant to F32 then NewFloat16StorageFromF32.
  - Acceptance: builds and tests pass.

- [ ] T1.4 Convert Q6_K dequant to FP16  Owner:  Est: 10m
  - File: `zerfoo/model/gguf/loader.go`, function `decodeQ6KTensor`
  - Same pattern as Q5_K.
  - Acceptance: builds and tests pass.

### E2: ztensor UploadWeights FP16 Path

Ensure UploadWeights uploads Float16Storage raw bytes to GPU instead of skipping.
Currently line 431 skips Float16Storage. Change to upload FP16 raw bytes and set
GPU pointer, similar to how BFloat16Storage is handled (lines 408-423).

- [ ] T2.1 Add Float16Storage upload to UploadWeights  Owner:  Est: 30m
  - File: `ztensor/compute/gpu_engine.go`, function UploadWeights
  - Replace the Float16Storage skip (line 431-432) with an upload path:
    ```
    if fs, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
        if fs.GPUPtr() != nil { continue }  // already on GPU
        rawBytes := fs.RawBytes()            // FP16 packed bytes
        devPtr := allocWeight(len(rawBytes))
        uploadBytes(devPtr, rawBytes)
        fs.SetGPUPtr(devPtr, len(rawBytes), deviceID)
        continue
    }
    ```
  - Check that Float16Storage has RawBytes(), GPUPtr(), SetGPUPtr() methods.
    If missing, add them (template from BFloat16Storage or Q4KStorage).
  - Acceptance: `go build ./...` passes in ztensor.

### E3: PreUploadFrozenWeights FP16 Skip

Prevent PreUploadFrozenWeights from converting Float16Storage to float32 via ToGPU.

- [ ] T3.1 Skip Float16Storage in PreUploadFrozenWeights  Owner:  Est: 15m
  - File: `ztensor/graph/compile.go`, function PreUploadFrozenWeights
  - Add check after the GPUStorage check (line 247):
    ```
    if _, ok := any(t.GetStorage()).(*tensor.Float16Storage); ok {
        continue  // FP16 weights uploaded by UploadWeights, used by fp16MatMulNative
    }
    ```
  - Acceptance: `go build ./...` and `go test ./graph/ -race` pass.

### E4: Float16Storage GPU Support (if missing)

Float16Storage may need GPU pointer fields for the upload path.

- [ ] T4.1 Add GPUPtr/SetGPUPtr/RawBytes to Float16Storage if missing  Owner:  Est: 20m
  - File: `ztensor/tensor/fp16_storage.go`
  - Check if Float16Storage already has: GPUPtr() (ptr, size, devID), SetGPUPtr(),
    RawBytes() []byte. If any are missing, add them.
  - Template: Q4KStorage or BFloat16Storage GPU pointer fields.
  - Acceptance: `go build ./...` and `go test ./tensor/ -race` pass.

### E5: Integration and Benchmark

- [ ] T5.1 Rebuild libkernels.so on DGX  Owner:  Est: 10m
  - No new kernels needed (cuBLAS handles FP16). Just rebuild to ensure consistency.
  - Pull latest code, vendor, build bench_tps binary.

- [ ] T5.2 Full test suite  Owner:  Est: 20m
  - Deps: T1.1-T1.4, T2.1, T3.1, T4.1
  - `go test ./... -race` in both ztensor and zerfoo.
  - Acceptance: zero failures.

- [ ] T5.3 DGX benchmark: throughput  Owner:  Est: 20m
  - Deps: T5.1, T5.2
  - bench_tps at 50, 256, 512 tokens with correct model file.
  - Ollama side-by-side comparison.
  - Acceptance: Zerfoo >= 231 tok/s at 256+ tokens.

- [ ] T5.4 DGX benchmark: output quality  Owner:  Est: 15m
  - Deps: T5.1, T5.2
  - Verify "The capital of France is" produces coherent continuation.
  - Acceptance: coherent English text.

- [ ] T5.5 Update CLAUDE.md  Owner:  Est: 10m
  - Deps: T5.3, T5.4
  - Update benchmark claim with verified FP16 numbers.
  - Acceptance: claim matches measured results.

---

## 4. Parallel Work

### Tracks

| Track | Tasks | Repo | Description |
|-------|-------|------|-------------|
| A: Loader | T1.1-T1.4 | zerfoo | FP16 dequant in GGUF loader |
| B: Upload | T2.1, T4.1 | ztensor | FP16 upload path |
| C: Compile | T3.1 | ztensor | PreUploadFrozenWeights skip |

All tracks are independent. One agent can handle all of Track A (same file).
Tracks B and C touch different files in ztensor.

### Maximum Parallelism

**Wave 1** (3 agents, all independent):
- Agent 1: T1.1, T1.2, T1.3, T1.4 (all loader changes, one file)
- Agent 2: T4.1, T2.1 (Float16Storage GPU support + UploadWeights)
- Agent 3: T3.1 (PreUploadFrozenWeights skip)

**Wave 2** (sequential, after merge):
T5.1, T5.2

**Wave 3** (after tests pass):
T5.3, T5.4, T5.5

---

## 5. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | FP16 precision loss causes output degradation | Low | Medium | FP16 has 10-bit mantissa; Q5_0 has 5-bit values. No precision loss for quantized source data. |
| R2 | cuBLAS FP16 GEMM slower than SGEMM on Blackwell | Low | High | Blackwell has 4th-gen tensor cores optimized for FP16. If slower, fall back to F32. |
| R3 | Float16Storage missing GPU pointer infrastructure | Medium | Low | Add GPUPtr/SetGPUPtr/RawBytes — straightforward, template from BFloat16Storage. |
| R4 | PreUploadFrozenWeights re-converts FP16 to F32 | Medium | High | T3.1 explicitly adds skip. Verified by benchmark. |
| R5 | 228 tok/s theoretical max insufficient for +18% claim | Medium | Medium | +18% over Ollama 196 = 231. Theoretical FP16 max = 228. Within 1%. GPU caching may push over. |

---

## 6. Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in both repos.
2. Tests pass: `go test ./... -race` in both repos.
3. Lint clean: `go vet ./...` in both repos.
4. DGX benchmark: throughput and quality verified.
5. CLAUDE.md updated with verified numbers only.

---

## 7. Progress Log

### 2026-03-17: Phase 25 plan created

**Change summary:** Replaced Phase 24 plan with Phase 25. Phase 24 (native GEMV kernels)
completed but kernels were 2-3x slower than cuBLAS on Blackwell. Kernel code remains
in ztensor. Loader reverted to Q4_0 re-quant. Phase 24 findings preserved in
docs/devlog.md and ADR-040.

Phase 25 approach: FP16 weight dequantization + cuBLAS FP16 GEMM with tensor cores.
2x bandwidth reduction targets 228 tok/s theoretical max (= +16% over Ollama).

---

## 8. Hand-off Notes

- **Repos**: ztensor at `/Users/dndungu/Code/zerfoo/ztensor/`, zerfoo at `../zerfoo/`
- **Key files**:
  - `zerfoo/model/gguf/loader.go` — dequant functions (decodeQ5_0Tensor, etc.)
  - `ztensor/compute/gpu_engine.go` — UploadWeights, FP16 MatMul dispatch
  - `ztensor/graph/compile.go` — PreUploadFrozenWeights
  - `ztensor/tensor/fp16_storage.go` — Float16Storage type
- **DGX**: `ssh ndungu@192.168.86.250`, `LD_LIBRARY_PATH=~/Code/zerfoo`
- **Model**: `~/models/gemma3-q4km/model.gguf` (815 MB, Ollama version, produces coherent text)
- **Ollama**: `/usr/local/bin/ollama`, model `gemma3:1b-it-q4_K_M`
- **Key API**: `tensor.NewFloat16StorageFromF32([]float32) *Float16Storage`
- **FP16 MatMul dispatch**: `gpu_engine.go:690-694` checks Float16Storage, calls fp16MatMulNative
