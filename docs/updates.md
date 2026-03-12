# S100.1.1 DGX Spark Integration Test Results

Date: 2026-03-11

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10)
- **OS**: Linux 6.17.0-1008-nvidia aarch64
- **Go**: 1.25.0 linux/arm64
- **CUDA**: 13.0 (/usr/local/cuda)
- **Code**: upstream/main at commit 765108e (Merge PR #45 feat/neon-softmax)
- **Build**: `go build -tags cuda` with CGO_CFLAGS/CGO_LDFLAGS pointing to
  /usr/local/cuda

## Performance Results

| Model | Device | Tokens | tok/s | Megakernel Log? |
|-------|--------|--------|-------|-----------------|
| gemma3 (F32) | cuda | 64 | 12.84 | NO |
| gemma3 (F32) | cuda | 16 | 12.19 | NO |
| gemma3-q4 | cuda | 64 | 8.61 | NO |
| gemma3-q4 | cpu | 16 | 5.82 | NO |

### Baselines (from plan.md)

| Config | tok/s |
|--------|-------|
| CPU ARM64 (post Track D) | 8.15 median |
| GPU cuda (previous) | 10.32 peak / 7.78 median |

## Findings

### 1. Megakernel Did Not Fire

The "megakernel: compiled and loaded" log message never appeared.
`tryCompileMegakernel` (generate/megakernel.go:21) is called at
generate/generator.go:152 but silently fails. All error paths in
`tryCompileMegakernel` return without logging, making it impossible to
determine from output alone which step failed:

- `codegen.CheckSupport` (unsupported ops)
- `codegen.EmitMegakernel` (source generation)
- `codegen.CachedCompile` (nvcc compilation)
- `codegen.LoadMegakernel` (dlopen)

The most likely failure point is `codegen.CheckSupport`, which probably finds
unsupported ops in the Gemma 3 execution plan (KV cache ops, rotary
embeddings, or attention ops). This aligns with T100.2 (GPU KV cache wiring)
being listed as a prerequisite.

### 2. GPU Throughput Improved

The F32 model at 12.84 tok/s exceeds the previous baseline of 10.32 peak.
This improvement comes from the regular (non-megakernel) GPU execution path.

### 3. Output Quality Issues

Both models produce gibberish/repetitive output on CPU and GPU. The F32 model
repeats "land" indefinitely. The Q4 model outputs random tokens. This may
indicate model or quantization issues unrelated to the megakernel path.

### 4. Q4 vs F32 Performance Gap

Q4 on GPU (8.61 tok/s) is slower than F32 on GPU (12.84 tok/s). This is
unexpected and may indicate the Q4 kernel path is not GPU-optimized.

## Recommendation

Add diagnostic logging to `tryCompileMegakernel` at each failure point so the
exact failure cause can be identified. Example:

```go
unsupported := codegen.CheckSupport(instructions)
if len(unsupported) > 0 {
    log.Printf("megakernel: %d unsupported ops: %v", len(unsupported), unsupported)
    return
}
```

T100.2 (GPU KV cache wiring) is likely required before the megakernel can
fire on a real model.

## Quality Gate Assessment

| Gate | Status |
|------|--------|
| "megakernel: compiled and loaded" appears | FAIL |
| bench_tps runs on DGX Spark | PASS |
| Performance baseline recorded | PASS |

---

# S100.2.1 KV Cache Integration Test Results

Date: 2026-03-11

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10)
- **OS**: Linux 6.17.0-1008-nvidia aarch64
- **Go**: 1.26.0 linux/arm64
- **CUDA**: 13.0 (/usr/local/cuda)
- **Code**: upstream/main at commit 17b0e8a
- **Build**: `go build -tags cuda` with CGO_CFLAGS/CGO_LDFLAGS pointing to
  /usr/local/cuda

## Build Fixes Required

### 1. Missing runner_stub.go methods (commit 2faa5b2)

T100.2 added `SetKVCache()` and `HasKVCache()` to `MegakernelRunner` in
`runner.go` (`//go:build !cuda`), but the corresponding `runner_stub.go`
(`//go:build cuda`) was not updated. Build failed with:

```
runner.SetKVCache undefined (type *codegen.MegakernelRunner has no field or method SetKVCache)
```

Fix: Added `SetKVCache` and `HasKVCache` stubs to `runner_stub.go`.

### 2. Purego/CGo linker conflict (commit 17b0e8a)

T87.3 added `cgo_import_dynamic` directives in `purego_linux_arm64.go` to
import dlopen/dlsym from libdl.so.2 via assembly trampolines. When building
with `-tags cuda`, other CGo files activate, and the Go linker cannot handle
`SDYNIMPORT` relocations alongside the external (CGo) linker:

```
internal/cuda.libc_dlopen_trampoline: unhandled relocation for libc_dlopen
(type 65 (SDYNIMPORT) rtype 9 (R_CALLARM64))
```

Fix: Split platform implementation into two files:
- `purego_linux_arm64.go` (`!cuda`): zero-overhead asm trampolines
- `purego_linux_arm64_cgo.go` (`cuda`): CGo-based dlopen/dlsym/ccall

## Performance Results

| Model | Device | Tokens | tok/s | Megakernel? |
|-------|--------|--------|-------|-------------|
| gemma3 (F32) | cuda | 16 | 11.81 | NO |
| gemma3 (F32) | cuda | 50 | 11.54 | NO |
| gemma3-q4 | cuda | 50 | 8.98 | NO |

### Comparison with S100.1.1

| Model | S100.1.1 tok/s | S100.2.1 tok/s | Delta |
|-------|----------------|----------------|-------|
| gemma3 (F32) | 12.84 (64 tok) | 11.54 (50 tok) | -10% |
| gemma3-q4 | 8.61 (64 tok) | 8.98 (50 tok) | +4% |

Small variance is expected; different token counts and Go version (1.25 vs 1.26).

## Findings

### 1. Megakernel Did Not Fire — 16 Unsupported Ops Identified

`codegen.CheckSupport` rejects 16 ops not in the emitter table:

```
AutoPositionIds AutoZeroKVCache Shape Unsqueeze Cast Equal Where
ConstantOfShape Expand Range Cos Sin Greater Trilu Max ScatterND
```

These ops fall into three categories:

**RoPE (Rotary Positional Embeddings)**: `Cos`, `Sin`, `Range`, `AutoPositionIds`
- RoPE computes position-dependent rotation matrices using sin/cos of positions.
- `Range` generates position indices; `Cos`/`Sin` compute rotation components.

**Attention masking**: `Equal`, `Where`, `Greater`, `Trilu`, `ConstantOfShape`, `Expand`
- Causal attention mask construction: `Trilu` creates triangular mask,
  `Where`/`Greater`/`Equal` apply conditional logic, `ConstantOfShape`
  fills with -inf, `Expand` broadcasts the mask.

**Utility/shape ops**: `Shape`, `Unsqueeze`, `Cast`, `AutoZeroKVCache`, `Max`, `ScatterND`
- `Shape`/`Unsqueeze` are tensor metadata ops (could be no-ops in megakernel).
- `Cast` converts types (e.g., int64 indices to float32).
- `AutoZeroKVCache` initializes KV cache (one-time setup, not per-token).
- `Max` is element-wise max (for clamping).
- `ScatterND` is an indexed write (for KV cache updates).

### 2. Architectural Issue: Megakernel Runner vs GPU Engine Build Tags

The megakernel runner (`runner.go`) has `//go:build !cuda` and uses purego
dlopen to load compiled .so files. The GPU engine (`gpu_engine.go`) has
`//go:build cuda` and uses CGo-based cuBLAS/cuDNN. These are mutually
exclusive — the megakernel runner cannot be active in a CUDA build.

This means even if all ops were supported, the megakernel runner stub
(`runner_stub.go`) would return `errStub` from `LoadMegakernel()`, and
the megakernel would never fire in a `-tags cuda` build.

This is a fundamental architectural blocker that requires either:
- **Option A**: Move the megakernel runner out of the `!cuda` constraint
  (use CGo-based dlopen when building with `-tags cuda`)
- **Option B**: Remove build tags entirely per ADR-025 (bigger refactor)

### 3. Output Quality

Both models continue to produce gibberish/repetitive output (consistent
with S100.1.1 findings). This is a pre-existing issue unrelated to the
megakernel path.

## Summary of Blockers

| Blocker | Severity | Fix Scope |
|---------|----------|-----------|
| 16 unsupported ops in CheckSupport | High | Add emitters for each op (~2-4h) |
| runner_stub.go returns errStub in cuda build | Critical | Move runner to work in cuda build (~1h) |
| Build tag architecture (purego vs CGo) | Architectural | ADR-025 phase 2 (TBD) |

## Recommendation

1. **Immediate**: Fix `runner_stub.go` to use real dlopen (not stub) when
   building with `-tags cuda`. The CGo dlopen fallback created in this
   session provides the infrastructure.

2. **Short-term**: Add emitters for the 16 unsupported ops. Priority order:
   - `Cos`, `Sin` (trivial: `unaryOp("cosf")`, `unaryOp("sinf")`)
   - `Max` (trivial: `funcBinaryOp("fmaxf")`)
   - `Shape`, `Unsqueeze`, `Reshape` (no-ops)
   - `Cast` (type conversion)
   - `Range`, `Expand`, `Repeat` (indexing)
   - `Equal`, `Greater`, `Where` (comparison/select)
   - `Trilu`, `ConstantOfShape` (mask construction)
   - `ScatterND` (indexed write)
   - `AutoPositionIds`, `AutoZeroKVCache` (model-specific setup)

3. **Long-term**: Complete ADR-025 — remove `//go:build cuda` tags entirely,
   use runtime dlopen detection for all GPU operations.

## Quality Gate Assessment

| Gate | Status |
|------|--------|
| Megakernel fires | FAIL |
| Blocker precisely identified | PASS |
| Performance numbers recorded | PASS |
| Results appended to docs/updates.md | PASS |

---

# GPU Memory Allocator Optimization Results

Date: 2026-03-12

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10)
- **Model**: gemma3-gguf (Q4_0 quantized), 64 tokens generated
- **Device**: cuda (sm_75 PTX JIT on Blackwell)
- **Ollama Baseline**: 187.2 tok/s (target: 178.7 tok/s = 95%)

## Performance Progression

| Optimization | tok/s | Delta | Commit |
|---|---|---|---|
| Starting point (previous session) | 60.59 | -- | -- |
| Pool-backed GPUStorage | 61.59 | +1.7% | 399baf9 |
| Transpose-as-reshape | 64.41 | +4.6% | cea6ff4 |
| TensorPool GPU release | 64.90 | +0.8% | e7e0820 |
| GPUStorage view fix | 65.88 | +1.5% | 631a29d |
| Parameter upload fix | 64.34 | -2.3% | f625c88 |
| MemPool bucket sizing (4KB) | 63.54 | -1.2% | f0278f6 |
| MemPool bucket sizing (256B) | 63.47 | -0.1% | f8130a9 |
| GPUStorage refcounting | 61.08 | -3.8% | 276cc72 |
| Arena allocator (2GB, no reset) | 80.35 | +31.5% | 33b0dee |

## Key Findings

### 1. cudaMalloc Was the #1 Bottleneck (~6ms/token, 39% of per-token budget)

Each forward pass made ~1,500 cudaMalloc calls because:
- The MemPool was keyed by exact byte size, causing 85% miss rate as attention
  intermediates grew with kvSeqLen on every pass
- GPUStorage views (from Reshape/Transpose) had no-op Free(), so memory only
  returned to the pool via GC finalizers between passes
- Within-node intermediates (GQA does ~50 allocations internally) were not
  tracked by the graph executor's refcount system

### 2. Arena Allocator Eliminated All cudaMalloc During Inference

A 2GB pre-allocated bump-pointer arena serves as the GPU memory pool:
- 119,419 allocations, 0 fallback to MemPool (100% arena hit rate)
- Each allocation is a pointer bump + 256-byte alignment (~5ns vs ~4us for cudaMalloc)
- Weight uploads use runtime.Malloc directly (permanent storage, not arena)
- Arena used 2093.8 MB for 64 tokens + warmup -- tight fit for 2GB

### 3. Pool Bucketing and Refcounting Did Not Help

- Power-of-2 bucket sizing: marginal improvement (85% to 92% hit rate in one
  config) but didn't address the core issue of within-node intermediates
- GPUStorage refcounting: added complexity without throughput gain because the
  graph executor doesn't call Release() on within-node intermediates
- Arena approach bypasses both problems entirely

## Remaining Gap: 80.35 tok/s vs 178.7 tok/s target (45%)

Per-token budget at 80 tok/s (~12.5ms/token):
- GPU compute (Q4 GEMV + cuBLAS): ~3.6ms (29%)
- D2H memory copies: ~1.9ms (15%)
- H2D memory copies: ~1.5ms (12%)
- Kernel launch overhead: ~1.6ms (13%)
- Other (CPU, Go runtime, scheduling): ~3.9ms (31%)

Next targets:
1. Eliminate unnecessary D2H copies (~13 per forward pass)
2. Eliminate unnecessary H2D copies (~143 per forward pass)
3. Reduce kernel launch overhead (batch or fuse operations)
4. Investigate Go runtime overhead vs C/C++ baseline

---

# Performance Optimization Session 3

Date: 2026-03-12

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10) -- offline during session
- **Model**: gemma3-gguf (Q4_0 quantized), greedy decoding, 64 tokens
- **Ollama Baseline**: 187.2 tok/s (target: 178.7 tok/s = 95%)
- **Previous best**: 86.63 tok/s (correct output)

## Optimizations Implemented (Not Yet Benchmarked)

### 1. Pre-allocated KV cache buffers (commit 7e80e21)
- Allocates `[batch, maxSeqLen, dim]` GPU buffers once at first Update
- Subsequent appends: D2D memcpy at offset (no cudaMalloc)
- Eliminates 104 cudaMalloc/Free + 52 redundant D2D copies per token

### 2. GQA KV head broadcast (commit e92a04a)
- When numKVHeads=1 (Gemma 3: 1 KV head, 8 Q heads), skip Repeat
- MatMul batch broadcasting handles Q=[8, seqLen, headDim] * K=[1, seqLen, headDim]
- Eliminates ~192MB of redundant GPU memory copies per decode step

### 3. MatMulTransposeB via cuBLAS SgemmNT (commits 74cac33, bb5e5fd)
- Computes A*B^T without explicit Transpose allocation + kernel launch
- SDPA now type-asserts for TransposeBMatMuler, falls back to Transpose+MatMul
- Added to both CGO and purego paths
- Eliminates 18 GPU Transpose allocations + kernel launches per token

### 4. ExecutionPlan.Run() pre-allocated buffers (commit 4655ed6)
- Pre-allocate scratch slot array and per-instruction input buffers once
- Eliminates ~101 slice heap allocations per token

### 5. TensorPool shapeKey optimization (commit 4655ed6)
- Use strconv for common rank 1-3 shapes instead of fmt.Sprint

### 6. noopCleanup in getDevicePtr (commit a370d21)
- Shared package-level no-op replaces per-call closure allocation
- Eliminates ~200 tiny heap allocations per token

### 7. MatMulTransposeB in traced execution plan (commit 6df83f4)
- makeTracedForward now handles "MatMulTransposeB" op
- Compiled plans dispatch to TransposeBMatMuler with fallback

### 8. cublasSgemmStridedBatched (commit 2bbbeb1)
- Extended purego trampoline from 14 to 20 args
- Single batched GEMM call replaces N sequential Sgemm calls
- For 8 query heads per attention layer: 1 call instead of 8

## DGX Status

DGX Spark has been unreachable (SSH timeout) throughout this session.
All optimizations are pushed to main and ready for benchmarking when it
comes back online.

## Expected Impact

| Optimization | Expected tok/s Impact |
|---|---|
| Pre-allocated KV cache | Moderate: eliminates malloc overhead |
| GQA broadcast | Moderate: eliminates ~192MB copies/decode |
| MatMulTransposeB | Moderate: saves 18 kernel launches/token |
| Batched GEMM | Moderate: reduces cuBLAS call overhead |
| Heap allocation reduction | Small: reduces GC pressure |

## Build/Test Command for DGX

```
cd ~/Code/zerfoo/zerfoo
git pull
export PATH=$PATH:/usr/local/cuda-13.0/bin:/usr/local/go/bin
cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_120
cd ~/Code/zerfoo/zerfoo
export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib
go build -o bench_tps_opt3 ./cmd/bench_tps/
./bench_tps_opt3 -model /home/ndungu/models/gemma3-gguf/model.gguf -device cuda -tokens 64
```

---

# Post-Target Optimization Attempts

Date: 2026-03-12

## NVCC -O3 --use_fast_math

Upgraded kernel compilation from `-O2` to `-O3 --use_fast_math`.

| Run | tok/s |
|-----|-------|
| 1 | 189.32 |
| 2 | 186.85 |
| 3 | 188.64 |
| 4 | 187.13 |
| 5 | 188.47 |
| **Average** | **188.08** |

Negligible improvement (+0.04%). Kernels are bandwidth-bound, not compute-bound.

## CUDA Graph Capture (Not Yet Feasible)

Implemented CUDA graph API wrappers (purego bindings for cudaStreamBeginCapture,
cudaStreamEndCapture, cudaGraphInstantiate, cudaGraphLaunch) and a
CUDAGraphExecutor that captures the decode forward pass. Graph capture fails
because the forward pass includes synchronous D2H memcpy calls:

1. `GPUEngine.Gather` reads indices via `.Data()` to convert int64 to int32
2. `GPUStorage.TrySlice` is called during GQA for CPU fallback paths
3. KV cache `appendGPU` falls back to `.Data()` for CPU-resident tensors

These D2H copies conflict with CUDA stream capture even in relaxed mode because
the data they read was produced by operations on the capturing stream. CUDA
correctly blocks reads of not-yet-computed data.

Infrastructure is in place (graph/cuda_graph.go, internal/cuda graph APIs).
To enable graph capture, eliminate ALL D2H copies from the decode forward pass:
- Upload Gather indices to GPU without reading on CPU
- Remove CPU fallback paths from splitMergedQKV during GPU inference
- Ensure KV cache operations are fully GPU-resident

Expected gain when enabled: ~1-2 tok/s (eliminates 338 kernel launch overheads).

---

# TARGET REACHED: 95% of Ollama Inference Performance

Date: 2026-03-12

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10 Blackwell)
- **Model**: gemma3-gguf (Q4_K_M quantized, Gemma 3 1B)
- **Tokens**: 256 (greedy decoding)
- **CUDA**: 13.0, sm_121

## Results

| Run | tok/s |
|-----|-------|
| 1 | 186.73 |
| 2 | **189.78** |
| 3 | 187.41 |
| 4 | 188.28 |
| 5 | 187.85 |
| **Average** | **188.01** |

**Target: 187.35 tok/s (95% of Ollama's 197.21 tok/s) -- ACHIEVED**

## Performance Progression

| Optimization | tok/s | Delta | Commit |
|---|---|---|---|
| Previous session best | 177.49 | -- | c684a92 |
| Fused QK norm+RoPE kernel | 183.23 | +3.2% | 42f4008 |
| Zero-copy Q+K view (avoid Concat) | 186.54 | +1.8% | 27bf4d3 |
| Fused post-FFN norm+add kernel | 189.78 | +1.7% | 6b22b47 |

## Optimizations in This Session

### 1. Fused QK RMSNorm + RoPE kernel (commit 42f4008)

Replaced 4 kernel launches per GQA layer (Q norm, K norm, Q RoPE, K RoPE)
with a single fused CUDA kernel. Per block handles one head: computes RMS
reduction, normalizes with the appropriate weight (Q vs K), applies RoPE
rotation. For 26 layers with 5 heads each (4Q + 1KV), saves 78 kernel
launches per token.

### 2. Zero-copy Q+K concatenation (commit 27bf4d3)

When Q and K come from merged QKV (adjacent GPU views), creates a single
GPUStorageView spanning both instead of launching a Concat kernel. Saves
26 additional kernel launches per token.

### 3. Fused post-FFN RMSNorm + residual Add (commit 6b22b47)

Replaced separate postFfnNorm (RMSNorm) + residualAdd (Add) with a single
fused kernel that computes output = rmsnorm(input, weight, eps) + residual.
Saves 26 kernel launches per token. Also introduced residualRefNode for
zero-cost retrieval of stored residuals from fusedAddRMSNormNode.

## Kernel Launch Count Reduction

| Phase | Per-layer launches | Total (26 layers) |
|---|---|---|
| Before this session | ~17 | ~442 |
| After fused QK norm+RoPE | ~14 | ~364 |
| After fused norm+add | ~13 | ~338 |

## Architecture Summary

Per decode token (Gemma 3, seqLen=1, 26 layers):
- inputNorm (RMSNorm): 1 kernel
- Merged QKV GEMV: 1 kernel
- Fused QK norm+RoPE: 1 kernel (was 4)
- SDPA (MatMulTransposeB + ScaledSoftmax + MatMul): 3 kernels
- O proj GEMV: 1 kernel
- postAttnNorm (RMSNorm): 1 kernel
- Fused Add+RMSNorm (residual + pre-FFN norm): 1 kernel
- GateUp GEMV: 1 kernel
- FusedSwiGLU: 1 kernel
- Down GEMV: 1 kernel
- Fused Norm+Add (post-FFN norm + residual): 1 kernel (was 2)
Total: ~13 kernels/layer x 26 layers = ~338 + overhead
