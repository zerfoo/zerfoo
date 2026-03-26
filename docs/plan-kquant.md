# Native K-Quant GPU Support: Q4_K and Q6_K

## Context

### Problem Statement

Zerfoo achieves 236 tok/s on Gemma 3 1B (Q4_0 native), 16% faster than Ollama.
However, on Mistral 7B Q4_K_M, Zerfoo manages only 11.6 tok/s vs Ollama's 46.8
tok/s -- a 4x deficit. The root cause: Q4_K_M GGUF files use Q4_K for most layers
and Q6_K for attention output and FFN down projections. Zerfoo re-quantizes both
formats to Q4_0 at load time (model/gguf/loader.go), which:

1. **Loses precision** -- Q4_K has per-sub-block 6-bit scales; Q4_0 has only a
   single fp16 scale per 32 values. This precision loss may affect output quality.
2. **Prevents native GEMV dispatch** -- The re-quantized Q4_0 weights work, but
   the existing Q4_K and Q6_K CUDA GEMV kernels (gemv_q4k.cu, gemv_q6k.cu) go
   unused because the storage type is Q4Storage, not Q4KStorage.
3. **Blocks merged QKV/GateUp optimization** -- arch_common.go only checks for
   `*tensor.Q4Storage` when building merged weight tensors for single-GEMV decode.
   Q4_K_M models that keep native Q4KStorage would not get merged QKV or GateUp.
4. **transposeWeight does not handle Q4K/Q6K** -- Only Q4Storage gets virtual
   transpose on GPU; Q4KStorage and Q6KStorage fall through to engine.Transpose,
   which dequantizes to F32, triggering D2H copies that break CUDA graph capture.

### Lessons Learned

The initial plan (Waves 1-3) assumed that existing K-quant GEMV kernels would
perform at parity with Q4_0 GEMV once native loading was enabled. This proved
wrong. Benchmark findings from executing T2.1 (native loading):

| Format | GEMV Kernel | Gemma 3 1B tok/s | vs Q4_0 |
|--------|-------------|------------------|---------|
| Q4_0   | gemv_q4.cu  | 236              | baseline |
| Q4_K (native) | gemv_q4k.cu | 128         | -45% |
| Q6_K (native) | gemv_q6k.cu | 158         | -33% |

The Q4_K format has more complex dequantization: 256-element super-blocks with
nested 32-element sub-blocks and per-sub-block 6-bit scales, vs Q4_0's simple
32-element blocks with one fp16 scale. This complexity translates to higher
register pressure and more memory accesses in the GEMV kernel.

**Decision:** All K-quant loading was reverted back to Q4_0 re-quantization
(T2.1 REVERTED). The infrastructure work (virtual transpose T2.2, merged
QKV/GateUp T2.3, merge functions T1.1/T1.2) remains merged and ready to
activate once kernel performance is resolved.

Additionally, Mistral 7B has two independent issues unrelated to K-quant
storage: (1) CUDA graph capture fails due to a D2H cudaMemcpy in instruction 5,
and (2) Mistral's GGUF declares `general.architecture = llama` causing wrong
tokenizer configuration and garbage text output.

### What Already Exists

The ztensor repo already has substantial K-quant infrastructure:

| Component | Location | Status |
|-----------|----------|--------|
| Q4KStorage, Q5KStorage, Q6KStorage types | ztensor/tensor/quantized_kquant.go | Complete |
| CPU dequantization (Q4_K, Q5_K, Q6_K) | ztensor/tensor/quantized_kquant.go | Complete |
| Dequantizer registry entries | ztensor/tensor/quant_registry.go | Complete |
| GPU weight upload for Q4K/Q5K/Q6K | ztensor/compute/gpu_engine.go (UploadWeights) | Complete |
| GPU MatMul dispatch for Q4K/Q6K/Q5K | ztensor/compute/gpu_engine.go (matMulQ4K, etc.) | Complete |
| Fused Q4_K GEMV kernel | ztensor/internal/cuda/kernels/gemv_q4k.cu | Complete (slow) |
| Fused Q4_K GEMV kernel (SM121 optimized) | ztensor/internal/cuda/kernels/gemv_q4k_sm121.cu | Complete (slow) |
| Fused Q6_K GEMV kernel | ztensor/internal/cuda/kernels/gemv_q6k.cu | Complete (slow) |
| Fused Q5_K GEMV kernel | ztensor/internal/cuda/kernels/gemv_q5k.cu | Complete |
| Q4_K dequant kernel (for batch>1) | ztensor/internal/cuda/kernels/dequant_q4k.cu | Complete |
| MergeQ4Storage (for merged QKV/GateUp) | ztensor/tensor/quantized.go | Complete |
| MergeQ4KStorage | ztensor/tensor/quantized_kquant.go | Complete (ztensor v0.5.0) |
| MergeQ6KStorage | ztensor/tensor/quantized_kquant.go | Complete (ztensor v0.5.0) |
| Virtual transpose for Q4K/Q6K | inference/arch_common.go | Complete (merged) |
| Merged QKV/GateUp for Q4K | inference/arch_common.go | Complete (merged) |

### Objectives

- Optimize Q4_K GEMV kernel to within 10% of Q4_0 GEMV performance.
- Fix CUDA graph capture for Mistral 7B (D2H copy in instruction 5).
- Fix Mistral architecture detection and tokenizer configuration.
- Once kernel is fast enough, re-enable native Q4_K loading (all infrastructure is ready).
- Validate with benchmarks: Mistral 7B >= 40 tok/s, Gemma 3 1B no regression.

### Success Metrics

| Metric | Target |
|--------|--------|
| Q4_K GEMV speed | >= 215 tok/s on Gemma 3 1B (within 10% of Q4_0's 236) |
| Mistral 7B decode tok/s | >= 40 tok/s (from 11.6, parity with Ollama's 46.8) |
| Gemma 3 1B decode tok/s | No regression (>= 230 tok/s) |
| CUDA graph capture | No D2H copies during decode for Mistral 7B |
| Mistral text output | Coherent text (not garbage) |

---

## Completed Work

### Wave 1: ztensor Merge Functions (completed 2026-03-22)

- [x] T1.1 Add MergeQ4KStorage to ztensor  (ztensor v0.5.0)
- [x] T1.2 Add MergeQ6KStorage to ztensor  (ztensor v0.5.0)

### Wave 2: GGUF Loader + arch_common Integration (completed 2026-03-23)

- [x] T2.1 Remove Q4_K/Q6_K re-quantization in GGUF loader -- REVERTED (2026-03-24)
  - Native loading worked but Q4_K GEMV was 45% slower than Q4_0 GEMV.
  - All K-quants reverted to Q4_0 re-quantization for performance.
  - Will re-enable after E4 kernel optimization.
- [x] T2.2 Add virtual transpose for Q4KStorage/Q6KStorage in transposeWeight (merged, stays)
- [x] T2.3 Add merged QKV/GateUp support for Q4KStorage/Q6KStorage (merged, stays)

---

## Remaining Work

### E4: Q4_K GEMV Kernel Optimization (ztensor repo)

The fused gemv_q4k.cu kernel is 45% slower than gemv_q4.cu on GB10. The Q4_K
format's 256-element super-blocks with nested 32-element sub-blocks and
per-sub-block 6-bit scales create higher register pressure and more complex
memory access patterns than Q4_0's simple 32-element blocks. The kernel must be
optimized to close this gap before native K-quant loading becomes viable.

- [ ] T4.1 Profile gemv_q4k.cu vs gemv_q4.cu  Est: 2h  repo: ztensor
  - Use `ncu` (Nsight Compute) to profile both kernels on DGX Spark.
  - Collect: register usage, occupancy, memory throughput, warp stalls,
    L1/L2 cache hit rates, instruction mix.
  - Compare SM121-optimized variant (gemv_q4k_sm121.cu) against generic.
  - Document bottleneck: register pressure? memory access pattern? warp
    divergence from 6-bit scale unpacking? instruction count?
  - Acceptance: profiling report in devlog with root cause identified.

- [ ] T4.2 Optimize Q4_K GEMV kernel  Est: 4h  repo: ztensor
  - Apply optimizations based on T4.1 profiling. Likely candidates:
    - Reduce register usage by recomputing values instead of caching.
    - Improve coalesced memory access for super-block layout (256-byte blocks
      vs Q4_0's 18-byte blocks).
    - Use shared memory for sub-block scales (avoid redundant global loads).
    - Reduce warp divergence in 6-bit scale unpacking.
    - Tune thread block size and elements-per-thread for Q4_K's larger blocks.
  - Benchmark after each change to measure incremental improvement.
  - Acceptance: `go test ./internal/cuda/ -run BenchmarkGEMV` shows improvement.

- [ ] T4.3 Benchmark Q4_K GEMV at target  Est: 1h  repo: ztensor
  - Run Gemma 3 1B Q4_K_M with native Q4_K loading (temporarily re-enable T2.1).
  - Target: >= 215 tok/s (within 10% of Q4_0's 236 tok/s).
  - If target met: proceed to T4.4. If not: iterate on T4.2.
  - Acceptance: benchmark result recorded in devlog.

- [ ] T4.4 Re-enable native Q4_K loading in zerfoo  Est: 1h  repo: zerfoo  depends: T4.3
  - Re-apply the T2.1 change: remove Q4_K re-quantization in model/gguf/loader.go.
  - All infrastructure (virtual transpose T2.2, merged QKV T2.3) is already
    merged and will activate automatically with native Q4KStorage.
  - Run full test suite: `go test ./...`
  - Acceptance: Gemma 3 1B Q4_K_M >= 215 tok/s, all tests pass.

### E5: CUDA Graph Capture Fix for Mistral 7B (zerfoo repo)

Instruction 5 in Mistral 7B's forward pass (ResidualAdd / fusedAddRMSNormNode)
triggers a D2H cudaMemcpy during CUDA graph capture, breaking it. This is a
pre-existing bug that occurs even with Q4_0 storage. The D2H copy comes from
GPUStorage.TrySlice() being called somewhere in the forward pass. Without CUDA
graphs, Mistral 7B gets only 11 tok/s instead of the expected 40+ tok/s.

- [ ] T5.1 Trace the D2H copy source  Est: 2h  repo: zerfoo
  - Enable CUDA API tracing or add logging to GPUStorage.TrySlice() and
    GPUStorage.Slice() to identify which operation triggers the D2H copy.
  - Identify the call chain: which layer/operation at instruction 5 calls
    Slice() that results in a cudaMemcpy D2H.
  - Check if this is specific to Mistral's architecture (sliding window
    attention?) or a general issue with certain residual connection patterns.
  - Acceptance: root cause documented, specific call site identified.

- [ ] T5.2 Eliminate the D2H copy  Est: 3h  repo: zerfoo  depends: T5.1
  - Fix the identified call site. Options:
    - Replace Slice() with a GPU-side view/offset that avoids D2H.
    - Pre-allocate the needed buffer so Slice() is not called during forward.
    - If the Slice() is in KV cache management, restructure to use GPU pointers.
  - Verify CUDA graph capture succeeds for Mistral 7B: all instructions captured.
  - Acceptance: `go test ./inference/ -run TestCUDAGraphCapture` passes for Mistral.

- [ ] T5.3 Benchmark Mistral 7B with CUDA graphs  Est: 1h  repo: zerfoo  depends: T5.2
  - Run `bench_tps -model mistral-7b-instruct-v0.3.Q4_K_M.gguf -tokens 128 -device cuda`
  - Target: >= 40 tok/s (from 11.6 baseline, parity with Ollama's 46.8).
  - Acceptance: benchmark result recorded in devlog.

### E6: Mistral Architecture Detection and Tokenizer Fix (zerfoo repo)

Mistral's GGUF file declares `general.architecture = llama`, so Zerfoo loads it
as Llama. The output is garbage text. Need to detect Mistral from GGUF metadata
and apply correct tokenizer handling. Mistral uses a 32768-token vocabulary vs
Llama's 128256 tokens.

- [ ] T6.1 Detect Mistral from GGUF metadata  Est: 2h  repo: zerfoo
  - Parse GGUF metadata fields to identify Mistral models. Possible signals:
    - `general.name` contains "mistral" or "Mistral".
    - Vocabulary size == 32768 (vs Llama's 128256).
    - Sliding window attention config present (`llama.attention.sliding_window`).
    - Presence of `tokenizer.ggml.pre` == "mistral" or similar.
  - Add an architecture detection step in inference/load_gguf.go or
    model/gguf/loader.go that sets a Mistral flag.
  - Acceptance: Mistral 7B GGUF is correctly identified as Mistral architecture.

- [ ] T6.2 Apply correct tokenizer for Mistral  Est: 2h  repo: zerfoo  depends: T6.1
  - Configure the BPE tokenizer (ztoken) with Mistral's vocabulary and special
    tokens when Mistral is detected.
  - Verify BOS/EOS token IDs match Mistral's expected values.
  - Handle any Mistral-specific prompt formatting (e.g., [INST] tags).
  - Acceptance: Mistral 7B produces coherent, non-garbage text output.

- [ ] T6.3 Benchmark Mistral 7B end-to-end  Est: 1h  repo: zerfoo  depends: T5.3, T6.2
  - Run full end-to-end test: load, generate 128 tokens, verify coherent output.
  - Verify tok/s is not degraded by the detection/tokenizer fix.
  - Compare against Ollama output for the same prompt (qualitative check).
  - Acceptance: coherent output + tok/s >= 40 (with CUDA graph fix from E5).

---

## Wave Plan

### Wave 4: Kernel Profiling + CUDA Graph Tracing (2 agents, parallel)

E4 is in the ztensor repo, E5 is in the zerfoo repo -- no conflicts.

| Agent | Task | Est | Repo |
|-------|------|-----|------|
| A | T4.1 Profile gemv_q4k.cu vs gemv_q4.cu | 2h | ztensor |
| B | T5.1 Trace the D2H copy source | 2h | zerfoo |

### Wave 5: Kernel Optimization + CUDA Graph Fix (2 agents, parallel)

| Agent | Task | Est | Repo |
|-------|------|-----|------|
| A | T4.2 Optimize Q4_K GEMV kernel | 4h | ztensor |
| B | T5.2 Eliminate the D2H copy | 3h | zerfoo |

### Wave 6: Benchmarks + Mistral Detection (2 agents, parallel)

| Agent | Task | Est | Repo |
|-------|------|-----|------|
| A | T4.3 Benchmark Q4_K GEMV + T4.4 Re-enable native loading | 2h | ztensor + zerfoo |
| B | T5.3 Benchmark Mistral CUDA graphs + T6.1 Detect Mistral + T6.2 Tokenizer fix | 5h | zerfoo |

### Wave 7: Final Validation (1 agent)

| Agent | Task | Est | Repo |
|-------|------|-----|------|
| A | T6.3 Benchmark Mistral 7B end-to-end | 1h | zerfoo |

---

## Dependencies

```
T4.1 ──► T4.2 ──► T4.3 ──► T4.4          (E4: kernel optimization, ztensor)
T5.1 ──► T5.2 ──► T5.3 ──────────┐
                                  ├──► T6.3  (E6: final validation)
T6.1 ──► T6.2 ───────────────────┘
```

- E4 (ztensor) and E5/E6 (zerfoo) run in parallel with no cross-repo conflicts.
- T4.4 depends on T4.3 (kernel must meet target before re-enabling native loading).
- T6.3 depends on both T5.3 (CUDA graphs working) and T6.2 (tokenizer fixed).
- T6.1 is independent and can start in any wave after Wave 4.

---

## Timeline and Milestones

| ID | Milestone | Exit Criteria | Depends On |
|----|-----------|---------------|------------|
| M1 | Merge functions ready | MergeQ4KStorage + MergeQ6KStorage in ztensor, tests pass | Wave 1 (DONE) |
| M2 | Infrastructure merged | Virtual transpose + merged QKV/GateUp in zerfoo | Wave 2 (DONE) |
| M3 | Kernel bottleneck identified | ncu profiling report with root cause | Wave 4 |
| M4 | CUDA graph capture fixed | Mistral 7B captures all instructions | Wave 5 |
| M5 | Q4_K GEMV at target speed | >= 215 tok/s on Gemma 3 1B | Wave 6 |
| M6 | Mistral 7B fully working | Coherent text, >= 40 tok/s, CUDA graphs | Wave 7 |

Estimated total: 4-5 days with 2 parallel agents.

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Q4_K GEMV cannot reach within 10% of Q4_0 | High | Medium | The Q4_K format is fundamentally more complex (256-byte super-blocks vs 18-byte blocks). If kernel optimization plateaus above 10% gap, consider: (a) hybrid approach where only Q6_K layers stay native, (b) accept the gap if quality improvement justifies it, (c) custom Q4_K memory layout optimized for GPU access patterns. |
| R2 | D2H copy in Mistral is architectural (sliding window) | Medium | Medium | Sliding window attention may require dynamic slice operations that inherently need CPU involvement. If so, consider pre-computing all slice offsets at graph build time and storing them as GPU constants. |
| R3 | Mistral tokenizer differences deeper than vocabulary | Medium | Low | Mistral may have BPE merge rule differences, special token handling, or byte-fallback differences. If simple vocabulary swap fails, may need a Mistral-specific tokenizer profile in ztoken. |
| R4 | go.mod dependency delay between ztensor and zerfoo | Low | Medium | T4.4 (re-enable native loading) needs the optimized ztensor kernel. Coordinate ztensor release before T4.4. T5/T6 work is independent of ztensor changes. |

---

## Operating Procedure

### Definition of Done

- Q4_K GEMV kernel achieves >= 215 tok/s on Gemma 3 1B Q4_K_M (within 10% of Q4_0).
- Native Q4_K loading re-enabled in GGUF loader (no re-quantization to Q4_0).
- CUDA graph capture succeeds for Mistral 7B with no D2H copies during decode.
- Mistral 7B correctly detected from GGUF metadata and uses correct tokenizer.
- Mistral 7B Q4_K_M achieves >= 40 tok/s on DGX Spark.
- Mistral 7B produces coherent text output (not garbage).
- Gemma 3 1B Q4_K_M shows no throughput regression (>= 230 tok/s).
- All existing tests pass (`go test ./...` in both repos).
