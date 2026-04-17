# Zerfoo Benchmarks

All benchmarks run on DGX Spark GB10 unless noted. Greedy sampling, 128 output tokens, 3-run median.

## Throughput vs Ollama (GPU, 2026-03-30, v1.38.4)

| Model | Quant | Zerfoo tok/s | Ollama tok/s | Ratio | Date |
|-------|-------|-------------|--------------|-------|------|
| Gemma 3 1B | Q4_K_M | 241 | 188 | **1.28x** | 2026-03-31 |
| DeepSeek-R1 1.5B | Q4_K_M | 186 | 168 | **1.11x** | 2026-03-30 |
| Llama 3.2 3B | Q4_K_M | 92 | 93 | 0.99x | 2026-03-30 |
| Mistral 7B | Q4_K_M | 44 | 44 | 1.00x | 2026-03-30 |
| Gemma 4 E2B (edge) | Q4_K_M | 3.85* / 2.69** / 1.23*** | N/A | — | 2026-04-15 / 2026-04-16 |

\* The `3.85 tok/s` figure originally recorded for commit `72828131`
could not be reproduced on 2026-04-16: the same binary at the same
commit, on the same DGX host, running `-mode generate -device cuda
-steps 64 -prompt "The quick brown fox"` with
`ZERFOO_DISABLE_CUDA_GRAPH=1`, measured 2.69 tok/s. The 3.85 number
was likely measured with different parameters (possibly `-mode
forward`, or different `-steps`/`-seq`) and should be treated as
unverified. See `docs/devlog.md` 2026-04-16 entry.

\*\* 2.69 tok/s = current best reproducible gemma4e generate
throughput on GPU, measured 2026-04-16 at commit `72828131`,
capture disabled.

\*\*\* 1.23 tok/s = main (`6ad8bceb`) gemma4e generate on GPU,
capture disabled. A ~2.2x regression vs 72828131, tracked as T99.2.1
and bisected to commit `96c7540a` (T99.1.2): the per-step
`refreshPerLayerSlices` in `inference/gemma4_edge_ple_nodes.go`
issued 2*numLayers = 70 synchronous H2D `CopyFromHost` launches per
decode step plus 2 `.Data()` D2H copies. Fix on branch
`perf/t99-2-1-gemma4e-ple-decode` replaces that with 2 full-width
D2D copies per decode step (numLayers per-layer tensors become
non-owning `NewGPUStorageView` SubSlice views into two stable GPU
buffers). DGX verification pending GPU availability (training pod
holding the GPU at fix commit time).

**Correctness caveat (2026-04-16):** gemma4e generate produces
degenerate tokens (`"ly\ns\ns\ns..."` on CPU, multilingual gibberish
on GPU) on both `72828131` and current main. The E98 note about
"40 bytes non-degenerate output" referred to `-mode forward`
integration, not decode. Do not trust any gemma4e generate
throughput number until T99.2.2 restores decode coherence. The
throughput figures above are only useful as a regression floor for
T99.2.1, not as user-facing performance claims.

Once E99 lands (T99.1.3/T99.1.4) and decode is fixed (T99.2.2) and
throughput is restored (T99.2.1), this row should rise substantially
(Gemma 3 1B with capture is 241 tok/s). Ollama comparison skipped
because Ollama does not support the gemma4 architecture yet (E97.2
DEFERRED).

CUDA graph capture: 184/185 instructions (99.5%). Fused kernels: softmax+V multiply, repeat-interleave for GQA, fused AddRMSNorm, fused SwiGLU, fused QKNormRoPE, merged QKV, merged gate+up.

Full results: `results/benchmark-2026-03-31.json` (3-run median)

## Training: PatchTST (GPU, 2026-04-09, v1.42+)

| Workload | v1.37 | v1.38.4 | v1.42+ | Speedup (v1.37→v1.42+) |
|----------|-------|---------|--------|------------------------|
| 28K×20×10 | 596s | 128.5s | **40.3s** | **14.8x** |
| 20K×20×5 | N/A | N/A | **15.0s** | — |
| 5K×10×3 | N/A | N/A | **3.0s** | — |

DGX GB10, commit `2ecf473a`. Pre-allocated batch workspace + GPU dst-memory
reuse (ztensor#84/#85). Loss converges 99.9% (0.0178 → 0.000022) on 28K×20×10.

Previous v1.38.4 baseline (128.5s at 28K×20×10) regressed to OOM after E85
buffer pre-allocation (commit `09a318c6`) introduced per-op GPU memory leaks.
Fixed in ztensor#85 by reusing dst device pointers instead of allocating per call.

## Over-RAM Inference (2026-03-29)

| Model | Params | Quant | File Size | Shards | RAM | Load Time | Tokens | Zerfoo | Ollama |
|-------|--------|-------|-----------|--------|-----|-----------|--------|--------|--------|
| MiniMax-M2 | 229B (MoE) | Q4_K_M | 128.8 GB | 3 | 128 GB | 6.3s | ✅ 0.06 tok/s | ✅ | ❌ 500 error |

Notes:
- CPU-only (no GPU acceleration for over-RAM path yet)
- Throughput is NVMe-bound; faster storage = faster tokens
- Ollama: "unable to load model" 500 error on same hardware

## Granite Time Series vs Python granite-tsfm

Results in devlog, 2026-03-27.

## Flash Decode (GPU, pending)

Target: ≥ 1.5x speedup at seqLen_KV > 1024 (E43). To be benchmarked on DGX Spark after GPU streaming GEMM lands.
