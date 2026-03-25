# Benchmarks

Performance measurements, comparisons with other runtimes, and methodology
for reproducing all benchmark results.

## Hardware

All official benchmarks run on a single machine:

| Component | Spec |
|-----------|------|
| System | NVIDIA DGX Spark |
| SoC | NVIDIA Grace Blackwell GB10 (sm_121) |
| Memory | 128 GB unified LPDDR5x (~200 GB/s bandwidth) |
| CUDA | 13.0 |
| OS | Linux (aarch64) |
| Go | 1.25.0 |

## Model

| Property | Value |
|----------|-------|
| Model | Gemma 3 1B Instruct |
| Format | GGUF |
| Quantization | Q4_K_M |
| File size | ~0.8 GB |
| Source | [google/gemma-3-1b-it-GGUF](https://huggingface.co/google/gemma-3-1b-it-GGUF) |

## Results

### Current Baseline (2026-03-20, main @ 22b1c31 + ztensor @ 4d56fd6)

| Model | Format | Tok/s | CUDA Graph | Tokens | Date | Commit | Notes |
|-------|--------|-------|------------|--------|------|--------|-------|
| Gemma 3 1B | Q4_K_M | 244.45 | Yes | 256 | 2026-03-20 | 22b1c31 | Regression fixed (E103) |
| Gemma 3 1B | Q4_K_M | 244.18 | Yes | 256 | 2026-03-20 | 22b1c31 | Run 2 |
| Gemma 3 1B | Q4_K_M | 244.62 | Yes | 256 | 2026-03-20 | 22b1c31 | Run 3 |

Roofline analysis: GB10 LPDDR5x ~200 GB/s, max ~257 tok/s for 778 MB model.
Current 244 tok/s = 95% bandwidth utilization.
500 tok/s target requires hardware with higher memory bandwidth (A100/H100).

### Previous Baseline (2026-03-17, ztensor @ 4e85b12)

| Model | Format | Tok/s | CUDA Graph | Tokens | Date | Commit |
|-------|--------|-------|------------|--------|------|--------|
| Gemma 3 1B | Q4_K_M | 219.17 | Yes | 50 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 245.15 | Yes | 256 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 248.47 | Yes | 512 | 2026-03-17 | 4e85b12 |
| Gemma 3 1B | Q4_K_M | 174.44 | No | 256 | 2026-03-17 | 8717a12 |
| Ollama gemma3:1b | - | 203.60 | - | 989 | 2026-03-17 | - |

Note: dp4a INT8 GEMV (T4.1) and arena free-list reuse (T4.2) merged into ztensor.
At batch=1 decode, throughput is memory-bandwidth-bound so dp4a shows parity as expected.
dp4a benefits will appear at larger batch sizes where compute becomes the bottleneck.

### Previous Baseline (2026-03-16, main @ a5c54c3)

| Model | Format | Tok/s | CUDA Graph % | Output Quality | Tokens | Date |
|-------|--------|-------|-------------|----------------|--------|------|
| Gemma 3 1B | GGUF Q4_0 | 103.22 (CUDA) / 9.20 (CPU) | 99.5% | Coherent | 50 | 2026-03-16 |
| Gemma 3 1B FP16 | GGUF Q4_0 + FP16 | 9.20 (CPU) | - | Valid | 30 | 2026-03-16 |
| Gemma 3 1B FP8 | GGUF Q4_0 + FP8 | ~0.5 (CPU) | - | Valid | 30 | 2026-03-16 |
| TinyLlama 1.1B | GGUF Q4_K_M | 7.18 (CPU) | - | Low (small model) | 50 | 2026-03-16 |
| Qwen 2.5 0.5B | GGUF Q4_K_M | ~13 (CPU) | - | Garbled (tokenizer bug) | 50 | 2026-03-16 |
| Mistral 7B | GGUF Q4_K_M | ~0.55 (CPU) | - | Low (loads as llama) | 50 | 2026-03-16 |
| Phi-3.5 mini | GGUF Q4_K_M | - | - | FAIL (merged QKV) | - | 2026-03-16 |

### Previous Baseline (2026-03-15, main @ 4724c47)

| Model | Format | Tok/s | CUDA Graph % | Output Quality | Tokens | Date |
|-------|--------|-------|-------------|----------------|--------|------|
| Gemma 3 1B | GGUF Q4_K | 232.21 | 99.5% | Baseline | 256 | 2026-03-15 |
| Llama 3 1B | GGUF | 12.93 | 2.0% | Semi-coherent | 20 | 2026-03-15 |
| Qwen 2.5 0.5B | GGUF | 15.79 | 1.8% | Working (rep. penalty helps) | 20 | 2026-03-15 |
| Mistral 7B | GGUF | 3.94 | 1.2% | Working (spaces fixed) | 20 | 2026-03-15 |
| Phi-3 mini | GGUF | 4.14 | 0.5% | Semi-coherent | 20 | 2026-03-15 |

## Comparison: Zerfoo vs Ollama vs llama.cpp

All measurements use a fixed prompt ("The meaning of life is") and measure
steady-state decode throughput (tokens per second) after warm-up.

| Framework | Version | Tokens | Tok/s (decode) | CUDA Graphs | Notes |
|-----------|---------|--------|----------------|-------------|-------|
| **Zerfoo** | v0.x (4e85b12) | 256 | **245.15** | Yes | Q4_K_M loaded, re-quantized to Q4_0 at load time |
| **Zerfoo** | v0.x (4e85b12) | 512 | **248.47** | Yes | Throughput stable at longer sequences |
| **Zerfoo** | v0.x (4e85b12) | 50 | 219.17 | Yes | Lower at short sequences (warm-up amortization) |
| **Zerfoo** | v0.x (8717a12) | 256 | 174.44 | No | Without CUDA graph capture |
| **Ollama** | latest | 989 | 203.60 | N/A | Default settings, `ollama run gemma3:1b` |
| **llama.cpp** | b5220+ | 256 | ~210-230 | No | Estimated from community reports on GB10-class hardware |

**Summary:**

- Zerfoo with CUDA graphs: **245 tok/s** (+20% vs Ollama, ~10-15% vs llama.cpp)
- Zerfoo without CUDA graphs: **174 tok/s** (CUDA graph capture adds +40%)
- Ollama: **204 tok/s** (uses llama.cpp under the hood with its own overhead)

> **Note on llama.cpp numbers:** Direct llama.cpp measurements on this exact
> DGX Spark unit are pending. The estimate above is based on published community
> benchmarks for GB10 / Blackwell-class hardware with Gemma 3 1B Q4_K_M. We
> will update this table when we complete our own llama.cpp runs.

### Why Zerfoo Is Faster

1. **CUDA graph capture (99.5% coverage):** The entire decode step (26
   transformer layers, attention, FFN, norms) is captured as a single CUDA
   graph. This eliminates per-kernel launch overhead (~5-10 us per launch x
   hundreds of kernels per token) and lets the GPU execute the full pipeline
   without returning control to the host.

2. **Fused kernels:** Operations that are separate kernel launches in other
   frameworks are fused in Zerfoo:
   - `FusedAddRMSNorm` (residual addition + RMS normalization in one pass)
   - `FusedQKNormRoPE` (QK normalization + rotary position embeddings)
   - `FusedSiluGate` (SiLU activation + gating in the FFN)
   - Merged QKV and Gate+Up projections (single GEMV instead of 2-3 separate)

3. **Zero CGo overhead:** GPU bindings use purego/dlopen instead of CGo. This
   avoids the ~200 ns per CGo call overhead that accumulates across thousands of
   CUDA API calls per token.

4. **Optimized Q4_0 GEMV:** The quantized matrix-vector multiply kernel is
   hand-tuned for the decode path with coalesced memory access patterns and
   efficient warp-level reductions.

### Expected Results by GPU Class

| GPU | Zerfoo (est.) | Notes |
|-----|---------------|-------|
| DGX Spark GB10 | 245 tok/s | Measured |
| RTX 4090 | TBD | Community contributions welcome |
| RTX 3090 | TBD | Community contributions welcome |
| A100 80GB | TBD | Community contributions welcome |
| Apple M-series (CPU) | ~8-15 tok/s | Metal backend not yet implemented |

## Vision Models

Vision model benchmarks use synthetic weights with small dimensions for CI,
and full GGUF models for hardware throughput validation.

| Model | Test | Status | Env Var |
|-------|------|--------|---------|
| LLaVA | BenchmarkLLaVA_Throughput | Synthetic (CI) | - |
| LLaVA | TestLLaVA_VisionPipeline | Full model | LLAVA_GGUF_PATH |
| Qwen-VL | BenchmarkQwenVL_Throughput | Synthetic (CI) | - |
| Qwen-VL | TestQwenVL_VisionPipeline | Full model | QWENVL_GGUF_PATH |

```bash
# Run synthetic benchmarks
go test -bench BenchmarkLLaVA -count=1 ./tests/parity/
go test -bench BenchmarkQwenVL -count=1 ./tests/parity/

# Run full-model vision pipeline tests (requires GGUF files)
LLAVA_GGUF_PATH=/path/to/llava.gguf go test -run TestLLaVA_VisionPipeline -count=1 -v ./tests/parity/
QWENVL_GGUF_PATH=/path/to/qwenvl.gguf go test -run TestQwenVL_VisionPipeline -count=1 -v ./tests/parity/
```

## Performance Milestones

| Date | Milestone | Tok/s | Notes |
|------|-----------|-------|-------|
| 2026-03-17 | dp4a + arena reuse (T4.1+T4.2) | 245.15 | Parity at batch=1 (memory-bound); dp4a benefits at larger batches |
| 2026-03-17 | Q4_0 re-quant restored | 244.99 | +32% vs regression, +20% vs Ollama |
| 2026-03-14 | CUDA graph capture | 234.30 | +26% vs non-graph baseline |
| 2026-03-13 | GPU-first pipeline | 6.84 | Phase 32, +33.6% from D2H elimination |
| 2026-03-13 | Graph compilation | 6.86 | Phase 30, +5% from worker pool |
| 2026-03-12 | NEON SIMD | 8.15 | Phase 34, +18.8% CPU acceleration |
| 2026-03-12 | CPU baseline | 6.5 | Phase 29, parallelFor + xblas |
| 2026-03-11 | Initial GPU | 5.12 | Phase 31, 43% cgocall overhead |
| 2026-03-10 | Initial CPU | 3.60 | Phase 26, Gemma 3 2B Q4 |

## Methodology

Every performance claim in this project is backed by a reproducible benchmark
run. This section describes the measurement procedure so that anyone can
independently verify the numbers.

### Software Versions

Always record exact versions alongside every benchmark result:

| Component | Version |
|-----------|---------|
| Go | 1.25.0 |
| CUDA Toolkit | 13.0 |
| Zerfoo | latest `main` (record commit hash with each run) |
| ztensor | v0.1.0 (see `go.mod`) |
| ztoken | v0.1.0 (see `go.mod`) |
| float16 | v0.2.0 |
| float8 | v0.2.0 |

### Measurement Procedure

1. **Warm-up phase** -- Run a short generation (16-32 tokens) to warm up the
   GPU, populate caches, and trigger JIT compilation / CUDA graph capture.
   Discard these results.

2. **Measurement window** -- Generate at least **256 tokens** in decode mode.
   Measure wall-clock time from the first decode token to the last.

3. **Decode-only measurement** -- Report only decode throughput (tokens per
   second). Prefill / prompt-processing time is excluded from the tok/s
   number.

4. **CUDA graph coverage** -- Record the percentage of operations captured in
   CUDA graphs. The current baseline achieves 99.5% coverage.

5. **Repeat** -- Run the benchmark at least 3 times and report the median
   result.

### Benchmark Commands

**Zerfoo:**

```bash
go run ./cmd/bench_tps \
  -model /path/to/gemma-3-1b-q4_k_m.gguf \
  -tokens 256 \
  -prompt "The quick brown fox"
```

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to GGUF model file |
| `-tokens` | 64 | Number of tokens to generate |
| `-prompt` | `""` | Input prompt text |

For official benchmarks, always use `-tokens 256` or higher.

**Ollama:**

```bash
ollama run gemma3:1b --verbose "The quick brown fox" 2>&1 | grep "eval rate"
```

**llama.cpp:**

```bash
./build/bin/llama-bench \
  -m /path/to/gemma-3-1b-it-Q4_K_M.gguf \
  -p 0 -n 256 -ngl 99
```

The `-p 0` flag skips prompt processing to measure pure decode throughput.
`-ngl 99` offloads all layers to GPU.

### How to Reproduce from Scratch

```bash
# 1. Clone the repo
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo

# 2. Ensure Go 1.25+ is installed
go version

# 3. Download dependencies
go mod tidy

# 4. Download the model (Gemma 3 1B Q4_K_M GGUF from HuggingFace)
#    Place it at a known path, e.g. ~/models/gemma-3-1b-q4_k_m.gguf

# 5. SSH into the DGX Spark (or use any CUDA-capable machine)
ssh ndungu@192.168.86.250

# 6. Run the benchmark (3 times, take the median)
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256

# 7. Record: median tok/s, CUDA graph %, commit hash
git rev-parse HEAD
```

### Comparison Methodology

When comparing against other runtimes (llama.cpp, Ollama, vLLM, etc.):

1. **Same hardware** -- Run both on the same DGX Spark machine.
2. **Same model** -- Use the identical GGUF file (or equivalent quantization).
3. **Same token count** -- Generate the same number of tokens (256+).
4. **Same prompt** -- Use the same input prompt.
5. **Decode-only** -- Compare decode tok/s, not end-to-end latency.
6. **Warm up both** -- Give the competing runtime the same warm-up opportunity.
7. **Report versions** -- Record exact version/commit of the competing runtime.

### Running Your Own Benchmarks

To get a fair comparison on your hardware:

1. **Use the same model file.** All three frameworks read GGUF, so use the
   exact same `.gguf` file for each run.
2. **Match token counts.** Set all frameworks to generate the same number of
   tokens (e.g., 256).
3. **Warm up.** Run at least 3 warm-up iterations before measuring.
4. **Isolate the GPU.** Close other GPU workloads. On Linux, check with
   `nvidia-smi` that no other processes are using the GPU.
5. **Report decode throughput.** All numbers in this guide are decode
   throughput (tokens per second during autoregressive generation), not prompt
   processing (prefill) speed.
6. **Record your environment.** Report: GPU model, CUDA version, driver
   version, CPU, RAM, OS, and framework version/commit hash.

## Contributing Benchmarks

We welcome benchmark contributions from the community. To submit results:

1. Run all three frameworks on the same hardware using the methodology above.
2. Open an issue or PR with your results, including full hardware and software
   version details.
3. Include the raw JSON output from `cmd/bench --output results.json` for
   Zerfoo runs.

---

## Historical Measurements

Older baselines moved here during /trim.
