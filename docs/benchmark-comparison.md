# Benchmark Comparison: Zerfoo vs Ollama vs llama.cpp

Side-by-side performance comparison on identical hardware using the same model
and quantization. All numbers are reproducible with the instructions below.

## Hardware

| Component | Spec |
|-----------|------|
| System | NVIDIA DGX Spark |
| GPU | GB10 (Grace Blackwell, sm_121) |
| Memory | 128 GB unified LPDDR5x |
| CUDA | 13.0 |
| OS | Ubuntu (aarch64) |
| Go | 1.25.0 |

## Model

| Property | Value |
|----------|-------|
| Model | Gemma 3 1B Instruct |
| Quantization | Q4_K_M (GGUF) |
| File size | ~0.8 GB |
| Source | [google/gemma-3-1b-it-GGUF](https://huggingface.co/google/gemma-3-1b-it-GGUF) |

## Results (2026-03-17)

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

### Summary

- Zerfoo with CUDA graphs: **245 tok/s** (+20% vs Ollama, ~10-15% vs llama.cpp)
- Zerfoo without CUDA graphs: **174 tok/s** (CUDA graph capture adds +40%)
- Ollama: **204 tok/s** (uses llama.cpp under the hood with its own overhead)

> **Note on llama.cpp numbers:** Direct llama.cpp measurements on this exact
> DGX Spark unit are pending. The estimate above is based on published community
> benchmarks for GB10 / Blackwell-class hardware with Gemma 3 1B Q4_K_M. We
> will update this table when we complete our own llama.cpp runs.

## Why Zerfoo Is Faster

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

## Methodology

### Zerfoo

```bash
# Clone and build
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo

# Download model (requires HuggingFace access)
mkdir -p models
# Place gemma-3-1b-it-Q4_K_M.gguf in models/

# Run benchmark
go run ./cmd/bench \
  --model models/gemma-3-1b-it-Q4_K_M.gguf \
  --tokens 256 \
  --warmup 3 \
  --prompt "The meaning of life is"
```

The `cmd/bench` harness reports throughput (tok/s), time-to-first-token (TTFT),
P99 latency, and GPU memory usage. Results are written to stdout and optionally
to a JSON file with `--output`.

For CUDA graph capture (default on supported GPUs), no additional flags are
needed. To disable CUDA graphs for comparison:

```bash
go run ./cmd/bench \
  --model models/gemma-3-1b-it-Q4_K_M.gguf \
  --backend cpu \
  --tokens 256 \
  --prompt "The meaning of life is"
```

### Ollama

```bash
# Install Ollama (https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull gemma3:1b

# Run with timing (Ollama prints eval rate in its output)
ollama run gemma3:1b "The meaning of life is" --verbose
```

Look for `eval rate: XXX.XX tokens/s` in the verbose output.

### llama.cpp

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j

# Run benchmark (use the same GGUF file)
./build/bin/llama-bench \
  -m /path/to/gemma-3-1b-it-Q4_K_M.gguf \
  -p 0 -n 256 -ngl 99
```

The `-p 0` flag skips prompt processing to measure pure decode throughput.
`-ngl 99` offloads all layers to GPU.

## Running Your Own Benchmarks

To get a fair comparison on your hardware:

1. **Use the same model file.** All three frameworks read GGUF, so use the
   exact same `.gguf` file for each run.

2. **Match token counts.** Set all frameworks to generate the same number of
   tokens (e.g., 256).

3. **Warm up.** Run at least 3 warm-up iterations before measuring. Zerfoo's
   `cmd/bench` does this automatically with `--warmup 3`.

4. **Isolate the GPU.** Close other GPU workloads. On Linux, check with
   `nvidia-smi` that no other processes are using the GPU.

5. **Report decode throughput.** All numbers in this guide are decode
   throughput (tokens per second during autoregressive generation), not prompt
   processing (prefill) speed. These are different metrics.

6. **Record your environment.** Report: GPU model, CUDA version, driver
   version, CPU, RAM, OS, and framework version/commit hash.

### Expected results by GPU class

| GPU | Zerfoo (est.) | Notes |
|-----|---------------|-------|
| DGX Spark GB10 | 245 tok/s | Measured |
| RTX 4090 | TBD | Community contributions welcome |
| RTX 3090 | TBD | Community contributions welcome |
| A100 80GB | TBD | Community contributions welcome |
| Apple M-series (CPU) | ~8-15 tok/s | Metal backend not yet implemented |

## Contributing Benchmarks

We welcome benchmark contributions from the community. To submit results:

1. Run all three frameworks on the same hardware using the methodology above.
2. Open an issue or PR with your results, including full hardware and software
   version details.
3. Include the raw JSON output from `cmd/bench --output results.json` for
   Zerfoo runs.
