# Benchmarking Methodology

How Zerfoo benchmark numbers are produced, verified, and compared against
competing runtimes.

## Purpose

Every performance claim in this project is backed by a reproducible benchmark
run. This document describes the hardware, software, model, and measurement
procedure so that anyone can independently verify the numbers.

## Hardware

All official benchmarks run on a single machine:

| Property | Value |
|----------|-------|
| System | NVIDIA DGX Spark |
| SoC | NVIDIA Grace Blackwell GB10 |
| GPU SM | sm_121 |
| Memory | 128 GB unified LPDDR5x |
| Access | `ssh ndungu@192.168.86.250` |

## Software

| Component | Version |
|-----------|---------|
| Go | 1.25.0 |
| CUDA Toolkit | 13.0 |
| Zerfoo | latest `main` (record the commit hash with each run) |
| ztensor | v0.1.0 (see `go.mod`) |
| ztoken | v0.1.0 (see `go.mod`) |
| float16 | v0.2.0 |
| float8 | v0.2.0 |
| OS | Linux (DGX Spark default image) |

Always record the exact commit hash (`git rev-parse HEAD`) alongside every
benchmark result.

## Model

| Property | Value |
|----------|-------|
| Model | Gemma 3 1B |
| Format | GGUF |
| Quantization | Q4_K_M |
| Source | HuggingFace (GGUF export) |

Use the same model file across all comparison runs. Do not re-quantize between
runs.

## Measurement Procedure

1. **Warm-up phase** — Run a short generation (16-32 tokens) to warm up the
   GPU, populate caches, and trigger JIT compilation / CUDA graph capture.
   Discard these results.

2. **Measurement window** — Generate at least **256 tokens** in decode mode.
   Measure wall-clock time from the first decode token to the last.

3. **Decode-only measurement** — Report only decode throughput (tokens per
   second). Prefill / prompt-processing time is excluded from the tok/s
   number.

4. **CUDA graph coverage** — Record the percentage of operations captured in
   CUDA graphs. The current baseline achieves 99.5% coverage.

5. **Repeat** — Run the benchmark at least 3 times and report the median
   result.

## Benchmark Command

Use the `cmd/bench_tps` tool:

```bash
go run ./cmd/bench_tps \
  -model /path/to/gemma-3-1b-q4_k_m.gguf \
  -tokens 256 \
  -prompt "The quick brown fox"
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-model` | (required) | Path to GGUF model file |
| `-tokens` | 64 | Number of tokens to generate |
| `-prompt` | `""` | Input prompt text |

For official benchmarks, always use `-tokens 256` or higher.

## Current Baseline

| Model | Tok/s (decode) | CUDA Graph % | Date | Commit |
|-------|---------------|-------------|------|--------|
| Gemma 3 1B Q4_K_M | 234.30 | 99.5% | 2026-03-15 | See `docs/benchmarks.md` |

This number represents a 26% improvement over the non-graph baseline,
achieved through CUDA graph capture of the decode path.

## How to Reproduce

Step-by-step from a fresh checkout:

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

# 6. Run the warm-up pass
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 32

# 7. Run the benchmark (3 times, take the median)
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256

# 8. Record: median tok/s, CUDA graph %, commit hash
git rev-parse HEAD
```

## Comparison Methodology

When comparing against other runtimes (llama.cpp, Ollama, vLLM, etc.):

1. **Same hardware** — run both on the same DGX Spark machine
2. **Same model** — use the identical GGUF file (or equivalent quantization)
3. **Same token count** — generate the same number of tokens (256+)
4. **Same prompt** — use the same input prompt
5. **Decode-only** — compare decode tok/s, not end-to-end latency
6. **Warm up both** — give the competing runtime the same warm-up opportunity
7. **Report versions** — record exact version/commit of the competing runtime

### Example: Ollama Comparison

```bash
# Ollama baseline (same machine, same model)
ollama run gemma3:1b --verbose "The quick brown fox" 2>&1 | grep "eval rate"

# Zerfoo (same machine, same model file)
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
```

Current comparison (2026-03-15):

| Runtime | Tok/s (decode) | Notes |
|---------|---------------|-------|
| Zerfoo | 234.30 | CUDA graph capture, Q4_K_M |
| Ollama | ~197 | Same model, same hardware |

Zerfoo is 18.8% faster than Ollama on this workload.
