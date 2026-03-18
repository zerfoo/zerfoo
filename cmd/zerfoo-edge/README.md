# zerfoo-edge

Minimal edge/embedded inference binary for Zerfoo. Supports CPU-only GGUF model inference with no training, distributed, serving, GPU, or AutoML dependencies.

## Build

```bash
go build -tags edge ./cmd/zerfoo-edge/
```

Cross-compile for ARM64 (e.g., Raspberry Pi 5):

```bash
GOOS=linux GOARCH=arm64 go build -tags edge ./cmd/zerfoo-edge/
```

## Usage

Interactive mode:

```bash
./zerfoo-edge google/gemma-3-1b
```

Single-shot mode:

```bash
./zerfoo-edge google/gemma-3-1b --prompt "What is 2+2?"
```

With generation parameters:

```bash
./zerfoo-edge google/gemma-3-1b \
  --temperature 0.7 \
  --max-tokens 512 \
  --system "You are a helpful assistant"
```

## Options

| Flag | Description |
|------|-------------|
| `--prompt <text>` | Single-shot prompt (exits after generating) |
| `--system <text>` | System prompt |
| `--temperature <float>` | Sampling temperature (default: 1.0) |
| `--top-k <int>` | Top-K sampling |
| `--top-p <float>` | Top-P nucleus sampling |
| `--repetition-penalty <float>` | Penalize repeated tokens |
| `--max-tokens <int>` | Maximum tokens to generate |
| `--cache-dir <dir>` | Override model cache directory |
| `--version` | Print version and exit |

## What's excluded

The edge binary intentionally excludes:

- Training (`training/`)
- Distributed training (`distributed/`)
- HTTP/API server (`serve/`)
- GPU backends (CUDA, ROCm, OpenCL)
- AutoML and NAS
- Tabular model support
