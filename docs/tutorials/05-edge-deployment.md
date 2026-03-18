# Tutorial 5: Edge Deployment with zerfoo-edge

`zerfoo-edge` is a minimal, self-contained inference binary that strips out training, distributed serving, GPU backends, and the HTTP API layer. The result is a small binary suitable for running GGUF models on resource-constrained ARM devices such as the Raspberry Pi 5 or NVIDIA Jetson Orin Nano.

Because Zerfoo has no CGo by default, cross-compilation is a standard `GOOS`/`GOARCH` flag — no toolchain setup, no Docker build containers.

## What Is Included (and Excluded)

| Included | Excluded |
|----------|----------|
| GGUF model loading | Training (`training/`) |
| CPU inference | Distributed training (`distributed/`) |
| Interactive REPL | HTTP API server (`serve/`) |
| Single-shot `--prompt` mode | GPU backends (CUDA, ROCm, OpenCL) |
| Chat and streaming generation | Tabular and AutoML |

## Step 1: Build for the Host Platform

```bash
# From the zerfoo repository root
go build -tags edge -o zerfoo-edge ./cmd/zerfoo-edge/
./zerfoo-edge --version
# zerfoo-edge dev
```

## Step 2: Cross-Compile for ARM64 (Raspberry Pi 5)

```bash
GOOS=linux GOARCH=arm64 go build -tags edge \
  -o zerfoo-edge-arm64 ./cmd/zerfoo-edge/
```

No C compiler or cross-toolchain is needed. The binary is statically linked pure Go.

Copy the binary and a GGUF model to the device:

```bash
scp zerfoo-edge-arm64 pi@raspberrypi.local:~/zerfoo-edge
scp gemma-3-1b-it-qat-q4_0.gguf pi@raspberrypi.local:~/models/
```

## Step 3: Cross-Compile for ARM64 with NEON SIMD

Zerfoo's `internal/xblas` package includes hand-written ARM NEON SIMD assembly for GEMV and GEMM operations. The assembly is automatically selected when `GOARCH=arm64` — no build tags required.

```bash
GOOS=linux GOARCH=arm64 go build -tags edge \
  -ldflags "-X main.version=1.0.0" \
  -o zerfoo-edge-arm64 ./cmd/zerfoo-edge/
```

The `-ldflags "-X main.version=..."` flag embeds a version string that appears in `--version` output.

## Step 4: Cross-Compile for x86-64 (Intel/AMD edge servers)

```bash
GOOS=linux GOARCH=amd64 go build -tags edge \
  -o zerfoo-edge-amd64 ./cmd/zerfoo-edge/
```

AVX2 SIMD paths in `internal/xblas` are used automatically on amd64.

## Step 5: Run on the Device

### Interactive Mode

```bash
./zerfoo-edge google/gemma-3-1b
# Loading model google/gemma-3-1b-it-qat-q4_0-gguf...
# Model loaded.
# Type your message (Ctrl-D to quit).
# > What is 2+2?
# 2+2 equals 4.
# >
```

The model ID resolves to the short alias registered in `inference.modelAliases`. You can also pass a full HuggingFace repo ID or a local file path.

### Single-Shot Mode

```bash
./zerfoo-edge google/gemma-3-1b --prompt "Explain neural networks briefly"
# Neural networks are...
```

### With a System Prompt

```bash
./zerfoo-edge google/gemma-3-1b \
  --system "You are a helpful assistant for embedded systems developers." \
  --prompt "What is I2C?"
```

### Control Generation Parameters

```bash
./zerfoo-edge google/gemma-3-1b \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9 \
  --max-tokens 256 \
  --prompt "Write a Python function to read a DHT22 sensor."
```

### Use a Local Model File

```bash
./zerfoo-edge /home/pi/models/gemma-3-1b-it-qat-q4_0.gguf \
  --prompt "Hello"
```

### Override the Cache Directory

```bash
./zerfoo-edge google/gemma-3-1b \
  --cache-dir /mnt/ssd/models \
  --prompt "Hello"
```

## Step 6: Use zerfoo-edge as a Library in Your Own Go Binary

If you want to embed inference into your own edge application rather than using the standalone binary, import the `inference` package directly:

```go
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	ctx := context.Background()

	// Load from a local GGUF file — no network access needed.
	mdl, err := inference.LoadFile("/opt/models/gemma-3-1b.gguf")
	if err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Fprint(os.Stderr, "> ")
		if !scanner.Scan() {
			break
		}
		prompt := scanner.Text()

		err := mdl.GenerateStream(ctx, prompt,
			generate.TokenStreamFunc(func(token string, done bool) error {
				fmt.Fprint(os.Stdout, token)
				return nil
			}),
			inference.WithTemperature(0.7),
			inference.WithMaxTokens(512),
		)
		if err != nil {
			log.Println("generate:", err)
		}
		fmt.Fprintln(os.Stdout)
	}
}
```

Build and cross-compile exactly as above:

```bash
GOOS=linux GOARCH=arm64 go build -o my-edge-app .
```

## Step 7: Minimize Binary Size

Strip debug symbols to reduce binary size for storage-constrained devices:

```bash
GOOS=linux GOARCH=arm64 go build -tags edge \
  -ldflags "-s -w -X main.version=1.0.0" \
  -o zerfoo-edge-arm64 ./cmd/zerfoo-edge/
```

| Flag | Effect |
|------|--------|
| `-s` | Strip symbol table |
| `-w` | Strip DWARF debug info |

Typical binary sizes after stripping:

| Target | Size |
|--------|------|
| `linux/arm64` | ~8 MB |
| `linux/amd64` | ~9 MB |
| `darwin/arm64` (Apple Silicon) | ~8 MB |

## Supported Targets

Any target supported by `go tool dist list` works. Common edge targets:

```
linux/arm64    — Raspberry Pi 4/5, Jetson Orin, Apple M1 servers
linux/amd64    — Intel/AMD edge servers, NUC devices
linux/arm      — Raspberry Pi 3 and earlier (32-bit ARMv7)
darwin/arm64   — Apple Silicon (M1/M2/M3)
```

## CLI Reference

```
zerfoo-edge <model-id> [OPTIONS]

OPTIONS:
  --prompt <text>              Single-shot prompt (non-interactive)
  --system <text>              System prompt
  --temperature <float>        Sampling temperature (default: 1.0)
  --top-k <int>                Top-K sampling
  --top-p <float>              Top-P nucleus sampling
  --repetition-penalty <float> Penalize repeated tokens
  --max-tokens <int>           Maximum tokens to generate
  --cache-dir <dir>            Override model cache directory
  --version                    Print version and exit
  --help                       Print this help
```

## Next Steps

- [Tutorial 1: Getting Started](01-getting-started.md) — use the full zerfoo library in a Go program
- [Tutorial 2: Chatbot API](02-chatbot-api.md) — serve over HTTP if your edge device has enough headroom
- Read `cmd/zerfoo-edge/main.go` to see the complete edge binary implementation (~300 lines)
