# zerfoo-edge — Raspberry Pi 5 Deployment

This directory contains tooling to cross-compile, deploy, and test the
`zerfoo-edge` binary on a Raspberry Pi 5 (ARM Cortex-A76, aarch64).

## Prerequisites

- Go 1.25+ on your build machine
- SSH access to the RPi5 (for `make deploy`)
- The RPi5 must run a 64-bit OS (Raspberry Pi OS 64-bit or Ubuntu 24.04 arm64)

## Quick Start

```bash
# From the repository root:

# 1. Cross-compile
make -C deploy/rpi5 build

# 2. Verify the binary was built
file zerfoo-edge-rpi5

# 3. Deploy to RPi5 (set RPI5_HOST if not at the default address)
RPI5_HOST=pi@192.168.1.x make -C deploy/rpi5 deploy

# 4. Validate on the RPi5
ssh pi@192.168.1.x '~/zerfoo/test.sh ~/zerfoo/zerfoo-edge'
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `build` | Cross-compile for arm64 linux; outputs `zerfoo-edge-rpi5` at repo root |
| `test` | Run `go vet` and check the binary exists |
| `deploy` | Build, copy binary and `test.sh` to RPi5 via `scp` |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT` | `<repo-root>/zerfoo-edge-rpi5` | Path for the output binary |
| `RPI5_HOST` | `pi@raspberrypi.local` | SSH host for `make deploy` |
| `REMOTE_DIR` | `~/zerfoo` | Remote directory on the RPi5 |
| `VERSION` | `git describe` | Version string embedded in binary |

## Build Details

The binary is compiled with:

- `GOOS=linux GOARCH=arm64` — targets RPi5's 64-bit ARM OS
- `CGO_ENABLED=0` — pure Go, no C toolchain required on the build host
- Build tags `edge,!cuda,!rocm,!opencl` — excludes GPU code paths
- `-ldflags "-s -w"` — strips debug info for a smaller binary

## Manual Cross-Compilation

```bash
GOARCH=arm64 GOOS=linux CGO_ENABLED=0 \
  go build -tags "edge,!cuda,!rocm,!opencl" \
  -ldflags "-s -w -X main.version=$(git describe --tags --always)" \
  -o zerfoo-edge-rpi5 \
  ./cmd/zerfoo-edge/
```

## Validating on RPi5

Copy `deploy/rpi5/test.sh` and the binary to the device, then run:

```bash
chmod +x test.sh zerfoo-edge
./test.sh ./zerfoo-edge
```

Expected output:

```
=== zerfoo-edge RPi5 validation ===
[PASS] binary is executable
[PASS] --version exits 0
[PASS] --help exits 0
[PASS] missing model ID returns non-zero exit code
[PASS] unknown flag returns non-zero exit code
[PASS] running on aarch64 (expected for RPi5)

=== Results: 6 passed, 0 failed ===
```

## Running Inference on RPi5

```bash
# Interactive mode
./zerfoo-edge google/gemma-3-1b

# Single-shot
./zerfoo-edge google/gemma-3-1b --prompt "What is 2+2?"

# With parameters
./zerfoo-edge google/gemma-3-1b \
  --temperature 0.7 \
  --max-tokens 256 \
  --system "You are a helpful assistant"
```

Models are cached in `~/.cache/zerfoo/` by default. Use `--cache-dir` to override.
