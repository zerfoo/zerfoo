# Jetson Orin Nano Deployment

Deploy `zerfoo-edge` to an NVIDIA Jetson Orin Nano for on-device ML inference.

## Prerequisites

### Host (cross-compilation machine)

- Go 1.25+ (`go version`)
- For CPU-only build: no extra toolchain required (`CGO_ENABLED=0`)
- For CUDA build: `aarch64-linux-gnu-gcc` cross-compiler
  ```bash
  # Debian/Ubuntu
  sudo apt-get install gcc-aarch64-linux-gnu
  ```

### Jetson device

- NVIDIA Jetson Orin Nano (8 GB or 4 GB module)
- JetPack 6.x (ships CUDA 12.x, cuDNN 9.x)
- SSH access from the host machine

JetPack SDK installs CUDA at `/usr/local/cuda` on the device. This is the default
`CUDA_PATH` used by `build-cuda.sh`.

## Quick Start

### 1. Build (CPU-only)

```bash
# From the repository root
make -f deploy/jetson/Makefile build
# or directly
bash deploy/jetson/build.sh
```

Produces `zerfoo-edge-jetson` in the repository root — a statically linked ARM64
Linux binary with no GPU dependencies.

### 2. Build (with CUDA)

```bash
make -f deploy/jetson/Makefile build-cuda
# or
bash deploy/jetson/build-cuda.sh
```

Produces `zerfoo-edge-jetson-cuda`. Requires the ARM64 cross-compiler and JetPack
headers in `CUDA_PATH`. If building natively on the Jetson itself, set
`CGO_ENABLED=1` and run `go build -tags "edge cuda" ./cmd/zerfoo-edge`.

### 3. Deploy to Jetson

```bash
# Set JETSON_HOST to your device's hostname or IP
make -f deploy/jetson/Makefile deploy JETSON_HOST=192.168.1.100

# Or deploy manually
scp zerfoo-edge-jetson user@jetson:~/
scp deploy/jetson/test.sh user@jetson:~/
```

### 4. Run on Jetson

```bash
ssh user@jetson

# Interactive inference
./zerfoo-edge-jetson google/gemma-3-1b

# Single-shot
./zerfoo-edge-jetson google/gemma-3-1b --prompt "What is 2+2?" --max-tokens 64
```

### 5. Validate

```bash
# From the host (runs test.sh over SSH)
make -f deploy/jetson/Makefile test JETSON_HOST=192.168.1.100

# Or directly on the device
bash ~/test.sh
```

## Make Targets

| Target | Description |
|--------|-------------|
| `build` | Cross-compile CPU-only binary for ARM64 Linux |
| `build-cuda` | Cross-compile with CUDA support (requires JetPack toolchain) |
| `deploy` | `scp` binary and test script to Jetson |
| `test` | Run on-device validation via SSH |
| `clean` | Remove built binaries |

Override SSH target:
```bash
make -f deploy/jetson/Makefile deploy JETSON_HOST=192.168.1.100 JETSON_USER=ndungu
```

## CUDA Toolkit Requirements (JetPack SDK)

The CUDA build (`build-cuda.sh`) relies on headers and libraries from the JetPack SDK.

| Component | JetPack 6.x path | Notes |
|-----------|------------------|-------|
| CUDA Toolkit | `/usr/local/cuda` | Set `CUDA_PATH` to override |
| cuDNN | `/usr/lib/aarch64-linux-gnu/libcudnn*` | Installed by JetPack |
| NCCL | `/usr/lib/aarch64-linux-gnu/libnccl*` | Optional, for distributed |

Zerfoo loads GPU libraries at runtime via `dlopen` — no static linking of CUDA
required. The binary itself remains CGo-free for the CPU build; CGo is only used
when the `cuda` build tag is set.

### Verifying CUDA on the Jetson

```bash
# On the Jetson device
nvcc --version          # CUDA compiler
nvidia-smi              # GPU status
ls /usr/local/cuda/lib64/libcudart*   # Runtime library
```

JetPack 6.0 ships CUDA 12.2. JetPack 6.1+ ships CUDA 12.6.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT` | `<repo-root>/zerfoo-edge-jetson` | Output binary path |
| `VERSION` | `git describe` | Version string embedded in binary |
| `CUDA_PATH` | `/usr/local/cuda` | CUDA Toolkit installation path |
| `JETSON_HOST` | `jetson` | SSH hostname/IP for `make deploy` and `make test` |
| `JETSON_USER` | current user | SSH username |
| `DEPLOY_DIR` | `~` | Deployment directory on device |
| `BINARY` | `~/zerfoo-edge-jetson` | Binary path used by `test.sh` |
| `MODEL` | `google/gemma-3-1b` | Model ID used by smoke test |

## Notes

- The CPU-only build uses `CGO_ENABLED=0` and produces a fully static binary — no
  shared library dependencies on the Jetson.
- The CUDA build enables runtime GPU acceleration via `dlopen`; the binary falls
  back to CPU if CUDA libraries are not present at runtime.
- Models are downloaded on first use to `~/.cache/zerfoo` (or `--cache-dir`).
- The Jetson Orin Nano's Ampere GPU supports FP16 natively; FP8 requires
  Ada Lovelace or newer (not available on Orin Nano).
