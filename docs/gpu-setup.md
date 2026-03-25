# GPU Setup Guide

Zerfoo supports GPU acceleration through three backends: NVIDIA CUDA, AMD ROCm, and OpenCL (via CLBlast). GPU support is entirely optional -- `go build ./...` produces a working CPU-only binary with zero configuration.

## Supported GPUs

| Backend | Vendor | Library | Loaded Via |
|---------|--------|---------|------------|
| CUDA | NVIDIA | `libcudart.so`, `libcublas.so`, `libcudnn.so` | purego/dlopen |
| ROCm | AMD | `libamdhip64.so`, `librocblas.so`, `libMIOpen.so` | purego/dlopen |
| OpenCL | Any | `libOpenCL.so` (+ CLBlast) | purego/dlopen |

Tested hardware includes NVIDIA DGX Spark, RTX 4090, A100, and H100 GPUs. AMD MI250X and MI300X are supported via ROCm. Any GPU with OpenCL support can use the CLBlast backend.

## How GPU Detection Works (purego/dlopen)

Zerfoo uses **purego/dlopen** to load GPU libraries at runtime. There are no build tags required for GPU support -- the same binary works on machines with or without a GPU.

At startup, Zerfoo calls `dlopen("libcudart.so")` (or the equivalent for ROCm/OpenCL). If the shared library is found, all CUDA runtime function pointers are resolved via `dlsym`. If the library is not found, Zerfoo falls back to CPU execution silently.

This means:

- **No CGo** -- GPU calls go through resolved function pointers, not `cgo` wrappers. Zero `runtime.cgocall` overhead.
- **No build tags** -- You do not need `-tags cuda` to build a GPU-capable binary. The default `go build ./...` produces a binary that detects GPUs at runtime.
- **Graceful fallback** -- If CUDA is not installed, `cuda.Available()` returns `false` and the engine falls back to CPU. No crashes, no missing symbol errors.

The detection flow (from `internal/cuda/purego.go`):

```
1. dlopen("libcudart.so.12") or dlopen("libcudart.so")
2. dlsym each required symbol (cudaMalloc, cudaFree, cudaMemcpy, ...)
3. Optionally resolve CUDA graph symbols (cudaStreamBeginCapture, ...)
4. cuda.Available() returns true if all required symbols resolved
```

## CUDA Setup (NVIDIA)

### Prerequisites

- NVIDIA GPU with compute capability 5.0+
- NVIDIA driver 525+ (CUDA 12.x compatible)
- CUDA Toolkit 12.0 or later (`libcudart.so.12` must be on the library path)

### Verify GPU and driver

```bash
nvidia-smi
```

Expected output shows your GPU model, driver version, and CUDA version:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.03              Driver Version: 560.35.03      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  GR Blackwell                  On   | 00000009:01:00.0  Off  |                    0 |
| N/A   40C    P0              18W / 100W |       0MiB / 131072MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Verify CUDA toolkit

```bash
nvcc --version
```

### Verify CUDA is visible to Zerfoo

Run a quick Go program to check:

```go
package main

import (
	"fmt"
	"github.com/zerfoo/zerfoo/internal/cuda"
)

func main() {
	fmt.Println("CUDA available:", cuda.Available())
	if cuda.Available() {
		count, _ := cuda.GetDeviceCount()
		fmt.Println("Device count:", count)
	}
}
```

Or from the CLI:

```bash
zerfoo run gemma-3-1b-q4  # defaults to CPU
```

To force GPU usage via the library API:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),      // use GPU 0
	inference.WithDevice("cuda:1"),    // use GPU 1
)
```

### cuBLAS

cuBLAS is included with the CUDA Toolkit. Zerfoo uses it for GEMM/GEMV operations. No additional installation needed.

### cuDNN

cuDNN provides accelerated convolution, batch norm, activation, pooling, and softmax operations. Install from the NVIDIA cuDNN archive:

```bash
# Ubuntu/Debian (CUDA 12)
sudo apt install libcudnn8 libcudnn8-dev

# Or download from NVIDIA: https://developer.nvidia.com/cudnn
```

cuDNN operations in Zerfoo are non-interface methods on `*GPUEngine` (not part of `Engine[T]`). Layers that want cuDNN acceleration type-assert to `*GPUEngine` and call these methods directly:

- `Conv2dForward` -- cuDNN convolution with grouped conv, bias, IMPLICIT_GEMM
- `BatchNormForwardInference` -- spatial batch norm
- `CudnnActivationForward` -- ReLU, Sigmoid, Tanh
- `CudnnPoolingForward` -- Max, AvgIncPad, AvgExcPad
- `CudnnSoftmaxForward` -- channel-mode softmax

See [ADR-008](adr/008-cudnn-integration.md) for architecture decisions.

### Building custom CUDA kernels

Zerfoo ships 25+ hand-written CUDA kernels for fused operations (RMSNorm, SwiGLU, RoPE, flash attention, quantized GEMM/GEMV). To build the custom kernel shared library:

```bash
cd internal/cuda/kernels && make shared && cd ../../../
```

This produces `libkernels.so`, which is loaded at runtime via `dlopen`. The kernels are optional -- without them, Zerfoo falls back to cuBLAS and element-wise GPU operations.

### NCCL (multi-GPU gradient exchange)

For distributed training with GPU-native gradient exchange, install NCCL:

```bash
# Ubuntu/Debian
sudo apt install libnccl2 libnccl-dev
```

NCCL enables direct GPU-to-GPU collective operations (AllReduce, Broadcast, Barrier) without CPU involvement. The `NcclStrategy[T]` in the `distributed/` package uses NCCL for intra-node GPU-GPU communication and falls back to gRPC for inter-node transport. Requires `//go:build cuda`.

See [ADR-007](adr/007-multi-gpu-architecture.md) for multi-GPU architecture decisions.

### TensorRT (optional)

TensorRT provides inference optimization via graph-to-TRT conversion, engine caching, and half-precision builds:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithBackend("tensorrt"),
	inference.WithPrecision("fp16"),
)
```

- Engine cache stored at `~/.cache/zerfoo/tensorrt/` (keyed by modelID, precision, gpuArch)
- Returns `UnsupportedOpError` for ops not yet mapped to TRT layers

See [ADR-009](adr/009-tensorrt-integration.md) for architecture decisions.

## ROCm Setup (AMD)

### Prerequisites

- AMD GPU (MI250X, MI300X, or other ROCm-supported cards)
- ROCm 6.0+ (`libamdhip64.so`, `librocblas.so`, `libMIOpen.so`)

### Verify GPU

```bash
rocm-smi
```

Expected output:

```
========================= ROCm System Management Interface =========================
================================ Concise Info =======================================
GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%
0    42.0c  30.0W   800Mhz  1600Mhz 0%    auto  300.0W  0%     0%
=====================================================================================
```

### Verify ROCm version

```bash
cat /opt/rocm/.info/version
# 6.0.0
```

### Use ROCm in Zerfoo

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("rocm"),
)
```

Ensure `libamdhip64.so` is on the library path:

```bash
ldconfig -p | grep amdhip
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## OpenCL Setup (CLBlast)

OpenCL works with any GPU vendor that provides an OpenCL driver.

### Install OpenCL and CLBlast

```bash
# Verify OpenCL is available
clinfo | head -20

# Ubuntu/Debian
sudo apt install libclblast-dev ocl-icd-opencl-dev

# Fedora
sudo dnf install clblast-devel ocl-icd-devel
```

### Use OpenCL in Zerfoo

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("opencl"),
)
```

## Multi-GPU

### Device selection

Specify a device ID to target a specific GPU:

```go
inference.WithDevice("cuda:0")  // first GPU
inference.WithDevice("cuda:1")  // second GPU
```

### Device affinity

The CUDA stack threads a device ID through all GPU objects:

- **GPUEngine** (`compute/gpu_engine.go`) -- `deviceID` field; calls `cuda.SetDevice()` before dispatching kernels
- **GPUStorage** (`tensor/gpu_storage.go`) -- `deviceID` field; `DeviceID()` method exposes which GPU a tensor resides on
- **MemPool** (`internal/cuda/mempool.go`) -- per-device-per-size cache prevents cross-device pointer reuse
- **CudaAllocator** (`device/cuda_allocator.go`) -- binds allocations to the correct GPU via `SetDevice()`

### Parallelism strategies

| Strategy | Complexity | Use Case |
|----------|-----------|----------|
| Data parallelism (same model on each GPU, split batches) | Medium | Training, batch inference |
| Pipeline parallelism (different layers on different GPUs) | High | Large models exceeding single GPU memory |

For distributed training across multiple GPUs or nodes, see the `distributed/` package which provides gRPC + NCCL gradient exchange. NCCL handles intra-node GPU-GPU communication; gRPC handles inter-node transport.

### Flash attention

Flash attention fuses Q*K^T -> scale -> softmax -> V into a single tiled CUDA kernel. Memory drops from O(n^2) to O(n) and four kernel launches collapse to one.

- Kernel: `internal/cuda/kernels/flash_attention.cu`
- Online softmax via log-sum-exp trick (no full n x n scores matrix)
- Shared memory staging for K/V tiles (BLOCK_SIZE=64)
- Causal masking via tile skipping + per-element mask
- MAX_HEAD_DIM=128 (covers Gemma, Llama, Mistral, Qwen, Phi)
- Automatically used by GQA and MLA when mask is nil and data is on GPU
- Falls back to naive attention for: CPU data, head_dim > 128, arbitrary masks

See [ADR-010](adr/010-cutlass-flash-attention.md) for architecture decisions.

## Performance Tuning

### Quantization choice

Lower-bit quantization reduces memory and increases throughput at the cost of quality:

| Quantization | Bits/Weight | Memory | Speed | Quality |
|-------------|-------------|--------|-------|---------|
| FP32 | 32 | Baseline | Slow | Best |
| FP16 | 16 | 2x smaller | Fast | Near-best |
| Q8_0 | 8 | 4x smaller | Faster | Good |
| Q4_K_M | 4 | 8x smaller | Fastest | Acceptable |

For most use cases, **Q4_K_M** provides the best speed/quality tradeoff. Zerfoo achieves **234 tok/s on Gemma 3 1B Q4_K_M** on a DGX Spark (19% faster than Ollama on the same hardware).

### Compute precision

Use FP16 compute precision for faster GPU inference with minimal quality loss:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),
	inference.WithDType("fp16"),     // FP16 activations
	inference.WithKVDtype("fp16"),   // FP16 KV cache (halves bandwidth)
)
```

### Context length

Longer context lengths increase KV cache memory. Reduce `MaxSeqLen` if you don't need full context:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),
	inference.WithMaxSeqLen(2048),   // default varies by model (up to 128K)
)
```

### CUDA graph capture

Zerfoo automatically captures the inference computation graph as a CUDA graph when CUDA 10.0+ graph APIs are available. This eliminates per-token kernel launch overhead and covers 99.5% of instructions on the GGUF inference path. No configuration needed -- it is enabled by default.

### VRAM requirements

| Model | Quantization | VRAM |
|-------|-------------|------|
| Gemma 3 1B | Q4_K_M | ~1 GB |
| Llama 3 8B | Q4_K_M | ~5 GB |
| Mistral 7B | Q4_K_M | ~5 GB |
| Qwen 2.5 7B | Q4_K_M | ~5 GB |

## Troubleshooting

### GPU not detected (CUDA)

**Symptom:** `cuda.Available()` returns `false` even though `nvidia-smi` shows a GPU.

**Fix:** Ensure `libcudart.so` is on the library search path:

```bash
# Check if libcudart is findable
ldconfig -p | grep cudart

# If not found, add the CUDA lib directory
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CUDA version mismatch

**Symptom:** `dlopen libcudart failed` or `dlsym cudaMalloc failed`.

**Fix:** Zerfoo looks for `libcudart.so.12` first, then `libcudart.so`. Ensure your CUDA toolkit is version 12.x:

```bash
ls /usr/local/cuda/lib64/libcudart.so*
# Should show libcudart.so.12.x.x
```

If you have CUDA 11.x, create a symlink (not recommended -- upgrade to CUDA 12):

```bash
sudo ln -s /usr/local/cuda/lib64/libcudart.so.11.8 /usr/local/cuda/lib64/libcudart.so.12
```

### Out of memory (OOM)

**Symptom:** `CUDA error: out of memory` during model loading or inference.

**Fix:** Use a more aggressively quantized model or reduce context length:

```bash
# Check GPU memory usage
nvidia-smi

# Use Q4 quantization (smallest memory footprint)
zerfoo pull gemma-3-1b-q4
zerfoo run gemma-3-1b-q4
```

From the API, reduce `max_tokens` to lower peak memory:

```go
mdl, err := inference.LoadFile("/path/to/model.gguf",
	inference.WithDevice("cuda"),
	inference.WithMaxSeqLen(2048),  // reduce from default 8192
)
```

### ROCm not detected

**Symptom:** `ROCm device requested but binary built without rocm build tag`

**Fix:** Check that `libamdhip64.so` is on the library path:

```bash
ldconfig -p | grep amdhip
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### OpenCL not detected

**Fix:** Ensure `libOpenCL.so` and CLBlast are installed:

```bash
ldconfig -p | grep -i opencl
# If missing:
sudo apt install ocl-icd-opencl-dev libclblast-dev
```

## Example: DGX Spark Setup

The DGX Spark at `ssh ndungu@192.168.86.250` provides a reference GPU environment:

```bash
ssh ndungu@192.168.86.250

# Verify GPU
nvidia-smi

# Run Gemma 3 1B inference
cd ~/Code/zerfoo/zerfoo
go run ./cmd/zerfoo pull gemma-3-1b-q4
go run ./cmd/zerfoo run gemma-3-1b-q4

# Run with GPU acceleration
go run ./cmd/zerfoo serve gemma-3-1b-q4 --port 8080
```

## Historical Investigation Reports

- [gpu-engine-audit.md](gpu-engine-audit.md) — Phase 32 audit of GPUEngine H2D/D2H transfer patterns
- [cuda-perop-perf-analysis.md](cuda-perop-perf-analysis.md) — Root cause analysis of CUDA per-op path being slower than CPU (resolved: cuBLAS converted to purego)

## Architecture Decision Records

- [ADR-006](adr/006-gpu-engine-architecture.md) — Pre-Phase-10 GPU architecture
- [ADR-007](adr/007-multi-gpu-architecture.md) — Multi-GPU architecture
- [ADR-008](adr/008-cudnn-integration.md) — cuDNN integration
- [ADR-009](adr/009-tensorrt-integration.md) — TensorRT integration
- [ADR-010](adr/010-cutlass-flash-attention.md) — CUTLASS flash attention
