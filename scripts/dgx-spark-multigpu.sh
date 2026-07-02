#!/usr/bin/env bash
# dgx-spark-multigpu.sh -- Run multi-GPU tests on DGX Spark.
#
# Usage:
#   ssh ndungu@192.168.86.250
#   cd /home/ndungu/Code/zerfoo
#   ./scripts/dgx-spark-multigpu.sh
#
# Prerequisites:
#   - Two DGX Spark GB10 units connected via ConnectX-7 200 Gb/s QSFP cable
#   - Go installed at $HOME/.local/go/bin
#   - CUDA toolkit at /usr/local/cuda
#   - CUTLASS headers at $HOME/cutlass/include
#   - NCCL >= 2.28.3 installed (libnccl-dev)
#   - cuda.GetDeviceCount() returns >= 2
set -euo pipefail

# CUDA / CGo environment
export PATH="/usr/local/cuda/bin:$HOME/.local/go/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu"
export CGO_CFLAGS="-I/usr/local/cuda/include -I/usr/include/aarch64-linux-gnu -I$HOME/cutlass/include"

# NCCL configuration for ConnectX-7 RoCE
# Set NCCL_SOCKET_IFNAME to the QSFP/RoCE network interface.
# Uncomment and adjust the interface name as needed:
# export NCCL_SOCKET_IFNAME=enp1s0f0
# export NCCL_DEBUG=INFO

echo "=== Multi-GPU Test Runner ==="
echo ""
echo "Checking GPU device count..."
DEVICE_COUNT=$(go run -tags cuda -e 'package main; import "fmt"; import "github.com/zerfoo/zerfoo/internal/cuda"; func main() { c, _ := cuda.GetDeviceCount(); fmt.Println(c) }' 2>/dev/null || echo "0")
echo "  CUDA devices detected: ${DEVICE_COUNT:-unknown}"
echo ""

echo "--- Memory pool cross-device tests ---"
go test -tags cuda ./internal/cuda/ -v -count=1 -run "NoCrossDevice|MultiDevice" -timeout 60s
echo ""

echo "--- NCCL multi-GPU tests ---"
go test -tags cuda ./internal/nccl/ -v -count=1 -run "TwoGPU" -timeout 120s
echo ""

echo "--- NCCL strategy multi-GPU tests ---"
go test -tags cuda ./distributed/ -v -count=1 -run "TwoGPU" -timeout 120s
echo ""

echo "--- Multi-GPU inference parity test ---"
go test -tags cuda,cutlass ./tests/parity/ -v -count=1 -run "MultiGPU" -timeout 300s
echo ""

echo "=== All multi-GPU tests complete ==="
