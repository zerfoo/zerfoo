#!/usr/bin/env bash
# build-cuda.sh — Cross-compile zerfoo-edge for NVIDIA Jetson Orin Nano with CUDA
#
# Requires JetPack SDK on the Jetson device or a cross-compilation sysroot with:
#   - CUDA Toolkit for ARM64 (typically /usr/local/cuda on Jetson)
#   - aarch64-linux-gnu-gcc toolchain
#
# JetPack 6.x ships CUDA 12.x. Set CUDA_PATH to override.
#
# Run from the repository root or deploy/jetson/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${OUTPUT:-${REPO_ROOT}/zerfoo-edge-jetson-cuda}"
VERSION="${VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty 2>/dev/null || echo dev)}"

# JetPack installs CUDA at /usr/local/cuda on the device.
# When cross-compiling from x86, point CUDA_PATH at the ARM64 sysroot's CUDA.
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

echo "Building zerfoo-edge (CUDA) for Jetson Orin Nano (GOARCH=arm64 GOOS=linux)..."
echo "  Version   : ${VERSION}"
echo "  Output    : ${OUTPUT}"
echo "  CUDA_PATH : ${CUDA_PATH}"

# Require the ARM64 cross-compiler when building from a non-ARM64 host.
HOST_ARCH="$(uname -m)"
if [[ "${HOST_ARCH}" != "aarch64" ]]; then
  if ! command -v aarch64-linux-gnu-gcc &>/dev/null; then
    echo "ERROR: aarch64-linux-gnu-gcc not found." >&2
    echo "  On Debian/Ubuntu: sudo apt-get install gcc-aarch64-linux-gnu" >&2
    exit 1
  fi
  export CC=aarch64-linux-gnu-gcc
  export CXX=aarch64-linux-gnu-g++
fi

GOOS=linux GOARCH=arm64 CGO_ENABLED=1 \
  go build \
    -tags "edge cuda" \
    -ldflags "-X main.version=${VERSION} -s -w" \
    -o "${OUTPUT}" \
    "${REPO_ROOT}/cmd/zerfoo-edge"

echo "Build successful: ${OUTPUT}"
file "${OUTPUT}"
