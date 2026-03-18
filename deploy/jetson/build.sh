#!/usr/bin/env bash
# build.sh — Cross-compile zerfoo-edge for NVIDIA Jetson Orin Nano (CPU-only)
#
# Produces: zerfoo-edge-jetson (ARM64 Linux binary, no GPU)
# Run from the repository root or deploy/jetson/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${OUTPUT:-${REPO_ROOT}/zerfoo-edge-jetson}"
VERSION="${VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty 2>/dev/null || echo dev)}"

echo "Building zerfoo-edge for Jetson Orin Nano (GOARCH=arm64 GOOS=linux)..."
echo "  Version : ${VERSION}"
echo "  Output  : ${OUTPUT}"

GOOS=linux GOARCH=arm64 CGO_ENABLED=0 \
  go build \
    -tags edge \
    -ldflags "-X main.version=${VERSION} -s -w" \
    -o "${OUTPUT}" \
    "${REPO_ROOT}/cmd/zerfoo-edge"

echo "Build successful: ${OUTPUT}"
file "${OUTPUT}"
