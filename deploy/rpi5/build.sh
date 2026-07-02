#!/usr/bin/env bash
# build.sh — Cross-compile zerfoo-edge for Raspberry Pi 5 (arm64 linux).
#
# Usage:
#   ./deploy/rpi5/build.sh
#   OUTPUT=/path/to/binary ./deploy/rpi5/build.sh
#
# Requirements:
#   - Go 1.25+
#   - Run from the zerfoo repository root

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT="${OUTPUT:-${REPO_ROOT}/zerfoo-edge-rpi5}"
VERSION="${VERSION:-$(git -C "${REPO_ROOT}" describe --tags --always --dirty 2>/dev/null || echo dev)}"

echo "Building zerfoo-edge for Raspberry Pi 5 (GOOS=linux GOARCH=arm64)..."
echo "  Output:  ${OUTPUT}"
echo "  Version: ${VERSION}"

cd "${REPO_ROOT}"

GOOS=linux \
GOARCH=arm64 \
CGO_ENABLED=0 \
go build \
  -tags "edge,!cuda,!rocm,!opencl" \
  -ldflags "-s -w -X main.version=${VERSION}" \
  -o "${OUTPUT}" \
  ./cmd/zerfoo-edge/

echo "Build successful: ${OUTPUT}"
file "${OUTPUT}" 2>/dev/null || true
