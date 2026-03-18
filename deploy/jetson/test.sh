#!/usr/bin/env bash
# test.sh — On-device validation for zerfoo-edge on Jetson Orin Nano
#
# Run this script *on the Jetson device* after deploying the binary.
# Assumes the binary is at ~/zerfoo-edge-jetson or $BINARY path.

set -euo pipefail

BINARY="${BINARY:-${HOME}/zerfoo-edge-jetson}"
MODEL="${MODEL:-google/gemma-3-1b}"

if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: Binary not found or not executable: ${BINARY}" >&2
  echo "  Deploy with: scp zerfoo-edge-jetson jetson:~/" >&2
  exit 1
fi

echo "=== zerfoo-edge Jetson validation ==="
echo "  Binary : ${BINARY}"
echo "  Model  : ${MODEL}"
echo ""

# 1. Version check
echo "--- version ---"
"${BINARY}" --version

# 2. Architecture sanity
echo ""
echo "--- binary info ---"
file "${BINARY}"
uname -m

# 3. Single-shot inference smoke test
echo ""
echo "--- inference smoke test ---"
OUTPUT=$("${BINARY}" "${MODEL}" --prompt "What is 2+2?" --max-tokens 16 2>/dev/null)
if [[ -z "${OUTPUT}" ]]; then
  echo "FAIL: no output produced" >&2
  exit 1
fi
echo "Output: ${OUTPUT}"
echo "PASS"

# 4. GPU availability check (informational, non-fatal)
echo ""
echo "--- GPU check (informational) ---"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi present but no GPU found"
else
  echo "nvidia-smi not in PATH — install JetPack SDK for GPU support"
fi

echo ""
echo "=== All checks passed ==="
