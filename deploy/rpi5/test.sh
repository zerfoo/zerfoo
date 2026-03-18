#!/usr/bin/env bash
# test.sh — Validate the zerfoo-edge binary on Raspberry Pi 5.
#
# Copy this script and the binary to the RPi5, then run:
#   chmod +x test.sh zerfoo-edge
#   ./test.sh ./zerfoo-edge
#
# Usage:
#   ./test.sh [path/to/zerfoo-edge]

set -euo pipefail

BINARY="${1:-./zerfoo-edge}"
PASS=0
FAIL=0

pass() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "=== zerfoo-edge RPi5 validation ==="
echo "Binary: ${BINARY}"
echo ""

# 1. Binary exists and is executable.
if [ -x "${BINARY}" ]; then
  pass "binary is executable"
else
  fail "binary not found or not executable: ${BINARY}"
  exit 1
fi

# 2. --version flag returns zero exit code.
if "${BINARY}" --version >/dev/null 2>&1; then
  VERSION_OUT=$("${BINARY}" --version 2>&1)
  pass "--version exits 0 (output: ${VERSION_OUT})"
else
  fail "--version returned non-zero"
fi

# 3. --help flag returns zero exit code.
if "${BINARY}" --help >/dev/null 2>&1; then
  pass "--help exits 0"
else
  fail "--help returned non-zero"
fi

# 4. Missing model ID returns error (non-zero).
if ! "${BINARY}" >/dev/null 2>&1; then
  pass "missing model ID returns non-zero exit code"
else
  fail "missing model ID should return non-zero"
fi

# 5. Unknown flag returns error (non-zero).
if ! "${BINARY}" --bogus-flag >/dev/null 2>&1; then
  pass "unknown flag returns non-zero exit code"
else
  fail "unknown flag should return non-zero"
fi

# 6. Architecture check.
ARCH="$(uname -m)"
if [ "${ARCH}" = "aarch64" ]; then
  pass "running on aarch64 (expected for RPi5)"
else
  echo "[WARN] running on ${ARCH}, not aarch64 — binary may not be native"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [ "${FAIL}" -gt 0 ]; then
  exit 1
fi
