#!/usr/bin/env bash
# nsight-decode-profile.sh -- Profile the decode hot path with Nsight Systems on DGX Spark.
#
# Usage (on DGX Spark):
#   ./scripts/nsight-decode-profile.sh /path/to/model.gguf [tokens] [prompt]
#
# Prerequisites:
#   - Nsight Systems installed (nsys)
#   - Go toolchain
#   - CUDA toolkit
#
# Output:
#   - nsight_decode_<timestamp>.nsys-rep  (Nsight Systems report)
#   - nsight_decode_<timestamp>.sqlite    (SQLite database for analysis)
#   - nsight_decode_stats.txt             (kernel summary)
#
# The script builds bench_tps with CUDA support, runs it under nsys for a
# controlled number of decode tokens, then extracts top kernel durations.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?Usage: $0 /path/to/model.gguf [tokens] [prompt]}"
TOKENS="${2:-128}"
PROMPT="${3:-The meaning of life is}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_NAME="nsight_decode_${TIMESTAMP}"

echo "=== Zerfoo Decode Hot Path Profiling ==="
echo "Model:  ${MODEL}"
echo "Tokens: ${TOKENS}"
echo "Prompt: \"${PROMPT}\""
echo ""

# Build bench_tps with CUDA support.
echo "Building bench_tps (CUDA)..."
cd "${REPO_ROOT}"
go build -tags cuda -o /tmp/bench_tps ./cmd/bench_tps/
echo "Build complete."
echo ""

# Run under Nsight Systems.
# - trace cuda,nvtx,osrt for GPU kernels, NVTX markers, and OS runtime
# - sample=none to reduce overhead on CPU sampling
# - capture-range=cudaProfilerApi allows programmatic capture (future)
# - gpu-metrics-device=all for SM occupancy and memory throughput
echo "Running under Nsight Systems..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    --output="${REPO_ROOT}/${REPORT_NAME}" \
    /tmp/bench_tps \
        -model "${MODEL}" \
        -device cuda \
        -tokens "${TOKENS}" \
        -temp 0 \
        -prompt "${PROMPT}"

echo ""
echo "Nsight report: ${REPO_ROOT}/${REPORT_NAME}.nsys-rep"
echo ""

# Extract kernel statistics from the report.
echo "=== Top 20 CUDA Kernels by Total Time ==="
nsys stats \
    --report gputrace \
    --format csv \
    "${REPO_ROOT}/${REPORT_NAME}.nsys-rep" 2>/dev/null | \
    head -25 || echo "(nsys stats not available — open .nsys-rep in Nsight Systems GUI)"

echo ""
echo "=== GPU Kernel Summary (aggregated) ==="
nsys stats \
    --report gpukernsum \
    --format csv \
    "${REPO_ROOT}/${REPORT_NAME}.nsys-rep" 2>/dev/null | \
    head -15 || echo "(nsys stats not available — open .nsys-rep in Nsight Systems GUI)"

echo ""
echo "=== CUDA API Summary ==="
nsys stats \
    --report cudaapisum \
    --format csv \
    "${REPO_ROOT}/${REPORT_NAME}.nsys-rep" 2>/dev/null | \
    head -15 || echo "(nsys stats not available — open .nsys-rep in Nsight Systems GUI)"

# Also run with cuBLAS profiling for GEMM breakdown.
echo ""
echo "=== cuBLAS Kernel Profiling ==="
ZERFOO_PROFILE_CUBLAS=1 /tmp/bench_tps \
    -model "${MODEL}" \
    -device cuda \
    -tokens "${TOKENS}" \
    -temp 0 \
    -prompt "${PROMPT}" 2>&1 | tee "${REPO_ROOT}/nsight_decode_stats.txt"

echo ""
echo "=== Profiling Complete ==="
echo "Reports:"
echo "  Nsight: ${REPO_ROOT}/${REPORT_NAME}.nsys-rep"
echo "  Stats:  ${REPO_ROOT}/nsight_decode_stats.txt"
echo ""
echo "Open in Nsight Systems GUI:"
echo "  nsys-ui ${REPO_ROOT}/${REPORT_NAME}.nsys-rep"
