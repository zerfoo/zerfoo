#!/usr/bin/env bash
# bench.sh -- Run key benchmarks and emit JSON metrics to stdout.
#
# Usage:
#   ./scripts/bench.sh
#   ./scripts/bench.sh | jq .
#
# Each line of output is a JSON object:
#   {"metric": "gemm_gflops_1024", "value": 18.57, "unit": "GFLOPS"}
#
# Requires: go, awk
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Run benchmarks and capture output.
GEMM_OUT=$(go test -bench='BenchmarkGEMM' -benchmem -count=1 -run='^$' \
    -timeout 300s "${REPO_ROOT}/tests/benchmark/" 2>&1) || true
KV_OUT=$(go test -bench='BenchmarkKVCacheUpdate' -benchmem -count=1 -run='^$' \
    -timeout 300s "${REPO_ROOT}/tests/benchmark/" 2>&1) || true
ALLOC_OUT=$(go test -bench='BenchmarkMemoryAllocs' -benchmem -count=1 -run='^$' \
    -timeout 300s "${REPO_ROOT}/tests/benchmark/" 2>&1) || true
Q4_OUT=$(go test -bench='BenchmarkGemmQ4F32_GEMV' -benchmem -count=1 -run='^$' \
    -timeout 300s "${REPO_ROOT}/internal/xblas/" 2>&1) || true

# Extract GEMM GFLOPS for each size.
extract_gemm_gflops() {
    local size="$1"
    echo "${GEMM_OUT}" | awk -v sz="${size}x${size}" '
        $0 ~ "BenchmarkGEMM/" sz {
            for (i = 1; i <= NF; i++) {
                if ($(i+1) == "GFLOPS") {
                    printf "{\"metric\": \"gemm_gflops_%s\", \"value\": %s, \"unit\": \"GFLOPS\"}\n", "'"${size}"'", $i
                    exit
                }
            }
        }'
}

for sz in 128 512 1024; do
    result=$(extract_gemm_gflops "${sz}")
    if [ -n "${result}" ]; then
        echo "${result}"
    fi
done

# Extract Q4 GEMV ns/op and convert to tok/s equivalent (1 op = 1 token).
echo "${Q4_OUT}" | awk '
    /BenchmarkGemmQ4F32_GEMV\/fused/ {
        for (i = 1; i <= NF; i++) {
            if ($(i+1) == "ns/op") {
                ns = $i
                toks = 1e9 / ns
                printf "{\"metric\": \"q4_gemv_tok_per_sec\", \"value\": %.2f, \"unit\": \"tok/s\"}\n", toks
                exit
            }
        }
    }'

# Extract KV cache updates/s.
echo "${KV_OUT}" | awk '
    /BenchmarkKVCacheUpdate/ {
        for (i = 1; i <= NF; i++) {
            if ($(i+1) == "updates/s") {
                printf "{\"metric\": \"kv_cache_updates_per_sec\", \"value\": %.2f, \"unit\": \"updates/s\"}\n", $i
                exit
            }
        }
    }'

# Extract memory allocs per op from BenchmarkMemoryAllocs.
echo "${ALLOC_OUT}" | awk '
    /BenchmarkMemoryAllocs/ {
        for (i = 1; i <= NF; i++) {
            if ($(i+1) == "allocs/op") {
                printf "{\"metric\": \"allocs_per_decode_step\", \"value\": %s, \"unit\": \"allocs/op\"}\n", $i
                exit
            }
        }
    }'
