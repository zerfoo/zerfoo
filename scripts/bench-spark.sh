#!/usr/bin/env bash
# bench-spark.sh — submit a PatchTST training bench to Spark on the DGX host,
# poll the pod until it terminates, stream its logs, and exit with its status.
#
# Usage:
#   scripts/bench-spark.sh -samples 5000 -channels 10 -epochs 3 [-cleanup]
#
# Flags are passed through to bench_train verbatim; -cleanup deletes the pod
# after logs are fetched (default: leave the pod for inspection via
# `curl http://${SPARK_HOST}/api/v1/pods/<name>`).
#
# Requires: bash, curl, python3,
#           network access to ${SPARK_HOST} (default 192.168.86.250:8080).
#
# This replaces the deprecated `ssh dgx 'bench_train ...'` pattern that
# leaked SSH channels and took the DGX down on 2026-04-07. See
# docs/adr/083-spark-bench-runner.md and docs/plans/spark-bench-runner.md.

set -euo pipefail

SPARK_HOST="${SPARK_HOST:-192.168.86.250:8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_TEMPLATE="${REPO_ROOT}/docs/bench/manifests/patchtst-train.yaml"

SAMPLES=""
CHANNELS=""
EPOCHS=""
CLEANUP=0

usage() {
  cat >&2 <<USAGE
Usage: $0 -samples N -channels C -epochs E [-cleanup]

Submits docs/bench/manifests/patchtst-train.yaml to Spark at
\${SPARK_HOST} (currently ${SPARK_HOST}), polls until the pod terminates,
prints logs, and exits with the pod's success/failure status.
USAGE
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -samples)  SAMPLES="$2"; shift 2 ;;
    -channels) CHANNELS="$2"; shift 2 ;;
    -epochs)   EPOCHS="$2"; shift 2 ;;
    -cleanup)  CLEANUP=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -z "${SAMPLES}" || -z "${CHANNELS}" || -z "${EPOCHS}" ]] && usage
[[ -f "${MANIFEST_TEMPLATE}" ]] || { echo "missing manifest: ${MANIFEST_TEMPLATE}" >&2; exit 3; }

RUN_ID="$(date -u +%Y%m%d-%H%M%S)-s${SAMPLES}-c${CHANNELS}-e${EPOCHS}"
POD_NAME="bench-patchtst-${RUN_ID}"

# Substitute only the four variables we own; anything else passes through.
MANIFEST_RENDERED="$(
  sed \
    -e "s|\${RUN_ID}|${RUN_ID}|g" \
    -e "s|\${SAMPLES}|${SAMPLES}|g" \
    -e "s|\${CHANNELS}|${CHANNELS}|g" \
    -e "s|\${EPOCHS}|${EPOCHS}|g" \
    "${MANIFEST_TEMPLATE}"
)"

# Spark's /api/v1/pods reads the request body via its own indent-based YAML
# parser, so we POST raw YAML (not JSON). Content-Type is informational.
echo "bench-spark: submitting ${POD_NAME} to http://${SPARK_HOST}"
SUBMIT_RESP="$(
  printf '%s' "${MANIFEST_RENDERED}" | curl -sf -X POST \
    -H 'Content-Type: application/yaml' \
    --data-binary @- \
    "http://${SPARK_HOST}/api/v1/pods"
)" || { echo "bench-spark: submit failed" >&2; exit 5; }

echo "bench-spark: submitted"
printf '%s\n' "${SUBMIT_RESP}" | head -c 500
echo

# Poll for terminal status. Spark phases: pending|scheduled|running|completed|failed.
PHASE=""
for _ in $(seq 1 7200); do  # 6 hours max at 3s/tick
  STATUS_JSON="$(curl -sf "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}" || echo '{}')"
  PHASE="$(
    printf '%s' "${STATUS_JSON}" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
s = d.get("status", "")
print(s if isinstance(s, str) else "")
'
  )"
  case "${PHASE}" in
    completed|failed) break ;;
    "")               echo "bench-spark: pod not found yet; retrying" >&2 ;;
    *)                ;;
  esac
  sleep 3
done

echo "bench-spark: pod status=${PHASE}; fetching logs"
curl -sf "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}/logs" || echo "(log fetch failed)"

if [[ "${CLEANUP}" -eq 1 ]]; then
  echo
  echo "bench-spark: deleting pod ${POD_NAME}"
  curl -sf -X DELETE "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}" >/dev/null || true
fi

case "${PHASE}" in
  completed) exit 0 ;;
  failed)    exit 1 ;;
  *)         exit 2 ;;
esac
