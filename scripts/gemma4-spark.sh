#!/usr/bin/env bash
# gemma4-spark.sh — submit the Gemma 4 E2B end-to-end test to Spark on the DGX
# host, poll the pod until it terminates, stream its logs, and exit with its
# status.
#
# Usage:
#   scripts/gemma4-spark.sh [-gguf /var/lib/zerfoo/models/gemma-4-E2B-it-Q4_K_M.gguf] [-cleanup]
#
# Prerequisites (one-time DGX staging):
#   - Build the binary for linux/arm64 and place it on the DGX host at
#     /var/lib/zerfoo/bin/gemma4_e2e:
#
#         GOOS=linux GOARCH=arm64 go build -o gemma4_e2e ./cmd/gemma4_e2e
#         rsync -av gemma4_e2e ndungu@192.168.86.250:/var/lib/zerfoo/bin/
#
#   - Copy the GGUF to /var/lib/zerfoo/models/ on the DGX (the manifest mounts
#     /var/lib/zerfoo/models read-only into the pod).
#
# Submit wrapper only — does no staging of binaries or models. See
# docs/adr/083-spark-bench-runner.md for the host-access policy.

set -euo pipefail

SPARK_HOST="${SPARK_HOST:-192.168.86.250:8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_TEMPLATE="${REPO_ROOT}/docs/bench/manifests/gemma4-e2e.yaml"

GGUF_PATH="/var/lib/zerfoo/models/gemma-4-E2B-it-Q4_K_M.gguf"
CLEANUP=0

usage() {
  cat >&2 <<USAGE
Usage: $0 [-gguf /path/to/model.gguf] [-cleanup]

Submits docs/bench/manifests/gemma4-e2e.yaml to Spark at \${SPARK_HOST}
(currently ${SPARK_HOST}), polls until the pod terminates, prints logs,
and exits with the pod's success/failure status.

Default GGUF path: ${GGUF_PATH}
USAGE
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -gguf)     GGUF_PATH="$2"; shift 2 ;;
    -cleanup)  CLEANUP=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -f "${MANIFEST_TEMPLATE}" ]] || { echo "missing manifest: ${MANIFEST_TEMPLATE}" >&2; exit 3; }

RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
POD_NAME="gemma4-e2e-${RUN_ID}"

MANIFEST_RENDERED="$(
  sed \
    -e "s|\${RUN_ID}|${RUN_ID}|g" \
    -e "s|\${GGUF_PATH}|${GGUF_PATH}|g" \
    "${MANIFEST_TEMPLATE}"
)"

echo "gemma4-spark: submitting ${POD_NAME} to http://${SPARK_HOST}"
SUBMIT_RESP="$(
  printf '%s' "${MANIFEST_RENDERED}" | curl -sf -X POST \
    -H 'Content-Type: application/yaml' \
    --data-binary @- \
    "http://${SPARK_HOST}/api/v1/pods"
)" || { echo "gemma4-spark: submit failed" >&2; exit 5; }

echo "gemma4-spark: submitted"
printf '%s\n' "${SUBMIT_RESP}" | head -c 500
echo

PHASE=""
for _ in $(seq 1 2400); do  # 2h cap at 3s/tick
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
    "")               echo "gemma4-spark: pod not found yet; retrying" >&2 ;;
    *)                ;;
  esac
  sleep 3
done

echo "gemma4-spark: pod status=${PHASE}; fetching logs"
curl -sf "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}/logs" || echo "(log fetch failed)"

if [[ "${CLEANUP}" -eq 1 ]]; then
  echo
  echo "gemma4-spark: deleting pod ${POD_NAME}"
  curl -sf -X DELETE "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}" >/dev/null || true
fi

case "${PHASE}" in
  completed) exit 0 ;;
  failed)    exit 1 ;;
  *)         exit 2 ;;
esac
