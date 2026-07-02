#!/usr/bin/env bash
# bisect-dgx.sh — build a specific commit on DGX and run bench via Spark.
# Usage: scripts/bisect-dgx.sh <commit-sha> <label> [-samples N] [-channels C] [-epochs E]
# Default: 10000 samples, 15 channels, 5 epochs. Timeout: 180s.

set -euo pipefail

SPARK_HOST="${SPARK_HOST:-192.168.86.250:8080}"
DGX="ndungu@192.168.86.250"
SHA="${1:?usage: bisect-dgx.sh <sha> <label> [-samples N] [-channels C] [-epochs E]}"
LABEL="${2:?usage: bisect-dgx.sh <sha> <label>}"
shift 2

SAMPLES=10000; CHANNELS=15; EPOCHS=5; TIMEOUT=180
while [[ $# -gt 0 ]]; do
  case "$1" in
    -samples)  SAMPLES="$2"; shift 2 ;;
    -channels) CHANNELS="$2"; shift 2 ;;
    -epochs)   EPOCHS="$2"; shift 2 ;;
    -timeout)  TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

POD_NAME="bisect-${LABEL}-$(date -u +%H%M%S)"
echo "=== Bisect: ${LABEL} @ ${SHA} ==="
echo "Shape: ${SAMPLES}x${CHANNELS}x${EPOCHS}, timeout=${TIMEOUT}s"

# 1. Build on DGX
echo "--- Building on DGX ---"
ssh "${DGX}" "export PATH=\$PATH:/usr/local/go/bin && cd ~/zerfoo && git fetch origin -q && git checkout ${SHA} -q && git reset --hard ${SHA} -q && go build -o /var/lib/zerfoo/bin/bench_train ./cmd/bench_train" || {
  echo "RESULT: BUILD_FAIL"; exit 1
}
echo "Build OK"

# 2. Submit manifest
echo "--- Submitting pod ${POD_NAME} ---"
MANIFEST="$(cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: docker.io/library/ubuntu:24.04
      command:
        - /var/lib/zerfoo/bin/bench_train
      args:
        - "-samples"
        - "${SAMPLES}"
        - "-channels"
        - "${CHANNELS}"
        - "-epochs"
        - "${EPOCHS}"
      env:
        - name: LD_LIBRARY_PATH
          value: /usr/local/cuda/lib64
      resources:
        limits:
          memory: 32Gi
          cpu: "8"
          nvidia.com/gpu: "1"
      volumeMounts:
        - name: zerfoo-bin
          mountPath: /var/lib/zerfoo/bin
          readOnly: true
        - name: zerfoo-lib
          mountPath: /opt/zerfoo/lib
          readOnly: true
        - name: cuda
          mountPath: /usr/local/cuda
          readOnly: true
        - name: bench-out
          mountPath: /var/lib/zerfoo/bench-out
  volumes:
    - name: zerfoo-bin
      hostPath:
        path: /var/lib/zerfoo/bin
        type: Directory
    - name: zerfoo-lib
      hostPath:
        path: /opt/zerfoo/lib
        type: Directory
    - name: cuda
      hostPath:
        path: /usr/local/cuda
        type: Directory
    - name: bench-out
      hostPath:
        path: /var/lib/zerfoo/bench-out
        type: DirectoryOrCreate
YAML
)"

echo "${MANIFEST}" | curl -sS -X POST "http://${SPARK_HOST}/api/v1/pods" \
  -H 'Content-Type: application/yaml' --data-binary @- || {
  echo "RESULT: SUBMIT_FAIL"; exit 1
}
echo ""

# 3. Poll
ELAPSED=0
while [[ ${ELAPSED} -lt ${TIMEOUT} ]]; do
  sleep 10
  ELAPSED=$((ELAPSED + 10))
  STATUS=$(curl -sS "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "error")
  echo "[${ELAPSED}s] status=${STATUS}"
  if [[ "${STATUS}" == "completed" ]] || [[ "${STATUS}" == "failed" ]]; then
    break
  fi
done

# 4. Fetch logs
echo "--- Logs ---"
LOGS=$(curl -sS "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}/logs" 2>/dev/null)
echo "${LOGS}"

# 5. Cleanup
curl -sS -X DELETE "http://${SPARK_HOST}/api/v1/pods/${POD_NAME}" >/dev/null 2>&1

# 6. Verdict
if [[ "${STATUS}" != "completed" ]]; then
  echo "RESULT: FAIL (status=${STATUS} after ${ELAPSED}s)"
  exit 1
fi

if echo "${LOGS}" | grep -q "out of memory\|OOM\|cudaMalloc failed"; then
  echo "RESULT: FAIL (OOM)"
  exit 1
fi

if echo "${LOGS}" | grep -q "convergence: OK"; then
  TOTAL=$(echo "${LOGS}" | grep "total:" | sed 's/.*total: //')
  echo "RESULT: PASS (${TOTAL})"
  exit 0
fi

echo "RESULT: UNKNOWN"
exit 1
