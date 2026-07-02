#!/usr/bin/env bash
# dgx-validate.sh -- one command to run zerfoo's GPU validation suite natively
# on the DGX GB10 via Spark.
#
# purego GPU bindings cannot cross-compile darwin->linux/arm64 (runtime.dlopen
# linknames need cgo), so all GPU-touching build+test must happen ON the DGX.
# This wraps docs/bench/manifests/validate-arm64.yaml: it renders a unique pod,
# POSTs it to Spark, polls to completion, streams logs, extracts the JSON
# report, deletes the pod, and propagates a green/red exit code.
#
# Usage:
#   scripts/dgx-validate.sh [-ref <git-ref>] [-timeout <seconds>] [-dry-run]
#
#   -ref <git-ref>     git ref (branch/tag/SHA) to validate. Default: the
#                      current origin/main SHA.
#   -timeout <seconds> max seconds to wait for the pod to terminate. Default 1800.
#   -dry-run           render the manifest and print the API calls; submit nothing.
#
# Environment:
#   SPARK        base URL of the Spark API. Default http://192.168.86.250:8080.
#                (SPARK_HOST=host:port is also honored, for parity with
#                 scripts/bench-spark.sh.)
#   SPARK_TOKEN  optional; sent as "Authorization: Bearer <token>" on every call.
#
# Requires: bash, curl, git, sed, grep, date, mktemp. No third-party deps.
#
# GPU runs are serialized on the host (SPARK_GPU_MAX=1): only one GPU pod runs
# at a time, so coordinate with any other DGX GPU work before submitting.

set -euo pipefail

# --- config ------------------------------------------------------------------

SPARK="${SPARK:-}"
if [ -z "$SPARK" ]; then
  if [ -n "${SPARK_HOST:-}" ]; then
    SPARK="http://${SPARK_HOST}"
  else
    SPARK="http://192.168.86.250:8080"
  fi
fi
SPARK="${SPARK%/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_TEMPLATE="${REPO_ROOT}/docs/bench/manifests/validate-arm64.yaml"

REF=""
DRY_RUN=0
TIMEOUT=1800
POLL_INTERVAL="${POLL_INTERVAL:-5}"

usage() {
  cat >&2 <<USAGE
Usage: $0 [-ref <git-ref>] [-timeout <seconds>] [-dry-run] [-keep] [-no-pull] [-delete <pod-name>]

Submits docs/bench/manifests/validate-arm64.yaml to Spark at \${SPARK}
(currently ${SPARK}), polls until the pod terminates, streams logs, extracts
the JSON report, deletes the pod, and exits 0 only on Succeeded + a report
with no failures. Pre-pulls the pod image via Spark's image API (skip with
-no-pull). On failure, pod events are fetched and printed before deletion;
-keep skips deletion entirely so the pod can be inspected.
USAGE
  exit 2
}

KEEP=0
PREPULL=1
DELETE_POD=""
while [ $# -gt 0 ]; do
  case "$1" in
    -ref)      [ $# -ge 2 ] || usage; REF="$2"; shift 2 ;;
    -timeout)  [ $# -ge 2 ] || usage; TIMEOUT="$2"; shift 2 ;;
    -dry-run)  DRY_RUN=1; shift ;;
    -keep)     KEEP=1; shift ;;
    -no-pull)  PREPULL=0; shift ;;
    -delete)   [ $# -ge 2 ] || usage; DELETE_POD="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

case "$TIMEOUT" in
  ''|*[!0-9]*) echo "dgx-validate: -timeout must be a positive integer" >&2; exit 2 ;;
esac

[ -f "$MANIFEST_TEMPLATE" ] || { echo "dgx-validate: missing manifest: $MANIFEST_TEMPLATE" >&2; exit 3; }

if [ -z "$REF" ]; then
  REF="$(git -C "$REPO_ROOT" rev-parse origin/main 2>/dev/null || true)"
  if [ -z "$REF" ]; then
    echo "dgx-validate: could not resolve origin/main; pass -ref explicitly" >&2
    exit 3
  fi
fi

# Unique pod name: zerfoo-validate-<shortref>-<epoch>. Strip anything that is
# not alphanumeric from the ref so branch names with slashes stay DNS-safe.
SHORT="$(printf '%s' "$REF" | tr -cd '[:alnum:]' | cut -c1-12)"
[ -n "$SHORT" ] || SHORT="ref"
EPOCH="$(date -u +%s)"
POD_NAME="zerfoo-validate-${SHORT}-${EPOCH}"

# Substitute only the two variables we own. Refs/SHAs cannot contain '|', so it
# is a safe sed delimiter here.
MANIFEST="$(
  sed \
    -e "s|\${POD_NAME}|${POD_NAME}|g" \
    -e "s|\${REF}|${REF}|g" \
    "$MANIFEST_TEMPLATE"
)"

# --- HTTP with retry ---------------------------------------------------------

# Auth header as an array so it expands to zero args when unset (bash 3.2 safe).
AUTH_HEADER=()
if [ -n "${SPARK_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${SPARK_TOKEN}")
fi

HTTP_BODY=""
HTTP_CODE=""

# http_req METHOD URL [extra curl args...]
# Retries 3x with 2s/4s/8s backoff on 429, 5xx, and connection errors.
# Never retries other 4xx. Sets HTTP_BODY and HTTP_CODE; returns 0 on 2xx.
http_req() {
  local method="$1" url="$2"
  shift 2
  local attempt=1 delay=2 code body tmp
  while : ; do
    tmp="$(mktemp)"
    code="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" \
              "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" "$@" "$url" 2>/dev/null)" || code=000
    body="$(cat "$tmp")"; rm -f "$tmp"
    [ -n "$code" ] || code=000
    HTTP_BODY="$body"; HTTP_CODE="$code"

    if [ "$code" -ge 200 ] 2>/dev/null && [ "$code" -lt 300 ] 2>/dev/null; then
      return 0
    fi
    # Any 4xx other than 429 is a client error: do not retry.
    if [ "$code" -ge 400 ] 2>/dev/null && [ "$code" -lt 500 ] 2>/dev/null && [ "$code" != "429" ]; then
      return 1
    fi
    if [ "$attempt" -ge 3 ]; then
      return 1
    fi
    echo "dgx-validate: ${method} ${url} -> HTTP ${code}; retry ${attempt}/3 in ${delay}s" >&2
    sleep "$delay"
    delay=$((delay * 2))
    attempt=$((attempt + 1))
  done
}

# Extract a terminal-ish phase from a pod JSON body. Handles both k8s-style
# "phase":"Succeeded" and Spark's flat "status":"completed" without a JSON
# parser (the values are simple string literals).
extract_phase() {
  printf '%s' "$1" \
    | grep -oiE '"(phase|status)"[[:space:]]*:[[:space:]]*"[a-z]+"' \
    | grep -oiE '"[a-z]+"$' \
    | tr -d '"' \
    | head -1
}

# --- delete-only mode ----------------------------------------------------------

if [ -n "$DELETE_POD" ]; then
  if http_req DELETE "${SPARK}/api/v1/pods/${DELETE_POD}"; then
    echo "dgx-validate: deleted pod ${DELETE_POD}"
    exit 0
  fi
  echo "dgx-validate: delete failed for ${DELETE_POD} (HTTP ${HTTP_CODE})" >&2
  exit 5
fi

# --- dry-run -----------------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
  echo "# dgx-validate DRY RUN -- nothing will be submitted"
  echo "# SPARK=${SPARK}"
  echo "# ref=${REF}"
  echo "# pod=${POD_NAME}"
  if [ -n "${SPARK_TOKEN:-}" ]; then
    echo "# auth: Authorization: Bearer <SPARK_TOKEN> (set)"
  else
    echo "# auth: none (SPARK_TOKEN unset)"
  fi
  echo
  echo "# --- rendered manifest ---"
  printf '%s\n' "$MANIFEST"
  echo
  echo "# --- API calls that would be issued ---"
  echo "POST   ${SPARK}/api/v1/pods            (Content-Type: application/yaml, body = manifest above)"
  echo "GET    ${SPARK}/api/v1/pods/${POD_NAME}   (poll every ${POLL_INTERVAL}s until Succeeded/Failed, timeout ${TIMEOUT}s)"
  echo "GET    ${SPARK}/api/v1/pods/${POD_NAME}/logs"
  echo "DELETE ${SPARK}/api/v1/pods/${POD_NAME}"
  exit 0
fi

# --- pre-pull image ------------------------------------------------------------

# A cold host has no arm64 golang image; an in-pod pull can exceed the pod's
# startup grace and surface as an instant Failed with no logs (observed on the
# first live run, 2026-07-02). Pull explicitly first; failure is non-fatal.
if [ "$PREPULL" -eq 1 ]; then
  IMAGE="$(printf '%s\n' "$MANIFEST" | grep -m1 -E '^[[:space:]]*image:' | sed -E 's/^[[:space:]]*image:[[:space:]]*//')"
  if [ -n "$IMAGE" ]; then
    echo "dgx-validate: ensuring image present on host: ${IMAGE}"
    if ! http_req POST "${SPARK}/api/v1/images/pull" \
          -H 'Content-Type: application/json' \
          --data "{\"image\":\"${IMAGE}\"}" --max-time 900; then
      echo "dgx-validate: warning: image pre-pull failed (HTTP ${HTTP_CODE}); continuing -- in-pod pull may still succeed" >&2
    fi
  fi
fi

# --- submit ------------------------------------------------------------------

echo "dgx-validate: submitting ${POD_NAME} to ${SPARK} (ref=${REF})"
if ! http_req POST "${SPARK}/api/v1/pods" \
      -H 'Content-Type: application/yaml' --data-binary "$MANIFEST"; then
  echo "dgx-validate: submit failed (HTTP ${HTTP_CODE})" >&2
  printf '%s\n' "$HTTP_BODY" | head -c 800 >&2
  echo >&2
  exit 5
fi
echo "dgx-validate: submitted; polling for terminal phase"

# --- poll --------------------------------------------------------------------

DEADLINE=$(( $(date +%s) + TIMEOUT ))
PHASE=""
while : ; do
  if http_req GET "${SPARK}/api/v1/pods/${POD_NAME}"; then
    RAW_PHASE="$(extract_phase "$HTTP_BODY")"
    case "$RAW_PHASE" in
      [Ss]ucceeded|[Cc]ompleted) PHASE="Succeeded"; break ;;
      [Ff]ailed)                 PHASE="Failed";    break ;;
    esac
  else
    # 404 right after submit is normal until the pod registers; keep polling.
    echo "dgx-validate: poll HTTP ${HTTP_CODE}; retrying" >&2
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    PHASE="timeout"
    break
  fi
  sleep "$POLL_INTERVAL"
done

# --- logs + report -----------------------------------------------------------

echo "dgx-validate: phase=${PHASE}; fetching logs"
if http_req GET "${SPARK}/api/v1/pods/${POD_NAME}/logs"; then
  LOGS="$HTTP_BODY"
else
  LOGS="(log fetch failed: HTTP ${HTTP_CODE})"
fi
printf '%s\n' "$LOGS"

# The in-pod script prints the report as the final line starting with {"ref":.
REPORT="$(printf '%s\n' "$LOGS" | grep -E '\{"ref":' | tail -1 || true)"
if [ -n "$REPORT" ]; then
  echo "dgx-validate: report: ${REPORT}"
else
  echo "dgx-validate: no JSON report line found in logs" >&2
fi

# On any non-success, pull the pod's event stream BEFORE deleting -- events
# carry image-pull / container-create diagnoses that logs cannot (a pod whose
# container never started has no logs at all).
if [ "$PHASE" != "Succeeded" ]; then
  echo "dgx-validate: fetching pod events for diagnosis"
  if http_req GET "${SPARK}/api/v1/pods/${POD_NAME}/events"; then
    printf '%s\n' "$HTTP_BODY"
  else
    echo "dgx-validate: events fetch failed (HTTP ${HTTP_CODE})" >&2
  fi
fi

# --- cleanup -----------------------------------------------------------------

if [ "$KEEP" -eq 1 ]; then
  echo "dgx-validate: -keep set; leaving pod ${POD_NAME} for inspection:"
  echo "  curl ${SPARK}/api/v1/pods/${POD_NAME}"
  echo "  curl ${SPARK}/api/v1/pods/${POD_NAME}/logs"
  echo "  curl -X DELETE ${SPARK}/api/v1/pods/${POD_NAME}"
elif [ "$PHASE" != "timeout" ]; then
  if http_req DELETE "${SPARK}/api/v1/pods/${POD_NAME}"; then
    echo "dgx-validate: deleted pod ${POD_NAME}"
  else
    echo "dgx-validate: warning: could not delete pod ${POD_NAME} (HTTP ${HTTP_CODE})" >&2
  fi
else
  echo "dgx-validate: timed out after ${TIMEOUT}s; leaving pod ${POD_NAME} for inspection:" >&2
  echo "  curl ${SPARK}/api/v1/pods/${POD_NAME}" >&2
  echo "  curl ${SPARK}/api/v1/pods/${POD_NAME}/logs" >&2
  echo "  curl -X DELETE ${SPARK}/api/v1/pods/${POD_NAME}" >&2
fi

# --- exit code ---------------------------------------------------------------

if [ "$PHASE" = "Succeeded" ]; then
  if [ -n "$REPORT" ] && printf '%s' "$REPORT" | grep -q '"failures":\[\]'; then
    echo "dgx-validate: PASS"
    exit 0
  fi
  echo "dgx-validate: pod Succeeded but report is missing or lists failures" >&2
  exit 1
fi

echo "dgx-validate: FAIL (phase=${PHASE})" >&2
exit 1
