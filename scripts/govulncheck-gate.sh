#!/usr/bin/env bash
# govulncheck-gate.sh -- run govulncheck and fail the build unless every reachable
# vulnerability it reports is on the explicit allowlist below.
#
# Context (CICD-6, docs/deep-reviews/002-full-codebase.md): go.etcd.io/bbolt v1.4.3 --
# the latest release, pinned in go.mod -- carries advisory GO-2026-4923 (index
# out-of-range panic on a zero-element branch page), which has no upstream fix. The CI
# vulnerability-check step used to work around that with a blanket
# `continue-on-error: true`, which silently swallowed *every* govulncheck finding, not
# just the unfixable bbolt one. This script replaces that: it fails on any reachable
# vulnerability except the ones named in ALLOWLIST below.
#
# Note: as of this writing GO-2026-4923 was itself WITHDRAWN by the reporter/maintainer
# as a false positive (2026-04-08), so a clean govulncheck run no longer reports it at
# all. It stays allowlisted defensively -- bbolt has shipped no newer release, so if the
# advisory database ever reinstates the report there is still nothing to upgrade to.
#
# Usage:
#   scripts/govulncheck-gate.sh ./...
#
# For testing, set GOVULNCHECK_REPORT to a pre-built `govulncheck -format json` report
# file to skip invoking govulncheck and filter that file instead.
set -uo pipefail

ALLOWLIST=(
  GO-2026-4923 # go.etcd.io/bbolt v1.4.3, no fix available; WITHDRAWN upstream 2026-04-08
)

report="${GOVULNCHECK_REPORT:-}"
workdir=""
if [ -z "$report" ]; then
  workdir=$(mktemp -d)
  report="$workdir/govulncheck.json"
  # govulncheck exits non-zero when it finds vulnerabilities -- that's expected here,
  # the JSON report is the source of truth this script filters below.
  govulncheck -format json "$@" > "$report" || true
fi
cleanup() {
  [ -n "$workdir" ] && rm -rf "$workdir"
}
trap cleanup EXIT

# A "finding" is reachable/actionable (what govulncheck's own text output numbers as
# "Vulnerability #N") when its trace terminates in a resolved call site, i.e. the last
# frame has a "function" key. Findings whose trace stops at a bare module or package
# name (no "function") are govulncheck's lower-priority "imported but not called" /
# "required but not imported" context rows and are not build-breaking on their own.
found=()
while IFS= read -r id; do
  [ -n "$id" ] && found+=("$id")
done < <(
  jq -r '
    select(.finding != null)
    | select(.finding.trace != null and (.finding.trace | length) > 0)
    | select(.finding.trace[-1] | has("function"))
    | .finding.osv
  ' "$report" | sort -u
)

if [ "${#found[@]}" -eq 0 ]; then
  echo "govulncheck: no reachable vulnerabilities found."
  exit 0
fi

bad=()
for id in "${found[@]}"; do
  allowed=false
  for a in "${ALLOWLIST[@]}"; do
    if [ "$id" = "$a" ]; then
      allowed=true
      break
    fi
  done
  [ "$allowed" = false ] && bad+=("$id")
done

if [ "${#bad[@]}" -gt 0 ]; then
  echo "::error::govulncheck found un-allowlisted vulnerabilities: ${bad[*]} (all reachable IDs: ${found[*]})" >&2
  exit 1
fi

echo "govulncheck: only allowlisted advisories present (${ALLOWLIST[*]}); reachable IDs were: ${found[*]}"
