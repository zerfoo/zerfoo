#!/usr/bin/env bash
# dgx-validate-inpod.sh -- the in-pod stage of scripts/dgx-validate.sh.
#
# Runs INSIDE the validate-arm64 Spark pod on the DGX GB10, from the root of a
# fresh clone at the target ref (the pod's bootstrap one-liner in
# docs/bench/manifests/validate-arm64.yaml clones and checks out, then execs
# this script). It is a committed file rather than inline pod YAML because
# Spark's indent-based YAML parser does not preserve literal block scalars in
# `args` (observed live 2026-07-02: the container ran `/bin/bash -c '|'`).
#
# Emits a single-line JSON report as the final stdout line and exits nonzero
# on any failure:
#   {"ref":...,"build":...,"vet":...,"cuda_tests":...,"parity":...,"failures":[...]}
set -uo pipefail

REF="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
MODELS_DIR="${MODELS_DIR:-/var/lib/zerfoo/models}"
CUDA_PKGS="${CUDA_PKGS:-./internal/cuda/... ./tabular/... ./tests/parity/...}"

FAILS=""
add_fail() { FAILS="${FAILS:+$FAILS,}\"$1\""; }

emit() {
  printf '{"ref":"%s","build":"%s","vet":"%s","cuda_tests":"%s","parity":"%s","failures":[%s]}\n' \
    "$REF" "$1" "$2" "$3" "$4" "$FAILS"
}

echo ">> zerfoo arm64 GPU validation: ref=$REF host=$(uname -m) go=$(go version)"

# The -tags cuda packages compile CGo against the CUDA headers/libs from the
# read-only /usr/local/cuda mount.
export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

BUILD=pass
echo ">> go build ./..."
go build ./... || { BUILD=fail; add_fail build; }

# Vet policy matches docs/QUALITY.md: the purego dlopen bindings carry known,
# intentional "possible misuse of unsafe.Pointer" findings (only visible when
# vetting on linux, where the linux_arm64 files are in scope). Fail vet only
# if anything OTHER than that class remains.
VET=pass
echo ">> go vet ./..."
VET_OUT="$(go vet ./... 2>&1)" || true
VET_RESIDUE="$(printf '%s\n' "$VET_OUT" | grep -vE 'possible misuse of unsafe\.Pointer' | grep -vE '^#|^$' || true)"
if [ -n "$VET_RESIDUE" ]; then
  printf '%s\n' "$VET_OUT"
  VET=fail; add_fail vet
else
  printf '%s\n' "$VET_OUT" | grep -E 'possible misuse' | sed 's/^/>> vet (allowed, QUALITY.md purego class): /' || true
fi

CUDA=pass
echo ">> go test -tags cuda $CUDA_PKGS"
# shellcheck disable=SC2086
go test -tags cuda -count=1 -timeout 900s $CUDA_PKGS || { CUDA=fail; add_fail cuda_tests; }

# Model parity: only when GGUF files are mounted. Each model parity test skips
# itself when its ModelDirEnvVar is unset, so we discover those env-var names
# from the source and point them all at MODELS_DIR.
PARITY=skip
if [ -d "$MODELS_DIR" ] && [ -n "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
  for v in $(grep -rhoE 'ModelDirEnvVar:[[:space:]]*"[A-Z0-9_]+"' tests/parity 2>/dev/null \
             | grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort -u); do
    export "$v=$MODELS_DIR"
  done
  PARITY=pass
  echo ">> go test -tags cuda ./tests/parity/... (models present)"
  go test -tags cuda -count=1 -timeout 900s ./tests/parity/... || { PARITY=fail; add_fail parity; }
else
  echo ">> model parity skipped (no files under $MODELS_DIR)"
fi

emit "$BUILD" "$VET" "$CUDA" "$PARITY"
[ -z "$FAILS" ] || exit 1
