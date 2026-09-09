#!/usr/bin/env bash
# Targeted Phase R eager-fp32 training gate. Run only through its Spark manifest.
set -uo pipefail
export ZERFOO_R_MODEL_GPU=1
export GOMAXPROCS=4
export GOFLAGS=-p=2
export CPATH="/usr/local/cuda/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="/usr/local/cuda/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
echo "reference=$(git rev-parse HEAD) architecture=$(uname -m)"
go version
log=$(mktemp)
go test -json -count=1 -timeout 900s -run '^TestClassifierTrainingGPU$' ./tabular 2>&1 | tee "$log"
result=${PIPESTATUS[0]}
if ! grep -q '"Action":"pass".*"Test":"TestClassifierTrainingGPU"' "$log"; then
  echo 'ERROR: required GPU test did not pass'
  result=1
fi
if grep -q '"Action":"skip".*"Test":"TestClassifierTrainingGPU"' "$log"; then
  echo 'ERROR: required GPU test skipped'
  result=1
fi
trials=$(grep -c 'device=.*hidden=.*seed=.*steps=1200' "$log" || true)
if [ "$trials" -ne 6 ]; then
  echo "ERROR: expected 6 completed GPU recipe/seed runs, got $trials"
  result=1
fi
printf '{"gate":"model-creation-gpu-training","completed_trials":%s,"exit_code":%s}\n' "$trials" "$result"
exit "$result"
