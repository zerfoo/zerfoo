#!/bin/bash
# Generate some random prompts
words=("alpha" "beta" "gamma" "delta" "stock" "crypto" "market" "trend" "signal" "risk" "matrix" "tensor")
for i in {1..10}; do
  sentence=""
  for j in {1..5}; do
    sentence+="${words[$RANDOM % ${#words[@]}]} "
  done
  echo $sentence
done
