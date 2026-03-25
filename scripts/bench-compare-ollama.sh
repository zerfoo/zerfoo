#!/usr/bin/env bash
# bench-compare-ollama.sh -- Benchmark Zerfoo vs Ollama on multiple models.
#
# Usage:
#   ./scripts/bench-compare-ollama.sh                    # Run all models
#   ./scripts/bench-compare-ollama.sh gemma3-1b          # Run single model
#   ./scripts/bench-compare-ollama.sh --list              # List configured models
#
# Prerequisites:
#   - Go installed and in PATH
#   - CUDA toolkit available (benchmarks use -device cuda)
#   - bench_tps binary built: go build -o bench_tps ./cmd/bench_tps/
#   - Ollama installed and serving (for Ollama-supported models)
#   - GGUF model files in $MODEL_BASE/<name>/model.gguf
#
# Outputs:
#   results/benchmark-YYYY-MM-DD.json    -- Machine-readable results
#   results/benchmark-YYYY-MM-DD.md      -- Formatted comparison table
#
# Methodology:
#   - Prompt: "Explain the theory of relativity in simple terms."
#   - Tokens: 128 generated tokens (decode phase)
#   - Warmup: bench_tps has built-in warmup; Ollama warms up on first run
#   - Sampling: greedy (temperature=0)
#   - Runs: 3 per model per runtime, median reported
#   - Metric: tokens per second during decode (eval rate)
set -euo pipefail

# --- Configuration ---

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="$(date +%Y-%m-%d)"
RESULTS_DIR="${REPO_ROOT}/results"
JSON_OUT="${RESULTS_DIR}/benchmark-${DATE}.json"
MD_OUT="${RESULTS_DIR}/benchmark-${DATE}.md"
MODEL_BASE="${MODEL_BASE:-$HOME/models}"
BENCH_TPS="${REPO_ROOT}/bench_tps"
PROMPT="Explain the theory of relativity in simple terms."
TOKENS=128
RUNS=3
COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# CUDA / Go environment
export PATH="/usr/local/cuda/bin:/usr/local/go/bin:$HOME/.local/go/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Model registry: name|architecture|gguf_path|ollama_name|size
# ollama_name is "NONE" for models without Ollama support.
MODELS=(
  "gemma3-1b|gemma3|gemma3-q4km/model.gguf|gemma3:1b|1B"
  "gemma3-4b|gemma3|gemma3-4b-q4km/model.gguf|gemma3:4b|4B"
  "llama3.2-3b|llama|llama3.2-3b-q4km/model.gguf|llama3.2:3b|3B"
  "llama3.1-8b|llama|llama3.1-8b-q4km/model.gguf|llama3.1:8b|8B"
  "mistral-7b|mistral|mistral-7b/model.gguf|mistral:7b|7B"
  "mixtral-8x7b|mixtral|mixtral-8x7b-q4km/model.gguf|mixtral:8x7b|47B"
  "qwen2.5-7b|qwen2|qwen2.5-7b-q4km/model.gguf|qwen2.5:7b|7B"
  "phi3-mini|phi3|phi-3.5-mini/model.gguf|phi3:mini|3.8B"
  "deepseek-r1-1.5b|deepseek2|deepseek-r1-1.5b-q4km/model.gguf|deepseek-r1:1.5b|1.5B"
  "command-r-35b|command-r|command-r-35b-q4km/model.gguf|command-r:35b|35B"
  "falcon-7b|falcon|falcon-7b-q4km/model.gguf|NONE|7B"
  "mamba-2.8b|mamba|mamba-2.8b-q4km/model.gguf|NONE|2.8B"
  "rwkv-7b|rwkv|rwkv-7b-q4km/model.gguf|NONE|7B"
)

# --- Functions ---

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }
err() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; }

# Build bench_tps if not already built.
build_bench_tps() {
  if [ ! -x "$BENCH_TPS" ]; then
    log "Building bench_tps..."
    (cd "$REPO_ROOT" && go build -o bench_tps ./cmd/bench_tps/)
  fi
}

# Run bench_tps and extract tok/s from output.
# Returns the tok/s value or "FAIL" on error.
run_zerfoo() {
  local gguf_path="$1"
  local output
  output=$("$BENCH_TPS" \
    -model "$gguf_path" \
    -prompt "$PROMPT" \
    -tokens "$TOKENS" \
    -device cuda \
    -temp 0 2>&1) || { echo "FAIL"; return; }

  # Parse: "Throughput: 245.32 tok/s"
  echo "$output" | awk '/Throughput:/ { for (i=1; i<=NF; i++) if ($i == "tok/s" && i > 1) print $(i-1) }' | tail -1
}

# Run Ollama and extract eval rate (tok/s) from verbose output.
# Returns the tok/s value or "FAIL" on error.
run_ollama() {
  local model_name="$1"
  local stderr_file
  stderr_file="$(mktemp)"

  # Use OLLAMA_NUM_PREDICT to control token count.
  OLLAMA_NUM_PREDICT="$TOKENS" ollama run "$model_name" \
    --verbose --nowordwrap \
    "$PROMPT" >/dev/null 2>"$stderr_file" || { rm -f "$stderr_file"; echo "FAIL"; return; }

  # Parse: "eval rate:       204.32 tokens/s"
  local tps
  # Match "eval rate:" but NOT "prompt eval rate:" -- strip leading whitespace first.
  tps=$(sed 's/^[[:space:]]*//' "$stderr_file" | awk '/^eval rate:/ { for (i=1; i<=NF; i++) if ($i == "tokens/s" && i > 1) print $(i-1) }')
  rm -f "$stderr_file"

  if [ -z "$tps" ]; then
    echo "FAIL"
  else
    echo "$tps"
  fi
}

# Compute median of space-separated values.
median() {
  echo "$@" | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {
    if (NR%2==1) print a[(NR+1)/2]
    else printf "%.2f\n", (a[NR/2]+a[NR/2+1])/2
  }'
}

# Benchmark a single model: run RUNS times, return median tok/s.
benchmark_zerfoo() {
  local gguf_path="$1"
  local results=()
  for i in $(seq 1 "$RUNS"); do
    local tps
    tps=$(run_zerfoo "$gguf_path")
    if [ "$tps" = "FAIL" ] || [ -z "$tps" ]; then
      err "Zerfoo run $i failed for $gguf_path"
      return 1
    fi
    results+=("$tps")
    log "  Zerfoo run $i: ${tps} tok/s"
  done
  median "${results[@]}"
}

benchmark_ollama() {
  local model_name="$1"
  local results=()
  for i in $(seq 1 "$RUNS"); do
    local tps
    tps=$(run_ollama "$model_name")
    if [ "$tps" = "FAIL" ] || [ -z "$tps" ]; then
      err "Ollama run $i failed for $model_name"
      return 1
    fi
    results+=("$tps")
    log "  Ollama run $i: ${tps} tok/s"
  done
  median "${results[@]}"
}

# List all configured models.
list_models() {
  echo "Configured models:"
  echo ""
  printf "%-20s %-12s %-8s %-20s\n" "Name" "Architecture" "Size" "Ollama"
  printf "%-20s %-12s %-8s %-20s\n" "----" "------------" "----" "------"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name arch gguf ollama size <<< "$entry"
    printf "%-20s %-12s %-8s %-20s\n" "$name" "$arch" "$size" "$ollama"
  done
}

# --- Main ---

if [ "${1:-}" = "--list" ]; then
  list_models
  exit 0
fi

# Filter to a single model if specified.
FILTER="${1:-}"

mkdir -p "$RESULTS_DIR"
build_bench_tps

log "Benchmark: Zerfoo vs Ollama"
log "Date: $DATE"
log "Commit: $COMMIT"
log "Hardware: DGX Spark GB10"
log "Prompt: \"$PROMPT\""
log "Tokens: $TOKENS, Runs: $RUNS, Sampling: greedy (temp=0)"
log ""

# JSON array accumulator.
JSON_ENTRIES=()

# Markdown table header.
MD_HEADER="| Model | Architecture | Size | Zerfoo (tok/s) | Ollama (tok/s) | Ratio | Winner |"
MD_SEP="|-------|-------------|------|----------------|----------------|-------|--------|"
MD_ROWS=()

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name arch gguf_rel ollama_name size <<< "$entry"

  # Apply filter if specified.
  if [ -n "$FILTER" ] && [ "$FILTER" != "$name" ]; then
    continue
  fi

  gguf_path="${MODEL_BASE}/${gguf_rel}"
  log "=== $name ($arch, $size) ==="

  # --- Zerfoo ---
  zerfoo_tps="N/A"
  if [ -f "$gguf_path" ]; then
    log "Running Zerfoo..."
    zerfoo_tps=$(benchmark_zerfoo "$gguf_path") || zerfoo_tps="FAIL"
    log "Zerfoo median: ${zerfoo_tps} tok/s"
  else
    log "SKIP Zerfoo: GGUF not found at $gguf_path"
    zerfoo_tps="SKIP"
  fi

  # --- Ollama ---
  ollama_tps="N/A"
  if [ "$ollama_name" != "NONE" ]; then
    # Check if the model is pulled.
    if ollama list 2>/dev/null | grep -q "^${ollama_name}"; then
      log "Running Ollama ($ollama_name)..."
      ollama_tps=$(benchmark_ollama "$ollama_name") || ollama_tps="FAIL"
      log "Ollama median: ${ollama_tps} tok/s"
    else
      log "SKIP Ollama: model $ollama_name not pulled"
      ollama_tps="SKIP"
    fi
  else
    log "Ollama: not supported for $name"
  fi

  # --- Ratio ---
  ratio="N/A"
  winner="N/A"
  if [[ "$zerfoo_tps" =~ ^[0-9] ]] && [[ "$ollama_tps" =~ ^[0-9] ]]; then
    ratio=$(awk "BEGIN { printf \"%.2f\", $zerfoo_tps / $ollama_tps }")
    if awk "BEGIN { exit ($zerfoo_tps >= $ollama_tps) ? 0 : 1 }"; then
      winner="Zerfoo"
    else
      winner="Ollama"
    fi
  elif [[ "$zerfoo_tps" =~ ^[0-9] ]] && [ "$ollama_name" = "NONE" ]; then
    winner="Zerfoo (only)"
  fi

  # --- Record ---
  JSON_ENTRIES+=("{
    \"model\": \"$name\",
    \"architecture\": \"$arch\",
    \"size\": \"$size\",
    \"zerfoo_tps\": $(if [[ "$zerfoo_tps" =~ ^[0-9] ]]; then echo "$zerfoo_tps"; else echo "null"; fi),
    \"ollama_tps\": $(if [[ "$ollama_tps" =~ ^[0-9] ]]; then echo "$ollama_tps"; else echo "null"; fi),
    \"ratio\": $(if [[ "$ratio" =~ ^[0-9] ]]; then echo "$ratio"; else echo "null"; fi),
    \"winner\": \"$winner\",
    \"zerfoo_status\": \"$zerfoo_tps\",
    \"ollama_status\": \"$ollama_tps\",
    \"ollama_model\": \"$ollama_name\"
  }")

  MD_ROWS+=("| $name | $arch | $size | $zerfoo_tps | $ollama_tps | ${ratio}x | $winner |")

  log ""
done

# --- Write JSON output ---
{
  echo "{"
  echo "  \"date\": \"$DATE\","
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"hardware\": \"DGX Spark GB10 (sm_121, 128GB LPDDR5x)\","
  echo "  \"prompt\": \"$PROMPT\","
  echo "  \"tokens\": $TOKENS,"
  echo "  \"runs\": $RUNS,"
  echo "  \"sampling\": \"greedy (temp=0)\","
  echo "  \"ollama_version\": \"$(ollama --version 2>/dev/null || echo 'unknown')\","
  echo "  \"results\": ["
  for i in "${!JSON_ENTRIES[@]}"; do
    if [ "$i" -lt "$(( ${#JSON_ENTRIES[@]} - 1 ))" ]; then
      echo "    ${JSON_ENTRIES[$i]},"
    else
      echo "    ${JSON_ENTRIES[$i]}"
    fi
  done
  echo "  ]"
  echo "}"
} > "$JSON_OUT"

log "JSON results written to $JSON_OUT"

# --- Write Markdown output ---
{
  echo "# Zerfoo vs Ollama Benchmark Comparison"
  echo ""
  echo "**Date:** $DATE"
  echo "**Commit:** $COMMIT"
  echo "**Hardware:** DGX Spark GB10 (sm_121, 128GB LPDDR5x)"
  echo "**Methodology:** $RUNS runs per model, median reported, 128 decode tokens, greedy sampling"
  echo ""
  echo "## Results"
  echo ""
  echo "$MD_HEADER"
  echo "$MD_SEP"
  for row in "${MD_ROWS[@]}"; do
    echo "$row"
  done
  echo ""
  echo "## Notes"
  echo ""
  echo "- Ratio > 1.0 means Zerfoo is faster."
  echo "- N/A in Ollama column means the model is not supported by Ollama."
  echo "- SKIP means the model file was not available or not pulled."
  echo "- FAIL means the benchmark run encountered an error."
} > "$MD_OUT"

log "Markdown results written to $MD_OUT"

# --- Summary ---
log ""
log "=== SUMMARY ==="
log ""
for row in "${MD_ROWS[@]}"; do
  log "$row"
done
