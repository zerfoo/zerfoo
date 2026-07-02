#!/usr/bin/env bash
# dgx-spark-parity.sh -- Run model parity tests on DGX Spark.
#
# Usage:
#   ssh ndungu@192.168.86.250
#   cd /home/ndungu/Code/zerfoo
#   ./scripts/dgx-spark-parity.sh
#
# Prerequisites:
#   - Go installed at $HOME/.local/go/bin
#   - CUDA toolkit at /usr/local/cuda
#   - CUTLASS headers at $HOME/cutlass/include
#   - Model ZMF files in $HOME/models/<family>/model.zmf
#   - Model configs in $HOME/models/<family>/config.json
set -euo pipefail

# CUDA / CGo environment
export PATH="/usr/local/cuda/bin:$HOME/.local/go/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CGO_LDFLAGS="-L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu"
export CGO_CFLAGS="-I/usr/local/cuda/include -I/usr/include/aarch64-linux-gnu -I$HOME/cutlass/include"

# Model paths -- set only if the ZMF file exists.
MODEL_BASE="$HOME/models"

set_model_env() {
    local family="$1" zmf_var="$2" dir_var="$3"
    local zmf="$MODEL_BASE/$family/model.zmf"
    local dir="$MODEL_BASE/$family/"
    if [ -f "$zmf" ]; then
        export "$zmf_var=$zmf"
        export "$dir_var=$dir"
        echo "  $zmf_var=$zmf"
    else
        echo "  $zmf_var (skipped -- $zmf not found)"
    fi
}

echo "Setting model environment variables:"
set_model_env "qwen25"   QWEN25_ZMF_PATH   QWEN25_MODEL_DIR
set_model_env "llama3"   LLAMA3_ZMF_PATH   LLAMA3_MODEL_DIR
set_model_env "mistral"  MISTRAL_ZMF_PATH  MISTRAL_MODEL_DIR
set_model_env "phi4"     PHI4_ZMF_PATH     PHI4_MODEL_DIR
set_model_env "gemma3"   GEMMA3_ZMF_PATH   GEMMA3_MODEL_DIR
set_model_env "deepseek" DEEPSEEK_ZMF_PATH DEEPSEEK_MODEL_DIR
set_model_env "siglip"   SIGLIP_ZMF_PATH   SIGLIP_MODEL_DIR
echo ""

echo "Running parity tests..."
go test -tags cuda,cutlass ./tests/parity/ -v -count=1 -timeout 600s
