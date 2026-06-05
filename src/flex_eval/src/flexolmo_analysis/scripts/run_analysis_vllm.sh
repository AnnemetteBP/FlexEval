#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="1000 65534 65534 65534 1000cd "$BUNDLE_ROOT/.." && pwd)"
export PYTHONPATH="$SRC_ROOT:${PYTHONPATH:-}"

MODEL=$1
EVAL_NAME=$2
EVAL_OUTPUT_DIR=$3
ANALYSIS_OUTPUT_DIR=$4
GPUS=$5

echo "Model: $MODEL"
echo "Eval name: $EVAL_NAME"
echo "Eval outputs: $EVAL_OUTPUT_DIR"
echo "Analysis outputs: $ANALYSIS_OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Backend: vllm"
echo
echo "Edit configs in src/flexolmo_analysis/configs before running if needed."
echo
python3 -m flexolmo_analysis.cli.run_capture
python3 -m flexolmo_analysis.cli.run_reports
