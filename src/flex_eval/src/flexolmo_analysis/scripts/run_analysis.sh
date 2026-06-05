#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="1000 65534 65534 65534 1000cd "$BUNDLE_ROOT/.." && pwd)"
export PYTHONPATH="$SRC_ROOT:${PYTHONPATH:-}"

CAPTURE_STAGES=(
  routing_light
  router_direction
  weight_analysis
  latent_space
  token_features
  expert_contribution
)

REPORT_STAGES=(
  coactivation
  top1_top2_confusion
  routing_confidence
  correctness_conditioned
  router_direction
  latent_space
  weight_analysis
  routing_weight_bridge
  token_features
  expert_contribution
  summary_tables
)

clear_runtime_caches() {
  python3 - <<'PY'
import gc

gc.collect()

try:
    import torch
except Exception:
    torch = None

if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
PY
}

clean_outputs() {
  rm -rf "$BUNDLE_ROOT/outputs/analysis/captures"
  rm -rf "$BUNDLE_ROOT/outputs/analysis/figures"
  rm -rf "$BUNDLE_ROOT/outputs/analysis/tables"
}

clean_pycaches() {
  find "$BUNDLE_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
}

if [[ "${CLEAN_OUTPUTS:-0}" == "1" ]]; then
  clean_outputs
fi

if [[ "${CLEAN_PYCACHE:-0}" == "1" ]]; then
  clean_pycaches
fi

for stage in "${CAPTURE_STAGES[@]}"; do
  echo "[capture] $stage"
  python3 -m flexolmo_analysis.cli.run_capture --stage "$stage"
  clear_runtime_caches
done

for analysis in "${REPORT_STAGES[@]}"; do
  echo "[report] $analysis"
  python3 -m flexolmo_analysis.cli.run_reports --analysis "$analysis"
  clear_runtime_caches
done
