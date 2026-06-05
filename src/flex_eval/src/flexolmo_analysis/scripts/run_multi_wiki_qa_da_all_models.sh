#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_ROOT="$BUNDLE_ROOT"
SRC_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
export PYTHONPATH="$SRC_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

find_project_root() {
  local current="$1"
  while [[ "$current" != "/" ]]; do
    if [[ -d "$current/models" ]]; then
      printf '%s\n' "$current"
      return 0
    fi
    current="$(dirname "$current")"
  done
  return 1
}

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PROJECT_ROOT="${PROJECT_ROOT:-$(find_project_root "$SRC_ROOT" || true)}"
if [[ -z "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-flex-olmo-analysis-am}"
DATE_STAMP="${DATE_STAMP:-$(date +%F)}"
DEFAULT_DATASET_PATH="$(first_existing_file \
  "$PROJECT_ROOT/eval/benchmarks/mix/data/multi-wiki-qa_da_jsonl/multi_wiki_qa_da.jsonl" \
  "$PROJECT_ROOT/eval_analysis/data/multi-wiki-qa_da_jsonl/multi_wiki_qa_da.jsonl" \
  "$PACKAGE_ROOT/eval/benchmarks/mix/data/multi-wiki-qa_da_jsonl/multi_wiki_qa_da.jsonl" \
  || true)"
DATASET_PATH="${DATASET_PATH:-$DEFAULT_DATASET_PATH}"
RESULTS_ROOT="${RESULTS_ROOT:-$PACKAGE_ROOT}"
DEFAULT_BENCHMARK_NAME="$(basename "${DATASET_PATH%.*}")"
BENCHMARK_NAME="${BENCHMARK_NAME:-${DEFAULT_BENCHMARK_NAME:-multi_wiki_qa_da}}"
RAW_BENCHMARK_ROOT="${BENCHMARK_ROOT:-$RESULTS_ROOT}"
if [[ "$(basename "$RAW_BENCHMARK_ROOT")" == "$BENCHMARK_NAME" ]]; then
  BENCHMARK_ROOT="$RAW_BENCHMARK_ROOT"
else
  BENCHMARK_ROOT="$RAW_BENCHMARK_ROOT/$BENCHMARK_NAME"
fi
DEFAULT_MODEL_ROOT="$(first_existing_dir \
  "$PROJECT_ROOT/models" \
  "/work/training/FlexMoRE/models" \
  || true)"
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
DEFAULT_TOKENIZER_PATH="$(first_existing_dir \
  "$MODEL_ROOT/tmp" \
  "$PROJECT_ROOT/models/tmp" \
  "/work/training/FlexMoRE/models/tmp" \
  || true)"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
MODEL_REGISTRY="${MODEL_REGISTRY:-$PACKAGE_ROOT/model_paths/all_models.txt}"
MODEL_CATALOG="${MODEL_CATALOG:-$PACKAGE_ROOT/model_paths/models.json}"
BASELINE_MODEL="${BASELINE_MODEL:-FlexOlmo-7x7B-1T-a4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-512}"
TASK_DOMAIN="${TASK_DOMAIN:-wiki}"
TASK_SCORING_MODE="${TASK_SCORING_MODE:-qa}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw_prompt}"
PROMPT_BACKEND="${PROMPT_BACKEND:-euroeval_compat}"
EUROEVAL_FEW_SHOT="${EUROEVAL_FEW_SHOT:-true}"
EUROEVAL_NUM_FEW_SHOT_EXAMPLES="${EUROEVAL_NUM_FEW_SHOT_EXAMPLES:-4}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-auto}"
CHAT_TEMPLATE_ENABLED="${CHAT_TEMPLATE_ENABLED:-$CHAT_TEMPLATE_MODE}"
PROMPT_ADD_SPECIAL_TOKENS="${PROMPT_ADD_SPECIAL_TOKENS:-false}"
REFERENCE_ADD_SPECIAL_TOKENS="${REFERENCE_ADD_SPECIAL_TOKENS:-false}"
DECODE_SKIP_SPECIAL_TOKENS="${DECODE_SKIP_SPECIAL_TOKENS:-true}"
MAX_EXAMPLES_PER_DATASET="${MAX_EXAMPLES_PER_DATASET:-500}"
SELECTED_LAYERS="${SELECTED_LAYERS:-all}"
PUBLIC_EXPERT_IDX="${PUBLIC_EXPERT_IDX:-0}"
COMBINED_ACTIVE_EXPERTS="${COMBINED_ACTIVE_EXPERTS:-2,4,7}"
ROUTING_RUN_MODE="${ROUTING_RUN_MODE:-native_only}"
INCLUDE_INDIVIDUAL_EXPERTS="${INCLUDE_INDIVIDUAL_EXPERTS:-false}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SAVE_DEEP_RAW="${SAVE_DEEP_RAW:-0}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif [[ -d "/home/ucloud/miniconda3" ]]; then
  CONDA_BASE="/home/ucloud/miniconda3"
else
  CONDA_BASE=""
fi

if [[ -n "${CONDA_ENV_NAME:-}" && -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    conda activate "$CONDA_ENV_NAME"
  fi
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Missing dataset: $DATASET_PATH" >&2
  exit 1
fi

if [[ -z "$TOKENIZER_PATH" || ! -d "$TOKENIZER_PATH" ]]; then
  echo "Missing tokenizer dir: $TOKENIZER_PATH" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_PATH/tokenizer_config.json" ]]; then
  echo "Tokenizer dir missing tokenizer_config.json: $TOKENIZER_PATH" >&2
  exit 1
fi

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

patch_model_tokenizer_dir() {
  local model_dir="$1"
  python3 - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
changed_any = False

def disable_bytelevel_regex(node):
    changed = False
    if isinstance(node, dict):
        if node.get("type") == "ByteLevel" and node.get("use_regex") is True:
            node["use_regex"] = False
            changed = True
        for value in node.values():
            if disable_bytelevel_regex(value):
                changed = True
    elif isinstance(node, list):
        for value in node:
            if disable_bytelevel_regex(value):
                changed = True
    return changed

def remove_fix_mistral_regex(node):
    changed = False
    if isinstance(node, dict):
        if "fix_mistral_regex" in node:
            node.pop("fix_mistral_regex", None)
            changed = True
        for value in node.values():
            if remove_fix_mistral_regex(value):
                changed = True
    elif isinstance(node, list):
        for value in node:
            if remove_fix_mistral_regex(value):
                changed = True
    return changed

for json_path in sorted(model_dir.rglob("*.json")):
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(obj, (dict, list)):
        continue
    changed = remove_fix_mistral_regex(obj)
    if json_path.name == "tokenizer.json" and disable_bytelevel_regex(obj):
        changed = True
    if changed:
        json_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        changed_any = True

print("patched" if changed_any else "ok")
PY
}

link_model_dir() {
  local source_dir="$1"
  local target_root="$2"
  local model_name="$3"
  mkdir -p "$target_root"
  rm -rf "$target_root/$model_name"
  ln -s "$source_dir" "$target_root/$model_name"
}

if [[ "${CLEAN_PYCACHE:-0}" == "1" ]]; then
  find "$PACKAGE_ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
fi

MANIFEST_PATH="$BENCHMARK_ROOT/${BENCHMARK_NAME}_manifest.json"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "$BENCHMARK_ROOT"
  python3 - <<PY
import json
from pathlib import Path
manifest = {
    "schema_version": 1,
    "datasets": [
        {
            "name": ${BENCHMARK_NAME@Q},
            "path": ${DATASET_PATH@Q},
            "num_examples": int(${MAX_EXAMPLES_PER_DATASET@Q}),
            "domain": ${TASK_DOMAIN@Q},
            "scoring_mode": ${TASK_SCORING_MODE@Q},
            "prompting": {
                "format": ${PROMPT_FORMAT@Q},
                "chat_template": {
                    "enabled": ${CHAT_TEMPLATE_ENABLED@Q},
                    "mode": ${CHAT_TEMPLATE_MODE@Q}
                },
                "euroeval_compat": {
                    "enabled": ${PROMPT_BACKEND@Q} == "euroeval_compat",
                    "few_shot": ${EUROEVAL_FEW_SHOT@Q} == "true",
                    "num_few_shot_examples": int(${EUROEVAL_NUM_FEW_SHOT_EXAMPLES@Q})
                }
            },
            "generation": {"max_new_tokens": int(${DEFAULT_MAX_NEW_TOKENS@Q})},
            "tokenization": {
                "prompt_add_special_tokens": ${PROMPT_ADD_SPECIAL_TOKENS@Q} == "true",
                "reference_add_special_tokens": ${REFERENCE_ADD_SPECIAL_TOKENS@Q} == "true",
                "decode_skip_special_tokens": ${DECODE_SKIP_SPECIAL_TOKENS@Q} == "true",
            },
        }
    ],
}
Path(${MANIFEST_PATH@Q}).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
fi

mapfile -t MODELS < <(
  python3 - <<PY
import json
from pathlib import Path
catalog = json.loads(Path(${MODEL_CATALOG@Q}).read_text())
models = [${BASELINE_MODEL@Q}, *catalog["flexolmo"]["8x7B"]["a4"]]
for name in models:
    print(name)
PY
)

PATCH_STATUS="$(patch_model_tokenizer_dir "$TOKENIZER_PATH")"
if [[ "$PATCH_STATUS" != "ok" ]]; then
  echo "[tokenizer] ${PATCH_STATUS} $(basename "$TOKENIZER_PATH")"
fi

run_step() {
  local label="$1"
  shift
  echo "[run] $label"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q' python3
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
    return 0
  fi
  local prev=""
  local arg=""
  for arg in "$@"; do
    if [[ "$prev" == "--output-root" ]]; then
      mkdir -p "$arg"
    fi
    prev="$arg"
  done
  local start_ts
  start_ts="$(date '+%F %T')"
  echo "[start] $label at $start_ts"
  python3 "$@"
  local end_ts
  end_ts="$(date '+%F %T')"
  echo "[done] $label at $end_ts"
  clear_runtime_caches
}

if [[ "$SELECTED_LAYERS" == "first_early_mid_late_last" ]]; then
  NON_BASE_SELECTED_LAYERS="first_early_mid_late_last"
else
  NON_BASE_SELECTED_LAYERS="$SELECTED_LAYERS"
fi

TOTAL_MODELS="${#MODELS[@]}"
MODEL_INDEX=0
declare -A MODEL_RUN_ROOTS

for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_INDEX=$((MODEL_INDEX + 1))
  RUN_ROOT="$BENCHMARK_ROOT/${MODEL_NAME}_${DATE_STAMP}"
  CAPTURE_ROOT="$RUN_ROOT/captures"
  FIGURE_ROOT="$RUN_ROOT/figures"
  TABLE_ROOT="$RUN_ROOT/tables"

  echo "[model] ${MODEL_INDEX}/${TOTAL_MODELS} ${MODEL_NAME}"
  echo "[root] $RUN_ROOT"
  echo "[config] device=${DEVICE} dtype=${DTYPE} selected_layers=${SELECTED_LAYERS} max_examples=${MAX_EXAMPLES_PER_DATASET} save_deep_raw=${SAVE_DEEP_RAW} scoring_mode=${TASK_SCORING_MODE} domain=${TASK_DOMAIN} prompt_format=${PROMPT_FORMAT} prompt_backend=${PROMPT_BACKEND} few_shot=${EUROEVAL_FEW_SHOT}/${EUROEVAL_NUM_FEW_SHOT_EXAMPLES} chat_template=${CHAT_TEMPLATE_ENABLED} tokenizer_path=${TOKENIZER_PATH}"
  MODEL_RUN_ROOTS["$MODEL_NAME"]="$RUN_ROOT"

  if [[ "${CLEAN_OUTPUTS:-0}" == "1" ]]; then
    rm -rf "$RUN_ROOT"
  fi

  DEEP_RAW_ARGS=()
  if [[ "$SAVE_DEEP_RAW" == "1" ]]; then
    DEEP_RAW_ARGS+=(--save-raw-artifacts)
  fi

  run_step "deep_base_capture:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/runners/run_mix_deep_base_capture.py" \
    --manifest-path "$MANIFEST_PATH" \
    --model-name "$MODEL_NAME" \
    --model-root "$MODEL_ROOT" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --model-registry "$MODEL_REGISTRY" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-length "$MAX_LENGTH" \
    --default-max-new-tokens "$DEFAULT_MAX_NEW_TOKENS" \
    --max-examples-per-dataset "$MAX_EXAMPLES_PER_DATASET" \
    --datasets "$BENCHMARK_NAME" \
    --selected-layers "$SELECTED_LAYERS" \
    --router-direction-output-root "$CAPTURE_ROOT/router_direction" \
    --router-direction-selected-layers "$NON_BASE_SELECTED_LAYERS" \
    --latent-space-output-root "$CAPTURE_ROOT/latent_space" \
    --latent-space-selected-layers "$NON_BASE_SELECTED_LAYERS" \
    --routing-run-mode "$ROUTING_RUN_MODE" \
    --public-expert-idx "$PUBLIC_EXPERT_IDX" \
    --combined-active-experts "$COMBINED_ACTIVE_EXPERTS" \
    "${DEEP_RAW_ARGS[@]}" \
    --output-root "$CAPTURE_ROOT/deep_base"

  run_step "weight_analysis:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/runners/run_mix_weight_analysis.py" \
    --model-name "$MODEL_NAME" \
    --model-root "$MODEL_ROOT" \
    --model-registry "$MODEL_REGISTRY" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --selected-layers "$NON_BASE_SELECTED_LAYERS" \
    --public-expert-idx "$PUBLIC_EXPERT_IDX" \
    --output-root "$CAPTURE_ROOT/weight_analysis"

  run_step "coactivation:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_coactivation.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/coactivation" \
    --model-name "$MODEL_NAME" \
    --dataset "$BENCHMARK_NAME" \
    --run-label native_full

  run_step "top1_top2_confusion:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_top1_top2_confusion.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/top1_top2_confusion" \
    --model-name "$MODEL_NAME" \
    --dataset "$BENCHMARK_NAME" \
    --run-label native_full

  run_step "routing_confidence:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_routing_confidence_outcomes.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/routing_confidence" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME"

  run_step "correctness_conditioned:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_correctness_conditioned.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/correctness_conditioned" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME" \
    --run-labels native_full

  run_step "routing_weight_bridge:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_routing_weight_bridge.py" \
    --routing-root "$CAPTURE_ROOT/deep_base" \
    --weight-root "$CAPTURE_ROOT/weight_analysis" \
    --output-root "$FIGURE_ROOT/routing_weight_bridge" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME" \
    --run-label native_full

  run_step "token_features_report:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_token_features.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/token_features" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME" \
    --run-labels native_full

  run_step "expert_contribution_report:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_expert_contribution.py" \
    --results-root "$CAPTURE_ROOT/deep_base" \
    --output-root "$FIGURE_ROOT/expert_contribution" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME" \
    --run-labels native_full

  run_step "router_geometry_report:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_router_geometry.py" \
    --results-root "$CAPTURE_ROOT/router_direction" \
    --output-root "$FIGURE_ROOT/router_geometry" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME"

  run_step "representation_geometry_report:$MODEL_NAME" \
    "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_representation_geometry.py" \
    --results-root "$CAPTURE_ROOT/latent_space" \
    --output-root "$FIGURE_ROOT/representation_geometry" \
    --model-names "$MODEL_NAME" \
    --datasets "$BENCHMARK_NAME"

  if [[ "$MODEL_NAME" != "$BASELINE_MODEL" ]]; then
    BASELINE_RUN_ROOT="${MODEL_RUN_ROOTS[$BASELINE_MODEL]}"
    PAIR_SLUG="${BASELINE_MODEL}__vs__${MODEL_NAME}"
    PAIR_ROOT="$BENCHMARK_ROOT/comparisons/$PAIR_SLUG"
    PAIR_INPUT_ROOT="$PAIR_ROOT/inputs"
    PAIR_FIGURE_ROOT="$PAIR_ROOT/figures"
    PAIR_TABLE_ROOT="$PAIR_ROOT/tables"

    link_model_dir "$BASELINE_RUN_ROOT/captures/deep_base/$BASELINE_MODEL" "$PAIR_INPUT_ROOT/deep_base" "$BASELINE_MODEL"
    link_model_dir "$RUN_ROOT/captures/deep_base/$MODEL_NAME" "$PAIR_INPUT_ROOT/deep_base" "$MODEL_NAME"
    link_model_dir "$BASELINE_RUN_ROOT/captures/router_direction/$BASELINE_MODEL" "$PAIR_INPUT_ROOT/router_direction" "$BASELINE_MODEL"
    link_model_dir "$RUN_ROOT/captures/router_direction/$MODEL_NAME" "$PAIR_INPUT_ROOT/router_direction" "$MODEL_NAME"
    link_model_dir "$BASELINE_RUN_ROOT/captures/latent_space/$BASELINE_MODEL" "$PAIR_INPUT_ROOT/latent_space" "$BASELINE_MODEL"
    link_model_dir "$RUN_ROOT/captures/latent_space/$MODEL_NAME" "$PAIR_INPUT_ROOT/latent_space" "$MODEL_NAME"
    link_model_dir "$BASELINE_RUN_ROOT/captures/weight_analysis/$BASELINE_MODEL" "$PAIR_INPUT_ROOT/weight_analysis" "$BASELINE_MODEL"
    link_model_dir "$RUN_ROOT/captures/weight_analysis/$MODEL_NAME" "$PAIR_INPUT_ROOT/weight_analysis" "$MODEL_NAME"

    run_step "router_direction_report:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_router_direction.py" \
      --results-root "$PAIR_INPUT_ROOT/router_direction" \
      --output-root "$PAIR_FIGURE_ROOT/router_direction" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME"

    run_step "router_geometry_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_router_geometry.py" \
      --results-root "$PAIR_INPUT_ROOT/router_direction" \
      --output-root "$PAIR_FIGURE_ROOT/router_geometry" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME"

    run_step "representation_geometry_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_representation_geometry.py" \
      --results-root "$PAIR_INPUT_ROOT/latent_space" \
      --output-root "$PAIR_FIGURE_ROOT/representation_geometry" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME"

    run_step "coactivation_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_coactivation.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/coactivation" \
      --model-name "$BASELINE_MODEL" \
      --model-name "$MODEL_NAME" \
      --dataset "$BENCHMARK_NAME" \
      --run-label native_full \
      --summary-only

    run_step "top1_top2_confusion_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_top1_top2_confusion.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/top1_top2_confusion" \
      --model-name "$BASELINE_MODEL" \
      --model-name "$MODEL_NAME" \
      --dataset "$BENCHMARK_NAME" \
      --run-label native_full

    run_step "routing_confidence_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_routing_confidence_outcomes.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/routing_confidence" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME"

    run_step "correctness_conditioned_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_correctness_conditioned.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/correctness_conditioned" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME" \
      --run-labels native_full

    run_step "latent_space_report:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_latent_space.py" \
      --results-root "$PAIR_INPUT_ROOT/latent_space" \
      --output-root "$PAIR_FIGURE_ROOT/latent_space" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME" \
      --pca-dataset "$BENCHMARK_NAME"

    run_step "weight_analysis_report:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_weight_analysis.py" \
      --results-root "$PAIR_INPUT_ROOT/weight_analysis" \
      --output-root "$PAIR_FIGURE_ROOT/weight_analysis" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME"

    run_step "token_features_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_token_features.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/token_features" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME" \
      --run-labels native_full

    run_step "expert_contribution_pair:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/analyze_mix_expert_contribution.py" \
      --results-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_FIGURE_ROOT/expert_contribution" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME" \
      --datasets "$BENCHMARK_NAME" \
      --run-labels native_full

    run_step "summary_tables:${PAIR_SLUG}" \
      "$PACKAGE_ROOT/eval/benchmarks/mix/plotting/generate_mix_summary_tables.py" \
      --coactivation-root "$PAIR_FIGURE_ROOT/coactivation" \
      --latent-root "$PAIR_FIGURE_ROOT/latent_space" \
      --top1-top2-root "$PAIR_FIGURE_ROOT/top1_top2_confusion" \
      --routing-confidence-root "$PAIR_FIGURE_ROOT/routing_confidence" \
      --correctness-conditioned-root "$PAIR_FIGURE_ROOT/correctness_conditioned" \
      --expert-contribution-root "$PAIR_FIGURE_ROOT/expert_contribution" \
      --router-geometry-root "$PAIR_FIGURE_ROOT/router_geometry" \
      --representation-geometry-root "$PAIR_FIGURE_ROOT/representation_geometry" \
      --routing-light-root "$PAIR_INPUT_ROOT/deep_base" \
      --output-root "$PAIR_TABLE_ROOT" \
      --model-names "$BASELINE_MODEL,$MODEL_NAME"
  fi
done

echo "Benchmark root: $BENCHMARK_ROOT"
