#!/usr/bin/env bash
set -euo pipefail

resolve_project_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

activate_conda_env_if_requested() {
  local env_name="$1"
  local python_version="$2"
  local use_conda="$3"

  if [[ "${use_conda}" == "0" || "${use_conda}" == "false" || "${use_conda}" == "no" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    if [[ "${use_conda}" == "1" || "${use_conda}" == "true" || "${use_conda}" == "yes" ]]; then
      echo "USE_CONDA was requested, but conda is not available." >&2
      exit 1
    fi
    return 0
  fi

  if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    conda create --name "${env_name}" "python=${python_version}" -y
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${env_name}"
}

build_root_install_target() {
  local project_root="$1"
  local mode="$2"
  local engine="$3"
  local architecture="$4"
  local extras=()

  if [[ "${mode}" == "dev" ]]; then
    extras+=("dev")
  fi

  if [[ "${engine}" == "vllm" ]]; then
    extras+=("engine-vllm")
  elif [[ "${engine}" != "transformers" ]]; then
    echo "Unsupported ENGINE='${engine}'." >&2
    exit 1
  fi

  case "${architecture}" in
    none|generic)
      ;;
    flex-family)
      extras+=("architecture-flex-family")
      ;;
    *)
      echo "Unsupported ARCHITECTURE='${architecture}'." >&2
      exit 1
      ;;
  esac

  if [[ ${#extras[@]} -eq 0 ]]; then
    printf '%s' "${project_root}"
    return 0
  fi

  local joined
  joined="$(IFS=,; echo "${extras[*]}")"
  printf '%s' "${project_root}[${joined}]"
}

install_root_package() {
  local project_root="$1"
  local mode="$2"
  local engine="$3"
  local architecture="$4"
  local install_target

  install_target="$(build_root_install_target "${project_root}" "${mode}" "${engine}" "${architecture}")"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${install_target}"
}

install_backend_selection() {
  local project_root="$1"
  local backend="$2"

  case "${backend}" in
    none)
      ;;
    euroeval)
      python -m pip install -r "${project_root}/env/requirements-backend-euroeval.txt"
      python -m pip install --no-deps -e "${project_root}/EuroEval"
      ;;
    olmes)
      python -m pip install -r "${project_root}/env/requirements-backend-olmes.txt"
      python -m pip install --no-deps -e "${project_root}/olmes"
      ;;
    all)
      python -m pip install -r "${project_root}/env/requirements-backend-euroeval.txt"
      python -m pip install -r "${project_root}/env/requirements-backend-olmes.txt"
      python -m pip install --no-deps -e "${project_root}/EuroEval"
      python -m pip install --no-deps -e "${project_root}/olmes"
      ;;
    *)
      echo "Unsupported BACKEND='${backend}'." >&2
      exit 1
      ;;
  esac
}

install_selected_profile() {
  local project_root="$1"
  local mode="$2"
  local backend="$3"
  local engine="$4"
  local architecture="$5"

  install_root_package "${project_root}" "${mode}" "${engine}" "${architecture}"
  install_backend_selection "${project_root}" "${backend}"
}

export_hf_credentials() {
  local token
  token="$(python -c "from env.hf_token import get_hf_token; print(get_hf_token())")"
  export HF_TOKEN="${token}"
  export HUGGING_FACE_HUB_TOKEN="${token}"
  export HF_HUB_TOKEN="${token}"

  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "${HF_TOKEN}"
  fi
}
