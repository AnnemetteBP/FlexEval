#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_common.sh"

ENV_NAME="${ENV_NAME:-flexeval-dev}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_ROOT="${PROJECT_ROOT:-$(resolve_project_root)}"
BACKEND="${BACKEND:-euroeval}"
ENGINE="${ENGINE:-transformers}"
ARCHITECTURE="${ARCHITECTURE:-flex-family}"
USE_CONDA="${USE_CONDA:-auto}"

activate_conda_env_if_requested "${ENV_NAME}" "${PYTHON_VERSION}" "${USE_CONDA}"
install_selected_profile "${PROJECT_ROOT}" "dev" "${BACKEND}" "${ENGINE}" "${ARCHITECTURE}"
export_hf_credentials
