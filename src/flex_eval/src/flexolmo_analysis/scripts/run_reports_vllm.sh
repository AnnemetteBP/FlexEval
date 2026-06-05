#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="1000 65534 65534 65534 1000cd "$BUNDLE_ROOT/.." && pwd)"
export PYTHONPATH="$SRC_ROOT:${PYTHONPATH:-}"

ANALYSIS=${1:-all}

python3 -m flexolmo_analysis.cli.run_reports --analysis "$ANALYSIS"
