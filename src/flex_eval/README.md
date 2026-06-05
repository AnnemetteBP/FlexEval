# FlexOlmo Analysis Bundle

This is a standalone Python project you can move as a single folder.

Top-level structure:

- `pyproject.toml`
- `requirements.txt`
- `scripts/`
- `src/flexolmo_analysis/`
- `vendor/`
- `outputs/`

Scope currently included:

- routing-light capture
- router-direction capture and plotting
- weight-analysis capture and plotting
- latent-space capture and PCA-style plotting
- token-feature capture and probe analysis
- coactivation heatmaps
- top1/top2 confusion
- routing-confidence analysis
- routing/weight bridge
- summary-table generation
- router-geometry analysis
- representation-geometry analysis

Inside `src/flexolmo_analysis/`:

- `cli/`
  - Python entrypoints
- `configs/`
  - generic analysis configs
- `metadata/`
  - expert-label mappings

Inside `vendor/`:

- `eval/benchmarks/mix/`
  - copied runners, plotting code, and data
- `src/flexolmo_analysis.toolkit/`
  - copied support library used by the vendor runners
- `model_paths/`
  - copied model registry files

Important:

- dependency setup is at the bundle root via `requirements.txt`
- capture stages require a `transformers` installation that provides `FlexOlmoForCausalLM`

Quick setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "from transformers import FlexOlmoForCausalLM; print(FlexOlmoForCausalLM)"
```

Main commands:

```bash
bash scripts/run_analysis.sh <MODEL_A> <EVAL_NAME> outputs/evals outputs/analysis 1
bash scripts/run_analysis_vllm.sh <MODEL_A> <EVAL_NAME> outputs/evals outputs/analysis 1
```

Mix reporting also includes the new geometry reports:

- `router_geometry`
- `representation_geometry`
