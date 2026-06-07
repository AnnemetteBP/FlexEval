<p align="center">
  <img src="assets/FlexEval.png" alt="FlexEval" width="720">
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-0f766e">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-1d4ed8">
  <img alt="PyTorch 2.8.0" src="https://img.shields.io/badge/PyTorch-2.8.0-ee4c2c">
  <img alt="Datasets 3.6.0" src="https://img.shields.io/badge/Datasets-3.6.0-f59e0b">
  <img alt="Accelerate 1.13.0" src="https://img.shields.io/badge/Accelerate-1.13.0-7c3aed">
</p>

<p align="center">
  <img alt="Flex-family transformers fork" src="https://img.shields.io/badge/Flex--family%20Transformers-peter--sk%2Ftransformers%409da5df2d2d2fe155f861d6248ba5bb0b1c769513-374151">
</p>

# FlexEval

FlexEval is a unified evaluation and analysis suite for flexible
architectures. It brings architecture-aware capture and analysis together with
integrated `EuroEval` and `olmes` support in one repository and one shared
installation flow.

## Requirements

- Python `3.11`
- PyTorch `2.8.0`
- Datasets `3.6.0`
- Accelerate `1.13.0`
- Flex-family architecture extra uses `peter-sk/transformers@9da5df2d2d2fe155f861d6248ba5bb0b1c769513`

## Installation

### From source

```bash
git clone https://github.com/AnnemetteBP/FlexEval.git
cd FlexEval
```

### Conda setup

Use one shared Conda environment and install the project with `pip`.

#### Development setup

```bash
BACKEND=euroeval ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_dev_env.sh
BACKEND=olmes ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_dev_env.sh
BACKEND=all ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_dev_env.sh
BACKEND=all ENGINE=vllm ARCHITECTURE=flex-family bash env/setup_dev_env.sh
```

#### Runtime setup

```bash
BACKEND=euroeval ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_runtime_env.sh
BACKEND=olmes ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_runtime_env.sh
BACKEND=all ENGINE=transformers ARCHITECTURE=flex-family bash env/setup_runtime_env.sh
BACKEND=all ENGINE=vllm ARCHITECTURE=flex-family bash env/setup_runtime_env.sh
```

Supported setup variables:

- `BACKEND=none|euroeval|olmes|all`
- `ENGINE=transformers|vllm`
- `ARCHITECTURE=generic|flex-family`
- `ENV_NAME=<name>`
- `PYTHON_VERSION=3.11`
- `USE_CONDA=auto|yes|no`

### Manual installation

If you already have a Python or Conda environment, install the root project
first and then add the backend you want.

#### Root package

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[engine-vllm]"
pip install -e ".[architecture-flex-family]"
pip install -e ".[dev,engine-vllm,architecture-flex-family]"
```

#### EuroEval support

```bash
pip install -r env/requirements-backend-euroeval.txt
pip install --no-deps -e ./EuroEval
```

#### olmes support

```bash
pip install -r env/requirements-backend-olmes.txt
pip install --no-deps -e ./olmes
```

## What FlexEval Provides

- One root project install in [`pyproject.toml`](./pyproject.toml)
- Backend adapters for `EuroEval` and `olmes`
- Engine adapters for `transformers` and `vllm`
- Architecture adapters for flexible model families such as `flex-family`
- Flex-family capture and analysis code under `src/flex_eval/src/flexolmo_analysis/`
- Unified CLI entry points under `src/flex_eval/src/flexeval/`
- Shared setup wrappers in `env/`

## Verification

```bash
python -c "import flexeval; print('ok')"
python -m flexeval.cli.run --help
python -c "from transformers import FlexOlmoForCausalLM; print(FlexOlmoForCausalLM)"  # with architecture-flex-family installed
```

## Usage

Run the unified CLI with the selected backend, architecture, dataset, model,
and engine.

```bash
python -m flexeval.cli.run \
  --backend euroeval \
  --architecture flex-family \
  --dataset <dataset_name> \
  --model <model_name_or_path> \
  --engine transformers
```

## Repository Layout

- `src/flex_eval/src/flexeval/` contains the unified root package
- `src/flex_eval/src/flexeval/architectures/` contains architecture-family adapters
- `src/flex_eval/src/flexeval/backends/` contains evaluation backend adapters
- `src/flex_eval/src/flexeval/engines/` contains inference engine adapters
- `src/flex_eval/src/flexolmo_analysis/` contains Flex capture and analysis code
- `env/` contains setup wrappers and optional backend or engine requirement files
- `EuroEval/` contains the integrated EuroEval source tree
- `olmes/` contains the integrated olmes source tree
- `OLMo-core/` and `Megatron-LM/` are vendored components used by the project
