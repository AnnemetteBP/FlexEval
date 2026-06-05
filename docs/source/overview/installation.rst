Installation
============

Requirements
------------

Supported Python
----------------

``Python 3.11``

Shared runtime baseline
-----------------------

+----------------------+------------------------------------------------------+
| Package              | Version                                              |
+======================+======================================================+
| ``torch``            | ``2.8.0``                                            |
+----------------------+------------------------------------------------------+
| ``transformers``     | ``peter-sk/transformers@9da5df2d2d2fe155f861d6248ba5bb0b1c769513`` |
+----------------------+------------------------------------------------------+
| ``datasets``         | ``3.6.0``                                            |
+----------------------+------------------------------------------------------+
| ``accelerate``       | ``1.13.0``                                           |
+----------------------+------------------------------------------------------+
| ``numpy``            | ``1.26.4``                                           |
+----------------------+------------------------------------------------------+
| ``pandas``           | ``3.0.3``                                            |
+----------------------+------------------------------------------------------+
| ``huggingface-hub``  | ``1.17.0``                                           |
+----------------------+------------------------------------------------------+
| ``scikit-learn``     | ``1.6.1``                                            |
+----------------------+------------------------------------------------------+
| ``matplotlib``       | ``3.10.9``                                           |
+----------------------+------------------------------------------------------+
| ``seaborn``          | ``0.13.2``                                           |
+----------------------+------------------------------------------------------+
| ``umap-learn``       | ``0.5.12``                                           |
+----------------------+------------------------------------------------------+

Optional Linux runtime engine
-----------------------------

+------------------+-------------+
| Package          | Version     |
+==================+=============+
| ``vllm``         | ``0.11.0``  |
+------------------+-------------+
| ``ray``          | ``2.55.1``  |
+------------------+-------------+
| ``bitsandbytes`` | ``0.49.2``  |
+------------------+-------------+

Developer installation
----------------------

FlexEval can be installed in your preferred Python environment. The setup
scripts use Conda for environment creation.

Install with the EuroEval backend::

    BACKEND=euroeval bash env/setup_dev_env.sh

Install with the olmes backend::

    BACKEND=olmes bash env/setup_dev_env.sh

Install with both backends::

    BACKEND=all bash env/setup_dev_env.sh

Runtime installation
--------------------

Install with the transformers engine::

    BACKEND=euroeval ENGINE=transformers bash env/setup_runtime_env.sh
    BACKEND=olmes ENGINE=transformers bash env/setup_runtime_env.sh
    BACKEND=all ENGINE=transformers bash env/setup_runtime_env.sh

Install with the vLLM engine::

    BACKEND=euroeval ENGINE=vllm bash env/setup_runtime_env.sh
    BACKEND=olmes ENGINE=vllm bash env/setup_runtime_env.sh
    BACKEND=all ENGINE=vllm bash env/setup_runtime_env.sh

Configuration
-------------

- ``BACKEND=euroeval|olmes|all``
- ``ENGINE=transformers|vllm``
- ``ENV_NAME=<name>``
- ``PYTHON_VERSION=3.11``
- ``USE_CONDA=auto|yes|no``

Verification
------------

Verify the package import::

    python -c "import flexeval; print('ok')"

Verify the CLI::

    python -m flexeval.cli.run --help

Verify the transformer fork::

    python -c "from transformers import FlexOlmoForCausalLM; print(FlexOlmoForCausalLM)"
