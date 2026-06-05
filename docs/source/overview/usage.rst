Usage
=====

Run the project CLI::

    python -m flexeval.cli.run \
      --backend euroeval \
      --dataset <dataset_name> \
      --model <model_name_or_path> \
      --engine transformers

Core arguments
--------------

- ``--backend``
- ``--dataset``
- ``--model``
- ``--engine``
- ``--device``
- ``--num-samples``
- ``--capture``
- ``--analyses``

Project layout
--------------

- ``src/flex_eval/src/flexeval/`` contains the project package
- ``src/flex_eval/src/flexolmo_analysis/`` contains analysis code
- ``env/`` contains setup wrappers and requirement files
- ``EuroEval/`` contains the integrated EuroEval backend source
- ``olmes/`` contains the integrated olmes backend source

Analysis coverage
-----------------

- routing and coactivation analysis
- weight-space analysis
- latent-space and token-feature analysis
- router geometry and expert separability analysis
- representation geometry for embedding, hidden-state, and pre-router latents
