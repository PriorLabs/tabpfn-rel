# TabPFN-Rel

[![PyPI version](https://badge.fury.io/py/tabpfn-rel.svg)](https://pypi.org/project/tabpfn-rel/)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://pypi.org/project/tabpfn-rel/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://docs.priorlabs.ai/capabilities/relational)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)

TabPFN-Rel applies TabPFN to prediction tasks over relational databases. It follows
relationships between tables, builds features with Deep Feature Synthesis, and
uses TabPFN to predict outcomes for entities such as customers or sellers.

[Documentation](https://docs.priorlabs.ai/capabilities/relational) ·
[Olist cookbook](https://docs.priorlabs.ai/cookbook/relational_predictions_tabpfn_rel) ·
[Technical report](https://arxiv.org/abs/2608.16319)

## Try the Olist notebook

> [!TIP]
> Start with the [Olist seller-churn notebook](https://github.com/PriorLabs/tabpfn-cookbook/blob/main/notebooks/relational_predictions_tabpfn_rel.ipynb).
> It walks through a seven-table e-commerce database, task definitions, fitting,
> prediction, caching and baseline comparisons.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/tabpfn-cookbook/blob/main/notebooks/relational_predictions_tabpfn_rel.ipynb)

The task predicts whether a seller active in the past 30 days will receive no
orders in the next 30 days. You can also [read the walkthrough in the docs](https://docs.priorlabs.ai/cookbook/relational_predictions_tabpfn_rel)
or run the [local script](#run-the-example-as-a-local-script).

## Installation

Requires Python 3.11 or 3.12. Choose a backend:

```bash
pip install "tabpfn-rel[api]"    # Hosted inference
# Or: pip install "tabpfn-rel[local]"  # Local inference; GPU recommended
```

For hosted inference, authenticate before fitting:

```python
from tabpfn_client import init

init()
```

For local inference, follow the [model access guide](https://docs.priorlabs.ai/models/accessing-model-weights).

## Run the example as a local script

Prefer a Python script to a notebook? [The Olist example](examples/olist_seller_churn.py)
includes the same database schema and prediction task as YAML files, and reports
held-out ROC-AUC. It runs on your machine with either hosted or local inference.

<details>
<summary>Download the data and run the script</summary>

Clone the repository, then download the data with the Kaggle CLI (requires a
Kaggle account and configured credentials):

```bash
git clone https://github.com/PriorLabs/tabpfn-rel.git
cd tabpfn-rel
uvx kaggle datasets download -d olistbr/brazilian-ecommerce -p data/olist --unzip
uv sync --extra api --group cpu
uv run --no-sync python -c "from tabpfn_client import init; init()"
OMP_NUM_THREADS=1 uv run --no-sync python examples/olist_seller_churn.py
```

The default fits one configuration through the hosted API. Add `--n-trials 3` to
try temporal tuning, or `--data-dir /path/to/olist` to use an existing download.
Hosted fitting and prediction consume API quota.

For local inference, install with `uv sync --extra local` and run the script with
`--backend local`.

</details>

## Predict on your own database

The Relational Predictive Interface (RPI) makes TabPFN-Rel easier to use on your
own database: define a prediction task, then call `fit` and `predict`. RPI is
provided by `relarena-core`, which is installed automatically with TabPFN-Rel.

Define your tables, keys and timestamps in a database YAML file, then define the
target and temporal splits in a task YAML file. The
[task-definition guide](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md)
describes these formats; the Olist example provides complete files to adapt.

```python
from tabpfn_rel import PredictiveContext, PredictiveQuery, TabPFNRel

context = PredictiveContext.from_yaml("task.yaml", data_dir="data/")
model = TabPFNRel(model="client")
model.fit(context, n_trials=0)
query = PredictiveQuery(entities="all", at_timestamp="test_timestamp")
predictions = model.predict(query)
```

Use `model="local"` for local inference. `n_trials=0` (the default) fits the
default configuration; a positive budget enables temporal tuning. Pass `seed`
and `cache_dir` to `fit` to control tuning randomness and feature caching.
`predict` reuses that cache unless given another `cache_dir`.

Both query fields are required. Use `at_timestamp="test_timestamp"` for the
context cutoff or an explicit date for another prediction anchor. The database
remains frozen at the context cutoff, including for later anchors, to follow
RelArena's fixed-snapshot evaluation protocol and prevent post-cutoff data from
entering predictions. See
[RelArena's temporal-validation protocol](https://github.com/PriorLabs/relarena/blob/main/docs/temporal-validation.md#why-the-database-cutoff-matters).

To benchmark TabPFN-Rel against other methods, see
[RelArena](https://github.com/PriorLabs/relarena).

## Development

```bash
uv sync --locked --group cpu
OMP_NUM_THREADS=1 uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
uv build
```

Tests cover feature generation, context selection, temporal tuning and prediction
without model downloads or hosted API calls.
