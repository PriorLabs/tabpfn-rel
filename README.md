# TabPFN-Rel

TabPFN-Rel applies TabPFN to prediction tasks over relational databases. It follows
relationships between tables, builds features with Deep Feature Synthesis, and
uses TabPFN to predict outcomes for entities such as customers or sellers.

[Documentation](https://docs.priorlabs.ai/capabilities/relational) ·
[Olist cookbook](https://docs.priorlabs.ai/cookbook/relational_predictions_tabpfn_rel) ·
[Technical report](https://arxiv.org/abs/2608.16319)

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

## Try it on a real database

The [Olist seller-churn example](examples/olist_seller_churn.py) uses seven tables
from the Brazilian E-Commerce dataset. It predicts whether a seller active in the
past 30 days will receive no orders in the next 30 days, then reports held-out
ROC-AUC. The database schema and prediction task are included as YAML files.

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
`--backend local`. The [cookbook](https://docs.priorlabs.ai/cookbook/relational_predictions_tabpfn_rel)
walks through the same task interactively, including caching and baseline comparisons.

## Predict on your own database

Define your tables, keys and timestamps in a database YAML file, then define the
target and temporal splits in a task YAML file. The
[task-definition guide](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md)
describes these formats; the Olist example provides complete files to adapt.

```python
from tabpfn_rel import PredictiveQuery, PredictiveQuerySpec

spec = PredictiveQuerySpec.from_yaml("task.yaml", data_dir="data/")
query = PredictiveQuery(spec).fit("tabpfn-rel-client", n_trials=0)
predictions = query.predict()
```

Use `tabpfn-rel-local` for local inference. `n_trials=0` fits the default
configuration; a positive budget enables temporal tuning.

The Relational Predictive Interface (RPI) comes from `relarena-core`, installed
automatically. To benchmark TabPFN-Rel against other methods, see
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
