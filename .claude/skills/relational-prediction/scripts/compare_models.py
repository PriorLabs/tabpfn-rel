"""Fit TabPFN-Rel and baselines on a task and print their test metrics."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from tabpfn_rel import PredictiveContext, PredictiveQuery

BASELINES = ["constant-global", "constant-per-entity", "lightgbm"]


def score(
    context: PredictiveContext, labels: pd.DataFrame, model: str, n_trials: int
) -> dict[str, object]:
    """Fit one model, predict every labelled test row and compute the metrics."""
    start = time.perf_counter()
    fitted = context.fit(model, n_trials=n_trials)
    batches = [
        fitted.predict(PredictiveQuery(entities=ids, at_timestamp=timestamp))
        for timestamp, ids in context.group_test_entities(labels)
    ]
    task = context.task
    scored = labels.merge(
        pd.concat(batches, ignore_index=True),
        on=[task.time_col, task.entity_col],
        how="left",
        validate="one_to_one",
    )
    predicted = scored[f"{task.target_col}_pred"]
    if predicted.isna().any():
        raise ValueError(f"{model} left {predicted.isna().sum()} test rows unscored.")
    row: dict[str, object] = {"model": model}
    for metric in task.metrics:
        row[metric.__name__] = round(metric(scored[task.target_col], predicted), 4)
    row["seconds"] = round(time.perf_counter() - start)
    return row


def main() -> None:
    """Score each model on the task's labelled test rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-end", default=None, help="Last date with full data.")
    parser.add_argument("--backend", choices=("client", "local"), default="client")
    parser.add_argument(
        "--models", nargs="+", default=None, help="Default: baselines and TabPFN-Rel."
    )
    parser.add_argument("--n-trials", type=int, default=0)
    args = parser.parse_args()
    models = args.models or [*BASELINES, f"tabpfn-rel-{args.backend}-latest"]

    context = PredictiveContext.from_yaml(args.task_yaml, data_dir=args.data_dir)
    labels = context.compute_test_labels(data_end_timestamp=args.data_end)
    print(f"{len(labels):,} labelled test rows")
    results = pd.DataFrame([score(context, labels, m, args.n_trials) for m in models])
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
