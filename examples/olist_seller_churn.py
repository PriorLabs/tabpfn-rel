"""Predict seller churn from the Olist tables and evaluate held-out ROC-AUC.

Download the data and configure a backend using the repository README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from tabpfn_rel import (
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
    TabPFNRel,
)


def prepare_olist_data(csv_dir: str) -> Path:
    """Add purchase timestamps to order items for temporal feature filtering."""
    csv = Path(csv_dir)
    orders = pd.read_csv(
        csv / "olist_orders_dataset.csv",
        usecols=["order_id", "order_purchase_timestamp"],
    )
    items = pd.read_csv(csv / "olist_order_items_dataset.csv").merge(
        orders, on="order_id", how="left"
    )
    items.rename(columns={"order_purchase_timestamp": "purchase_ts"}).to_csv(
        csv / "order_items.csv", index=False
    )
    return csv


def fit_predict_and_evaluate(
    spec: PredictiveQuerySpec,
    model: TabPFNRel,
    *,
    n_trials: int,
) -> None:
    """Fit the model and score sellers with observed test-window outcomes."""
    pq = PredictiveContext(spec)
    model.fit(pq, n_trials=n_trials, seed=0)
    labels = pq.compute_test_labels()
    # For multiple test timestamps, see:
    # https://github.com/PriorLabs/relarena/blob/adrian/context-query/examples/relbench_test_rows.py
    preds = model.predict(
        PredictiveQuery(
            entities=labels[pq.task.entity_col].tolist(),
            at_timestamp="test_timestamp",
        )
    )
    scored = labels.merge(
        preds,
        on=[pq.task.time_col, pq.task.entity_col],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not scored["_merge"].eq("both").all():
        raise ValueError("Prediction rows must exactly match test label rows.")
    scored = scored.drop(columns="_merge")
    if scored[f"{pq.task.target_col}_pred"].isna().any():
        raise RuntimeError("Predictions are missing rows from the test cohort.")
    roc_auc = roc_auc_score(
        scored[pq.task.target_col], scored[f"{pq.task.target_col}_pred"]
    )
    print(preds.head())
    print(f"Test ROC-AUC: {roc_auc:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse the example's local-versus-hosted inference choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("local", "client"),
        default="client",
        help="Hosted TabPFN API (default), or local TabPFN without text features.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="Temporal tuning trials; 0 fits the default configuration.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/olist",
        help="Directory containing the downloaded Olist CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    here = Path(__file__).parent
    data_dir = prepare_olist_data(args.data_dir)
    spec = PredictiveQuerySpec.from_yaml(
        str(here / "olist_seller_churn.yaml"), data_dir=str(data_dir)
    )
    model = TabPFNRel(model=args.backend)
    fit_predict_and_evaluate(spec, model=model, n_trials=args.n_trials)
