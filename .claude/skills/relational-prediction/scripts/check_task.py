"""Print label counts and balance per split for a task YAML."""

from __future__ import annotations

import argparse

import pandas as pd
from relbench.base import TaskType

from tabpfn_rel import PredictiveContext


def describe(name: str, labels: pd.DataFrame, context: PredictiveContext) -> dict:
    """Summarize one split's labelled rows."""
    task = context.task
    target = labels[task.target_col]
    row = {
        "split": name,
        "rows": len(labels),
        "anchors": labels[task.time_col].nunique(),
        "entities": labels[task.entity_col].nunique(),
        "first_anchor": labels[task.time_col].min(),
        "last_anchor": labels[task.time_col].max(),
    }
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        row["positive_rate"] = round(target.mean(), 4)
    else:
        row["mean"], row["std"] = round(target.mean(), 4), round(target.std(), 4)
        row["zero_share"] = round((target == 0).mean(), 4)
    return row


def main() -> None:
    """Load the task, compute its labels and print one row per split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-end", default=None, help="Last date with full data.")
    args = parser.parse_args()
    context = PredictiveContext.from_yaml(args.task_yaml, data_dir=args.data_dir)
    splits = {
        "train": context.task.get_table("train").df,
        "val": context.task.get_table("val").df,
        "test": context.compute_test_labels(data_end_timestamp=args.data_end),
    }
    summary = pd.DataFrame([describe(n, df, context) for n, df in splits.items()])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
