"""Automatic hurdle regression with stage-specific recency contexts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from relarena_core.tfm import FittedTFM
from relbench.base import TaskType
from sklearn.base import BaseEstimator
from tabpfn_extensions.hurdle import AutoHurdleRegressor

from tabpfn_rel.context import ContextStrategy


class ContextEstimator(BaseEstimator):
    """Adapt context fitting to the cloneable hurdle estimator interface."""

    def __init__(
        self,
        context: ContextStrategy,
        tfm: str,
        seed: int,
        task_type: TaskType,
        context_time: np.ndarray | None,
    ) -> None:
        """Store the context recipe and row-aligned training timestamps."""
        self.context = context
        self.tfm = tfm
        self.seed = seed
        self.task_type = task_type
        self.context_time = context_time

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ContextEstimator:
        """Select contexts after the hurdle has filtered positive rows."""
        cutoff = (
            None if self.context_time is None else self.context_time[X.index.to_numpy()]
        )
        self.fitted_ = self.context.fit(
            X.reset_index(drop=True),
            pd.Series(y),
            self.task_type,
            tfm=self.tfm,
            seed=self.seed,
            context_time=cutoff,
        )
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.classes_ = self.fitted_.estimator.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return classifier probabilities in fitted class order."""
        return self.fitted_.estimator.predict_proba(
            X.reindex(columns=self.fitted_.feature_cols)
        )

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> Any:
        """Forward distribution requests to the fitted positive-stage estimator."""
        return self.fitted_.estimator.predict(
            X.reindex(columns=self.fitted_.feature_cols), **kwargs
        )


def fit_auto_hurdle(
    df: pd.DataFrame,
    y: pd.Series,
    context: ContextStrategy,
    *,
    tfm: str,
    seed: int,
    context_time: np.ndarray | None,
    zero_threshold: float,
) -> FittedTFM:
    """Fit the extension on all targets, preserving row identity during splitting."""
    estimator = AutoHurdleRegressor(
        classifier=ContextEstimator(
            context, tfm, seed, TaskType.BINARY_CLASSIFICATION, context_time
        ),
        regressor=ContextEstimator(
            context, tfm, seed, TaskType.REGRESSION, context_time
        ),
        hurdle="auto",
        zero_threshold=zero_threshold,
    )
    estimator.fit(df.reset_index(drop=True), y.to_numpy())
    return FittedTFM(estimator, list(df.columns), TaskType.REGRESSION)
