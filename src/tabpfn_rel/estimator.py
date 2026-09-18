"""Model-facing interface to relational fitting and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from relarena_core.userdb import PredictiveQuery


class TabPFNRel:
    """Fit a relational query with local or hosted TabPFN inference."""

    def __init__(self, *, model: Literal["client", "local"]) -> None:
        """Select the inference backend."""
        if model not in ("client", "local"):
            raise ValueError("model must be 'client' or 'local'.")
        self._model = f"tabpfn-rel-{model}"
        self._query: PredictiveQuery | None = None

    def fit(
        self,
        query: PredictiveQuery,
        *,
        n_trials: int = 0,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> TabPFNRel:
        """Fit the supplied query in place and return self.

        The query retains fitted state and tuning results. Use a separate query
        for each model. Positive n_trials enables temporal tuning.
        """
        self._query = None
        query.fit(self._model, n_trials=n_trials, seed=seed, cache_dir=cache_dir)
        self._query = query
        return self

    def predict(self, *, cache_dir: str | Path | None = None) -> pd.DataFrame:
        """Predict the fitted query's configured entities and timestamp."""
        if self._query is None:
            raise RuntimeError("Call fit(...) before predict().")
        return self._query.predict(cache_dir=cache_dir)
