"""Model-facing interface to relational fitting and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from relarena_core.userdb import PredictiveQuery


class TabPFNRel:
    """Fit a relational query with local or hosted TabPFN inference."""

    def __init__(
        self,
        *,
        model: Literal["client", "local", "client-2026-08-15", "client-2026-09-18"],
    ) -> None:
        """Select the inference backend."""
        if model not in ("client", "local", "client-2026-08-15", "client-2026-09-18"):
            raise ValueError(
                "model must be 'client', 'local', 'client-2026-08-15', "
                "or 'client-2026-09-18'."
            )
        self._model = (
            "tabpfn-rel-client-2026-08-15"
            if model == "client"
            else f"tabpfn-rel-{model}"
        )
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
