"""Model-facing interface to relational fitting and prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from relarena_core.results import TrialResult
from relarena_core.userdb import FittedPredictor, PredictiveContext, PredictiveQuery


class TabPFNRel:
    """Fit a relational context with local or hosted TabPFN inference."""

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
        self._fitted: FittedPredictor | None = None

    @property
    def trials(self) -> list[TrialResult] | None:
        """Return tuning results, or None when no tuning has completed."""
        return None if self._fitted is None else self._fitted.trials

    @property
    def config(self) -> dict[str, Any] | None:
        """Return the fitted configuration, or None before a successful fit."""
        return None if self._fitted is None else self._fitted.config

    def fit(
        self,
        context: PredictiveContext,
        *,
        n_trials: int = 0,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> TabPFNRel:
        """Fit a model on the context and return self."""
        self._fitted = None
        self._fitted = context.fit(
            self._model, n_trials=n_trials, seed=seed, cache_dir=cache_dir
        )
        return self

    def predict(
        self, query: PredictiveQuery, *, cache_dir: str | Path | None = None
    ) -> pd.DataFrame:
        """Predict the query's entities and timestamp using the fitted context."""
        if self._fitted is None:
            raise RuntimeError("Call fit(...) before predict().")
        return self._fitted.predict(query, cache_dir=cache_dir)
