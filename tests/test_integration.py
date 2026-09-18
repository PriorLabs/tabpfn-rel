"""Real DFS, temporal tuning and prediction with an in-memory estimator backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from relarena_core.tfm import TFMSpec

from tabpfn_rel import (
    PredictiveContext,
    PredictiveQuery,
    PredictiveQuerySpec,
    TabPFNRel,
    tfm,
)

from ._data import write_database


class _Estimator:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _Estimator:
        assert len(X) == len(y) > 0
        assert X.shape[1] > 0
        self.classes_ = np.unique(y)
        self.mean_ = float(np.mean(y))
        self.columns_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def _make_estimator(**kwargs: object) -> _Estimator:
    return _Estimator()


@pytest.fixture
def query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> PredictiveContext:
    for variable in (
        "RELARENA_CACHE_DIR",
        "RELARENA_DISABLE_CACHE",
        "RELARENA_DISABLE_FEATURE_CACHE",
    ):
        monkeypatch.delenv(variable, raising=False)
    for name in ("tabpfn-v3", "tabpfn-v3-api"):
        monkeypatch.setitem(
            tfm.TFM_REGISTRY,
            name,
            TFMSpec(_make_estimator, _make_estimator, 100_000, supports_text=True),
        )
    task_path = write_database(
        tmp_path, getattr(request, "param", "binary_classification")
    )
    spec = PredictiveQuerySpec.from_yaml(str(task_path), data_dir=str(tmp_path))
    return PredictiveContext(spec, data_version="test-v1")


@pytest.mark.parametrize(
    "query", ["binary_classification", "regression"], indirect=True
)
@pytest.mark.parametrize("backend", ["local", "client"])
@pytest.mark.parametrize("n_trials", [0, 2])
def test_rpi_fits_tunes_and_reuses_prediction_cache(
    query: PredictiveContext, backend: str, n_trials: int, tmp_path: Path
) -> None:
    model = TabPFNRel(model=backend)
    assert (
        model.fit(query, n_trials=n_trials, seed=7, cache_dir=tmp_path / "cache")
        is model
    )
    predictions = model.predict(
        PredictiveQuery(entities="all", at_timestamp="test_timestamp")
    )
    pd.testing.assert_frame_equal(
        predictions,
        model._fitted.predict(
            PredictiveQuery(entities="all", at_timestamp="test_timestamp")
        ),
    )
    pd.testing.assert_frame_equal(
        predictions,
        model.predict(
            PredictiveQuery(entities="all", at_timestamp="test_timestamp"),
            cache_dir=tmp_path / "prediction-cache",
        ),
    )
    assert list((tmp_path / "prediction-cache").rglob("*.parquet"))
    assert sorted(predictions["customer_id"]) == ["a", "b", "c", "d"]
    assert predictions["y_pred"].notna().all()
    assert len(query.compute_test_labels()) == 4
    assert model.config["max_depth"] in (2, 3)
    if n_trials:
        assert len(model.trials) == 2
        assert all(trial.val_score is not None for trial in model.trials)
    else:
        assert model.trials is None
    if backend == "client":
        assert (
            "description__raw_text" in model._fitted._model._fitted.estimator.columns_
        )
    assert list((tmp_path / "cache").rglob("*.parquet"))


def test_wrapper_default_fit_and_failed_refit(
    query: PredictiveContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = TabPFNRel(model="local").fit(query)
    assert model.trials is None
    assert (
        len(
            model.predict(
                PredictiveQuery(entities="all", at_timestamp="test_timestamp")
            )
        )
        == 4
    )

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise ValueError("Fit failed")

    monkeypatch.setattr(query, "fit", fail_fit)
    with pytest.raises(ValueError, match="Fit failed"):
        model.fit(query)
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict(PredictiveQuery(entities="all", at_timestamp="test_timestamp"))
