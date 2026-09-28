from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from relarena_core.tfm import TFMSpec, predict_tfm
from relbench.base import TaskType

from tabpfn_rel.context import HardPoolContext, hard_pool_subsample_indices
from tabpfn_rel.model import (
    TabPFNRelClient20260928Model,
    TabPFNRelLocal20260928Model,
)
from tabpfn_rel.tfm import TFM_REGISTRY

fit_auto_hurdle = pytest.importorskip("tabpfn_rel.hurdle").fit_auto_hurdle


class RecordingEstimator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RecordingEstimator:
        self.X = X.copy()
        self.y = np.asarray(y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.25, 0.75], (len(X), 1))

    def predict(
        self, X: pd.DataFrame, *, output_type: str, quantiles: list[float]
    ) -> Any:
        if output_type == "quantiles":
            return [np.full(len(X), q * 12) for q in quantiles]
        return np.full(len(X), 6.0)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        TFM_REGISTRY,
        "hurdle-test",
        TFMSpec(RecordingEstimator, RecordingEstimator, 200_000, supports_text=True),
    )


@pytest.mark.parametrize(
    "zeros,active", [(0, False), (1, False), (2, True), (20, True)]
)
@pytest.mark.parametrize("text_dtype", ["string", "object"])
def test_extension_gate_and_median(
    backend: None, zeros: int, active: bool, text_dtype: str
) -> None:
    X = pd.DataFrame({"id": range(20), "text": pd.Series(["a"] * 20, dtype=text_dtype)})
    y = pd.Series([0.0] * zeros + [2.0] * (20 - zeros))
    fitted = fit_auto_hurdle(
        X,
        y,
        HardPoolContext(32, 2, 2.0, None),
        tfm="hurdle-test",
        seed=0,
        context_time=np.arange(20),
        zero_threshold=0.05,
    )
    hurdle = fitted.estimator
    assert hurdle.hurdle_ == active
    expected = 0.0 if zeros == 20 else 4.0 if active else 6.0
    np.testing.assert_allclose(hurdle.predict(X), expected)
    if zeros == 20:
        assert hurdle.regressor_ is None and hurdle.classifier_ is None


def test_positive_stage_contexts_keep_row_times_and_text(backend: None) -> None:
    X = pd.DataFrame({"id": range(16), "text": pd.Series(["a"] * 16, dtype="string")})
    y = pd.Series([0, 2] * 8)
    times = np.array([7, 2, 15, 1, 10, 8, 5, 3, 9, 0, 6, 12, 4, 11, 14, 13])
    fitted = fit_auto_hurdle(
        X,
        y,
        HardPoolContext(2, 2, 2.0, None),
        tfm="hurdle-test",
        seed=3,
        context_time=times,
        zero_threshold=0.25,
    )
    for stage, rows in [
        (fitted.estimator.classifier_, np.arange(16)),
        (fitted.estimator.regressor_, np.flatnonzero(y > 0)),
    ]:
        indices = hard_pool_subsample_indices(
            times[rows], K=2, M=4, n_estimators=2, seed=3
        )
        union = np.unique(np.concatenate(indices))
        estimator = stage.fitted_.estimator
        np.testing.assert_array_equal(estimator.X.id, rows[union])
        assert isinstance(estimator.X.text.dtype, pd.StringDtype)
        for actual, original in zip(
            estimator.kwargs["inference_config"]["SUBSAMPLE_SAMPLES"],
            indices,
            strict=True,
        ):
            np.testing.assert_array_equal(
                estimator.X.id.to_numpy()[actual], rows[original]
            )


@pytest.mark.parametrize(
    "model_class",
    [TabPFNRelClient20260928Model, TabPFNRelLocal20260928Model],
)
@pytest.mark.parametrize(
    "task_type", [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION]
)
def test_recipe_backend_dispatch(
    backend: None, model_class: type, task_type: TaskType
) -> None:
    model = model_class({"hurdle_zero_threshold": 0.05})
    model._tfm = "hurdle-test"
    X = pd.DataFrame({"id": range(16)})
    y = pd.Series([0, 1] * 8)
    fitted = model._fit_backend(
        X,
        y,
        task_type,
        HardPoolContext(32, 2, 2.0, None),
        seed=0,
        context_time=np.arange(16),
    )
    expected = 4.0 if task_type == TaskType.REGRESSION else 0.75
    np.testing.assert_allclose(predict_tfm(fitted, X), expected)
