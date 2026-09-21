"""Exercise oversized fit and prediction requests against the configured API."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from time import monotonic

import numpy as np
import pandas as pd
from relarena_core.registry import registry
from tabpfn_client.client import ServiceClient

from tabpfn_rel.tfm import TFM_REGISTRY


def main() -> None:
    """Fit eight 200k-row contexts and predict one synthetic row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("classification", "regression"), required=True
    )
    args = parser.parse_args()
    recipe = registry.search_space("tabpfn-rel-client-2026-09-18").configs(
        n_trials=1, seed=0
    )[0]
    n_rows = recipe["subsample_samples"]
    indices = [np.arange(n_rows) for _ in range(8)]
    payload_bytes = len(
        json.dumps(
            {"SUBSAMPLE_SAMPLES": [rows.tolist() for rows in indices]},
            separators=(",", ":"),
        ).encode()
    )
    assert payload_bytes > 10_000_000
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=list("abcd"))
    target = data["a"] + rng.normal(scale=0.1, size=n_rows)
    if args.task == "classification":
        target = (target > 0).astype(int)
    spec = TFM_REGISTRY[recipe["tfm"]]
    factory = (
        spec.make_classifier if args.task == "classification" else spec.make_regressor
    )
    estimator = factory(
        device="cpu",
        seed=0,
        n_estimators=len(indices),
        inference_config={"SUBSAMPLE_SAMPLES": indices},
    )
    print(
        json.dumps(
            {
                "task": args.task,
                "endpoint": ServiceClient.base_url,
                "client_version": version("tabpfn-client"),
                "model_path": estimator.model_path,
                "index_config_bytes": payload_bytes,
                "rows": n_rows,
                "estimators": len(indices),
            }
        ),
        flush=True,
    )
    started = monotonic()
    estimator.fit(data, target)
    print(f"Fit passed after {monotonic() - started:.2f}s", flush=True)
    prediction = np.asarray(estimator.predict(data.iloc[:1]))
    assert prediction.shape == (1,), prediction.shape
    assert np.isfinite(prediction).all(), prediction
    print(f"Fit and prediction passed after {monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
