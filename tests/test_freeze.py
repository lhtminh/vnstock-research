"""Frozen-model tests.

The failure this guards against is silent. A model scores a feature matrix
positionally, so handing it the right columns in the wrong order — or a column
set that has moved since the fit — produces confident numbers and no error. In
a backtest that shows up as a bad result; in paper trading it shows up as a
portfolio, which is worse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vnresearch.model import freeze


class _Recorder:
    """Estimator stand-in that reports the column order it was handed."""

    def __init__(self):
        self.seen: list[str] | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.seen = list(X.columns)
        return np.arange(len(X), dtype=float)


def _frozen(features: list[str], est=None) -> freeze.FrozenModel:
    return freeze.FrozenModel(
        estimator=est or _Recorder(),
        manifest={"features": features, "model_hash": "deadbeef", "train_end": "2026-08-03"},
        path=Path("lightgbm_2026-08-03_deadbeef.pkl"),
    )


def test_predict_reorders_to_the_trained_order():
    """Columns in a different order must be corrected, not trusted."""
    est = _Recorder()
    fm = _frozen(["a", "b", "c"], est)
    fm.predict(pd.DataFrame({"c": [1.0], "a": [2.0], "b": [3.0]}))
    assert est.seen == ["a", "b", "c"]


def test_predict_refuses_a_missing_feature():
    fm = _frozen(["a", "b", "c"])
    with pytest.raises(ValueError, match="missing=\\['c'\\]"):
        fm.predict(pd.DataFrame({"a": [1.0], "b": [2.0]}))


def test_predict_refuses_an_extra_feature():
    """The dangerous direction. Silently dropping an unknown column would let a
    newly registered feature change nothing and still look like the registry and
    the model agree."""
    fm = _frozen(["a", "b"])
    with pytest.raises(ValueError, match="extra=\\['zzz'\\]"):
        fm.predict(pd.DataFrame({"a": [1.0], "b": [2.0], "zzz": [3.0]}))


def test_load_refuses_without_a_manifest(tmp_path):
    """A bare .pkl has no feature order, so nothing can verify a score from it."""
    pkl = tmp_path / "lightgbm_2026-08-03_abc.pkl"
    pkl.write_bytes(b"not really a model")
    with pytest.raises(FileNotFoundError, match="manifest"):
        freeze.load(pkl)


def test_latest_prefers_the_newest_training_window(tmp_path):
    """Re-freezing an older window later gives it a newer mtime. Picking by
    mtime would then quietly trade a stale model, so the date in the name wins."""
    old = tmp_path / "lightgbm_2026-01-15_aaaaaaaa.pkl"
    new = tmp_path / "lightgbm_2026-08-03_bbbbbbbb.pkl"
    new.write_bytes(b"x")
    old.write_bytes(b"x")  # written second -> newer mtime
    assert freeze.latest("lightgbm", model_dir=tmp_path) == new


def test_latest_raises_when_nothing_is_frozen(tmp_path):
    with pytest.raises(FileNotFoundError, match="vnr freeze"):
        freeze.latest("lightgbm", model_dir=tmp_path)


def test_manifest_round_trips(tmp_path):
    """The manifest must survive JSON. It carries numpy scalars and Paths from
    get_params(), which json refuses without a default — a freeze that writes
    the pickle then dies on the manifest leaves an unloadable artifact."""
    manifest = {
        "features": ["a_rank", "b_rank"],
        "params": {"n_estimators": np.int64(400), "learning_rate": np.float64(0.03)},
        "created_at": pd.Timestamp("2026-08-03"),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    back = json.loads(text)
    assert back["features"] == ["a_rank", "b_rank"]
    assert back["params"]["n_estimators"] in (400, "400")
