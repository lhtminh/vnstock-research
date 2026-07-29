"""Purged walk-forward splitter tests.

The failure mode is silent: a splitter that forgets to purge reports a better
score and nobody notices until the strategy trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vnresearch.model.cv import PurgedWalkForward


def _dates(n=1200, per_day=5):
    days = pd.bdate_range("2015-01-01", periods=n).date
    return pd.Series(np.repeat(days, per_day))


def test_train_is_always_before_test():
    d = _dates()
    s = PurgedWalkForward(n_splits=4, horizon=5, min_train_days=300)
    for tr, te in s.split(d):
        assert pd.to_datetime(d.iloc[tr]).max() < pd.to_datetime(d.iloc[te]).min()


def test_purge_gap_covers_the_label_window():
    """No training label may still be running when the test block opens."""
    d = _dates()
    horizon = 5
    s = PurgedWalkForward(n_splits=4, horizon=horizon, min_train_days=300)
    sessions = np.unique(pd.to_datetime(d).to_numpy())
    for tr, te in s.split(d):
        train_end = pd.to_datetime(d.iloc[tr]).max().to_datetime64()
        test_start = pd.to_datetime(d.iloc[te]).min().to_datetime64()
        # Count sessions strictly between the two.
        gap = np.searchsorted(sessions, test_start) - np.searchsorted(sessions, train_end)
        assert gap >= horizon + 1, f"purge gap {gap} < horizon+1 ({horizon + 1})"


def test_folds_do_not_overlap_each_other():
    d = _dates()
    s = PurgedWalkForward(n_splits=5, horizon=5, min_train_days=300)
    seen: set[int] = set()
    for _, te in s.split(d):
        idx = set(te.tolist())
        assert not (idx & seen), "test blocks overlap"
        seen |= idx


def test_train_window_expands():
    d = _dates()
    s = PurgedWalkForward(n_splits=4, horizon=5, min_train_days=300)
    sizes = [len(tr) for tr, _ in s.split(d)]
    assert sizes == sorted(sizes), "training window must grow, not slide"


def test_longer_horizon_widens_the_purge():
    d = _dates()
    gaps = {}
    sessions = np.unique(pd.to_datetime(d).to_numpy())
    for h in (1, 21):
        s = PurgedWalkForward(n_splits=3, horizon=h, min_train_days=300)
        tr, te = next(iter(s.split(d)))
        train_end = pd.to_datetime(d.iloc[tr]).max().to_datetime64()
        test_start = pd.to_datetime(d.iloc[te]).min().to_datetime64()
        gaps[h] = np.searchsorted(sessions, test_start) - np.searchsorted(sessions, train_end)
    assert gaps[21] > gaps[1]


def test_refuses_when_history_is_too_short():
    d = _dates(n=50)
    s = PurgedWalkForward(n_splits=5, horizon=5, min_train_days=500)
    with pytest.raises(ValueError, match="not enough"):
        list(s.split(d))


def test_describe_reports_every_fold():
    d = _dates()
    s = PurgedWalkForward(n_splits=4, horizon=5, min_train_days=300)
    out = s.describe(d)
    assert len(out) == 4
    assert (out["gap_days"] > 0).all()
