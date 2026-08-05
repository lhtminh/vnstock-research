"""Walk-forward training and the leakage controls.

Everything is scored by IC on the held-out fold — the rank correlation between
prediction and forward return. Not RMSE: the strategy trades the ORDER of the
predictions, and a model can improve its RMSE while ranking no better.

Results are reported per fold, never pooled into one number. Pooling hides the
thing you most need to see, which is whether the edge survived every regime or
came from one good year.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.model import dataset as ds
from vnresearch.model.cv import PurgedWalkForward


def make_model(name: str, params: dict | None = None):
    """Build a model with importance measured by GAIN.

    Both libraries default to counting SPLITS, which answers "how often did the
    tree cut on this?" rather than "how much did it learn from it?". The two
    disagree badly here: hl_range ranks 18th by split and 1st by gain, ret_1
    ranks 20th and 2nd. Split counts favour features with many distinct values,
    not useful ones.
    """
    cfg = config.load("model")
    p = {**cfg[name], **(params or {})}
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(importance_type="gain", **p)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(importance_type="gain", **p)
    raise ValueError(f"unknown model: {name}")


@dataclass
class FoldResult:
    fold: int
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    mean_ic: float
    ir: float
    hit_rate: float


@dataclass
class Run:
    model: str
    folds: list[FoldResult]
    importance: pd.Series
    oos: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Per-fold gain, normalised within each fold. Kept alongside the average
    # because the average hides the case that matters most when judging a new
    # feature: one that carries a whole fold and does nothing in the other three
    # is noise that happened to fit, and it reads as respectable once averaged.
    importance_by_fold: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])

    @property
    def mean_ic(self) -> float:
        return float(np.nanmean([f.mean_ic for f in self.folds]))


def walk_forward(
    model_name: str = "lightgbm",
    data: ds.Dataset | None = None,
    horizon: int | None = None,
    verbose: bool = True,
    params: dict | None = None,
) -> Run:
    cfg = config.load("model")["cv"]
    horizon = horizon or config.load("features")["label"]["horizon"]
    data = data or ds.load(horizon)

    splitter = PurgedWalkForward(
        n_splits=cfg["n_splits"],
        horizon=horizon,
        embargo=cfg["embargo"],
        min_train_days=cfg["min_train_days"],
    )

    folds, importances, oos = [], [], []
    for i, (tr, te) in enumerate(splitter.split(data.dates)):
        model = make_model(model_name, params)
        model.fit(data.X.iloc[tr], data.y.iloc[tr])
        pred = model.predict(data.X.iloc[te])

        ic = ds.daily_ic(pred, data.y.iloc[te].to_numpy(), data.dates.iloc[te]).dropna()
        dates_te = pd.to_datetime(data.dates.iloc[te])
        folds.append(
            FoldResult(
                fold=i,
                train_end=str(pd.to_datetime(data.dates.iloc[tr]).max().date()),
                test_start=str(dates_te.min().date()),
                test_end=str(dates_te.max().date()),
                n_train=len(tr),
                n_test=len(te),
                mean_ic=float(ic.mean()),
                ir=float(ic.mean() / ic.std()) if ic.std() else float("nan"),
                hit_rate=float((ic > 0).mean()),
            )
        )
        importances.append(pd.Series(model.feature_importances_, index=data.X.columns))
        oos.append(
            pd.DataFrame(
                {
                    "date": data.dates.iloc[te].to_numpy(),
                    "ticker": data.tickers.iloc[te].to_numpy(),
                    "pred": pred,
                    "fwd_ret": data.fwd_ret.iloc[te].to_numpy(),
                    "fold": i,
                }
            )
        )
        if verbose:
            f = folds[-1]
            print(
                f"  fold {i}  test {f.test_start}..{f.test_end}"
                f"  n={f.n_test:>7,}  IC {f.mean_ic:+.4f}  IR {f.ir:+.2f}"
                f"  hit {f.hit_rate:.0%}"
            )

    # Normalise per fold before averaging: gain is on an arbitrary scale that
    # grows with training set size, so a raw mean would let the last (largest)
    # fold decide the ranking on its own.
    by_fold = pd.concat([s / s.sum() for s in importances], axis=1).mul(100)
    by_fold.columns = [f"fold_{i}" for i in range(by_fold.shape[1])]
    imp = by_fold.mean(axis=1).sort_values(ascending=False)
    return Run(
        model=model_name,
        folds=folds,
        importance=imp,
        oos=pd.concat(oos, ignore_index=True),
        importance_by_fold=by_fold.loc[imp.index],
    )


def evaluate_holdout(model_name: str = "lightgbm", horizon: int | None = None) -> dict:
    """Train once on everything before the holdout, score once on the holdout.

    This is the only number here that was not influenced by development. The
    walk-forward score was watched while the pipeline was being built, so it is
    optimistic by an unknown amount. Run this sparingly — every look spends a
    little of what makes it trustworthy.
    """
    horizon = horizon or config.load("features")["label"]["horizon"]
    dev = ds.load(horizon)
    hold = ds.load(horizon, holdout_only=True)

    model = make_model(model_name)
    model.fit(dev.X, dev.y)
    pred = model.predict(hold.X)

    ic = ds.daily_ic(pred, hold.y.to_numpy(), hold.dates).dropna()
    return {
        "train_rows": len(dev),
        "train_end": str(pd.to_datetime(dev.dates).max().date()),
        "holdout_rows": len(hold),
        "holdout_start": str(pd.to_datetime(hold.dates).min().date()),
        "holdout_end": str(pd.to_datetime(hold.dates).max().date()),
        "mean_ic": float(ic.mean()),
        "ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "oos": pd.DataFrame(
            {
                "date": hold.dates.to_numpy(),
                "ticker": hold.tickers.to_numpy(),
                "pred": pred,
                "fwd_ret": hold.fwd_ret.to_numpy(),
            }
        ),
    }


def leakage_controls(model_name: str = "lightgbm", horizon: int | None = None) -> pd.DataFrame:
    """Two controls that must both pass before believing any result.

    NEGATIVE — shuffle the label within each day. The features are untouched and
    the cross-section is intact, only the pairing is destroyed. IC must collapse
    to ~0. If it does not, something in the pipeline is leaking.

    POSITIVE — hand the model the answer as a feature. IC must go to ~1. This
    proves the harness can SEE leakage; without it, a clean negative control
    might just mean the test is blind.
    """
    horizon = horizon or config.load("features")["label"]["horizon"]
    data = ds.load(horizon)

    rng = np.random.default_rng(0)
    shuffled = data.y.groupby(pd.to_datetime(data.dates).to_numpy()).transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    negative = walk_forward(
        model_name,
        data=ds.Dataset(data.X, shuffled, data.dates, data.tickers, data.fwd_ret),
        horizon=horizon,
        verbose=False,
    )

    X_leak = data.X.copy()
    X_leak["ANSWER"] = data.y.to_numpy()
    positive = walk_forward(
        model_name,
        data=ds.Dataset(X_leak, data.y, data.dates, data.tickers, data.fwd_ret),
        horizon=horizon,
        verbose=False,
    )

    real = walk_forward(model_name, data=data, horizon=horizon, verbose=False)
    return pd.DataFrame(
        [
            {"control": "real features", "mean_ic": real.mean_ic, "expect": "small, positive"},
            {"control": "shuffled labels", "mean_ic": negative.mean_ic, "expect": "~0.00"},
            {"control": "answer as feature", "mean_ic": positive.mean_ic, "expect": "~1.00"},
        ]
    )
