"""Assemble the training matrix from features and labels.

The target is the RESIDUAL forward return — what the stock did beyond what its
beta and the index explain — ranked within the day.

Why not simple excess return. Every stock on a given date shares the same
benchmark window, so `fwd_ret - bench_ret` subtracts a constant from the whole
cross-section and leaves the ranking exactly as it was. Measured on this data:
99.8% of rows share their day's window. Only the beta term varies per stock,
and it is what stops the model from rewarding a name that rose purely because
it is high beta and the market rose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.alpha.dataset import feature_names, open_joined


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series  # rank of the target within the day, 0..1
    dates: pd.Series
    tickers: pd.Series
    fwd_ret: pd.Series  # raw return, for turning predictions into money
    bench_ret: pd.Series | None = None

    def __len__(self) -> int:
        return len(self.X)


def _target_sql(horizon: int, target: str) -> str:
    if target == "residual":
        return f"fwd_ret_{horizon} - beta_60 * bench_ret_{horizon}"
    if target == "raw":
        return f"fwd_ret_{horizon}"
    raise ValueError(f"unknown target {target!r}; use 'residual' or 'raw'")


def load(
    horizon: int | None = None,
    include_holdout: bool = False,
    holdout_only: bool = False,
) -> Dataset:
    """Load the modelling sample.

    The frozen holdout is excluded unless asked for. Development happened while
    watching walk-forward results, so that score is optimistic by an unknown
    amount and the holdout is the only clean measurement left.
    """
    cfg = config.load("model")["dataset"]
    horizon = horizon or config.load("features")["label"]["horizon"]
    feats = feature_names(ranked=cfg["use_ranked_features"])
    target = cfg.get("target", "residual")

    where = [
        f"label_ok_{horizon}",
        f"fwd_ret_{horizon} IS NOT NULL",
    ]
    if target == "residual":
        # Both are NULL before the index series starts, so this is also what
        # enforces the 2019-09-12 floor.
        where += [f"bench_ret_{horizon} IS NOT NULL", "beta_60 IS NOT NULL"]
    if cfg.get("start_date"):
        where.append(f"date >= DATE '{cfg['start_date']}'")

    holdout = cfg.get("holdout_start")
    if holdout:
        if holdout_only:
            where.append(f"date >= DATE '{holdout}'")
        elif not include_holdout:
            where.append(f"date < DATE '{holdout}'")

    con = open_joined()
    try:
        df = con.execute(
            f"""
WITH base AS (
    SELECT ticker, date, {", ".join(feats)},
           fwd_ret_{horizon} AS fwd_ret,
           bench_ret_{horizon} AS bench_ret,
           ({_target_sql(horizon, target)}) AS target
    FROM d
    WHERE {" AND ".join(where)}
)
SELECT *,
       PERCENT_RANK() OVER (PARTITION BY date ORDER BY target, ticker) AS y
FROM base
ORDER BY date, ticker"""
        ).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("no rows matched — check start_date and holdout settings")

    X = df[feats].astype(np.float32)
    return Dataset(
        X=X,
        y=df["y"].astype(np.float32),
        dates=df["date"],
        tickers=df["ticker"],
        fwd_ret=df["fwd_ret"].astype(np.float32),
        bench_ret=df["bench_ret"].astype(np.float32),
    )


def daily_ic(pred: np.ndarray, y: np.ndarray, dates: pd.Series) -> pd.Series:
    """Spearman IC per day between predictions and the target rank.

    This is the only score that matters here. RMSE on a rank target is close to
    meaningless — the strategy trades the ORDER of the predictions, not their
    level, and a model can halve its RMSE while ranking no better.
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "p": pred, "y": y})
    return df.groupby("d").apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    )
