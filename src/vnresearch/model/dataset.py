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
    """The value the model ranks. ONE definition — stress.py imports this.

    It used to keep its own copy, which meant changing the target here left the
    2008 stress test quietly scoring the old one.
    """
    if target == "residual":
        return f"fwd_ret_{horizon} - beta_60 * bench_ret_{horizon}"
    if target == "residual_vol_scaled":
        # Residual return per unit of the stock's OWN recent volatility.
        #
        # Ranking the raw residual pays for volatility as much as for skill: a
        # name with twice the vol has roughly twice the spread and reaches the
        # top decile more often for that reason alone. Dividing asks whether the
        # move was large *for this stock*.
        #
        # The explicit CASE rather than GREATEST(vol_21, floor): GREATEST(NULL,
        # x) returns x in DuckDB, so a missing vol would be handed a fabricated
        # floor and a row with no computable target would rank as a real one.
        # Invariant 3, and the same trap as downside_vol_21.
        return (
            f"CASE WHEN vol_21 IS NOT NULL AND vol_21 > 0 "
            f"THEN (fwd_ret_{horizon} - beta_60 * bench_ret_{horizon}) / vol_21 END"
        )
    if target == "raw":
        return f"fwd_ret_{horizon}"
    raise ValueError(f"unknown target {target!r}; use 'residual', 'residual_vol_scaled' or 'raw'")


def target_filters(horizon: int, target: str) -> list[str]:
    """Rows on which the target cannot be computed, so they must not enter.

    Lives beside _target_sql because the two move together: a target that
    divides by vol_21 needs vol_21 present, and forgetting that leaves rows with
    a NULL target being percentile-ranked as though they were real values.
    """
    if target not in ("residual", "residual_vol_scaled"):
        return []
    # Both are NULL before the index series starts, so this is also what keeps
    # unbenchmarked early rows out of the sample.
    f = [f"bench_ret_{horizon} IS NOT NULL", "beta_60 IS NOT NULL"]
    if target == "residual_vol_scaled":
        f += ["vol_21 IS NOT NULL", "vol_21 > 0"]
    return f


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
        *target_filters(horizon, target),
    ]
    if cfg.get("start_date"):
        where.append(f"date >= DATE '{cfg['start_date']}'")

    holdout = cfg.get("holdout_start")
    if holdout:
        if holdout_only:
            where.append(f"date >= DATE '{holdout}'")
        elif not include_holdout:
            where.append(f"date < DATE '{holdout}'")

    # include_holdout=True because this function does its own holdout filtering
    # below — the view must not have already removed the rows holdout_only wants.
    con = open_joined(include_holdout=True)
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
