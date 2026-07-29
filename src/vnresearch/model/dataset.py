"""Assemble the training matrix from features and labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.alpha.dataset import feature_names, open_joined


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series  # forward-return rank, 0..1
    dates: pd.Series
    tickers: pd.Series
    fwd_ret: pd.Series  # raw return, for turning predictions into money

    def __len__(self) -> int:
        return len(self.X)


def load(horizon: int | None = None) -> Dataset:
    cfg = config.load("model")["dataset"]
    horizon = horizon or config.load("features")["label"]["horizon"]
    feats = feature_names(ranked=cfg["use_ranked_features"])

    where = [f"label_ok_{horizon}", f"fwd_rank_{horizon} IS NOT NULL"]
    if cfg.get("start_date"):
        where.append(f"date >= DATE '{cfg['start_date']}'")

    con = open_joined()
    try:
        df = con.execute(
            f"""SELECT ticker, date, {", ".join(feats)},
                       fwd_rank_{horizon} AS y, fwd_ret_{horizon} AS fwd_ret
                FROM d
                WHERE {" AND ".join(where)}
                ORDER BY date, ticker"""
        ).df()
    finally:
        con.close()

    # float32 halves the memory and costs nothing: these are ranks in [0, 1].
    X = df[feats].astype(np.float32)
    return Dataset(
        X=X,
        y=df["y"].astype(np.float32),
        dates=df["date"],
        tickers=df["ticker"],
        fwd_ret=df["fwd_ret"].astype(np.float32),
    )


def daily_ic(pred: np.ndarray, y: np.ndarray, dates: pd.Series) -> pd.Series:
    """Spearman IC per day between predictions and the forward-return rank.

    This is the only score that matters here. RMSE on a rank target is close to
    meaningless — the strategy trades the ORDER of the predictions, not their
    level, and a model can halve its RMSE while ranking no better.
    """
    df = pd.DataFrame({"d": pd.to_datetime(dates).to_numpy(), "p": pred, "y": y})
    return df.groupby("d").apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 20 else np.nan,
        include_groups=False,
    )
