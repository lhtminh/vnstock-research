"""Score the model on a crisis it was never trained on.

2008 is deliberately excluded from training: the market had ~30 liquid names,
a ±5% price band that the regulator narrowed mid-crash, and T+3 settlement. A
model fitted there would learn a market that no longer exists. But a model that
FALLS APART there has learned something fragile, and that is worth knowing
before trusting it with money.

This is the cleanest measurement available in the whole project. The holdout has
been looked at repeatedly; 2008 sits entirely outside every sample used to
build, tune or select anything, and it is a regime — a -66% year — that nothing
in the training data resembles.

Read it as a survival check, not a performance estimate. The universe is small
enough that the cross-section is coarse, and the emergency-band window
(2008-03-25 to 2008-08-18) is excluded from trading because the operative limit
there cannot be stated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.alpha.dataset import feature_names, open_joined
from vnresearch.model import dataset as ds
from vnresearch.model import train

STRESS_START = "2007-01-01"
STRESS_END = "2009-01-01"


def load_window(start: str = STRESS_START, end: str = STRESS_END) -> ds.Dataset:
    """The stress sample, built exactly like the training sample."""
    cfg = config.load("model")["dataset"]
    horizon = config.load("features")["label"]["horizon"]
    feats = feature_names(ranked=cfg["use_ranked_features"])
    target = cfg.get("target", "residual")

    # Both imported, never restated. A local copy of either one means this test
    # keeps scoring the previous target after someone changes it, and the result
    # still looks plausible.
    where = [
        f"label_ok_{horizon}",
        f"fwd_ret_{horizon} IS NOT NULL",
        f"date >= DATE '{start}'",
        f"date < DATE '{end}'",
        *ds.target_filters(horizon, target),
    ]
    expr = ds._target_sql(horizon, target)

    con = open_joined(include_holdout=True, apply_start=False)
    try:
        df = con.execute(
            f"""
WITH base AS (
    SELECT ticker, date, {", ".join(feats)},
           fwd_ret_{horizon} AS fwd_ret, bench_ret_{horizon} AS bench_ret,
           ({expr}) AS target
    FROM d WHERE {" AND ".join(where)}
)
SELECT *, PERCENT_RANK() OVER (PARTITION BY date ORDER BY target, ticker) AS y
FROM base ORDER BY date, ticker"""
        ).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"no rows in {start}..{end}")

    return ds.Dataset(
        X=df[feats].astype(np.float32),
        y=df["y"].astype(np.float32),
        dates=df["date"],
        tickers=df["ticker"],
        fwd_ret=df["fwd_ret"].astype(np.float32),
        bench_ret=df["bench_ret"].astype(np.float32),
    )


def run(model_name: str = "lightgbm") -> dict:
    """Train on the normal sample, score on the crisis window."""
    dev = ds.load()
    stress = load_window()

    model = train.make_model(model_name)
    model.fit(dev.X, dev.y)
    pred = model.predict(stress.X)

    ic = ds.daily_ic(pred, stress.y.to_numpy(), stress.dates).dropna()
    dates = pd.to_datetime(stress.dates)
    names_per_day = stress.tickers.groupby(dates.to_numpy()).nunique()

    return {
        "train_rows": len(dev),
        "train_start": str(pd.to_datetime(dev.dates).min().date()),
        "stress_rows": len(stress),
        "stress_start": str(dates.min().date()),
        "stress_end": str(dates.max().date()),
        "median_names_per_day": float(names_per_day.median()),
        "mean_ic": float(ic.mean()),
        "ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
        "hit_rate": float((ic > 0).mean()),
        "scored_days": len(ic),
        "oos": pd.DataFrame(
            {
                "date": stress.dates.to_numpy(),
                "ticker": stress.tickers.to_numpy(),
                "pred": pred,
                "fwd_ret": stress.fwd_ret.to_numpy(),
            }
        ),
    }
