"""Information coefficient: does a feature rank tomorrow's winners?

IC is the correlation between a feature's cross-sectional rank today and the
forward-return rank. Both sides are already ranks, so this is Spearman.

How to read the output. Mean IC of 0.02-0.05 is a genuinely useful equity
feature; above 0.15 on daily data almost always means leakage rather than
skill. IR (mean/sd) matters more than mean IC alone — a feature that is right
by a little, consistently, beats one that is right by a lot, occasionally.

Every statistic is computed per day and then averaged across days. Pooling all
observations instead would let a single volatile month dominate, and would
count 300 names on one day as 300 independent facts when they are mostly one.
"""

from __future__ import annotations

import pandas as pd

from vnresearch.alpha.dataset import MIN_NAMES_PER_DAY, feature_names, open_joined


def _daily_ic_sql(features: list[str], horizon: int) -> str:
    cors = [f"CORR({f}, fwd_rank_{horizon}) AS {f}" for f in features]
    return f"""
SELECT date, count(*) AS n, {", ".join(cors)}
FROM d
WHERE label_ok_{horizon}
GROUP BY date
HAVING count(*) >= {MIN_NAMES_PER_DAY}
"""


def ic_table(horizon: int = 5, features: list[str] | None = None) -> pd.DataFrame:
    """Per-feature IC statistics at one horizon, strongest first.

    The per-day correlations are computed in SQL because that pass touches ~1M
    rows. Summarising a few thousand daily values afterwards is pandas work.
    """
    features = features or feature_names()
    con = open_joined()
    try:
        daily = con.execute(_daily_ic_sql(features, horizon)).df()
    finally:
        con.close()

    rows = []
    for f in features:
        s = daily[f].dropna()
        if s.empty:
            continue
        mean, sd = s.mean(), s.std(ddof=1)
        ir = mean / sd if sd else float("nan")
        rows.append(
            {
                "feature": f,
                "days": len(s),
                "mean_ic": mean,
                "sd_ic": sd,
                "ir": ir,
                # t-stat of the mean IC: is this distinguishable from zero at all?
                "t_stat": ir * (len(s) ** 0.5),
                "hit_rate": float((s > 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.reindex(out["mean_ic"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def ic_decay(horizons: list[int], features: list[str] | None = None) -> pd.DataFrame:
    """Mean IC at each horizon — where a signal stops paying is its holding period."""
    frames = []
    for h in horizons:
        t = ic_table(h, features)[["feature", "mean_ic", "ir"]]
        t = t.rename(columns={"mean_ic": f"ic_{h}", "ir": f"ir_{h}"})
        frames.append(t.set_index("feature"))
    return pd.concat(frames, axis=1).reset_index()


def daily_ic(feature: str, horizon: int = 5) -> pd.DataFrame:
    """The IC time series for one feature — for plotting or regime checks."""
    con = open_joined()
    try:
        return con.execute(_daily_ic_sql([feature], horizon)).df()
    finally:
        con.close()


def correlation(features: list[str] | None = None) -> pd.DataFrame:
    """Average daily cross-sectional correlation between features.

    Two features correlated above ~0.9 carry the same information; feeding both
    to a tree splits their importance and tells you neither one matters.
    """
    features = features or feature_names()
    con = open_joined()
    try:
        df = con.execute(f"SELECT date, {', '.join(features)} FROM d").df()
    finally:
        con.close()
    # Correlate within each day, then average, so the market's own moves do not
    # create correlation that is not cross-sectional.
    return df.groupby("date")[features].corr().groupby(level=1).mean().loc[features, features]
