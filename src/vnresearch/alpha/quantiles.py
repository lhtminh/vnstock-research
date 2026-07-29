"""Quantile spread and signal turnover.

IC says a feature ranks correctly. These two say whether that ranking is
tradeable:

SPREAD — sort the universe into buckets each day and compare their forward
returns. A useful feature is monotonic across buckets. One that only separates
the extreme tails is fragile: it depends on a handful of names per day.

TURNOVER — how much the top bucket changes from one rebalance to the next. In
Vietnam a round trip costs well over 0.5% once the 0.1% sell tax is included,
so a feature that replaces its whole book weekly needs a very large spread just
to break even.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd

from vnresearch.alpha.dataset import MIN_NAMES_PER_DAY, open_joined


def spread(feature: str, horizon: int = 5, buckets: int = 5) -> pd.DataFrame:
    """Mean forward return per bucket, plus the count behind each number."""
    con = open_joined()
    try:
        sql = f"""
WITH elig AS (
    SELECT ticker, date, {feature} AS f, fwd_ret_{horizon} AS r
    FROM d
    WHERE label_ok_{horizon} AND {feature} IS NOT NULL
),
-- Days with too few names would put one or two stocks in each bucket, so the
-- bucket means would be noise. Filtered here because a window function cannot
-- live in a WHERE clause.
big_days AS (
    SELECT date FROM elig GROUP BY date HAVING count(*) >= {MIN_NAMES_PER_DAY}
),
b AS (
    SELECT
        e.date, e.f, e.r,
        -- ticker breaks ties. Percentile ranks tie often, and without a total
        -- ordering NTILE splits tied rows differently on every run, so the same
        -- query returns slightly different bucket means each time.
        NTILE({buckets}) OVER (PARTITION BY e.date ORDER BY e.f, e.ticker) AS bucket
    FROM elig e JOIN big_days USING (date)
),
per_day AS (
    SELECT date, bucket, avg(r) AS mean_r FROM b GROUP BY date, bucket
)
SELECT bucket,
       count(*)                 AS days,
       100 * avg(mean_r)        AS mean_fwd_pct,
       100 * stddev_samp(mean_r) AS sd_pct
FROM per_day
GROUP BY bucket
ORDER BY bucket
"""
        return con.execute(sql).df()
    finally:
        con.close()


def spread_summary(feature: str, horizon: int = 5, buckets: int = 5) -> dict:
    """Top-minus-bottom spread and whether the buckets increase in order."""
    df = spread(feature, horizon, buckets)
    if df.empty:
        return {"feature": feature, "spread_pct": None, "monotonic": False}
    vals = df["mean_fwd_pct"].tolist()
    return {
        "feature": feature,
        "spread_pct": vals[-1] - vals[0],
        "monotonic": all(b >= a for a, b in pairwise(vals))
        or all(b <= a for a, b in pairwise(vals)),
        "top_pct": vals[-1],
        "bottom_pct": vals[0],
    }


def turnover(feature: str, top_frac: float = 0.2, step: int = 5) -> float:
    """Fraction of the top bucket replaced between rebalances, 0..1.

    Measured every `step` sessions to match the rebalance the strategy would
    actually run, not daily — daily turnover overstates the cost of a weekly
    strategy by roughly the step size.
    """
    con = open_joined()
    try:
        df = con.execute(
            f"""SELECT date, ticker FROM d
                WHERE {feature} >= {1 - top_frac} AND {feature} IS NOT NULL
                ORDER BY date"""
        ).df()
    finally:
        con.close()
    if df.empty:
        return float("nan")

    days = sorted(df["date"].unique())[::step]
    by_day = {d: set(g["ticker"]) for d, g in df[df["date"].isin(days)].groupby("date")}
    changes = []
    prev = None
    for d in days:
        cur = by_day.get(d, set())
        if prev and cur:
            changes.append(len(cur - prev) / len(cur))
        prev = cur
    return float(sum(changes) / len(changes)) if changes else float("nan")
