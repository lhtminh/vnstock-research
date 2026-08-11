"""Cross-sectional transforms: rank and z-score within each day's universe.

Why rank rather than raw values. A tree splits on absolute thresholds, so a
raw feature drags the market regime into the model: 2% daily volatility means
"calm" in 2021 and "panic" in 2008, and the same split cannot be right in both.
Ranking within the day removes the level and keeps the ordering, which is all a
cross-sectional strategy trades on.

Ranks are computed over the investable universe only. Including illiquid names
would shift every rank by however many unbuyable stocks happened to sit below.
"""

from __future__ import annotations


def finite(col: str) -> str:
    """NaN and infinity to NULL, applied to a raw feature before it is ranked.

    THIS IS THE NULL-RANK DEFECT WEARING A DIFFERENT HAT. `rank_expr` below
    guards `IS NOT NULL`, and NaN is not NULL — it is a value, it sorts last,
    and so every row carrying one was handed a rank of exactly 1.0. Measured on
    the published table before this existed: 177 rows of `skew_21` and 128 of
    `mkt_corr_60`, every one of them rated the most extreme name in the market
    that day on the strength of an undefined calculation.

    Where they come from: SKEWNESS and CORR return NaN, not NULL, when a window
    has no variance — a stock that did not move for 21 sessions, or one whose
    volume never changed. `label/speculation.py` hit the same thing from the
    other end, where the NaN overflowed a STDDEV instead of quietly winning.

    Applied to the materialised raw column rather than to the expression, so the
    window function underneath is computed once.
    """
    return f"CASE WHEN isnan({col}) OR isinf({col}) THEN NULL ELSE {col} END"


def rank_expr(col: str) -> str:
    """Percentile rank of col within the day's universe, 0..1. NULL stays NULL.

    A missing value must not be given a rank. SQL `ORDER BY` sorts NULLs LAST,
    so ranking them alongside real values bunched every missing row at the TOP
    of the day: on 2026-08-04 the 47 names with no `peer_corr` all scored 0.836
    while the 234 real values spanned 0.000-0.832, making "no peer data"
    indistinguishable from "more correlated than 83% of the market". That held
    on every partial-coverage day of every affected feature — 48 of 59.

    Splitting the partition on nullity fixes two things at once:

      1. The missing rows rank among themselves and the outer CASE discards
         that, so they leave as NULL and LightGBM routes them natively.
      2. The real values regain the full 0..1 range. Ranking against a
         denominator that counted the NULLs compressed them into
         0..n_real/n_total — which is why 2026-08-04's real values stopped at
         0.832, not 1.0. That ceiling moved with each day's coverage, so the
         rank was not even comparable across dates, which is the one thing
         ranking exists to guarantee.

    Do not "fix" this by imputing 0.5 instead. That reintroduces the same
    defect in the middle of the distribution, and missingness is not neutral
    here: missing rows underperform slightly (mean target rank 0.495 vs 0.503).
    """
    # The PARTITION includes in_universe so the two groups rank separately, and
    # the outer CASE keeps only the universe values.
    return (
        f"CASE WHEN in_universe AND {col} IS NOT NULL THEN PERCENT_RANK() OVER "
        f"(PARTITION BY date, in_universe, ({col} IS NULL) ORDER BY {col}) END"
    )


def zscore_expr(col: str) -> str:
    """Standardised value within the day's universe."""
    return (
        f"CASE WHEN in_universe THEN ({col} - AVG({col}) OVER (PARTITION BY date, in_universe)) "
        f"/ NULLIF(STDDEV_SAMP({col}) OVER (PARTITION BY date, in_universe), 0) END"
    )
