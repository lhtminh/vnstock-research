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


def rank_expr(col: str) -> str:
    """Percentile rank of col within the day's universe, 0..1."""
    # The PARTITION includes in_universe so the two groups rank separately, and
    # the outer CASE keeps only the universe values.
    return (
        f"CASE WHEN in_universe THEN PERCENT_RANK() OVER "
        f"(PARTITION BY date, in_universe ORDER BY {col}) END"
    )


def zscore_expr(col: str) -> str:
    """Standardised value within the day's universe."""
    return (
        f"CASE WHEN in_universe THEN ({col} - AVG({col}) OVER (PARTITION BY date, in_universe)) "
        f"/ NULLIF(STDDEV_SAMP({col}) OVER (PARTITION BY date, in_universe), 0) END"
    )
