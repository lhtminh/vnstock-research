"""The feature registry: one name -> one SQL expression.

Features are SQL rather than Python callables because they run over 4.8M rows
and DuckDB does that in seconds. Keeping them in a registry means alpha
analysis, model training and the backtest all read the same definition, so they
cannot drift apart.

THE ONE RULE: every window is `ROWS BETWEEN n PRECEDING AND CURRENT ROW`. No
feature may reference a following row. A single forward-looking frame produces
a backtest that looks excellent and cannot be traded. `win()` is the only way
to build a frame here, and it cannot express FOLLOWING.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The dimensions a portfolio manager actually asks about. These are the
# `category` strings, and they are also the emission order — build.py sorts by
# (category, name) and derived features resolve by lateral alias, so a feature
# may only reference one whose (category, name) sorts EARLIER. Renaming a
# category can therefore break a reference silently; the two that matter today
# are peer_gap_5 -> ret_5 (peer after momentum) and vol_ratio_21_63 -> vol_21
# (same dimension, and "vol_2" < "vol_r").
BETA = "beta"
LIQUIDITY = "liquidity"
MOMENTUM = "momentum"
PEER = "peer"
RANGE = "range"
SEASON = "season"
SPECULATION = "speculation"
TECHNICAL = "technical"
VOLATILITY = "volatility"

# Which way is "good" for the rating, per dimension. Used only by
# `rating/score.py`; nothing in training or the backtest reads it, because a
# tree does not need to be told which end of a feature it likes.
#
# DECLARED, not measured. A rating a portfolio manager cannot explain is not a
# rating, and a sign fitted to the sample flips between periods and takes the
# explanation with it. `vnr rating --check` reports where the data disagrees
# with these, which is a finding to discuss rather than something to silently
# invert.
_DIRECTION_BY_CATEGORY = {
    MOMENTUM: +1,  # winners keep winning, over the horizons here
    VOLATILITY: -1,  # paid for in drawdown, not returns
    RANGE: -1,  # wide intraday swings are instability, not opportunity
    LIQUIDITY: +1,  # you can actually get out
    BETA: -1,  # market sensitivity is beta, not alpha; invariant 8
    TECHNICAL: +1,  # oriented individually below where +1 is wrong
    SPECULATION: -1,  # the whole point of the mentor's document
    PEER: +1,
    SEASON: 0,  # a calendar effect is not a quality judgement
}


@dataclass(frozen=True)
class Feature:
    name: str
    sql: str
    category: str
    lookback: int  # sessions of history needed before the value means anything
    # +1 higher is better, -1 lower is better, 0 excluded from the rating.
    direction: int = 0


_REGISTRY: dict[str, Feature] = {}
_WINDOWS: dict[str, Any] | None = None


def windows() -> dict[str, Any]:
    """The `features:` block of config/features.yaml.

    Cached because every feature module reads it at import time and the modules
    are imported on each all_features() call.
    """
    global _WINDOWS
    if _WINDOWS is None:
        from vnresearch import config

        _WINDOWS = config.load("features").get("features", {})
    return _WINDOWS


def win(n: int) -> str:
    """A backward-only window frame of n sessions, ending at the current row."""
    return (
        f"OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW)"
    )


def lag(col: str, n: int) -> str:
    """Value of col n sessions ago, within the ticker."""
    return f"LAG({col}, {n}) OVER (PARTITION BY ticker ORDER BY date)"


def register(
    name: str, sql: str, category: str, lookback: int, direction: int | None = None
) -> Feature:
    """Register one feature. `direction` defaults to the dimension's own.

    Pass it explicitly only where the dimension default is wrong for this
    feature — an RSI is not "higher is better" the way a 6-month return is.
    """
    if name in _REGISTRY:
        raise ValueError(f"duplicate feature: {name}")
    if direction is None:
        direction = _DIRECTION_BY_CATEGORY.get(category, 0)
    if direction not in (-1, 0, 1):
        raise ValueError(f"{name}: direction must be -1, 0 or 1, got {direction!r}")
    f = Feature(
        name=name,
        sql=" ".join(sql.split()),
        category=category,
        lookback=lookback,
        direction=direction,
    )
    _REGISTRY[name] = f
    return f


def all_features() -> dict[str, Feature]:
    # Import for side effects: each module registers on import.
    from vnresearch.features import (  # noqa: F401
        cluster,
        market,
        momentum,
        price,
        shape,
        technical,
        volume,
    )

    return dict(_REGISTRY)


def by_category(category: str) -> dict[str, Feature]:
    return {k: v for k, v in all_features().items() if v.category == category}
