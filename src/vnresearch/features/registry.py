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


@dataclass(frozen=True)
class Feature:
    name: str
    sql: str
    category: str
    lookback: int  # sessions of history needed before the value means anything


_REGISTRY: dict[str, Feature] = {}


def win(n: int) -> str:
    """A backward-only window frame of n sessions, ending at the current row."""
    return (
        f"OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN {n - 1} PRECEDING AND CURRENT ROW)"
    )


def lag(col: str, n: int) -> str:
    """Value of col n sessions ago, within the ticker."""
    return f"LAG({col}, {n}) OVER (PARTITION BY ticker ORDER BY date)"


def register(name: str, sql: str, category: str, lookback: int) -> Feature:
    if name in _REGISTRY:
        raise ValueError(f"duplicate feature: {name}")
    f = Feature(name=name, sql=" ".join(sql.split()), category=category, lookback=lookback)
    _REGISTRY[name] = f
    return f


def all_features() -> dict[str, Feature]:
    # Import for side effects: each module registers on import.
    from vnresearch.features import market, momentum, price, volume  # noqa: F401

    return dict(_REGISTRY)


def by_category(category: str) -> dict[str, Feature]:
    return {k: v for k, v in all_features().items() if v.category == category}
