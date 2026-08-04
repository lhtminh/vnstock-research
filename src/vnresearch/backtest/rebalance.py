"""Selecting what to hold, and — more importantly — what to keep holding.

The naive rule (hold exactly the top N, refresh every period) throws away a
position the moment it slips to rank N+1, then often buys it back weeks later.
Measured here: 73.7% of the book replaced every rebalance, costing 22.3% a year
against a gross edge worth about 19%.

A buffer fixes that. Enter at rank <= entry_rank, but only EXIT once a name
falls past exit_rank. Between the two ranks a holding is left alone. The signal
is unchanged; you simply stop paying to act on noise around the boundary.

    rank  1 .......... 30 .............. 60 ................
          |  buy zone   |  hold zone      |  sell zone
          entry_rank=30                   exit_rank=60

Vietnam-specific note: this is the main lever available here. Retail short
selling of individual stocks does not exist on HOSE/HNX — only VN30 futures —
so the bottom decile, where most of this model's edge lives, cannot be traded
directly. Reducing turnover on the long book is what is actually actionable.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd


def select(
    preds: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    entry_rank: int,
    exit_rank: int | None = None,
    initial_holdings: list[str] | None = None,
) -> dict[pd.Timestamp, list[str]]:
    """Holdings on each rebalance date, applying buffer hysteresis.

    exit_rank=None (or == entry_rank) reproduces plain top-N rebalancing.

    initial_holdings seeds the book. A backtest starts flat and leaves it None;
    live paper trading passes what it actually owns, because the buffer is a
    statement about POSITIONS — "keep what you hold until it falls past
    exit_rank" — and starting empty each session would re-derive the whole book
    from scratch and generate a full turnover every time.
    """
    exit_rank = exit_rank or entry_rank
    if exit_rank < entry_rank:
        raise ValueError("exit_rank must be >= entry_rank")

    by_date = {d: g for d, g in preds.groupby("date")}
    holdings: dict[pd.Timestamp, list[str]] = {}
    held: list[str] = list(initial_holdings or [])

    for d in rebal_dates:
        day = by_date.get(d)
        if day is None or day.empty:
            if held:
                holdings[d] = list(held)
            continue

        ranked = day.sort_values("pred", ascending=False)["ticker"].tolist()
        rank_of = {t: i + 1 for i, t in enumerate(ranked)}

        # Survivors first: anything still inside the sell threshold stays, in
        # signal order. This is the whole saving — no trade is generated for a
        # name that merely drifted from rank 28 to rank 41.
        keep = [t for t in ranked if t in set(held) and rank_of[t] <= exit_rank]

        # Then fill the remaining slots from the buy zone.
        for t in ranked[:entry_rank]:
            if len(keep) >= entry_rank:
                break
            if t not in keep:
                keep.append(t)

        held = keep[:entry_rank]
        holdings[d] = list(held)

    return holdings


def turnover(holdings: dict[pd.Timestamp, list[str]]) -> float:
    """Average fraction of the book replaced per rebalance."""
    dates = sorted(holdings)
    changes = []
    for prev, cur in pairwise(dates):
        a, b = set(holdings[prev]), set(holdings[cur])
        if b:
            changes.append(len(b - a) / len(b))
    return float(np.mean(changes)) if changes else float("nan")
