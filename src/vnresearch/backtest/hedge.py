"""Beta hedge with VN30 index futures.

WHAT THIS DOES AND DOES NOT DO.

A long book earns `alpha + beta x market`. Shorting the index removes the
second term, so what is left is what the stock selection actually contributed.
The hedge does not CREATE edge — it isolates it. Applied to a book with no
alpha it returns roughly zero, minus the cost of hedging.

It also does not capture the long-short spread. This model is far better at
finding losers than winners (-1.077% vs +0.380% per 5 sessions on the holdout),
but shorting the INDEX is not shorting the bottom decile — the index is close to
the average. What a futures hedge harvests is `top decile - market`, the
smaller half.

WHY FUTURES AT ALL. Vietnam has no retail short selling of individual stocks on
HOSE or HNX. VN30F1M futures are the only shortable instrument available, which
makes this the only form of hedge that can actually be traded here.

APPROXIMATIONS, all of which flatter the result slightly:
  - The VN30 INDEX is used in place of the futures price. Futures carry a basis
    that moves independently, and in Vietnam VN30F1M often trades at a discount
    to spot. Ignoring it is the largest unmodelled risk here.
  - Monthly roll cost is a flat estimate, not the observed calendar spread.
  - Margin and its financing are ignored. Real futures margin ties up roughly
    15-20% of notional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# VN30 futures are far cheaper than stocks: fees are per contract on a notional
# of index x 100,000 VND, so a round trip runs a few basis points rather than
# the 60 bp a stock round trip costs.
FUTURES_COST_PER_SIDE = 0.0003
ROLLS_PER_YEAR = 12


@dataclass
class HedgeResult:
    gross: pd.Series  # unhedged daily returns of the long book
    hedged: pd.Series  # after the short futures overlay and its costs
    beta: pd.Series  # the point-in-time hedge ratio actually applied
    cost: pd.Series

    def summary(self, bench: pd.Series) -> dict:
        def ann(r):
            return (1 + r).prod() ** (252 / len(r)) - 1

        def vol(r):
            return r.std() * np.sqrt(252)

        b = bench.reindex(self.gross.index).dropna()
        return {
            "gross_cagr": float(ann(self.gross)),
            "hedged_cagr": float(ann(self.hedged)),
            "benchmark_cagr": float(ann(b)),
            "gross_vol": float(vol(self.gross)),
            "hedged_vol": float(vol(self.hedged)),
            "gross_sharpe": float(ann(self.gross) / vol(self.gross)),
            "hedged_sharpe": float(ann(self.hedged) / vol(self.hedged)),
            "residual_beta": float(np.cov(self.hedged, b)[0, 1] / np.var(b)),
            "mean_hedge_ratio": float(self.beta.mean()),
            "hedge_cost_pa": float(self.cost.mean() * 252),
        }


def apply(
    long_returns: pd.Series,
    hedge_returns: pd.Series,
    window: int = 60,
    cost_per_side: float = FUTURES_COST_PER_SIDE,
    rolls_per_year: int = ROLLS_PER_YEAR,
) -> HedgeResult:
    """Overlay a short futures position sized to the book's rolling beta.

    The beta is estimated on a trailing window and then SHIFTED, so the ratio
    applied on day t was knowable on day t-1. Using a full-sample beta here
    would be look-ahead of the most flattering kind: the hedge would be sized
    with knowledge of how the market actually behaved.
    """
    df = pd.DataFrame({"s": long_returns, "m": hedge_returns}).dropna()
    cov = df["s"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    beta = (cov / var).shift(1)

    # Before the window fills there is no estimate; run unhedged rather than
    # guessing a ratio.
    beta = beta.fillna(0.0).clip(0.0, 2.0)

    # Adjusting the hedge is a trade; so is rolling the contract each month.
    rebal_cost = beta.diff().abs().fillna(0.0) * cost_per_side
    roll_cost = beta.abs() * cost_per_side * 2 * rolls_per_year / 252
    cost = rebal_cost + roll_cost

    hedged = df["s"] - beta * df["m"] - cost
    return HedgeResult(gross=df["s"], hedged=hedged, beta=beta, cost=cost)
