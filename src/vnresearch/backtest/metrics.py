"""Performance statistics on an equity curve.

Separate from `engine.py` because these are pure pandas and have no business
requiring vectorbt. The paper trader reports a live book with them daily, and
making that import a backtesting library — one that pins numpy and pandas hard
enough to be an optional extra here — was wrong.

`engine` re-exports both names, so nothing that already imported them from
there needs to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many return observations the split is noise. Roughly a quarter of
# trading sessions.
MIN_OBSERVATIONS = 60


def performance(equity: pd.Series, benchmark: pd.Series) -> dict:
    """Split the return into what the market gave and what is left.

    Total return on its own is not a result. A long-only book in a rising market
    earns beta times the index for free, and reporting the sum of that and any
    genuine edge as one number makes a tracker look like a strategy.
    """
    eq = equity.dropna()
    bm = benchmark.reindex(eq.index).ffill()
    r = pd.DataFrame({"s": eq.pct_change(), "b": bm.pct_change()}).dropna()
    if len(r) < MIN_OBSERVATIONS:
        return {}

    years = len(r) / 252
    cagr_s = (1 + r["s"]).prod() ** (1 / years) - 1
    cagr_b = (1 + r["b"]).prod() ** (1 / years) - 1
    beta = float(np.cov(r["s"], r["b"])[0, 1] / np.var(r["b"]))
    return {
        "cagr": float(cagr_s),
        "benchmark_cagr": float(cagr_b),
        "beta": beta,
        "market_contribution": float(beta * cagr_b),
        "alpha": float(cagr_s - beta * cagr_b),
        "correlation": float(r["s"].corr(r["b"])),
        "years": float(years),
    }


def annual_table(equity: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Year-by-year strategy vs benchmark. The table that ends arguments."""
    eq = equity.dropna()
    bm = benchmark.reindex(eq.index).ffill()
    r = pd.DataFrame({"strategy": eq.pct_change(), "benchmark": bm.pct_change()}).dropna()
    out = (1 + r).groupby(r.index.year).prod() - 1
    out["excess"] = out["strategy"] - out["benchmark"]
    return out
