"""Peer-cluster features: what did the stocks that move with this one just do?

THE PROBLEM THIS SOLVES. Vietnamese listings are heavily interlinked — Vingroup
alone has VIC, VHM, VRE and VEF — and the linkages are not confined to sector
codes. Measured on 2022-2026 daily returns:

    VHM-VIC  0.664
    VHM-VRE  0.627     the Vin cluster
    VIC-VRE  0.494
    ------------------
    FPT-HPG  0.416     unrelated large caps, i.e. plain market beta
    FPT-VHM  0.251

WHY CORRELATION AND NOT A PARENT-COMPANY TABLE. Ownership metadata would find
the Vin group and miss everything else — the supply-chain and sentiment linkages
between firms in different sectors, which is exactly the case that is hard to
encode by hand. Correlation finds whatever is actually there, including
relationships nobody wrote down.

RESIDUAL, NOT RAW. Every Vietnamese stock correlates with every other through
the market itself; FPT-HPG at 0.416 above is almost entirely that. Returns are
demeaned cross-sectionally each day, which removes the market move exactly and
leaves the co-movement that is specific to the pair.

POINT-IN-TIME. Peers are chosen on a rebalance date from the TRAILING window
and then held until the next rebalance, so a peer set is never informed by the
returns it is used to predict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vnresearch import config
from vnresearch.io import duck

# Trailing window for estimating co-movement. A year balances a stable estimate
# against a relationship that has since changed.
CORR_WINDOW = 252

# How often peer sets are re-estimated. Monthly: correlation structure moves far
# too slowly to justify daily re-estimation, and re-estimating rarely keeps the
# peer set stable enough that the resulting feature is not pure noise.
REBALANCE = 21

# Peers per ticker. Small enough that the group means something, large enough
# that one broken series cannot define it.
TOP_K = 10

# A pair needs this much residual correlation to count as a peer at all. Below
# it the "cluster" is just whatever survived the ranking.
MIN_CORR = 0.25


def _residual_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x ticker) returns with each day's cross-sectional mean removed."""
    wide = panel.pivot(index="date", columns="ticker", values="ret")
    # Subtracting the daily mean removes the market factor exactly, without
    # needing a beta estimate that would itself have to be point-in-time.
    return wide.sub(wide.mean(axis=1), axis=0)


def build(verbose: bool = True) -> Path:
    """Write data/features/peers.parquet."""
    panel_path = config.path("data/clean/panel.parquet")
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing — run `vnr panel` first")

    con = duck.open_mirror()
    try:
        df = con.execute(
            f"""SELECT ticker, date, ret, in_universe
                FROM read_parquet('{panel_path.as_posix()}')
                WHERE ret IS NOT NULL"""
        ).df()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    resid = _residual_returns(df)
    universe = df.pivot(index="date", columns="ticker", values="in_universe").fillna(False)
    dates = resid.index

    out = []
    # Start once a full estimation window exists; anything earlier would rank
    # peers on a handful of observations.
    for i in range(CORR_WINDOW, len(dates), REBALANCE):
        # Strictly BEFORE dates[i]: the peer set for a date is never informed
        # by that date's own returns.
        window = resid.iloc[i - CORR_WINDOW : i]
        names = universe.iloc[i][universe.iloc[i]].index
        names = [n for n in names if window[n].notna().sum() >= CORR_WINDOW // 2]
        if len(names) < 20:
            continue

        corr = window[names].corr(min_periods=CORR_WINDOW // 2)
        # A stock is not its own peer. .to_numpy(copy=False) can hand back a
        # read-only view, so mask through pandas instead of writing in place.
        corr = corr.mask(np.eye(len(corr), dtype=bool))

        # Dates this peer set governs: from asof to the next rebalance.
        end = min(i + REBALANCE, len(dates))
        span = dates[i:end]

        for t in names:
            s = corr[t].dropna()
            s = s[s >= MIN_CORR].nlargest(TOP_K)
            if s.empty:
                continue
            peers = list(s.index)
            # Equal-weighted rather than correlation-weighted: the correlations
            # are estimates, and weighting by them just concentrates the feature
            # on whichever pair happened to be noisiest.
            pr = resid.loc[span, peers].mean(axis=1)
            out.append(
                pd.DataFrame(
                    {
                        "ticker": t,
                        "date": span,
                        "peer_ret_1": pr.to_numpy(),
                        "peer_corr": float(s.mean()),
                        "peer_n": len(peers),
                    }
                )
            )

    if not out:
        raise ValueError("no peer sets built — check the universe and window settings")

    peers = pd.concat(out, ignore_index=True)
    # Trailing peer moves, computed per ticker AFTER assembly so they run across
    # rebalance boundaries rather than resetting at each one.
    peers = peers.sort_values(["ticker", "date"])
    g = peers.groupby("ticker")["peer_ret_1"]
    peers["peer_ret_5"] = g.transform(lambda s: s.rolling(5, min_periods=3).sum())
    peers["peer_ret_21"] = g.transform(lambda s: s.rolling(21, min_periods=10).sum())

    target = config.path("data/features") / "peers.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    peers.to_parquet(target, index=False)

    if verbose:
        print(f"  rows            {len(peers):>9,}")
        print(f"  tickers         {peers['ticker'].nunique():>9,}")
        print(f"  dates           {peers['date'].nunique():>9,}")
        print(f"  mean peer corr  {peers['peer_corr'].mean():>9.3f}")
        print(f"  mean peers/name {peers['peer_n'].mean():>9.1f}")
    return target
