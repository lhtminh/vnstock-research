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

REPRODUCIBLE, which is a SEPARATE property from point-in-time and was missing.
Nothing here looked forward, and yet rebuilding after new data arrived rewrote
peer values across the whole history — measured, 73% of all rows moved between
two builds three days apart, back as far as 2007. Two causes, both fixed below
and both worth knowing because the same shapes appear elsewhere:

  1. The daily mean was taken over EVERY ticker in the panel. A ticker
     discovered today enters the historical cross-section, moves the mean on
     dates it happened to trade, and every residual on those dates moves with
     it. Now the mean is over the INVESTABLE cross-section only.

  2. The rebalance grid was `range(252, len(dates), 21)` — positions in an
     array. One extra date anywhere shifts every later rebalance, so a
     completely different set of peer sets governs a completely different set of
     spans. Now rebalance dates come from the CALENDAR, so they cannot move.

A backtest whose inputs change when unrelated data arrives cannot be
reproduced, cannot be diffed against a previous run, and cannot tell a code
change from a data change.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vnresearch import config

# Trailing window for estimating co-movement. A year balances a stable estimate
# against a relationship that has since changed.
CORR_WINDOW = 252

# How often peer sets are re-estimated. Monthly: correlation structure moves far
# too slowly to justify daily re-estimation, and re-estimating rarely keeps the
# peer set stable enough that the resulting feature is not pure noise.
#
# Realised as the first trading day of each calendar month rather than every
# 21st row, which is the same cadence and is anchored to something that cannot
# move. See the module docstring.
REBALANCE = 21

# Peers per ticker. Small enough that the group means something, large enough
# that one broken series cannot define it.
TOP_K = 10

# A pair needs this much residual correlation to count as a peer at all. Below
# it the "cluster" is just whatever survived the ranking.
MIN_CORR = 0.25


def _residual_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (date x ticker) returns demeaned daily, plus the universe mask.

    The mean is taken over the INVESTABLE names on each date, not over every
    ticker that happened to print a return. Two reasons, and the second is the
    one that was actually wrong:

    - It is the right market. This book holds liquid names equal-weighted, so
      the equal-weighted liquid cross-section is the factor to remove. Several
      hundred UPCOM listings trading a few million VND a day are not part of
      the market being traded.
    - It is stable. The universe on a past date is a function of that date's
      own trailing ADTV, so discovering a ticker today cannot move a 2007 mean
      unless that ticker was itself liquid in 2007 — in which case it genuinely
      belonged there. Under the old full-panel mean, ANY new listing moved
      every residual on every date it traded.
    """
    wide = panel.pivot(index="date", columns="ticker", values="ret")
    uni = panel.pivot(index="date", columns="ticker", values="in_universe").fillna(False)
    uni = uni.reindex(columns=wide.columns, fill_value=False)
    return wide.sub(wide.where(uni).mean(axis=1), axis=0), uni


def _rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first trading day of each calendar month.

    Derived from the dates themselves, so it does not depend on how many rows
    precede them. `range(CORR_WINDOW, len(dates), REBALANCE)` did, and one extra
    date shifted every rebalance after it.
    """
    return dates[~dates.to_period("M").duplicated()]


def build(verbose: bool = True) -> Path:
    """Write data/features/peers.parquet."""
    panel_path = config.path("data/clean/panel.parquet")
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing — run `vnr panel` first")

    # Plain connection, not open_mirror(): this query reads one Parquet file by
    # path and touches no mirrored table, so requiring a mirror was a false
    # dependency — and it is why this module had no test.
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""SELECT ticker, date, ret, in_universe
                FROM read_parquet('{panel_path.as_posix()}')
                WHERE ret IS NOT NULL"""
        ).df()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    resid, universe = _residual_returns(df)
    dates = pd.DatetimeIndex(resid.index)
    rebal = _rebalance_dates(dates)

    out = []
    for k, asof in enumerate(rebal):
        # Strictly BEFORE asof: the peer set for a date is never informed by
        # that date's own returns. Sliced by date rather than by position, so
        # the window is "the last CORR_WINDOW sessions" no matter how many rows
        # sit in front of it.
        window = resid.loc[resid.index < asof].tail(CORR_WINDOW)
        # Start once a full estimation window exists; anything earlier would
        # rank peers on a handful of observations.
        if len(window) < CORR_WINDOW:
            continue
        names = universe.loc[asof][universe.loc[asof]].index
        names = [n for n in names if window[n].notna().sum() >= CORR_WINDOW // 2]
        if len(names) < 20:
            continue

        corr = window[names].corr(min_periods=CORR_WINDOW // 2)
        # A stock is not its own peer. .to_numpy(copy=False) can hand back a
        # read-only view, so mask through pandas instead of writing in place.
        corr = corr.mask(np.eye(len(corr), dtype=bool))

        # Dates this peer set governs: from asof up to the next rebalance.
        nxt = rebal[k + 1] if k + 1 < len(rebal) else None
        span = dates[(dates >= asof) & (dates < nxt)] if nxt is not None else dates[dates >= asof]

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
