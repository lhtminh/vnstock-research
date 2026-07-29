"""Backtest model predictions with vectorbt.

Long the top N of the investable universe, equal weighted, rebalanced every
few sessions.

THE PART THAT MATTERS: execution prices are NaN on any bar that was not
tradeable. vectorbt skips an order whose price is NaN, which is exactly the
right behaviour — on a limit-up day there is no offer side, so the trade
simply does not happen and the position carries. Filling those bars is the
usual way a Vietnamese backtest manufactures returns, and it flatters exactly
the momentum names a signal most wants to buy.

`allow_untradeable_fills=True` turns that off deliberately, so you can measure
how much of the result the filter was holding back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from vnresearch import config
from vnresearch.backtest.costs import Costs
from vnresearch.io import duck


@dataclass
class Result:
    portfolio: vbt.Portfolio
    stats: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame

    @property
    def equity(self) -> pd.Series:
        return self.portfolio.value()


def _price_matrices(tickers: list[str], start, end, allow_untradeable: bool):
    """Wide (date x ticker) close, execution price, and adtv matrices."""
    con = duck.open_mirror()
    try:
        df = con.execute(
            f"""SELECT ticker, date, close, open, adtv, tradeable
                FROM read_parquet('{config.path("data/clean/panel.parquet").as_posix()}')
                WHERE ticker IN ({",".join(f"'{t}'" for t in tickers)})
                  AND date BETWEEN DATE '{start}' AND DATE '{end}'"""
        ).df()
    finally:
        con.close()

    close = df.pivot(index="date", columns="ticker", values="close")
    open_ = df.pivot(index="date", columns="ticker", values="open")
    adtv = df.pivot(index="date", columns="ticker", values="adtv")
    ok = df.pivot(index="date", columns="ticker", values="tradeable").fillna(False)

    # Execution at the open. NaN where untradeable, so no order can fill there.
    price = open_ if allow_untradeable else open_.where(ok.astype(bool))
    # Valuation may still use the close on an untradeable day — the position is
    # worth something even when it cannot be sold.
    return close.ffill(), price, adtv


def _target_weights(
    preds: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    top_n: int,
    every: int,
    adtv: pd.DataFrame,
    init_cash: float,
    max_adtv_frac: float,
) -> pd.DataFrame:
    """Equal-weight the top N on each rebalance date, held until the next one."""
    w = pd.DataFrame(0.0, index=dates, columns=tickers)
    rebal = dates[::every]

    for d in rebal:
        day = preds[preds["date"] == d]
        if day.empty:
            continue
        picks = day.nlargest(top_n, "pred")["ticker"].tolist()
        picks = [t for t in picks if t in w.columns]
        if not picks:
            continue
        weight = 1.0 / len(picks)

        # Capacity: never take more than a slice of the name's daily turnover.
        # Without this the backtest buys volume that did not exist.
        if d in adtv.index:
            cap = (max_adtv_frac * adtv.loc[d, picks] / init_cash).astype(float)
            sized = np.minimum(weight, cap.fillna(0.0).to_numpy())
        else:
            sized = np.full(len(picks), weight)

        w.loc[d, picks] = sized

    # Hold between rebalances; a 0 row would be read as "sell everything".
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


def run(
    oos: pd.DataFrame,
    allow_untradeable_fills: bool = False,
    top_n: int | None = None,
    verbose: bool = True,
) -> Result:
    """Backtest out-of-sample predictions (date, ticker, pred)."""
    cfg = config.load("backtest")
    pcfg = cfg["portfolio"]
    top_n = top_n or pcfg["top_n"]
    every = pcfg["rebalance_every"]
    init_cash = float(pcfg["init_cash"])
    costs = Costs.load()

    preds = oos.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    tickers = sorted(preds["ticker"].unique())
    start, end = preds["date"].min().date(), preds["date"].max().date()

    close, price, adtv = _price_matrices(tickers, start, end, allow_untradeable_fills)
    close.index = pd.to_datetime(close.index)
    price.index = pd.to_datetime(price.index)
    adtv.index = pd.to_datetime(adtv.index)
    tickers = [t for t in tickers if t in close.columns]
    close, price, adtv = close[tickers], price[tickers], adtv[tickers]

    weights = _target_weights(
        preds, close.index, tickers, top_n, every, adtv, init_cash, pcfg["max_adtv_frac"]
    )

    # Signals are formed on a session's close, so they can only be acted on at
    # the NEXT session's open. Shifting the weight matrix is what enforces that.
    weights = weights.shift(1).fillna(0.0)

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=weights,
        size_type="targetpercent",
        price=price,
        fees=costs.symmetric_fee,
        slippage=costs.slippage,
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",  # sell before buy, so cash is available to rebalance
        freq="1D",
    )

    bench = _benchmark(close.index, cfg["benchmark"])
    stats = pf.stats()
    if verbose:
        _print(stats, bench, pf, costs)
    return Result(portfolio=pf, stats=stats, benchmark=bench, weights=weights)


def _benchmark(index: pd.DatetimeIndex, code: str) -> pd.Series:
    con = duck.open_mirror()
    try:
        df = con.execute(
            f"SELECT date, close FROM index_series WHERE index_code = '{code}' ORDER BY date"
        ).df()
    finally:
        con.close()
    s = df.set_index(pd.to_datetime(df["date"]))["close"]
    return s.reindex(index).ffill()


def _cagr(first: float, last: float, years: float) -> float:
    if years <= 0 or first <= 0:
        return float("nan")
    return 100 * ((last / first) ** (1 / years) - 1)


def _print(stats: pd.Series, bench: pd.Series, pf, costs: Costs) -> None:
    keys = [
        "Start",
        "End",
        "Total Return [%]",
        "Max Drawdown [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Win Rate [%]",
        "Total Trades",
    ]
    for k in keys:
        if k not in stats.index:
            continue
        v = stats[k]
        shown = f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) else str(v)
        print(f"  {k:<22} {shown}")

    eq = pf.value().dropna()
    if len(eq) > 1:
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        print(f"  {'Strategy CAGR [%]':<22} {_cagr(eq.iloc[0], eq.iloc[-1], yrs):,.2f}")

    b = bench.dropna()
    if len(b) > 1:
        yrs = (b.index[-1] - b.index[0]).days / 365.25
        print(f"  {'VNINDEX Total [%]':<22} {100 * (b.iloc[-1] / b.iloc[0] - 1):,.2f}")
        print(f"  {'VNINDEX CAGR [%]':<22} {_cagr(b.iloc[0], b.iloc[-1], yrs):,.2f}")
    print(f"  {'round-trip cost':<22} {100 * costs.round_trip:.2f}%")
