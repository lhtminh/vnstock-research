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
from vnresearch.backtest import rebalance
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

    @property
    def performance(self) -> dict:
        return performance(self.equity, self.benchmark)

    @property
    def annual(self) -> pd.DataFrame:
        return annual_table(self.equity, self.benchmark)


def performance(equity: pd.Series, benchmark: pd.Series) -> dict:
    """Split the return into what the market gave and what is left.

    Total return on its own is not a result. A long-only book in a rising market
    earns beta times the index for free, and reporting the sum of that and any
    genuine edge as one number makes a tracker look like a strategy.
    """
    eq = equity.dropna()
    bm = benchmark.reindex(eq.index).ffill()
    r = pd.DataFrame({"s": eq.pct_change(), "b": bm.pct_change()}).dropna()
    if len(r) < 60:
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
    equity: pd.Series,
    max_adtv_frac: float,
    exit_rank: int | None = None,
) -> pd.DataFrame:
    """Equal-weight the top N on each rebalance date, held until the next one.

    `equity` is the portfolio value at each date, NOT the starting cash. The cap
    is a constraint on the position's VND size, so it has to be divided by the
    money actually being deployed. Using starting cash makes the cap loosen by
    exactly the factor the book has compounded — a 7x gain silently turns a
    10%-of-ADTV limit into 70%.
    """
    # NaN means "no instruction", which is what lets ffill carry a target
    # between rebalances. A rebalance date writes the COMPLETE vector, zeros
    # included, so names that dropped out are actually sold.
    w = pd.DataFrame(np.nan, index=dates, columns=tickers)
    rebal = dates[::every]
    chosen = rebalance.select(preds, rebal, top_n, exit_rank)

    for d in rebal:
        picks = [t for t in chosen.get(d, []) if t in w.columns]
        if not picks:
            continue
        weight = 1.0 / len(picks)

        # Capacity: never take more than a slice of the name's daily turnover.
        # Without this the backtest buys volume that did not exist.
        capital = float(equity.get(d, equity.iloc[0]))
        if d in adtv.index and capital > 0:
            cap = (max_adtv_frac * adtv.loc[d, picks] / capital).astype(float)
            sized = np.minimum(weight, cap.fillna(0.0).to_numpy())
        else:
            sized = np.full(len(picks), weight)

        row = pd.Series(0.0, index=w.columns)
        row[picks] = sized
        w.loc[d] = row

    # Carry the last full target forward; flat before the first rebalance.
    return w.ffill().fillna(0.0)


def run(
    oos: pd.DataFrame,
    allow_untradeable_fills: bool = False,
    top_n: int | None = None,
    verbose: bool = True,
    costs: Costs | None = None,
    rebalance_every: int | None = None,
    exit_rank: int | None = None,
) -> Result:
    """Backtest out-of-sample predictions (date, ticker, pred).

    `costs` and `rebalance_every` override the config so the same predictions
    can be run under different frictions. Running with zero costs is the way to
    separate "the signal is wrong" from "the wrapper is too expensive" — they
    look identical in a single equity curve.
    """
    cfg = config.load("backtest")
    pcfg = cfg["portfolio"]
    top_n = top_n or pcfg["top_n"]
    every = rebalance_every or pcfg["rebalance_every"]
    init_cash = float(pcfg["init_cash"])
    costs = costs if costs is not None else Costs.load()

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

    exit_rank = exit_rank if exit_rank is not None else pcfg.get("exit_rank")

    def _simulate(equity: pd.Series):
        w = _target_weights(
            preds,
            close.index,
            tickers,
            top_n,
            every,
            adtv,
            equity,
            pcfg["max_adtv_frac"],
            exit_rank,
        )
        # Signals are formed on a session's close, so they can only be acted on
        # at the NEXT session's open. Shifting the weights is what enforces that.
        w = w.shift(1).fillna(0.0)
        return w, vbt.Portfolio.from_orders(
            close=close,
            size=w,
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

    # Two passes, because the capacity cap depends on portfolio value and
    # portfolio value depends on the cap. Pass one prices the cap off starting
    # cash; pass two re-prices it off the equity curve that produced. One
    # iteration is enough — the cap only binds on thin names, so the feedback is
    # small and converges rather than oscillating.
    flat = pd.Series(init_cash, index=close.index)
    _, first = _simulate(flat)
    weights, pf = _simulate(first.value().reindex(close.index).ffill().fillna(init_cash))

    bench = _benchmark(close.index, cfg["benchmark"])
    stats = pf.stats()
    if verbose:
        _print(stats, bench, pf, costs, weights)
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


def _print(stats: pd.Series, bench: pd.Series, pf, costs: Costs, weights: pd.DataFrame) -> None:
    for k in ["Start", "End", "Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio"]:
        if k not in stats.index:
            continue
        v = stats[k]
        shown = f"{v:,.2f}" if isinstance(v, int | float | np.floating) else str(v)
        print(f"  {k:<22} {shown}")

    eq = pf.value()
    perf = performance(eq, bench)
    if perf:
        print()
        print("  --- decomposition (this is the result; total return is not) ---")
        print(f"  {'CAGR [%]':<22} {100 * perf['cagr']:8.2f}")
        print(f"  {'benchmark CAGR [%]':<22} {100 * perf['benchmark_cagr']:8.2f}")
        print(f"  {'beta':<22} {perf['beta']:8.2f}")
        print(f"  {'correlation':<22} {perf['correlation']:8.2f}")
        print(f"  {'market gave [%]':<22} {100 * perf['market_contribution']:8.2f}")
        print(f"  {'ALPHA [%]':<22} {100 * perf['alpha']:8.2f}  <- what is left")
        print()
        print("  --- annual ---")
        for yr, row in annual_table(eq, bench).iterrows():
            flag = "  <-- negative" if row["excess"] < 0 else ""
            print(
                f"   {yr}  strat {100 * row['strategy']:7.1f}%"
                f"   bench {100 * row['benchmark']:7.1f}%"
                f"   excess {100 * row['excess']:7.1f}%{flag}"
            )

    # Capped positions leave cash idle. Realistic, but it has to be visible: a
    # book that is only half deployed is not the strategy you think you ran.
    print()
    print(f"  {'avg deployed [%]':<22} {100 * weights.sum(axis=1).mean():8.1f}")
    print(f"  {'round-trip cost [%]':<22} {100 * costs.round_trip:8.2f}")
