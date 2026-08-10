"""Speculation labelling, from the mentor's document.

"Phương Pháp Nhận Diện Dấu Hiệu Đầu Cơ Cổ Phiếu" — four components scored 0-3
and combined:

    Score = 0.40·PVDI + 0.30·Turnover + 0.15·Volatility + 0.15·Range

WHAT KIND OF LABEL THIS IS, because it is not the other one. `label/forward.py`
produces a TARGET: what a stock did next, which is what the alpha model predicts.
This produces a DESCRIPTION: how the stock is behaving now, from trailing data
only. Nothing here looks forward, so it cannot be a prediction target — it is a
feature, a filter, or a report. Training on it would teach a model to recognise
speculation it can already see, not to anticipate returns.

WHERE IT DEPARTS FROM THE DOCUMENT, all three forced rather than chosen:

1. TURNOVER'S DENOMINATOR. The document divides volume by free float. There is
   no free-float history in this database — 51 tickers on one date in
   company_snapshot, 1,620 tickers on 4 recent sessions in market_snapshot. A
   2026 share count applied to a 2010 bar would be wrong and forward-looking
   both. The denominator is the stock's own trailing average volume instead,
   which keeps the measure dimensionless and cross-sectionally comparable —
   the property the percentile step depends on.

2. THE ρ_observed WINDOW is given only as "giai đoạn ngắn". 21 sessions.

3. THE DAILY/WEEKLY BLEND weights are "optimised on the last 12 months" against
   no stated objective, and there is no ground-truth speculation label to
   optimise against. Daily only by default.

Two more places the document contradicts itself; both are config, both default
to the formula as literally written, and `report()` measures what the choice
costs — see `config/speculation.yaml`.

WHY THIS IS NOT IN THE FEATURE REGISTRY. Registry features are single SQL
expressions over a bounded window. Two steps here are neither: σ_correlation is
the standard deviation OF a rolling correlation, and the percentile thresholds
are pooled across every stock and the trailing year at once. Both need more than
one pass, like `features/peers.py`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from vnresearch import config

# Score for each label, and the order they are indexed in.
POINTS = {"binh_thuong": 0, "dau_co_nhe": 1, "dau_co": 2, "dau_co_manh": 3}


def _cfg() -> dict:
    return config.load("speculation")


def _bucket(value: str, thresholds: list[float], labels: list[str]) -> str:
    """CASE that maps a value to a label by ascending thresholds.

    Written with explicit NULL handling rather than nested LEAST/GREATEST: in
    DuckDB `GREATEST(NULL, 0)` is 0, so a missing input would silently be
    labelled Bình thường — a stock with no data reading as one with no problem.
    """
    lo, mid, hi = thresholds
    return (
        f"CASE WHEN {value} IS NULL THEN NULL "
        f"WHEN {value} < {lo} THEN '{labels[0]}' "
        f"WHEN {value} < {mid} THEN '{labels[1]}' "
        f"WHEN {value} < {hi} THEN '{labels[2]}' "
        f"ELSE '{labels[3]}' END"
    )


def _points(label_col: str) -> str:
    whens = " ".join(f"WHEN '{k}' THEN {v}" for k, v in POINTS.items())
    return f"CASE {label_col} {whens} END"


# Which config weight scores which label column.
_COMPONENTS = (
    ("pvdi", "pvdi"),
    ("turnover", "turnover"),
    ("volatility", "vol"),
    ("range", "range"),
)


# Enough to erase binary-representation noise, far more than any weighting
# scheme resolves. See weighted_score.
_SCORE_DP = 6


def weighted_score(weights: dict[str, float]) -> str:
    """Score = 0.40·PVDI + 0.30·Turnover + 0.15·Volatility + 0.15·Range.

    BOTH the cast and the rounding are load-bearing, and each fixes what the
    other causes.

    DuckDB reads a bare `0.40` as DECIMAL, so the sum came out DECIMAL(16,2):
    exact for these weights, but two decimal places is a silent trap for any
    other, and DECIMAL lands in pandas as Decimal objects — the hazard
    `io/duck.py` casts away on the way out of Postgres.

    Casting to DOUBLE instead put float error on the thresholds. Of the 256
    reachable label combinations, 100 differ in binary representation and THREE
    land on 1.5 as 1.4999999999999998, taking 5,633 rows from Đầu cơ down to
    Đầu cơ nhẹ purely by rounding. Rounding first restores all 256 exactly.

    NULL propagates through the sum on purpose: a composite from three of four
    components is not this score, and scoring a missing component as zero would
    read as "normal on that dimension".
    """
    terms = " + ".join(f"{weights[k]}::DOUBLE * {_points(f'{c}_label')}" for k, c in _COMPONENTS)
    return f"ROUND({terms}, {_SCORE_DP})"


def _pooled_quantiles(con, source: str, value: str, window: int, out: str) -> None:
    """Q75/Q90/Q95 of `value` pooled over EVERY stock and the trailing `window`
    sessions, one row per date.

    The document is specific about this: "tính phân vị của Volatility5-day,i,s
    cho tất cả cổ phiếu s và tất cả ngày i từ t-248 đến t". It is a distribution
    over the whole market-year, not over one day's cross-section, so a name is
    called speculative relative to how the market has behaved for a year rather
    than relative to today alone.

    The range join is what costs the time — roughly 250 x the panel. Grouping by
    date first would be cheaper and would answer a different question.
    """
    qs = _cfg()["quantiles"]
    con.execute(
        f"""CREATE OR REPLACE TABLE {out} AS
            SELECT d.sess, d.date,
                   quantile_cont(a.{value}, {qs[0]}) AS q_lo,
                   quantile_cont(a.{value}, {qs[1]}) AS q_mid,
                   quantile_cont(a.{value}, {qs[2]}) AS q_hi
            FROM (SELECT DISTINCT sess, date FROM {source}) d
            JOIN {source} a
              ON a.sess BETWEEN d.sess - {window - 1} AND d.sess
             AND a.{value} IS NOT NULL
            GROUP BY 1, 2"""
    )


def _base_sql(cfg: dict) -> str:
    """The panel, filtered to the market the document means, with a session index.

    TWO counters, and they answer different questions. `sess` is a dense rank
    over distinct DATES — the pooled percentile window is "the last 252 sessions
    of the market", so it must be the same window for every name. `tsess` counts
    a ticker's own bars, and is what says whether a 252-session measure has 252
    sessions behind it.

    `tsess` is not optional. A frame written ROWS BETWEEN 251 PRECEDING AND
    CURRENT ROW does not require 252 rows — it computes over whatever exists, so
    a stock's third session gets a "12-month correlation" from three points and
    a number that looks exactly like a real one. `features/build.py` NULLs its
    warm-up for the same reason; without it here, PVDI was non-NULL on 99.85% of
    rows when a 252-session measure cannot be.
    """
    u = cfg["universe"]
    panel = config.path("data/clean/panel.parquet").as_posix()
    return f"""
CREATE OR REPLACE TABLE base AS
SELECT ticker, date, exchange, close, prev_close, high, low, volume, ret,
       DENSE_RANK() OVER (ORDER BY date) AS sess,
       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date) AS tsess
FROM read_parquet('{panel}')
WHERE bar_status = '{u["bar_status"]}' AND volume >= {u["min_volume"]}"""


def finite(col: str) -> str:
    """NaN and infinity to NULL.

    NOT decoration. `CORR` returns NaN — not NULL — when a window has no
    variance, which happens whenever a stock trades the same volume or does not
    move for the whole window: 2,874 rows on this panel. NaN is not NULL, so it
    survives every guard written for missing data, and feeding it to the
    STDDEV_SAMP below overflows the accumulator outright rather than producing a
    wrong number. It is invariant 3 with a different disguise — an undefined
    result must be absent, not a value.

    Applied to a materialised column, never to an expression, so the window
    function underneath is not computed twice.
    """
    return f"CASE WHEN isnan({col}) OR isinf({col}) THEN NULL ELSE {col} END"


def _pvdi_sql(cfg: dict) -> str:
    """PVDI = (ρ_expected − ρ_observed) / σ_correlation.

    Several passes, because each needs the one before it materialised: the
    rolling correlations, then the standard deviation OF the short correlation,
    then the ratio. SQL will not nest window functions, and σ here is exactly
    that — a window over a window.

    Sign convention is the document's. ρ_expected − ρ_observed is POSITIVE when
    the observed price-volume relationship is weaker than the stock's own norm,
    which is the direction the theory calls divergence.
    """
    p = cfg["pvdi"]
    x = p["price_term"]
    w = "PARTITION BY ticker ORDER BY date ROWS BETWEEN"
    return f"""
CREATE OR REPLACE TABLE pvdi AS
WITH c AS (
    SELECT ticker, date, sess, tsess,
           CORR({x}, volume) OVER ({w} {p["long"] - 1} PRECEDING AND CURRENT ROW)  AS rho_long,
           CORR({x}, volume) OVER ({w} {p["short"] - 1} PRECEDING AND CURRENT ROW) AS rho_short,
           AVG({x})      OVER ({w} {p["weekly_sessions"] - 1} PRECEDING AND CURRENT ROW) AS x_wk,
           AVG(volume)   OVER ({w} {p["weekly_sessions"] - 1} PRECEDING AND CURRENT ROW) AS v_wk
    FROM base
),
g AS (
    SELECT ticker, date, sess, tsess, x_wk, v_wk,
           {finite("rho_long")} AS rho_long, {finite("rho_short")} AS rho_short
    FROM c
),
cw AS (
    SELECT *,
           CORR(x_wk, v_wk) OVER ({w} {p["long"] - 1} PRECEDING AND CURRENT ROW)  AS rho_long_wk,
           CORR(x_wk, v_wk) OVER ({w} {p["short"] - 1} PRECEDING AND CURRENT ROW) AS rho_short_wk
    FROM g
),
gw AS (
    SELECT * EXCLUDE (rho_long_wk, rho_short_wk),
           {finite("rho_long_wk")} AS rho_long_wk, {finite("rho_short_wk")} AS rho_short_wk
    FROM cw
),
s AS (
    SELECT *,
           STDDEV_SAMP(rho_short)    OVER ({w} {p["long"] - 1} PRECEDING AND CURRENT ROW) AS sigma,
           STDDEV_SAMP(rho_short_wk) OVER ({w} {p["long"] - 1} PRECEDING AND CURRENT ROW) AS sigma_wk
    FROM gw
)
SELECT ticker, date, sess, tsess, rho_long, rho_short, sigma,
       CASE WHEN tsess >= {p["long"]}
            THEN (rho_long - rho_short) / NULLIF(sigma, 0) END AS pvdi_daily,
       CASE WHEN tsess >= {p["long"] + p["weekly_sessions"]}
            THEN (rho_long_wk - rho_short_wk) / NULLIF(sigma_wk, 0) END AS pvdi_weekly
FROM s"""


def build(verbose: bool = True) -> Path:
    """Write data/clean/speculation.parquet."""
    cfg = _cfg()
    win = cfg["windows"]
    labels = cfg["labels"]
    panel = config.path("data/clean/panel.parquet")
    if not panel.exists():
        raise FileNotFoundError(f"{panel} missing — run `vnr panel` first")

    con = duckdb.connect()
    try:
        con.execute(_base_sql(cfg))
        con.execute(_pvdi_sql(cfg))

        # The three percentile-scored components. Each is a per-ticker measure
        # first, then scored against the pooled market-year distribution.
        vol_expr = "100 * ret" if cfg["volatility"]["measure"] == "signed" else "100 * ABS(ret)"
        rng_expr = (
            "high - low"
            if cfg["range"]["measure"] == "absolute"
            else "(high - low) / NULLIF(prev_close, 0)"
        )
        t = cfg["turnover"]
        if t["denominator"] == "free_float":
            raise ValueError(
                "turnover.denominator: free_float — there is no free-float history in "
                "this database (51 tickers on one date). See config/speculation.yaml."
            )
        w = "PARTITION BY ticker ORDER BY date ROWS BETWEEN"
        # Each measure NULLed until its own lookback is met — see _base_sql on
        # why the frame alone does not do this.
        con.execute(
            f"""CREATE OR REPLACE TABLE m AS
                SELECT ticker, date, sess, tsess,
                       CASE WHEN tsess >= {win["smooth"]} THEN
                         AVG({vol_expr}) OVER ({w} {win["smooth"] - 1} PRECEDING AND CURRENT ROW)
                       END AS vol_5d,
                       CASE WHEN tsess >= {win["smooth"]} THEN
                         AVG({rng_expr}) OVER ({w} {win["smooth"] - 1} PRECEDING AND CURRENT ROW)
                       END AS range_5d,
                       CASE WHEN tsess >= {t["slow"]} THEN
                         AVG(volume) OVER ({w} {t["fast"] - 1} PRECEDING AND CURRENT ROW)
                         / NULLIF(AVG(volume)
                             OVER ({w} {t["slow"] - 1} PRECEDING AND CURRENT ROW), 0)
                       END AS turnover_ratio
                FROM base"""
        )

        for col in ("vol_5d", "range_5d", "turnover_ratio"):
            if verbose:
                print(f"  pooled percentiles for {col} ...")
            _pooled_quantiles(con, "m", col, win["percentile"], f"q_{col}")

        # Daily labels: each measure against that date's pooled thresholds.
        joins, cols = [], []
        for col, name in (
            ("vol_5d", "vol"),
            ("range_5d", "range"),
            ("turnover_ratio", "turnover"),
        ):
            joins.append(f"LEFT JOIN q_{col} q{name} ON q{name}.date = m.date")
            cols.append(
                f"{_bucket(f'm.{col}', [f'q{name}.q_lo', f'q{name}.q_mid', f'q{name}.q_hi'], labels)}"
                f" AS {name}_label"
            )
        con.execute(
            f"""CREATE OR REPLACE TABLE daily AS
                SELECT m.ticker, m.date, m.sess, m.tsess,
                       m.vol_5d, m.range_5d, m.turnover_ratio,
                       {", ".join(cols)}
                FROM m {" ".join(joins)}"""
        )

        # "Tỷ lệ đầu cơ": share of the trailing year a name spent labelled Đầu cơ
        # or Đầu cơ mạnh, then ranked cross-sectionally. The document defines
        # this for volatility and range only.
        con.execute(
            f"""CREATE OR REPLACE TABLE ratios AS
                SELECT ticker, date, sess, tsess,
                       CASE WHEN tsess >= {win["ratio"]} THEN
                         AVG(CASE WHEN vol_label IN ('{labels[2]}', '{labels[3]}') THEN 1.0
                                  WHEN vol_label IS NULL THEN NULL ELSE 0.0 END)
                           OVER ({w} {win["ratio"] - 1} PRECEDING AND CURRENT ROW)
                       END AS vol_ratio,
                       CASE WHEN tsess >= {win["ratio"]} THEN
                         AVG(CASE WHEN range_label IN ('{labels[2]}', '{labels[3]}') THEN 1.0
                                  WHEN range_label IS NULL THEN NULL ELSE 0.0 END)
                           OVER ({w} {win["ratio"] - 1} PRECEDING AND CURRENT ROW)
                       END AS range_ratio
                FROM daily"""
        )
        for col in ("vol_ratio", "range_ratio"):
            _pooled_quantiles(con, "ratios", col, win["percentile"], f"q_{col}")

        cw = cfg["composite"]["weights"]
        pv = cfg["pvdi"]
        pvdi_expr = (
            f"({pv['w_daily']} * pvdi_daily + {pv['w_weekly']} * COALESCE(pvdi_weekly, 0))"
            if pv["w_weekly"]
            else "pvdi_daily"
        )
        score = weighted_score(cw)
        out = config.path("data/clean/speculation.parquet")
        con.execute(
            f"""COPY (
    SELECT d.ticker, d.date,
           p.rho_long, p.rho_short, p.sigma,
           {pvdi_expr} AS pvdi,
           d.turnover_ratio, d.vol_5d, d.range_5d,
           {_bucket(pvdi_expr, pv["thresholds"], labels)} AS pvdi_label,
           d.turnover_label, d.vol_label, d.range_label,
           r.vol_ratio, r.range_ratio,
           {_bucket("r.vol_ratio", ["qv.q_lo", "qv.q_mid", "qv.q_hi"], labels)} AS vol_label_12m,
           {_bucket("r.range_ratio", ["qr.q_lo", "qr.q_mid", "qr.q_hi"], labels)}
               AS range_label_12m,
           ({score}) AS spec_score,
           {_bucket(f"({score})", cfg["composite"]["thresholds"], labels)} AS spec_label
    FROM daily d
    JOIN pvdi p   ON p.ticker = d.ticker AND p.date = d.date
    LEFT JOIN ratios r ON r.ticker = d.ticker AND r.date = d.date
    LEFT JOIN q_vol_ratio   qv ON qv.date = d.date
    LEFT JOIN q_range_ratio qr ON qr.date = d.date
    ORDER BY d.ticker, d.date
) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)"""
        )
        if verbose:
            _summary(con, out.as_posix())
        return out
    finally:
        con.close()


def _summary(con, path: str) -> None:
    n, first, last = con.execute(
        f"SELECT count(*), min(date), max(date) FROM read_parquet('{path}')"
    ).fetchone()
    print(f"\n  {n:,} rows  {first} .. {last}")
    rows = con.execute(
        f"""SELECT spec_label, count(*) n, round(100.0 * count(*) / sum(count(*)) OVER (), 2) pct
            FROM read_parquet('{path}') WHERE spec_label IS NOT NULL
            GROUP BY 1 ORDER BY min(spec_score)"""
    ).fetchall()
    print("\n  composite label")
    for label, cnt, pct in rows:
        print(f"    {label:<14} {cnt:>9,}  {pct:>5.2f}%")
