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


def weighted_score(weights: dict[str, float], suffix: str = "") -> str:
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
    terms = " + ".join(
        f"{weights[k]}::DOUBLE * {_points(f'{c}_label{"" if c == "turnover" else suffix}')}"
        for k, c in _COMPONENTS
    )
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


# The two readings of the volatility and range formulas, computed side by side.
# Which one a variant uses is config; both always exist in the file.
_VOL = {"signed": "100 * ret", "abs": "100 * ABS(ret)"}
_RANGE = {"absolute": "high - low", "pct": "(high - low) / NULLIF(prev_close, 0)"}

# Column suffix per variant. The mentor's carries no suffix, so every name the
# document uses means what the document means by it.
_SUFFIX = {"mentor": "", "adjusted": "_adj"}


def _measure_sql(cfg: dict) -> str:
    """Every per-ticker measure both variants need, in one pass.

    All four of vol/range are computed whichever variant is selected — they are
    cheap window averages over the same panel, and having both in the file is
    what turns "the formula is wrong" into a query someone can run.
    """
    win, t = cfg["windows"], cfg["turnover"]
    w = "PARTITION BY ticker ORDER BY date ROWS BETWEEN"
    smooth = f"{w} {win['smooth'] - 1} PRECEDING AND CURRENT ROW"
    # Each measure NULLed until its own lookback is met — see _base_sql on why
    # the frame alone does not do this.
    cols = [
        f"CASE WHEN tsess >= {win['smooth']} THEN AVG({expr}) OVER ({smooth}) END AS {name}"
        for name, expr in (
            ("vol_signed", _VOL["signed"]),
            ("vol_abs", _VOL["abs"]),
            ("range_absolute", _RANGE["absolute"]),
            ("range_pct", _RANGE["pct"]),
        )
    ]
    cols.append(
        f"""CASE WHEN tsess >= {t["slow"]} THEN
              AVG(volume) OVER ({w} {t["fast"] - 1} PRECEDING AND CURRENT ROW)
              / NULLIF(AVG(volume) OVER ({w} {t["slow"] - 1} PRECEDING AND CURRENT ROW), 0)
            END AS turnover_ratio"""
    )
    # The windows are computed in their own CTE, where only `base` is in scope.
    # Written as one join instead, `PARTITION BY ticker` is ambiguous — the pvdi
    # table carries ticker, date, sess and tsess as well.
    return f"""
CREATE OR REPLACE TABLE m AS
WITH mm AS (SELECT *, {", ".join(cols)} FROM base)
SELECT mm.*, p.pvdi
FROM mm JOIN (SELECT ticker, date, pvdi FROM pvdi) p
     ON p.ticker = mm.ticker AND p.date = mm.date"""


def _variant_columns(cfg: dict) -> tuple[list[str], list[str], set[str]]:
    """(daily label expressions, composite expressions, measures needing quantiles).

    Reading this is the whole design: one loop over the two variants, and the
    only thing that differs between them is which measure column each component
    reads and whether PVDI is scored on percentiles or the document's fixed
    thresholds.
    """
    labels = cfg["labels"]
    need = {"turnover_ratio"}
    label_cols, score_cols = [], []

    for variant, suffix in _SUFFIX.items():
        v = cfg["variants"][variant]
        vol_col, rng_col = f"vol_{v['volatility']}", f"range_{v['range']}"
        need |= {vol_col, rng_col}

        pairs = [("vol", vol_col), ("range", rng_col)]
        if suffix == "":
            # Turnover is identical in both, so it is labelled once, unsuffixed.
            pairs.append(("turnover", "turnover_ratio"))
        for name, col in pairs:
            q = f"q_{col}"
            label_cols.append(
                f"{_bucket(f'm.{col}', [f'{q}.q_lo', f'{q}.q_mid', f'{q}.q_hi'], labels)}"
                f" AS {name}_label{suffix}"
            )

        if v["pvdi_scoring"] == "percentile":
            need.add("pvdi")
            thresholds = ["q_pvdi.q_lo", "q_pvdi.q_mid", "q_pvdi.q_hi"]
        else:
            thresholds = cfg["pvdi"]["thresholds"]
        label_cols.append(f"{_bucket('m.pvdi', thresholds, labels)} AS pvdi_label{suffix}")

        score = weighted_score(cfg["composite"]["weights"], suffix)
        score_cols.append(f"({score}) AS spec_score{suffix}")
        score_cols.append(
            f"{_bucket(f'({score})', cfg['composite']['thresholds'], labels)} AS spec_label{suffix}"
        )
    return label_cols, score_cols, need


def build(verbose: bool = True) -> Path:
    """Write data/clean/speculation.parquet, with BOTH label sets."""
    cfg = _cfg()
    win, labels = cfg["windows"], cfg["labels"]
    panel = config.path("data/clean/panel.parquet")
    if not panel.exists():
        raise FileNotFoundError(f"{panel} missing — run `vnr panel` first")
    if cfg["turnover"]["denominator"] == "free_float":
        raise ValueError(
            "turnover.denominator: free_float — there is no free-float history in this "
            "database (51 tickers on one date). See config/speculation.yaml."
        )

    con = duckdb.connect()
    try:
        con.execute(_base_sql(cfg))
        con.execute(_pvdi_sql(cfg))
        # The document's PVDI_composite. Weight 0 on weekly by default, because
        # its "optimised" blend has no objective to optimise against.
        pv = cfg["pvdi"]
        if pv["w_weekly"]:
            con.execute(
                f"""CREATE OR REPLACE TABLE pvdi AS SELECT * EXCLUDE (pvdi_daily),
                    {pv["w_daily"]}::DOUBLE * pvdi_daily
                      + {pv["w_weekly"]}::DOUBLE * pvdi_weekly AS pvdi FROM pvdi"""
            )
        else:
            con.execute("CREATE OR REPLACE TABLE pvdi AS SELECT *, pvdi_daily AS pvdi FROM pvdi")

        con.execute(_measure_sql(cfg))
        label_cols, score_cols, need = _variant_columns(cfg)

        for col in sorted(need):
            if verbose:
                print(f"  pooled percentiles for {col} ...", flush=True)
            _pooled_quantiles(con, "m", col, win["percentile"], f"q_{col}")

        joins = " ".join(f"LEFT JOIN q_{c} ON q_{c}.date = m.date" for c in sorted(need))
        con.execute(
            f"""CREATE OR REPLACE TABLE daily AS
                SELECT m.*, {", ".join(label_cols)} FROM m {joins}"""
        )

        # "Tỷ lệ đầu cơ": the share of the trailing year a name spent labelled
        # Đầu cơ or Đầu cơ mạnh, then ranked across the market. The document
        # defines it for volatility and range; computed for both variants.
        w = "PARTITION BY ticker ORDER BY date ROWS BETWEEN"
        ratio_cols = [
            f"""CASE WHEN tsess >= {win["ratio"]} THEN
                  AVG(CASE WHEN {c}_label{s} IN ('{labels[2]}', '{labels[3]}') THEN 1.0
                           WHEN {c}_label{s} IS NULL THEN NULL ELSE 0.0 END)
                    OVER ({w} {win["ratio"] - 1} PRECEDING AND CURRENT ROW)
                END AS {c}_ratio{s}"""
            for s in _SUFFIX.values()
            for c in ("vol", "range")
        ]
        con.execute(
            f"""CREATE OR REPLACE TABLE ratios AS
                SELECT ticker, date, sess, tsess, {", ".join(ratio_cols)} FROM daily"""
        )
        ratio_names = [f"{c}_ratio{s}" for s in _SUFFIX.values() for c in ("vol", "range")]
        for col in ratio_names:
            if verbose:
                print(f"  pooled percentiles for {col} ...", flush=True)
            _pooled_quantiles(con, "ratios", col, win["percentile"], f"q_{col}")

        twelve = [
            f"{_bucket(f'r.{n}', [f'q_{n}.q_lo', f'q_{n}.q_mid', f'q_{n}.q_hi'], labels)}"
            f" AS {n.replace('_ratio', '_label_12m')}"
            for n in ratio_names
        ]
        rjoins = " ".join(f"LEFT JOIN q_{n} ON q_{n}.date = d.date" for n in ratio_names)

        out = config.path("data/clean/speculation.parquet")
        con.execute(
            f"""COPY (
    SELECT d.* EXCLUDE (sess, tsess),
           {", ".join(f"r.{n}" for n in ratio_names)},
           {", ".join(twelve)},
           {", ".join(score_cols)}
    FROM daily d
    LEFT JOIN ratios r ON r.ticker = d.ticker AND r.date = d.date
    {rjoins}
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
    print(f"\n  {n:,} rows  {first} .. {last}\n")
    print(f"  {'':<14} {'MENTOR':>22}   {'ADJUSTED':>22}")
    for label in _cfg()["labels"]:
        a, b = con.execute(
            f"""SELECT count(*) FILTER (WHERE spec_label = '{label}'),
                       count(*) FILTER (WHERE spec_label_adj = '{label}')
                FROM read_parquet('{path}')"""
        ).fetchone()
        ta, tb = con.execute(
            f"SELECT count(spec_score), count(spec_score_adj) FROM read_parquet('{path}')"
        ).fetchone()
        print(f"  {label:<14} {a:>12,} {100 * a / ta:>6.2f}%   {b:>12,} {100 * b / tb:>6.2f}%")

    moved = con.execute(
        f"""SELECT count(*) FROM read_parquet('{path}')
            WHERE spec_label IS NOT NULL AND spec_label_adj IS NOT NULL
              AND spec_label <> spec_label_adj"""
    ).fetchone()[0]
    print(f"\n  the two disagree on {moved:,} rows")
