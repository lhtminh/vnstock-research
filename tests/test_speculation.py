"""The mentor's speculation labelling.

The formulas came from a document, so the tests pin them to hand-computed values
rather than to whatever the code currently produces. A labelling scheme that
drifts from its specification is worse than one that is wrong in a known way.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from vnresearch import config
from vnresearch.label import speculation as sp

LABELS = ["binh_thuong", "dau_co_nhe", "dau_co", "dau_co_manh"]


def _label(con, value, thresholds=(1.0, 2.0, 3.5)):
    sql = sp._bucket("v", list(thresholds), LABELS)
    v = "NULL" if value is None else str(value)
    return con.execute(f"SELECT {sql} FROM (SELECT {v}::DOUBLE AS v)").fetchone()[0]


def test_the_pvdi_thresholds_are_the_documents():
    """< 1 bình thường, 1-2 nhẹ, 2-3.5 đầu cơ, >= 3.5 mạnh. Boundaries are
    closed on the left, per the document's tables."""
    con = duckdb.connect()
    assert _label(con, 0.99) == "binh_thuong"
    assert _label(con, 1.0) == "dau_co_nhe"
    assert _label(con, 1.99) == "dau_co_nhe"
    assert _label(con, 2.0) == "dau_co"
    assert _label(con, 3.49) == "dau_co"
    assert _label(con, 3.5) == "dau_co_manh"
    assert _label(con, 100.0) == "dau_co_manh"
    con.close()


def test_a_missing_value_is_unlabelled_not_normal():
    """The trap this whole repo keeps hitting. A stock with no data must not
    read as a stock with no problem — that is invariant 3, and it is why
    _bucket tests IS NULL first instead of leaning on comparison semantics."""
    con = duckdb.connect()
    assert _label(con, None) is None
    con.close()


def test_the_composite_thresholds_are_the_documents():
    con = duckdb.connect()
    t = config.load("speculation")["composite"]["thresholds"]
    assert t == [0.75, 1.5, 2.25]
    assert _label(con, 0.74, t) == "binh_thuong"
    assert _label(con, 0.75, t) == "dau_co_nhe"
    assert _label(con, 1.5, t) == "dau_co"
    assert _label(con, 2.25, t) == "dau_co_manh"
    con.close()


def test_the_weights_are_the_documents_and_sum_to_one():
    w = config.load("speculation")["composite"]["weights"]
    assert w == {"pvdi": 0.40, "turnover": 0.30, "volatility": 0.15, "range": 0.15}
    assert sum(w.values()) == pytest.approx(1.0)


def test_the_composite_arithmetic_matches_a_hand_computation():
    """PVDI đầu cơ (2), turnover đầu cơ mạnh (3), volatility nhẹ (1), range
    bình thường (0):  0.4*2 + 0.3*3 + 0.15*1 + 0.15*0 = 1.85  ->  đầu cơ."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS SELECT 'dau_co' AS pvdi_label, 'dau_co_manh' AS turnover_label, "
        "'dau_co_nhe' AS vol_label, 'binh_thuong' AS range_label"
    )
    score = sp.weighted_score(config.load("speculation")["composite"]["weights"])
    got = con.execute(f"SELECT {score} FROM t").fetchone()[0]
    assert isinstance(got, float), "DECIMAL arithmetic rounds to 2dp — see weighted_score"
    assert got == pytest.approx(1.85)

    label = con.execute(
        f"SELECT {sp._bucket(f'({score})', [0.75, 1.5, 2.25], LABELS)} FROM t"
    ).fetchone()[0]
    assert label == "dau_co"
    con.close()


def test_no_reachable_score_is_decided_by_float_error():
    """Every combination of four labels, against exact decimal arithmetic.

    Three of the 256 land on 1.5 exactly, and in raw DOUBLE they compute as
    1.4999999999999998 — which moved 5,633 real rows from Đầu cơ to Đầu cơ nhẹ
    on nothing but binary representation. A threshold is only meaningful if the
    value reaching it is the value that was computed."""
    con = duckdb.connect()
    vals = ", ".join(f"('{x}')" for x in LABELS)
    con.execute(
        f"""CREATE TABLE g AS SELECT a.p pvdi_label, b.p turnover_label,
                                    x.p vol_label, y.p range_label
            FROM (VALUES {vals}) a(p), (VALUES {vals}) b(p),
                 (VALUES {vals}) x(p), (VALUES {vals}) y(p)"""
    )
    w = config.load("speculation")["composite"]["weights"]
    exact = " + ".join(f"{w[k]} * {sp._points(f'{c}_label')}" for k, c in sp._COMPONENTS)
    got = sp.weighted_score(w)

    assert con.execute("SELECT count(*) FROM g").fetchone()[0] == 256
    assert (
        con.execute(f"SELECT count(*) FROM g WHERE {got} <> CAST(({exact}) AS DOUBLE)").fetchone()[
            0
        ]
        == 0
    )
    con.close()


def test_points_are_zero_to_three():
    assert sp.POINTS == {"binh_thuong": 0, "dau_co_nhe": 1, "dau_co": 2, "dau_co_manh": 3}


def test_one_missing_component_leaves_the_score_absent():
    """A composite from three of four components is not the document's score.
    NULL must propagate rather than be treated as zero points, which would read
    as 'normal on that dimension' and drag every score down."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t AS SELECT NULL::VARCHAR AS pvdi_label, 'dau_co_manh' AS turnover_label, "
        "'dau_co_manh' AS vol_label, 'dau_co_manh' AS range_label"
    )
    score = sp.weighted_score(config.load("speculation")["composite"]["weights"])
    assert con.execute(f"SELECT {score} FROM t").fetchone()[0] is None
    con.close()


def test_nan_from_a_flat_window_becomes_null():
    """CORR returns NaN — not NULL — when a window has no variance, which is
    every stock that traded the same volume or did not move for the whole
    window. 2,874 rows on the real panel. NaN passes every IS NULL guard and
    overflows the STDDEV that consumes it, so it is converted at the source."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1.0), ('NaN'::DOUBLE), (2.0)) v(x)")
    got = con.execute(f"SELECT {sp.finite('x')} AS x FROM t ORDER BY x NULLS LAST").fetchall()
    assert [r[0] for r in got] == [1.0, 2.0, None]

    # And the point of it. Unguarded, this does not return a wrong number — it
    # raises, which is how the bug was found.
    assert con.execute(f"SELECT stddev_samp({sp.finite('x')}) FROM t").fetchone()[0] is not None
    with pytest.raises(duckdb.OutOfRangeException):
        con.execute("SELECT stddev_samp(x) FROM t").fetchone()
    con.close()


def test_infinity_is_caught_too():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 'Infinity'::DOUBLE AS x")
    assert con.execute(f"SELECT {sp.finite('x')} FROM t").fetchone()[0] is None
    con.close()


def _panel(tmp_path, n_days=300, n_tickers=4, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days).date
    rows = []
    for t in range(n_tickers):
        px = 20_000.0
        for d in dates:
            r = rng.normal(0, 0.02)
            prev = px
            px = max(px * (1 + r), 1_000.0)
            rows.append(
                {
                    "ticker": f"T{t}",
                    "date": d,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "prev_close": prev,
                    "volume": int(rng.integers(10_000, 100_000)),
                    "exchange": "HOSE",
                    "bar_status": "normal",
                    "ret": px / prev - 1,
                }
            )
    path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_the_adjusted_variant_changes_only_what_the_evidence_supports():
    """Three changes were proposed from reading the formulas. Tested against
    forward returns, only the volatility one held up — the PVDI change hurt in
    8 of 8 pairwise comparisons, and the range change was inconsistent. Both
    were reverted, so the adjusted variant differs in exactly one setting.

    Pinned because reverting a change on evidence is easy to undo by accident
    later, when only the argument for it is remembered and not the measurement.
    """
    cfg = config.load("speculation")
    m, a = cfg["variants"]["mentor"], cfg["variants"]["adjusted"]

    assert set(m) == set(a) == {"volatility", "range", "pvdi_scoring"}
    assert m == {"volatility": "signed", "range": "absolute", "pvdi_scoring": "fixed"}
    assert a == {"volatility": "abs", "range": "absolute", "pvdi_scoring": "fixed"}
    assert [k for k in m if m[k] != a[k]] == ["volatility"]

    # The mentor's variant carries no suffix, so every column the document names
    # means what the document means by it.
    assert sp._SUFFIX["mentor"] == ""
    assert sp._SUFFIX["adjusted"] == "_adj"


def test_turnover_is_scored_once_because_both_variants_share_it():
    """The free-float substitution is forced on both, so a second copy under a
    different name would imply a difference that does not exist."""
    labels, _scores, _need = sp._variant_columns(config.load("speculation"))
    assert sum("AS turnover_label" in c for c in labels) == 1
    assert not any("turnover_label_adj" in c for c in labels)
    assert "turnover_label" in sp.weighted_score(
        config.load("speculation")["composite"]["weights"], "_adj"
    )


def test_a_measure_is_absent_until_its_history_exists(tmp_path, monkeypatch):
    """ROWS BETWEEN 251 PRECEDING does NOT require 252 rows — it computes over
    whatever is there, so a stock's third session would get a '12-month'
    correlation from three points. Observed: before the guard, PVDI was
    non-NULL on 99.85% of rows."""
    monkeypatch.setattr(config, "path", lambda rel: tmp_path / Path(rel).name)
    _panel(tmp_path)

    out = sp.build(verbose=False)
    df = pd.read_parquet(out)
    long = config.load("speculation")["pvdi"]["long"]

    per_ticker = df.sort_values(["ticker", "date"]).groupby("ticker").cumcount() + 1
    assert df.loc[per_ticker < long, "pvdi"].isna().all()
    assert df.loc[per_ticker >= long, "pvdi"].notna().any()
