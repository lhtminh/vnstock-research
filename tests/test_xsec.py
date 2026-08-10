"""Cross-sectional rank correctness.

The rank pass had no test, and a defect lived in it that the feature-level
tests could not see: they check raw expressions, and this happens one layer
above. A missing value was still given a rank, and because SQL sorts NULLs
LAST that rank was an extreme rather than an absence.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vnresearch.features import xsec


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rank_expr to a small frame of (date, in_universe, x)."""
    con = duckdb.connect()
    try:
        con.register("t", df)
        return con.execute(
            f"SELECT *, {xsec.rank_expr('x')} AS x_rank FROM t ORDER BY date, x"
        ).df()
    finally:
        con.close()


def _frame(values, date="2026-08-04", in_universe=True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date).date()] * len(values),
            "in_universe": [in_universe] * len(values),
            "x": values,
        }
    )


def test_missing_value_gets_no_rank():
    """The regression. A NULL raw value must leave as a NULL rank.

    Before the fix these rows were bunched at the top of the day: 47 names with
    no peer_corr all scored 0.836 while the real values stopped at 0.832.
    """
    out = _rank(_frame([0.1, 0.2, 0.3, None, None]))
    assert out[out["x"].isna()]["x_rank"].isna().all()
    assert out[out["x"].notna()]["x_rank"].notna().all()


def test_missing_values_are_not_given_an_extreme_rank():
    """Specifically: they must not land above every real value."""
    out = _rank(_frame([0.1, 0.2, 0.3, None, None]))
    real = out[out["x"].notna()]["x_rank"]
    missing = out[out["x"].isna()]["x_rank"]
    assert missing.isna().all(), (
        f"missing rows received ranks {missing.tolist()}, real values reached {real.max()}"
    )


def test_real_values_span_the_full_range_whatever_the_coverage():
    """Ranking against a denominator that counted NULLs compressed the scale.

    Three real values among five rows used to top out at 3/5, so the same
    ordering produced different numbers depending on how many names happened to
    be missing that day — and the rank stopped being comparable across dates.
    """
    for n_missing in (0, 2, 20):
        out = _rank(_frame([0.1, 0.2, 0.3] + [None] * n_missing))
        real = out[out["x"].notna()]["x_rank"]
        assert real.min() == pytest.approx(0.0)
        assert real.max() == pytest.approx(1.0), (
            f"with {n_missing} missing rows the top real value ranked {real.max()}"
        )


def test_coverage_does_not_move_the_ranks_of_present_names():
    """The same real values must rank identically however many NULLs join them."""
    a = _rank(_frame([0.1, 0.2, 0.3]))
    b = _rank(_frame([0.1, 0.2, 0.3, None, None, None]))
    pd.testing.assert_series_equal(
        a[a["x"].notna()]["x_rank"].reset_index(drop=True),
        b[b["x"].notna()]["x_rank"].reset_index(drop=True),
        check_names=False,
    )


def test_all_null_cross_section_is_null_not_zero():
    """A day with no data anywhere is an absence, not a market of identical names."""
    out = _rank(_frame([None, None, None]))
    assert out["x_rank"].isna().all()


def test_rows_outside_the_universe_get_no_rank():
    """Unchanged behaviour, kept under test because the CASE now carries two terms."""
    df = pd.concat([_frame([0.1, 0.2]), _frame([0.3, 0.4], in_universe=False)])
    out = _rank(df)
    assert out[~out["in_universe"]]["x_rank"].isna().all()
    assert out[out["in_universe"]]["x_rank"].notna().all()


def test_each_date_ranks_independently():
    """Ranks are within the day; one date's coverage must not reach another."""
    df = pd.concat(
        [
            _frame([0.1, 0.2, 0.3], date="2026-08-04"),
            _frame([5.0, None, 7.0], date="2026-08-05"),
        ]
    )
    out = _rank(df)
    for _, g in out.groupby("date"):
        real = g[g["x"].notna()]["x_rank"]
        assert real.min() == pytest.approx(0.0)
        assert real.max() == pytest.approx(1.0)
