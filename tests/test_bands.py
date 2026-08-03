import duckdb
import pytest

from vnresearch.clean import bands, bars


@pytest.mark.parametrize(
    "price,exchange,expected",
    [
        (9_990, "HOSE", 10),
        (10_000, "HOSE", 50),
        (49_950, "HOSE", 50),
        (50_000, "HOSE", 100),
        (12_000, "HNX", 100),
        (12_000, "UPCOM", 100),
        (12_000, None, 100),
    ],
)
def test_tick_size(price, exchange, expected):
    assert bands.tick_size(price, exchange) == expected


def test_band_widths():
    assert bands.band("HOSE") == 0.07
    assert bands.band("HNX") == 0.10
    assert bands.band("UPCOM") == 0.15
    # An unknown venue gets the widest band, so we under-claim limits rather
    # than label an ordinary move as limit-locked.
    assert bands.band(None) == bands.DEFAULT_BAND == 0.15


def test_ceiling_and_floor_are_tick_aligned():
    # FPT on HOSE: ref 62,200 -> ceiling 66,500, floor 57,900 (verified live).
    assert bands.ceiling_price(62_200, "HOSE") == 66_500
    assert bands.floor_price(62_200, "HOSE") == 57_900


def test_ceiling_always_moves_at_least_one_tick():
    # On a very cheap stock the band rounds to nothing; the ceiling must still
    # be above the reference or the price could never legally move.
    ref = 1_000
    assert bands.ceiling_price(ref, "HNX") > ref
    assert bands.floor_price(ref, "HNX") < ref


def test_tolerance_is_clamped():
    # Cheap stock: one tick is a huge fraction of price, so the cap binds.
    assert bands.tolerance(1_000, "HNX") == bands.MAX_TOLERANCE
    # Expensive stock: one tick is negligible, so the floor binds.
    assert bands.tolerance(200_000, "HOSE") == bands.MIN_TOLERANCE


def test_sql_tick_matches_python():
    """bars.py builds SQL from these constants; the two must not drift apart."""
    con = duckdb.connect()
    cases = [
        (5_000, "HOSE"),
        (9_999, "HOSE"),
        (10_000, "HOSE"),
        (49_999, "HOSE"),
        (50_000, "HOSE"),
        (120_000, "HOSE"),
        (5_000, "HNX"),
        (80_000, "UPCOM"),
    ]
    for price, exchange in cases:
        sql = bars._TICK_SQL.replace("lagged.prev_close", str(price)).replace(
            "s.exchange", f"'{exchange}'"
        )
        got = con.execute(f"SELECT {sql}").fetchone()[0]
        assert got == bands.tick_size(price, exchange), f"{price} {exchange}"
    con.close()


def test_sql_band_matches_python():
    """Every exchange, on both sides of the 2013 reform."""
    con = duckdb.connect()
    for exchange in ["HOSE", "HNX", "UPCOM", "SOMETHING_ELSE"]:
        for date in ["2010-06-01", "2020-06-01"]:
            sql = bars._BAND_SQL.replace("s.exchange", f"'{exchange}'").replace(
                "lagged.date", f"DATE '{date}'"
            )
            got = con.execute(f"SELECT {sql}").fetchone()[0]
            assert got == pytest.approx(bands.band(exchange, date)), f"{exchange} {date}"
    con.close()


def test_bands_widened_in_2013():
    """Limits are not constant: on 2013-01-15 every band widened.

    Verified from the data before being encoded — positive HOSE returns pile up
    in the 4.5-5.0% bucket through 2012 and in 6.5-7.0% from 2013, and HNX hands
    over from ~7% to ~10% in the same month. Applying today's 7% to a 2010 bar
    reclassifies 66,245 genuinely limit-locked sessions as ordinary ones, which
    the backtest would then fill.
    """
    assert bands.band("HOSE", "2010-06-01") == 0.05
    assert bands.band("HOSE", "2020-06-01") == 0.07
    assert bands.band("HNX", "2010-06-01") == 0.07
    assert bands.band("HNX", "2020-06-01") == 0.10
    assert bands.band("UPCOM", "2010-06-01") == 0.10
    assert bands.band("UPCOM", "2020-06-01") == 0.15


def test_band_reform_boundary_is_exact():
    assert bands.band("HOSE", "2013-01-14") == 0.05
    assert bands.band("HOSE", "2013-01-15") == 0.07


def test_band_without_a_date_is_the_current_regime():
    """Callers asking about today should not have to know the history."""
    assert bands.band("HOSE") == 0.07
    assert bands.band("HNX") == 0.10


def test_sql_band_is_date_aware():
    """bars.py generates its own SQL; it must agree with band() on both eras."""
    con = duckdb.connect()
    for date, expected in [("2010-06-01", 0.05), ("2020-06-01", 0.07)]:
        sql = bars._BAND_SQL.replace("s.exchange", "'HOSE'").replace(
            "lagged.date", f"DATE '{date}'"
        )
        got = con.execute(f"SELECT {sql}").fetchone()[0]
        assert got == pytest.approx(expected), f"{date}: got {got}, want {expected}"
        assert got == pytest.approx(bands.band("HOSE", date))
    con.close()
