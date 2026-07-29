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
    con = duckdb.connect()
    for exchange in ["HOSE", "HNX", "UPCOM", "SOMETHING_ELSE"]:
        sql = bars._BAND_SQL.replace("s.exchange", f"'{exchange}'")
        got = con.execute(f"SELECT {sql}").fetchone()[0]
        assert got == pytest.approx(bands.band(exchange)), exchange
    con.close()
