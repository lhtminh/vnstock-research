"""Trading days.

trading_calendar only covers 2019-09-12 onward, but bars go back to 2002. For
earlier dates we take the days VNINDEX actually printed, falling back to the
days any ticker traded. Tet is a week-long gap every year — it must read as
"market closed", not "data missing".
"""

from __future__ import annotations

import pandas as pd

from vnresearch.io import duck

_SQL = """
SELECT date, true AS is_trading_day, 'calendar' AS src FROM trading_calendar WHERE is_trading_day
UNION
SELECT DISTINCT date, true, 'index' FROM index_series WHERE index_code = 'VNINDEX'
UNION
SELECT DISTINCT date, true, 'bars'  FROM daily_prices
ORDER BY 1
"""


def trading_days() -> pd.DatetimeIndex:
    """Every date the market was open, oldest first."""
    con = duck.open_mirror()
    try:
        df = con.execute("SELECT DISTINCT date FROM (" + _SQL + ") ORDER BY date").df()
    finally:
        con.close()
    return pd.DatetimeIndex(df["date"])


def session_gaps(min_days: int = 5) -> pd.DataFrame:
    """Gaps between consecutive trading days — mostly Tet, occasionally an outage."""
    days = trading_days()
    gap = days.to_series().diff().dt.days
    out = pd.DataFrame({"date": days, "gap_days": gap.values})
    return out[out["gap_days"] >= min_days].reset_index(drop=True)
