"""Peer-cluster and seasonality features.

Both answer the same complaint about a pure price-and-volume model: it looks at
each stock in isolation and out of time. A Vietnamese stock is neither — VIC,
VHM, VRE and VPL move together, and the market empties out before Tet every
year on a date that drifts by three weeks.

The columns themselves are computed elsewhere (peers.py in pandas, panel.py in
SQL); this only registers them so alpha analysis and the model see them.
"""

from __future__ import annotations

from vnresearch.features.registry import register

# --- peers -----------------------------------------------------------------
_P = "peer"

# What this ticker's cluster did recently, market move already removed.
#
# The trade this expresses: when the names that genuinely move with a stock have
# run and it has not, does it catch up? That is the lead-lag question, and it is
# per-ticker — each stock has its own cluster — so it ranks across the
# cross-section rather than being another day-constant.
register("peer_ret_5", "peer_ret_5", _P, 1)
register("peer_ret_21", "peer_ret_21", _P, 1)

# The stock's own move against its cluster's. Positive means it has outrun the
# names it usually tracks, which is the mean-reversion side of the same idea.
register("peer_gap_5", "ret_5 - peer_ret_5", _P, 5)

# How tightly the cluster holds together. A name with strong peers is one whose
# peer signal is worth believing; a loose one is nearly independent, and this
# lets the model discount the features above accordingly rather than treating
# every cluster as equally informative.
register("peer_corr", "peer_corr", _P, 1)

# --- seasonality -----------------------------------------------------------
_S = "season"

# This ticker's own tendency in this calendar month, averaged over PRIOR years
# only (see panel.py — the window excludes the current year).
#
# Why not a month dummy: "it is February" is identical for every stock on the
# date and cannot order one above another. "THIS stock usually does well in
# February" varies across the universe, which is the whole difference.
register("seas_month", "seas_month", _S, 1)

# This ticker's own behaviour in the +/-14 day Tet window, from prior years,
# and only applied within three weeks of the holiday.
#
# A FIRST ATTEMPT AT THIS WAS WRONG and the mistake is worth keeping written
# down. It was `seas_month * 1/(1+|days_from_tet|)` — an "interaction" with
# distance to Tet. But days_from_tet is the same for every stock on a date, so
# multiplying by it is multiplying the whole cross-section by one positive
# constant, which leaves the ranking exactly as it was. The IC came back
# identical to seas_month to four decimal places, which is what exposed it.
#
# A day-constant cannot be rescued by multiplying it into something else. The
# stock-specific part has to be the SEASONAL ESTIMATE itself, which is what
# this is.
register("seas_tet", "seas_tet", _S, 1)
