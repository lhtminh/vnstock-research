"""Mirror-manifest tests, mostly about the adjustment-epoch guard.

The guard has to fire on a genuinely mixed-basis series and stay silent on
ordinary maintenance. It got that backwards once: it tested whether the TABLE
held more than one epoch, which is true every time the service repairs a single
restatement, and refused to build anything until someone investigated a
non-problem.
"""

from __future__ import annotations

import json

import pytest

from vnresearch.io.manifest import Manifest, load_manifest, write_manifest


def _m(**kw) -> Manifest:
    base = {"row_counts": {"daily_prices": 4_777_868}, "adjustment_epochs": [1]}
    return Manifest(**{**base, **kw})


def test_several_epochs_across_the_table_is_fine():
    """The state that exposed the original bug: 12 tickers repaired onto epoch 2,
    1,675 still on epoch 1, and not one ticker spanning both."""
    _m(adjustment_epochs=[1, 2], seam_tickers=[]).check_single_epoch()


def test_a_ticker_spanning_two_epochs_raises():
    with pytest.raises(ValueError, match="ACG"):
        _m(adjustment_epochs=[1, 2], seam_tickers=["ACG"]).check_single_epoch()


def test_the_error_names_the_tickers_and_counts_the_rest():
    seams = [f"T{i:02d}" for i in range(14)]
    with pytest.raises(ValueError) as e:
        _m(adjustment_epochs=[1, 2], seam_tickers=seams).check_single_epoch()
    msg = str(e.value)
    assert "14 ticker(s)" in msg
    assert "T00" in msg
    assert "+4 more" in msg, "a long list must be truncated, not dumped"


def test_single_epoch_passes():
    _m().check_single_epoch()


def test_round_trips_through_disk(tmp_path):
    m = _m(adjustment_epochs=[1, 2], seam_tickers=["ABC"], max_ingested_at="2026-08-03")
    write_manifest(tmp_path, m)
    back = load_manifest(tmp_path)
    assert back.seam_tickers == ["ABC"]
    assert back.adjustment_epochs == [1, 2]
    assert back.row_counts["daily_prices"] == 4_777_868


def test_loads_a_manifest_written_before_seam_tickers_existed(tmp_path):
    """Old mirrors on disk must not become unloadable. They predate the field,
    so they report no seams — which matches what the old guard could observe."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"row_counts": {}, "adjustment_epochs": [1], "as_of": "2026-07-29T00:00:00"}),
        encoding="utf-8",
    )
    m = load_manifest(tmp_path)
    assert m.seam_tickers == []
    m.check_single_epoch()
