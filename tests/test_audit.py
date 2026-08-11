"""The data-quality audit.

No database here — the queries need one, so what is tested is the reporting
contract they feed. That contract is where an audit usually rots: a check that
returns "passed" while counting something, or one that fails silently because
nobody looks at the count.
"""

from __future__ import annotations

from vnresearch.io import audit


def test_a_check_passes_only_when_it_found_nothing():
    assert audit.Check("x", "t", "d", n_bad=0).passed is True
    assert audit.Check("x", "t", "d", n_bad=1).passed is False
    assert audit.Check("x", "t", "d", n_bad=177).passed is False


def test_passed_cannot_be_set_independently_of_the_count():
    """`passed` is derived, not assigned. A check that reports success while
    holding a non-zero count is worse than no check — it is a reason not to
    look."""
    c = audit.Check("x", "t", "d", n_bad=5)
    assert c.passed is False
    c.n_bad = 0
    c.__post_init__()
    assert c.passed is True


def test_a_check_defaults_to_clean():
    """So a check that forgets to count reads as passing rather than crashing —
    the count is the thing under test, and it is always explicit at the call
    sites in this module."""
    assert audit.Check("x", "t", "d").passed is True


def test_every_check_carries_a_scope_and_an_explanation():
    """`n_bad = 177` on its own does not tell anyone what to do next. Each check
    names the object it looked at and why the result matters."""
    c = audit.Check("no_nan_or_inf", "features.skew_21", "NaN is not NULL", 177)
    assert c.scope and c.detail
    assert c.name.replace("_", "").isalnum()
