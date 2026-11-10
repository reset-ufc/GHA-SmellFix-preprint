"""Tests for the live progress dashboard (gha_agents.progress)."""

from __future__ import annotations

import io

from gha_agents.progress import (
    NullReporter,
    RichBarsReporter,
    _LiveFooter,
    count_pairs,
)
from rich.console import Console


def test_count_pairs_sums_present_and_absent_smells():
    entries = [
        {
            "present_smells": [{"pascal": "A"}],
            "absent_smells": [{"pascal": "B"}, {"pascal": "C"}],
        },
        {"present_smells": [], "absent_smells": []},
    ]
    assert count_pairs(entries) == 3


def test_null_reporter_is_a_noop():
    reporter = NullReporter()
    with reporter as ctx:
        assert ctx is reporter
        reporter.pair_done("D1", "m", {"status": "OK"})  # must not raise


def _reporter() -> RichBarsReporter:
    return RichBarsReporter([("D1", "m1"), ("D2", "m1"), ("D1", "m2")], pairs_per_cell=2)


def test_pair_done_tallies_counts_without_a_live_region():
    """pair_done updates the counters under its lock and never calls Live.update,
    so it is safe to drive even before/outside the context manager."""
    reporter = _reporter()
    reporter.pair_done("D1", "m1", {"smells": "1", "solved": "1", "status": "OK"})
    reporter.pair_done("D2", "m1", {"smells": "?", "solved": "?", "status": "ERROR"})
    reporter.pair_done("D1", "m2", {"smells": "·", "solved": "·", "status": "SKIP"})

    assert reporter._counts["detected"] == 1
    assert reporter._counts["solved"] == 1
    assert reporter._counts["failed"] == 1
    assert reporter._counts["skipped"] == 1


def test_live_footer_rebuilds_from_current_state():
    """The footer renderable reflects the latest counts every time it renders —
    this is what lets the dashboard track the run off the single auto-refresh
    tick instead of a per-pair (per-thread) Live.update that tore frames."""
    reporter = _reporter()
    footer = _LiveFooter(reporter)

    first = Console(file=io.StringIO(), force_terminal=False, width=80)
    first.print(footer)
    assert "detected 0" in first.file.getvalue()

    reporter.pair_done("D1", "m1", {"smells": "1", "solved": "1", "status": "OK"})

    second = Console(file=io.StringIO(), force_terminal=False, width=80)
    second.print(footer)  # same footer object, new state
    assert "detected 1" in second.file.getvalue()
