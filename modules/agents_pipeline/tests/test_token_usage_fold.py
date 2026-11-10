"""Tests for RoleTokenUsage.fold — the typed per-role + total token rollup.

``fold`` accumulates the five summable token counts of one :class:`CallMetrics`
into the per-role bucket and the grand total, counting the call on both. It is
framework-free and mutates the :class:`RoleTokenUsage` in place, so the
``token_usage`` JSON the result dict reads stays byte-identical. (This fold used
to live in a separate MetricsLedger object; it now lives on the state model.)
"""

from __future__ import annotations

from gha_agents.state import CallMetrics, RoleTokenUsage


def test_fold_accumulates_summable_counts_into_per_role_and_total():
    usage = RoleTokenUsage()
    call = CallMetrics(
        input_tokens=100,
        output_tokens=40,
        reasoning_tokens=25,
        cache_read_tokens=10,
        cache_write_tokens=0,
        total_tokens=150,
        time_to_first_token=0.5,
        duration_ms=1234,
    )

    usage.fold(call, "detection")

    role = usage.by_role["detection"]
    assert role.input_tokens == 100
    assert role.output_tokens == 40
    assert role.reasoning_tokens == 25
    assert role.cache_read_tokens == 10
    assert role.calls == 1
    assert usage.total.input_tokens == 100
    assert usage.total.reasoning_tokens == 25
    assert usage.total.calls == 1


def test_fold_returns_the_same_metrics_object():
    # teams._fold_metrics relies on `cm is call` at the call sites.
    usage = RoleTokenUsage()
    call = CallMetrics(input_tokens=5)
    assert usage.fold(call, "refactor") is call


def test_fold_empty_call_still_counts_the_call():
    usage = RoleTokenUsage()
    usage.fold(CallMetrics(duration_ms=7), "refactor")
    assert usage.by_role["refactor"].calls == 1
    assert usage.total.calls == 1
    assert usage.by_role["refactor"].input_tokens == 0


def test_reasoning_tokens_folded_but_not_added_onto_output():
    usage = RoleTokenUsage()
    usage.fold(CallMetrics(output_tokens=40, reasoning_tokens=25), "combined")
    bucket = usage.by_role["combined"]
    # reasoning_tokens is a subset of output_tokens, never additive.
    assert bucket.output_tokens == 40
    assert bucket.reasoning_tokens == 25


def test_fold_accumulates_across_calls_and_roles():
    usage = RoleTokenUsage()
    usage.fold(CallMetrics(input_tokens=100), "detection")
    usage.fold(CallMetrics(input_tokens=10), "detection")
    usage.fold(CallMetrics(input_tokens=7), "refactor")
    assert usage.by_role["detection"].input_tokens == 110
    assert usage.by_role["detection"].calls == 2
    assert usage.by_role["refactor"].input_tokens == 7
    assert usage.total.input_tokens == 117
    assert usage.total.calls == 3


def test_default_usage_starts_empty():
    usage = RoleTokenUsage()
    assert usage.total.calls == 0
    assert all(b.calls == 0 for b in usage.by_role.values())
