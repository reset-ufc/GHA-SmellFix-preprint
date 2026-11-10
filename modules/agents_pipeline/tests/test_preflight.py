"""Unit tests for the pre-flight model gate.

A fake factory (returns a canned built-model) and a fake probe (returns a canned
call result, or raises) drive every branch with no LLM and no framework runtime —
the same substrate boundary the invoker/loop tests use.
"""

from types import SimpleNamespace

import pytest
from gha_agents.model_factory import _ANTHROPIC_THINKING_TOP_P
from gha_agents.preflight import (
    ModelCheck,
    PreflightError,
    _temperature_ok,
    _thinking_ok,
    assert_models_ready,
    format_report,
    run_model_preflight,
)


def _run(alias="m", provider="openrouter", *, thinking=True, model_id="ns/model"):
    """A duck-typed ResolvedRun — only the fields run_model_preflight reads."""
    return SimpleNamespace(
        model_alias=alias,
        provider=provider,
        model_map={"detection": model_id},
        sampling={},
        thinking=thinking,
        json_mode=True,
    )


def _model(temperature=0.0, top_p=1.0, reasoning_effort=None):
    return SimpleNamespace(temperature=temperature, top_p=top_p, reasoning_effort=reasoning_effort)


def _result(reasoning_content=None, reasoning_tokens=0):
    return SimpleNamespace(
        reasoning_content=reasoning_content,
        metrics=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


class _FakeFactory:
    def __init__(self, model):
        self._model = model

    def build(self, role):
        return self._model


def _preflight(run, model, probe):
    return run_model_preflight([run], factory=_FakeFactory(model), probe=probe)


# ─── _temperature_ok (static, pure) ───────────────────────────────────────────


def test_temperature_ok_non_anthropic_requires_zero():
    ok, _ = _temperature_ok("openrouter", _model(temperature=0.0), thinking_on=True)
    assert ok is True


def test_temperature_fail_non_anthropic_nonzero():
    ok, detail = _temperature_ok("openrouter", _model(temperature=0.7), thinking_on=True)
    assert ok is False
    assert "0.0" in detail


def test_temperature_ok_anthropic_unset_with_floor_top_p():
    ok, _ = _temperature_ok(
        "anthropic", _model(temperature=None, top_p=_ANTHROPIC_THINKING_TOP_P), thinking_on=True
    )
    assert ok is True


def test_temperature_fail_anthropic_temperature_set():
    ok, detail = _temperature_ok("anthropic", _model(temperature=1.0, top_p=0.95), thinking_on=True)
    assert ok is False
    assert "unset" in detail


def test_temperature_fail_anthropic_wrong_top_p():
    ok, detail = _temperature_ok("anthropic", _model(temperature=None, top_p=1.0), thinking_on=True)
    assert ok is False
    assert "top_p" in detail


def test_temperature_ok_anthropic_thinking_off_requires_zero():
    # With thinking OFF the factory leaves the matrix sampling untouched
    # (temperature=0.0, top_p=1.0), so anthropic follows the ordinary 0.0 rule —
    # the gate must NOT demand the thinking-on (unset temp / 0.95 top_p) shape.
    ok, _ = _temperature_ok("anthropic", _model(temperature=0.0, top_p=1.0), thinking_on=False)
    assert ok is True


def test_temperature_fail_anthropic_thinking_off_nonzero():
    # Thinking off still enforces temperature==0.0 like every other provider.
    ok, detail = _temperature_ok("anthropic", _model(temperature=0.7), thinking_on=False)
    assert ok is False
    assert "0.0" in detail


def test_temperature_ok_openai_reasoning_requires_unset():
    # OpenAI reasoning models (reasoning_effort set) reject temperature != 1, so
    # the factory drops it — the gate expects it unset, like anthropic. Keyed on
    # reasoning_effort presence, not thinking_on (pass thinking_on=False to prove it).
    ok, _ = _temperature_ok(
        "openai", _model(temperature=None, reasoning_effort="medium"), thinking_on=False
    )
    assert ok is True


def test_temperature_fail_openai_reasoning_temperature_set():
    ok, detail = _temperature_ok(
        "openai", _model(temperature=0.0, reasoning_effort="medium"), thinking_on=True
    )
    assert ok is False
    assert "unset" in detail


def test_temperature_ok_openai_without_reasoning_requires_zero():
    # A non-reasoning OpenAI model still follows the temperature==0.0 rule.
    ok, _ = _temperature_ok("openai", _model(temperature=0.0, reasoning_effort=None), thinking_on=True)
    assert ok is True


# ─── _thinking_ok (the unified signal) ────────────────────────────────────────


def test_thinking_ok_via_reasoning_content():
    assert _thinking_ok(_result(reasoning_content="walked the steps")) is True


def test_thinking_ok_via_reasoning_tokens_only():
    # OpenAI: text is always hidden, only the token count proves reasoning.
    assert _thinking_ok(_result(reasoning_content=None, reasoning_tokens=13)) is True


def test_thinking_fail_when_neither_present():
    assert _thinking_ok(_result(reasoning_content=None, reasoning_tokens=0)) is False


# ─── run_model_preflight ──────────────────────────────────────────────────────


def test_passing_model_is_marked_passed():
    [check] = _preflight(
        _run(provider="openrouter"),
        _model(temperature=0.0),
        lambda inv, m: _result(reasoning_content="thoughts", reasoning_tokens=31),
    )
    assert check.passed
    assert check.accessible and check.temperature_ok and check.thinking_ok


def test_openai_passes_on_tokens_without_text():
    [check] = _preflight(
        _run(provider="openai"),
        _model(temperature=0.0),
        lambda inv, m: _result(reasoning_content=None, reasoning_tokens=13),
    )
    assert check.passed


def test_thinking_failure_blocks_with_reason():
    [check] = _preflight(
        _run(provider="openrouter"),
        _model(temperature=0.0),
        lambda inv, m: _result(reasoning_content=None, reasoning_tokens=0),
    )
    assert not check.passed
    assert check.accessible  # the call succeeded; it just didn't reason
    assert not check.thinking_ok
    assert "reasoning" in check.detail


def test_temperature_failure_blocks():
    [check] = _preflight(
        _run(provider="openrouter"),
        _model(temperature=0.7),
        lambda inv, m: _result(reasoning_content="thoughts"),
    )
    assert not check.passed
    assert not check.temperature_ok


def test_probe_exception_marks_inaccessible():
    def boom(inv, m):
        raise RuntimeError("401 invalid api key")  # noqa: TRY003

    [check] = _preflight(_run(provider="openai"), _model(temperature=0.0), boom)
    assert not check.passed
    assert not check.accessible
    assert "401" in check.detail


def test_models_deduped_by_alias():
    runs = [_run(alias="dup"), _run(alias="dup")]
    checks = run_model_preflight(
        runs,
        factory=_FakeFactory(_model(temperature=0.0)),
        probe=lambda inv, m: _result(reasoning_content="x"),
    )
    assert len(checks) == 1


# ─── assert_models_ready (the gate) ───────────────────────────────────────────


def test_assert_raises_on_any_failure():
    with pytest.raises(PreflightError) as excinfo:
        assert_models_ready(
            [_run(provider="openrouter")],
            factory=_FakeFactory(_model(temperature=0.7)),
            probe=lambda inv, m: _result(reasoning_content="x"),
        )
    assert "temp✖" in str(excinfo.value)


def test_assert_returns_checks_on_all_pass():
    checks = assert_models_ready(
        [_run(provider="anthropic")],
        factory=_FakeFactory(_model(temperature=None, top_p=_ANTHROPIC_THINKING_TOP_P)),
        probe=lambda inv, m: _result(reasoning_content="thinking"),
    )
    assert all(c.passed for c in checks)


def test_format_report_marks_pass_and_fail():
    report = format_report(
        [
            ModelCheck("ok", "anthropic", True, True, True),
            ModelCheck("bad", "openai", True, True, False, "thinking not observed"),
        ]
    )
    assert "✔ ok" in report
    assert "✖ bad" in report
    assert "thinking not observed" in report
