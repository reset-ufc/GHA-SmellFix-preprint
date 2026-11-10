"""Unit tests for AgentInvoker — the single framework-call facade (#32, #33).

These pin the keystone seam: a fake agent (``SimpleNamespace`` with a ``.run``
returning a framework-shaped response) drives every behaviour with no LLM and
no framework runtime — the same substrate boundary the loop tests already use.
"""

from types import SimpleNamespace

import pytest
from gha_agents.credit import OutOfCreditError
from gha_agents.errors import (
    GhaConfigError,
    GhaError,
    GhaRateLimitError,
    GhaStructuredOutputError,
    GhaTransientError,
)
from gha_agents.invoker import AgentCallResult, AgentInvoker
from gha_agents.state import CallMetrics
from pydantic import BaseModel


def _metrics(**fields) -> SimpleNamespace:
    """Framework-shaped RunMetrics stub — concrete values, not MagicMocks."""
    base = {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "time_to_first_token": None,
    }
    base.update(fields)
    return SimpleNamespace(**base)


def _fake_agent(response):
    """Agent whose ``.run(prompt)`` returns ``response`` and records the prompt."""
    calls: list[str] = []

    def run(prompt):
        calls.append(prompt)
        return response

    return SimpleNamespace(run=run, calls=calls)


# ─── slice #32: happy path ───────────────────────────────────────────────────


def test_invoke_returns_call_result_with_metrics():
    response = SimpleNamespace(
        content="DETECTION",
        status="completed",
        metrics=_metrics(
            input_tokens=120, output_tokens=40, reasoning_tokens=8, total_tokens=160
        ),
        reasoning_content="walked each job before deciding",
    )

    result = AgentInvoker().invoke("detection", _fake_agent(response), "<workflow>x</workflow>")

    assert isinstance(result, AgentCallResult)
    assert result.content == "DETECTION"
    assert isinstance(result.metrics, CallMetrics)
    assert result.metrics.input_tokens == 120
    assert result.metrics.reasoning_tokens == 8
    assert result.metrics.total_tokens == 160
    assert result.duration_ms >= 0
    assert result.started_at  # ISO timestamp captured
    assert result.reasoning_content == "walked each job before deciding"


def test_invoke_without_metrics_still_returns_valid_row():
    response = SimpleNamespace(content="X", status="completed", metrics=None)

    result = AgentInvoker().invoke("refactor", _fake_agent(response), "prompt")

    assert isinstance(result.metrics, CallMetrics)
    assert result.metrics.input_tokens == 0
    assert result.metrics.duration_ms >= 0


def test_invoke_forwards_user_turn_verbatim_to_agent():
    response = SimpleNamespace(content="ok", status="completed", metrics=None)
    agent = _fake_agent(response)

    AgentInvoker().invoke("detection", agent, "<workflow>data</workflow>")

    assert agent.calls == ["<workflow>data</workflow>"]


def test_invoke_falls_back_to_message_reasoning_when_top_level_absent():
    """Capture is robust: when the run's top-level ``reasoning_content`` is empty
    but an assistant message carries one, it is still saved (any reasoning text
    Agno surfaces lands in reasoning_content)."""
    response = SimpleNamespace(
        content="X",
        status="completed",
        metrics=None,
        reasoning_content=None,
        messages=[
            SimpleNamespace(role="user", reasoning_content=None),
            SimpleNamespace(role="assistant", reasoning_content="thought it through"),
        ],
    )

    result = AgentInvoker().invoke("detection", _fake_agent(response), "p")

    assert result.reasoning_content == "thought it through"


def test_invoke_reasoning_content_none_when_no_channel_returned():
    """No reasoning anywhere → None (not a crash), e.g. OpenAI o-series over chat."""
    response = SimpleNamespace(content="X", status="completed", metrics=None)

    result = AgentInvoker().invoke("detection", _fake_agent(response), "p")

    assert result.reasoning_content is None


# ─── slice #33: status guard ─────────────────────────────────────────────────


def test_error_status_raises_typed_failure_not_content_string():
    response = SimpleNamespace(content="boom: something failed", status="error", metrics=None)

    with pytest.raises(GhaError):
        AgentInvoker().invoke("detection", _fake_agent(response), "prompt")


def test_error_status_classifies_underlying_cause():
    response = SimpleNamespace(content="429 rate limit exceeded", status="error", metrics=None)

    # max_retries=1 asserts the classification without exercising the backoff loop.
    with pytest.raises(GhaRateLimitError):
        AgentInvoker(max_retries=1).invoke("detection", _fake_agent(response), "prompt")


def test_error_status_raises_before_building_metrics():
    class Boom:
        content = "error string"
        status = "error"

        @property
        def metrics(self):  # must never be read on the error path
            raise AssertionError

    with pytest.raises(GhaError):
        AgentInvoker().invoke("detection", _fake_agent(Boom()), "prompt")


# ─── #54: bounded rate-limit retry ───────────────────────────────────────────


def _counting_agent(side_effect):
    """Agent whose ``.run`` invokes ``side_effect(n)`` on the n-th (1-based) call:
    whatever it returns is returned, whatever it raises propagates. Records n."""
    state = {"n": 0}

    def run(prompt):
        state["n"] += 1
        return side_effect(state["n"])

    return SimpleNamespace(run=run, state=state)


def test_invoke_retries_rate_limit_then_succeeds():
    ok = SimpleNamespace(content="OK", status="completed", metrics=None)

    def side_effect(n):
        if n <= 2:
            raise GhaRateLimitError("429 rate limit")  # noqa: TRY003
        return ok

    agent = _counting_agent(side_effect)
    sleeps: list[float] = []
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=sleeps.append)

    result = invoker.invoke("detection", agent, "p")

    assert result.content == "OK"
    assert agent.state["n"] == 3  # 2 failures + 1 success
    assert sleeps == [1.0, 2.0]  # doubling backoff, no real waits


def test_invoke_raises_after_exhausting_retries():
    def side_effect(n):
        raise GhaRateLimitError("429 rate limit")  # noqa: TRY003

    agent = _counting_agent(side_effect)
    sleeps: list[float] = []
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=sleeps.append)

    with pytest.raises(GhaRateLimitError):
        invoker.invoke("detection", agent, "p")

    assert agent.state["n"] == 3  # exactly max_retries attempts
    assert sleeps == [1.0, 2.0]  # slept between attempts, not after the last


def test_invoke_does_not_retry_or_wrap_non_rate_limit_raise():
    """A non-rate-limit provider exception is neither retried nor wrapped — it
    propagates unchanged so its payload survives for downstream handling."""

    def side_effect(n):
        raise RuntimeError("model not found: does not exist")  # noqa: TRY003

    agent = _counting_agent(side_effect)
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=lambda _: None)

    with pytest.raises(RuntimeError):
        invoker.invoke("detection", agent, "p")

    assert agent.state["n"] == 1  # one attempt only, no retry


def test_invoke_retries_rate_limit_surfaced_as_error_status():
    """A rate limit that comes back as an error STATUS (not a raised exception)
    is retried too."""
    rate_limited = SimpleNamespace(content="429 too many requests", status="error", metrics=None)
    ok = SimpleNamespace(content="OK", status="completed", metrics=None)

    agent = _counting_agent(lambda n: rate_limited if n == 1 else ok)
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=lambda _: None)

    result = invoker.invoke("detection", agent, "p")

    assert result.content == "OK"
    assert agent.state["n"] == 2


# ─── #52: out-of-credit surfaces as OutOfCreditError ──────────────────────────


class _HTTP402(Exception):
    """Stand-in for an httpx 402 carrying a ``.response`` (no httpx dependency)."""

    def __init__(self):
        super().__init__("HTTP 402 Payment Required")
        self.response = SimpleNamespace(status_code=402, text="Insufficient credits")


def test_raised_402_becomes_out_of_credit_error_without_retry():
    def side_effect(n):
        raise _HTTP402()

    agent = _counting_agent(side_effect)
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=lambda _: None)

    with pytest.raises(OutOfCreditError):
        invoker.invoke("detection", agent, "p")
    assert agent.state["n"] == 1  # out-of-credit is permanent — one attempt only


def test_error_status_with_credit_message_becomes_out_of_credit_error():
    response = SimpleNamespace(
        content="402: you have insufficient credit", status="error", metrics=None
    )
    invoker = AgentInvoker(max_retries=3, initial_backoff=1.0, sleep=lambda _: None)

    with pytest.raises(OutOfCreditError):
        invoker.invoke("detection", _fake_agent(response), "p")


# ─── #58: structured-output coercion failure ─────────────────────────────────


class _Schema(BaseModel):
    smell_detected: bool


class _SchemaAgent:
    """Agent declaring a Pydantic ``output_schema`` and returning a canned response."""

    def __init__(self, response):
        self.output_schema = _Schema
        self.model = SimpleNamespace(id="test-model")
        self._response = response

    def run(self, prompt):
        return self._response


def test_structured_output_failure_raises_typed_error_with_raw_response():
    """Agno leaves the raw string as content on a coercion failure (status stays
    'completed'); the invoker turns that into a typed structured-output error."""
    raw = '{"oops": not valid json'
    agent = _SchemaAgent(SimpleNamespace(content=raw, status="completed", metrics=None))

    with pytest.raises(GhaStructuredOutputError) as excinfo:
        AgentInvoker().invoke("detection", agent, "p")
    assert excinfo.value.raw_response == raw


def test_structured_output_none_content_raises_typed_error():
    """Native structured outputs yield content=None on a parse failure."""
    agent = _SchemaAgent(SimpleNamespace(content=None, status="completed", metrics=None))

    with pytest.raises(GhaStructuredOutputError):
        AgentInvoker().invoke("detection", agent, "p")


def test_structured_output_valid_model_passes():
    """Content that IS an instance of the schema is accepted unchanged."""
    parsed = _Schema(smell_detected=True)
    agent = _SchemaAgent(SimpleNamespace(content=parsed, status="completed", metrics=None))

    result = AgentInvoker().invoke("detection", agent, "p")
    assert result.content is parsed


# ─── transient retry: critical-path safety (no abort / no corrupt / no stall) ─
#
# The retry lives ENTIRELY in AgentInvoker.invoke(): it adds bounded in-place
# re-calls and never touches persistence, so the no-corrupt / resume-safe
# guarantees are unchanged (see test_store.py + test_resume_with_trace.py). These
# tests pin the retry's own safety properties.


def test_default_max_retries_is_five():
    """Lock the bumped default so a regression can't silently shrink the budget."""
    assert AgentInvoker()._max_retries == 5


def test_retries_openrouter_provider_envelope_status_then_succeeds():
    """The exact production bug: a 429 arrives as an error STATUS whose content is
    the generic 'Provider returned error' (real cause buried in metadata we never
    see). It was classified UNKNOWN and NOT retried; now it retries and recovers."""
    envelope = SimpleNamespace(content="Provider returned error", status="error", metrics=None)
    ok = SimpleNamespace(content="OK", status="completed", metrics=None)
    agent = _counting_agent(lambda n: envelope if n <= 2 else ok)
    sleeps: list[float] = []
    invoker = AgentInvoker(max_retries=5, initial_backoff=1.0, sleep=sleeps.append)

    result = invoker.invoke("detection", agent, "p")

    assert result.content == "OK"
    assert agent.state["n"] == 3  # 2 transient failures + 1 success
    assert sleeps == [1.0, 2.0]


def test_retries_5xx_raised_then_succeeds():
    """A raised 5xx (provider/upstream) is transient and recovers in place."""
    ok = SimpleNamespace(content="OK", status="completed", metrics=None)

    def side_effect(n):
        if n == 1:
            raise RuntimeError("503 Service Unavailable")  # noqa: TRY003
        return ok

    agent = _counting_agent(side_effect)
    result = AgentInvoker(max_retries=5, initial_backoff=1.0, sleep=lambda _: None).invoke(
        "detection", agent, "p"
    )

    assert result.content == "OK"
    assert agent.state["n"] == 2


def test_transient_terminates_and_raises_after_bounded_attempts():
    """STARVATION / DEADLOCK guard: a permanently-transient provider can never
    loop forever. The loop is bounded to exactly ``max_retries`` attempts with a
    finite, monotonic backoff, then raises so the pair fails per-cell (the run
    continues, resume re-runs it)."""

    def side_effect(n):
        raise RuntimeError("provider returned error")  # noqa: TRY003

    agent = _counting_agent(side_effect)
    sleeps: list[float] = []
    invoker = AgentInvoker(max_retries=5, initial_backoff=2.0, sleep=sleeps.append)

    with pytest.raises(GhaTransientError):
        invoker.invoke("detection", agent, "p")

    assert agent.state["n"] == 5  # bounded: exactly max_retries attempts, never more
    assert sleeps == [2.0, 4.0, 8.0, 16.0]  # 4 finite sleeps (between attempts), then raise
    assert sum(sleeps) == 30.0  # total backoff is finite and bounded


def test_permanent_error_not_retried_no_wasted_budget():
    """A permanent error that matches no transient signal fails on the FIRST
    attempt — the retry budget and wall-clock are not burned on it."""

    def side_effect(n):
        raise RuntimeError("invalid request: unsupported parameter")  # noqa: TRY003

    agent = _counting_agent(side_effect)
    sleeps: list[float] = []
    invoker = AgentInvoker(max_retries=5, initial_backoff=1.0, sleep=sleeps.append)

    with pytest.raises(RuntimeError):
        invoker.invoke("detection", agent, "p")

    assert agent.state["n"] == 1  # no retry
    assert sleeps == []  # no backoff slept


def test_structured_output_failure_is_not_retried():
    """A structured-output coercion failure is model-reliability, NOT transient:
    it fails on the first attempt. (Retrying would re-sample and change what the
    detection F1 measures — that is a separate, opt-in decision.)"""
    raw = '{"oops": not valid json'
    state = {"n": 0}

    def run(prompt):
        state["n"] += 1
        return SimpleNamespace(content=raw, status="completed", metrics=None)

    agent = SimpleNamespace(run=run, output_schema=_Schema, model=SimpleNamespace(id="m"))

    with pytest.raises(GhaStructuredOutputError):
        AgentInvoker(max_retries=5, initial_backoff=1.0, sleep=lambda _: None).invoke(
            "detection", agent, "p"
        )

    assert state["n"] == 1  # not retried


def test_out_of_credit_is_not_retried_even_via_transient_path():
    """A 429 that is actually out-of-credit must HALT, not retry into an empty
    wallet: out-of-credit is diverted before the transient classification."""
    response = SimpleNamespace(
        content="429: insufficient credit, please add funds", status="error", metrics=None
    )
    agent = _counting_agent(lambda n: response)
    invoker = AgentInvoker(max_retries=5, initial_backoff=1.0, sleep=lambda _: None)

    with pytest.raises(OutOfCreditError):
        invoker.invoke("detection", agent, "p")
    assert agent.state["n"] == 1  # permanent — one attempt only


def test_transient_error_is_not_a_fatal_pipeline_error():
    """NO-ABORT guard (type level): the per-pair handler re-raises only
    GhaConfigError / OutOfCreditError as fatal; everything else is caught by
    ``except Exception`` and fails one pair. GhaTransientError must be neither
    fatal type yet still an Exception, so an exhausted retry can never abort the
    whole run."""
    err = GhaTransientError("exhausted")
    assert isinstance(err, Exception)
    assert isinstance(err, GhaError)
    assert not isinstance(err, GhaConfigError)
    assert not isinstance(err, OutOfCreditError)


def test_concurrent_invokes_are_independent_no_shared_state():
    """THREAD-SAFETY / no-starvation guard: one shared AgentInvoker is stateless
    across calls. Concurrent invokes keep independent retry counters and all
    terminate within a bounded time — one worker's backoff cannot block another,
    and there is no shared lock to deadlock on."""
    import threading

    invoker = AgentInvoker(max_retries=4, initial_backoff=0.0, sleep=lambda _: None)

    def always_transient(n):
        raise RuntimeError("provider returned error")  # noqa: TRY003

    results: dict[int, tuple[str, int]] = {}

    def worker(i: int) -> None:
        agent = _counting_agent(always_transient)
        try:
            invoker.invoke("detection", agent, f"p{i}")
            results[i] = ("ok", agent.state["n"])
        except GhaTransientError:
            results[i] = ("raised", agent.state["n"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(not t.is_alive() for t in threads)  # no deadlock / hang
    assert len(results) == 8
    assert all(v == ("raised", 4) for v in results.values())  # independent, bounded each
