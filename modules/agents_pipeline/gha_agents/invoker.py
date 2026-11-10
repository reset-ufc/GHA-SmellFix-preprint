"""invoker.py — AgentInvoker: the single facade over the framework agent call.

The per-agent call ritual (wall-clock timing, ``agent.run``, ``RunMetrics`` ->
``CallMetrics`` mapping, native reasoning capture, and the framework's
error-status-as-content guard) was copy-pasted across every call site. This
module collapses it into one place.

AgentInvoker is the sole seam the domain crosses to reach the framework, so a
fake of it makes every gate and synthesis testable without an LLM or the
framework runtime. It duck-types the response
(``content``/``status``/``metrics``/``reasoning_content``) and reuses
:meth:`CallMetrics.from_agno_response`, so it imports no framework symbols.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .credit import OutOfCreditError, is_out_of_credit_error, message_signals_out_of_credit
from .errors import (
    GhaStructuredOutputError,
    GhaTransientError,
    classify_agent_error,
)
from .state import CallMetrics
from .trace import ToolCall

# The framework returns this run status (with the exception string as ``content``)
# instead of raising when an agent run fails. Mirrors agno ``RunStatus.error.value``.
_ERROR_STATUS = "error"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _model_id_of(agent: Any) -> str:
    """Best-effort lookup of the underlying model_id from a framework agent."""
    return str(getattr(getattr(agent, "model", None), "id", "") or "")


def _system_prompt_of(agent: Any) -> str:
    """The rendered system prompt the agent was built with (its ``instructions``).

    Best-effort: the pipeline always passes a ``str`` (the ``build_*_prompt``
    output), but a framework may store a list/None — anything non-str collapses
    to ``""`` so the trace field stays a plain string.
    """
    instructions = getattr(agent, "instructions", "")
    return instructions if isinstance(instructions, str) else ""


def _extract_tool_calls(response: Any) -> list[ToolCall]:
    """Convert ``RunOutput.tools`` into our :class:`ToolCall` model.

    Tolerates both dict-shaped and attribute-shaped tool entries the framework
    has used across versions; missing fields default to safe values.
    """
    raw_tools = getattr(response, "tools", None) or []
    captured: list[ToolCall] = []
    for tool in raw_tools:
        if isinstance(tool, dict):
            name = str(tool.get("name") or tool.get("tool_name") or "")
            args = tool.get("args") or tool.get("arguments") or {}
            output_full = str(tool.get("output") or tool.get("result") or "")
            duration_ms = int(tool.get("duration_ms") or 0)
        else:
            name = str(getattr(tool, "name", "") or getattr(tool, "tool_name", "") or "")
            args = getattr(tool, "args", None) or getattr(tool, "arguments", None) or {}
            output_full = str(getattr(tool, "output", "") or getattr(tool, "result", "") or "")
            duration_ms = int(getattr(tool, "duration_ms", 0) or 0)
        if not isinstance(args, dict):
            args = {"value": args}
        captured.append(
            ToolCall(name=name, args=args, output_full=output_full, duration_ms=duration_ms)
        )
    return captured


def _extract_reasoning(response: Any) -> str | None:
    """The provider's native reasoning text for this call, or ``None``.

    Primary source is the framework's top-level ``reasoning_content``: Agno maps
    the OpenRouter response's ``reasoning`` field onto it (the request opts into
    that channel — see :mod:`gha_agents.model_factory`). As a fallback — for
    paths that populate only a per-message copy — scan the run's messages for the
    first carrying ``reasoning_content`` (or the redacted variant). OpenRouter's
    structured ``reasoning_details`` array is not modelled by the framework and is
    dropped at the SDK seam, but it carries the same text as the flat ``reasoning``
    field that already feeds ``reasoning_content``, so nothing recoverable is lost.
    """
    top = getattr(response, "reasoning_content", None)
    if top:
        return top
    for message in getattr(response, "messages", None) or ():
        text = getattr(message, "reasoning_content", None) or getattr(
            message, "redacted_reasoning_content", None
        )
        if text:
            return text
    return None


@dataclass(frozen=True, slots=True)
class AgentCallResult:
    """Typed outcome of one :meth:`AgentInvoker.invoke` call.

    ``content`` is the validated output (the framework already coerced it into
    the agent's output_schema). ``metrics`` is the full per-call record;
    ``reasoning_content`` is the provider's native reasoning text where
    returned. ``response`` keeps the raw run output so the trace sink can still
    extract tool calls / model provider without re-crossing the seam.
    """

    content: Any
    metrics: CallMetrics
    started_at: str
    duration_ms: int
    reasoning_content: str | None = None
    model_id: str = ""
    model_provider: str = ""
    system_prompt: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    response: Any = None


class AgentInvoker:
    """Facade collapsing the per-agent call ritual into one tested place.

    Transient failures are retried in place with bounded exponential backoff
    (``max_retries`` attempts, doubling from ``initial_backoff`` seconds). This
    covers explicit rate limits (HTTP 429) AND the generic provider/5xx hiccups
    OpenRouter surfaces as a ``"provider returned error"`` status whose real
    cause never reaches ``str(exc)`` — both classify as
    :class:`~gha_agents.errors.GhaTransientError`. It is the only retry on
    the call path: with a per-call HTTP timeout active the provider SDK is built
    with ``max_retries=0`` (see :mod:`gha_agents.model_factory`), so without
    this the path would not retry at all. Every non-transient failure — auth,
    model-not-found, structured-output, out-of-credit — is permanent and
    propagates immediately. ``sleep`` is injectable so tests exercise the
    backoff without real waits.
    """

    def __init__(
        self,
        *,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._sleep = sleep

    def invoke(self, role: str, agent: Any, user_turn: str) -> AgentCallResult:
        """Run ``agent`` on ``user_turn``, retrying transient failures, and
        return a typed call result.

        Each attempt measures a single monotonic clock around the call only and
        surfaces the framework's error status as a hard typed failure BEFORE any
        metrics row is produced, so an error string can never masquerade as a
        result. A classified transient failure — rate limit (429) or a generic
        provider/5xx error, raised or surfaced as an error status — is retried
        up to ``max_retries`` times with doubling backoff; every other failure
        propagates on the first attempt.
        """
        attempt = 0
        while True:
            try:
                return self._invoke_once(role, agent, user_turn)
            except GhaTransientError:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                self._sleep(self._initial_backoff * (2 ** (attempt - 1)))

    def _invoke_once(self, role: str, agent: Any, user_turn: str) -> AgentCallResult:
        """One attempt: run the agent, guard the status, build the typed result.

        A raised provider exception that classifies as transient (rate limit or
        generic provider/5xx) is re-raised as the typed
        :class:`GhaTransientError` so :meth:`invoke` can retry it; every other
        raised error propagates unchanged so its original payload (status code,
        response body) survives for downstream handling. The status guard
        (:meth:`_guard_status`) likewise raises a typed transient error for the
        framework's error-status path, which :meth:`invoke` retries the same way.
        """
        started_at = _now_iso()
        t0 = time.monotonic()
        try:
            response = agent.run(user_turn)
        except (GhaTransientError, OutOfCreditError):
            raise
        except Exception as exc:
            if is_out_of_credit_error(exc) or message_signals_out_of_credit(str(exc)):
                raise OutOfCreditError(str(exc)) from exc
            typed = classify_agent_error(exc)
            if isinstance(typed, GhaTransientError):
                raise typed from exc
            raise
        duration_ms = int((time.monotonic() - t0) * 1000)
        self._guard_status(role, response)
        self._guard_structured_output(role, agent, response)
        metrics = CallMetrics.from_agno_response(response, duration_ms=duration_ms)
        return AgentCallResult(
            content=getattr(response, "content", None),
            metrics=metrics,
            started_at=started_at,
            duration_ms=duration_ms,
            reasoning_content=_extract_reasoning(response),
            model_id=_model_id_of(agent),
            model_provider=str(getattr(response, "model_provider", "") or ""),
            system_prompt=_system_prompt_of(agent),
            tool_calls=tuple(_extract_tool_calls(response)),
            response=response,
        )

    @staticmethod
    def _guard_status(role: str, response: Any) -> None:
        """Turn the framework's error-status into a hard, typed cell failure.

        The framework returns an error status with the exception string as
        ``content`` rather than raising. Left unguarded, that string would flow
        downstream as if it were a validated result. We classify the underlying
        cause and raise a typed :class:`GhaError` before metrics are built. An
        out-of-credit signal in the content is raised as :class:`OutOfCreditError`
        so the dispatcher fast-fails the whole run rather than logging it per-cell.
        """
        status = getattr(response, "status", None)
        status_value = getattr(status, "value", status)
        if str(status_value).lower() == _ERROR_STATUS:
            content = getattr(response, "content", None)
            cause = content if isinstance(content, str) and content else f"{role} agent run failed"
            if message_signals_out_of_credit(cause):
                raise OutOfCreditError(cause)
            raise classify_agent_error(Exception(cause))

    @staticmethod
    def _guard_structured_output(role: str, agent: Any, response: Any) -> None:
        """Fail a call whose content didn't coerce into the agent's output_schema.

        On a structured-output coercion failure Agno neither raises nor sets an
        error status — its parser leaves the raw model string as ``content``
        (JSON mode) or yields ``None`` (native structured outputs), which would
        otherwise flow downstream and only blow up later as an ``AttributeError``
        on a plain string. When the agent declares a class ``output_schema`` and
        the content is not an instance of it, raise a typed
        :class:`GhaStructuredOutputError` carrying the raw payload so the
        failure is categorized and the unparsed response is preserved.
        """
        schema = getattr(agent, "output_schema", None)
        if not isinstance(schema, type):
            return  # no schema, or a dict/JSON-schema we don't introspect here
        content = getattr(response, "content", None)
        if isinstance(content, schema):
            return
        raw = content if isinstance(content, str) else (None if content is None else str(content))
        raise GhaStructuredOutputError(
            f"{role} agent returned content that is not a {schema.__name__} "
            f"(got {type(content).__name__})",
            raw_response=raw,
        )
