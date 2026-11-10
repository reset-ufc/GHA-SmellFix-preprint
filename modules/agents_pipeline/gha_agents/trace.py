"""Audit-trail Pydantic models — opt-in via ``execution.persist_traces``.

Each pair run mirrors its result JSON with a sibling ``traces/{slug}.json``
that captures every agent call (rendered prompt, response content,
reasoning_content, tool calls, per-call metrics + duration). Useful for
post-mortem analysis and replication; default off because it doubles the
on-disk footprint.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentRole = Literal[
    "detection",
    "detection_validation",
    "refactor",
    "refactor_validation",
    "combined",
]
LoopName = Literal["detection", "refactor"]


class CallMetrics(BaseModel):
    """Full per-call record for one ``Agent.run()`` invocation.

    Unlike :class:`~gha_agents.state.TokenUsage` — a pure summable
    accumulator — this carries the non-summable per-call fields
    (``time_to_first_token``, wall-clock ``duration_ms``) plus the complete
    token breakdown. Populated from Agno's ``RunMetrics`` + ``RunOutput``
    wherever the provider reports the field; capture-only, no derived metrics.

    ``reasoning_tokens`` is a subset of ``output_tokens`` (not additive);
    ``total_tokens`` is taken verbatim from the provider so it is never
    double-counted here.

    Lives here (not in ``state``) so the audit-trail models can type their
    ``metrics`` field without ``state`` having to import ``trace`` — the
    dependency runs one way, ``state`` → ``trace``.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: float | None = None
    duration_ms: int = 0

    @classmethod
    def from_agno_response(cls, response: Any, *, duration_ms: int = 0) -> CallMetrics:
        """Build a record from an Agno ``RunOutput``.

        Token counts and ``time_to_first_token`` come from
        ``response.metrics`` (Agno's ``RunMetrics``) wherever the provider
        populated them. ``duration_ms`` is the caller's wall-clock measurement —
        Agno's own ``duration`` is unreliable across providers. A response with
        no ``metrics`` still yields a valid (all-default) capture row.
        """
        metrics = getattr(response, "metrics", None)
        if metrics is None:
            return cls(duration_ms=duration_ms)
        return cls(
            input_tokens=int(getattr(metrics, "input_tokens", 0) or 0),
            output_tokens=int(getattr(metrics, "output_tokens", 0) or 0),
            reasoning_tokens=int(getattr(metrics, "reasoning_tokens", 0) or 0),
            cache_read_tokens=int(getattr(metrics, "cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(metrics, "cache_write_tokens", 0) or 0),
            total_tokens=int(getattr(metrics, "total_tokens", 0) or 0),
            time_to_first_token=getattr(metrics, "time_to_first_token", None),
            duration_ms=duration_ms,
        )


class ToolCall(BaseModel):
    """One Agno tool invocation captured during an agent call."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict
    output_full: str
    duration_ms: int = 0


class AgentCall(BaseModel):
    """One ``Agent.run()`` call — system + user prompt, response, metrics, tools.

    ``system_prompt`` is the agent's rendered instructions (the ``build_*_prompt``
    output); ``user_prompt`` is the per-call user turn. Both are captured so a
    trace reconstructs the exact two-message exchange sent to the model.
    """

    model_config = ConfigDict(extra="forbid")

    loop: LoopName | None = None
    iteration: int | None = None
    role: AgentRole
    model_id: str
    model_provider: str = ""
    system_prompt: str = ""
    user_prompt: str
    response_content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metrics: CallMetrics = Field(default_factory=CallMetrics)
    started_at: str = ""


class TraceError(BaseModel):
    """Captured exception for partial-trail emission on agent failure."""

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str
    raised_at_call_index: int


class TraceFile(BaseModel):
    """Full per-pair audit trail. Persisted to ``traces/{slug}.json``."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    design: Literal["D1", "D2"]
    calls: list[AgentCall] = Field(default_factory=list)
    error: TraceError | None = None
