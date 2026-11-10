"""Pydantic models for in-memory workflow state.

These mirror the legacy plain-dict shapes that the team builders used to return,
so ``state.model_dump()`` produces JSON identical to the previous result schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from .agents import CombinedResult, DetectionContract, EvidenceRange, RefactorValidationResult
from .trace import AgentCall, CallMetrics

_ROLES = ("detection", "detection_validation", "refactor", "refactor_validation", "combined")

# The summable token counts folded across calls. ``reasoning_tokens`` is included
# because it is itself summable; it is a subset of ``output_tokens`` and is never
# added on top of it. The non-summable per-call fields (total_tokens,
# time_to_first_token, duration_ms) are deliberately NOT rolled up.
_SUMMABLE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


class TokenUsage(BaseModel):
    """Per-role or total token counters folded across all agent calls.

    A pure summable accumulator — every field adds across calls. ``reasoning_tokens``
    is a subset of ``output_tokens`` (a count of how many output tokens were
    reasoning), kept here because it is still summable; it is never added on top
    of ``output_tokens``.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0


def _default_by_role() -> dict[str, TokenUsage]:
    return {role: TokenUsage() for role in _ROLES}


class RoleTokenUsage(BaseModel):
    """Token usage aggregator with a grand total and per-role buckets."""

    model_config = ConfigDict(extra="forbid")

    total: TokenUsage = Field(default_factory=TokenUsage)
    by_role: dict[str, TokenUsage] = Field(default_factory=_default_by_role)

    @model_serializer(mode="wrap")
    def _drop_idle_roles(self, handler):
        """Serialize only the roles that actually ran (``calls > 0``).

        Every role is pre-seeded with a zero bucket so :meth:`fold` never has to
        branch on first use, but emitting all five in the result JSON reads as if
        a D1 run drove four idle agents — readers have miscounted the agents from
        it. The in-memory object is untouched; only the dumped ``by_role`` is
        pruned. A role with no call contributes no
        tokens, so the ``total`` is unaffected and stats default the absent role
        back to zero on load.
        """
        data = handler(self)
        data["by_role"] = {
            role: bucket
            for role, bucket in data["by_role"].items()
            if self.by_role[role].calls > 0
        }
        return data

    def fold(self, metrics: CallMetrics, role: str) -> CallMetrics:
        """Fold one call's summable token counts into the ``role`` bucket and the
        total, counting the call on both; returns ``metrics`` unchanged.

        Capture only — no metric is derived. A call with no token data still
        counts against both buckets. Raises ``KeyError`` for an unregistered role.
        """
        role_bucket = self.by_role[role]
        for field in _SUMMABLE_FIELDS:
            value = getattr(metrics, field)
            setattr(role_bucket, field, getattr(role_bucket, field) + value)
            setattr(self.total, field, getattr(self.total, field) + value)
        role_bucket.calls += 1
        self.total.calls += 1
        return metrics


class DetectionPhaseEntry(BaseModel):
    """A1's per-iteration detection output, with the full per-call record.

    ``count`` and ``evidence_ranges`` carry the multi-instance detection
    signal for this iteration (one ``EvidenceRange`` per localized
    instance; possibly empty when ``count > 0`` for absence smells). They
    were previously only kept on the final detection — keeping them per
    iteration lets the downstream analysis layer measure how localization
    evolves across the A1↔A2 feedback loop. ``raw_response`` keeps the full
    detection response verbatim so no per-iteration signal is dropped before
    the analysis layer sees it.
    """

    model_config = ConfigDict(extra="forbid")

    smell_detected: bool
    detection_justification: str
    count: int = 0
    evidence_ranges: list[EvidenceRange] = Field(default_factory=list)
    raw_response: str = ""
    # Provider's native reasoning/thinking stream for this call (the OpenRouter
    # "Reasoning" channel), distinct from the contract's in-answer CoT inside
    # raw_response and from the reasoning_tokens count. None when the model/provider
    # returned no reasoning text. Surfaces only in the per-iteration files.
    reasoning_content: str | None = None
    metrics: CallMetrics = Field(default_factory=CallMetrics)


class DetectionValidationPhaseEntry(BaseModel):
    """A2's per-iteration validator output, with the full per-call record.

    ``validated_count`` and ``validated_evidence_ranges`` carry the validator's
    own (possibly corrected) localization for this iteration — the operands a
    detection-loop predicate compares against the detection. Keeping them per
    iteration (not just on the converged result) lets the downstream analysis
    layer recompute count-match / evidence consistency post-hoc. ``raw_response`` keeps
    the full validator response verbatim so no per-iteration signal is dropped.
    """

    model_config = ConfigDict(extra="forbid")

    agrees: bool
    validated_smell_detected: bool
    validated_count: int = 0
    validated_evidence_ranges: list[EvidenceRange] = Field(default_factory=list)
    disagreement_feedback: str | None = None
    raw_response: str = ""
    # Provider's native reasoning/thinking stream for this call (the OpenRouter
    # "Reasoning" channel), distinct from the contract's in-answer CoT inside
    # raw_response and from the reasoning_tokens count. None when the model/provider
    # returned no reasoning text. Surfaces only in the per-iteration files.
    reasoning_content: str | None = None
    metrics: CallMetrics = Field(default_factory=CallMetrics)


class DetectionIterationLogEntry(BaseModel):
    """Per-iteration verdict for the A1 ↔ A2 detection loop."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    detection: DetectionPhaseEntry
    validation: DetectionValidationPhaseEntry


class RefactorPhaseEntry(BaseModel):
    """A3's per-iteration refactor output. Patched YAML stored elsewhere; here we
    keep the refactor agent's ``justification`` plus just enough to align the
    per-call record with the per-iter file artefacts."""

    model_config = ConfigDict(extra="forbid")

    patched_chars: int
    patched_sha256: str
    justification: str = ""
    # Provider's native reasoning/thinking stream for this refactor call; see
    # DetectionPhaseEntry.reasoning_content. None when no reasoning text returned.
    reasoning_content: str | None = None
    metrics: CallMetrics = Field(default_factory=CallMetrics)


class RefactorValidationPhaseEntry(BaseModel):
    """A4's per-iteration refactor-validator output.

    Field names mirror :class:`gha_agents.agents.RefactorValidationResult`:
    ``agrees`` replaces the previous ``smell_resolved``, and
    ``disagreement_feedback`` (nullable) replaces the dual-purpose
    ``justification`` — it is ``None`` on acceptance and carries the
    validator's actionable rejection text otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    agrees: bool
    remaining_count: int
    disagreement_feedback: str | None = None
    raw_response: str = ""
    # Provider's native reasoning/thinking stream for this call (the OpenRouter
    # "Reasoning" channel), distinct from the contract's in-answer CoT inside
    # raw_response and from the reasoning_tokens count. None when the model/provider
    # returned no reasoning text. Surfaces only in the per-iteration files.
    reasoning_content: str | None = None
    metrics: CallMetrics = Field(default_factory=CallMetrics)


class RefactorIterationLogEntry(BaseModel):
    """Per-iteration verdict for the A3 ↔ A4 refactor loop."""

    model_config = ConfigDict(extra="forbid")

    iteration: int
    refactor: RefactorPhaseEntry
    validation: RefactorValidationPhaseEntry


class WorkflowState(BaseModel):
    """Mutable per-run state populated by D1/D2 step executors.

    Holds a superset of fields needed by every variant; unused fields stay at
    their defaults so ``model_dump()`` matches the legacy dict literal each
    builder used to return.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    detection_loop_converged: bool = False
    detection_loop_iterations: int = 0
    detection_loop_iterations_log: list[DetectionIterationLogEntry] = Field(default_factory=list)
    refactor_loop_converged: bool | None = None
    refactor_loop_iterations: int = 0
    refactor_loop_iterations_log: list[RefactorIterationLogEntry] = Field(default_factory=list)
    refactor_patches_per_iteration: list[str] = Field(default_factory=list)
    last_refactor_validation: RefactorValidationResult | None = None
    last_patched: str | None = None
    final_detection: DetectionContract | None = None
    combined_result: CombinedResult | None = None
    token_usage: RoleTokenUsage = Field(default_factory=RoleTokenUsage)
    # Audit-trail buffer: per-call :class:`~gha_agents.trace.AgentCall`
    # records, populated only when ``execution.persist_traces`` is on. Typed
    # directly now that ``CallMetrics`` lives in ``trace`` (state → trace, one way).
    trace_calls: list[AgentCall] = Field(default_factory=list)
