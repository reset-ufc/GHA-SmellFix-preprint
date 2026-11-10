"""
teams.py — Two agentic workflow variants for Agents (binary per-smell).

Each workflow targets ONE smell per run (callers iterate over ValidSmells) and is
smell-specialized at construction: the target smell definition is interpolated into
each agent's system prompt via the build_*_prompt helpers, so user turns carry only
data wrapped in XML. The framework is call-substrate only, reached solely through
:class:`AgentInvoker` — there is no Agno Workflow/Step/Condition orchestration. Each
variant is a typed :class:`SmellWorkflow` Strategy whose ``run`` drives the agents in
plain Python (the D2 gate is an ``if``) and leaves the populated state on
``self.state``.

Variants, named after the paper's designs:
  D1Workflow(target_smell)        → D1: 1 combined detect+refactor agent
  D2Workflow(target_smell)        → D2: detection + presence-gated refactor

User turns are built exclusively via the typed ``*_BUILDER`` UserTurnBuilders in
``user_turn.py``; the XML tag names below match the ``<inputs>`` blocks each system
template advertises (XML wrapping is the Anthropic/OpenAI recommendation against
prompt injection):
  <workflow>                      Original YAML for the detection agent
  <original_workflow>             Original YAML for the refactor agent
  <detection_result>              Detection JSON (refactor agent input)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable

from agno.agent import Agent

from .agents import (
    CombinedResult,
    DetectionContract,
    RefactorResult,
    ValidSmells,
    make_combined_agent,
    make_detection_agent,
    make_refactor_agent,
)
from .env import is_persist_traces_active
from .invoker import AgentCallResult, AgentInvoker
from .prompts import Scenario
from .record import CombinedResultAssembler
from .state import (
    CallMetrics,
    RoleTokenUsage,
    WorkflowState,
)
from .trace import AgentCall
from .user_turn import (
    D2_REFACTOR_BUILDER,
    WORKFLOW_BUILDER,
)
from .utils import logger

# Default facade over the framework agent call. Loops and builders take an
# ``invoker`` parameter (defaulting to this) so tests can inject a fake and
# exercise the whole orchestration without the framework runtime.
_INVOKER = AgentInvoker()


def _empty_token_usage() -> RoleTokenUsage:
    """Return a fresh token-usage accumulator with per-role buckets.

    Only the roles that actually fired will carry non-zero values.
    """
    return RoleTokenUsage()


def _fold_metrics(bucket: RoleTokenUsage, call: CallMetrics, role: str) -> CallMetrics:
    """Fold the summable token counts of one :class:`CallMetrics` into ``bucket``
    (per-role + total) and count the call.

    The fold itself lives in :meth:`RoleTokenUsage.fold`; this thin wrapper
    delegates to it and keeps the per-call debug line (which reads the running
    totals) and the call-site contract of returning ``call`` unchanged. Capture
    only — no metric is derived here. A call with no token data still counts it.
    """
    bucket.fold(call, role)
    total = bucket.total
    logger.debug(
        "    [%s] tokens: in=%d out=%d reasoning=%d cache_r=%d (run total in=%d out=%d)",
        role,
        call.input_tokens,
        call.output_tokens,
        call.reasoning_tokens,
        call.cache_read_tokens,
        total.input_tokens,
        total.output_tokens,
    )
    return call


def _record_trace_call(
    state: WorkflowState | None,
    *,
    role: str,
    result: AgentCallResult,
    prompt: str,
    loop: str | None = None,
    iteration: int | None = None,
) -> None:
    """Append an :class:`AgentCall` to ``state.trace_calls`` when the audit
    trail is active. Silent no-op when ``persist_traces`` is off so the capture
    cost stays out of the default path. All per-call signal (metrics, native
    reasoning, tool calls, model id) is carried on the :class:`AgentCallResult`
    the invoker already produced."""
    if state is None or not is_persist_traces_active():
        return
    content = result.content
    if content is None:
        rendered = ""
    elif hasattr(content, "model_dump_json"):
        rendered = content.model_dump_json()
    else:
        rendered = str(content)
    state.trace_calls.append(
        AgentCall(
            loop=loop,
            iteration=iteration,
            role=role,
            model_id=result.model_id,
            model_provider=result.model_provider,
            system_prompt=result.system_prompt,
            user_prompt=prompt,
            response_content=rendered,
            reasoning_content=result.reasoning_content,
            tool_calls=list(result.tool_calls),
            metrics=result.metrics,
            started_at=result.started_at,
        )
    )


def _detection_payload(d: DetectionContract) -> str:
    """Serialize a detection result for the ``<detection_result>`` tag handed to
    the refactor and refactor-validation agents.

    Dumps the full structured decision — including ``evidence_ranges``, which
    ``REFACTOR_CORE_RULES`` instructs the refactor agent to anchor edits on. The
    detection contract no longer echoes the workflow YAML, so nothing has to be
    excluded to avoid duplicating it alongside the ``<original_workflow>`` tag.
    """
    return d.model_dump_json(indent=2)


def _summarise_detection(d: DetectionContract) -> str:
    if d.evidence_ranges:
        evidence = ",".join(f"L{r.start_line}-L{r.end_line}" for r in d.evidence_ranges)
    else:
        evidence = "no-lines"
    return f"smell_detected={d.smell_detected} count={d.count} evidence={evidence}"


def _truncate(text: str, n: int = 160) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


# ─── SmellWorkflow Strategy — D1 / D2 ─────────────────────────────────────────
# The framework no longer orchestrates: there is no Agno Workflow / Step /
# Condition. Each variant is a typed Strategy whose ``run`` drives the agents in
# plain Python — the D2 gate is an ``if`` — reaching the framework only through
# :class:`AgentInvoker`. ``run`` returns the CombinedResult and leaves the
# populated state on ``self.state``.


class AgentRegistry:
    """Thread-confined builder of smell-specialized agents.

    Yields one agent per ``(role, smell, scenario)``, built lazily in the calling
    thread and cached in thread-local storage so a worker reuses its own bundle
    and never shares an agent across the parallel matrix. Construction goes
    through the module-level ``make_*_agent`` factories (also the unit-test seam),
    so a fake agent monkeypatched onto this module is honoured.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _cache(self) -> dict[tuple, Agent]:
        cache = getattr(self._local, "agents", None)
        if cache is None:
            cache = self._local.agents = {}
        return cache

    def _get(self, key: tuple, build: Callable[[], Agent]) -> Agent:
        cache = self._cache()
        agent = cache.get(key)
        if agent is None:
            agent = cache[key] = build()
        return agent

    def detection_agent(self, smell: ValidSmells, scenario: Scenario) -> Agent:
        return self._get(
            ("detection", smell, scenario), lambda: make_detection_agent(smell, scenario)
        )

    def refactor_agent(self, smell: ValidSmells, scenario: Scenario) -> Agent:
        return self._get(
            ("refactor", smell, scenario), lambda: make_refactor_agent(smell, scenario)
        )

    def combined_agent(self, smell: ValidSmells) -> Agent:
        return self._get(("combined", smell, None), lambda: make_combined_agent(smell))


class SmellWorkflow(ABC):
    """Base Strategy for one (variant, target_smell) run.

    Subclasses build their own agents and implement :meth:`run`, accumulating
    per-call token metrics and (when tracing is on) the audit trail into
    ``self.state``. A workflow instance is single-use: one per (file, smell) run.
    Agents are sourced from an :class:`AgentRegistry` (a fresh one per workflow by
    default) so each run's bundle stays confined to its own thread.
    """

    def __init__(
        self,
        target_smell: ValidSmells,
        *,
        invoker: AgentInvoker = _INVOKER,
        registry: AgentRegistry | None = None,
    ) -> None:
        self.target_smell = target_smell
        self.invoker = invoker
        self.registry = registry or AgentRegistry()
        self.state = WorkflowState()

    @abstractmethod
    def run(self, yaml_content: str, repository_context: str | None = None) -> CombinedResult:
        """Drive the variant end-to-end and return its ``CombinedResult``.

        ``repository_context`` is the rendered ``<repository_context>`` block (the
        workflow's repository metadata), or ``None`` when the ground_truth entry carries
        none — in which case the block is omitted from every user turn.
        """


class D1Workflow(SmellWorkflow):
    """One Agent (D1) — single-pass detection + refactor in one combined call."""

    def run(self, yaml_content: str, repository_context: str | None = None) -> CombinedResult:
        agent = self.registry.combined_agent(self.target_smell)
        prompt = WORKFLOW_BUILDER.build(
            workflow=yaml_content, repository_context=repository_context
        )
        logger.debug(
            "D1 [%s] → combined agent (input=%d chars)", self.target_smell.value, len(prompt)
        )
        call_result = self.invoker.invoke("combined", agent, prompt)
        _fold_metrics(self.state.token_usage, call_result.metrics, "combined")
        _record_trace_call(self.state, role="combined", result=call_result, prompt=prompt)
        result: CombinedResult = call_result.content
        logger.debug(
            "D1 [%s] ← smell_detected=%s patched=%d chars — %s",
            self.target_smell.value,
            result.smell_detected,
            len(result.patched_workflow),
            _truncate(result.detection_justification),
        )
        return CombinedResultAssembler.from_d1(result)


class D2Workflow(SmellWorkflow):
    """Two Agents (D2) — detection, then a presence gate to a single refactor pass.

    Absent → the original workflow is returned verbatim (byte-identical
    passthrough). Present → the refactor agent produces the patch. Both branches
    populate ``self.state.combined_result``.
    """

    def run(self, yaml_content: str, repository_context: str | None = None) -> CombinedResult:
        detection_agent = self.registry.detection_agent(self.target_smell, "D2")
        prompt = WORKFLOW_BUILDER.build(
            workflow=yaml_content, repository_context=repository_context
        )
        logger.debug(
            "D2 [%s] → detection agent (input=%d chars)", self.target_smell.value, len(prompt)
        )
        det_result = self.invoker.invoke("detection", detection_agent, prompt)
        _fold_metrics(self.state.token_usage, det_result.metrics, "detection")
        _record_trace_call(self.state, role="detection", result=det_result, prompt=prompt)
        detection: DetectionContract = det_result.content
        self.state.final_detection = detection
        logger.debug(
            "D2 [%s] ← detection: %s — %s",
            self.target_smell.value,
            _summarise_detection(detection),
            _truncate(detection.detection_justification),
        )

        if not detection.smell_detected:
            logger.debug("D2 [%s] ⊘ smell absent — skipping refactor", self.target_smell.value)
            combined = CombinedResultAssembler.from_d2_absent(
                self.target_smell, detection, yaml_content
            )
            self.state.combined_result = combined
            return combined

        refactor_agent = self.registry.refactor_agent(self.target_smell, "D2")
        rep_prompt = D2_REFACTOR_BUILDER.build(
            original_workflow=yaml_content,
            repository_context=repository_context,
            detection_result=_detection_payload(detection),
        )
        logger.debug(
            "D2 [%s] → refactor agent (input=%d chars)", self.target_smell.value, len(rep_prompt)
        )
        rep_result = self.invoker.invoke("refactor", refactor_agent, rep_prompt)
        _fold_metrics(self.state.token_usage, rep_result.metrics, "refactor")
        _record_trace_call(self.state, role="refactor", result=rep_result, prompt=rep_prompt)
        refactor_res: RefactorResult = rep_result.content
        logger.debug(
            "D2 [%s] ← refactor returned patched_workflow (%d chars)",
            self.target_smell.value,
            len(refactor_res.patched_workflow),
        )

        combined = CombinedResultAssembler.from_d2_present(
            self.target_smell, detection, refactor_res
        )
        self.state.combined_result = combined
        return combined
