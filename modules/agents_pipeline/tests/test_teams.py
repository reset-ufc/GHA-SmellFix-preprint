"""Unit tests for teams.py — binary per-smell orchestration."""

from unittest.mock import MagicMock

import pytest
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from gha_agents.invoker import AgentCallResult, AgentInvoker
from gha_agents.state import CallMetrics, RoleTokenUsage
from gha_agents.teams import (
    D1Workflow,
    D2Workflow,
    _fold_metrics,
)

# ─── Test helpers ────────────────────────────────────────────────────────────

_SMELL = next(iter(ValidSmells))
_YAML = "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n"


def _mock_agent(contents: list) -> MagicMock:
    """Agent whose .run() returns successive mocked responses with .content.

    ``.metrics`` is explicitly set to ``None`` so :class:`FakeAgentInvoker`
    treats the call as "no metrics available" instead of introspecting a
    default MagicMock (which would crash on int() coercion).
    """
    agent = MagicMock()
    agent.run.side_effect = [MagicMock(content=c, metrics=None) for c in contents]
    return agent


class FakeAgentInvoker(AgentInvoker):
    """Test double for the framework-call facade.

    Still calls ``agent.run(prompt)`` so the loops and builders exercise their
    real prompt-assembly path (and ``call_args_list`` prompt assertions hold),
    but skips the tool-extraction / status-guard machinery the production
    invoker runs against real ``RunOutput`` objects — keeping the mock-agent
    responses (whose ``.tools`` would not iterate) usable as the substrate.
    """

    def invoke(self, role: str, agent, user_turn: str) -> AgentCallResult:
        response = agent.run(user_turn)
        metrics = CallMetrics.from_agno_response(response, duration_ms=0)
        raw_reasoning = getattr(response, "reasoning_content", None)
        return AgentCallResult(
            content=getattr(response, "content", None),
            metrics=metrics,
            started_at="2026-01-01T00:00:00+00:00",
            duration_ms=0,
            # MagicMock responses auto-create attributes; keep only real reasoning text
            # (production RunOutput yields str | None) so the str|None phase field holds.
            reasoning_content=raw_reasoning if isinstance(raw_reasoning, str) else None,
        )


_FAKE_INVOKER = FakeAgentInvoker()


def _detection(
    *,
    smell_detected: bool = True,
    justification: str = "stub",
    count: int | None = None,
    evidence_ranges: list[EvidenceRange] | None = None,
) -> DetectionContract:
    """Test-default helper for detection: count defaults to 1 / 0 from
    ``smell_detected`` so callers that don't care about the multi-instance schema
    stay terse, but callers that do can pass an explicit count + ranges to
    exercise multi-instance behavior or absence-smell shape (count > 0 with
    empty ranges)."""
    if count is None:
        count = 1 if smell_detected else 0
    if evidence_ranges is None:
        evidence_ranges = []
    return DetectionContract(
        reasoning="Walked each job and step against the smell definition before deciding.",
        target_smell=_SMELL,
        smell_detected=smell_detected,
        detection_justification=justification,
        count=count,
        evidence_ranges=evidence_ranges,
    )


def _refactor_result(
    patched: str = "patched: yaml", justification: str = "stub refactor rationale"
) -> RefactorResult:
    return RefactorResult(
        reasoning="Planned the minimal edit that removes the smell while preserving behavior.",
        patched_workflow=patched,
        refactor_justification=justification,
    )


# ─── _fold_metrics ───────────────────────────────────────────────────────────


def test_fold_metrics_folds_summable_counts_into_buckets():
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
    bucket = RoleTokenUsage()

    cm = _fold_metrics(bucket, call, "detection")

    assert cm is call
    # Summable token counts fold into the per-role and total buckets.
    assert bucket.by_role["detection"].input_tokens == 100
    assert bucket.by_role["detection"].reasoning_tokens == 25
    assert bucket.by_role["detection"].calls == 1
    assert bucket.total.reasoning_tokens == 25
    assert bucket.total.calls == 1


def test_fold_metrics_empty_call_still_counts_the_call():
    bucket = RoleTokenUsage()

    cm = _fold_metrics(bucket, CallMetrics(duration_ms=7), "refactor")

    assert isinstance(cm, CallMetrics)
    assert cm.input_tokens == 0
    assert bucket.by_role["refactor"].calls == 1
    assert bucket.total.calls == 1


# ─── D1Workflow ───────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_env(monkeypatch):
    import gha_agents.env as env_mod

    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    for role in env_mod._AGENT_ROLES:
        monkeypatch.delenv(f"AGENT_{role.upper()}_MODEL", raising=False)
    token = env_mod._model_override_var.set(dict.fromkeys(env_mod._AGENT_ROLES, "fake/model"))
    yield
    env_mod._model_override_var.reset(token)


def _combined(patched: str = "fixed: yaml") -> CombinedResult:
    return CombinedResult(
        reasoning="Reasoned through detection and the fix before emitting the result.",
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="Identified the smell.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
        patched_workflow=patched,
        refactor_justification="Applied the minimal fix.",
    )


def test_c1_workflow_runs_combined_agent_with_target_smell(fake_env, monkeypatch):
    combined = CombinedResult(
        reasoning="Detected the missing timeout and reasoned through the one-line fix.",
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="Identified missing timeout on build job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        patched_workflow="fixed: yaml",
        refactor_justification="Added timeout-minutes: 30 to the build job.",
    )
    agent = _mock_agent([combined])
    monkeypatch.setattr("gha_agents.teams.make_combined_agent", lambda smell: agent)

    workflow = D1Workflow(_SMELL, invoker=_FAKE_INVOKER)
    result = workflow.run(_YAML)

    assert agent.run.call_count == 1
    prompt = agent.run.call_args_list[0].args[0]
    assert "<workflow>" in prompt
    assert _YAML in prompt
    # Smell metadata is injected into the system prompt at agent-construction time.
    assert "<target_smell>" not in prompt

    assert result.target_smell == _SMELL
    assert result.smell_detected is True
    assert result.patched_workflow == "fixed: yaml"
    assert result.detection_justification.startswith("Identified")
    # D1's combined agent emits a distinct refactoring rationale.
    assert result.refactor_justification.startswith("Added timeout-minutes")
    # D1 does not stash combined_result on state (parity with the legacy executor).
    assert workflow.state.combined_result is None


def test_c1_workflow_factory_receives_target_smell(fake_env, monkeypatch):
    seen_smells: list = []

    def fake_factory(smell):
        seen_smells.append(smell)
        return _mock_agent([_combined()])

    monkeypatch.setattr("gha_agents.teams.make_combined_agent", fake_factory)
    D1Workflow(_SMELL, invoker=_FAKE_INVOKER).run(_YAML)
    assert seen_smells == [_SMELL]


def test_run_forwards_repository_context_into_the_user_prompt(fake_env, monkeypatch):
    """repository_context handed to .run() reaches the agent's user turn verbatim,
    immediately after the workflow — the end-to-end wiring the ground_truth entry rides."""
    agent = _mock_agent([_combined()])
    monkeypatch.setattr("gha_agents.teams.make_combined_agent", lambda smell: agent)

    repo_ctx = "- current_repository: o/r\n- default_branch: main"
    D1Workflow(_SMELL, invoker=_FAKE_INVOKER).run(_YAML, repo_ctx)

    prompt = agent.run.call_args_list[0].args[0]
    assert f"<repository_context>\n{repo_ctx}\n</repository_context>" in prompt
    assert prompt.index("<repository_context>") > prompt.index("<workflow>")


def test_run_omits_repository_context_when_absent(fake_env, monkeypatch):
    """No repository_context → the block is omitted entirely (not rendered "None")."""
    agent = _mock_agent([_combined()])
    monkeypatch.setattr("gha_agents.teams.make_combined_agent", lambda smell: agent)

    D1Workflow(_SMELL, invoker=_FAKE_INVOKER).run(_YAML)
    assert "repository_context" not in agent.run.call_args_list[0].args[0]


# ─── D2Workflow ───────────────────────────────────────────────────────────────


def _patch_c2_factories(monkeypatch, *, det_agent, rep_agent):
    monkeypatch.setattr(
        "gha_agents.teams.make_detection_agent", lambda smell, scenario: det_agent
    )
    monkeypatch.setattr(
        "gha_agents.teams.make_refactor_agent", lambda smell, scenario: rep_agent
    )


def test_c2_workflow_short_circuits_when_smell_not_detected(fake_env, monkeypatch):
    not_detected = _detection(smell_detected=False, justification="no match")
    det_agent = _mock_agent([not_detected])
    rep_agent = MagicMock()  # the presence gate must skip refactor entirely
    _patch_c2_factories(monkeypatch, det_agent=det_agent, rep_agent=rep_agent)

    workflow = D2Workflow(_SMELL, invoker=_FAKE_INVOKER)
    result = workflow.run(_YAML)

    # Absent → original YAML returned verbatim, no refactor rationale.
    assert result.target_smell == _SMELL
    assert result.smell_detected is False
    assert result.patched_workflow == _YAML
    assert result.detection_justification == "no match"
    assert result.refactor_justification == ""
    # Both branches stash the CombinedResult on state.
    assert workflow.state.combined_result is result

    assert det_agent.run.call_count == 1
    rep_agent.run.assert_not_called()

    det_prompt = det_agent.run.call_args_list[0].args[0]
    assert "<workflow>" in det_prompt
    assert "<target_smell>" not in det_prompt


def test_c2_workflow_runs_refactor_when_smell_detected(fake_env, monkeypatch):
    detected = DetectionContract(
        reasoning="Read lines 12-18 against the smell definition in one shot before deciding.",
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="found it",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=12, end_line=18)],
    )
    refactored = _refactor_result(
        patched="fixed_c2: yaml",
        justification="Scoped GITHUB_TOKEN to contents: read.",
    )
    det_agent = _mock_agent([detected])
    rep_agent = _mock_agent([refactored])
    _patch_c2_factories(monkeypatch, det_agent=det_agent, rep_agent=rep_agent)

    result = D2Workflow(_SMELL, invoker=_FAKE_INVOKER).run(_YAML)

    # Refactor prompt composes <original_workflow> + <detection_result>; no
    # <target_smell> (it is in the system prompt), no <previous_answer> (no loop).
    rep_prompt = rep_agent.run.call_args_list[0].args[0]
    assert "<original_workflow>" in rep_prompt
    assert "<detection_result>" in rep_prompt
    assert "12" in rep_prompt and "18" in rep_prompt
    assert "<target_smell>" not in rep_prompt
    assert "<previous_answer>" not in rep_prompt

    assert result.target_smell == _SMELL
    assert result.smell_detected is True
    assert result.patched_workflow == "fixed_c2: yaml"
    assert result.detection_justification == "found it"
    # The refactor agent's own rationale is carried into the final CombinedResult.
    assert result.refactor_justification == "Scoped GITHUB_TOKEN to contents: read."


def test_c2_workflow_factories_receive_target_smell(fake_env, monkeypatch):
    seen_smells: list[tuple[str, object, str]] = []

    def fake_detect(smell, scenario):
        seen_smells.append(("detection", smell, scenario))
        return _mock_agent([_detection(smell_detected=True)])

    def fake_refactor(smell, scenario):
        seen_smells.append(("refactor", smell, scenario))
        return _mock_agent([_refactor_result()])

    monkeypatch.setattr("gha_agents.teams.make_detection_agent", fake_detect)
    monkeypatch.setattr("gha_agents.teams.make_refactor_agent", fake_refactor)

    # A present detection exercises both the detection and refactor factories.
    D2Workflow(_SMELL, invoker=_FAKE_INVOKER).run(_YAML)
    assert ("detection", _SMELL, "D2") in seen_smells
    assert ("refactor", _SMELL, "D2") in seen_smells


