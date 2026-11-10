"""Tests for the audit-trail / persist_traces opt-in path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from gha_agents.agents import (
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from gha_agents.env import persist_traces
from gha_agents.state import CallMetrics, WorkflowState
from gha_agents.store import RunStore
from gha_agents.teams import (
    D1Workflow,
    D2Workflow,
)
from gha_agents.trace import AgentCall, ToolCall, TraceFile

_SMELL = next(iter(ValidSmells))
_YAML = "name: CI\non: [push]\n"


def _detection(
    smell_detected: bool = True,
    justification: str = "ok",
    *,
    count: int | None = None,
    evidence_ranges: list[EvidenceRange] | None = None,
) -> DetectionContract:
    if count is None:
        count = 1 if smell_detected else 0
    if evidence_ranges is None:
        evidence_ranges = []
    return DetectionContract(
        reasoning="Walked the workflow against the smell definition step by step before deciding.",
        target_smell=_SMELL,
        smell_detected=smell_detected,
        detection_justification=justification,
        count=count,
        evidence_ranges=evidence_ranges,
    )


def _refactor(
    patched: str = "patched: yaml", justification: str = "stub refactor rationale"
) -> RefactorResult:
    return RefactorResult(
        reasoning="Reasoned through the minimal smell-removing edit.",
        patched_workflow=patched,
        refactor_justification=justification,
    )


def _agent_with_id(model_id: str, contents: list, metrics_value=None) -> MagicMock:
    """Build an Agno-shaped agent mock with a model.id and reasoning_content."""
    agent = MagicMock()
    agent.model.id = model_id
    # Real agents carry their rendered system prompt as a str on .instructions;
    # the invoker captures it into the trace's system_prompt.
    agent.instructions = f"system::{model_id}"
    agent.run.side_effect = [
        MagicMock(
            content=c,
            metrics=metrics_value,
            reasoning_content=None,
            tools=[],
            model_provider="",
        )
        for c in contents
    ]
    return agent


# ─── persist_traces contextvar ───────────────────────────────────────────────


@pytest.fixture()
def fake_env(monkeypatch):
    import gha_agents.env as env_mod

    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    token = env_mod._model_override_var.set(dict.fromkeys(env_mod._AGENT_ROLES, "fake/model"))
    yield
    env_mod._model_override_var.reset(token)


def _patch_c2(monkeypatch, *, det_agent, rep_agent) -> None:
    monkeypatch.setattr("gha_agents.teams.make_detection_agent", lambda smell, scenario: det_agent)
    monkeypatch.setattr("gha_agents.teams.make_refactor_agent", lambda smell, scenario: rep_agent)


def test_persist_traces_off_skips_trace_capture(fake_env, monkeypatch):
    det_agent = _agent_with_id("alpha/detector", [_detection()])
    rep_agent = _agent_with_id("alpha/refactor", [_refactor()])
    _patch_c2(monkeypatch, det_agent=det_agent, rep_agent=rep_agent)
    workflow = D2Workflow(target_smell=_SMELL)
    workflow.run(_YAML)
    # Off by default → empty buffer.
    assert workflow.state.trace_calls == []


def test_persist_traces_on_captures_c2_calls(fake_env, monkeypatch):
    det_agent = _agent_with_id("alpha/detector", [_detection()])
    rep_agent = _agent_with_id("alpha/refactor", [_refactor()])
    _patch_c2(monkeypatch, det_agent=det_agent, rep_agent=rep_agent)
    workflow = D2Workflow(target_smell=_SMELL)
    with persist_traces(True):
        workflow.run(_YAML)
    state = workflow.state
    roles = [call.role for call in state.trace_calls]
    assert roles == ["detection", "refactor"]
    det_call = state.trace_calls[0]
    assert isinstance(det_call, AgentCall)
    assert det_call.model_id == "alpha/detector"
    assert det_call.loop is None  # D2 has no feedback loop
    assert "<workflow>" in det_call.user_prompt
    assert det_call.system_prompt == "system::alpha/detector"


def test_persist_traces_captures_call_metrics_and_provider(fake_env, monkeypatch):
    """The audit trail carries the full CallMetrics record + model_provider per call."""
    det_metrics = SimpleNamespace(
        input_tokens=1500,
        output_tokens=300,
        reasoning_tokens=120,
        cache_read_tokens=900,
        cache_write_tokens=0,
        total_tokens=1800,
        time_to_first_token=0.74,
    )
    det_agent = MagicMock()
    det_agent.model.id = "anthropic/claude"
    det_agent.instructions = "system::anthropic/claude"
    det_agent.run.return_value = MagicMock(
        content=_detection(),
        metrics=det_metrics,
        reasoning_content="weighing the evidence",
        tools=[],
        model_provider="Anthropic",
    )
    rep_agent = _agent_with_id("anthropic/refactor", [_refactor()])
    _patch_c2(monkeypatch, det_agent=det_agent, rep_agent=rep_agent)
    workflow = D2Workflow(target_smell=_SMELL)
    with persist_traces(True):
        workflow.run(_YAML)
    det_call = workflow.state.trace_calls[0]
    assert isinstance(det_call.metrics, CallMetrics)
    assert det_call.metrics.input_tokens == 1500
    assert det_call.metrics.reasoning_tokens == 120
    assert det_call.metrics.total_tokens == 1800
    assert det_call.metrics.time_to_first_token == 0.74
    assert det_call.model_provider == "Anthropic"
    assert det_call.reasoning_content == "weighing the evidence"


def test_persist_traces_on_captures_c1_combined_call(monkeypatch):
    import gha_agents.env as env_mod
    from gha_agents.agents import CombinedResult

    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    token = env_mod._model_override_var.set(
        dict.fromkeys(env_mod._AGENT_ROLES, "gamma/combined")
    )

    combined = CombinedResult(
        reasoning="Detected the smell and reasoned through the fix in a single combined pass.",
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="found",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
        patched_workflow="fixed: yaml",
    )
    agent = _agent_with_id("gamma/combined", [combined])
    monkeypatch.setattr("gha_agents.teams.make_combined_agent", lambda smell: agent)

    workflow = D1Workflow(target_smell=_SMELL)
    with persist_traces(True):
        workflow.run(_YAML)
    state = workflow.state

    env_mod._model_override_var.reset(token)
    assert len(state.trace_calls) == 1
    assert state.trace_calls[0].role == "combined"
    assert state.trace_calls[0].loop is None  # D1 has no loop


# ─── RunStore.save_trace emits the JSON ──────────────────────────────────────


def test_persist_trace_file_writes_json_with_calls(tmp_path: Path):
    state = WorkflowState()
    state.trace_calls.append(
        AgentCall(
            loop="detection",
            iteration=1,
            role="detection",
            model_id="alpha/det",
            user_prompt="<workflow>...",
            response_content='{"smell_detected": true}',
            metrics=CallMetrics(input_tokens=100, total_tokens=130, duration_ms=900),
            tool_calls=[
                ToolCall(
                    name="enumerate_workflow_lines",
                    args={"workflow": "..."},
                    output_full="L1: name",
                ),
            ],
        )
    )
    path = RunStore(tmp_path).save_trace(
        slug="repo@file.yml@D2@gemma@rep1",
        design="D2",
        calls=state.trace_calls,
        error=None,
        persist_enabled=True,
    )
    assert path is not None and path.exists()
    parsed = TraceFile.model_validate_json(path.read_text())
    assert parsed.slug == "repo@file.yml@D2@gemma@rep1"
    assert parsed.design == "D2"
    assert parsed.error is None
    assert len(parsed.calls) == 1
    assert parsed.calls[0].tool_calls[0].name == "enumerate_workflow_lines"


def test_persist_trace_file_partial_trail_on_error(tmp_path: Path):
    state = WorkflowState()
    state.trace_calls.append(
        AgentCall(
            role="detection",
            model_id="alpha/det",
            user_prompt="<workflow>...",
            response_content='{"smell_detected": true}',
        )
    )
    err = RuntimeError("model 502 upstream error")
    path = RunStore(tmp_path).save_trace(
        slug="slug",
        design="D2",
        calls=state.trace_calls,
        error=err,
        persist_enabled=True,
    )
    assert path is not None
    parsed = TraceFile.model_validate_json(path.read_text())
    assert parsed.error is not None
    assert parsed.error.type == "RuntimeError"
    assert parsed.error.message == "model 502 upstream error"
    # raised_at_call_index points just past the calls captured before failure.
    assert parsed.error.raised_at_call_index == 1
    assert len(parsed.calls) == 1


def test_persist_trace_file_creates_traces_dir(tmp_path: Path):
    state = WorkflowState()
    out_dir = tmp_path / "deep" / "traces"
    assert not out_dir.exists()
    RunStore(tmp_path / "deep").save_trace(
        slug="x",
        design="D1",
        calls=state.trace_calls,
        error=None,
        persist_enabled=True,
    )
    assert out_dir.is_dir()


# ─── Persisted trace serializes via load → roundtrip ─────────────────────────


def test_trace_file_roundtrips_through_disk(tmp_path: Path):
    state = WorkflowState()
    state.trace_calls.extend(
        [
            AgentCall(
                role="detection",
                model_id="alpha/det",
                user_prompt="p1",
                response_content="r1",
            ),
            AgentCall(
                role="detection_validation",
                model_id="alpha/val",
                user_prompt="p2",
                response_content="r2",
            ),
        ]
    )
    path = RunStore(tmp_path).save_trace(
        slug="s",
        design="D2",
        calls=state.trace_calls,
        error=None,
        persist_enabled=True,
    )
    assert path is not None
    raw = json.loads(path.read_text())
    assert raw["slug"] == "s"
    assert raw["design"] == "D2"
    assert [c["role"] for c in raw["calls"]] == [
        "detection",
        "detection_validation",
    ]


# ─── save_trace hybrid policy: failure always, success flag-gated ─────────────


def test_failure_writes_trace_even_when_persist_disabled(tmp_path: Path):
    path = RunStore(tmp_path).save_trace(
        slug="s",
        design="D2",
        calls=[],
        error=RuntimeError("boom upstream"),
        persist_enabled=False,
    )
    assert path is not None and path.exists()
    parsed = TraceFile.model_validate_json(path.read_text())
    assert parsed.error is not None
    assert parsed.error.type == "RuntimeError"
    assert parsed.error.message == "boom upstream"


def test_success_skips_trace_when_persist_disabled(tmp_path: Path):
    traces_dir = tmp_path / "traces"
    path = RunStore(tmp_path).save_trace(
        slug="s", design="D2", calls=[], error=None, persist_enabled=False
    )
    assert path is None
    assert not traces_dir.exists()  # nothing written, dir untouched


def test_success_writes_trace_when_persist_enabled(tmp_path: Path):
    path = RunStore(tmp_path).save_trace(
        slug="s", design="D2", calls=[], error=None, persist_enabled=True
    )
    assert path is not None and path.exists()
    assert TraceFile.model_validate_json(path.read_text()).error is None


def test_persist_preserves_native_reasoning_content(tmp_path: Path):
    call = AgentCall(
        role="detection",
        model_id="m",
        user_prompt="p",
        response_content="r",
        reasoning_content="weighed the evidence step by step",
    )
    path = RunStore(tmp_path).save_trace(
        slug="s", design="D1", calls=[call], error=None, persist_enabled=True
    )
    assert path is not None
    parsed = TraceFile.model_validate_json(path.read_text())
    assert parsed.calls[0].reasoning_content == "weighed the evidence step by step"
