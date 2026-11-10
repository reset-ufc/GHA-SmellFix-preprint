"""Characterization snapshots of the per-run result dict + WorkflowState dump (#31).

Pins the CURRENT, observable output shape of the D1/D2 runners BEFORE any
typed-record extraction (#43+), so later refactors can prove byte-equivalence
against a committed golden instead of trusting that "nothing changed".

What is pinned, per scenario, is exactly what ``main._run_standard`` returns —
the assembled result dict — plus ``WorkflowState.model_dump(mode="json")``. The
workflows are driven end-to-end through Agno's ``workflow.run`` with the existing
content-fake pattern: stub agents whose ``.run()`` returns a fixed typed result,
and a deterministic ``AgentInvoker.invoke`` (``duration_ms=0``) so the captured
``CallMetrics`` are all-zero and the snapshot is stable. No LLM, no network.

Beyond the two required configs the file also snapshots the byte-identical
passthrough path (D2 absent) — the exact behavior the HITL slices (#41
detection-contract collapse, #42 strategy de-orchestration) must preserve
unchanged.

Regenerate goldens after an *intended* shape change:
    study_UPDATE_SNAPSHOTS=1 uv run pytest \
        modules/agents_pipeline/tests/test_characterization_snapshot.py
Review the resulting diff under tests/snapshots/ before committing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import gha_agents.main as main
import gha_agents.teams as teams
import pytest
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from gha_agents.invoker import AgentCallResult, AgentInvoker
from gha_agents.state import CallMetrics

_SMELL = next(iter(ValidSmells))
_YAML = "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n"
_R = "Walked each job and step against the smell definition before deciding."

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
_UPDATE = os.environ.get("study_UPDATE_SNAPSHOTS") == "1"


# ─── Deterministic runtime substitution ──────────────────────────────────────


def _fake_invoke(self, role: str, agent, user_turn: str) -> AgentCallResult:
    """Stand-in for ``AgentInvoker.invoke`` that skips the framework ritual.

    Still calls ``agent.run(user_turn)`` (so prompt assembly runs for real) but
    pins ``duration_ms=0`` and a fixed timestamp, so every captured
    ``CallMetrics`` is all-zero and the snapshot is reproducible.
    """
    response = agent.run(user_turn)
    metrics = CallMetrics.from_agno_response(response, duration_ms=0)
    raw_reasoning = getattr(response, "reasoning_content", None)
    return AgentCallResult(
        content=getattr(response, "content", None),
        metrics=metrics,
        started_at="2026-01-01T00:00:00+00:00",
        duration_ms=0,
        # MagicMock responses auto-create attributes; keep only real reasoning text
        # so the snapshot records a deterministic null, not a mock repr.
        reasoning_content=raw_reasoning if isinstance(raw_reasoning, str) else None,
    )


@pytest.fixture(autouse=True)
def _deterministic_runtime(monkeypatch):
    monkeypatch.setattr(AgentInvoker, "invoke", _fake_invoke)
    # Keep the trace buffer empty regardless of the developer's environment.
    monkeypatch.setattr(teams, "is_persist_traces_active", lambda: False)


def _mock_agent(contents: list) -> MagicMock:
    agent = MagicMock()
    agent.run.side_effect = [MagicMock(content=c, metrics=None, status=None) for c in contents]
    return agent


def _assert_snapshot(name: str, payload: dict) -> None:
    """Compare ``payload`` to the committed golden, or (re)write it on update."""
    # Round-trip through JSON so the comparison is over plain JSON types only
    # (this is exactly what ``save_result`` persists for the result dict).
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    path = _SNAPSHOT_DIR / f"{name}.json"
    if _UPDATE:
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", "utf-8")
        return
    assert path.exists(), (
        f"golden snapshot '{name}' missing — regenerate with "
        f"study_UPDATE_SNAPSHOTS=1 and review the diff before committing"
    )
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert normalized == expected


def _capture(result: dict, state) -> dict:
    return {"result": result, "state": state.model_dump(mode="json")}


# ─── D1 — single combined agent ──────────────────────────────────────────────


def test_snapshot_c1_present(monkeypatch):
    combined = CombinedResult(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="Missing timeout-minutes on the build job.",
        detection_confidence="high",
        detection_confidence_justification="The job clearly defines no job-level timeout.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        patched_workflow="name: CI\njobs:\n  build:\n    timeout-minutes: 30\n",
        refactor_justification="Added timeout-minutes: 30 to the build job.",
        refactor_confidence="high",
        refactor_confidence_justification="The edit is localized and behavior-preserving.",
    )
    monkeypatch.setattr(teams, "make_combined_agent", lambda smell: _mock_agent([combined]))

    result, state = main._run_standard(_YAML, teams.D1Workflow, _SMELL)

    _assert_snapshot("d1_present", _capture(result, state))


# ─── D2 — detection + conditional refactor ───────────────────────────────────


def _patch_c2(monkeypatch, *, det_agent, rep_agent) -> None:
    monkeypatch.setattr(teams, "make_detection_agent", lambda smell, scenario: det_agent)
    monkeypatch.setattr(teams, "make_refactor_agent", lambda smell, scenario: rep_agent)


def test_snapshot_c2_present(monkeypatch):
    det = DetectionContract(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="Unscoped GITHUB_TOKEN at the workflow level.",
        detection_confidence="medium",
        detection_confidence_justification="Token use is implied by the checkout step.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=12, end_line=18)],
    )
    rep = RefactorResult(
        reasoning=_R,
        patched_workflow="name: CI\npermissions:\n  contents: read\n",
        refactor_justification="Scoped GITHUB_TOKEN to contents: read.",
        refactor_confidence="high",
        refactor_confidence_justification="Adding a permissions block is a safe, localized edit.",
    )
    _patch_c2(monkeypatch, det_agent=_mock_agent([det]), rep_agent=_mock_agent([rep]))

    result, state = main._run_standard(_YAML, teams.D2Workflow, _SMELL)

    _assert_snapshot("d2_present", _capture(result, state))


def test_snapshot_c2_absent_is_byte_identical_passthrough(monkeypatch):
    det = DetectionContract(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=False,
        detection_justification="No matching smell instance found.",
        count=0,
        evidence_ranges=[],
    )
    rep_agent = MagicMock()  # the gate must skip refactor entirely
    _patch_c2(monkeypatch, det_agent=_mock_agent([det]), rep_agent=rep_agent)

    result, state = main._run_standard(_YAML, teams.D2Workflow, _SMELL)

    rep_agent.run.assert_not_called()
    # The contract the collapse slice (#41) must preserve: absent → original YAML.
    assert result["patched_workflow"] == _YAML
    _assert_snapshot("d2_absent", _capture(result, state))
