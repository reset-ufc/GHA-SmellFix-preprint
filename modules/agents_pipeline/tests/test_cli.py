"""Tests for ``gha-smellfix`` (gha_agents.cli): one design on one workflow.

The agents are fakes (the pattern of test_characterization_snapshot.py), so no
test calls a model. The fakes record, at build time, the model id and provider
the run bound, and every user turn, to show that the command runs the paper's
settings on the given inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import gha_agents.teams as teams
import pytest
from gha_agents import cli
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from gha_agents.config import expand_runs, load_config
from gha_agents.env import _resolve_provider, get_model_id
from gha_agents.invoker import AgentCallResult, AgentInvoker
from gha_agents.state import CallMetrics
from typer.testing import CliRunner

_CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
_YAML = (
    "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
    "    steps:\n      - run: make test\n"
)
_PATCHED = (
    "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
    "    timeout-minutes: 30\n    steps:\n      - run: make test\n"
)
_R = "Walked each job against the smell definition."


@pytest.fixture
def calls(monkeypatch) -> list[dict]:
    """Fake invoker: runs the fake agent and records each user turn."""
    seen: list[dict] = []

    def fake_invoke(self, role, agent, user_turn):
        seen.append({"role": role, "user_turn": user_turn})
        response = agent.run(user_turn)
        return AgentCallResult(
            content=response.content,
            metrics=CallMetrics(input_tokens=100, output_tokens=20),
            started_at="2026-01-01T00:00:00+00:00",
            duration_ms=0,
        )

    monkeypatch.setattr(AgentInvoker, "invoke", fake_invoke)
    monkeypatch.setattr(teams, "is_persist_traces_active", lambda: False)
    return seen


def _agent(content, built: list[tuple[str, str]], role: str) -> MagicMock:
    built.append((get_model_id(role), _resolve_provider(get_model_id(role))))
    agent = MagicMock()
    agent.run.return_value = MagicMock(content=content)
    return agent


def _workflow(tmp_path: Path, text: str = _YAML) -> Path:
    path = tmp_path / "ci.yml"
    path.write_text(text, encoding="utf-8")
    return path


def _invoke(*args: str):
    return CliRunner().invoke(cli.app, list(args))


def test_paper_models_are_the_ones_of_the_experiment_runs():
    """Every run of every config resolves to the same settings through the CLI."""
    aliases: set[str] = set()
    for path in _CONFIGS:
        for spec in expand_runs(load_config(path)):
            aliases.add(spec.model_alias)
            mine = cli.resolve_run(spec.design, spec.model_alias)
            assert (mine.agents, mine.provider, mine.model_map) == (
                spec.agents,
                spec.provider,
                spec.model_map,
            ), path.name
            assert (mine.sampling, mine.thinking, mine.json_mode) == (
                spec.sampling,
                spec.thinking,
                spec.json_mode,
            ), path.name
    assert aliases == set(cli.PAPER_MODELS)


def test_d1_present_prints_the_verdict_and_writes_the_patch(tmp_path, monkeypatch, calls):
    built: list[tuple[str, str]] = []
    combined = CombinedResult(
        reasoning=_R,
        target_smell=ValidSmells.MissingJobTimeout,
        smell_detected=True,
        detection_justification="The build job sets no timeout-minutes.",
        detection_confidence="high",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        patched_workflow=_PATCHED,
        refactor_justification="Added timeout-minutes: 30 to the build job.",
        refactor_confidence="high",
    )
    monkeypatch.setattr(
        teams, "make_combined_agent", lambda smell: _agent(combined, built, "combined")
    )
    workflow = _workflow(tmp_path)
    out, result_json = tmp_path / "out" / "ci.patched.yml", tmp_path / "result.json"

    result = _invoke(
        str(workflow),
        "--smell", "MissingJobTimeout",
        "--design", "D1",
        "--model", "haiku45",
        "--repository", "octo/app",
        "--default-branch", "main",
        "--file-path", ".github/workflows/ci.yml",
        "--out", str(out),
        "--json", str(result_json),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert built == [("claude-haiku-4-5", "anthropic")]
    assert [c["role"] for c in calls] == ["combined"]
    turn = calls[0]["user_turn"]
    assert "<workflow>" in turn
    assert "- current_repository: octo/app" in turn
    assert "- repository_owner: octo" in turn
    assert "- default_branch: main" in turn
    assert "- workflow_file_path: .github/workflows/ci.yml" in turn
    assert "Verdict: present (1 instance(s); lines 4-6; confidence high)" in result.output
    assert "+    timeout-minutes: 30" in result.output
    assert out.read_text(encoding="utf-8") == _PATCHED

    record = json.loads(result_json.read_text(encoding="utf-8"))
    assert record["workflow"] == str(workflow)
    assert record["design"] == "D1"
    assert record["model_alias"] == "haiku45"
    assert record["model_ids"] == {"combined": "claude-haiku-4-5"}
    assert record["sampling_settings"] == {"top_p": 0.95}
    assert record["target_smell"] == "MissingJobTimeout"
    assert record["smell_detected"] is True
    assert record["patch_valid"] is True
    assert record["patched_workflow"] == _PATCHED
    assert set(record["prompt_hashes"]) == {"combined"}


def test_d2_absent_returns_the_workflow_unchanged(tmp_path, monkeypatch, calls):
    built: list[tuple[str, str]] = []
    detection = DetectionContract(
        reasoning=_R,
        target_smell=ValidSmells.MissingJobTimeout,
        smell_detected=False,
        detection_justification="Every job sets timeout-minutes.",
        count=0,
    )
    refactor = MagicMock()
    monkeypatch.setattr(
        teams,
        "make_detection_agent",
        lambda smell, scenario: _agent(detection, built, "detection"),
    )
    monkeypatch.setattr(teams, "make_refactor_agent", lambda smell, scenario: refactor)
    workflow = _workflow(tmp_path)
    out = tmp_path / "ci.patched.yml"

    result = _invoke(
        str(workflow),
        "--smell", "MissingJobTimeout",
        "--design", "D2",
        "--model", "gpt54-nano",
        "--out", str(out),
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert built == [("gpt-5.4-nano", "openai")]
    assert [c["role"] for c in calls] == ["detection"]
    assert "<repository_context>" not in calls[0]["user_turn"]
    refactor.run.assert_not_called()
    assert "Verdict: absent" in result.output
    assert "The workflow comes back unchanged." in result.output
    assert out.read_text(encoding="utf-8") == _YAML


def test_d2_present_runs_the_refactoring_agent(tmp_path, monkeypatch, calls):
    built: list[tuple[str, str]] = []
    detection = DetectionContract(
        reasoning=_R,
        target_smell=ValidSmells.MissingJobTimeout,
        smell_detected=True,
        detection_justification="The build job sets no timeout-minutes.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=4)],
    )
    refactor = RefactorResult(
        reasoning=_R,
        patched_workflow=_PATCHED,
        refactor_justification="Added timeout-minutes: 30 to the build job.",
    )
    monkeypatch.setattr(
        teams,
        "make_detection_agent",
        lambda smell, scenario: _agent(detection, built, "detection"),
    )
    monkeypatch.setattr(
        teams,
        "make_refactor_agent",
        lambda smell, scenario: _agent(refactor, built, "refactor"),
    )

    result = _invoke(
        str(_workflow(tmp_path)),
        "--smell", "MissingJobTimeout",
        "--design", "D2",
        "--model", "deepseek-v4-flash",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    expected = ("deepseek/deepseek-v4-flash", "openrouter")
    assert built == [expected, expected]
    assert [c["role"] for c in calls] == ["detection", "refactor"]
    assert "<detection_result>" in calls[1]["user_turn"]
    assert "Verdict: present (1 instance(s); lines 4)" in result.output
    assert "Refactoring: Added timeout-minutes: 30 to the build job." in result.output


def test_rejects_a_file_that_is_not_a_workflow(tmp_path):
    result = _invoke(
        str(_workflow(tmp_path, "- just\n- a list\n")),
        "--smell", "MissingJobTimeout",
        "--design", "D1",
        "--model", "haiku45",
    )  # fmt: skip

    assert result.exit_code == 2
    assert "is not a GitHub Actions workflow" in result.output


def test_rejects_an_unknown_smell(tmp_path):
    result = _invoke(
        str(_workflow(tmp_path)),
        "--smell", "NoSuchSmell",
        "--design", "D1",
        "--model", "haiku45",
    )  # fmt: skip

    assert result.exit_code == 2
    assert "MissingJobTimeout" in result.output


def test_reports_a_failed_run(tmp_path, monkeypatch, calls):
    def no_key(smell):
        message = "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        raise OSError(message)

    monkeypatch.setattr(teams, "make_combined_agent", no_key)

    result = _invoke(
        str(_workflow(tmp_path)),
        "--smell", "MissingJobTimeout",
        "--design", "D1",
        "--model", "haiku45",
    )  # fmt: skip

    assert result.exit_code == 1
    assert "The run failed (OSError): ANTHROPIC_API_KEY is not set." in result.output
