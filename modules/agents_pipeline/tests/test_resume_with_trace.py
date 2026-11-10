"""Tests for resume semantics with persist_traces.

When persist_traces is on, resume must skip only when both result.json
and traces/{slug}.json exist. Otherwise the pair is re-run so the trace
is regenerated alongside the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gha_agents.agents import ValidSmells
from gha_agents.env import persist_traces
from gha_agents.main import _process_entry

_SMELL = next(iter(ValidSmells))


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


def _write_existing_artifacts(
    output_dir: Path,
    *,
    slug: str,
    write_result: bool,
    write_trace: bool,
) -> None:
    """Pre-seed the result + trace files inside ``output_dir`` for resume tests."""
    if write_result:
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / f"{slug}.json").write_text("{}", encoding="utf-8")
    if write_trace:
        traces_dir = output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        (traces_dir / f"{slug}.json").write_text("{}", encoding="utf-8")


def _slug_for(target: str, *, agents: int = 1, repetition: int = 1) -> str:
    """Mirror the slug format produced by main._build_slug for our fixture."""
    config_label = {1: "D1", 2: "D2"}[agents]
    return f"owner@repo@ci@{config_label}@rep{repetition}@{target.lower()}"


def _entry() -> dict:
    return {
        "file_key": "owner@repo@ci.yml",
        "yaml_content": "name: CI\non: [push]\n",
        "repository": "owner/repo",
        "file_path": ".github/workflows/ci.yml",
        "default_branch": "main",
        "present_smells": [{"pascal": _SMELL.name, "count": 1}],
        "absent_smells": [],
    }


def _process_with_resume(
    *,
    output_dir: Path,
    error_log: Path,
    persist_active: bool,
    echo_pairs: bool = True,
) -> dict | None:
    """Drive ``_process_entry`` with the persist_traces contextvar toggled.

    Returns the row produced by the entry — ``status="SKIP"`` means resume
    skipped, anything else means it executed (or short-circuited elsewhere).
    """
    with persist_traces(persist_active):
        return _process_entry(
            entry=_entry(),
            target_pascal=_SMELL.name,
            ground_truth_positive=True,
            agents=1,
            repetition=1,
            max_iterations=3,
            error_log=error_log,
            output_dir=output_dir,
            model_alias=None,
            resume=True,
            echo_pairs=echo_pairs,
        )


def test_resume_skips_when_result_exists_and_persist_traces_off(fake_env, tmp_path: Path):
    output_dir = tmp_path / "output"
    error_log = tmp_path / "error.log"
    slug = _slug_for(_SMELL.name)
    _write_existing_artifacts(output_dir, slug=slug, write_result=True, write_trace=False)
    row = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=False
    )
    assert row is not None
    assert row["status"] == "SKIP"


def test_resume_skip_echo_is_gated_by_echo_pairs(fake_env, tmp_path: Path, capsys):
    """Resume-skips print the ⏭ line in plain mode but are silenced under the
    live dashboard (echo_pairs=False) so fast skips don't flood and corrupt it."""
    output_dir = tmp_path / "output"
    error_log = tmp_path / "error.log"
    slug = _slug_for(_SMELL.name)
    _write_existing_artifacts(output_dir, slug=slug, write_result=True, write_trace=False)

    # Plain mode: the skip line is echoed.
    row = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=False
    )
    assert row is not None and row["status"] == "SKIP"
    assert "skipped (resume)" in capsys.readouterr().out

    # Progress mode: the ⏭ skip echo is suppressed (the row still skips and the
    # event is still logged to the run file — not asserted here).
    row2 = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=False, echo_pairs=False
    )
    assert row2 is not None and row2["status"] == "SKIP"
    captured = capsys.readouterr()
    assert "skipped (resume)" not in captured.out
    assert "⏭" not in captured.out


def test_resume_skips_when_both_result_and_trace_exist(fake_env, tmp_path: Path):
    output_dir = tmp_path / "output"
    error_log = tmp_path / "error.log"
    slug = _slug_for(_SMELL.name)
    _write_existing_artifacts(output_dir, slug=slug, write_result=True, write_trace=True)
    row = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=True
    )
    assert row is not None
    assert row["status"] == "SKIP"


def test_resume_reruns_when_trace_missing_and_persist_on(
    fake_env, tmp_path: Path, monkeypatch
):
    """persist_traces=true + result exists + trace missing → re-run.

    We don't run the agent for real (no API key); instead we monkeypatch
    ``_run_for_smell`` to detect that resume *did* call into the runner.
    """
    output_dir = tmp_path / "output"
    error_log = tmp_path / "error.log"
    slug = _slug_for(_SMELL.name)
    _write_existing_artifacts(output_dir, slug=slug, write_result=True, write_trace=False)

    called = {"count": 0}

    def fake_run_for_smell(**kwargs):
        called["count"] += 1
        raise RuntimeError("test sentinel — runner reached")  # noqa: TRY003

    monkeypatch.setattr("gha_agents.main._run_for_smell", fake_run_for_smell)

    row = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=True
    )
    assert called["count"] == 1
    # Runner raised → status is ERROR, not SKIP.
    assert row is not None
    assert row["status"] == "ERROR"


def test_resume_reruns_when_only_trace_exists_defensive(
    fake_env, tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "output"
    error_log = tmp_path / "error.log"
    slug = _slug_for(_SMELL.name)
    _write_existing_artifacts(output_dir, slug=slug, write_result=False, write_trace=True)

    called = {"count": 0}

    def fake_run_for_smell(**kwargs):
        called["count"] += 1
        raise RuntimeError("sentinel")

    monkeypatch.setattr("gha_agents.main._run_for_smell", fake_run_for_smell)
    row = _process_with_resume(
        output_dir=output_dir, error_log=error_log, persist_active=True
    )
    # Result missing → resume's existing-result short-circuit doesn't fire,
    # so the runner is invoked. (Trace alone never satisfies skip.)
    assert called["count"] == 1
    assert row is not None
    assert row["status"] == "ERROR"


def test_out_of_credit_propagates_instead_of_becoming_error_row(
    fake_env, tmp_path: Path, monkeypatch
):
    """Mid-run out-of-credit must propagate out of ``_process_entry`` so the
    dispatcher aborts the run fast — not be swallowed into a per-cell ERROR row."""
    from gha_agents.credit import OutOfCreditError

    def fake_run_for_smell(**kwargs):
        raise OutOfCreditError("balance exhausted")  # noqa: TRY003

    monkeypatch.setattr("gha_agents.main._run_for_smell", fake_run_for_smell)

    with pytest.raises(OutOfCreditError):
        _process_entry(
            entry=_entry(),
            target_pascal=_SMELL.name,
            ground_truth_positive=True,
            agents=1,
            repetition=1,
            max_iterations=3,
            error_log=tmp_path / "error.log",
            output_dir=tmp_path / "output",
            model_alias=None,
            resume=False,
        )
