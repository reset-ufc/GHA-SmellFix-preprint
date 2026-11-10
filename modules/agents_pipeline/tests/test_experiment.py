"""Unit tests for the experiment dispatcher (gha_agents.experiment)."""

from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path
from textwrap import dedent

import gha_agents.experiment as exp_mod
import pytest
import typer
from gha_agents.config import ResolvedRun
from typer.testing import CliRunner

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_spec(
    tmp_path: Path,
    *,
    name: str = "r1",
    design: str = "D1",
    agents: int = 1,
    model_alias: str = "m",
    model_map: dict | None = None,
    provider: str = "openrouter",
    limit: int | None = None,
    repetition: int = 1,
) -> ResolvedRun:
    return ResolvedRun(
        name=name,
        design=design,  # type: ignore[arg-type]
        agents=agents,
        model_alias=model_alias,
        model_map=model_map or {"detection": "x/y"},
        provider=provider,  # type: ignore[arg-type]
        ground_truth_file=tmp_path / "ground_truth.json",
        output_dir=tmp_path / "output",
        limit=limit,
        repetition=repetition,
        max_iterations=3,
        verbose=False,
    )


def _write_ground_truth(path: Path, n: int = 2) -> None:
    entries = [
        {
            "file_key": f"repo@wf{i}.yml",
            "yaml_content": "name: wf\non: push\njobs: {}\n",
            "target_smell": {"pascal": "HardCodedSecrets", "paper_id": "S1"},
            "repository": "owner/repo",
            "file_path": f".github/workflows/wf{i}.yml",
        }
        for i in range(n)
    ]
    path.write_text(json.dumps(entries), encoding="utf-8")


# ─── run_one_experiment ───────────────────────────────────────────────────────


def test_run_one_experiment_truncates_to_limit(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=5)
    spec = _make_spec(tmp_path, limit=2)

    captured: dict = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return [{"file": "x", "smells": "1", "solved": "—", "status": "OK"}]

    monkeypatch.setattr(exp_mod, "run_pipeline", fake_run_pipeline)

    rows = exp_mod.run_one_experiment(spec, resume=False)

    assert len(rows) == 1
    assert len(captured["entries"]) == 2
    assert captured["agents"] == 1
    assert captured["model_alias"] == "m"
    assert captured["label"] == "r1"
    assert captured["resume"] is False


def test_run_one_experiment_applies_model_overrides(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    spec = _make_spec(tmp_path, model_map={"detection": "acme/foo", "refactor": "acme/bar"})

    import gha_agents.env as env_mod

    seen_override: dict = {}

    def fake_run_pipeline(**_):
        # Snapshot the override visible inside the worker.
        seen_override["val"] = env_mod._model_override_var.get()
        return []

    monkeypatch.setattr(exp_mod, "run_pipeline", fake_run_pipeline)

    exp_mod.run_one_experiment(spec, resume=True)

    assert seen_override["val"] == {"detection": "acme/foo", "refactor": "acme/bar"}
    # Override cleared after context manager exits.
    assert env_mod._model_override_var.get() is None


def test_run_one_experiment_pins_provider_override(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    spec = _make_spec(tmp_path, provider="anthropic")

    import gha_agents.env as env_mod

    seen: dict = {}

    def fake_run_pipeline(**_):
        # Snapshot the provider visible inside the worker.
        seen["val"] = env_mod._provider_override_var.get()
        return []

    monkeypatch.setattr(exp_mod, "run_pipeline", fake_run_pipeline)

    exp_mod.run_one_experiment(spec, resume=False)

    assert seen["val"] == "anthropic"
    # Override cleared after context manager exits.
    assert env_mod._provider_override_var.get() is None


def test_run_one_experiment_applies_request_timeout(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    spec = _make_spec(tmp_path)

    import gha_agents.env as env_mod

    seen_timeout: dict = {}

    def fake_run_pipeline(**_):
        seen_timeout["val"] = env_mod._timeout_override_var.get()
        return []

    monkeypatch.setattr(exp_mod, "run_pipeline", fake_run_pipeline)

    exp_mod.run_one_experiment(spec, resume=False, request_timeout=45.0)

    assert seen_timeout["val"] == 45.0
    # Override cleared after context manager exits.
    assert env_mod._timeout_override_var.get() is None


def test_run_one_experiment_empty_ground_truth_returns_empty(tmp_path: Path, monkeypatch):
    (tmp_path / "ground_truth.json").write_text("[]", encoding="utf-8")
    spec = _make_spec(tmp_path)

    monkeypatch.setattr(exp_mod, "run_pipeline", lambda **_: [{"unreachable": True}])
    assert exp_mod.run_one_experiment(spec, resume=False) == []


# ─── dispatch: sequential ─────────────────────────────────────────────────────


def test_dispatch_sequential_runs_all(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    specs = [_make_spec(tmp_path, name=f"r{i}") for i in range(3)]

    executed: list[str] = []

    def fake_runner(
        spec,
        *,
        resume,
        sampling=None,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        executed.append(spec.name)
        return [{"file": spec.name, "smells": "1", "solved": "—", "status": "OK"}]

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)

    rows = exp_mod.dispatch(specs, max_workers=1, resume=False, stop_on_error=False)

    assert executed == ["r0", "r1", "r2"]
    assert len(rows) == 3


def test_dispatch_sequential_continues_on_failure(tmp_path: Path, monkeypatch):
    specs = [_make_spec(tmp_path, name=f"r{i}") for i in range(3)]

    def fake_runner(
        spec,
        *,
        resume,
        sampling=None,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        if spec.name == "r1":
            raise RuntimeError("boom")
        return [{"file": spec.name, "smells": "1", "solved": "—", "status": "OK"}]

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)

    rows = exp_mod.dispatch(specs, max_workers=1, resume=False, stop_on_error=False)

    # r0 and r2 succeed; r1 fails silently.
    assert {r["file"] for r in rows} == {"r0", "r2"}


def test_dispatch_sequential_stops_on_error(tmp_path: Path, monkeypatch):
    specs = [_make_spec(tmp_path, name=f"r{i}") for i in range(3)]

    executed: list[str] = []

    def fake_runner(
        spec,
        *,
        resume,
        sampling=None,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        executed.append(spec.name)
        if spec.name == "r1":
            raise RuntimeError("boom")
        return []

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)

    with pytest.raises(RuntimeError, match="boom"):
        exp_mod.dispatch(specs, max_workers=1, resume=False, stop_on_error=True)

    assert executed == ["r0", "r1"]  # r2 never reached


def test_dispatch_sequential_fast_fails_on_out_of_credit_even_without_stop_on_error(
    tmp_path: Path, monkeypatch
):
    """Out-of-credit aborts the whole dispatch regardless of stop_on_error —
    continuing would only burn calls against an empty balance."""
    from gha_agents.credit import OutOfCreditError

    specs = [_make_spec(tmp_path, name=f"r{i}") for i in range(3)]
    executed: list[str] = []

    def fake_runner(
        spec,
        *,
        resume,
        sampling=None,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        executed.append(spec.name)
        if spec.name == "r1":
            raise OutOfCreditError("balance exhausted")  # noqa: TRY003
        return []

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)

    with pytest.raises(OutOfCreditError):
        exp_mod.dispatch(specs, max_workers=1, resume=False, stop_on_error=False)

    assert executed == ["r0", "r1"]  # r2 never reached — aborted on credit


# ─── per-model credit circuit breaker ─────────────────────────────────────────


def test_model_credit_tracker_marks_and_reports():
    tracker = exp_mod._ModelCreditTracker()
    assert tracker.is_exhausted("x") is False
    tracker.mark("x")
    assert tracker.is_exhausted("x") is True
    assert tracker.exhausted == frozenset({"x"})


def test_dispatch_with_tracker_contains_out_of_credit_and_continues(tmp_path: Path, monkeypatch):
    """With a tracker, out-of-credit marks that model and is contained — sibling
    models keep running instead of the whole dispatch aborting."""
    from gha_agents.credit import OutOfCreditError

    specs = [
        _make_spec(tmp_path, name="r0", model_alias="A"),
        _make_spec(tmp_path, name="r1", model_alias="A"),
        _make_spec(tmp_path, name="r2", model_alias="B"),
    ]
    executed: list[str] = []

    def fake_runner(
        spec,
        *,
        resume,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        executed.append(spec.name)
        if spec.name == "r1":
            raise OutOfCreditError("balance exhausted")  # noqa: TRY003
        return [{"file": spec.name, "smells": "0", "solved": "—", "status": "OK"}]

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)
    tracker = exp_mod._ModelCreditTracker()

    rows = exp_mod.dispatch(
        specs, max_workers=1, resume=False, stop_on_error=False, credit_tracker=tracker
    )

    assert executed == ["r0", "r1", "r2"]  # r1's credit error did NOT abort r2
    assert tracker.is_exhausted("A") is True
    assert tracker.is_exhausted("B") is False
    assert {r["file"] for r in rows} == {"r0", "r2"}


def test_run_one_experiment_skips_when_model_already_exhausted(tmp_path: Path):
    """A spec whose model is already flagged out-of-credit returns [] before it
    even touches the ground_truth file (which here does not exist — proving no load)."""
    tracker = exp_mod._ModelCreditTracker()
    tracker.mark("m")
    spec = _make_spec(tmp_path, model_alias="m")  # ground_truth_file path does not exist

    rows = exp_mod.run_one_experiment(spec, resume=False, credit_tracker=tracker)

    assert rows == []


def test_report_credit_exhaustion_exits_when_all_models_out(tmp_path: Path):
    tracker = exp_mod._ModelCreditTracker()
    tracker.mark("A")
    tracker.mark("B")
    runs = [
        _make_spec(tmp_path, name="r0", model_alias="A"),
        _make_spec(tmp_path, name="r1", model_alias="B"),
    ]
    with pytest.raises(typer.Exit) as excinfo:
        exp_mod._report_credit_exhaustion(tracker, runs)
    assert excinfo.value.exit_code == 2


def test_report_credit_exhaustion_warns_when_some_models_survive(tmp_path: Path, capsys):
    tracker = exp_mod._ModelCreditTracker()
    tracker.mark("A")
    runs = [
        _make_spec(tmp_path, name="r0", model_alias="A"),
        _make_spec(tmp_path, name="r1", model_alias="B"),
    ]
    exp_mod._report_credit_exhaustion(tracker, runs)  # must NOT raise
    err = capsys.readouterr().err
    assert "out of credit" in err and "A" in err


def test_report_credit_exhaustion_noop_when_nothing_exhausted(tmp_path: Path, capsys):
    tracker = exp_mod._ModelCreditTracker()
    runs = [_make_spec(tmp_path, model_alias="A")]
    exp_mod._report_credit_exhaustion(tracker, runs)
    assert capsys.readouterr().err == ""


# ─── dispatch: parallel ───────────────────────────────────────────────────────


def test_dispatch_parallel_runs_all(tmp_path: Path, monkeypatch):
    specs = [_make_spec(tmp_path, name=f"r{i}") for i in range(4)]
    seen_threads: set[int] = set()
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def fake_runner(
        spec,
        *,
        resume,
        sampling=None,
        request_timeout=None,
        persist_traces_flag=False,
        reporter=None,
        credit_tracker=None,
    ):
        with lock:
            seen_threads.add(threading.get_ident())
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=0.5)
        return [{"file": spec.name, "smells": "1", "solved": "—", "status": "OK"}]

    monkeypatch.setattr(exp_mod, "run_one_experiment", fake_runner)

    rows = exp_mod.dispatch(specs, max_workers=3, resume=False, stop_on_error=False)

    assert len(rows) == 4
    assert {r["file"] for r in rows} == {"r0", "r1", "r2", "r3"}
    # At least two worker threads participated — proves parallelism actually kicked in.
    assert len(seen_threads) >= 2


def test_dispatch_parallel_single_run_degrades_to_sequential(tmp_path: Path, monkeypatch):
    specs = [_make_spec(tmp_path, name="only")]
    called = {"seq": 0, "par": 0}

    # Wrap the private helpers to detect which path was taken.
    real_seq = exp_mod._run_sequential
    real_par = exp_mod._run_parallel

    def spy_seq(*a, **kw):
        called["seq"] += 1
        return real_seq(*a, **kw)

    def spy_par(*a, **kw):
        called["par"] += 1
        return real_par(*a, **kw)

    monkeypatch.setattr(exp_mod, "_run_sequential", spy_seq)
    monkeypatch.setattr(exp_mod, "_run_parallel", spy_par)
    def _noop_runner(spec, *, resume, **_kw):
        return []

    monkeypatch.setattr(exp_mod, "run_one_experiment", _noop_runner)

    exp_mod.dispatch(specs, max_workers=8, resume=False, stop_on_error=False)

    assert called == {"seq": 1, "par": 0}


# ─── CLI: --dry-run ───────────────────────────────────────────────────────────


def test_cli_dry_run_does_not_execute(tmp_path: Path, monkeypatch):
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        dedent(
            f"""
            version: 1
            defaults:
              ground_truth_file: {tmp_path / "ground_truth.json"}
              output_dir: {tmp_path / "output"}
            models:
              glm: z-ai/glm-4.5-air:free
            runs:
              - name: smoke
                design: D1
                model: glm
                limit: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )

    called = {"n": 0}

    def fake_dispatch(*_, **__):
        called["n"] += 1
        return []

    monkeypatch.setattr(exp_mod, "dispatch", fake_dispatch)

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "smoke" in result.output
    assert called["n"] == 0


def _write_basic_config(tmp_path: Path) -> Path:
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        dedent(
            f"""
            version: 1
            defaults:
              ground_truth_file: {tmp_path / "ground_truth.json"}
              output_dir: {tmp_path / "output"}
            models:
              glm: z-ai/glm-4.5-air:free
            runs:
              - name: smoke
                design: D1
                model: glm
                limit: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_cli_runs_preflight_credit_check_before_dispatch(tmp_path: Path, monkeypatch):
    """Pre-flight balance check must run before dispatch consumes any tokens."""
    config_path = _write_basic_config(tmp_path)

    call_order: list[str] = []

    def fake_preflight(**_):
        call_order.append("preflight")

    def fake_dispatch(*_, **__):
        call_order.append("dispatch")
        return []

    monkeypatch.setattr(exp_mod, "preflight_credit_check", fake_preflight)
    monkeypatch.setattr(exp_mod, "assert_models_ready", lambda *a, **k: [])
    monkeypatch.setattr(exp_mod, "dispatch", fake_dispatch)

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path)])

    assert result.exit_code == 0, result.output
    assert call_order == ["preflight", "dispatch"]


def test_cli_skips_credit_check_for_non_openrouter_runs(tmp_path: Path, monkeypatch):
    """The OpenRouter balance check must NOT gate a run set with no OpenRouter
    model — an unrelated OPENROUTER_API_KEY's balance/validity is irrelevant to an
    openai/anthropic-only experiment (regression: the check was unconditional)."""
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        dedent(
            f"""
            version: 1
            defaults:
              ground_truth_file: {tmp_path / "ground_truth.json"}
              output_dir: {tmp_path / "output"}
            models:
              haiku:
                id: claude-haiku-4-5
                provider: anthropic
            runs:
              - name: smoke
                design: D1
                model: haiku
                limit: 1
            """
        ).lstrip(),
        encoding="utf-8",
    )

    called = {"n": 0}

    def fake_preflight(**_):
        called["n"] += 1

    monkeypatch.setattr(exp_mod, "preflight_credit_check", fake_preflight)
    monkeypatch.setattr(exp_mod, "assert_models_ready", lambda *a, **k: [])
    monkeypatch.setattr(exp_mod, "dispatch", lambda *a, **k: [])

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path)])

    assert result.exit_code == 0, result.output
    assert called["n"] == 0


def _write_config_with_timeout(tmp_path: Path, *, request_timeout: str) -> Path:
    """Same as _write_basic_config but lets the test set execution.request_timeout.

    ``request_timeout`` is the raw YAML value (e.g. ``"60"`` or ``"null"``).
    """
    _write_ground_truth(tmp_path / "ground_truth.json", n=1)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        dedent(
            f"""
            version: 1
            defaults:
              ground_truth_file: {tmp_path / "ground_truth.json"}
              output_dir: {tmp_path / "output"}
            models:
              glm: z-ai/glm-4.5-air:free
            runs:
              - name: smoke
                design: D1
                model: glm
                limit: 1
            execution:
              request_timeout: {request_timeout}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_cli_timeout_flag_overrides_config(tmp_path: Path, monkeypatch):
    """--timeout on the CLI must beat execution.request_timeout from the YAML."""
    config_path = _write_config_with_timeout(tmp_path, request_timeout="60")

    captured: dict = {}

    def fake_dispatch(*_, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(exp_mod, "preflight_credit_check", lambda **_: None)
    monkeypatch.setattr(exp_mod, "assert_models_ready", lambda *a, **k: [])
    monkeypatch.setattr(exp_mod, "dispatch", fake_dispatch)

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path), "--timeout", "30"])

    assert result.exit_code == 0, result.output
    assert captured["request_timeout"] == 30.0


def test_cli_timeout_falls_back_to_config_value(tmp_path: Path, monkeypatch):
    """No --timeout flag → execution.request_timeout from the YAML wins."""
    config_path = _write_config_with_timeout(tmp_path, request_timeout="60")

    captured: dict = {}

    def fake_dispatch(*_, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(exp_mod, "preflight_credit_check", lambda **_: None)
    monkeypatch.setattr(exp_mod, "assert_models_ready", lambda *a, **k: [])
    monkeypatch.setattr(exp_mod, "dispatch", fake_dispatch)

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured["request_timeout"] == 60.0


def test_cli_exits_with_code_2_when_dispatch_raises_out_of_credit(tmp_path: Path, monkeypatch):
    """OutOfCreditError mid-dispatch should exit cleanly with code 2."""
    from gha_agents.credit import OutOfCreditError

    config_path = _write_basic_config(tmp_path)

    def fake_dispatch(*_, **__):
        raise OutOfCreditError("balance exhausted")  # noqa: TRY003

    monkeypatch.setattr(exp_mod, "dispatch", fake_dispatch)
    monkeypatch.setattr(exp_mod, "preflight_credit_check", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(exp_mod.app, [str(config_path)])

    assert result.exit_code == 2
    assert "credit" in result.output.lower() or "credit" in (result.stderr or "").lower()
