"""Merge semantics: re-running resume in a folder after a ground-truth edit.

Resume is content-addressed by slug (``file_key+config+rep+smell+model``) — NOT
by polarity nor yaml_content. These tests pin the behaviour for every way a
ground truth can change under an existing output dir, so the guarantees (and the
two known gaps that :mod:`gha_agents.reconcile` closes) are regression-locked.

Signal convention (mirrors test_resume_with_trace): ``_run_for_smell`` is
monkeypatched to raise, so a row with ``status == "ERROR"`` means the runner was
reached (the cell RE-RAN) and ``status == "SKIP"`` means resume skipped it.

    Case                         | slug vs existing | resume does | coherent?
    new file_key                 | new              | re-runs     | ✓
    new pair on existing file    | new              | re-runs     | ✓
    removed pair                 | not enumerated   | never runs  | ✓ (orphan on disk)
    polarity flip (same fk+smell)| SAME             | SKIPS       | ✗ gap → reconcile
    yaml edit   (same fk+smell)  | SAME             | SKIPS       | ✗ gap → reconcile
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gha_agents.agents import ValidSmells
from gha_agents.main import _build_slug, _process_entry, enumerate_pairs

_SMELLS = list(ValidSmells)
_SMELL_A = _SMELLS[0]
_SMELL_B = _SMELLS[1]


@pytest.fixture()
def fake_env(monkeypatch):
    import gha_agents.env as env_mod

    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    for role in env_mod._AGENT_ROLES:
        monkeypatch.delenv(f"AGENT_{role.upper()}_MODEL", raising=False)
    token = env_mod._model_override_var.set(
        dict.fromkeys(env_mod._AGENT_ROLES, "fake/model")
    )
    yield
    env_mod._model_override_var.reset(token)


@pytest.fixture()
def sentinel_runner(monkeypatch):
    """Make the runner raise so 'reached the runner' is observable as ERROR."""

    def _raise(**_kwargs):
        raise RuntimeError("runner reached — cell re-ran")  # noqa: TRY003

    monkeypatch.setattr("gha_agents.main._run_for_smell", _raise)


def _entry(file_key: str, *, smell=_SMELL_A, present=True, yaml="name: CI\n") -> dict:
    smell_obj = {"pascal": smell.name, "count": 1}
    return {
        "file_key": file_key,
        "yaml_content": yaml,
        "repository": "owner/repo",
        "file_path": ".github/workflows/ci.yml",
        "present_smells": [smell_obj] if present else [],
        "absent_smells": [] if present else [smell_obj],
    }


def _seed_result(output_dir: Path, slug: str) -> Path:
    """Pre-seed a minimal existing result so resume's skip predicate can fire."""
    results = output_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{slug}.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _process(
    entry: dict, *, smell, ground_truth_positive: bool, output_dir: Path, error_log: Path
) -> dict | None:
    return _process_entry(
        entry=entry,
        target_pascal=smell.name,
        ground_truth_positive=ground_truth_positive,
        agents=1,
        repetition=1,
        max_iterations=3,
        error_log=error_log,
        output_dir=output_dir,
        model_alias=None,
        resume=True,
        echo_pairs=False,
    )


def _slug(file_key: str, smell) -> str:
    return _build_slug(
        file_key=file_key, agents=1, repetition=1, target_smell=smell, model_alias=None
    )


# ─── Case 1: new row (new file_key) → runs fresh ───────────────────────────────


def test_new_file_key_runs_fresh(fake_env, sentinel_runner, tmp_path: Path):
    output_dir = tmp_path / "out"
    # A different file already has a result; the new file_key has none.
    _seed_result(output_dir, _slug("owner@repo@old.yml", _SMELL_A))

    row = _process(
        _entry("owner@repo@brand-new.yml"),
        smell=_SMELL_A,
        ground_truth_positive=True,
        output_dir=output_dir,
        error_log=tmp_path / "error.log",
    )
    assert row is not None and row["status"] == "ERROR"  # reached runner → fresh run


# ─── Case 2: new pair on an existing file_key → runs fresh, survivor kept ──────


def test_new_pair_on_existing_file_runs_fresh_and_keeps_survivor(
    fake_env, sentinel_runner, tmp_path: Path
):
    output_dir = tmp_path / "out"
    file_key = "owner@repo@ci.yml"
    # The file already has a result for smell A only.
    _seed_result(output_dir, _slug(file_key, _SMELL_A))
    entry = _entry(file_key, smell=_SMELL_A)

    # Re-processing the SAME (file, smell A) → skipped (kept).
    kept = _process(
        entry,
        smell=_SMELL_A,
        ground_truth_positive=True,
        output_dir=output_dir,
        error_log=tmp_path / "error.log",
    )
    assert kept is not None and kept["status"] == "SKIP"

    # A NEW smell B on the same file → different slug → runs fresh.
    fresh = _process(
        entry,
        smell=_SMELL_B,
        ground_truth_positive=True,
        output_dir=output_dir,
        error_log=tmp_path / "error.log",
    )
    assert fresh is not None and fresh["status"] == "ERROR"


# ─── Case 3: removed pair → never enumerated, stale result left on disk ────────


def test_removed_pair_is_not_enumerated(fake_env, tmp_path: Path):
    output_dir = tmp_path / "out"
    removed_slug = _slug("owner@repo@dropped.yml", _SMELL_A)
    orphan = _seed_result(output_dir, removed_slug)

    # The NEW ground truth no longer contains the dropped file at all.
    new_entries = [_entry("owner@repo@kept.yml")]
    pairs = enumerate_pairs(new_entries)
    enumerated_keys = {(e["file_key"], pascal) for e, pascal, _ in pairs}

    assert ("owner@repo@dropped.yml", _SMELL_A.name) not in enumerated_keys
    # Resume never touches it → the stale result survives untouched (an orphan
    # that reconcile.py later quarantines; see test_reconcile).
    assert orphan.exists()


# ─── Case 4: polarity flip → resume GAP (wrongly skipped) ──────────────────────


def test_polarity_flip_is_wrongly_skipped_by_resume(
    fake_env, sentinel_runner, tmp_path: Path
):
    """The slug omits polarity, so a flipped pair keeps the same slug and resume
    skips it — documenting the gap reconcile.py closes (rerun_flip)."""
    output_dir = tmp_path / "out"
    file_key = "owner@repo@flip.yml"
    # Existing result was produced as PRESENT; same slug regardless of polarity.
    _seed_result(output_dir, _slug(file_key, _SMELL_A))

    # New GT flips it to ABSENT — slug is identical, so resume still skips.
    row = _process(
        _entry(file_key, smell=_SMELL_A, present=False),
        smell=_SMELL_A,
        ground_truth_positive=False,
        output_dir=output_dir,
        error_log=tmp_path / "error.log",
    )
    assert row is not None and row["status"] == "SKIP"  # the gap


# ─── Case 5: yaml edit → resume GAP (wrongly skipped) ──────────────────────────


def test_yaml_edit_is_wrongly_skipped_by_resume(
    fake_env, sentinel_runner, tmp_path: Path
):
    """yaml_content is not in the slug, so an in-place file edit keeps the slug and
    resume skips the stale result — the gap reconcile.py closes via input hash."""
    output_dir = tmp_path / "out"
    file_key = "owner@repo@edited.yml"
    _seed_result(output_dir, _slug(file_key, _SMELL_A))

    row = _process(
        _entry(file_key, smell=_SMELL_A, yaml="name: CI\non: [push, pull_request]\n"),
        smell=_SMELL_A,
        ground_truth_positive=True,
        output_dir=output_dir,
        error_log=tmp_path / "error.log",
    )
    assert row is not None and row["status"] == "SKIP"  # the gap
