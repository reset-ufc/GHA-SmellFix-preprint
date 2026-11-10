"""Tests for the ground-truth drift gate (experiment.reconcile_gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from gha_agents.agents import ValidSmells
from gha_agents.config import ResolvedRun
from gha_agents.experiment import reconcile_gate
from gha_agents.main import _build_slug
from gha_agents.reconcile import yaml_fingerprint

_SMELLS = list(ValidSmells)
_SMELL_A = _SMELLS[0]
_SMELL_B = _SMELLS[1]
_AGENTS = {"D1": 1, "D2": 2}


def _spec(
    *, config: str, model: str, gt_file: Path, out_dir: Path, name: str = "r"
) -> ResolvedRun:
    return ResolvedRun(
        name=name,
        design=config,  # type: ignore[arg-type]
        agents=_AGENTS[config],
        model_alias=model,
        model_map={"detection": "x/y"},
        provider="openrouter",
        ground_truth_file=gt_file,
        output_dir=out_dir,
        limit=None,
        repetition=1,
        max_iterations=3,
        verbose=False,
    )


def _write_gt(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _gt_entry(file_key, *, present=(), absent=(), yaml="name: CI\n") -> dict:
    return {
        "file_key": file_key,
        "yaml_content": yaml,
        "present_smells": [{"pascal": s.name, "count": 1} for s in present],
        "absent_smells": [{"pascal": s.name} for s in absent],
    }


def _seed_result(out_dir: Path, *, config: str, model: str, file_key: str, smell,
                 present: bool, yaml="name: CI\n", with_hash: bool = True) -> Path:
    slug = _build_slug(
        file_key=file_key,
        agents=_AGENTS[config],
        repetition=1,
        target_smell=smell,
        model_alias=model,
    )
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    body = {"file_key": file_key, "target_smell": smell.name, "ground_truth": present}
    if with_hash:
        body["input_yaml_sha256"] = yaml_fingerprint(yaml)
    path = results / f"{slug}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ─── drift → abort vs --reconcile ──────────────────────────────────────────────


def test_drift_without_reconcile_aborts_exit_2(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("kept.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    # On-disk result for a pair the GT no longer contains → orphan → drift.
    _seed_result(out, config="D1", model="m", file_key="dropped.yml", smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)

    with pytest.raises(typer.Exit) as ei:
        reconcile_gate([spec], resume=True, apply=False)
    assert ei.value.exit_code == 2


def test_drift_with_reconcile_applies_and_returns(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("kept.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    orphan = _seed_result(out, config="D1", model="m", file_key="dropped.yml",
                          smell=_SMELL_A, present=True)
    keep = _seed_result(out, config="D1", model="m", file_key="kept.yml",
                        smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)

    reconcile_gate([spec], resume=True, apply=True)  # must NOT raise

    assert not orphan.exists()  # quarantined
    assert (out / "_orphaned" / "results" / orphan.name).exists()
    assert keep.exists()  # coherent result kept in place


def test_flip_with_reconcile_supersedes(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("f.yml", absent=[_SMELL_A])])
    out = tmp_path / "out"
    # Result baked PRESENT, GT now ABSENT → rerun_flip.
    flip = _seed_result(out, config="D1", model="m", file_key="f.yml", smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)

    reconcile_gate([spec], resume=True, apply=True)
    assert not flip.exists()
    assert (out / "_superseded" / "results" / flip.name).exists()


# ─── gating conditions ─────────────────────────────────────────────────────────


def test_resume_false_skips_gate_even_with_drift(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("kept.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    _seed_result(out, config="D1", model="m", file_key="dropped.yml", smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)
    # resume=False → no gate, no raise.
    reconcile_gate([spec], resume=False, apply=False)


def test_additive_only_does_not_abort(tmp_path: Path):
    # GT keeps the existing pair AND adds a new one → keep + added, no drift.
    gt = _write_gt(
        tmp_path / "gt.json",
        [_gt_entry("a.yml", present=[_SMELL_A]), _gt_entry("new.yml", present=[_SMELL_B])],
    )
    out = tmp_path / "out"
    _seed_result(out, config="D1", model="m", file_key="a.yml", smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)
    reconcile_gate([spec], resume=True, apply=False)  # must NOT raise


def test_out_of_matrix_model_drift_is_ignored(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("a.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    # 'other' model has an orphan, but the run set only contains model 'm' (coherent).
    _seed_result(out, config="D1", model="other", file_key="dropped.yml",
                 smell=_SMELL_A, present=True)
    _seed_result(out, config="D1", model="m", file_key="a.yml", smell=_SMELL_A, present=True)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)
    reconcile_gate([spec], resume=True, apply=False)  # 'm' is clean → no raise


def test_cold_start_no_results_no_raise(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("a.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    out.mkdir()  # exists but empty — nothing scanned
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)
    reconcile_gate([spec], resume=True, apply=False)


def test_missing_output_dir_no_raise(tmp_path: Path):
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("a.yml", present=[_SMELL_A])])
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=tmp_path / "nope")
    reconcile_gate([spec], resume=True, apply=False)


def test_legacy_no_hash_orphan_still_caught(tmp_path: Path):
    """A pre-hash result still yields orphan/flip drift (from baked file_key+gt)."""
    gt = _write_gt(tmp_path / "gt.json", [_gt_entry("kept.yml", present=[_SMELL_A])])
    out = tmp_path / "out"
    _seed_result(out, config="D1", model="m", file_key="dropped.yml", smell=_SMELL_A,
                 present=True, with_hash=False)
    spec = _spec(config="D1", model="m", gt_file=gt, out_dir=out)
    with pytest.raises(typer.Exit):
        reconcile_gate([spec], resume=True, apply=False)
