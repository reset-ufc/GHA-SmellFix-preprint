"""Tests for gha_agents.reconcile — classify/plan/apply against an edited GT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gha_agents.agents import ValidSmells
from gha_agents.main import _build_slug
from gha_agents.reconcile import (
    apply_plan,
    build_gt_index,
    build_plan,
    classify,
    iter_result_files,
    yaml_fingerprint,
)

_SMELLS = list(ValidSmells)
_SMELL_A = _SMELLS[0]
_SMELL_B = _SMELLS[1]
_MODEL = "deepseek-v4-flash"


def _gt_entry(file_key: str, *, present=(), absent=(), yaml="name: CI\n") -> dict:
    return {
        "file_key": file_key,
        "yaml_content": yaml,
        "present_smells": [{"pascal": s.name, "count": 1} for s in present],
        "absent_smells": [{"pascal": s.name} for s in absent],
    }


def _result(file_key: str, smell, *, present: bool, yaml="name: CI\n") -> dict:
    return {
        "file_key": file_key,
        "target_smell": smell.name,
        "ground_truth": present,
        "input_yaml_sha256": yaml_fingerprint(yaml),
    }


# ─── build_gt_index ────────────────────────────────────────────────────────────


def test_yaml_fingerprint_matches_main_hashing():
    yaml = "name: CI\non: [push]\n"
    assert yaml_fingerprint(yaml) == hashlib.sha256(yaml.encode("utf-8")).hexdigest()


def test_build_gt_index_polarity_yaml_and_pairs():
    gt = build_gt_index(
        [
            _gt_entry("a.yml", present=[_SMELL_A], absent=[_SMELL_B], yaml="A"),
            _gt_entry("b.yml", present=[_SMELL_B], yaml="B"),
        ]
    )
    assert gt.polarity[("a.yml", _SMELL_A.name)] is True
    assert gt.polarity[("a.yml", _SMELL_B.name)] is False
    assert gt.polarity[("b.yml", _SMELL_B.name)] is True
    assert gt.yaml_sha["a.yml"] == yaml_fingerprint("A")
    assert ("a.yml", _SMELL_A.name, True) in gt.pairs
    assert not gt.conflicts


def test_build_gt_index_flags_present_and_absent_conflict():
    gt = build_gt_index([_gt_entry("a.yml", present=[_SMELL_A], absent=[_SMELL_A])])
    assert ("a.yml", _SMELL_A.name) in gt.conflicts


# ─── classify ──────────────────────────────────────────────────────────────────


def _gt_one(file_key="a.yml", *, present=(_SMELL_A,), absent=(), yaml="name: CI\n"):
    return build_gt_index([_gt_entry(file_key, present=present, absent=absent, yaml=yaml)])


def test_classify_keep():
    gt = _gt_one()
    assert classify(_result("a.yml", _SMELL_A, present=True), gt).kind == "keep"


def test_classify_orphan_when_pair_gone():
    gt = _gt_one(present=(_SMELL_A,))
    # Result is for smell B, which the GT does not evaluate on this file.
    assert classify(_result("a.yml", _SMELL_B, present=True), gt).kind == "orphan"


def test_classify_orphan_when_file_gone():
    gt = _gt_one(file_key="kept.yml")
    assert classify(_result("dropped.yml", _SMELL_A, present=True), gt).kind == "orphan"


def test_classify_rerun_flip_on_polarity_change():
    gt = _gt_one(present=(), absent=(_SMELL_A,))  # GT now says ABSENT
    v = classify(_result("a.yml", _SMELL_A, present=True), gt)  # result baked PRESENT
    assert v.kind == "rerun_flip"
    assert "present" in v.reason and "absent" in v.reason


def test_classify_rerun_yaml_on_input_hash_change():
    gt = _gt_one(present=(_SMELL_A,), yaml="EDITED CONTENT")
    v = classify(_result("a.yml", _SMELL_A, present=True, yaml="OLD CONTENT"), gt)
    assert v.kind == "rerun_yaml"


def test_classify_keep_when_yaml_hash_absent_legacy_result():
    """A pre-hash result (no input_yaml_sha256) can't self-report yaml drift, so it
    is kept rather than falsely flagged — yaml-edit detection is best-effort."""
    gt = _gt_one(present=(_SMELL_A,), yaml="WHATEVER")
    legacy = {"file_key": "a.yml", "target_smell": _SMELL_A.name, "ground_truth": True}
    assert classify(legacy, gt).kind == "keep"


def test_classify_unreadable_when_identity_missing():
    gt = _gt_one()
    assert classify({"ground_truth": True}, gt).kind == "unreadable"


def test_classify_reads_legacy_ground_truth_positive_key():
    gt = _gt_one(present=(), absent=(_SMELL_A,))
    legacy = {
        "file_key": "a.yml",
        "target_smell": _SMELL_A.name,
        "ground_truth_positive": True,  # legacy key name
    }
    assert classify(legacy, gt).kind == "rerun_flip"


# ─── disk: iter_result_files + build_plan + apply_plan ─────────────────────────


def _seed(root: Path, design: str, slug: str, result: dict) -> Path:
    """Write a result + sibling patched/trace under root/design_<d>/."""
    base = root / f"design_{design}"
    (base / "results").mkdir(parents=True, exist_ok=True)
    (base / "patched_workflows").mkdir(parents=True, exist_ok=True)
    (base / "traces").mkdir(parents=True, exist_ok=True)
    rp = base / "results" / f"{slug}.json"
    rp.write_text(json.dumps(result), encoding="utf-8")
    (base / "patched_workflows" / f"{slug}.yml").write_text("patched", encoding="utf-8")
    (base / "traces" / f"{slug}.json").write_text("{}", encoding="utf-8")
    return rp


def _slug(file_key: str, smell, *, model: str | None = _MODEL) -> str:
    return _build_slug(
        file_key=file_key,
        agents=2,
        repetition=1,
        target_smell=smell,
        model_alias=model,
    )


def test_iter_result_files_walks_design_dirs_and_skips_quarantine(tmp_path: Path):
    root = tmp_path / "out"
    _seed(root, "d1", _slug("a.yml", _SMELL_A), _result("a.yml", _SMELL_A, present=True))
    # A file already quarantined must NOT be re-collected.
    q = root / "_orphaned" / "design_d1" / "results"
    q.mkdir(parents=True, exist_ok=True)
    (q / "old.json").write_text("{}", encoding="utf-8")

    found = list(iter_result_files(root))
    assert len(found) == 1
    assert "_orphaned" not in found[0].parts


def test_build_plan_classifies_and_counts_added(tmp_path: Path):
    root = tmp_path / "out"
    # KEEP: still present.
    _seed(root, "d1", _slug("keep.yml", _SMELL_A), _result("keep.yml", _SMELL_A, present=True))
    # ORPHAN: file dropped from GT.
    _seed(root, "d1", _slug("drop.yml", _SMELL_A), _result("drop.yml", _SMELL_A, present=True))
    # RERUN flip: GT now absent.
    _seed(root, "d1", _slug("flip.yml", _SMELL_A), _result("flip.yml", _SMELL_A, present=True))

    gt = build_gt_index(
        [
            _gt_entry("keep.yml", present=[_SMELL_A]),
            _gt_entry("flip.yml", absent=[_SMELL_A]),
            _gt_entry("added.yml", present=[_SMELL_B]),  # no result yet → added
        ]
    )
    plan = build_plan(root, gt, model=_MODEL)
    counts = plan.counts
    assert counts.get("keep") == 1
    assert counts.get("orphan") == 1
    assert counts.get("rerun_flip") == 1
    assert counts.get("added") == 1  # added.yml#B has no on-disk result
    assert plan.needs_action


def test_build_plan_model_filter_scopes_to_one_model(tmp_path: Path):
    root = tmp_path / "out"
    _seed(root, "d1", _slug("a.yml", _SMELL_A, model="deepseek-v4-flash"),
          _result("a.yml", _SMELL_A, present=True))
    _seed(root, "d1", _slug("a.yml", _SMELL_A, model="gemma4"),
          _result("a.yml", _SMELL_A, present=True))
    gt = build_gt_index([_gt_entry("a.yml", absent=[_SMELL_A])])  # flip for both

    plan = build_plan(root, gt, model="deepseek-v4-flash")
    assert len(plan.cells) == 1
    assert plan.cells[0].slug.endswith("@deepseek-v4-flash")


def test_apply_quarantines_orphan_and_supersedes_rerun(tmp_path: Path):
    root = tmp_path / "out"
    keep_rp = _seed(root, "d1", _slug("keep.yml", _SMELL_A),
                    _result("keep.yml", _SMELL_A, present=True))
    orphan_rp = _seed(root, "d1", _slug("drop.yml", _SMELL_A),
                      _result("drop.yml", _SMELL_A, present=True))
    flip_rp = _seed(root, "d1", _slug("flip.yml", _SMELL_A),
                    _result("flip.yml", _SMELL_A, present=True))

    gt = build_gt_index(
        [_gt_entry("keep.yml", present=[_SMELL_A]), _gt_entry("flip.yml", absent=[_SMELL_A])]
    )
    plan = build_plan(root, gt, model=_MODEL)
    moved = apply_plan(plan)

    # keep untouched
    assert keep_rp.exists()
    # orphan moved under _orphaned/ (result + patched + trace = 3 files)
    assert not orphan_rp.exists()
    assert (root / "_orphaned" / "design_d1" / "results" / orphan_rp.name).exists()
    assert (root / "_orphaned" / "design_d1" / "patched_workflows"
            / f"{_slug('drop.yml', _SMELL_A)}.yml").exists()
    # rerun (flip) moved under _superseded/ → resume will re-run it
    assert not flip_rp.exists()
    assert (root / "_superseded" / "design_d1" / "results" / flip_rp.name).exists()
    assert moved["_orphaned"] == 3
    assert moved["_superseded"] == 3


def test_apply_then_replan_is_clean(tmp_path: Path):
    root = tmp_path / "out"
    _seed(root, "d1", _slug("drop.yml", _SMELL_A), _result("drop.yml", _SMELL_A, present=True))
    _seed(root, "d1", _slug("keep.yml", _SMELL_A), _result("keep.yml", _SMELL_A, present=True))
    gt = build_gt_index([_gt_entry("keep.yml", present=[_SMELL_A])])

    apply_plan(build_plan(root, gt, model=_MODEL))
    # Second pass sees only the coherent KEEP — quarantined files are skipped.
    replan = build_plan(root, gt, model=_MODEL)
    assert not replan.needs_action
    assert replan.counts.get("keep") == 1
    assert replan.counts.get("orphan", 0) == 0
