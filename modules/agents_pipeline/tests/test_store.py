"""Unit tests for the RunStore Repository (#45).

Pins the on-disk layout (the slug → path grammar, incl. per-iter patch
filenames), the trace-aware resume predicate, and the save operations. The
end-to-end resume behaviour stays covered by test_resume_with_trace.py driving
_process_entry; these isolate the Repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gha_agents.store import RunStore

_SLUG = "owner@repo@ci@D2@rep1@unscoped_permissions@gpt"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed(store: RunStore, *, result: bool, trace: bool) -> None:
    if result:
        store.result_path(_SLUG).parent.mkdir(parents=True, exist_ok=True)
        store.result_path(_SLUG).write_text("{}", encoding="utf-8")
    if trace:
        store.trace_path(_SLUG).parent.mkdir(parents=True, exist_ok=True)
        store.trace_path(_SLUG).write_text("{}", encoding="utf-8")


# ─── path grammar ─────────────────────────────────────────────────────────────


def test_path_grammar(tmp_path: Path):
    store = RunStore(tmp_path)
    assert store.result_path(_SLUG) == tmp_path / "results" / f"{_SLUG}.json"
    assert store.patched_path(_SLUG) == tmp_path / "patched_workflows" / f"{_SLUG}.yml"
    assert store.trace_path(_SLUG) == tmp_path / "traces" / f"{_SLUG}.json"
    assert store.iters_dir == tmp_path / "patched_workflows_iters"
    assert store.iter_logs_dir == tmp_path / "iteration_logs"
    assert store.traces_dir == tmp_path / "traces"
    assert store.iter_patch_path(_SLUG, 2) == store.iters_dir / f"{_SLUG}@iter2.yml"
    assert store.iter_log_path(_SLUG, "detection", 1) == store.iter_logs_dir / f"{_SLUG}@detection1.json"
    assert store.iter_log_path(_SLUG, "refactor", 3) == store.iter_logs_dir / f"{_SLUG}@refactor3.json"


# ─── exists() — trace-aware resume rule ───────────────────────────────────────


def test_exists_skips_when_result_present_and_traces_off(tmp_path: Path):
    store = RunStore(tmp_path)
    _seed(store, result=True, trace=False)
    assert store.exists(_SLUG, trace_required=False) is True


def test_exists_skips_when_result_and_trace_present_under_persist(tmp_path: Path):
    store = RunStore(tmp_path)
    _seed(store, result=True, trace=True)
    assert store.exists(_SLUG, trace_required=True) is True


def test_exists_reruns_when_trace_missing_under_persist(tmp_path: Path):
    store = RunStore(tmp_path)
    _seed(store, result=True, trace=False)
    assert store.exists(_SLUG, trace_required=True) is False


def test_exists_reruns_when_result_missing(tmp_path: Path):
    store = RunStore(tmp_path)
    _seed(store, result=False, trace=True)  # trace alone never satisfies skip
    assert store.exists(_SLUG, trace_required=False) is False
    assert store.exists(_SLUG, trace_required=True) is False


# ─── save_patched_workflow ────────────────────────────────────────────────────


def test_save_patched_workflow_writes_and_returns_path(tmp_path: Path):
    store = RunStore(tmp_path)
    path = store.save_patched_workflow(_SLUG, "name: CI\n")
    assert path == store.patched_path(_SLUG)
    assert path.read_text(encoding="utf-8") == "name: CI\n"


# ─── save_iter_patches — per-iter filename grammar ────────────────────────────


def test_save_iter_patches_one_file_per_iteration(tmp_path: Path):
    store = RunStore(tmp_path)
    written = store.save_iter_patches(_SLUG, ["v1: yaml\n", "v2: yaml\n"])
    assert written == [
        store.iters_dir / f"{_SLUG}@iter1.yml",
        store.iters_dir / f"{_SLUG}@iter2.yml",
    ]
    assert (store.iters_dir / f"{_SLUG}@iter1.yml").read_text(encoding="utf-8") == "v1: yaml\n"
    assert (store.iters_dir / f"{_SLUG}@iter2.yml").read_text(encoding="utf-8") == "v2: yaml\n"


def test_save_iter_patches_empty_skips_directory(tmp_path: Path):
    store = RunStore(tmp_path)
    assert store.save_iter_patches(_SLUG, []) == []
    assert not store.iters_dir.exists()  # D1/D2 path never creates the iters dir


def test_save_iter_patches_preserves_bytes(tmp_path: Path):
    store = RunStore(tmp_path)
    patches = ["alpha\n", "beta\n", "gamma\n"]
    written = store.save_iter_patches(_SLUG, patches)
    for path, original in zip(written, patches, strict=True):
        assert _sha256(path.read_text(encoding="utf-8")) == _sha256(original)


# ─── save_iter_logs — per-iteration self-describing JSON ──────────────────────


def test_save_iter_logs_one_self_describing_file_per_iteration(tmp_path: Path):
    store = RunStore(tmp_path)
    header = {"workflow_id": "wf", "design": "D2", "ground_truth": True}
    detection = [
        {"iteration": 1, "detection": {"smell_detected": True}, "validation": {"agrees": False}},
        {"iteration": 2, "detection": {"smell_detected": True}, "validation": {"agrees": True}},
    ]
    refactor = [
        {"iteration": 1, "refactor": {"patched_chars": 10}, "validation": {"agrees": True}},
    ]
    written = store.save_iter_logs(_SLUG, header=header, detection=detection, refactor=refactor)
    # Detection files first (in order), then refactor; filename carries loop + index.
    assert written == [
        store.iter_logs_dir / f"{_SLUG}@detection1.json",
        store.iter_logs_dir / f"{_SLUG}@detection2.json",
        store.iter_logs_dir / f"{_SLUG}@refactor1.json",
    ]
    d1 = json.loads((store.iter_logs_dir / f"{_SLUG}@detection1.json").read_text(encoding="utf-8"))
    # header merged + loop tag + the iteration entry, all in one self-describing file
    assert d1["workflow_id"] == "wf"
    assert d1["design"] == "D2"
    assert d1["ground_truth"] is True
    assert d1["loop"] == "detection"
    assert d1["iteration"] == 1
    assert d1["detection"] == {"smell_detected": True}
    assert d1["validation"] == {"agrees": False}
    r1 = json.loads((store.iter_logs_dir / f"{_SLUG}@refactor1.json").read_text(encoding="utf-8"))
    assert r1["loop"] == "refactor"
    assert r1["refactor"] == {"patched_chars": 10}


def test_save_iter_logs_empty_skips_directory(tmp_path: Path):
    store = RunStore(tmp_path)
    assert store.save_iter_logs(_SLUG, header={}, detection=[], refactor=[]) == []
    assert not store.iter_logs_dir.exists()  # D1/D2 path never creates the dir


def test_save_iter_logs_detection_only_writes_no_refactor_files(tmp_path: Path):
    # Detection iterations only, no refactor loop (a smell-absent bypass).
    store = RunStore(tmp_path)
    detection = [{"iteration": 1, "detection": {"smell_detected": False}, "validation": {"agrees": True}}]
    written = store.save_iter_logs(_SLUG, header={"workflow_id": "wf"}, detection=detection, refactor=[])
    assert written == [store.iter_logs_dir / f"{_SLUG}@detection1.json"]
    assert not (store.iter_logs_dir / f"{_SLUG}@refactor1.json").exists()


# ─── save_result + prepare ────────────────────────────────────────────────────


def test_save_result_writes_json(tmp_path: Path):
    store = RunStore(tmp_path)
    data = {"smell_detected": True, "count": 1}
    path = store.save_result(_SLUG, data)
    assert path == store.result_path(_SLUG)
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_prepare_creates_result_and_patched_dirs(tmp_path: Path):
    store = RunStore(tmp_path)
    store.prepare()
    assert (tmp_path / "results").is_dir()
    assert (tmp_path / "patched_workflows").is_dir()
