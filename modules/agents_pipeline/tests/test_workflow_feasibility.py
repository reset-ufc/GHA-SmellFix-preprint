"""Tests for scripts/workflow_feasibility.py — annotation sheet → ground truth."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from gha_agents.main import enumerate_pairs
from openpyxl import load_workbook

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "workflow_feasibility.py"
_spec = importlib.util.spec_from_file_location("workflow_feasibility", _SCRIPT)
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)

ENTRY = {
    "file_key": "o@r@ci.yml",
    "yaml_content": "on: push\n",
    "present_smells": [],
    "absent_smells": [],
}


def _sheet(overrides: dict[str, tuple[str, str]]) -> list[dict]:
    return [
        {
            "workflow_id": "o@r@ci.yml",
            "smell_pascal": pascal,
            "manual_label": overrides.get(pascal, ("absent", ""))[0],
            "number_of_instances": overrides.get(pascal, ("absent", ""))[1],
            "expected_refactoring_scope": "job-level key",
        }
        for pascal, _ in wf.SMELLS
    ]


def test_every_smell_becomes_one_run_and_uncertain_is_tracked():
    rows = _sheet(
        {
            "MissingJobTimeout": ("present", "2"),
            "MissingConcurrencyPR": ("present", "1"),
            "UnpinnedDependencies": ("Uncertain", ""),
        }
    )
    [entry] = wf.build_ground_truth([ENTRY], rows)

    assert {"pascal": "MissingJobTimeout", "count": 2} in entry["present_smells"]
    assert len(entry["present_smells"]) == 2
    assert entry["uncertain_smells"] == ["UnpinnedDependencies"]
    assert {"pascal": "UnpinnedDependencies"} in entry["absent_smells"]
    pairs = enumerate_pairs([entry])
    assert len(pairs) == 19
    assert {p for _, p, _ in pairs} == wf.SMELL_PASCALS


def test_incomplete_sheet_is_rejected():
    rows = _sheet({"MissingJobTimeout": ("present", "")})[:-1]
    with pytest.raises(ValueError, match="number_of_instances") as exc:
        wf.build_ground_truth([ENTRY], rows)
    assert "missing 2 smells" in str(exc.value)


def test_inconsistent_rows_and_single_smell_workflow_are_rejected():
    rows = _sheet({"MissingJobTimeout": ("present", "1"), "MissingConcurrencyPR": ("absent", "2")})
    rows[2]["number_of_instances"] = "n/a"
    with pytest.raises(ValueError) as exc:
        wf.build_ground_truth([ENTRY], rows)
    message = str(exc.value)
    assert "absent cannot have number_of_instances 2" in message
    assert "'n/a' is not a whole number" in message

    rows = _sheet({"MissingJobTimeout": ("present", "1")})
    with pytest.raises(ValueError, match="only 1 present smell"):
        wf.build_ground_truth([ENTRY], rows)

    rows = _sheet({"MissingJobTimeout": ("present", "1"), "MissingConcurrencyPR": ("present", "1")})
    for row in rows:
        row["expected_refactoring_scope"] = ""
    with pytest.raises(ValueError, match="expected_refactoring_scope"):
        wf.build_ground_truth([ENTRY], rows)


def _export_tab_as_csv(ws, path: Path) -> None:
    """What Google Sheets' File > Download > CSV writes for the open tab."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(
            ["" if v is None else v for v in row] for row in ws.iter_rows(values_only=True)
        )


def test_tabs_exported_as_csv_round_trip_through_consensus(tmp_path):
    entry = ENTRY | {"repository": "o/r", "file_path": ".github/workflows/ci.yml"}
    path = tmp_path / "sheet.xlsx"
    wf.write_annotation_sheet([entry], path, {"MissingJobTimeout": "ADD timeout-minutes"})

    book = load_workbook(path)
    assert book.sheetnames == list(wf.TABS)
    header = [c.value for c in book[wf.TABS[0]][1]]
    col = {h: i + 1 for i, h in enumerate(header)}
    yaml_cell = book[wf.TABS[0]].cell(2, col["workflow_yaml"]).value
    assert yaml_cell == "  1  on: push"
    # A marks two smells present; B agrees on one and misses the other.
    for tab, present in (
        (wf.TABS[0], {"MissingJobTimeout", "MissingConcurrencyPR"}),
        (wf.TABS[1], {"MissingJobTimeout"}),
    ):
        ws = book[tab]
        for row in range(2, ws.max_row + 1):
            smell = ws.cell(row, col["smell_pascal"]).value
            ws.cell(row, col["manual_label"]).value = "present" if smell in present else "absent"
            if smell in present:
                ws.cell(row, col["number_of_instances"]).value = 1
                ws.cell(row, col["expected_refactoring_scope"]).value = "jobs.a"
    exports = [tmp_path / "annotator_a.csv", tmp_path / "annotator_b.csv"]
    for tab, export in zip(wf.TABS, exports, strict=True):
        _export_tab_as_csv(book[tab], export)

    merged, disagreements = wf.merge_annotations(*(wf.read_annotation_csv(p) for p in exports))
    assert disagreements == 1
    [unresolved] = [r for r in merged if not r["manual_label"]]
    assert (unresolved["smell_pascal"], unresolved["annotator_a"]) == (
        "MissingConcurrencyPR",
        "present/1",
    )
    with pytest.raises(ValueError, match="manual_label ''"):
        wf.build_ground_truth([entry], merged)

    unresolved["manual_label"] = "present"
    [gt] = wf.build_ground_truth([entry], merged)
    assert len(gt["present_smells"]) == 2 and len(enumerate_pairs([gt])) == 19


def test_csv_export_without_the_sheet_header_is_rejected(tmp_path):
    path = tmp_path / "annotator_a.csv"
    path.write_text("workflow_id,label\no@r@ci.yml,present\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manual_label"):
        wf.read_annotation_csv(path)


def test_build_gt_only_warns_about_smells_without_a_present_instance(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wf, "OUT_DIR", tmp_path)
    (tmp_path / "selected_workflows.json").write_text(json.dumps([ENTRY]), encoding="utf-8")
    rows = _sheet({"MissingJobTimeout": ("present", "1"), "MissingConcurrencyPR": ("present", "1")})
    annotations = tmp_path / "annotations_consensus.csv"
    with annotations.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    wf.build_gt(annotations)

    [entry] = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))
    assert len(entry["present_smells"]) == 2 and len(enumerate_pairs([entry])) == 19
    out = capsys.readouterr().out
    assert "no present instance of 17 smells" in out and "MissingForkGuardPR" in out


def test_agreement_is_cohens_kappa_over_the_two_annotators_labels():
    verdicts = [("present/2", "present/1"), ("absent", "absent"), ("present/1", "absent")]
    rows = [{"annotator_a": a, "annotator_b": b} for a, b in [*verdicts, ("absent", "absent")]]
    # labels A = p a p a, B = p a a a: observed 3/4, expected (2*1 + 2*3)/16 = 1/2
    assert wf.agreement(rows) == (4, 3, 0.5)


def test_freeze_records_a_file_again_only_when_its_bytes_change(tmp_path):
    export, manifest = tmp_path / "annotator_a.csv", tmp_path / "manifest.json"
    export.write_text("workflow_id\no@r@ci.yml\n", encoding="utf-8")
    wf.freeze((export, tmp_path / "annotations_consensus.csv"), manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    assert list(entries) == ["annotator_a.csv"]  # a missing file is not recorded
    first = entries["annotator_a.csv"]
    assert first["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest()
    assert first["bytes"] == export.stat().st_size
    assert datetime.fromisoformat(first["completed_at"]).utcoffset() == timedelta(0)  # UTC

    os.utime(export, (0, 0))  # a fresh checkout: a new mtime, the same bytes
    wf.freeze((export,), manifest)
    assert json.loads(manifest.read_text(encoding="utf-8"))["annotator_a.csv"] == first

    export.write_text("workflow_id\no@r@other.yml\n", encoding="utf-8")
    wf.freeze((export,), manifest)
    changed = json.loads(manifest.read_text(encoding="utf-8"))["annotator_a.csv"]
    assert changed["sha256"] == hashlib.sha256(export.read_bytes()).hexdigest() != first["sha256"]


@pytest.mark.skipif(not wf.MANIFEST.exists(), reason="no frozen feasibility inputs")
def test_frozen_feasibility_files_still_match_the_manifest():
    """Every hand-made file is frozen, and one edited after ``freeze`` has to be frozen again."""
    entries = json.loads(wf.MANIFEST.read_text(encoding="utf-8"))
    assert set(entries) == {path.name for path in wf.FROZEN}, "run freeze"
    for path in wf.FROZEN:
        assert path.exists(), f"{path.name} is in the manifest but missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        changed = f"{path.name} changed since it was frozen; run freeze"
        assert digest == entries[path.name]["sha256"], changed


def test_pick_covering_prefers_new_smells_and_keeps_quotas():
    smells = sorted(wf.SMELL_PASCALS)
    # Plenty of look-alike candidates per bucket, plus one per bucket holding rare smells;
    # the first rare 5+ candidate is invalid and must be skipped.
    common = [
        {"file_key": f"{b}-{i}", "bucket": b, "chance": dict.fromkeys(smells[:2], 1.0)}
        for b in ("2", "3-4", "5+")
        for i in range(10)
    ]
    rare = [
        {"file_key": "bad", "bucket": "5+", "chance": dict.fromkeys(smells[2:], 1.0)},
        {"file_key": "r5", "bucket": "5+", "chance": dict.fromkeys(smells[2:10], 1.0)},
        {"file_key": "r3", "bucket": "3-4", "chance": dict.fromkeys(smells[10:14], 1.0)},
        {"file_key": "r2", "bucket": "2", "chance": dict.fromkeys(smells[14:], 0.9)},
    ]
    picked = wf.pick_covering(
        common + rare,
        random.Random(2027),  # noqa: S311
        lambda r: r["file_key"] != "bad",
    )
    keys = {r["file_key"] for r in picked}
    assert {"r5", "r3", "r2"} <= keys and "bad" not in keys
    buckets = [r["bucket"] for r in picked]
    assert (buckets.count("2"), buckets.count("3-4"), buckets.count("5+")) == (2, 3, 3)
