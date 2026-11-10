"""Unit tests for the Step 13 synthetic-merge core (pure funcs) + a reproduction
guard that locks the final ground truth's byte hash to the frozen inputs."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from dataset_utils import synthetic_merge as sm

SMELL = sm.SMELL_PASCAL
MARK = sm.SYNTH_KEY_MARKER


def _record(file_key, present=(), absent=(), loc=10, extra=None):
    """A full RECORD_FIELDS record (+ optional extra unrecognized fields)."""
    r = {
        "file_key": file_key,
        "repository": "o/r",
        "file_path": ".github/workflows/x.yml",
        "yaml_content": "\n".join(["line"] * loc),
        "file_line_count": loc,
        "workflow_run_count": 1,
        "repo_language": "Python",
        "default_branch": "main",
        "present_smells": [{"pascal": p, "count": 1} for p in present],
        "absent_smells": [{"pascal": a} for a in absent],
    }
    if extra:
        r.update(extra)
    return r


# --- normalize_record ----------------------------------------------------

def test_normalize_drops_unrecognized_and_orders_fields():
    r = _record(f"f0{MARK}", present=[SMELL], extra={"injection_confidence": "high"})
    out = sm.normalize_record(r)
    assert list(out.keys()) == sm.RECORD_FIELDS  # fixed order, extra dropped
    assert "injection_confidence" not in out
    assert out["present_smells"] == [{"pascal": SMELL, "count": 1}]
    assert out["absent_smells"] == []


def test_normalize_normalizes_smell_key_order():
    # present -> {pascal, count}; absent -> {pascal} only.
    r = _record("f0", present=[SMELL], absent=["Other"])
    r["present_smells"] = [{"count": 2, "pascal": SMELL}]  # reversed keys, count=2
    out = sm.normalize_record(r)
    assert out["present_smells"] == [{"pascal": SMELL, "count": 2}]
    assert out["absent_smells"] == [{"pascal": "Other"}]


def test_normalize_positive_requires_present():
    with pytest.raises(ValueError):
        sm.normalize_record(_record("f0"))  # empty present_smells, require_present default
    # a genuine negative normalizes fine with require_present=False
    out = sm.normalize_record(_record("f0", absent=[SMELL]), require_present=False)
    assert out["present_smells"] == []


# --- split_by_loc -----------------------------------------------------------

def test_split_by_loc_partitions_on_cap():
    recs = [_record("a", present=[SMELL], loc=100), _record("b", present=[SMELL], loc=210)]
    within, over = sm.split_by_loc(recs, max_loc=209)
    assert [r["file_key"] for r in within] == ["a"]
    assert [r["file_key"] for r in over] == ["b"]


def test_record_loc_recomputes_when_field_bad():
    r = _record("a", present=[SMELL], loc=5)
    r["file_line_count"] = "not-an-int"
    r["yaml_content"] = "a\nb\nc"
    assert sm.record_loc(r) == 3


# --- approval_from_csv (the gate that replaces annotations.json) ------------

def _write_csv(path, rows):
    cols = ["file_key", "target_smell", "tool_output", "final_decision"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_approval_present_approved_absent_rejected(tmp_path):
    csv_path = tmp_path / "eval.csv"
    _write_csv(csv_path, [
        {"file_key": f"{key}{MARK}", "target_smell": smell, "tool_output": "positive",
         "final_decision": decision}
        for key, smell, decision in [
            ("a", SMELL, "present"), ("b", SMELL, "absent"), ("c", "Other", "present")
        ]
    ])
    approved, rejected, reviewed = sm.approval_from_csv(csv_path, SMELL)
    assert approved == {f"a{MARK}"}          # present
    assert rejected == {f"b{MARK}"}          # absent (flip, dropped)
    assert reviewed == {f"a{MARK}", f"b{MARK}"}  # filtered to SMELL; 'Other' excluded


def test_approved_synth_records_preserve_input_order():
    synth = [_record(f"{k}{MARK}", present=[SMELL]) for k in ("z", "a", "m")]
    approved = {f"z{MARK}", f"m{MARK}"}
    got = sm.approved_synth_records(synth, approved)
    assert [r["file_key"] for r in got] == [f"z{MARK}", f"m{MARK}"]  # 12d order, not sorted


# --- fold_negatives ---------------------------------------------------------

def test_fold_negatives_enrich_add_already():
    gt = [
        _record("existing", absent=["Other"]),      # will be enriched with SMELL
        _record("already", absent=[SMELL]),          # already absent for SMELL -> no-op
    ]
    targets = [
        _record("existing", absent=[SMELL]),
        _record("already", absent=[SMELL]),
        _record("brand_new", absent=[SMELL]),        # appended as new negative
    ]
    out, counts = sm.fold_negatives(gt, targets)
    assert counts == {"enriched": 1, "added": 1, "already": 1}
    enriched = next(r for r in out if r["file_key"] == "existing")
    assert {s["pascal"] for s in enriched["absent_smells"]} == {"Other", SMELL}
    assert any(r["file_key"] == "brand_new" for r in out)
    # input not mutated
    assert len(gt[0]["absent_smells"]) == 1


def test_fold_negatives_does_not_mutate_input():
    gt = [_record("x", absent=["Other"])]
    before = json.dumps(gt, sort_keys=True)
    sm.fold_negatives(gt, [_record("x", absent=[SMELL])])
    assert json.dumps(gt, sort_keys=True) == before


# --- merge + summary --------------------------------------------------------

def test_merge_into_ground_truth_appends():
    gt = [_record("g", absent=[SMELL])]
    adds = [_record(f"s{MARK}", present=[SMELL])]
    merged = sm.merge_into_ground_truth(gt, adds)
    assert [r["file_key"] for r in merged] == ["g", f"s{MARK}"]


def test_smell_summary_splits_synth():
    recs = [
        _record("real", present=[SMELL]),
        _record(f"synth{MARK}", present=[SMELL]),
        _record("neg", absent=[SMELL]),
    ]
    row = next(r for r in sm.smell_summary(recs) if r["smell"] == SMELL)
    assert row == {"smell": SMELL, "present": 2, "present_synth": 1, "absent": 1, "absent_synth": 0}


# --- reproduction guard (skips if the frozen inputs are absent) -------------

_REPO_DATA = Path(__file__).resolve().parents[3] / "data"
_EXPECTED_FINAL_SHA = "a90c1675a6fa8c77a276ef86544c463da49e410d6334b6e94347c6b276b60dbb"


@pytest.mark.skipif(
    not (_REPO_DATA / "12d_synth_UnsafeDeployOnFork_present.json").exists(),
    reason="frozen synthesis inputs not present",
)
def test_run_reproduces_frozen_final_ground_truth(tmp_path):
    """Given the frozen inputs, Step 13 reproduces the final ground
    truth byte-for-byte (== the previous ad-hoc 12e_with_synth)."""
    out_path, report = sm.run(
        synth_path=_REPO_DATA / "12d_synth_UnsafeDeployOnFork_present.json",
        ground_truth_path=_REPO_DATA / "12b_final_ground_truth.json",
        negatives_path=_REPO_DATA / "12c_synthesis_targets_UnsafeDeployOnFork_absent.json",
        eval_csv_path=_REPO_DATA / "12e_sinth_manual_evaluation.csv",
        output_path=tmp_path / "13_final.json",
        report_path=tmp_path / "step13.json",
    )
    sha = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    assert sha == _EXPECTED_FINAL_SHA, "Step 13 no longer reproduces the frozen final ground truth"
    assert report["n_approved"] == 45
    assert report["n_rejected"] == 4
    assert report["n_entries_out"] == 902
