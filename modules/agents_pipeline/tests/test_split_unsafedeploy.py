"""Tests for scripts/split_unsafedeploy_for_agents.py — the n15 ground truth of the runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_unsafedeploy_for_agents.py"
_spec = importlib.util.spec_from_file_location("split_unsafedeploy_for_agents", _SCRIPT)
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)

SMELL = "UnsafeDeployOnFork"


def _record(key, present=(), absent=()):
    return {
        "file_key": key,
        "present_smells": [{"pascal": s, "count": 1} for s in present],
        "absent_smells": [{"pascal": s} for s in absent],
    }


def test_merge_n15_folds_negatives_and_appends_the_rest():
    n15 = [_record("a", present=["X"]), _record("b", absent=["Y"])]
    fill = [
        _record(f"b{split.SYNTH_MARKER}", present=[SMELL]),
        _record("b", absent=[SMELL]),
        _record("c", absent=[SMELL]),
    ]
    merged = split.merge_n15(n15, fill)
    assert [r["file_key"] for r in merged] == ["a", "b", f"b{split.SYNTH_MARKER}", "c"]
    assert merged[1]["absent_smells"] == [{"pascal": "Y"}, {"pascal": SMELL}]
    assert n15[1]["absent_smells"] == [{"pascal": "Y"}]


def test_merge_n15_keeps_a_negative_already_listed_once():
    n15 = [_record("b", absent=[SMELL])]
    merged = split.merge_n15(n15, [_record("b", absent=[SMELL])])
    assert merged[0]["absent_smells"] == [{"pascal": SMELL}]


# --- reproduction guard (skips if the frozen inputs are absent) -------------

_REPO_DATA = Path(__file__).resolve().parents[3] / "data"
_EXPECTED_N15_SHA = "4f5f7e3046b068ef27083a8684f17cf0969a49d82cb828fb21ccc654d63c0ef7"


@pytest.mark.skipif(
    not (_REPO_DATA / "12f_ground_truth_n15fill_UnsafeDeployOnFork.json").exists(),
    reason="frozen n15 inputs not present",
)
def test_merge_n15_reproduces_the_n15_ground_truth_of_the_runs():
    """The 12b n15 records plus the n15 fill give, byte for byte, the 570 pairs that every
    model ran on and that the analysis scores."""
    n15 = json.loads((_REPO_DATA / "12b_ground_truth_n15.json").read_text(encoding="utf-8"))
    fill = json.loads(
        (_REPO_DATA / "12f_ground_truth_n15fill_UnsafeDeployOnFork.json").read_text(
            encoding="utf-8"
        )
    )
    payload = split.dump_json(split.merge_n15(n15, fill)).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == _EXPECTED_N15_SHA
