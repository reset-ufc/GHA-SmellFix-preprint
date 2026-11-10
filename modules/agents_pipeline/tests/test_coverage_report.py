"""Unit tests for the coverage_report counting logic.

The report is read-only diagnostics (it never touches the run), so these pin the
pure counting/attribution functions: expected-pair math, slug→alias parsing,
result counting, and error.log → (model, category) attribution including the
unmapped-model bucket.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from gha_agents.coverage_report import (
    _alias_of_slug,
    _count_failures,
    _count_results,
    _done_keys,
    _scenario_dir,
    expected_pairs,
)


def test_scenario_dir_maps_design_to_layout():
    assert _scenario_dir("D1") == "design_d1"
    assert _scenario_dir("D2") == "design_d2"


def test_alias_of_slug_is_final_at_segment():
    p = Path("Repo@wf@D2@rep1@somesmell@deepseek-v4-flash.json")
    assert _alias_of_slug(p) == "deepseek-v4-flash"


def test_expected_pairs_sums_present_and_absent(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    ground_truth.write_text(
        json.dumps(
            [
                {"present_smells": [{"pascal": "A"}, {"pascal": "B"}], "absent_smells": [{"pascal": "C"}]},
                {"present_smells": [], "absent_smells": [{"pascal": "D"}]},
            ]
        ),
        encoding="utf-8",
    )
    pairs, entries = expected_pairs(ground_truth, limit=None)
    assert (pairs, entries) == (4, 2)  # (2+1) + (0+1)


def test_expected_pairs_honours_limit(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth.json"
    ground_truth.write_text(
        json.dumps(
            [
                {"present_smells": [{"pascal": "A"}], "absent_smells": []},
                {"present_smells": [{"pascal": "B"}], "absent_smells": [{"pascal": "C"}]},
            ]
        ),
        encoding="utf-8",
    )
    pairs, entries = expected_pairs(ground_truth, limit=1)  # truncates to entries[:1]
    assert (pairs, entries) == (1, 1)


def test_count_results_groups_by_model_alias(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    for name in (
        "R@wf1@D1@rep1@s@deepseek-v4-flash.json",
        "R@wf2@D1@rep1@s@deepseek-v4-flash.json",
        "R@wf1@D1@rep1@s@gemma4.json",
    ):
        (results / name).write_text("{}", encoding="utf-8")

    counts = _count_results(tmp_path)
    assert counts["deepseek-v4-flash"] == 2
    assert counts["gemma4"] == 1


def test_done_keys_reads_result_identity_tuples(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "R@wf@D2@rep1@s@deepseek-v4-flash.json").write_text(
        json.dumps({"file_key": "R@wf", "target_smell": "Avoid X", "model_alias": "deepseek-v4-flash"}),
        encoding="utf-8",
    )
    assert _done_keys(tmp_path) == {("R@wf", "Avoid X", "deepseek-v4-flash")}


def _write_log(dir_: Path, entries: list) -> None:
    (dir_ / "error.log").write_text(
        "\n".join(e if isinstance(e, str) else json.dumps(e) for e in entries), encoding="utf-8"
    )


def test_count_failures_attributes_and_buckets_unmapped(tmp_path: Path):
    alias_by_id = {"deepseek/deepseek-v4-flash": "deepseek-v4-flash", "google/gemma-4-31b-it": "gemma4"}
    _write_log(
        tmp_path,
        [
            {"file": "a", "target_smell": "S1", "category": "structured_output",
             "model_ids": {"detection": "deepseek/deepseek-v4-flash"}},
            {"file": "b", "target_smell": "S2", "category": "rate_limit",
             "model_ids": {"detection": "google/gemma-4-31b-it"}},
            {"file": "c", "target_smell": "S3", "category": "unknown",
             "model_ids": {"detection": "who/knows"}},  # unmapped -> "?"
            "not json — must be skipped",
        ],
    )
    failures = _count_failures(tmp_path, alias_by_id, done_keys=set())
    assert failures["deepseek-v4-flash"]["structured_output"] == 1
    assert failures["gemma4"]["rate_limit"] == 1
    assert failures["?"]["unknown"] == 1  # unmapped model id falls into the "?" bucket


def test_count_failures_excludes_pairs_that_since_succeeded(tmp_path: Path):
    """Append-only error.log keeps a fixed pair's stale line; reconciliation must
    drop it once a result JSON exists for that (file, smell, model)."""
    alias_by_id = {"deepseek/deepseek-v4-flash": "deepseek-v4-flash"}
    _write_log(
        tmp_path,
        [
            {"file": "fixed", "target_smell": "S", "category": "rate_limit",
             "model_ids": {"detection": "deepseek/deepseek-v4-flash"}},
            {"file": "stillbad", "target_smell": "S", "category": "structured_output",
             "model_ids": {"detection": "deepseek/deepseek-v4-flash"}},
        ],
    )
    done = {("fixed", "S", "deepseek-v4-flash")}  # 'fixed' now has a result JSON
    failures = _count_failures(tmp_path, alias_by_id, done_keys=done)
    assert failures["deepseek-v4-flash"] == Counter({"structured_output": 1})  # only the survivor


def test_count_failures_dedups_append_only_retries(tmp_path: Path):
    """The same pair failing on two runs is ONE current hole, categorized by its
    most recent failure — not double-counted from the append-only history."""
    alias_by_id = {"deepseek/deepseek-v4-flash": "deepseek-v4-flash"}
    _write_log(
        tmp_path,
        [
            {"file": "p", "target_smell": "S", "category": "rate_limit",
             "model_ids": {"detection": "deepseek/deepseek-v4-flash"}},
            {"file": "p", "target_smell": "S", "category": "structured_output",
             "model_ids": {"detection": "deepseek/deepseek-v4-flash"}},  # later attempt
        ],
    )
    failures = _count_failures(tmp_path, alias_by_id, done_keys=set())
    assert failures["deepseek-v4-flash"] == Counter({"structured_output": 1})  # once, latest category


def test_count_failures_unmapped_reconciled_on_file_smell(tmp_path: Path):
    """An unmappable model id (alias '?') can't match a full result key, so it is
    reconciled on (file, smell): dropped when that pair succeeded under any model
    (otherwise an unattributable stale line would show as a phantom hole)."""
    _write_log(
        tmp_path,
        [
            {"file": "ghost", "target_smell": "S", "category": "unknown",
             "model_ids": {"detection": "who/knows"}},
        ],
    )
    # same (file, smell) has a result under a real model → phantom, dropped
    assert _count_failures(tmp_path, {}, done_keys={("ghost", "S", "deepseek-v4-flash")}) == {}
    # no matching result → genuinely unreconcilable, kept as "?"
    kept = _count_failures(tmp_path, {}, done_keys=set())
    assert kept["?"]["unknown"] == 1


def test_count_failures_empty_when_no_log(tmp_path: Path):
    assert _count_failures(tmp_path, {}, done_keys=set()) == {}
