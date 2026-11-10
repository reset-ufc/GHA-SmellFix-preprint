"""Tests for utils.workflow_feasibility — detection outcomes, smell sets, patch review."""

from __future__ import annotations

import pandas as pd
import pytest
from utils.workflow_feasibility import (
    _parses,
    pair_metrics,
    pair_table,
    refactoring_summary,
    review_sheet,
    validate_review,
    workflow_sets,
    workflow_summary,
    write_diffs,
)

ORIGINAL = "on: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
PATCHED = ORIGINAL + "    timeout-minutes: 10\n"
GROUND_TRUTH = [
    {
        "file_key": "w1.yml",
        "repository": "o/r",
        "file_path": ".github/w1.yml",
        "yaml_content": ORIGINAL,
    }
]
ANNOTATIONS = pd.DataFrame(
    {
        "workflow_id": ["w1.yml"] * 5,
        "target_smell": ["A", "B", "C", "D", "E"],
        "manual_label": ["present", "present", "absent", "uncertain", "absent"],
        "number_of_instances": ["2", "1", "", "", ""],
        "expected_refactoring_scope": ["jobs.a.timeout-minutes", "x", "", "", ""],
    }
)


def _pairs(tmp_path, runs: dict[str, tuple[bool, str] | None]) -> pd.DataFrame:
    """``runs`` maps smell -> (verdict, candidate YAML); None = the run is missing."""
    patched_dir = tmp_path / "out" / "design_d1" / "patched_workflows"
    patched_dir.mkdir(parents=True)
    rows = []
    for smell, run in runs.items():
        if run is None:
            continue
        (patched_dir / f"w1@D1@rep1@{smell}.yml").write_text(run[1], encoding="utf-8")
        rows.append(
            {
                "design": "D1",
                # The result JSON carries both: a stem workflow_id and the file_key.
                "workflow_id": "w1",
                "file_key": "w1.yml",
                "target_smell": smell,
                "smell_detected": run[0],
                "patch_valid": True,
                "detection_justification": "",
                "refactor_justification": "",
                "run_duration_seconds": 1.0,
                "model_alias": "haiku45",
                "token_usage": {"total": {"input_tokens": 1000, "output_tokens": 200}},
                "file": f"w1@D1@rep1@{smell}.json",
                "source_dir": "out",
            }
        )
    pairs = pair_table(pd.DataFrame(rows), ANNOTATIONS, GROUND_TRUTH, tmp_path)
    pairs["diff_path"] = write_diffs(pairs, GROUND_TRUTH, tmp_path, tmp_path / "art" / "diffs")
    return pairs


def test_detection_scores_only_certain_pairs_with_a_run(tmp_path):
    # A hit, B missing run, C false alarm, D uncertain, E correct rejection.
    pairs = _pairs(
        tmp_path,
        {
            "A": (True, PATCHED),
            "B": None,
            "C": (True, ORIGINAL),
            "D": (True, PATCHED),
            "E": (False, ORIGINAL),
        },
    )
    outcome = pairs[pairs["design"] == "D1"].set_index("target_smell")["outcome"]
    assert list(outcome[["A", "C", "E"]]) == ["TP", "FP", "TN"]
    assert pd.isna(outcome["B"]) and pd.isna(outcome["D"])

    # D2 never ran: every one of its pairs is a missing run, not a silently absent design.
    [m, m2] = pair_metrics(pairs).to_dict("records")
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 0, 1)
    assert m["n_uncertain"] == 1 and m["n_missing_runs"] == 1
    assert m2["design"] == "D2" and m2["n_missing_runs"] == 5
    assert pairs["estimated_cost"].dropna().iloc[0] == pytest.approx(0.002)

    [s, _] = workflow_sets(pairs).to_dict("records")
    # The missing run on B is not a missed smell.
    assert s["annotated"] == ["A"] and s["predicted"] == ["A", "C"]
    assert s["missed"] == [] and s["extra"] == ["C"] and s["n_unscored"] == 2
    assert workflow_summary(workflow_sets(pairs))["mean_extra"].iloc[0] == 1


def test_refactoring_outcomes_and_review_round_trip(tmp_path):
    # A patched; B detected but only a comment and blank lines changed; C rejected yet rewritten;
    # D uncertain but patched; E abstains.
    pairs = _pairs(
        tmp_path,
        {
            "A": (True, PATCHED),
            "B": (True, "# note\n\n" + ORIGINAL + "\n\n"),
            "C": (False, PATCHED),
            "D": (True, PATCHED),
            "E": (False, ORIGINAL),
        },
    )
    d1 = pairs[pairs["design"] == "D1"].set_index("target_smell")
    assert list(d1["refactoring_outcome"]) == [
        "review",
        "no_patch_abstention",
        "unnecessary_patch",
        "uncertain_patch",
        "correct_abstention",
    ]
    assert d1["diff_path"]["C"].endswith("D1@w1.yml@C.diff")

    sheet = review_sheet(pairs, None)
    assert list(sheet["target_smell"]) == ["A", "C", "D"]
    assert sheet.loc[0, ["number_of_instances", "expected_refactoring_scope"]].tolist() == [
        "2",
        "jobs.a.timeout-minutes",
    ]
    sheet.loc[sheet["target_smell"] == "A", "feasibility_category"] = "plausible"
    sheet.loc[sheet["target_smell"] == "C", "feasibility_category"] = "localized_unnecessary"
    validate_review(sheet)

    # Re-running keeps the hand-filled answers.
    kept = review_sheet(pairs, sheet.astype(str))
    assert list(kept["feasibility_category"]) == ["plausible", "localized_unnecessary", ""]
    [present], [absent] = (f.to_dict("records") for f in refactoring_summary(pairs, kept))
    assert (present["present_runs"], present["patches_generated"]) == (2, 1)
    assert (present["no_patch_abstention"], present["no_patch_missed_detection"]) == (1, 0)
    assert present["excluded_uncertain_patches"] == 1
    assert present["plausible"] == 1 and present["plausible_refactoring_rate"] == 0.5
    assert (absent["correct_abstentions"], absent["unnecessary_patches"]) == (1, 1)
    assert absent["localized_unnecessary"] == 1

    sheet.loc[sheet["target_smell"] == "A", "feasibility_category"] = "localized_unnecessary"
    with pytest.raises(ValueError, match="feasibility_category"):
        validate_review(sheet)

    orphan = sheet.assign(target_smell="Z", feasibility_category="plausible").astype(str)
    with pytest.raises(ValueError, match="no longer have a generated patch"):
        review_sheet(pairs, orphan)


def test_only_a_mapping_counts_as_valid_yaml():
    assert _parses(ORIGINAL)
    assert not _parses("I cannot find this smell in the workflow.")
    assert not _parses("- a\n- b\n") and not _parses("") and not _parses("a: [")
