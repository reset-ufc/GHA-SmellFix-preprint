"""Build the RQ2 manual-validation scope independent of construct filtering.

The validation app needs the full behavioral universe first:

* 570 n15 GT pairs × 4 models × 2 designs × rep1 = 4,560 outputs.
* All GT-positive outputs are in scope for smell-removal assessment.
* GT-negative outputs enter manual inspection only when the YAML really changed.

Construct diffs/trees are still generated for the manual scope, but they are
evidence for inspection, not the filter that decides whether an output is shown.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils import refactored_instances
from utils.construct_diff import construct_trees, unified_diff
from utils.dataloader import ROOT
from utils import detection

_VALIDATION_OUT = ROOT / "data" / "artifacts" / "validation"
_CONSTRUCTS_OUT = ROOT / "data" / "artifacts" / "constructs"
GROUND_TRUTH = ROOT / "data" / "ground_truth_n15.json"


BEHAVIOR_INDEX = _VALIDATION_OUT / "behavior_index_rep1.csv"
MANUAL_INDEX = _VALIDATION_OUT / "rq2_manual_index_rep1.csv"
SUMMARY = _VALIDATION_OUT / "rq2_validation_scope_summary.csv"
ARTIFACT_STATUS = _VALIDATION_OUT / "rq2_validation_artifact_status.csv"


def _boolify(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].astype(bool)
    return df


def _originals() -> dict[str, str]:
    raw = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    return {e["file_key"]: e.get("yaml_content", "") for e in raw if e.get("file_key")}


def _result_path(row: pd.Series) -> Path:
    patched = refactored_instances._patched_path(  # noqa: SLF001 - shared path reconstruction.
        {
            "model_alias": row["model"],
            "design": row["design"],
            "repetition": int(row["rep"]),
            "target_smell": row["smell"],
            "file_key": row["file_key"],
        }
    )
    if patched is None:
        return Path()
    return patched.parent.parent / "results" / f"{row['stem']}.json"


def _result_metadata(row: pd.Series) -> dict[str, str]:
    path = _result_path(row)
    if not path.is_file():
        return {
            "refactor_justification": "",
            "refactor_confidence": "",
            "refactor_confidence_justification": "",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "refactor_justification": "",
            "refactor_confidence": "",
            "refactor_confidence_justification": "",
        }
    return {
        "refactor_justification": data.get("refactor_justification", ""),
        "refactor_confidence": data.get("refactor_confidence", ""),
        "refactor_confidence_justification": data.get("refactor_confidence_justification", ""),
    }


def build_behavior_index(repetition: int = 1) -> pd.DataFrame:
    """Return all rep1 GT-scoped outputs with real modification labels."""
    df = refactored_instances.load_refactored_instances(repetition=repetition)
    df = detection.add_authoritative(df)
    df = df[df["gt_n15"].notna()].copy()
    df = _boolify(
        df,
        ("ground_truth", "gt_n15", "smell_detected", "patched_file_exists", "refactored"),
    )

    out = df.rename(
        columns={
            "model_alias": "model",
            "repetition": "rep",
            "target_smell": "smell",
            "gt_n15": "gt_positive",
        }
    ).copy()
    out["population"] = out["gt_positive"].map({True: "positive", False: "negative"})
    out["expected_action_by_gt"] = out["gt_positive"].map({True: "modify", False: "do_not_modify"})
    out["should_modify_strict_gt_and_detected"] = out["gt_positive"] & out["smell_detected"]
    out["behavior_outcome_by_gt"] = "correctly_not_modified"
    out.loc[out["gt_positive"] & out["refactored"], "behavior_outcome_by_gt"] = "expected_modified"
    out.loc[out["gt_positive"] & ~out["refactored"], "behavior_outcome_by_gt"] = "missed_required_modification"
    out.loc[~out["gt_positive"] & out["refactored"], "behavior_outcome_by_gt"] = "unexpected_modification"
    out["needs_manual_validation"] = out["gt_positive"] | ((~out["gt_positive"]) & out["refactored"])
    out["stem"] = [
        refactored_instances.result_stem(file_key, design, int(rep), smell, model)
        for file_key, design, rep, smell, model in zip(
            out["file_key"], out["design"], out["rep"], out["smell"], out["model"], strict=True
        )
    ]
    out["diff_path"] = "yaml_diffs/" + out["stem"] + ".diff"

    columns = [
        "stem",
        "file_key",
        "smell",
        "design",
        "model",
        "rep",
        "gt_positive",
        "population",
        "ground_truth",
        "smell_detected",
        "patched_file_exists",
        "refactored",
        "patch_valid",
        "expected_action_by_gt",
        "should_modify_strict_gt_and_detected",
        "behavior_outcome_by_gt",
        "needs_manual_validation",
        "diff_path",
    ]
    return out[columns].sort_values(["population", "file_key", "smell", "design", "model"]).reset_index(drop=True)


def _ensure_patch_artifacts(row: pd.Series, original: str) -> dict:
    stem = row["stem"]
    tree_path = _CONSTRUCTS_OUT / "construct_trees" / f"{stem}.json"
    diff_path = _CONSTRUCTS_OUT / "yaml_diffs" / f"{stem}.diff"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.parent.mkdir(parents=True, exist_ok=True)

    patched_path = refactored_instances._patched_path(  # noqa: SLF001 - shared path reconstruction.
        {
            "model_alias": row["model"],
            "design": row["design"],
            "repetition": int(row["rep"]),
            "target_smell": row["smell"],
            "file_key": row["file_key"],
        }
    )
    if patched_path is None or not patched_path.is_file():
        return {
            "stem": stem,
            "status": "missing_patch",
            "parseable": False,
            "n_added": 0,
            "n_removed": 0,
            "n_changed": int(bool(row["refactored"])),
            "tree_exists": tree_path.is_file(),
            "diff_exists": diff_path.is_file(),
        }

    patched = patched_path.read_text(encoding="utf-8")
    if not diff_path.is_file():
        diff_path.write_text(unified_diff(original, patched), encoding="utf-8")

    tree = construct_trees(original, patched)
    if tree is None:
        return {
            "stem": stem,
            "status": "tree_unparseable",
            "parseable": False,
            "n_added": 0,
            "n_removed": 0,
            "n_changed": int(bool(row["refactored"])),
            "tree_exists": tree_path.is_file(),
            "diff_exists": diff_path.is_file(),
        }

    if not tree_path.is_file():
        tree_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stem": stem,
        "status": "ok",
        "parseable": True,
        "n_added": len(tree["added"]),
        "n_removed": len(tree["removed"]),
        "n_changed": len(tree["changed"]),
        "tree_exists": tree_path.is_file(),
        "diff_exists": diff_path.is_file(),
    }


def build_scope_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write behavior index, manual index, artifact status, and summary CSVs."""
    _VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    behavior = build_behavior_index(repetition=1)
    manual = behavior[behavior["needs_manual_validation"]].copy().reset_index(drop=True)
    if manual["stem"].duplicated().any():
        dup = manual[manual["stem"].duplicated(keep=False)]["stem"].tolist()
        raise AssertionError(f"manual validation stems must be unique; duplicates: {dup[:5]}")

    originals = _originals()
    statuses = []
    metadata_rows = []
    for row in manual.itertuples(index=False):
        series = pd.Series(row._asdict())
        original = originals.get(series["file_key"], "")
        statuses.append(_ensure_patch_artifacts(series, original))
        metadata_rows.append(_result_metadata(series))

    status = pd.DataFrame(statuses)
    metadata = pd.DataFrame(metadata_rows)
    manual = pd.concat([manual, status[["parseable", "n_added", "n_removed", "n_changed"]], metadata], axis=1)
    manual["n_diff"] = manual[["n_added", "n_removed", "n_changed"]].fillna(0).sum(axis=1)

    summary = pd.DataFrame(
        [
            {"metric": "rep1_gt_scoped_outputs", "value": len(behavior)},
            {"metric": "manual_scope_outputs", "value": len(manual)},
            {"metric": "manual_scope_positive_outputs", "value": int((manual["population"] == "positive").sum())},
            {"metric": "manual_scope_modified_negative_outputs", "value": int((manual["population"] == "negative").sum())},
            {"metric": "manual_scope_existing_tree", "value": int(status["tree_exists"].sum())},
            {"metric": "manual_scope_existing_diff", "value": int(status["diff_exists"].sum())},
            {"metric": "manual_scope_unparseable_tree", "value": int((status["status"] == "tree_unparseable").sum())},
        ]
    )

    behavior.to_csv(BEHAVIOR_INDEX, index=False)
    manual.to_csv(MANUAL_INDEX, index=False)
    status.to_csv(ARTIFACT_STATUS, index=False)
    summary.to_csv(SUMMARY, index=False)
    return behavior, manual, summary


def main() -> None:
    behavior, manual, summary = build_scope_artifacts()
    print(f"behavior index: {BEHAVIOR_INDEX} ({len(behavior):,} rows)")
    print(f"manual index:   {MANUAL_INDEX} ({len(manual):,} rows)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
