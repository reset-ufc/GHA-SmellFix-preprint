"""Aggregate the workflow-level feasibility diagnostic (8 workflows x 19 smells x D1/D2).

Implements the protocol in
``modules/agents_pipeline/docs/workflow_level_feasibility_detection_refactoring_protocol.md``.
Descriptive only, by design: pair-level TP/FP/FN/TN with precision/recall/F1,
workflow-level smell sets (exact/partial match, missed, extra), and the refactoring
summaries fed by a hand-filled patch review sheet. ``uncertain`` annotations and missing
runs are counted, never scored. Ground truth is the consensus annotation CSV, not the
``ground_truth_positive`` field of the runs (which folds ``uncertain`` into absent).

    uv run python -m utils.workflow_feasibility
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import pandas as pd
import yaml

from utils.cost_latency import add_cost
from utils.dataloader import ROOT, load_results
from utils.designs import DESIGNS, run_dir
from utils.refactored_instances import _normalize_yaml

FEASIBILITY_DIR = ROOT / "data" / "workflow_feasibility"
OUT_DIR = ROOT / "data" / "artifacts" / "workflow_feasibility"
LABELS = ("present", "absent", "uncertain")

# Feasibility categories 1-4; category 5 (no patch) and the unnecessary-patch label for
# absent smells are derived from the runs, not typed by the reviewer.
PRESENT_CATEGORIES = ("plausible", "plausible_external_validation", "risky_or_unrelated", "invalid")
# Optional sub-classification of unnecessary patches on manually absent smells.
ABSENT_CATEGORIES = ("localized_unnecessary", "behavior_relevant_unnecessary")
_YES_NO_UNCLEAR = ("yes", "no", "unclear")
REVIEW_CHOICES = {
    "target_smell_appears_addressed": (*_YES_NO_UNCLEAR, "not_applicable"),
    "modifies_expected_construct": ("yes", "no", "partial", "unclear"),
    "modifies_other_smell_construct": _YES_NO_UNCLEAR,
    "requires_external_validation": ("yes", "no"),
    "potential_behavior_impact": _YES_NO_UNCLEAR,
    "other_smell_effect": ("likely_improves", "likely_worsens", "unclear"),
}
REVIEW_KEY = ["design", "workflow_id", "target_smell"]
REVIEW_AUTO_FIELDS = [
    *REVIEW_KEY,
    "repository",
    "workflow_path",
    "manual_label",
    "number_of_instances",
    "expected_refactoring_scope",
    "detection_verdict",
    "patch_generated",
    "yaml_valid",
    "gha_structurally_valid",
    "refactoring_outcome",
    "diff_path",
]
# The protocol's manual review form (minus brief_justification) plus its multi-smell
# interaction notes.
REVIEW_MANUAL_FIELDS = [
    "target_smell_appears_addressed",
    "modifies_expected_construct",
    "modifies_other_smell_construct",
    "requires_external_validation",
    "potential_behavior_impact",
    "feasibility_category",
    "other_smell_name",
    "other_smell_effect",
    "interaction_notes",
]


def load_annotations(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    ann["manual_label"] = ann["manual_label"].str.strip().str.lower()
    unknown = set(ann["manual_label"]) - set(LABELS)
    if unknown:
        msg = f"manual_label must be one of {LABELS}, got {sorted(unknown)}"
        raise ValueError(msg)
    # The annotators' instance count and expected scope go to the patch reviewer; the
    # detection metrics stay per (workflow, smell), as in the protocol.
    columns = ["workflow_id", "smell_pascal", "manual_label", "number_of_instances"]
    return ann[[*columns, "expected_refactoring_scope"]].rename(
        columns={"smell_pascal": "target_smell"}
    )


def _parses(text: str) -> bool:
    """A workflow is a mapping; prose or a bare list also loads as YAML but is not one."""
    try:
        return isinstance(yaml.safe_load(text), dict)
    except yaml.YAMLError:
        return False


def _candidate_path(runs_root: Path, r) -> Path:
    design_dir = runs_root / r.source_dir / run_dir(r.design)
    return design_dir / "patched_workflows" / Path(r.file).with_suffix(".yml").name


def pair_table(
    results: pd.DataFrame, annotations: pd.DataFrame, ground_truth: list[dict], runs_root: Path
) -> pd.DataFrame:
    """One row per annotated (design, workflow, smell): the protocol's stored output per run.

    Missing runs keep a NA verdict. ``patch_generated`` is judged on the candidate itself,
    whatever the verdict (a run can reject the smell and still rewrite the workflow), with
    the main study's parsed-YAML equality, so comment or formatting edits are no patch.
    """
    entries = {e["file_key"]: e for e in ground_truth}
    tokens = results["token_usage"].map(lambda u: (u or {}).get("total") or {})
    runs = pd.DataFrame(
        {
            "design": results["design"],
            "workflow_id": results["file_key"],
            "target_smell": results["target_smell"],
            "detection_verdict": results["smell_detected"].astype("boolean"),
            "gha_structurally_valid": results["patch_valid"],
            "detection_rationale": results["detection_justification"],
            "refactor_rationale": results["refactor_justification"],
            "latency_s": results["run_duration_seconds"],
            "model_alias": results["model_alias"],
            **{
                k: tokens.map(lambda t, k=k: t.get(k, 0) or 0)
                for k in ("input_tokens", "output_tokens", "reasoning_tokens")
            },
            "candidate_path": [_candidate_path(runs_root, r) for r in results.itertuples()],
        }
    )
    runs = add_cost(runs).rename(columns={"cost_usd": "estimated_cost"})
    candidates = runs["candidate_path"].map(
        lambda p: p.read_text(encoding="utf-8") if p.exists() else ""
    )
    originals = runs["workflow_id"].map(lambda w: entries[w]["yaml_content"])
    runs["candidate_exists"] = candidates.str.strip() != ""
    runs["yaml_valid"] = candidates.map(_parses) & runs["candidate_exists"]
    changed = [
        _normalize_yaml(c) != _normalize_yaml(o) for c, o in zip(candidates, originals, strict=True)
    ]
    runs["patch_generated"] = runs["candidate_exists"] & pd.Series(changed, index=runs.index)
    runs["candidate_path"] = runs["candidate_path"].map(lambda p: str(p.relative_to(runs_root)))

    # Fixed design grid: a design with no runs at all shows up as missing runs.
    grid = annotations.merge(pd.DataFrame({"design": DESIGNS}), how="cross")
    pairs = grid.merge(runs, on=REVIEW_KEY, how="left")
    pairs["repository"] = pairs["workflow_id"].map(lambda w: entries[w]["repository"])
    pairs["workflow_path"] = pairs["workflow_id"].map(lambda w: entries[w]["file_path"])

    ran = pairs["detection_verdict"].notna()
    detected = pairs["detection_verdict"].fillna(False).astype(bool)
    generated = pairs["patch_generated"].fillna(False).astype(bool)
    # An empty candidate on a detected smell is an unusable patch (category 4), not category 5.
    attempted = generated | ~pairs["candidate_exists"].fillna(True).astype(bool)
    present, absent = pairs["manual_label"] == "present", pairs["manual_label"] == "absent"
    uncertain = pairs["manual_label"] == "uncertain"
    pairs["outcome"] = pd.NA
    pairs.loc[ran & present & detected, "outcome"] = "TP"
    pairs.loc[ran & present & ~detected, "outcome"] = "FN"
    pairs.loc[ran & absent & detected, "outcome"] = "FP"
    pairs.loc[ran & absent & ~detected, "outcome"] = "TN"
    # "review" rows get feasibility categories 1-4; both "no_patch_*" are category 5, split
    # so a missed detection is told apart from a detected smell left unpatched.
    pairs["refactoring_outcome"] = pd.NA
    pairs.loc[ran & present & detected & attempted, "refactoring_outcome"] = "review"
    pairs.loc[ran & present & ~detected, "refactoring_outcome"] = "no_patch_missed_detection"
    pairs.loc[ran & present & detected & ~attempted, "refactoring_outcome"] = "no_patch_abstention"
    pairs.loc[ran & absent & generated, "refactoring_outcome"] = "unnecessary_patch"
    pairs.loc[ran & absent & ~generated, "refactoring_outcome"] = "correct_abstention"
    # Reviewed like any generated output, but kept out of every metric.
    pairs.loc[ran & uncertain & generated, "refactoring_outcome"] = "uncertain_patch"
    return pairs


def write_diffs(
    pairs: pd.DataFrame, ground_truth: list[dict], runs_root: Path, diff_dir: Path
) -> pd.Series:
    """Unified diff per generated patch; returns the diff path ('' when no patch)."""
    originals = {e["file_key"]: e["yaml_content"] for e in ground_truth}
    diff_dir.mkdir(parents=True, exist_ok=True)
    paths = pd.Series("", index=pairs.index)
    for i, r in pairs[pairs["patch_generated"].fillna(False).astype(bool)].iterrows():
        patched = (runs_root / r["candidate_path"]).read_text(encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                originals[r["workflow_id"]].splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{r['workflow_id']}",
                tofile=f"b/{r['workflow_id']}",
            )
        )
        path = diff_dir / f"{r['design']}@{r['workflow_id']}@{r['target_smell']}.diff"
        path.write_text(diff, encoding="utf-8")
        paths[i] = str(path.relative_to(diff_dir.parent))
    return paths


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def pair_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for design, g in pairs.groupby("design"):
        n = g["outcome"].value_counts().to_dict()
        tp, fp, fn, tn = (n.get(k, 0) for k in ("TP", "FP", "FN", "TN"))
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        rows.append(
            {
                "design": design,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "f1": _ratio(2 * precision * recall, precision + recall),
                "n_uncertain": int((g["manual_label"] == "uncertain").sum()),
                "n_missing_runs": int(g["detection_verdict"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def workflow_sets(pairs: pd.DataFrame) -> pd.DataFrame:
    """Annotated vs. predicted smell sets per (design, workflow) over the scored pairs only.

    Uncertain labels and missing runs are left out of both sets, as in ``pair_metrics``.
    """
    rows = []
    for (design, wf), g in pairs.groupby(["design", "workflow_id"]):
        scored = g[g["outcome"].notna()]
        annotated = set(scored.loc[scored["manual_label"] == "present", "target_smell"])
        predicted = set(scored.loc[scored["detection_verdict"].astype(bool), "target_smell"])
        rows.append(
            {
                "design": design,
                "workflow_id": wf,
                "annotated": sorted(annotated),
                "predicted": sorted(predicted),
                "missed": sorted(annotated - predicted),
                "extra": sorted(predicted - annotated),
                "exact_match": annotated == predicted,
                "partial_match": bool(annotated & predicted),
                "n_unscored": len(g) - len(scored),
            }
        )
    return pd.DataFrame(rows)


def workflow_summary(sets: pd.DataFrame) -> pd.DataFrame:
    return (
        sets.assign(
            n_missed=sets["missed"].map(len),
            n_extra=sets["extra"].map(len),
        )
        .groupby("design")
        .agg(
            workflows=("workflow_id", "count"),
            exact_match=("exact_match", "sum"),
            partial_match=("partial_match", "sum"),
            mean_missed=("n_missed", "mean"),
            mean_extra=("n_extra", "mean"),
        )
        .reset_index()
    )


def review_sheet(pairs: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """One row per generated output (protocol's review form), keeping hand-filled answers.

    Rebuilt on every call so resumed runs join the sheet; raises instead of dropping a
    filled row whose patch is no longer generated.
    """
    todo = pairs["refactoring_outcome"].isin(["review", "unnecessary_patch", "uncertain_patch"])
    sheet = pairs.loc[todo, REVIEW_AUTO_FIELDS].sort_values(REVIEW_KEY).reset_index(drop=True)
    if previous is None:
        return sheet.assign(**dict.fromkeys(REVIEW_MANUAL_FIELDS, ""))
    kept = previous[REVIEW_KEY + REVIEW_MANUAL_FIELDS]
    matched = kept.merge(sheet[REVIEW_KEY], on=REVIEW_KEY, how="left", indicator=True)
    orphans = matched[matched["_merge"] == "left_only"]
    filled = orphans[orphans[REVIEW_MANUAL_FIELDS].ne("").any(axis=1)]
    if not filled.empty:
        msg = "reviewed rows no longer have a generated patch:\n" + filled[REVIEW_KEY].to_string(
            index=False
        )
        raise ValueError(msg)
    merged = sheet.merge(kept, on=REVIEW_KEY, how="left")
    return merged.fillna(dict.fromkeys(REVIEW_MANUAL_FIELDS, ""))


def validate_review(sheet: pd.DataFrame) -> None:
    problems = []
    for field, choices in REVIEW_CHOICES.items():
        bad = set(sheet[field]) - {"", *choices}
        if bad:
            problems.append(f"{field}: {sorted(bad)} not in {choices}")
    categories = (
        ("present", PRESENT_CATEGORIES),
        ("absent", ABSENT_CATEGORIES),
        ("uncertain", PRESENT_CATEGORIES + ABSENT_CATEGORIES),
    )
    for label, choices in categories:
        bad = set(sheet.loc[sheet["manual_label"] == label, "feasibility_category"])
        if bad - {"", *choices}:
            problems.append(f"feasibility_category ({label}): {sorted(bad - {'', *choices})}")
    if problems:
        msg = "patch_review.csv has invalid values:\n" + "\n".join(problems)
        raise ValueError(msg)


def refactoring_summary(
    pairs: pd.DataFrame, sheet: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-design counts and rates for manually present and manually absent smells."""
    category = pairs.merge(sheet[[*REVIEW_KEY, "feasibility_category"]], on=REVIEW_KEY, how="left")
    pairs = pairs.assign(category=category["feasibility_category"].fillna("").to_numpy())
    present_rows, absent_rows = [], []
    for design, g in pairs[pairs["detection_verdict"].notna()].groupby("design"):
        p, a = g[g["manual_label"] == "present"], g[g["manual_label"] == "absent"]
        reviewed = p[p["refactoring_outcome"] == "review"]
        counts = {c: int((reviewed["category"] == c).sum()) for c in PRESENT_CATEGORIES}
        plausible = counts["plausible"] + counts["plausible_external_validation"]
        present_rows.append(
            {
                "design": design,
                "present_runs": len(p),
                "predicted_present": int(p["detection_verdict"].astype(bool).sum()),
                "patches_generated": len(reviewed),
                **counts,
                "unclassified": int((reviewed["category"] == "").sum()),
                **{
                    k: int((p["refactoring_outcome"] == k).sum())
                    for k in ("no_patch_missed_detection", "no_patch_abstention")
                },
                "excluded_uncertain_patches": int(
                    (g["refactoring_outcome"] == "uncertain_patch").sum()
                ),
                "patch_generation_rate": _ratio(len(reviewed), len(p)),
                "plausible_refactoring_rate": _ratio(plausible, len(p)),
                "conditional_plausible_rate": _ratio(plausible, len(reviewed)),
            }
        )
        unnecessary = a[a["refactoring_outcome"] == "unnecessary_patch"]
        absent_rows.append(
            {
                "design": design,
                "absent_runs": len(a),
                "correct_abstentions": int(
                    (a["refactoring_outcome"] == "correct_abstention").sum()
                ),
                "unnecessary_patches": len(unnecessary),
                **{c: int((unnecessary["category"] == c).sum()) for c in ABSENT_CATEGORIES},
                "unnecessary_patch_rate": _ratio(len(unnecessary), len(a)),
            }
        )
    return pd.DataFrame(present_rows), pd.DataFrame(absent_rows)


def _report(
    metrics: pd.DataFrame, summary: pd.DataFrame, present: pd.DataFrame, absent: pd.DataFrame
) -> None:
    """The protocol's D1 vs D2 summary, descriptive only (no tests, no superiority claim)."""
    print("Detection:")
    for m, s in zip(metrics.itertuples(), summary.itertuples(), strict=True):
        print(
            f"  {m.design} recovered {m.TP} of {m.TP + m.FN} manually annotated smell "
            f"instances and suggested {m.FP} additional; "
            f"exact-match workflows {s.exact_match}/{s.workflows}, "
            f"partial {s.partial_match}/{s.workflows}; P={m.precision:.2f} R={m.recall:.2f} "
            f"(uncertain={m.n_uncertain}, missing runs={m.n_missing_runs})"
        )
    print("Refactoring (manually present smells):")
    for p in present.itertuples():
        print(
            f"  {p.design} generated {p.patches_generated} patches: {p.plausible} plausible "
            f"smell-specific, {p.plausible_external_validation} requiring external validation, "
            f"{p.risky_or_unrelated} risky or unrelated, {p.invalid} invalid or unusable, "
            f"{p.unclassified} not yet classified; no patch: {p.no_patch_missed_detection} "
            f"missed detection, {p.no_patch_abstention} abstention"
        )
    print("Abstention (manually absent smells):")
    for a in absent.itertuples():
        print(
            f"  {a.design} generated {a.unnecessary_patches} unnecessary patches over "
            f"{a.absent_runs} runs ({a.behavior_relevant_unnecessary} behavior-relevant)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=FEASIBILITY_DIR / "runs")
    parser.add_argument(
        "--annotations", type=Path, default=FEASIBILITY_DIR / "annotations_consensus.csv"
    )
    parser.add_argument("--ground-truth", type=Path, default=FEASIBILITY_DIR / "ground_truth.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    results = load_results(args.runs)
    if results.empty:
        msg = f"No result JSONs under {args.runs}/*/design_d*/results/"
        raise SystemExit(msg)
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    pairs = pair_table(results, load_annotations(args.annotations), ground_truth, args.runs)
    pairs["diff_path"] = write_diffs(pairs, ground_truth, args.runs, args.out / "diffs")
    metrics = pair_metrics(pairs)
    sets = workflow_sets(pairs)
    summary = workflow_summary(sets)

    # The review sheet is filled by hand: merge, validate, and only then overwrite it.
    review_path = args.out / "patch_review.csv"
    previous = (
        pd.read_csv(review_path, dtype=str, keep_default_na=False) if review_path.exists() else None
    )
    sheet = review_sheet(pairs, previous)
    validate_review(sheet)
    present, absent = refactoring_summary(pairs, sheet)

    args.out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out / "pairs.csv", index=False)
    metrics.to_csv(args.out / "pair_metrics.csv", index=False)
    sets.to_csv(args.out / "workflow_sets.csv", index=False)
    summary.to_csv(args.out / "workflow_summary.csv", index=False)
    sheet.to_csv(review_path, index=False)
    present.to_csv(args.out / "refactoring_present.csv", index=False)
    absent.to_csv(args.out / "refactoring_absent.csv", index=False)
    todo = (sheet["manual_label"] == "present") & (sheet["feasibility_category"] == "")
    print(f"-> {review_path} ({int(todo.sum())} present-smell patches to classify)")
    _report(metrics, summary, present, absent)


if __name__ == "__main__":
    main()
