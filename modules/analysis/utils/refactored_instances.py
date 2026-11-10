"""Count instances where the agent actually changed the workflow.

The JSON ``patched_workflow_path`` field is stale (it points at the old
``output-<model>/...`` tree), so this module rebuilds the real path from
``model_alias``, ``design``, ``repetition``, ``target_smell`` and
``file_key`` and compares the patched YAML with the original workflow from the
n15 ground truth.

Example:

    from utils import refactored_instances as ri

    df = ri.load_refactored_instances()
    print(df["refactored"].sum())                         # total changed
    print(df.groupby("model_alias")["refactored"].sum())  # per model
    print(df.groupby(["model_alias", "design"])["refactored"].sum())
    print(df.groupby("target_smell")["refactored"].sum())  # per smell
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from utils import detection
from utils.dataloader import ROOT, DATA_ROOT
from utils.designs import run_dir

GT_PATH = ROOT / "data" / "ground_truth_n15.json"

# The result directories in DATA_ROOT use a slightly different basename than
# model_alias. This mapping lets us locate the patched_workflows/ folder.
_MODEL_DIR = {
    "deepseek-v4-flash": "output-final-deepseek",
    "gemini35-flash": "output-final-gemini-n15",
    "gemma4": "output-final-gemma",
    "gpt54-nano": "output-final-gpt-n15",
    "haiku45": "output-final-haiku-n15",
}


def result_stem(file_key: str, design: str, rep: int, smell: str, model: str) -> str:
    """Return the result/patch stem used on disk, preserving synthetic markers.

    Synthetic GT entries are keyed as ``workflow.yml##synth`` but the generated
    files encode that marker as a final ``@synth`` suffix:
    ``workflow@D1@rep1@smell@model@synth.yml``. Plain ``Path(file_key).stem``
    drops the marker and can collide with the real workflow, so all reconstructed
    stems should go through this helper.
    """
    synth = str(file_key).endswith("##synth")
    raw_file_key = str(file_key).removesuffix("##synth")
    file_stem = Path(raw_file_key).stem
    suffix = "@synth" if synth else ""
    return f"{file_stem}@{design}@rep{rep}@{smell.lower()}@{model}{suffix}"


def _stringify_keys(obj):
    """Recursively convert mapping keys to strings so mixed-type keys
    (e.g., ``True`` vs ``'True'``) can be sorted for stable comparison."""
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj


def _normalize_yaml(text: str) -> str:
    """String-normalized YAML for stable equality comparison."""
    try:
        parsed = yaml.safe_load(text)
    except Exception:
        return str(text).strip()
    parsed = _stringify_keys(parsed)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)


def _patched_path(row: dict) -> Path | None:
    """Reconstruct the actual patched-workflow path from result metadata."""
    model = row.get("model_alias")
    design = row.get("design")
    rep = row.get("repetition")
    smell = row.get("target_smell")
    file_key = row.get("file_key")
    if not all((model, design, rep is not None, smell, file_key)):
        return None

    model_dir = _MODEL_DIR.get(model)
    if model_dir is None:
        return None

    stem = result_stem(file_key, design, rep, smell, model)
    return DATA_ROOT / model_dir / run_dir(design) / "patched_workflows" / f"{stem}.yml"


def load_refactored_instances(
    data: Path = DATA_ROOT,
    gt_path: Path = GT_PATH,
    exclude: tuple[str, ...] = ("gemma4",),
    repetition: int | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Return one row per result JSON with a ``refactored`` boolean flag.

    ``refactored == True`` means the patched workflow content differs from the
    original workflow in the n15 GT. Only instances that are part of the n15 GT
    are kept, matching ``detection.load_results`` semantics.

    ``repetition`` can be used to restrict to specific runs, e.g. ``repetition=1``.
    """
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    originals = {e["file_key"]: e["yaml_content"] for e in gt}
    gt_pairs = {
        (e["file_key"], s["pascal"])
        for e in gt
        for group in ("present_smells", "absent_smells")
        for s in e.get(group, [])
    }

    rep_filter = repetition
    if isinstance(repetition, int):
        rep_filter = (repetition,)

    rows = []
    for f in sorted(Path(data).glob("*/design_*/results/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        model = d.get("model_alias")
        if model in exclude:
            continue

        rep = d.get("repetition")
        if rep_filter is not None and rep not in rep_filter:
            continue

        file_key = d.get("file_key")
        smell = d.get("target_smell")
        if (file_key, smell) not in gt_pairs:
            continue

        patched_file = _patched_path(d)
        original = originals.get(file_key, "")
        patched = patched_file.read_text(encoding="utf-8") if patched_file and patched_file.exists() else None

        refactored = False
        if patched is not None:
            refactored = _normalize_yaml(original) != _normalize_yaml(patched)

        rows.append(
            {
                "model_alias": model,
                "design": d.get("design"),
                "repetition": rep,
                "file_key": file_key,
                "target_smell": d.get("target_smell"),
                "ground_truth": d.get("ground_truth"),
                "smell_detected": d.get("smell_detected"),
                "patch_valid": d.get("patch_valid"),
                "patched_file_exists": patched is not None,
                "refactored": refactored,
            }
        )

    df = pd.DataFrame(rows)
    # A result file can appear in two output dirs once gap-fill / top-up runs are
    # merged alongside originals (e.g. output-final-deepseek + output-*-fill).
    # Keep one row per (model, design, rep, file_key, smell) to match
    # ``detection.load_results`` and avoid inflating counts.
    df = df.drop_duplicates(
        subset=["model_alias", "design", "repetition", "file_key", "target_smell"],
        keep="first",
    ).reset_index(drop=True)
    return df


def _selfcheck() -> None:
    for rep_label, rep_filter in [("all reps", None), ("rep1 only", 1)]:
        df = load_refactored_instances(repetition=rep_filter)
        total = len(df)
        changed = int(df["refactored"].sum())
        pct = changed / total * 100 if total else 0
        print(f"\n=== {rep_label} ===")
        print(f"n15 instances: {total:,}")
        print(f"refactored (real change): {changed:,} ({pct:.1f}%)")
        print("any patched file missing:", (~df["patched_file_exists"]).any())

        print("\nBy model:")
        print(df.groupby("model_alias")["refactored"].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "refactored"}))

        print("\nBy model x design:")
        print(
            df.groupby(["model_alias", "design"])["refactored"]
            .sum()
            .unstack(fill_value=0)
        )

        print("\nBy smell:")
        print(df.groupby("target_smell")["refactored"].sum().sort_values(ascending=False))


if __name__ == "__main__":
    _selfcheck()
