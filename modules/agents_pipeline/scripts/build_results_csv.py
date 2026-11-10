"""Build per-result CSV directly from output JSONs.

Two outputs, both derived only from per-run JSONs (no ground_truth cross-reference):

  paper_artifacts/csv/results.csv
      One row per result JSON.
      Ground truth uses baked ``ground_truth_positive`` from the JSON, so the CSV
      cannot become orphaned by future ground_truth edits.

  paper_artifacts/csv/iterations.csv
      Long format. One row per per-iteration critic record (detection or refactor);
      empty for the D1/D2 designs, which run no feedback loop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Both tier roots are merged into a single bundle. ``model_alias`` is in
# every result filename, so paths don't collide across tiers.
OUTPUT_DIRS = (REPO_ROOT / "output", REPO_ROOT / "output-slm")
RESULTS_CSV = REPO_ROOT / "paper_artifacts" / "csv" / "results.csv"
ITERATIONS_CSV = REPO_ROOT / "paper_artifacts" / "csv" / "iterations.csv"
DESIGNS = ("d1", "d2")
MODEL_TIER = {
    "gemma": "llm", "glm": "llm",
    "qwen": "slm", "mistral": "slm", "llama": "slm",
}


def patch_metadata(raw: dict) -> dict:
    rel = raw.get("patched_workflow_path")
    if not rel:
        return {"patch_exists": False, "patch_line_count": None}
    p = REPO_ROOT / rel
    if not p.is_file():
        return {"patch_exists": False, "patch_line_count": None}
    try:
        n = sum(1 for _ in p.open())
    except OSError:
        n = None
    return {"patch_exists": True, "patch_line_count": n}


def confusion(pred: bool | None, truth: bool | None) -> str:
    if pred is None or truth is None:
        return "NA"
    if pred and truth:
        return "TP"
    if pred and not truth:
        return "FP"
    if not pred and truth:
        return "FN"
    return "TN"


def role_tokens(by_role: dict, role: str) -> dict:
    r = by_role.get(role) or {}
    return {
        f"tokens_input_{role}": r.get("input_tokens"),
        f"tokens_output_{role}": r.get("output_tokens"),
        f"tokens_cache_read_{role}": r.get("cache_read_tokens"),
        f"tokens_cache_write_{role}": r.get("cache_write_tokens"),
        f"tokens_calls_{role}": r.get("calls"),
    }


def flatten_result(raw: dict, design: str) -> dict:
    tu = raw.get("token_usage") or {}
    tu_total = tu.get("total") or {}
    tu_by_role = tu.get("by_role") or {}
    sampling = raw.get("sampling_settings") or {}
    model_ids = raw.get("model_ids") or {}
    prompt_hashes = raw.get("prompt_hashes") or {}
    pmeta = patch_metadata(raw)
    truth = raw.get("ground_truth_positive")
    pred = raw.get("smell_detected")
    alias = raw.get("model_alias")

    row = {
        # identity
        "file_key": raw.get("file_key"),
        "workflow_id": raw.get("workflow_id"),
        "repository": raw.get("repository", ""),
        "file_path": raw.get("file_path", ""),
        "target_smell_pascal": raw.get("target_smell_pascal"),
        "target_smell_human": raw.get("target_smell"),
        # run identity
        "design": design.upper(),
        "model_alias": alias,
        "model_tier": MODEL_TIER.get(alias, "unknown"),
        "rep": raw.get("repetition"),
        "agent_count": raw.get("agent_count"),
        # ground truth + prediction
        "ground_truth_positive": truth,
        "smell_detected": pred,
        "confusion": confusion(pred, truth),
        "refactor_validation_agrees": raw.get("refactor_validation_agrees"),
        "evidence_start_line": raw.get("evidence_start_line"),
        "evidence_end_line": raw.get("evidence_end_line"),
        # critic loop totals
        "detection_loop_converged": raw.get("detection_loop_converged"),
        "detection_loop_iterations": raw.get("detection_loop_iterations"),
        "refactor_loop_converged": raw.get("refactor_loop_converged"),
        "refactor_loop_iterations": raw.get("refactor_loop_iterations"),
        # patch
        "patched_workflow_path": raw.get("patched_workflow_path", ""),
        "patch_exists": pmeta["patch_exists"],
        "patch_line_count": pmeta["patch_line_count"],
        # timing
        "run_duration_seconds": raw.get("run_duration_seconds"),
        "agent_call_count": raw.get("agent_call_count"),
        "run_failed": raw.get("run_failed"),
        # tokens (total)
        "tokens_input_total": tu_total.get("input_tokens"),
        "tokens_output_total": tu_total.get("output_tokens"),
        "tokens_cache_read_total": tu_total.get("cache_read_tokens"),
        "tokens_cache_write_total": tu_total.get("cache_write_tokens"),
        "tokens_calls_total": tu_total.get("calls"),
    }
    for role in ("detection", "detection_validation", "refactor", "refactor_validation"):
        row.update(role_tokens(tu_by_role, role))
    row.update(
        {
            # reproducibility
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "seed": sampling.get("seed"),
            "model_id_combined": model_ids.get("combined", ""),
            "model_id_detection": model_ids.get("detection", ""),
            "model_id_detection_validation": model_ids.get("detection_validation", ""),
            "model_id_refactor": model_ids.get("refactor", ""),
            "model_id_refactor_validation": model_ids.get("refactor_validation", ""),
            "prompt_hash_combined": prompt_hashes.get("combined", ""),
            "prompt_hash_detection": prompt_hashes.get("detection", ""),
            "prompt_hash_detection_validation": prompt_hashes.get(
                "detection_validation", ""
            ),
            "prompt_hash_refactor": prompt_hashes.get("refactor", ""),
            "prompt_hash_refactor_validation": prompt_hashes.get(
                "refactor_validation", ""
            ),
            # provenance
            "run_started_at": raw.get("run_started_at"),
        }
    )
    return row


def flatten_iterations(raw: dict, design: str) -> list[dict]:
    alias = raw.get("model_alias")
    common = {
        "file_key": raw.get("file_key"),
        "target_smell_pascal": raw.get("target_smell_pascal"),
        "design": design.upper(),
        "model_alias": alias,
        "model_tier": MODEL_TIER.get(alias, "unknown"),
        "rep": raw.get("repetition"),
    }
    rows: list[dict] = []
    for entry in raw.get("detection_loop_iterations_log") or []:
        rows.append(
            {
                **common,
                "loop": "detection",
                "iteration": entry.get("iteration"),
                "agent_smell_detected": entry.get("detection_agent_smell_detected"),
                "validator_agreed": entry.get("validator_agreed"),
                "validated_smell_detected": entry.get("validated_smell_detected"),
                "refactor_validator_agrees": None,
            }
        )
    for entry in raw.get("refactor_loop_iterations_log") or []:
        validation = entry.get("validation") or {}
        rows.append(
            {
                **common,
                "loop": "refactor",
                "iteration": entry.get("iteration"),
                "agent_smell_detected": None,
                "validator_agreed": None,
                "validated_smell_detected": None,
                "refactor_validator_agrees": validation.get("agrees"),
            }
        )
    return rows


def main() -> None:
    result_rows: list[dict] = []
    iter_rows: list[dict] = []
    parse_errors: list[Path] = []
    by_tier: dict[str, int] = {}

    for output_dir in OUTPUT_DIRS:
        if not output_dir.is_dir():
            continue
        for design in DESIGNS:
            results_dir = output_dir / f"design_{design}" / "results"
            if not results_dir.is_dir():
                continue
            for path in sorted(results_dir.glob("*.json")):
                try:
                    with path.open() as fh:
                        raw = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    parse_errors.append(path)
                    continue
                row = flatten_result(raw, design)
                tier = row.get("model_tier") or "unknown"
                by_tier[tier] = by_tier.get(tier, 0) + 1
                result_rows.append(row)
                iter_rows.extend(flatten_iterations(raw, design))

    print(f"Result rows: {len(result_rows)} (by tier: {by_tier})")
    print(f"Iteration rows: {len(iter_rows)}")
    print(f"Parse errors: {len(parse_errors)}")

    if not result_rows:
        raise SystemExit("No result rows — aborting.")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {RESULTS_CSV} ({len(result_rows)} rows, {len(result_rows[0])} cols)")

    if iter_rows:
        with ITERATIONS_CSV.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(iter_rows[0].keys()))
            writer.writeheader()
            writer.writerows(iter_rows)
        print(
            f"Wrote {ITERATIONS_CSV} ({len(iter_rows)} rows, {len(iter_rows[0])} cols)"
        )


if __name__ == "__main__":
    main()
