from __future__ import annotations

import json
from pathlib import Path

from dataset_utils.hashing import compute_file_hash


class PipelineIntegrityError(RuntimeError):
    pass


def assert_input_matches_report(
    input_path: Path,
    prior_report_path: Path,
    label: str,
) -> str:
    if not input_path.exists():
        msg = f"{label}: input file missing: {input_path}"
        raise PipelineIntegrityError(msg)

    current_sha = compute_file_hash(input_path)

    if not prior_report_path.exists():
        msg = (
            f"{label}: upstream report missing: {prior_report_path}. "
            f"Run the previous step first to produce it."
        )
        raise PipelineIntegrityError(msg)

    try:
        with prior_report_path.open("r", encoding="utf-8") as f:
            prior = json.load(f)
    except json.JSONDecodeError as exc:
        msg = f"{label}: upstream report {prior_report_path} is not valid JSON: {exc}"
        raise PipelineIntegrityError(msg) from exc

    expected_sha = prior.get("output_sha256")
    if not expected_sha:
        msg = (
            f"{label}: upstream report {prior_report_path} has no 'output_sha256' field. "
            f"Regenerate the upstream report with a compatible producing step."
        )
        raise PipelineIntegrityError(msg)

    if current_sha != expected_sha:
        msg = (
            f"{label}: input SHA drift detected.\n"
            f"  input file : {input_path}\n"
            f"  current sha: {current_sha}\n"
            f"  expected   : {expected_sha} (from {prior_report_path})\n"
            f"Re-run the producing step to regenerate the input."
        )
        raise PipelineIntegrityError(msg)

    return current_sha


def hash_source_file(path: Path, label: str) -> str:
    if not path.exists():
        msg = f"{label}: source file missing: {path}"
        raise PipelineIntegrityError(msg)
    return compute_file_hash(path)
