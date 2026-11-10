#!/usr/bin/env python3
"""Build the smoke-test ground_truth by sampling the final ground_truth.

Source
------
``final_ground_truth.json`` — the single source of truth (new schema). Each entry
carries ``present_smells`` (``{pascal, count}``, smells confirmed present) and
``absent_smells`` (``{pascal}``, smells confirmed absent). The pipeline derives
its evaluation pairs as ``present ∪ absent`` per file, so a smoke slice is just
a balanced subset of those same pairs — no database, no separate provenance.

Selection
---------
For every smell, take up to ``--per-smell`` (default 3) entries where it is
present and up to ``--per-smell`` where it is absent — i.e. K positive + K
negative pairs per smell, so the negative side stays measurable. Candidates are
ordered by ``file_key`` and the first K taken, so the slice is **deterministic
and reproducible without an RNG seed**. A smell with fewer than K candidates on
one polarity contributes whatever it has.

Each emitted entry is the original ground_truth entry with its ``present_smells`` /
``absent_smells`` **filtered to only the sampled (pascal, polarity) targets**,
so the pipeline enumerates exactly the sampled pairs (not every smell the file
carries). An entry sampled for several smells appears once, carrying the union
of its sampled targets.

Output
------
An ground-truth-schema JSON array (same schema as ``final_ground_truth.json``) consumable
directly by ``gha_agents.experiment``.

Usage (from repo root)
----------------------
    python modules/agents_pipeline/scripts/sample_smoke_ground_truth.py
    python modules/agents_pipeline/scripts/sample_smoke_ground_truth.py --per-smell 5
    python modules/agents_pipeline/scripts/sample_smoke_ground_truth.py \
        --source data/12b_final_ground_truth.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = REPO_ROOT / "final_ground_truth.json"
DEFAULT_OUTPUT = REPO_ROOT / "modules" / "agents_pipeline" / "ground_truth-smoke.json"
DEFAULT_PER_SMELL = 3


def sample_smoke_ground_truth(entries: list[dict], per_smell: int) -> list[dict]:
    """Return a balanced K-per-smell-per-polarity slice of ``entries``.

    Deterministic: candidates are ordered by ``file_key`` and the first
    ``per_smell`` taken on each polarity. Emitted entries carry only the sampled
    targets in their ``present_smells`` / ``absent_smells``.
    """
    # smell -> polarity -> ordered list of file_keys exhibiting it
    candidates: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"present": [], "absent": []}
    )
    by_key: dict[str, dict] = {}
    for entry in sorted(entries, key=lambda e: e.get("file_key", "")):
        fk = entry.get("file_key", "")
        by_key[fk] = entry
        for smell in entry.get("present_smells", []):
            candidates[smell["pascal"]]["present"].append(fk)
        for smell in entry.get("absent_smells", []):
            candidates[smell["pascal"]]["absent"].append(fk)

    # file_key -> {"present": {pascals}, "absent": {pascals}} sampled targets
    selected: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"present": set(), "absent": set()}
    )
    for pascal in sorted(candidates):
        for polarity in ("present", "absent"):
            for fk in candidates[pascal][polarity][:per_smell]:
                selected[fk][polarity].add(pascal)

    smoke: list[dict] = []
    for fk in sorted(selected):
        entry = by_key[fk]
        keep = selected[fk]
        sliced = dict(entry)
        sliced["present_smells"] = [
            s for s in entry.get("present_smells", []) if s["pascal"] in keep["present"]
        ]
        sliced["absent_smells"] = [
            s for s in entry.get("absent_smells", []) if s["pascal"] in keep["absent"]
        ]
        smoke.append(sliced)
    return smoke


def _summarize(smoke: list[dict]) -> None:
    pos: dict[str, int] = defaultdict(int)
    neg: dict[str, int] = defaultdict(int)
    for entry in smoke:
        for s in entry["present_smells"]:
            pos[s["pascal"]] += 1
        for s in entry["absent_smells"]:
            neg[s["pascal"]] += 1
    total_pos = sum(pos.values())
    total_neg = sum(neg.values())
    print(f"  entries : {len(smoke)}")
    print(f"  pairs   : {total_pos + total_neg}  ({total_pos} pos / {total_neg} neg)")
    print(f"  smells  : {len(set(pos) | set(neg))}")
    for pascal in sorted(set(pos) | set(neg)):
        print(f"    {pascal:<32} {pos.get(pascal, 0):>2} pos / {neg.get(pascal, 0):>2} neg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Final ground_truth JSON to sample (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Smoke ground_truth JSON to write (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--per-smell",
        type=int,
        default=DEFAULT_PER_SMELL,
        help=f"Max entries per smell per polarity (default: {DEFAULT_PER_SMELL}).",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Source ground_truth not found: {args.source}\n"
            "Place final_ground_truth.json at the repo root, or pass "
            "--source data/12b_final_ground_truth.json."
        )
    entries = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("Source ground_truth must be a JSON array at the top level.")

    smoke = sample_smoke_ground_truth(entries, args.per_smell)
    args.output.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    print(f"Smoke ground_truth -> {args.output}")
    _summarize(smoke)


if __name__ == "__main__":
    main()
