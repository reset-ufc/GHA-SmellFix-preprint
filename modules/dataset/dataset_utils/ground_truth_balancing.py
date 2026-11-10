"""Balance the human-validated ground truth: cap each (smell, polarity) bucket
to N, then trim each smell to min(present, absent) for an exact per-smell
balance.

Extracted from ``12b_balanced_ground_truth.ipynb`` so the notebook, the
parametric-N variant generator, and the unit tests all share ONE
implementation. Split into:

  * :func:`balance` - PURE. Takes the in-memory unbalanced ground truth + a
    priority lookup and returns the balanced list plus the report body. No IO,
    no hashing - this is what the tests exercise directly.
  * :func:`run` - IO wrapper. Verifies upstream integrity (optional), loads the
    inputs, calls :func:`balance`, and writes the balanced ground truth + report
    with input/output hashes.

Cap-then-min semantics (the balanced ground truth):
  each smell row ends at ``min(N, present, absent)`` on BOTH polarities. The cap
  N only bites when a bucket already exceeds N (N mirrors the paper target). The
  fine present/absent balance is decided manually upstream; this step enforces
  the symmetric floor and records what it dropped.

Tiebreak when a bucket exceeds its target: keep lowest Step-10 ``priority``
(highest sampling rank), then ``file_key``. Priority is recovered from
``evaluation_targets`` in the Step-10 / Step-10b samples.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataset_utils.ground_truth_sampling import N_PER_POLARITY, SMELL_CATALOG

# Cap target per (smell, polarity). Mirrors the paper's N_PER_POLARITY.
N = N_PER_POLARITY

_PAPER_ID = {pascal: meta[0] for pascal, meta in SMELL_CATALOG.items()}
# evaluation_targets polarity -> ground-truth polarity bucket.
_POL = {"positive": "present", "negative": "absent"}


def priority_lookup(sample_path: Path, resample_path: Path) -> dict:
    """(file_key, pascal, polarity) -> Step-10 priority (lower = higher rank)."""
    pri: dict[tuple, int] = {}
    for path in (sample_path, resample_path):
        with Path(path).open(encoding="utf-8") as f:
            for e in json.load(f):
                fk = e["file_key"]
                for t in e.get("evaluation_targets", []):
                    pol = _POL.get(t.get("polarity"))
                    if pol is None:
                        continue
                    pri.setdefault((fk, t["pascal"], pol), t.get("priority", 10**9))
    return pri


# --- Shared selection core --------------------------------------------------
# ONE implementation of "bucket by (smell, polarity)" and "sort by Step-10
# priority then file_key, keep first n". Both the primary balancer (`balance`)
# and the synthesis-target selector (`select_targets`) call these, so the choice
# of WHICH instances win is provably identical across the two steps.

def _bucketize(ground_truth: list[dict], priority: dict) -> dict[tuple, list]:
    """(smell, polarity) -> [(priority, file_key), ...] (unsorted)."""
    buckets: dict[tuple, list] = {}
    for e in ground_truth:
        fk = e["file_key"]
        for s in e.get("present_smells", []):
            buckets.setdefault((s["pascal"], "present"), []).append(
                (priority.get((fk, s["pascal"], "present"), 10**9), fk))
        for s in e.get("absent_smells", []):
            buckets.setdefault((s["pascal"], "absent"), []).append(
                (priority.get((fk, s["pascal"], "absent"), 10**9), fk))
    return buckets


def _cap_bucket(instances: list, n: int) -> list:
    """Sort by (priority ASC, file_key ASC) and keep the first ``n``."""
    return sorted(instances, key=lambda x: (x[0], x[1]))[:n]


def balance(
    ground_truth: list[dict],
    priority: dict | None = None,
    *,
    n: int = N,
    balance_to_min: bool = True,
) -> tuple[list[dict], dict]:
    """Cap each (smell, polarity) bucket to ``n``, then (optionally) trim each
    smell to ``min(present, absent)``. PURE: no IO, no hashing.

    ``priority`` maps ``(file_key, pascal, polarity) -> rank`` (lower kept
    first); missing entries sort last. Returns ``(balanced_entries, report)``
    where ``report`` holds the audit body (no hashes / output path).
    """
    priority = priority or {}

    # 1. Collect instances per (smell, polarity): each = (priority, file_key).
    buckets = _bucketize(ground_truth, priority)

    # 2. Cap each bucket to n (lowest priority first, then file_key).
    survivors: dict[tuple, list] = {key: _cap_bucket(insts, n) for key, insts in buckets.items()}

    # 3. Optional exact balance to min(present, absent) per smell.
    if balance_to_min:
        smells = {s for s, _ in survivors}
        for smell in smells:
            p = survivors.get((smell, "present"), [])
            a = survivors.get((smell, "absent"), [])
            t = min(len(p), len(a))
            survivors[(smell, "present")] = p[:t]
            survivors[(smell, "absent")] = a[:t]

    keep: set = set()
    dropped: list[dict] = []
    for key, insts in buckets.items():
        smell, pol = key
        kept_fks = {fk for _, fk in survivors.get(key, [])}
        for prio, fk in insts:
            if fk in kept_fks:
                keep.add((fk, smell, pol))
            else:
                dropped.append({"file_key": fk, "smell": smell, "polarity": pol, "priority": prio})

    # 4. Rebuild entries with only surviving smells; drop entries left empty.
    out: list[dict] = []
    for e in ground_truth:
        fk = e["file_key"]
        present = [s for s in e.get("present_smells", []) if (fk, s["pascal"], "present") in keep]
        absent = [s for s in e.get("absent_smells", []) if (fk, s["pascal"], "absent") in keep]
        if not present and not absent:
            continue
        ne = dict(e)
        ne["present_smells"] = present
        ne["absent_smells"] = absent
        out.append(ne)
    out.sort(key=lambda e: e["file_key"])

    # 5. Tallies before/after for the report.
    def _tally(survivor_map):
        per = {}
        for (smell, pol), insts in survivor_map.items():
            per.setdefault(smell, {"present": 0, "absent": 0})[pol] = len(insts)
        return per

    before = _tally(buckets)
    after = _tally(survivors)
    before = {s: before[s] for s in sorted(before, key=lambda s: _PAPER_ID.get(s, 999))}
    after = {s: after[s] for s in sorted(after, key=lambda s: _PAPER_ID.get(s, 999))}
    deficits = {
        s: {
            "present_deficit": max(0, n - v["present"]),
            "absent_deficit": max(0, n - v["absent"]),
        }
        for s, v in after.items()
        if v["present"] < n or v["absent"] < n
    }

    report = {
        "variant": "balanced",
        "n_per_polarity_cap": n,
        "balance_to_min": balance_to_min,
        "n_entries_in": len(ground_truth),
        "n_entries_out": len(out),
        "instances_dropped": len(dropped),
        "per_smell_before": before,
        "per_smell_after": after,
        "deficits_vs_cap": deficits,
        "dropped_instances": sorted(
            dropped, key=lambda d: (d["smell"], d["polarity"], d["priority"], d["file_key"])
        ),
    }
    return out, report


# --- IO wrapper -------------------------------------------------------------

INPUT_PATH = Path("data/12_ground_truth_unbalanced.json")
GROUND_TRUTH_SAMPLE_PATH = Path("data/10_ground_truth_sample.json")
RESAMPLE_PATH = Path("data/10b_ground_truth_resample.json")
OUTPUT_PATH = Path("data/12b_final_ground_truth.json")
REPORT_PATH = Path("data/step12b_report.json")


def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
    *,
    n: int = N,
    balance_to_min: bool = True,
    sample_path: Path = GROUND_TRUTH_SAMPLE_PATH,
    resample_path: Path = RESAMPLE_PATH,
    verify_integrity: bool = True,
) -> tuple[Path, dict]:
    """Load the unbalanced ground truth, balance it, and write the result +
    report. ``verify_integrity`` checks the input against ``step12_report.json``
    (off for unit tests that feed synthetic inputs)."""
    from dataset_utils.hashing import compute_file_hash

    input_path = Path(input_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    if verify_integrity:
        from dataset_utils.integrity import assert_input_matches_report

        assert_input_matches_report(
            input_path, Path("data/step12_report.json"), "step 12b: ground_truth_unbalanced"
        )

    with input_path.open(encoding="utf-8") as f:
        ground_truth: list[dict] = json.load(f)
    pri = priority_lookup(sample_path, resample_path)

    out, report = balance(ground_truth, pri, n=n, balance_to_min=balance_to_min)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    report = {
        **report,
        "input_sha256": {
            "unbalanced_ground_truth": compute_file_hash(input_path),
            "ground_truth_sample": compute_file_hash(sample_path),
            "resample": compute_file_hash(resample_path),
        },
        "output_file": str(output_path),
        "output_sha256": compute_file_hash(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path, report


# --- Synthesis-target selection (Step 12c) ----------------------------------
# Select up to N instances of ONE (smell, polarity) bucket as injection targets
# for synthetic-smell generation. Uses the SAME priority sort + cap as `balance`
# (via the shared core above), with NO min-balancing: the goal is the full N
# targets, not a Pos==Neg row. Output records are in the 12b_final schema so the
# result can later be merged into data/12b_final_ground_truth.json.

POLARITIES = ("present", "absent")
# Convenience aliases so callers may pass paper-style polarity names.
_POLARITY_ALIAS = {"positive": "present", "negative": "absent"}

SYNTHESIS_OUTPUT_TEMPLATE = "data/12c_synthesis_targets_{smell}_{polarity}.json"
SYNTHESIS_REPORT_PATH = Path("data/step12c_report.json")


def _normalize_polarity(polarity: str) -> str:
    pol = _POLARITY_ALIAS.get(polarity, polarity)
    if pol not in POLARITIES:
        raise ValueError(f"polarity must be one of {POLARITIES} (got {polarity!r})")
    return pol


def _project_to_target(entry: dict, smell: str, polarity: str) -> dict:
    """Copy ``entry`` keeping ONLY the target (smell, polarity) smell-instance in
    its present_smells / absent_smells (mirrors how `balance` emits a row whose
    only surviving smell is this one). Keeps yaml_content and all other fields."""
    ne = dict(entry)
    ne["present_smells"] = [
        s for s in entry.get("present_smells", [])
        if polarity == "present" and s["pascal"] == smell
    ]
    ne["absent_smells"] = [
        s for s in entry.get("absent_smells", [])
        if polarity == "absent" and s["pascal"] == smell
    ]
    return ne


def select_targets(
    ground_truth: list[dict],
    priority: dict | None,
    smell: str,
    polarity: str,
    *,
    n: int = N,
) -> list[str]:
    """File_keys of the top ``min(n, available)`` instances of (smell, polarity),
    in priority order (rank 1 kept first). SAME sort/cap as `balance`. PURE."""
    polarity = _normalize_polarity(polarity)
    bucket = _bucketize(ground_truth, priority or {}).get((smell, polarity), [])
    return [fk for _, fk in _cap_bucket(bucket, n)]


def build_target_ground_truth(
    ground_truth: list[dict],
    priority: dict | None,
    smell: str,
    polarity: str,
    *,
    n: int = N,
) -> tuple[list[dict], dict]:
    """Return ``(records, report)``. ``records`` are 12b_final-schema entries for
    the selected (smell, polarity) instances, sorted by file_key (12b_final
    ordering). PURE: no IO, no hashing, no subset check."""
    polarity = _normalize_polarity(polarity)
    by_key = {e["file_key"]: e for e in ground_truth}
    ranked = select_targets(ground_truth, priority, smell, polarity, n=n)
    records = [_project_to_target(by_key[fk], smell, polarity) for fk in ranked]
    records.sort(key=lambda e: e["file_key"])

    available = len(_bucketize(ground_truth, priority or {}).get((smell, polarity), []))
    report = {
        "variant": "synthesis_targets",
        "target_smell": smell,
        "target_polarity": polarity,
        "n_per_polarity_cap": n,
        "available_in_bucket": available,
        "n_selected": len(records),
        "capped": available > n,
        "selected_file_keys_by_priority": ranked,
    }
    return records, report


# --- IO wrapper (Step 12c) --------------------------------------------------

def run_selection(
    smell: str,
    polarity: str,
    *,
    n: int = N,
    input_path: Path = INPUT_PATH,
    output_path: Path | None = None,
    report_path: Path = SYNTHESIS_REPORT_PATH,
    sample_path: Path = GROUND_TRUTH_SAMPLE_PATH,
    resample_path: Path = RESAMPLE_PATH,
    final_ground_truth_path: Path = OUTPUT_PATH,
    verify_integrity: bool = True,
    verify_subset: bool = True,
) -> tuple[Path, dict]:
    """Select synthesis targets for (smell, polarity) and write the 12b-schema
    JSON + report with input/output hashes.

    ``verify_integrity`` checks the input against step12_report.json (input drift
    guard, same as 12b). ``verify_subset`` asserts that the (smell, polarity)
    instances already in ``final_ground_truth_path`` (the 12b ground
    truth) are a SUBSET of this selection -- i.e. this step never drops an
    instance the balanced ground truth already committed. Both raise on failure.
    """
    from dataset_utils.hashing import compute_file_hash

    polarity = _normalize_polarity(polarity)
    input_path = Path(input_path)
    report_path = Path(report_path)
    if output_path is None:
        output_path = Path(SYNTHESIS_OUTPUT_TEMPLATE.format(smell=smell, polarity=polarity))
    output_path = Path(output_path)

    if verify_integrity:
        from dataset_utils.integrity import assert_input_matches_report

        assert_input_matches_report(
            input_path, Path("data/step12_report.json"), "step 12c: ground_truth_unbalanced"
        )

    with input_path.open(encoding="utf-8") as f:
        ground_truth: list[dict] = json.load(f)
    pri = priority_lookup(sample_path, resample_path)

    records, body = build_target_ground_truth(ground_truth, pri, smell, polarity, n=n)
    selected = set(body["selected_file_keys_by_priority"])

    # Subset integrity: the 12b ground truth's instances of this
    # (smell, polarity) must all be present in the selection.
    subset = {"checked": False, "ok": None, "final_gt_target_keys": [], "missing_from_selection": []}
    final_ground_truth_path = Path(final_ground_truth_path)
    if verify_subset:
        from dataset_utils.integrity import PipelineIntegrityError

        if not final_ground_truth_path.exists():
            raise PipelineIntegrityError(
                f"step 12c: 12b ground truth missing: {final_ground_truth_path}. "
                f"Run step 12b first (subset check needs it; pass verify_subset=False to skip)."
            )
        with final_ground_truth_path.open(encoding="utf-8") as f:
            final_gt = json.load(f)
        field = f"{polarity}_smells"
        final_keys = sorted(
            e["file_key"] for e in final_gt
            if any(s["pascal"] == smell for s in e.get(field, []))
        )
        missing = sorted(set(final_keys) - selected)
        subset = {
            "checked": True,
            "ok": not missing,
            "final_gt_target_keys": final_keys,
            "missing_from_selection": missing,
        }
        if missing:
            raise PipelineIntegrityError(
                f"step 12c: subset check FAILED for ({smell}, {polarity}). "
                f"{len(missing)} instance(s) in {final_ground_truth_path} are NOT in the "
                f"selection: {missing}. The selection logic has drifted from step 12b."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    report = {
        **body,
        "subset_of_final_ground_truth": subset,
        "input_sha256": {
            "unbalanced_ground_truth": compute_file_hash(input_path),
            "ground_truth_sample": compute_file_hash(sample_path),
            "resample": compute_file_hash(resample_path),
        },
        "output_file": str(output_path),
        "output_sha256": compute_file_hash(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path, report
