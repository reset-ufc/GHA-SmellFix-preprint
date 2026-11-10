"""Step 13 - build the FINAL human-validated ground truth by folding validated
synthetic positives + the 12c negatives into the balanced ground truth (12b).

Why this exists
---------------
A low-positive smell (``UnsafeDeployOnFork``: only 5 real positives) cannot be
balanced. We synthesize positives by injecting the smell into confirmed-clean
files (the 12c negatives). That injection is the ONE non-deterministic step
(an LLM call, ``smell-synthesizer``); its output ``12d_synth_..._present.json`` is
**frozen** in ``data/`` as a versioned input. Everything here is deterministic:
given (12b_final, the frozen 12d, the validation CSV, and the 12c negatives) this
module rebuilds the final ground truth byte-for-byte, with NO dependency on the
sibling ``smell-synthesizer`` repo.

This is the pure/IO port of ``smell-synthesizer``'s ``records.py`` +
``merge_cli.py`` merge core (``normalize_record``, ``split_by_loc``,
``fold_negatives``, ``merge_into_ground_truth``, ``smell_summary``), so it can be
unit-tested and driven from notebook ``13_final_ground_truth.ipynb``.

Approval gate (the one intentional change vs the source)
--------------------------------------------------------
``smell-synthesizer`` gated the merge on the viewer's ``annotations.json``
(agree/disagree). Here approval comes from the manual-evaluation CSV
(``12e_sinth_manual_evaluation.csv``, SAME schema as ``11_manual_evaluation.csv``),
so ONE CSV feeds both Step 12e's kappa and this merge. A synthetic positive is
approved iff its row is concordant-present (``tool_output == 'positive'`` and
``final_decision == 'present'``); a ``final_decision == 'absent'`` row is a
rejected injection (a flip) and is dropped - never reused as a negative, matching
the polarity-bucket rule the rest of the pipeline enforces. The two trails were
verified to agree (45 approved / 4 rejected either way), so this reproduces the
same final ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

# --- constants (mirror smell-synthesizer records.py) -------------------------
SMELL_PASCAL = "UnsafeDeployOnFork"
SYNTH_KEY_MARKER = "##synth"
RECORD_FIELDS = [
    "file_key",
    "repository",
    "file_path",
    "yaml_content",
    "file_line_count",
    "workflow_run_count",
    "repo_language",
    "default_branch",
    "present_smells",
    "absent_smells",
]
# Tukey IQR upper fence on workflow LOC from Step 09b (data/9b_loc_cap.json:
# ceil(Q3 + 1.5*IQR) = 209). A synthetic positive longer than this is out of
# distribution (a synthetic tell) and is excluded from the merge. If Step 09b is
# re-run on a new snapshot, re-sync this constant (or pass max_loc).
MAX_LOC = 209


# --- pure merge core (ported verbatim from records.py) ----------------------

def recompute_line_count(yaml_text: str) -> int:
    """Line count using the ground truth's convention (len(splitlines()))."""
    return len(yaml_text.splitlines())


def record_loc(record: dict) -> int:
    """A record's line count: its file_line_count, or recomputed from yaml_content
    if that field is missing/non-integer (so the LOC pre-check can't be dodged)."""
    loc = record.get("file_line_count")
    if isinstance(loc, int):
        return loc
    return recompute_line_count(record.get("yaml_content", ""))


def is_synth_key(file_key: str) -> bool:
    """True if a file_key carries the synthetic-positive marker (an injected row)."""
    return file_key.endswith(SYNTH_KEY_MARKER)


def yaml_sha256(yaml_text: str) -> str:
    """Content fingerprint (sha256 of yaml_content); matches reconcile.py."""
    return hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()


def normalize_record(record: dict, *, require_present: bool = True) -> dict:
    """Rebuild a record into 12b's exact byte-shape so it can enter the ground truth.

    Drops any unrecognized field (e.g. the review-only ``injection_confidence``),
    emits RECORD_FIELDS in a fixed order, and normalizes the smell dicts to 12b's
    key order (present_smells as {pascal, count}, absent_smells as {pascal}).
    Raises KeyError if any required field is missing, and (by default) ValueError
    if present_smells is empty - a positive with no present smell would otherwise
    normalize into an effective negative. Pass ``require_present=False`` for a
    genuine negative (a 12c target whose present_smells is empty by design).
    """
    if require_present and not record["present_smells"]:
        raise ValueError(
            f"normalize_record: empty present_smells on {record.get('file_key', '?')!r} "
            "(a positive must carry at least one present smell)"
        )
    out: dict = {}
    for field in RECORD_FIELDS:
        if field == "present_smells":
            out[field] = [
                {"pascal": s["pascal"], "count": s.get("count", 1)}
                for s in record["present_smells"]
            ]
        elif field == "absent_smells":
            out[field] = [{"pascal": s["pascal"]} for s in record["absent_smells"]]
        else:
            out[field] = record[field]
    return out


def split_by_loc(records_in: list[dict], max_loc: int = MAX_LOC) -> tuple[list[dict], list[dict]]:
    """Partition records into (within_loc, over_loc). over_loc is the merge blocker:
    an approved instance longer than ``max_loc`` must not enter the ground truth."""
    within: list[dict] = []
    over: list[dict] = []
    for r in records_in:
        (over if record_loc(r) > max_loc else within).append(r)
    return within, over


def fold_negatives(ground_truth: list[dict], targets: list[dict]) -> tuple[list[dict], dict]:
    """Fold the absent-smell synthesis targets (12c) into a COPY of the ground truth.

    Each 12c target is a confirmed NEGATIVE (present_smells empty, absent_smells
    listing the smell it was manually judged to lack). For each target:
    - key already in the ground truth: union the smell(s) into that record's
      absent_smells, preserving every existing label (enrich, never clobber);
    - key not present: append the normalized target as a new negative record.

    These absent labels are real two-annotator decisions (11_manual_evaluation.csv)
    that 12b's class-balancing capped out; folding restores the full bucket. The
    input is never mutated. Returns (merged, counts) with enriched/added/already.
    """
    index = {r.get("file_key"): i for i, r in enumerate(ground_truth)}
    out = [dict(r) for r in ground_truth]
    counts = {"enriched": 0, "added": 0, "already": 0}
    for target in targets:
        smells = [s["pascal"] for s in target.get("absent_smells", [])]
        idx = index.get(target.get("file_key"))
        if idx is None:
            out.append(normalize_record(target, require_present=False))
            counts["added"] += 1
            continue
        record = out[idx]
        existing = {s["pascal"] for s in record.get("absent_smells", [])}
        fresh = [p for p in smells if p not in existing]
        if fresh:
            record["absent_smells"] = [
                *record.get("absent_smells", []),
                *({"pascal": p} for p in fresh),
            ]
            counts["enriched"] += 1
        else:
            counts["already"] += 1
    return out, counts


def merge_into_ground_truth(ground_truth: list[dict], additions: list[dict]) -> list[dict]:
    """Append approved synthetic positives after the ground-truth records (new list).
    Keys never collide - synthetic keys carry SYNTH_KEY_MARKER, GT keys do not."""
    return [*ground_truth, *additions]


def smell_summary(records_in: list[dict]) -> list[dict]:
    """Per-smell present/absent tallies, split real vs synth (rows carrying
    SYNTH_KEY_MARKER). One row per smell, sorted by name."""
    rows: dict[str, dict] = {}

    def bump(pascal: str, field: str, synth: bool) -> None:
        row = rows.setdefault(
            pascal,
            {"smell": pascal, "present": 0, "present_synth": 0, "absent": 0, "absent_synth": 0},
        )
        row[field] += 1
        if synth:
            row[field + "_synth"] += 1

    for record in records_in:
        synth = is_synth_key(record.get("file_key", ""))
        for s in record.get("present_smells", []):
            bump(s["pascal"], "present", synth)
        for s in record.get("absent_smells", []):
            bump(s["pascal"], "absent", synth)
    return [rows[k] for k in sorted(rows)]


# --- approval gate: the manual-evaluation CSV (replaces annotations.json) ----

def read_eval_csv(csv_path: Path, smell: str | None = None) -> list[dict]:
    """Rows of the synthetic-evaluation CSV, optionally filtered to one smell."""
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if smell is not None:
        rows = [r for r in rows if r.get("target_smell") == smell]
    return rows


def approval_from_csv(csv_path: Path, smell: str | None = None) -> tuple[set[str], set[str], set[str]]:
    """Partition the CSV's file_keys into (approved, rejected, reviewed).

    approved = concordant present (tool_output == 'positive' and
    final_decision == 'present'); rejected = final_decision == 'absent' (a flip,
    dropped - never reused as a negative); reviewed = every file_key in the CSV
    (the coverage set). A row that is neither present nor absent is ignored for
    approval but still counts as reviewed.
    """
    approved: set[str] = set()
    rejected: set[str] = set()
    reviewed: set[str] = set()
    for r in read_eval_csv(csv_path, smell):
        fk = r["file_key"]
        reviewed.add(fk)
        decision = r.get("final_decision")
        if decision == "present":
            approved.add(fk)
        elif decision == "absent":
            rejected.add(fk)
    return approved, rejected, reviewed


def approved_synth_records(synth: list[dict], approved: set[str]) -> list[dict]:
    """Synthetic positives whose file_key is approved, in the input (12d) order -
    matching the order smell-synthesizer's review_buckets produced."""
    return [r for r in synth if r["file_key"] in approved]


# --- IO wrapper (Step 13) ----------------------------------------------------

SYNTH_PATH = Path("data/12d_synth_UnsafeDeployOnFork_present.json")
GROUND_TRUTH_PATH = Path("data/12b_final_ground_truth.json")
NEGATIVES_PATH = Path("data/12c_synthesis_targets_UnsafeDeployOnFork_absent.json")
EVAL_CSV_PATH = Path("data/12e_sinth_manual_evaluation.csv")
OUTPUT_PATH = Path("data/13_final_ground_truth.json")
REPORT_PATH = Path("data/step13_report.json")


def load_json(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _write_json_with_sha(records_out: list[dict], out_path: Path) -> str:
    """Write the ground truth exactly as smell-synthesizer's engine.write_output:
    json.dumps(indent=2, ensure_ascii=False) + trailing newline, utf-8, plus a
    sibling .sha256.txt (``<sha>  <basename>\\n``). Returns the sha256 hex."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records_out, indent=2, ensure_ascii=False) + "\n"
    data = payload.encode("utf-8")
    out_path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    (out_path.parent / (out_path.name + ".sha256.txt")).write_text(
        f"{sha}  {out_path.name}\n", encoding="utf-8"
    )
    return sha


def run(
    synth_path: Path = SYNTH_PATH,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    negatives_path: Path = NEGATIVES_PATH,
    eval_csv_path: Path = EVAL_CSV_PATH,
    output_path: Path = OUTPUT_PATH,
    report_path: Path = REPORT_PATH,
    *,
    smell: str = SMELL_PASCAL,
    max_loc: int = MAX_LOC,
    fold_negatives_in: bool = True,
    strict_loc: bool = False,
    require_all_reviewed: bool = True,
) -> tuple[Path, dict]:
    """Build data/13_final_ground_truth.json from the frozen synthesis inputs.

    Order of operations mirrors smell-synthesizer's smell-merge exactly:
    approve (CSV) -> LOC pre-check -> normalize -> fold 12c negatives ->
    append positives -> write. ``require_all_reviewed`` aborts if any synthetic
    positive has no CSV row (coverage gate); ``strict_loc`` aborts on any
    over-LOC approved instance instead of excluding it.
    """
    from dataset_utils.hashing import compute_file_hash

    synth = load_json(synth_path)
    ground_truth = load_json(ground_truth_path)
    approved, rejected, reviewed = approval_from_csv(eval_csv_path, smell)

    unreviewed = [r["file_key"] for r in synth if r["file_key"] not in reviewed]
    if unreviewed and require_all_reviewed:
        raise ValueError(
            f"Step 13: {len(unreviewed)} synthetic positive(s) have no row in "
            f"{eval_csv_path} - finish manual validation first (or pass "
            f"require_all_reviewed=False): {sorted(unreviewed)}"
        )

    agree = approved_synth_records(synth, approved)
    within, over = split_by_loc(agree, max_loc)
    if over and strict_loc:
        raise ValueError(
            f"Step 13: {len(over)} approved instance(s) exceed {max_loc} LOC "
            f"(Tukey IQR cap): {[r['file_key'] for r in over]}"
        )

    normalized = [normalize_record(r) for r in within]

    base = ground_truth
    fold = {"enriched": 0, "added": 0, "already": 0}
    if fold_negatives_in:
        negatives = load_json(negatives_path)
        base, fold = fold_negatives(ground_truth, negatives)

    merged = merge_into_ground_truth(base, normalized)
    output_sha = _write_json_with_sha(merged, output_path)

    report = {
        "variant": "final_ground_truth",
        "target_smell": smell,
        "max_loc": max_loc,
        "fold_negatives": fold_negatives_in,
        "n_synth_candidates": len(synth),
        "n_approved": len(approved),
        "n_rejected": len(rejected),
        "rejected_keys": sorted(rejected),
        "n_over_loc": len(over),
        "over_loc_keys": [r["file_key"] for r in over],
        "n_positives_merged": len(normalized),
        "fold_counts": fold,
        "n_entries_in": len(ground_truth),
        "n_entries_out": len(merged),
        "smell_summary": smell_summary(merged),
        "input_sha256": {
            "synth_12d": compute_file_hash(synth_path),
            "ground_truth_12b": compute_file_hash(ground_truth_path),
            "negatives_12c": compute_file_hash(negatives_path),
            "eval_csv_12e_sinth": compute_file_hash(eval_csv_path),
        },
        "output_file": str(output_path),
        "output_sha256": output_sha,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(report_path).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return Path(output_path), report
