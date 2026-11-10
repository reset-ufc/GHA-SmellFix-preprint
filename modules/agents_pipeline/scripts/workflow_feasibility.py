#!/usr/bin/env python3
"""Inputs for the workflow-level feasibility diagnostic (8 workflows x 19 smells).

Implements the protocol in
``modules/agents_pipeline/docs/workflow_level_feasibility_detection_refactoring_protocol.md``.
Four commands, run from the repo root:

``select``
    Picks 8 real workflows from the Step 9 candidate pool that were never run in the
    main study (except, disclosed per entry, confirmed cases of a smell no such workflow
    likely has), fit the Step 9b LOC cap, pass the main study's strict filter (no smell
    with more than one instance), have run history, and pass the GitHub Actions
    schema, so that all 19 smells are likely present somewhere in the set. Each
    (workflow, smell) gets a chance of being present: 1 or 0 when the study's
    annotators already labeled it (the only way to reach UnsafeDeployOnFork, whose
    GHA flag they confirmed in 3% of cases), the flag's confirmed share when GHA
    flags it, else 0. Slot by slot, the candidate that most lowers the expected
    number of smells with no present instance wins, within the protocol's quotas
    (2 / 3-4 / 5+ smells -> 2 / 3 / 3 workflows, counting flags confirmed in at
    least half the cases plus labeled-present smells); seed 2027 breaks ties.
    None of this is ground truth, and none of it is shown to the annotators.
    Writes ``selected_workflows.json`` and ``annotation_sheet.xlsx``: one identical
    tab per author, 152 rows each, with the mined YAML, the smell definition the
    pipeline receives, and the study's expected constructs next to the fields to fill.

``consensus``
    Merges the two tabs, each exported from the shared sheet as CSV
    (``annotator_a.csv``, ``annotator_b.csv``), into ``annotations_consensus.csv``.
    Agreed rows are kept; every disagreement gets a blank ``manual_label`` (both
    verdicts in ``annotator_a`` / ``annotator_b``), so ``build-gt`` refuses the sheet
    until the authors resolve it.

``build-gt``
    Converts the consensus annotation CSV into a pipeline ground truth where every
    workflow lists all 19 smells, so the dispatcher runs 19 target-conditioned
    pairs per workflow. ``uncertain`` smells still run (listed under
    ``absent_smells``) and are also recorded in ``uncertain_smells`` so the
    analysis layer can exclude them from the metrics. A workflow the annotators
    find fewer than two present smells in is rejected: it no longer meets the
    protocol's inclusion criteria and must be replaced. A smell with no present
    instance in the whole set is only reported: it still runs on every workflow,
    as absent pairs only.

``freeze``
    Records the SHA-256, size and completion time (the file's mtime, in UTC) of the files no
    command can regenerate, in ``manifest.json``: the two exports, the resolved
    consensus, the two patch reviews, their adjudication, the findings of the final
    check, and the resolved review sheet. Run it each time one of them is completed or
    changed. An entry only changes when the file's bytes do.

Usage
-----
    uv run python modules/agents_pipeline/scripts/workflow_feasibility.py select
    uv run python modules/agents_pipeline/scripts/workflow_feasibility.py consensus
    uv run python modules/agents_pipeline/scripts/workflow_feasibility.py build-gt
    uv run python modules/agents_pipeline/scripts/workflow_feasibility.py freeze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from gha_agents.prompts import SMELL_CATALOG
from gha_agents.tools import validate_github_actions_syntax
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
OUT_DIR = DATA / "workflow_feasibility"
SHEET = OUT_DIR / "annotation_sheet.xlsx"
# The filled tabs, one CSV each (Google Sheets' File > Download > CSV exports the open tab).
ANNOTATOR_CSVS = (OUT_DIR / "annotator_a.csv", OUT_DIR / "annotator_b.csv")
CONSENSUS = OUT_DIR / "annotations_consensus.csv"
# The reviewers' patch sheets, their adjudication, the findings of a final check, and the
# resolved sheet that utils.workflow_feasibility reads (only its review columns are by hand).
PATCH_REVIEWS = (
    *(OUT_DIR / f"patch_review_{part}.csv" for part in ("a", "b", "adjudication", "findings")),
    DATA / "artifacts" / "workflow_feasibility" / "patch_review.csv",
)
# Hand-made, so no command can regenerate them: recorded in MANIFEST, keyed by file name.
FROZEN = (*ANNOTATOR_CSVS, CONSENSUS, *PATCH_REVIEWS)
MANIFEST = OUT_DIR / "manifest.json"
CANDIDATES = DATA / "9_ground_truth_candidates.csv"
ENRICHED = DATA / "8_dataset_enriched.csv"
LOC_CAP = DATA / "9b_loc_cap.json"
YAML_TARBALL = DATA / "downloaded_workflows.tar.xz"
CONSTRUCT_MAP = DATA / "smell_expected_construct_mapping_refined.csv"
# Every manually labeled GHA candidate: the agreeing rows plus the ones that overturn GHA.
MANUAL_LABELS = [DATA / "11_manual_evaluation.csv", DATA / "manual_evaluation_gha_overturned.csv"]
MIN_FLAG_PRECISION = 0.5
DEFAULT_SEED = 2027

SMELLS = [(s["pascal"], s["name"]) for s in SMELL_CATALOG]
SMELL_PASCALS = {p for p, _ in SMELLS}
DEFINITIONS = {s["pascal"]: s["description"] for s in SMELL_CATALOG}
LABELS = ("present", "absent", "uncertain")
MIN_PRESENT_SMELLS = 2
# The protocol's annotation fields, plus smell_pascal as the join key.
ANNOTATION_FIELDS = [
    "workflow_id",
    "repository",
    "workflow_path",
    "smell_pascal",
    "smell_name",
    "manual_label",
    "number_of_instances",
    "smell_evidence",
    "expected_refactoring_scope",
    "notes",
]
# (label, min counted smells, max counted smells, workflows to draw)
BUCKETS = [("2", 2, 2, 2), ("3-4", 3, 4, 3), ("5+", 5, len(SMELLS), 3)]
# Sampling pools, not evaluation sets: their files were drawn or annotated but only the
# ones that reached the other ground truths were ever run.
SAMPLING_POOLS = {
    "10_ground_truth_sample.json",
    "10b_ground_truth_resample.json",
    "12_ground_truth_unbalanced.json",
}

TABS = ("Annotator A", "Annotator B")
# (header, column width, header comment); the fill columns sit between the context the
# annotator reads first (smell + definition) and the YAML block.
SHEET_COLUMNS = [
    ("workflow_id", 34, None),
    ("smell_name", 26, None),
    ("smell_definition", 70, "The exact target smell definition the pipeline receives."),
    (
        "manual_label",
        14,
        "present = at least one instance; absent = none; uncertain = the "
        "static YAML is not enough for a reliable decision.",
    ),
    ("number_of_instances", 12, "Whole number >= 1 when present; blank when absent."),
    (
        "smell_evidence",
        40,
        "Line numbers of the YAML block and what makes the smell present "
        "(or why it is absent/uncertain).",
    ),
    (
        "expected_refactoring_scope",
        40,
        "Required when present: the construct a fix must change "
        "(e.g. jobs.build.timeout-minutes). See expected_constructs.",
    ),
    ("notes", 30, None),
    (
        "workflow_yaml",
        90,
        "The mined version the pipeline receives; annotate this, not the "
        "live file behind github_url.",
    ),
    ("expected_constructs", 45, "Primary constructs the study maps to this smell (reference)."),
    ("repository", 30, None),
    ("workflow_path", 36, None),
    ("github_url", 50, "Live file; it may have changed since mining."),
    ("smell_pascal", 26, None),
]
FILL_FIELDS = {
    "manual_label",
    "number_of_instances",
    "smell_evidence",
    "expected_refactoring_scope",
    "notes",
}


def _file_key(repository: str, file_path: str) -> str:
    return repository.replace("/", "@") + "@" + Path(file_path).name


def _evaluated_file_keys() -> set[str]:
    """Every workflow in a ground truth the pipeline ran on, synthetic variants included."""
    keys: set[str] = set()
    for path in DATA.glob("*.json"):
        if path.name in SAMPLING_POOLS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            keys |= {
                e["file_key"].split("##")[0]
                for e in payload
                if isinstance(e, dict) and "file_key" in e
            }
    return keys


def _manual_rows() -> list[dict]:
    rows = []
    for path in MANUAL_LABELS:
        with path.open(encoding="utf-8", newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _flag_precision(rows: list[dict]) -> dict[str, float]:
    """Share of GHA-flagged annotated pairs that the annotators labeled present, per smell."""
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["tool_output"] == "positive":
            hits[row["target_smell"]][0] += row["final_decision"] == "present"
            hits[row["target_smell"]][1] += 1
    return {smell: present / flagged for smell, (present, flagged) in hits.items()}


def _analyzer_smells(smells_json: str) -> dict[str, int]:
    return {
        s["pascal"]: int(s.get("instance_count") or 0)
        for s in json.loads(smells_json or "[]")
        if s.get("pascal") in SMELL_PASCALS
    }


def _eligible_candidates(
    loc_cap: int, precision: dict[str, float], labels: dict[str, dict]
) -> list[dict]:
    """Pool rows with their bucket and per-smell chance of a present instance."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with CANDIDATES.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if not 0 < int(row["file_line_count"]) <= loc_cap:
                continue
            if int(float(row["workflow_run_count"] or 0)) <= 0:
                continue
            smells = _analyzer_smells(row["smells_json"])
            if max(smells.values(), default=0) > 1:  # main study's strict filter (Step 10)
                continue
            known = labels.get(row["file_key"], {})
            chance = {s: precision.get(s, 0.0) for s in smells} | {
                s: float(label == "present") for s, label in known.items()
            }
            chance = {s: c for s, c in chance.items() if c > 0 and s in SMELL_PASCALS}
            counted = sorted(s for s, c in chance.items() if c >= MIN_FLAG_PRECISION)
            bucket = next((b for b, lo, hi, _ in BUCKETS if lo <= len(counted) <= hi), None)
            if bucket is None:
                continue
            rows.append(
                row
                | {
                    "analyzer_smells": smells,
                    "counted_smells": counted,
                    "chance": chance,
                    "bucket": bucket,
                }
            )
    return sorted(rows, key=lambda r: r["file_key"])


def pick_covering(candidates: list[dict], rng: random.Random, is_valid) -> list[dict]:
    """Greedy cover of the 19 smells within the bucket quotas, rarest smell first.

    Each slot targets the smell most likely to end up with no present instance that an
    open-bucket candidate can still likely reach (the scarcest one on ties), and takes the valid candidate most likely to
    have it, then the one that most lowers the expected number of uncovered smells; the
    seeded shuffle decides the remaining ties. A swap pass then replaces any pick with a
    same-bucket candidate that raises the chance that all 19 smells are covered at once.
    ponytail: greedy plus swaps, not an exact optimum; enough for 8 slots over a 12k pool.
    """
    pool = list(candidates)
    rng.shuffle(pool)
    missing = {pascal: 1.0 for pascal, _ in SMELLS}
    left = {label: quota for label, _, _, quota in BUCKETS}
    picked: list[dict] = []

    def gain(row: dict) -> float:
        return sum(missing[s] * c for s, c in row["chance"].items())

    while any(left.values()):
        open_rows = [r for r in pool if left[r["bucket"]]]
        if not open_rows:
            msg = f"no valid candidates left for buckets {[b for b, n in left.items() if n]}"
            raise ValueError(msg)
        # Scarcity breaks ties between equally uncovered smells: fewest likely candidates first.
        likely = dict.fromkeys(missing, 0)
        for r in open_rows:
            for p, c in r["chance"].items():
                likely[p] += c >= MIN_FLAG_PRECISION
        reachable = [p for p in missing if likely[p]]
        target = max(reachable, key=lambda p: (missing[p], -likely[p]), default=None)
        best = max(open_rows, key=lambda r: (r["chance"].get(target, 0.0), gain(r)))
        pool.remove(best)
        if not is_valid(best):
            continue
        picked.append(best)
        left[best["bucket"]] -= 1
        for s, c in best["chance"].items():
            missing[s] *= 1 - c

    def all_covered(rows: list[dict]) -> float:
        return math.prod(1 - math.prod(1 - r["chance"].get(p, 0.0) for r in rows) for p in missing)

    improved = True
    while improved:
        improved = False
        for i, current in enumerate(picked):
            others = picked[:i] + picked[i + 1 :]
            rest = {p: math.prod(1 - r["chance"].get(p, 0.0) for r in others) for p in missing}

            def with_row(row: dict, rest: dict = rest) -> float:
                return math.prod(1 - rest[p] * (1 - row["chance"].get(p, 0.0)) for p in rest)

            while True:
                same = [r for r in pool if r["bucket"] == current["bucket"]]
                best = max(same, key=with_row, default=None)
                if best is None or with_row(best) <= all_covered(picked) + 1e-12:
                    break
                pool.remove(best)
                if is_valid(best):
                    pool.append(current)
                    picked[i] = best
                    improved = True
                    break
    return picked


def _read_yamls(file_keys: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with tarfile.open(YAML_TARBALL, "r:xz") as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in file_keys:
                found[name] = tar.extractfile(member).read().decode("utf-8", errors="replace")
    return found


def _default_branches() -> dict[str, str]:
    branches: dict[str, str] = {}
    with ENRICHED.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = _file_key(row["repository"], row["file_path"])
            branches.setdefault(key, row.get("repo_default_branch", ""))
    return branches


def _expected_constructs() -> dict[str, str]:
    constructs: dict[str, list[str]] = defaultdict(list)
    with CONSTRUCT_MAP.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["expectedness"] == "PRIMARY":
                item = f"{row['expected_modification_type']} {row['expected_construct']}"
                if item not in constructs[row["smell_pascal"]]:
                    constructs[row["smell_pascal"]].append(item)
    return {smell: "\n".join(items) for smell, items in constructs.items()}


def write_annotation_sheet(selected: list[dict], path: Path, constructs: dict[str, str]) -> None:
    """One identical tab per author; each workflow's numbered YAML spans its 19 rows."""
    headers = [h for h, _, _ in SHEET_COLUMNS]
    col = {h: i + 1 for i, h in enumerate(headers)}
    wrap = Alignment(wrap_text=True, vertical="top")
    to_fill = PatternFill("solid", fgColor="FFF2CC")
    book = Workbook()
    book.remove(book.active)
    for tab in TABS:
        ws = book.create_sheet(tab)
        ws.append(headers)
        for (header, width, note), cell in zip(SHEET_COLUMNS, ws[1], strict=True):
            ws.column_dimensions[cell.column_letter].width = width
            cell.font = Font(bold=True)
            if note:
                cell.comment = Comment(note, "workflow_feasibility.py")
            if header in FILL_FIELDS:
                cell.fill = to_fill
        labels = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True)
        counts = DataValidation(
            type="whole", operator="greaterThanOrEqual", formula1="0", allow_blank=True
        )
        ws.add_data_validation(labels)
        ws.add_data_validation(counts)
        for entry in selected:
            first = ws.max_row + 1
            yaml_lines = entry["yaml_content"].splitlines()
            branch = entry.get("default_branch") or "HEAD"
            for pascal, name in SMELLS:
                ws.append(
                    [
                        entry["file_key"],
                        name,
                        DEFINITIONS[pascal],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        constructs.get(pascal, ""),
                        entry["repository"],
                        entry["file_path"],
                        f"https://github.com/{entry['repository']}/blob/{branch}/{entry['file_path']}",
                        pascal,
                    ]
                )
                row = ws.max_row
                for c in ws[row]:
                    c.alignment = wrap
                    if c.column in {col[f] for f in FILL_FIELDS}:
                        c.fill = to_fill
                labels.add(ws.cell(row, col["manual_label"]))
                counts.add(ws.cell(row, col["number_of_instances"]))
                # Tall enough for the wrapped definition and for this row's share of the YAML.
                definition_lines = math.ceil(len(DEFINITIONS[pascal]) / 75) + 1
                ws.row_dimensions[row].height = 15 * max(
                    definition_lines, math.ceil(len(yaml_lines) / len(SMELLS))
                )
            yaml_cell = ws.cell(first, col["workflow_yaml"])
            yaml_cell.value = "\n".join(f"{i:>3}  {line}" for i, line in enumerate(yaml_lines, 1))
            yaml_cell.font = Font(name="Courier New", size=9)
            ws.merge_cells(
                start_row=first,
                end_row=ws.max_row,
                start_column=col["workflow_yaml"],
                end_column=col["workflow_yaml"],
            )
        ws.freeze_panes = "C2"
    book.save(path)


def read_annotation_csv(path: Path) -> list[dict]:
    """One tab exported as CSV, as ANNOTATION_FIELDS dicts of strings."""
    csv.field_size_limit(sys.maxsize)  # a workflow's whole YAML sits in one cell
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in ANNOTATION_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            msg = f"{path} lacks the sheet columns {', '.join(missing)}"
            raise ValueError(msg)
        return [
            {field: (row[field] or "").strip() for field in ANNOTATION_FIELDS}
            for row in reader
            if any(row.values())
        ]


def select(seed: int) -> None:
    if SHEET.exists():
        msg = f"{SHEET} exists and may hold annotations; move it first."
        raise SystemExit(msg)
    loc_cap = int(json.loads(LOC_CAP.read_text(encoding="utf-8"))["cap"])
    rows = _manual_rows()
    precision = _flag_precision(rows)
    labels: dict[str, dict] = defaultdict(dict)
    for row in rows:
        labels[row["file_key"]][row["target_smell"]] = row["final_decision"]
    evaluated = _evaluated_file_keys()
    eligible = _eligible_candidates(loc_cap, precision, labels)
    outside = [c for c in eligible if c["file_key"] not in evaluated]
    # A smell no outside workflow likely has (UnsafeDeployOnFork, MissingPermExternalActions)
    # can only be covered by a real workflow the study already confirmed; the protocol
    # merely prefers workflows outside the evaluation set, so this is disclosed per entry.
    unreachable = {
        p
        for p, _ in SMELLS
        if not any(c["chance"].get(p, 0) >= MIN_FLAG_PRECISION for c in outside)
    }
    inside = [
        c | {"in_main_evaluation": True}
        for c in eligible
        if c["file_key"] in evaluated and any(c["chance"].get(p) == 1.0 for p in unreachable)
    ]
    candidates = outside + inside
    yamls = _read_yamls({r["file_key"] for r in candidates})
    branches = _default_branches()

    def is_valid(row: dict) -> bool:
        return validate_github_actions_syntax(yamls.get(row["file_key"], "")) == "OK"

    rng = random.Random(seed)  # noqa: S311 -- reproducible sampling, not crypto
    try:
        picked = pick_covering(candidates, rng, is_valid)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected = [
        {
            "file_key": row["file_key"],
            "repository": row["repository"],
            "file_path": row["file_path"],
            "yaml_content": yamls[row["file_key"]],
            "file_line_count": int(row["file_line_count"]),
            "workflow_run_count": int(float(row["workflow_run_count"])),
            "repo_language": row["repo_language"],
            "default_branch": branches.get(row["file_key"], ""),
            "selection_bucket": row["bucket"],
            "in_main_evaluation": row.get("in_main_evaluation", False),
            "selection_chance": row["chance"],
            "analyzer_smells": row["analyzer_smells"],
            "present_smells": [],
            "absent_smells": [],
        }
        for row in sorted(picked, key=lambda r: (r["bucket"], r["file_key"]))
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selected_workflows.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    write_annotation_sheet(selected, SHEET, _expected_constructs())

    print(
        f"Eligible pool: {len(outside)} outside the evaluation set + {len(inside)} inside it, "
        f"confirmed for {sorted(unreachable)} (LOC cap {loc_cap}, seed {seed})"
    )
    for entry in selected:
        inside_note = "  <- in the main evaluation set" if entry["in_main_evaluation"] else ""
        print(
            f"  [{entry['selection_bucket']:>3}] {entry['file_key']:<60} "
            f"LOC={entry['file_line_count']:<4} smells={len(entry['selection_chance'])}"
            f"{inside_note}"
        )
    print("Chance that at least one selected workflow has each smell:")
    covered = 1.0
    for pascal, _ in SMELLS:
        miss = math.prod(1 - e["selection_chance"].get(pascal, 0.0) for e in selected)
        covered *= 1 - miss
        print(f"  {1 - miss:6.1%}  {pascal}")
    print(f"Chance that all 19 are covered: {covered:.1%} (build-gt enforces it after annotation)")
    print(f"-> {OUT_DIR / 'selected_workflows.json'}")
    print(f"-> {SHEET} (tabs: {', '.join(TABS)})")


def _verdict(row: dict) -> tuple[str, str]:
    label = (row.get("manual_label") or "").strip().lower()
    count = (row.get("number_of_instances") or "").strip()
    return label, count if label == "present" else ""


def _join(a: str | None, b: str | None) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    return a if b in ("", a) else b if not a else f"A: {a} | B: {b}"


def merge_annotations(sheet_a: list[dict], sheet_b: list[dict]) -> tuple[list[dict], int]:
    """Consensus draft from two annotators' rows and the number of disagreements.

    A disagreement is a different label, or a different instance count on a present
    smell; its ``manual_label`` is left blank for the authors to resolve.
    """
    by_key = {(r["workflow_id"], r["smell_pascal"]): r for r in sheet_b}
    keys_a = [(r["workflow_id"], r["smell_pascal"]) for r in sheet_a]
    if sorted(keys_a) != sorted(by_key):
        msg = "the two tabs do not cover the same (workflow_id, smell_pascal) rows"
        raise ValueError(msg)
    merged, disagreements = [], 0
    for row_a in sheet_a:
        row_b = by_key[(row_a["workflow_id"], row_a["smell_pascal"])]
        agree = _verdict(row_a) == _verdict(row_b)
        disagreements += not agree
        merged.append(
            {field: row_a.get(field, "") for field in ANNOTATION_FIELDS}
            | {
                "manual_label": _verdict(row_a)[0] if agree else "",
                "smell_evidence": _join(row_a.get("smell_evidence"), row_b.get("smell_evidence")),
                "expected_refactoring_scope": _join(
                    row_a.get("expected_refactoring_scope"), row_b.get("expected_refactoring_scope")
                ),
                "notes": _join(row_a.get("notes"), row_b.get("notes")),
                "annotator_a": "/".join(filter(None, _verdict(row_a))),
                "annotator_b": "/".join(filter(None, _verdict(row_b))),
            }
        )
    return merged, disagreements


def consensus(exports: tuple[Path, Path], out: Path) -> None:
    if out.exists():
        msg = f"{out} exists and may hold resolved disagreements; move it first."
        raise SystemExit(msg)
    try:
        merged, disagreements = merge_annotations(*(read_annotation_csv(p) for p in exports))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*ANNOTATION_FIELDS, "annotator_a", "annotator_b"])
        writer.writeheader()
        writer.writerows(merged)
    print(f"-> {out}: {len(merged)} rows, {disagreements} disagreements to resolve by consensus")


def build_ground_truth(selected: list[dict], annotations: list[dict]) -> list[dict]:
    """Merge consensus annotations into the selected entries (one run per smell).

    Raises ``ValueError`` listing every problem, so a half-filled sheet never
    produces a ground truth.
    """
    by_workflow = {e["file_key"]: e for e in selected}
    labels: dict[str, dict[str, dict]] = defaultdict(dict)
    problems: list[str] = []
    for i, row in enumerate(annotations, start=2):
        wf, pascal = row.get("workflow_id", ""), row.get("smell_pascal", "")
        status = (row.get("manual_label") or "").strip().lower()
        count = (row.get("number_of_instances") or "").strip()
        if wf not in by_workflow:
            problems.append(f"line {i}: unknown workflow_id {wf!r}")
        elif pascal not in SMELL_PASCALS:
            problems.append(f"line {i}: unknown smell_pascal {pascal!r}")
        elif pascal in labels[wf]:
            problems.append(f"line {i}: duplicate ({wf}, {pascal})")
        elif status not in LABELS:
            problems.append(f"line {i}: manual_label {status!r} not in {LABELS}")
        elif count and not count.isdigit():
            problems.append(f"line {i}: number_of_instances {count!r} is not a whole number")
        elif status == "present" and int(count or 0) < 1:
            problems.append(f"line {i}: present needs number_of_instances >= 1")
        elif status == "present" and not (row.get("expected_refactoring_scope") or "").strip():
            problems.append(f"line {i}: present needs expected_refactoring_scope")
        elif status == "absent" and int(count or 0) > 0:
            problems.append(f"line {i}: absent cannot have number_of_instances {count}")
        else:
            labels[wf][pascal] = {"status": status, "count": int(count or 0)}
    for wf in by_workflow:
        missing = [p for p, _ in SMELLS if p not in labels[wf]]
        if missing:
            problems.append(f"{wf}: missing {len(missing)} smells ({', '.join(missing)})")
            continue
        n_present = sum(v["status"] == "present" for v in labels[wf].values())
        if n_present < MIN_PRESENT_SMELLS:
            problems.append(
                f"{wf}: only {n_present} present smell(s); replace it with a workflow that has "
                f">= {MIN_PRESENT_SMELLS}"
            )
    if problems:
        raise ValueError("\n".join(problems))

    ground_truth = []
    for wf, entry in by_workflow.items():
        wf_labels = labels[wf]
        ground_truth.append(
            entry
            | {
                "present_smells": [
                    {"pascal": p, "count": wf_labels[p]["count"]}
                    for p, _ in SMELLS
                    if wf_labels[p]["status"] == "present"
                ],
                "absent_smells": [
                    {"pascal": p} for p, _ in SMELLS if wf_labels[p]["status"] != "present"
                ],
                "uncertain_smells": [p for p, _ in SMELLS if wf_labels[p]["status"] == "uncertain"],
            }
        )
    return ground_truth


def agreement(rows: list[dict]) -> tuple[int, int, float]:
    """Pairs, pairs with the same label, and Cohen's kappa of the two annotators.

    Reads the verdict columns ``consensus`` writes (``annotator_a`` / ``annotator_b``,
    e.g. ``present/2``), so it measures agreement before the disagreements were resolved.
    """
    labels = [(r["annotator_a"].split("/")[0], r["annotator_b"].split("/")[0]) for r in rows]
    n = len(labels)
    agree = sum(a == b for a, b in labels)
    count_a, count_b = Counter(a for a, _ in labels), Counter(b for _, b in labels)
    expected = sum(count_a[k] * count_b[k] for k in count_a) / n**2
    kappa = float("nan") if expected == 1 else (agree / n - expected) / (1 - expected)
    return n, agree, kappa


def build_gt(annotations_path: Path) -> None:
    selected = json.loads((OUT_DIR / "selected_workflows.json").read_text(encoding="utf-8"))
    with annotations_path.open(encoding="utf-8-sig", newline="") as fh:
        annotations = list(csv.DictReader(fh))
    if annotations and "annotator_a" in annotations[0]:
        n, agree, kappa = agreement(annotations)
        print(
            f"   before consensus: same label on {agree} of {n} pairs ({agree / n:.1%}), "
            f"Cohen's kappa = {kappa:.3f}"
        )
    try:
        ground_truth = build_ground_truth(selected, annotations)
    except ValueError as exc:
        msg = f"Annotation sheet is not complete:\n{exc}"
        raise SystemExit(msg) from exc
    uncovered = sorted(
        SMELL_PASCALS - {s["pascal"] for e in ground_truth for s in e["present_smells"]}
    )
    out = OUT_DIR / "ground_truth.json"
    out.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
    n_present = sum(len(e["present_smells"]) for e in ground_truth)
    n_uncertain = sum(len(e["uncertain_smells"]) for e in ground_truth)
    n_pairs = sum(len(e["present_smells"]) + len(e["absent_smells"]) for e in ground_truth)
    print(f"-> {out}: {len(ground_truth)} workflows, {n_pairs} pairs per design")
    print(
        f"   present={n_present} uncertain={n_uncertain} absent={n_pairs - n_present - n_uncertain}"
    )
    if uncovered:
        print(
            f"   no present instance of {len(uncovered)} smells ({', '.join(uncovered)}): "
            "they still run on every workflow, as absent pairs only"
        )


def freeze(paths: tuple[Path, ...] = FROZEN, manifest: Path = MANIFEST) -> None:
    """Record each existing file's SHA-256, size and mtime (in UTC) in ``manifest``.

    An entry is rewritten only when the file's bytes change, so a fresh checkout (new
    mtimes) keeps the recorded times.
    """
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    for path in paths:
        if not path.exists():
            print(f"   {path.name}: missing, not recorded")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if entries.get(path.name, {}).get("sha256") == digest:
            print(f"   {path.name}: unchanged since {entries[path.name]['completed_at']}")
            continue
        completed = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        entries[path.name] = {
            "sha256": digest,
            "bytes": len(data),
            "completed_at": completed.isoformat(timespec="seconds"),
        }
        print(f"   {path.name}: recorded {digest[:12]}, completed {entries[path.name]['completed_at']}")
    manifest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"-> {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select", help="Sample the 8 workflows and write the annotation sheet.")
    sel.add_argument("--seed", type=int, default=DEFAULT_SEED)
    con = sub.add_parser("consensus", help="Merge the two annotators' CSV exports for consensus.")
    con.add_argument("--annotator-a", type=Path, default=ANNOTATOR_CSVS[0])
    con.add_argument("--annotator-b", type=Path, default=ANNOTATOR_CSVS[1])
    con.add_argument("--out", type=Path, default=CONSENSUS)
    gt = sub.add_parser("build-gt", help="Turn the consensus annotation CSV into a ground truth.")
    gt.add_argument(
        "--annotations",
        type=Path,
        default=CONSENSUS,
        help="Consensus CSV with every disagreement resolved.",
    )
    sub.add_parser("freeze", help="Record SHA-256, size and completion time of the hand-made files.")
    args = parser.parse_args()
    if args.command == "select":
        select(args.seed)
    elif args.command == "consensus":
        consensus((args.annotator_a, args.annotator_b), args.out)
    elif args.command == "freeze":
        freeze()
    else:
        build_gt(args.annotations)


if __name__ == "__main__":
    main()
