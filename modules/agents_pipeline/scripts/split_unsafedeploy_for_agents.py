#!/usr/bin/env python3
"""Build the UnsafeDeployOnFork agent-pipeline ground-truth splits for the
gemma/deepseek (full) and GPT/Haiku (close-n15) catch-up runs.

Context
-------
The synthesis delta for the ``UnsafeDeployOnFork`` smell lives, fully
single-smell stripped, in this repo's ``data/`` (frozen Step 12c / 12d artifacts):

  * ``12d_synth_UnsafeDeployOnFork_present.json``           -> synthetic positives (``##synth``)
  * ``12c_synthesis_targets_UnsafeDeployOnFork_absent.json`` -> manual-validated negatives

Those source files are *candidate* sets, not the experiment delta:

  * ``12d`` holds 49 candidates; 4 were rejected when the final GT
    (``13_final_ground_truth.json``, Step 13) was merged.
  * ``12c`` holds 50 negatives; 5 of them are the *originals* already present in
    ``12b_final_ground_truth.json`` (so gemma/deepseek already ran them, and
    GPT/Haiku already ran them at n15).

This script filters both source files against the final GT (Step 13)
and against 12b_final so that we emit EXACTLY the 45 + 45 = 90 instances that are
(a) part of the official final ground truth and (b) NOT already evaluated on the
other machine.

Outputs (written to ``<repo>/data``)
------------------------------------
``12f_unsafedeploy_synth90_full.json``
    90 records (45 positive + 45 negative), single-smell. Feeds the gemma and
    deepseek *full* catch-up configs. Merged with their 12b_final results ->
    50 + 50 for this smell.

``12f_ground_truth_n15fill_UnsafeDeployOnFork.json``
    20 records (10 positive + 10 negative), drawn as 10 *paired* base workflows
    (each contributes its ``##synth`` positive AND its clean negative). Feeds the
    GPT and Haiku n15-fill configs. Merged with their existing 5 + 5 -> 15 + 15,
    i.e. the n15 (<=15/polarity) cap is closed for this smell.

``ground_truth_n15.json``
    The n15 ground truth of the paper: the records of ``12b_ground_truth_n15.json``
    plus the n15 fill (:func:`merge_n15`), 570 pairs (15 + 15 per smell). Every model
    ran on these pairs, and the analysis scores them.

Determinism: every selection is sorted by file_key; no randomness. Re-running
overwrites the outputs byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SMELL = "UnsafeDeployOnFork"
SYNTH_MARKER = "##synth"
N_FILL_PER_POLARITY = 10  # GPT/Haiku already have 5/polarity at n15; +10 -> 15.

# Resolve repo paths from this file's location:
#   <repo>/modules/agents_pipeline/scripts/split_unsafedeploy_for_agents.py
SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]                 # replication-package


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        sys.exit(f"FATAL: {path} is not a JSON array")
    return data


def _keys_for_polarity(entries: list[dict], polarity: str) -> dict[str, str]:
    """Map file_key -> yaml_content for every record carrying SMELL in the given
    polarity bucket ('present_smells' or 'absent_smells')."""
    out: dict[str, str] = {}
    for e in entries:
        if any(s.get("pascal") == SMELL for s in e.get(polarity, [])):
            out[e["file_key"]] = e.get("yaml_content", "")
    return out


def _strip_synth(file_key: str) -> str:
    return file_key[: -len(SYNTH_MARKER)] if file_key.endswith(SYNTH_MARKER) else file_key


def merge_n15(n15: list[dict], fill: list[dict]) -> list[dict]:
    """The n15 records followed by the fill, without mutating either.

    A fill negative whose workflow is already in ``n15`` adds its smell to that record's
    ``absent_smells``. The fill positives come right after the n15 records, then the other
    fill negatives, both in fill order.
    """
    out = [dict(r) for r in n15]
    index = {r["file_key"]: i for i, r in enumerate(out)}
    positives, negatives = [], []
    for r in fill:
        if r["present_smells"]:
            positives.append(r)
        elif r["file_key"] in index:
            record = out[index[r["file_key"]]]
            listed = {s["pascal"] for s in record["absent_smells"]}
            record["absent_smells"] = [
                *record["absent_smells"],
                *(s for s in r["absent_smells"] if s["pascal"] not in listed),
            ]
        else:
            negatives.append(r)
    return out + positives + negatives


def dump_json(records: list[dict]) -> str:
    return json.dumps(records, indent=2, ensure_ascii=False) + "\n"


def _assert_single_smell(record: dict, expect_present: bool, src: str) -> None:
    pres = [s.get("pascal") for s in record.get("present_smells", [])]
    absent = [s.get("pascal") for s in record.get("absent_smells", [])]
    if expect_present:
        ok = pres == [SMELL] and absent == []
    else:
        ok = pres == [] and absent == [SMELL]
    if not ok:
        sys.exit(
            f"FATAL: {src} record {record.get('file_key')!r} is not single-smell "
            f"(present={pres}, absent={absent}). Source must be pre-stripped."
        )


def _rel(path: Path, root: Path) -> str:
    """Display path relative to root; full path if outside (e.g. a custom --out-dir)."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None, help="defaults to <repo-root>/data")
    ap.add_argument("--n-fill", type=int, default=N_FILL_PER_POLARITY)
    args = ap.parse_args()

    rp_data = args.repo_root / "data"
    out_dir = args.out_dir or rp_data

    p_12d = rp_data / "12d_synth_UnsafeDeployOnFork_present.json"
    p_12c = rp_data / "12c_synthesis_targets_UnsafeDeployOnFork_absent.json"
    p_12b = rp_data / "12b_final_ground_truth.json"
    p_final = rp_data / "13_final_ground_truth.json"
    p_n15 = rp_data / "12b_ground_truth_n15.json"
    for p in (p_12d, p_12c, p_12b, p_final, p_n15):
        if not p.exists():
            sys.exit(f"FATAL: missing input {p}")

    d_present = _load(p_12d)   # synthetic positive candidates (49), frozen (Step 12d)
    c_absent = _load(p_12c)    # negative targets (50)
    b_final = _load(p_12b)     # full base GT (originals)
    e_gt = _load(p_final)      # final experiment GT (50 pos + 50 neg for SMELL) = Step 13

    # --- reference sets from the official 12e GT -----------------------------
    e_pos = _keys_for_polarity(e_gt, "present_smells")   # 50
    e_neg = _keys_for_polarity(e_gt, "absent_smells")    # 50
    o_pos = set(_keys_for_polarity(b_final, "present_smells"))  # 5 originals
    o_neg = set(_keys_for_polarity(b_final, "absent_smells"))   # 5 originals

    new_pos_keys = set(e_pos) - o_pos      # expect 45
    new_neg_keys = set(e_neg) - o_neg      # expect 45

    # --- flip / drift guards -------------------------------------------------
    if o_pos - set(e_pos):
        sys.exit(f"FATAL: original positive(s) dropped from 12e: {sorted(o_pos - set(e_pos))}")
    if o_neg - set(e_neg):
        sys.exit(f"FATAL: original negative(s) dropped from 12e: {sorted(o_neg - set(e_neg))}")
    flips = (o_pos & set(e_neg)) | (o_neg & set(e_pos))
    if flips:
        sys.exit(f"FATAL: polarity flip on original instance(s): {sorted(flips)}")

    # --- positives: take the 12d records that survived into 12e --------------
    pos_records, pos_rejected = [], []
    for r in d_present:
        fk = r["file_key"]
        if fk in new_pos_keys:
            _assert_single_smell(r, expect_present=True, src="12d")
            if r.get("yaml_content", "") != e_pos[fk]:
                sys.exit(f"FATAL: 12d yaml_content for {fk!r} != 12e yaml (stale source)")
            pos_records.append(r)
        else:
            pos_rejected.append(fk)

    # --- negatives: take the 12c records that are new (not originals) --------
    neg_records, neg_excluded_orig = [], []
    for r in c_absent:
        fk = r["file_key"]
        if fk in new_neg_keys:
            _assert_single_smell(r, expect_present=False, src="12c")
            if r.get("yaml_content", "") != e_neg[fk]:
                sys.exit(f"FATAL: 12c yaml_content for {fk!r} != 12e yaml (stale source)")
            neg_records.append(r)
        elif fk in o_neg:
            neg_excluded_orig.append(fk)

    # --- coverage checks: every official new key must be sourced -------------
    missing_pos = new_pos_keys - {r["file_key"] for r in pos_records}
    missing_neg = new_neg_keys - {r["file_key"] for r in neg_records}
    if missing_pos:
        sys.exit(f"FATAL: new positives absent from 12d: {sorted(missing_pos)}")
    if missing_neg:
        sys.exit(f"FATAL: new negatives absent from 12c: {sorted(missing_neg)}")
    if len(pos_records) != 45 or len(neg_records) != 45:
        sys.exit(f"FATAL: expected 45+45, got {len(pos_records)}+{len(neg_records)}")

    pos_records.sort(key=lambda r: r["file_key"])
    neg_records.sort(key=lambda r: r["file_key"])
    ninety = pos_records + neg_records

    # --- n15 fill: 10 PAIRED base workflows (synth-pos + its clean neg) -------
    pos_by_base = {_strip_synth(r["file_key"]): r for r in pos_records}
    neg_by_key = {r["file_key"]: r for r in neg_records}
    paired_bases = sorted(set(pos_by_base) & set(neg_by_key))
    if len(paired_bases) < args.n_fill:
        sys.exit(f"FATAL: only {len(paired_bases)} paired bases, need {args.n_fill}")
    chosen = paired_bases[: args.n_fill]
    fill = [pos_by_base[b] for b in chosen] + [neg_by_key[b] for b in chosen]

    # fill must be a strict subset of the 90
    ninety_keys = {r["file_key"] for r in ninety}
    fill_keys = {r["file_key"] for r in fill}
    if not fill_keys <= ninety_keys:
        sys.exit(f"FATAL: fill not subset of 90: {sorted(fill_keys - ninety_keys)}")
    if len(fill) != 2 * args.n_fill:
        sys.exit(f"FATAL: expected {2 * args.n_fill} fill records, got {len(fill)}")

    n15_gt = merge_n15(_load(p_n15), fill)

    # --- write ---------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    p_full = out_dir / "12f_unsafedeploy_synth90_full.json"
    p_fill = out_dir / "12f_ground_truth_n15fill_UnsafeDeployOnFork.json"
    p_gt = out_dir / "ground_truth_n15.json"
    p_full.write_text(dump_json(ninety), encoding="utf-8")
    p_fill.write_text(dump_json(fill), encoding="utf-8")
    p_gt.write_text(dump_json(n15_gt), encoding="utf-8")

    # --- audit ---------------------------------------------------------------
    print(f"smell                 : {SMELL}")
    print(f"12d candidates        : {len(d_present)}  -> kept {len(pos_records)} pos "
          f"(rejected {len(pos_rejected)} not in 12e: {sorted(pos_rejected)})")
    print(f"12c targets           : {len(c_absent)}  -> kept {len(neg_records)} neg "
          f"(excluded {len(neg_excluded_orig)} originals already run: {sorted(neg_excluded_orig)})")
    print(f"FULL  (gemma/deepseek): {len(ninety):>3} records  -> {_rel(p_full, args.repo_root)}")
    print(f"N15-FILL (gpt/haiku)  : {len(fill):>3} records  ({args.n_fill}+{args.n_fill}, "
          f"{len(chosen)} paired bases)  -> {_rel(p_fill, args.repo_root)}")
    print(f"N15 GROUND TRUTH      : {len(n15_gt):>3} records  -> {_rel(p_gt, args.repo_root)}")
    print("paired bases chosen   :")
    for b in chosen:
        print(f"    {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
