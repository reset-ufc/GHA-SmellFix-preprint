"""Shared single-occurrence filter + inter-rater reliability (Cohen's kappa).

ONE implementation of the manual-evaluation summary, used by both:

  * Step 11 (``11_summarize_evaluation``) - the base manual evaluation, and
  * Step 12e (``12e_recalc_kappa``) - the recalculation that folds in the
    synthetic-instance evaluations.

Before this module the two steps carried byte-copies of the same functions,
which had already drifted (Step 12e's Landis-Koch band said ``Subsubstantial``
where Step 11 correctly said ``Substantial``). Both now import from here, so the
kappa math and the interpretation bands cannot diverge again. Step 12e adds only
:func:`combine_evaluations` on top; everything else lives here.

All functions are pure w.r.t. their arguments (they only read the CSV path they
are given and write the CSV path they are given); no notebook globals, no chdir.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def is_multi_occurrence_present(row: dict) -> bool:
    """True for a consensus present whose final_count exceeds one occurrence."""
    if row.get("final_decision") != "present":
        return False
    try:
        return int(float(row.get("final_count") or 0)) > 1
    except (TypeError, ValueError):
        return False


def filter_one_occurrence(eval_path: Path, filtered_path: Path) -> dict:
    """Write the single-occurrence variant; return drop statistics.

    Keeps every row except multi-occurrence present rows (final_decision ==
    "present" and final_count > 1), preserving the original header and column
    order so downstream steps read an identical schema.
    """
    with Path(eval_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [r for r in rows if not is_multi_occurrence_present(r)]
    dropped = [r for r in rows if is_multi_occurrence_present(r)]

    filtered_path = Path(filtered_path)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with filtered_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    dropped_by_smell: dict[str, int] = {}
    for r in dropped:
        dropped_by_smell[r["target_smell"]] = dropped_by_smell.get(r["target_smell"], 0) + 1

    return {
        "n_in": len(rows),
        "n_out": len(kept),
        "n_dropped": len(dropped),
        "dropped_by_smell": dict(
            sorted(dropped_by_smell.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }


def landis_koch(kappa: float | None) -> str:
    """Landis & Koch (1977) strength-of-agreement band for a kappa value."""
    if kappa is None or math.isnan(kappa):
        return "undefined"
    if kappa <= 0.00:
        return "Poor"
    if kappa <= 0.20:
        return "Slight"
    if kappa <= 0.40:
        return "Fair"
    if kappa <= 0.60:
        return "Moderate"
    if kappa <= 0.80:
        return "Substantial"
    return "Almost perfect"


def compute_irr(eval_path: Path) -> dict:
    """Cohen's kappa per annotator pair + overall (mean +/- std) over the rows of
    ``eval_path`` (decisions from decision_a / decision_b)."""
    rows: list[dict] = []
    with Path(eval_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    pairs: dict[str, list[dict]] = {}
    for row in rows:
        pair_id = f"{row['annotator_a']}-{row['annotator_b']}"
        pairs.setdefault(pair_id, []).append(row)

    pair_results: list[dict] = []
    for pair_id, pair_rows in sorted(pairs.items()):
        pp = pa = ap = aa = 0
        a_name = pair_rows[0]["annotator_a"]
        b_name = pair_rows[0]["annotator_b"]
        y_a: list[str] = []
        y_b: list[str] = []
        for r in pair_rows:
            da, db = r["decision_a"], r["decision_b"]
            y_a.append(da)
            y_b.append(db)
            if da == "present" and db == "present":
                pp += 1
            elif da == "present" and db == "absent":
                pa += 1
            elif da == "absent" and db == "present":
                ap += 1
            else:
                aa += 1
        n = pp + pa + ap + aa
        kappa = cohen_kappa_score(y_a, y_b) if n > 0 else None
        if kappa is not None and math.isnan(kappa):
            kappa = None
        interp = landis_koch(kappa) if kappa is not None else "undefined"
        pair_results.append(
            {
                "pair_id": pair_id,
                "annotator_a": a_name,
                "annotator_b": b_name,
                "n": n,
                "confusion_matrix": {"pp": pp, "pa": pa, "ap": ap, "aa": aa},
                "concordance_pct": round((pp + aa) / n * 100, 2) if n > 0 else None,
                "kappa": round(kappa, 4) if kappa is not None else None,
                "kappa_interpretation": interp,
            }
        )

    total_pp = sum(p["confusion_matrix"]["pp"] for p in pair_results)
    total_pa = sum(p["confusion_matrix"]["pa"] for p in pair_results)
    total_ap = sum(p["confusion_matrix"]["ap"] for p in pair_results)
    total_aa = sum(p["confusion_matrix"]["aa"] for p in pair_results)
    total_n = total_pp + total_pa + total_ap + total_aa

    valid_kappas = [p["kappa"] for p in pair_results if p["kappa"] is not None]
    mean_kappa = round(statistics.mean(valid_kappas), 4) if valid_kappas else None
    std_kappa = round(statistics.stdev(valid_kappas), 4) if len(valid_kappas) > 1 else None
    mean_interp = landis_koch(mean_kappa) if mean_kappa is not None else "undefined"

    return {
        "n_pairs": len(pair_results),
        "n_rows": len(rows),
        "pairs": pair_results,
        "overall": {
            "n": total_n,
            "confusion_matrix": {
                "pp": total_pp,
                "pa": total_pa,
                "ap": total_ap,
                "aa": total_aa,
            },
            "concordance_pct": round((total_pp + total_aa) / total_n * 100, 2)
            if total_n > 0
            else None,
        },
        "mean_kappa": mean_kappa,
        "std_kappa": std_kappa,
        "mean_kappa_interpretation": mean_interp,
    }


def write_irr_csv(result: dict, output_path: Path) -> None:
    """Write the per-pair + mean IRR table to ``output_path`` (CSV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "annotator_a",
        "annotator_b",
        "n",
        "pp",
        "pa",
        "ap",
        "aa",
        "concordance_pct",
        "kappa",
        "kappa_interpretation",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for p in result["pairs"]:
            cm = p["confusion_matrix"]
            w.writerow(
                {
                    "pair_id": p["pair_id"],
                    "annotator_a": p["annotator_a"],
                    "annotator_b": p["annotator_b"],
                    "n": p["n"],
                    "pp": cm["pp"],
                    "pa": cm["pa"],
                    "ap": cm["ap"],
                    "aa": cm["aa"],
                    "concordance_pct": p["concordance_pct"],
                    "kappa": p["kappa"],
                    "kappa_interpretation": p["kappa_interpretation"],
                }
            )
        ov = result["overall"]
        w.writerow(
            {
                "pair_id": "mean",
                "annotator_a": "",
                "annotator_b": "",
                "n": ov["n"],
                "pp": ov["confusion_matrix"]["pp"],
                "pa": ov["confusion_matrix"]["pa"],
                "ap": ov["confusion_matrix"]["ap"],
                "aa": ov["confusion_matrix"]["aa"],
                "concordance_pct": ov["concordance_pct"],
                "kappa": result["mean_kappa"],
                "kappa_interpretation": result["mean_kappa_interpretation"],
            }
        )


def print_summary(report: dict) -> None:
    """Pretty-print the per-pair + mean IRR table to stdout."""
    header = f"  {'Pair':<12} {'N':>5} {'Kappa':>7} {'Interpretation':<18} {'Concordance':>12}"
    sep = f"  {'-' * 12} {'-' * 5} {'-' * 7} {'-' * 18} {'-' * 12}"
    print()
    print(header)
    print(sep)
    for p in report["pairs"]:
        k = f"{p['kappa']:.4f}" if p["kappa"] is not None else "   N/A"
        c = f"{p['concordance_pct']:.2f}%" if p["concordance_pct"] is not None else "    N/A"
        print(f"  {p['pair_id']:<12} {p['n']:>5} {k:>7} {p['kappa_interpretation']:<18} {c:>12}")
    print(sep)
    mk = report["mean_kappa"]
    sk = report["std_kappa"]
    k_str = f"{mk:.4f}" if mk is not None else "   N/A"
    std_str = f" ±{sk:.4f}" if sk is not None else ""
    ov = report["overall"]
    c = f"{ov['concordance_pct']:.2f}%" if ov["concordance_pct"] is not None else "    N/A"
    print(f"  {'mean':<12} {ov['n']:>5} {k_str:>7}{std_str:<18} {c:>12}")
    print()
