"""Summary of the manual-validation result — "what went in, what came out".

CSV-native and offline: reads the published ``data/validation/manual_validation_final.csv``
(one resolved label per judged instance) plus ``disagreement_resolutions.csv``, and
returns a handful of small DataFrames describing the intake (what was judged) and the
outcomes (the resolved labels), so a notebook can display them without any logic of its
own.

    from utils import validation_summary as vs
    s = vs.summarize()
    s["intake"]; s["positive_outcomes"]; s["srr_by_design"]; s["headline"]
"""
from __future__ import annotations

import pandas as pd

from utils.dataloader import ROOT
from utils.designs import DESIGNS

FINAL_VALIDATION_CSV = ROOT / "data" / "validation" / "manual_validation_final.csv"
RESOLUTIONS_CSV = ROOT / "data" / "validation" / "disagreement_resolutions.csv"


VERDICT_POS = ("removed", "not removed", "uncertain")
VERDICT_NEG = ("unnecessary", "necessary", "uncertain", "preserved")


def _counts(series: pd.Series, order: tuple[str, ...]) -> pd.DataFrame:
    """Value counts as a (label, n, pct) frame, ordered, including zero-count labels."""
    vc = series.value_counts()
    labels = list(order) + [x for x in vc.index if x and x not in order]
    total = int(vc.sum())
    return pd.DataFrame([
        {"label": lab, "n": int(vc.get(lab, 0)),
         "pct": round(100 * int(vc.get(lab, 0)) / total, 1) if total else 0.0}
        for lab in labels if (lab in order or vc.get(lab, 0))
    ])


def summarize(final_csv: str | None = None, resolutions_csv: str | None = None) -> dict:
    """Build the intake + outcome summary. Returns a dict of DataFrames plus a flat
    ``headline`` dict of the key numbers."""
    final_csv = final_csv or FINAL_VALIDATION_CSV
    resolutions_csv = resolutions_csv or RESOLUTIONS_CSV
    f = pd.read_csv(final_csv, dtype=str).fillna("")
    f = f[f["design"].isin(DESIGNS)]
    res = pd.read_csv(resolutions_csv, dtype=str).fillna("")

    pos = f[f["gt_positive"] == "True"]
    neg = f[f["gt_positive"] == "False"]
    adj = f["is_adjudicated"].astype(str) == "True"
    removed = int((pos["resolved_verdict"] == "removed").sum())
    unc_pos = int((pos["resolved_verdict"] == "uncertain").sum())
    n_validators = int(f["validators"].str.split("; ").explode().str.strip().replace("", pd.NA).dropna().nunique())

    intake = pd.DataFrame([
        ("judged instances (stems)", len(f)),
        ("  GT-positive", len(pos)),
        ("  GT-negative", len(neg)),
        ("derivation groups (gkey)", int(f["gkey"].nunique())),
        ("validators", n_validators),
        ("disagreements resolved (gkey)", len(res)),
        ("  adjudicated instances", int(adj.sum())),
        ("  unanimous instances", int((~adj).sum())),
    ], columns=["item", "count"])

    # SRR per design (GT-positives only)
    srr = pos.assign(rem=(pos["resolved_verdict"] == "removed").astype(int),
                     unc=(pos["resolved_verdict"] == "uncertain").astype(int))
    by_design = srr.groupby("design").agg(n=("rem", "size"), removed=("rem", "sum"),
                                          uncertain=("unc", "sum")).reset_index()
    by_design["SRR"] = (by_design["removed"] / by_design["n"]).round(3)
    by_design["SRR_excl"] = (by_design["removed"]
                             / (by_design["n"] - by_design["uncertain"])).round(3)

    headline = {
        "judged_instances": len(f),
        "positives": len(pos), "negatives": len(neg),
        "gkeys": int(f["gkey"].nunique()),
        "resolved_disagreements": len(res),
        "adjudicated_instances": int(adj.sum()),
        "unanimous_instances": int((~adj).sum()),
        "SRR": round(removed / len(pos), 3) if len(pos) else None,
        "SRR_excl": round(removed / (len(pos) - unc_pos), 3) if (len(pos) - unc_pos) else None,
    }

    return {
        "intake": intake,
        "positive_outcomes": _counts(pos["resolved_verdict"], VERDICT_POS),
        "negative_outcomes": _counts(neg["resolved_verdict"], VERDICT_NEG),
        "construct_ok": _counts(pos["resolved_construct_ok"], ("yes", "no", "uncertain")),
        "potential_behavior_impact": _counts(
            f["resolved_potential_behavior_impact"], ("no", "yes", "uncertain")
        ),
        "resolution_source": _counts(f["resolution_source"], ("unanimous", "adjudicated")),
        "srr_by_design": by_design,
        "headline": headline,
    }
