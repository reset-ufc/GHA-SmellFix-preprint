"""Inter-rater agreement (Cohen's kappa) for the manual validation — CSV-native.

Same duplas, same per-field + combined kappa, same "third question is polarity-gated"
rule (construct_ok for GT-positives, edit_type for GT-negatives). Reads the published
``data/validation/verdicts_raters.csv`` — no ``.db`` dependency, reproduces offline.

The unit is the derivation group (``gkey``): each rater has one decision per gkey
(``save_group`` writes the same label to every member stem), so rows are collapsed
to one per ``(gkey, validator)`` before measuring agreement — counting per stem would
inflate n by the group size.

    from utils import inter_rater
    inter_rater.agreement_summary()        # one row per dupla + OVERALL
    inter_rater.analyze_pair(df, "R1", "R2")
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from utils.dataloader import ROOT

VERDICTS_RATERS = ROOT / "data" / "validation" / "verdicts_raters.csv"

# The three validation duplas (inter-rater pairs).
DUPLAS = {
    "R1 & R2": ("R1", "R2"),
    "R3 & R4": ("R3", "R4"),
    "R5 & R6": ("R5", "R6"),
}
# Question D of the refactoring codebook: does the change introduce an evident config risk?
IMPACT = "potential_behavior_impact"
FIELDS = ("verdict", IMPACT, "construct_ok", "edit_type")
_LABEL_COLS = ("gkey", "validator", "verdict", IMPACT, "construct_ok", "gt_positive", "edit_type")


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def _is_pos(v) -> bool:
    return _clean(v).lower() == "true"


def load_rater_labels(csv: str | None = None) -> pd.DataFrame:
    """Load one decision per ``(gkey, validator)`` from the raters CSV (the verdicts
    of a gkey's member stems are identical, so the first is taken)."""
    csv = csv or VERDICTS_RATERS
    df = pd.read_csv(csv, dtype=str).fillna("")
    for c in _LABEL_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[df["validator"].map(_clean) != ""]
    df = df[df["gkey"].map(_clean) != ""]
    return df.drop_duplicates(["gkey", "validator"], keep="first")[list(_LABEL_COLS)].reset_index(drop=True)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two aligned label lists; None if < 2 pairs, NaN if degenerate
    (sklearn returns NaN when a rater used a single category throughout)."""
    if len(a) < 2 or len(a) != len(b):
        return None
    with warnings.catch_warnings():  # a degenerate (single-category) slice -> NaN, not a warning
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b))


def _third(row: pd.Series) -> str:
    """The polarity-gated third field: construct_ok for GT-positives, else edit_type."""
    return _clean(row["construct_ok"]) if _is_pos(row["gt_positive"]) else _clean(row["edit_type"])


def analyze_pair(df: pd.DataFrame, v1: str, v2: str) -> dict:
    """Progress, per-field + combined Cohen's kappa, and disagreement counts for one
    dupla, over the gkeys both raters judged."""
    a = df[df["validator"] == v1].set_index("gkey")
    b = df[df["validator"] == v2].set_index("gkey")
    keys_a, keys_b = set(a.index), set(b.index)
    common = sorted(keys_a & keys_b)
    stats = {
        "dupla": f"{v1} & {v2}", "v1": v1, "v2": v2,
        "assigned": len(keys_a | keys_b), "completed": len(common),
        "remaining": len(keys_a | keys_b) - len(common),
        "disagreements": {f: 0 for f in FIELDS}, "any_disagreements": 0,
        "kappa": {f: None for f in FIELDS}, "combined_kappa": None,
    }
    if not common:
        return stats

    cols = {f: ([], []) for f in FIELDS}      # aligned label lists per field
    comb_a, comb_b = [], []
    for gk in common:
        r1, r2 = a.loc[gk], b.loc[gk]
        pos = _is_pos(r1["gt_positive"])
        per = {f: (_clean(r1[f]), _clean(r2[f])) for f in FIELDS}
        for f, (x, y) in per.items():
            cols[f][0].append(x); cols[f][1].append(y)
            if x != y:
                stats["disagreements"][f] += 1
        third1 = per["construct_ok"] if pos else per["edit_type"]
        third_diff = third1[0] != third1[1]
        if per["verdict"][0] != per["verdict"][1] or per[IMPACT][0] != per[IMPACT][1] or third_diff:
            stats["any_disagreements"] += 1
        comb_a.append("||".join([per["verdict"][0], per[IMPACT][0], third1[0]]))
        comb_b.append("||".join([per["verdict"][1], per[IMPACT][1], third1[1]]))

    for f in FIELDS:
        stats["kappa"][f] = cohen_kappa(cols[f][0], cols[f][1])
    stats["combined_kappa"] = cohen_kappa(comb_a, comb_b)
    stats["agreement_pct"] = round((stats["completed"] - stats["any_disagreements"]) / stats["completed"] * 100, 2)
    return stats


def agreement_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per dupla + an OVERALL row: co-judged groups, percent agreement,
    per-field and combined Cohen's kappa. The OVERALL kappa pools every dupla's
    combined labels (not an average of kappas)."""
    df = load_rater_labels() if df is None else df
    rows, all_a, all_b = [], [], []
    tot_completed = tot_disagree = 0
    for _, (v1, v2) in DUPLAS.items():
        s = analyze_pair(df, v1, v2)
        rows.append({
            "dupla": s["dupla"], "co_judged": s["completed"],
            "agreement_pct": s.get("agreement_pct", 0.0),
            "kappa_verdict": s["kappa"]["verdict"], f"kappa_{IMPACT}": s["kappa"][IMPACT],
            "kappa_construct": s["kappa"]["construct_ok"], "kappa_edit_type": s["kappa"]["edit_type"],
            "kappa_combined": s["combined_kappa"],
        })
        tot_completed += s["completed"]; tot_disagree += s["any_disagreements"]
        # pooled combined labels for the OVERALL kappa
        a = df[df["validator"] == v1].set_index("gkey")
        b = df[df["validator"] == v2].set_index("gkey")
        for gk in sorted(set(a.index) & set(b.index)):
            r1, r2 = a.loc[gk], b.loc[gk]
            all_a.append("||".join([_clean(r1["verdict"]), _clean(r1[IMPACT]), _third(r1)]))
            all_b.append("||".join([_clean(r2["verdict"]), _clean(r2[IMPACT]), _third(r2)]))
    rows.append({
        "dupla": "OVERALL", "co_judged": tot_completed,
        "agreement_pct": round((tot_completed - tot_disagree) / tot_completed * 100, 2) if tot_completed else 0.0,
        "kappa_verdict": None, f"kappa_{IMPACT}": None,
        "kappa_construct": None, "kappa_edit_type": None,
        "kappa_combined": cohen_kappa(all_a, all_b),
    })
    return pd.DataFrame(rows)


def disagreements(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every gkey where a dupla's two raters differ on any (polarity-gated) field —
    the reconciliation worklist, with both sides shown for audit."""
    df = load_rater_labels() if df is None else df
    out = []
    for _, (v1, v2) in DUPLAS.items():
        a = df[df["validator"] == v1].set_index("gkey")
        b = df[df["validator"] == v2].set_index("gkey")
        for gk in sorted(set(a.index) & set(b.index)):
            r1, r2 = a.loc[gk], b.loc[gk]
            pos = _is_pos(r1["gt_positive"])
            third = "construct_ok" if pos else "edit_type"
            diffs = [f for f in ("verdict", IMPACT, third) if _clean(r1[f]) != _clean(r2[f])]
            if diffs:
                out.append({"dupla": f"{v1} & {v2}", "gkey": gk, "fields": ", ".join(diffs),
                            "gt_positive": "True" if pos else "False",
                            f"{v1}_verdict": _clean(r1["verdict"]), f"{v2}_verdict": _clean(r2["verdict"]),
                            f"v1_{IMPACT}": _clean(r1[IMPACT]), f"v2_{IMPACT}": _clean(r2[IMPACT]),
                            "v1_third": _clean(r1[third]), "v2_third": _clean(r2[third])})
    return pd.DataFrame(out)
