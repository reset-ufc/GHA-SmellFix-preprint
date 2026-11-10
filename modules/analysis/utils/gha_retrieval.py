"""GHA-based candidate retrieval: GHA's flags checked against the manual labels.

Robustness diagnostic behind the paper's *GHA-based candidate retrieval* paragraph
(section *Robustness and Diagnostic Analyses*). The dataset pipeline drew every
``(workflow, smell)`` candidate from GHA's output: a ``positive`` candidate
(``tool_output``) is a workflow where GHA flags the target smell, a ``negative`` one
a workflow where GHA does not. The annotators then labeled each candidate
``present`` or ``absent`` (``final_decision``).

The manual labels come in two disjoint files that together hold every candidate
with a final label:

- ``data/11_manual_evaluation.csv`` — the candidates whose label agrees with GHA.
  The ground truth is built from these only.
- ``data/manual_evaluation_gha_overturned.csv`` — the candidates whose label
  contradicts GHA: flagged candidates labeled ``absent`` (*rejected*) and unflagged
  candidates labeled ``present`` (*missed* by GHA).

A flagged candidate is *confirmed* when its label is ``present``, whatever its
occurrence count.

Run ``python gha_retrieval.py`` for a self-check.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from utils.dataloader import ROOT
from utils.detection import SMELL_ORDER


AGREED_LABELS = ROOT / "data" / "11_manual_evaluation.csv"
OVERTURNED_LABELS = ROOT / "data" / "manual_evaluation_gha_overturned.csv"
LABEL_COLUMNS = ["file_key", "target_smell", "tool_output", "final_decision"]

# Confirmation-rate thresholds of the paper's paragraph: smells with at least 90%
# confirmation and smells below 70%.
HIGH_CONFIRMATION_PCT = 90.0
LOW_CONFIRMATION_PCT = 70.0

BY_SMELL_COLUMNS = [
    "smell",
    "flagged_inspected",
    "flagged_confirmed",
    "flagged_rejected",
    "confirmation_pct",
    "unflagged_inspected",
    "unflagged_missed",
    "miss_pct",
]


def _pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 2) if whole else None


def _contradicts_gha(labels: pd.DataFrame) -> pd.Series:
    return (labels["tool_output"] == "positive") != (labels["final_decision"] == "present")


def combine_labels(agreed: pd.DataFrame, overturned: pd.DataFrame) -> pd.DataFrame:
    """Every labeled candidate, from the two files; ``ValueError`` if they break the split.

    Checks that the values are known, that every agreed row agrees with GHA and every
    overturned row contradicts it, and that no ``(file_key, target_smell)`` pair
    appears twice.
    """
    labels = pd.concat([agreed[LABEL_COLUMNS], overturned[LABEL_COLUMNS]], ignore_index=True)
    for column, allowed in [
        ("target_smell", set(SMELL_ORDER)),
        ("tool_output", {"positive", "negative"}),
        ("final_decision", {"present", "absent"}),
    ]:
        unexpected = sorted(set(labels[column]) - allowed)
        if unexpected:
            raise ValueError(f"unexpected {column} values: {unexpected}")
    if _contradicts_gha(agreed).any():
        raise ValueError("an agreed row contradicts GHA")
    if not _contradicts_gha(overturned).all():
        raise ValueError("an overturned row agrees with GHA")
    duplicated = labels.duplicated(["file_key", "target_smell"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} (file_key, target_smell) pairs appear twice")
    return labels


def _read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_labels(
    agreed_path: Path = AGREED_LABELS, overturned_path: Path = OVERTURNED_LABELS
) -> pd.DataFrame:
    """Read and combine the two label files (see :func:`combine_labels`)."""
    return combine_labels(_read_labels(agreed_path), _read_labels(overturned_path))


def retrieval_by_smell(labels: pd.DataFrame, smells: list[str] | None = None) -> pd.DataFrame:
    """Confirmed, rejected and missed candidates per smell, in ``smells`` order
    (default: the paper order). A rate is NaN when its smell has no candidate."""
    records = []
    for smell in SMELL_ORDER if smells is None else smells:
        rows = labels[labels["target_smell"] == smell]
        present = rows["final_decision"] == "present"
        flagged = rows["tool_output"] == "positive"
        n_flagged, n_unflagged = int(flagged.sum()), int((~flagged).sum())
        confirmed, missed = int((flagged & present).sum()), int((~flagged & present).sum())
        records.append({
            "smell": smell,
            "flagged_inspected": n_flagged,
            "flagged_confirmed": confirmed,
            "flagged_rejected": n_flagged - confirmed,
            "confirmation_pct": _pct(confirmed, n_flagged),
            "unflagged_inspected": n_unflagged,
            "unflagged_missed": missed,
            "miss_pct": _pct(missed, n_unflagged),
        })
    return pd.DataFrame.from_records(records, columns=BY_SMELL_COLUMNS)


def _extremes(by_smell: pd.DataFrame, part: str, whole: str) -> dict | None:
    """Lowest and highest rate, with every smell that has it; rates are exact fractions.
    ``None`` when no smell has a candidate."""
    rows = by_smell[by_smell[whole] > 0]
    if rows.empty:
        return None
    rate = rows[part] / rows[whole]
    extremes = {}
    for side, pick in [("min", rate == rate.min()), ("max", rate == rate.max())]:
        first = rows[pick].iloc[0]
        extremes[side] = {
            "pct": _pct(int(first[part]), int(first[whole])),
            "smells": rows.loc[pick, "smell"].tolist(),
        }
    return extremes


def summarize(by_smell: pd.DataFrame) -> dict:
    """Totals over all smells, the smells on each side of the thresholds, and the ranges.

    The thresholds compare exact rates, not the rounded percentages.
    """
    flagged = int(by_smell["flagged_inspected"].sum())
    confirmed = int(by_smell["flagged_confirmed"].sum())
    unflagged = int(by_smell["unflagged_inspected"].sum())
    missed = int(by_smell["unflagged_missed"].sum())
    inspected = by_smell[by_smell["flagged_inspected"] > 0]
    scaled = 100 * inspected["flagged_confirmed"]
    high = scaled >= HIGH_CONFIRMATION_PCT * inspected["flagged_inspected"]
    low = scaled < LOW_CONFIRMATION_PCT * inspected["flagged_inspected"]
    return {
        "n_smells": len(by_smell),
        "flagged": {
            "inspected": flagged,
            "confirmed": confirmed,
            "rejected": flagged - confirmed,
            "confirmation_pct": _pct(confirmed, flagged),
        },
        "unflagged": {
            "inspected": unflagged,
            "missed": missed,
            "miss_pct": _pct(missed, unflagged),
        },
        "high_confirmation": {
            "threshold_pct": HIGH_CONFIRMATION_PCT,
            "smells": inspected.loc[high, "smell"].tolist(),
        },
        "low_confirmation": {
            "threshold_pct": LOW_CONFIRMATION_PCT,
            "smells": inspected.loc[low, "smell"].tolist(),
        },
        "confirmation_range": _extremes(by_smell, "flagged_confirmed", "flagged_inspected"),
        "miss_range": _extremes(by_smell, "unflagged_missed", "unflagged_inspected"),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gha_retrieval_report(
    out_dir: Path,
    agreed_path: Path = AGREED_LABELS,
    overturned_path: Path = OVERTURNED_LABELS,
) -> tuple[pd.DataFrame, dict]:
    """Write ``gha_retrieval_by_smell.csv`` and ``gha_retrieval.json`` to ``out_dir``.

    The JSON holds the :func:`summarize` values plus the path and SHA-256 of each input.
    """
    by_smell = retrieval_by_smell(load_labels(agreed_path, overturned_path))
    summary = summarize(by_smell)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_smell.to_csv(out_dir / "gha_retrieval_by_smell.csv", index=False, lineterminator="\n")
    report = summary | {
        "inputs": [
            {"path": os.path.relpath(path, ROOT), "sha256": _sha256(path)}
            for path in (agreed_path, overturned_path)
        ],
    }
    (out_dir / "gha_retrieval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return by_smell, summary


def _selfcheck() -> None:
    def rows(smell, confirmed=0, rejected=0, missed=0, clean=0):
        spec = [("positive", "present", confirmed), ("positive", "absent", rejected),
                ("negative", "present", missed), ("negative", "absent", clean)]
        return [
            {"file_key": f"o@r@{flag}-{label}-{i}.yml", "target_smell": smell,
             "tool_output": flag, "final_decision": label}
            for flag, label, n in spec for i in range(n)
        ]

    def split(records):
        frame = pd.DataFrame(records, columns=LABEL_COLUMNS)
        overturned = _contradicts_gha(frame)
        return frame[~overturned], frame[overturned]

    a, b, c = SMELL_ORDER[:3]
    labels = combine_labels(*split(
        rows(a, confirmed=9, rejected=1, missed=2, clean=6)
        + rows(b, confirmed=7, rejected=3)
        + rows(c, confirmed=69, rejected=31, clean=4)
    ))
    by_smell = retrieval_by_smell(labels, [a, b, c])
    first = by_smell.iloc[0]
    assert (first["flagged_inspected"], first["flagged_rejected"]) == (10, 1), first
    assert (first["confirmation_pct"], first["unflagged_missed"], first["miss_pct"]) == (
        90.0, 2, 25.0), first
    assert pd.isna(by_smell.iloc[1]["miss_pct"])  # no unflagged candidate inspected

    # 90% is high, 70% is not low, 69% is low.
    summary = summarize(by_smell)
    assert summary["high_confirmation"]["smells"] == [a], summary
    assert summary["low_confirmation"]["smells"] == [c], summary
    assert summary["flagged"] == {"inspected": 120, "confirmed": 85, "rejected": 35,
                                  "confirmation_pct": 70.83}, summary
    assert summary["confirmation_range"]["min"] == {"pct": 69.0, "smells": [c]}, summary
    assert summary["miss_range"] == {"min": {"pct": 0.0, "smells": [c]},
                                     "max": {"pct": 25.0, "smells": [a]}}, summary

    # Exact rates: 8999/10000 rounds to 90.0% but stays below the threshold.
    close = retrieval_by_smell(combine_labels(*split(rows(a, confirmed=8999, rejected=1001))), [a])
    assert close.iloc[0]["confirmation_pct"] == 89.99
    assert summarize(close)["high_confirmation"]["smells"] == []

    # The split must hold: agreed rows agree, overturned rows contradict, no pair twice.
    agreed, overturned = split(rows(a, confirmed=1, rejected=1))
    broken = [(overturned, overturned), (agreed, agreed), (agreed, pd.concat([overturned] * 2))]
    for pair in broken:
        try:
            combine_labels(*pair)
        except ValueError:
            continue
        raise AssertionError("combine_labels accepted a broken split")
    print("gha_retrieval self-check: OK")


if __name__ == "__main__":
    _selfcheck()
