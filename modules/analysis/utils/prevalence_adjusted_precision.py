"""Prevalence-adjusted precision: detection precision under the candidate-pool prevalence.

Robustness diagnostic behind the paper's *Balanced evaluation* paragraph (section
*Robustness and Diagnostic Analyses*). The n15 set holds 15 present and 15 absent pairs per
smell, so its precision assumes that half of the pairs are positive. This module recomputes
precision under the prevalence of the candidate pool that the ground truth was sampled from,
"using the observed recall and false-positive rate while keeping the same predictions fixed",
as the paragraph puts it.

**Candidate pool.** The workflows that Step 10 of the dataset pipeline sampled candidates from:
the rows of ``data/9_ground_truth_candidates.csv`` with at most the LOC cap of
``data/9b_loc_cap.json``, where GHA reports at most one instance of each study smell (Step 10's
strict filter, ``_passes_strict`` in ``modules/dataset/dataset_utils/ground_truth_sampling.py``;
the three presentational smells of the catalog do not count). Every pool workflow is a candidate
for every smell: *flagged* when GHA reports the smell, *unflagged* otherwise. The pool size and
each smell's flagged and unflagged counts must equal the ones in ``data/step10_report.json``.

**Prevalence.** For smell ``s``, with ``f_s`` the share of pool workflows that GHA flags::

    pi_s = f_s * c_s + (1 - f_s) * m_s

``c_s`` (confirmation rate) and ``m_s`` (miss rate) are the shares of inspected flagged and
unflagged candidates labeled ``present`` (see ``utils.gha_retrieval``). Only the labeled
candidates whose workflow is in the pool count. The pool prevalence ``pi`` is the mean of
``pi_s`` over the smells: the estimated share of present pairs among all ``(workflow, smell)``
pairs of the pool, where each smell has one pair per workflow.

**Adjusted metrics.** For one model and design, with ``Re`` and ``FPR`` its recall and
false-positive rate over the n15 pairs (rep1)::

    precision = pi Re / (pi Re + (1 - pi) FPR)

Recall does not depend on prevalence; F1 follows from the adjusted precision and the recall.
Every smell has as many positive as negative n15 pairs, so this is also the precision computed
smell by smell with the prevalence ``pi`` in every smell: it assumes one prevalence for all
smells, not equal error rates across smells. With ``pi = 0.5`` it gives the balanced values of
the paper's overall detection table. The precision order of two models or designs does
not depend on ``pi``: it follows the ratio ``FPR / Re``. A design's value is the mean over
the four models, as in that table's Avg. row.

Run ``python prevalence_adjusted_precision.py`` for a self-check.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from utils import detection
from utils.dataloader import ROOT
from utils.designs import DESIGNS
from utils.detection import SMELL_ORDER, TABLE_MODELS
from utils.gha_retrieval import AGREED_LABELS, OVERTURNED_LABELS, load_labels, retrieval_by_smell


CANDIDATES = ROOT / "data" / "9_ground_truth_candidates.csv"
LOC_CAP = ROOT / "data" / "9b_loc_cap.json"
STEP10_REPORT = ROOT / "data" / "step10_report.json"

PREVALENCE_COLUMNS = [
    "smell",
    "pool_flagged",
    "pool_unflagged",
    "flag_pct",
    "flagged_inspected",
    "flagged_confirmed",
    "confirmation_pct",
    "unflagged_inspected",
    "unflagged_missed",
    "miss_pct",
    "labeled_outside_pool",
    "prevalence",
    "prevalence_pct",
]
# Metrics of metrics_by_model, as fractions; the CSV writes them as percentages.
METRICS = [
    "balanced_precision",
    "adjusted_precision",
    "recall",
    "balanced_f1",
    "adjusted_f1",
]
MODEL_COLUMNS = ["design", "model", "model_alias", "TP", "FP", "FN", "TN", *METRICS]


def _ratio(part: float, whole: float) -> float:
    return part / whole if whole else float("nan")


def _pct(value: float) -> float:
    return round(100 * float(value), 2)


def _f1(precision: float, recall: float) -> float:
    return _ratio(2 * precision * recall, precision + recall)


def candidate_pool(candidates: pd.DataFrame, loc_cap: int) -> pd.DataFrame:
    """GHA's flags on the pool workflows: one row per workflow (index ``file_key``) and one bool
    column per study smell, True when GHA flags the smell.

    ``candidates`` holds ``file_key``, ``file_line_count`` and ``smells_json`` (GHA's instance
    count of each catalog smell it reports). A workflow is in the pool when it has at most
    ``loc_cap`` lines and GHA reports at most one instance of each study smell.
    """
    study = set(SMELL_ORDER)
    flags = {}
    capped = candidates[candidates["file_line_count"] <= loc_cap]
    for file_key, smells_json in zip(capped["file_key"], capped["smells_json"], strict=True):
        counts = {
            smell["pascal"]: smell["instance_count"]
            for smell in json.loads(smells_json)
            if smell["pascal"] in study
        }
        if all(count <= 1 for count in counts.values()):
            flags[file_key] = [counts.get(smell, 0) >= 1 for smell in SMELL_ORDER]
    pool = pd.DataFrame.from_dict(flags, orient="index", columns=SMELL_ORDER, dtype=bool)
    return pool.rename_axis("file_key")


def load_candidate_pool(
    candidates_path: Path = CANDIDATES, loc_cap_path: Path = LOC_CAP
) -> pd.DataFrame:
    """Read the Step 09 candidates and the Step 09b LOC cap (see :func:`candidate_pool`)."""
    loc_cap = int(json.loads(Path(loc_cap_path).read_text(encoding="utf-8"))["cap"])
    candidates = pd.read_csv(
        candidates_path,
        usecols=["file_key", "file_line_count", "smells_json"],
        dtype={"file_key": str, "file_line_count": int, "smells_json": str},
        keep_default_na=False,
    )
    return candidate_pool(candidates, loc_cap)


def check_pool_against_step10(pool: pd.DataFrame, report_path: Path = STEP10_REPORT) -> None:
    """``ValueError`` unless ``pool`` has the size and the per-smell flagged and unflagged
    counts that Step 10 recorded when it sampled the candidates."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    recorded_size = report["sampling_config"]["strict_pool_size"]
    if len(pool) != recorded_size:
        raise ValueError(f"the pool has {len(pool)} workflows, Step 10 recorded {recorded_size}")
    for smell in SMELL_ORDER:
        stats = report["per_smell"][smell]
        flagged = int(pool[smell].sum())
        recorded = (stats["available_pos_strict"], stats["available_neg_strict"])
        if (flagged, len(pool) - flagged) != recorded:
            raise ValueError(
                f"{smell}: the pool has {flagged} flagged and {len(pool) - flagged} unflagged "
                f"workflows, Step 10 recorded {recorded[0]} and {recorded[1]}"
            )


def prevalence_by_smell(
    pool: pd.DataFrame, labels: pd.DataFrame, smells: list[str] | None = None
) -> pd.DataFrame:
    """Estimated prevalence of each smell in ``pool``, with its inputs, in ``smells`` order
    (default: the paper order).

    ``labels`` holds the labeled candidates (``utils.gha_retrieval.load_labels``). Only those
    whose workflow is in the pool count; ``labeled_outside_pool`` gives the number of the others.
    ``prevalence`` is exact, the ``*_pct`` columns are rounded. ``ValueError`` if a pool
    candidate's ``tool_output`` disagrees with GHA's flag in the pool, or if a smell has no
    inspected candidate among its flagged or its unflagged workflows.
    """
    smells = SMELL_ORDER if smells is None else smells
    inside = labels["file_key"].isin(pool.index)
    pooled = labels[inside]
    flagged_in_pool = [
        bool(pool.at[file_key, smell])
        for file_key, smell in zip(pooled["file_key"], pooled["target_smell"], strict=True)
    ]
    if [output == "positive" for output in pooled["tool_output"]] != flagged_in_pool:
        raise ValueError("a pool candidate's tool_output disagrees with GHA's flag in the pool")
    retrieval = retrieval_by_smell(pooled, smells).set_index("smell")
    outside = labels.loc[~inside, "target_smell"].value_counts()
    records = []
    for smell in smells:
        row = retrieval.loc[smell]
        n_flagged, n_unflagged = int(row["flagged_inspected"]), int(row["unflagged_inspected"])
        if not n_flagged or not n_unflagged:
            raise ValueError(f"{smell}: no inspected candidate among its flagged or unflagged pool")
        pool_flagged = int(pool[smell].sum())
        flag_rate = pool_flagged / len(pool)
        confirmation = int(row["flagged_confirmed"]) / n_flagged
        miss = int(row["unflagged_missed"]) / n_unflagged
        prevalence = flag_rate * confirmation + (1 - flag_rate) * miss
        records.append({
            "smell": smell,
            "pool_flagged": pool_flagged,
            "pool_unflagged": len(pool) - pool_flagged,
            "flag_pct": _pct(flag_rate),
            "flagged_inspected": n_flagged,
            "flagged_confirmed": int(row["flagged_confirmed"]),
            "confirmation_pct": _pct(confirmation),
            "unflagged_inspected": n_unflagged,
            "unflagged_missed": int(row["unflagged_missed"]),
            "miss_pct": _pct(miss),
            "labeled_outside_pool": int(outside.get(smell, 0)),
            "prevalence": prevalence,
            "prevalence_pct": _pct(prevalence),
        })
    return pd.DataFrame.from_records(records, columns=PREVALENCE_COLUMNS)


def pool_prevalence(prevalence: pd.DataFrame) -> float:
    """The pool prevalence: the mean over smells of :func:`prevalence_by_smell`'s
    ``prevalence``, i.e. the estimated share of present pairs among the pool's pairs."""
    return float(prevalence["prevalence"].mean())


def adjusted_precision(recall: float, false_positive_rate: float, prevalence: float) -> float:
    """Precision at ``prevalence`` of a detector with this recall and false-positive rate."""
    hits = prevalence * recall
    return _ratio(hits, hits + (1 - prevalence) * false_positive_rate)


def metrics_by_model(counts: pd.DataFrame, prevalence: float) -> pd.DataFrame:
    """Balanced and adjusted precision and F1, and the recall, of each model and design,
    as fractions, in the order of ``DESIGNS`` and ``TABLE_MODELS``.

    ``counts`` holds TP, FP, FN and TN per ``model_alias`` and ``design``, pooled over the
    smells (as ``utils.detection.metrics_by`` returns them); ``prevalence`` is the pool
    prevalence. Balanced values pool the counts, as the overall detection table does; adjusted
    values apply :func:`adjusted_precision` (module docstring). ``ValueError`` if a model and
    design has several rows, or not as many positive as negative pairs, or no pair.
    """
    records = []
    for design in DESIGNS:
        for model, alias in TABLE_MODELS:
            rows = counts[(counts["design"] == design) & (counts["model_alias"] == alias)]
            if rows.empty:
                continue
            if len(rows) > 1:
                raise ValueError(f"{alias} {design}: {len(rows)} rows of counts")
            tp, fp, fn, tn = (int(rows[column].iloc[0]) for column in ("TP", "FP", "FN", "TN"))
            if tp + fn != fp + tn or not tp + fn:
                raise ValueError(
                    f"{alias} {design}: {tp + fn} positive and {fp + tn} negative pairs, "
                    "expected as many of each and at least one"
                )
            recall = tp / (tp + fn)
            balanced = _ratio(tp, tp + fp)
            adjusted = adjusted_precision(recall, fp / (fp + tn), prevalence)
            records.append({
                "design": design,
                "model": model,
                "model_alias": alias,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "balanced_precision": balanced,
                "adjusted_precision": adjusted,
                "recall": recall,
                "balanced_f1": _f1(balanced, recall),
                "adjusted_f1": _f1(adjusted, recall),
            })
    return pd.DataFrame.from_records(records, columns=MODEL_COLUMNS)


def summarize(prevalence: pd.DataFrame, by_model: pd.DataFrame) -> dict:
    """The pool, the design means and the D2 - D1 differences, as percentages.

    - ``candidate_pool``: the pool size, the labeled candidates used and left out, the pool
      prevalence (:func:`pool_prevalence`) and the lowest and highest smell prevalence, with
      every smell that has it;
    - ``designs``: the mean over models of each metric of :func:`metrics_by_model`, as in
      the Avg. row of the overall detection table, the change in precision in percentage points
      and the false positives summed over the models;
    - ``d2_minus_d1``: the D2 mean minus the D1 mean of each metric, in percentage points.

    The changes and differences come from the unrounded means.
    """
    rates = prevalence["prevalence"]
    means = by_model.groupby("design", sort=False)[METRICS].mean()
    designs = {}
    for design, row in means.iterrows():
        designs[design] = {f"{metric}_pct": _pct(row[metric]) for metric in METRICS} | {
            "precision_change_pp": _pct(row["adjusted_precision"] - row["balanced_precision"]),
            "false_positives": int(by_model.loc[by_model["design"] == design, "FP"].sum()),
        }
    first = prevalence.iloc[0]
    return {
        "candidate_pool": {
            "workflows": int(first["pool_flagged"] + first["pool_unflagged"]),
            "labeled_candidates_used": int(
                prevalence["flagged_inspected"].sum() + prevalence["unflagged_inspected"].sum()
            ),
            "labeled_outside_pool": int(prevalence["labeled_outside_pool"].sum()),
            "prevalence_pct": _pct(pool_prevalence(prevalence)),
            "smell_prevalence_range": {
                side: {
                    "pct": _pct(value),
                    "smells": prevalence.loc[rates == value, "smell"].tolist(),
                }
                for side, value in [("min", rates.min()), ("max", rates.max())]
            },
        },
        "designs": designs,
        "d2_minus_d1": {
            f"{metric}_pp": _pct(means.at["D2", metric] - means.at["D1", metric])
            for metric in METRICS
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prevalence_adjusted_precision_report(
    out_dir: Path,
    results: pd.DataFrame,
    candidates_path: Path = CANDIDATES,
    loc_cap_path: Path = LOC_CAP,
    step10_report_path: Path = STEP10_REPORT,
    agreed_path: Path = AGREED_LABELS,
    overturned_path: Path = OVERTURNED_LABELS,
    n15_path: Path = detection.N15_GT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Write ``prevalence_by_smell.csv``, ``prevalence_adjusted_precision_by_model.csv`` and
    ``prevalence_adjusted_precision.json`` to ``out_dir``.

    ``results`` holds the agent runs (``utils.detection.load_results``); the detection counts
    are the rep1 counts on the n15 pairs of ``n15_path``. ``ValueError`` unless the pool matches
    Step 10 and every model has both designs. The model CSV writes the metrics as
    percentages; the JSON holds the :func:`summarize` values plus the path and SHA-256 of each
    input file.
    """
    pool = load_candidate_pool(candidates_path, loc_cap_path)
    check_pool_against_step10(pool, step10_report_path)
    prevalence = prevalence_by_smell(pool, load_labels(agreed_path, overturned_path))
    counts = detection.metrics_by(
        detection.add_n15_label(results, n15_path), "model_alias", "design"
    )
    by_model = metrics_by_model(counts, pool_prevalence(prevalence))
    expected_rows = len(TABLE_MODELS) * len(DESIGNS)
    if len(by_model) != expected_rows:
        raise ValueError(f"expected {expected_rows} model rows, got {len(by_model)}")
    summary = summarize(prevalence, by_model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(out_dir / "prevalence_by_smell.csv", index=False, lineterminator="\n")
    table = by_model.copy()
    for metric in METRICS:
        table[f"{metric}_pct"] = table.pop(metric).map(_pct)
    table.to_csv(
        out_dir / "prevalence_adjusted_precision_by_model.csv", index=False, lineterminator="\n"
    )
    inputs = [
        candidates_path, loc_cap_path, step10_report_path, agreed_path, overturned_path, n15_path
    ]
    report = summary | {
        "inputs": [
            {"path": os.path.relpath(path, ROOT), "sha256": _sha256(path)} for path in inputs
        ],
    }
    (out_dir / "prevalence_adjusted_precision.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return prevalence, by_model, summary


def _selfcheck() -> None:
    a, b = SMELL_ORDER[:2]

    def smells(counts: dict[str, int]) -> str:
        return json.dumps([{"pascal": s, "instance_count": n} for s, n in counts.items()])

    # Pool: the LOC cap, the strict filter over the study smells only, and GHA's flags.
    candidates = pd.DataFrame(
        [
            ("w1", 10, smells({a: 1})),
            ("w2", 10, smells({a: 1, "NameRunSteps": 5})),  # presentational smell: kept
            ("w3", 10, smells({b: 2})),  # two instances of a study smell: dropped
            ("w4", 11, smells({})),  # above the cap: dropped
            ("w5", 10, smells({})),
        ],
        columns=["file_key", "file_line_count", "smells_json"],
    )
    pool = candidate_pool(candidates, loc_cap=10)
    assert pool.index.tolist() == ["w1", "w2", "w5"], pool
    assert pool[a].tolist() == [True, True, False] and not pool[b].any(), pool

    # Step 10's counts must match.
    per_smell = {s: {"available_pos_strict": 0, "available_neg_strict": 3} for s in SMELL_ORDER}
    per_smell[a] = {"available_pos_strict": 2, "available_neg_strict": 1}
    step10 = {"sampling_config": {"strict_pool_size": 3}, "per_smell": per_smell}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step10_report.json"
        path.write_text(json.dumps(step10), encoding="utf-8")
        check_pool_against_step10(pool, path)
        step10["per_smell"][a]["available_pos_strict"] = 1
        path.write_text(json.dumps(step10), encoding="utf-8")
        try:
            check_pool_against_step10(pool, path)
        except ValueError:
            pass
        else:
            raise AssertionError("check_pool_against_step10 accepted a different pool")

    # Prevalence: 10 workflows; GHA flags a in 4 (f = 0.4) and b in 1 (f = 0.1).
    keys = [f"w{i}" for i in range(10)]
    pool = pd.DataFrame(
        {a: [i < 4 for i in range(10)], b: [i < 1 for i in range(10)]}, index=keys
    ).rename_axis("file_key")

    def label(file_key, smell, flagged, present):
        return {"file_key": file_key, "target_smell": smell,
                "tool_output": "positive" if flagged else "negative",
                "final_decision": "present" if present else "absent"}

    labels = pd.DataFrame([
        label("w0", a, True, True), label("w1", a, True, False),  # c = 1/2
        label("w4", a, False, True), label("w5", a, False, False),
        label("w6", a, False, False), label("w7", a, False, False),  # m = 1/4
        label("x9", a, True, False),  # outside the pool: left out
        label("w0", b, True, True),  # c = 1
        label("w1", b, False, False), label("w2", b, False, False),  # m = 0
    ])
    prevalence = prevalence_by_smell(pool, labels, [a, b])
    by_smell = prevalence.set_index("smell")
    assert abs(by_smell.at[a, "prevalence"] - (0.4 * 0.5 + 0.6 * 0.25)) < 1e-12, by_smell
    assert abs(by_smell.at[b, "prevalence"] - 0.1) < 1e-12, by_smell
    assert by_smell.at[a, "flagged_inspected"] == 2, by_smell
    assert by_smell.at[a, "labeled_outside_pool"] == 1, by_smell
    assert by_smell.at[a, "prevalence_pct"] == 35.0, by_smell
    assert abs(pool_prevalence(prevalence) - (0.35 + 0.1) / 2) < 1e-12, prevalence

    mismatch = labels.copy()
    mismatch.loc[2, "tool_output"] = "positive"  # w4 is unflagged for a in the pool
    try:
        prevalence_by_smell(pool, mismatch, [a, b])
    except ValueError:
        pass
    else:
        raise AssertionError("prevalence_by_smell accepted a tool_output that contradicts the pool")
    _selfcheck_metrics(prevalence)
    print("prevalence_adjusted_precision self-check: OK")


def _selfcheck_metrics(prevalence: pd.DataFrame) -> None:
    """The metrics and summary part of the self-check, on the toy ``prevalence`` of
    :func:`_selfcheck` (smell ``a`` at 35%, ``b`` at 10%)."""
    a = SMELL_ORDER[0]
    # Metrics: the pooled counts of one model, 30 positive and 30 negative pairs per design.
    alias = TABLE_MODELS[0][1]
    counts = pd.DataFrame([
        {"model_alias": alias, "design": "D1", "TP": 27, "FP": 3, "FN": 3, "TN": 27},
        {"model_alias": alias, "design": "D2", "TP": 29, "FP": 6, "FN": 1, "TN": 24},
    ])
    for _, row in metrics_by_model(counts, 0.5).iterrows():  # 50% gives the balanced values
        assert abs(row["adjusted_precision"] - row["balanced_precision"]) < 1e-12, row
        assert abs(row["adjusted_f1"] - row["balanced_f1"]) < 1e-12, row
    by_model = metrics_by_model(counts, 0.2)
    d1 = by_model.iloc[0]
    assert abs(d1["adjusted_precision"] - 0.2 * 0.9 / (0.2 * 0.9 + 0.8 * 0.1)) < 1e-12, d1
    assert abs(d1["recall"] - 0.9) < 1e-12, d1
    # D1's counts are two smells with 15 positive and 15 negative pairs each: the pooled rates
    # give the smell-by-smell precision with the same prevalence in both smells.
    two_smells = [(12, 3, 3, 12), (15, 0, 0, 15)]  # TP, FP, FN, TN
    hits = sum(0.2 * tp / (tp + fn) for tp, _, fn, _ in two_smells)
    alarms = sum(0.8 * fp / (fp + tn) for _, fp, _, tn in two_smells)
    assert abs(d1["adjusted_precision"] - hits / (hits + alarms)) < 1e-12, d1
    for broken in ({"TP": 26}, {"TP": 0, "FN": 0, "FP": 0, "TN": 0}):
        try:
            metrics_by_model(counts.iloc[:1].assign(**broken), 0.2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"metrics_by_model accepted the counts {broken}")

    summary = summarize(prevalence, by_model)
    pool = summary["candidate_pool"]
    assert pool["prevalence_pct"] == 22.5, summary
    assert pool["smell_prevalence_range"]["max"] == {"pct": 35.0, "smells": [a]}, summary
    assert summary["designs"]["D2"]["false_positives"] == 6, summary
    assert summary["d2_minus_d1"]["recall_pp"] == _pct(29 / 30 - 27 / 30), summary


if __name__ == "__main__":
    _selfcheck()
