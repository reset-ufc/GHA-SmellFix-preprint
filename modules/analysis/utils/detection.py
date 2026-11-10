"""RQ1 detection metrics: the model's prediction (``smell_detected``) scored
against ground truth, per model / design / smell.

Two truth scopes (``label``):

- ``gt_n15`` (default) — the balanced **n15** ground truth (570 pairs: 285
  present + 285 absent), joined from ``data/ground_truth_n15.json``. Every model
  is scored on the SAME pairs, so F1 is comparable across models. This is the
  apples-to-apples headline (paper ``tab:rq1_overall_detection``).
- ``ground_truth`` — the per-instance label baked into each result JSON. Scores
  each model on its own corpus (DeepSeek ran the full 1710-pair GT, the others
  the n15 subset), so per-model F1 is NOT directly comparable across models.

All metrics use rep1 (first execution) only, so models that ran k>1 reps are
compared on the same footing as single-run ones. The exception is the accuracy
spread (``accuracy_spread``), which compares the repetitions.

Run ``python detection.py`` for a self-check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from utils import dataloader
from utils.dataloader import ROOT
from utils.designs import DESIGNS


N15_GT = ROOT / "data" / "ground_truth_n15.json"
TABLE_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]
METRICS = [
    ("f1", "F1"),
    ("precision", "Pr"),
    ("recall", "Re"),
    ("accuracy", "Acc"),
]
SMELL_ORDER = [
    "MissingPermExternalActions",
    "UnpinnedActionVersion",
    "MissingTokenPermissions",
    "MissingForkGuardPR",
    "MissingJobTimeout",
    "MissingConcurrencyPR",
    "MissingConcurrencyBranch",
    "MissingPathFilters",
    "MissingForkGuardSchedule",
    "UnsafeArtifactUploadFork",
    "MissingUploadCondition",
    "UnsafeDeployOnFork",
    "MissingConcurrencyGlobal",
    "MutableRunnerVersion",
    "SuboptimalCacheUsage",
    "MultiCommandStep",
    "SingleOSExecution",
    "UnpinnedDependencies",
    "SingleLangVersionCI",
]
# Smell type (Step 2 of the paper's method), in SMELL_ORDER within each type. The
# paper's smell-level tables group their rows this way.
SMELL_TYPES = {
    "Localized": [
        "MissingPermExternalActions",
        "UnpinnedActionVersion",
        "MissingTokenPermissions",
        "MissingJobTimeout",
        "UnsafeArtifactUploadFork",
        "MissingUploadCondition",
        "MutableRunnerVersion",
        "SuboptimalCacheUsage",
        "MultiCommandStep",
        "UnpinnedDependencies",
    ],
    "Context-dependent": [
        "MissingForkGuardPR",
        "MissingConcurrencyPR",
        "MissingConcurrencyBranch",
        "MissingPathFilters",
        "MissingForkGuardSchedule",
        "UnsafeDeployOnFork",
        "MissingConcurrencyGlobal",
        "SingleOSExecution",
        "SingleLangVersionCI",
    ],
}
# Rotated label of each type in the smell-level detection table, as the paper writes
# it: ``\groupcell`` is defined in the paper's preamble.
_TYPE_LABELS = {
    "Localized": r"\groupcell{\textbf{Localized}}",
    "Context-dependent": "\n".join(
        [
            "",
            r"    \rotatebox[origin=c]{90}{",
            r"        \shortstack{",
            r"            \textsc{\scriptsize \textbf{Context-}}\\",
            r"            \textsc{\scriptsize \textbf{dependent}}",
            r"        }",
            r"    }",
            "",
        ]
    ),
}
SHADED_SMELL_COUNT = 5


def load_results(*args, **kwargs) -> pd.DataFrame:
    """Load result JSONs through the shared dataloader."""
    return dataloader.load_results(*args, **kwargs)


def add_n15_label(df: pd.DataFrame, gt_path: Path = N15_GT) -> pd.DataFrame:
    """Add ``gt_n15``: True/False when the ``(file_key, target_smell)`` pair is a
    present/absent pair in the n15 ground truth, else ``pd.NA`` (pair outside the
    n15 scope — dropped when scoring on this label)."""
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    present = frozenset(
        (e["file_key"], s["pascal"])
        for e in gt
        for s in e.get("present_smells", [])
    )
    absent = frozenset((e["file_key"], s["pascal"]) for e in gt for s in e.get("absent_smells", []))
    pairs = zip(df["file_key"], df["target_smell"])
    label = [True if p in present else False if p in absent else pd.NA for p in pairs]
    df = df.copy()
    df["gt_n15"] = pd.array(label, dtype="boolean")
    return df


def add_authoritative(df: pd.DataFrame, gt_path: Path = N15_GT) -> pd.DataFrame:
    """Compatibility alias for the validated n15 ground-truth label."""
    return add_n15_label(df, gt_path)


def metrics(truth: pd.Series, pred: pd.Series) -> dict:
    """Detection metrics for a boolean ``(truth, pred)`` pair, via sklearn.

    Positive class = smell present. ``confusion_matrix`` is pinned to
    ``labels=[False, True]`` so an absent class still yields a full 2x2; degenerate
    rates (``zero_division``) become NaN."""
    t = truth.astype(bool).to_numpy()
    p = pred.astype(bool).to_numpy()
    tn, fp, fn, tp = confusion_matrix(t, p, labels=[False, True]).ravel()
    pr, re, f1, _ = precision_recall_fscore_support(
        t,
        p,
        average="binary",
        pos_label=True,
        zero_division=np.nan,
    )
    return {"n": len(t), "precision": float(pr), "recall": float(re), "f1": float(f1),
            "accuracy": (int(tp) + int(tn)) / len(t),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


METRIC_COLUMNS = ["n", "precision", "recall", "f1", "accuracy", "TP", "FP", "FN", "TN"]


def metrics_by(
    df: pd.DataFrame, *keys: str, label: str = "gt_n15", repetition: int | None = 1
) -> pd.DataFrame:
    """Per-group detection metrics scored on ``label``, on one ``repetition`` (rep1 by
    default) or, with ``repetition=None``, on every run."""
    if label == "gt_n15" and "gt_n15" not in df.columns:
        df = add_n15_label(df)
    if "design" in df.columns:
        df = df[df["design"].isin(DESIGNS)].copy()
    if repetition is not None:
        df = df[df["repetition"] == repetition]
    sub = df.dropna(subset=[label, "smell_detected"])
    recs = []
    for key, group in sub.groupby(list(keys), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        recs.append(dict(zip(keys, key)) | metrics(group[label], group["smell_detected"]))
    return pd.DataFrame.from_records(recs, columns=[*keys, *METRIC_COLUMNS])


def accuracy_spread(df: pd.DataFrame, label: str = "gt_n15") -> pd.DataFrame:
    """Detection accuracy of each repetition, and its spread, per model and design.

    The spread is the maximum minus the minimum accuracy over the repetitions of one
    (model, design) pair: the stability measure of the paper's Step 6. Each repetition
    is scored on its own runs. Pairs run only once (Gemini 3.5 Flash and DeepSeek V4
    Flash) are left out.
    """
    by_rep = metrics_by(df, "model_alias", "design", "repetition", label=label, repetition=None)
    acc = by_rep.pivot(index=["model_alias", "design"], columns="repetition", values="accuracy")
    acc = acc[acc.notna().sum(axis=1) > 1]
    acc.columns = [f"rep{r}" for r in acc.columns]
    acc["spread"] = acc.max(axis=1) - acc.min(axis=1)
    return acc.reset_index()


def _pct(value: float, rounded: bool = True) -> str:
    if pd.isna(value):
        return "--"
    if rounded:
        return f"{round(value * 100):.0f}"
    return f"{value * 100:.1f}"


def _distinct_f1_ranks(rows: pd.DataFrame) -> tuple[float | None, float | None]:
    values = sorted(rows["f1"].dropna().unique(), reverse=True)
    best = values[0] if values else None
    second = values[1] if len(values) > 1 else None
    return best, second


def _format_f1(value: float, best: float | None, second: float | None, rounded: bool) -> str:
    cell = _pct(value, rounded=rounded)
    if pd.isna(value):
        return cell
    if best is not None and value == best:
        return rf"\textbf{{{cell}}}"
    if second is not None and value == second:
        return rf"\underline{{{cell}}}"
    return cell


def _smell_cells(smell: str, pivot: pd.DataFrame, rank_lookup: dict, rounded: bool) -> list[str]:
    """Metric cells of one smell row: F1, Pr, Re and Acc per model and design."""
    values = []
    for _, alias in TABLE_MODELS:
        best, second = rank_lookup.get((smell, alias), (None, None))
        for design in DESIGNS:
            key = (smell, alias, design)
            for metric, _ in METRICS:
                value = pivot.loc[key, metric] if key in pivot.index else np.nan
                if metric == "f1":
                    values.append(_format_f1(value, best, second, rounded))
                else:
                    values.append(_pct(value, rounded=rounded))
    return values


def build_smell_level_latex_table(
    df: pd.DataFrame,
    label: str = "tab:rq1_detection_grouped_by_model",
    caption: str | None = None,
    rounded: bool = True,
) -> str:
    """Render per-smell detection metrics by model and design.

    Each row is one target smell, grouped by smell type (``SMELL_TYPES``) under a
    rotated Type label. Each model contributes D1/D2 blocks with
    F1, precision, recall, and accuracy columns. F1 cells are ranked within each
    smell/model across designs: best distinct F1 is bold, second-best is
    underlined. Gray rows are the ``SHADED_SMELL_COUNT`` smells with the lowest mean
    F1. The Avg. row is the macro-average across smells of each model/design.
    """
    if caption is None:
        caption = (
            "Smell-level detection effectiveness by model and \\textsc{GHA-SmellFix} "
            "orchestration design. "
            "Best F1-score per smell/model is in \\textbf{bold}; second-best "
            "distinct F1-score is \\underline{underlined}. The Avg. row reports "
            "macro-averages across smells for each model/design pair. "
            "Gray rows highlight selected difficult smells discussed in the text."
        )

    if df.empty:
        raise ValueError("No result rows found for smell-level table")

    mt = metrics_by(df, "target_smell", "model_alias", "design")
    if mt.empty:
        raise ValueError("No scored rep1 rows found for smell-level table")

    mt = mt[mt["design"].isin(DESIGNS)].copy()
    mt["target_smell"] = pd.Categorical(mt["target_smell"], SMELL_ORDER, ordered=True)
    mt["model_alias"] = pd.Categorical(
        mt["model_alias"],
        [alias for _, alias in TABLE_MODELS],
        ordered=True,
    )
    mt["design"] = pd.Categorical(mt["design"], DESIGNS, ordered=True)
    mt = mt.sort_values(["target_smell", "model_alias", "design"])

    pivot = mt.set_index(["target_smell", "model_alias", "design"])
    shaded_smells = set(
        mt.groupby("target_smell", observed=False)["f1"]
        .mean()
        .sort_values()
        .head(SHADED_SMELL_COUNT)
        .index.astype(str)
    )
    rank_lookup = {}
    for (smell, model), rows in mt.groupby(["target_smell", "model_alias"], observed=False):
        rank_lookup[(smell, model)] = _distinct_f1_ranks(rows)

    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\tiny",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{1.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        "",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{",
        "@{}",
        r">{\centering\arraybackslash}p{0.65cm}",
        r">{\raggedright\arraybackslash}p{3.0cm}",
        r"!{\vrule width 0.35pt}",
        r"rrrr!{\vrule width 0.25pt}rrrr",
        r"!{\vrule width 0.35pt}",
        r"rrrr!{\vrule width 0.25pt}rrrr",
        r"!{\vrule width 0.35pt}",
        r"rrrr!{\vrule width 0.25pt}rrrr",
        r"!{\vrule width 0.35pt}",
        r"rrrr!{\vrule width 0.25pt}rrrr",
        "@{}",
        "}",
        r"\toprule",
        "",
        r"\multirow{3}{*}{\textbf{Type}}",
        r"& \multirow{3}{*}{\textbf{Smell}}",
        r"& \multicolumn{8}{!{\vrule width 0.35pt}c!{\vrule width 0.35pt}}{\textbf{GPT-5.4-nano}}",
        r"& \multicolumn{8}{c!{\vrule width 0.35pt}}{\textbf{Gemini 3.5 Flash}}",
        r"& \multicolumn{8}{c!{\vrule width 0.35pt}}{\textbf{Claude Haiku 4.5}}",
        r"& \multicolumn{8}{c}{\textbf{DeepSeek V4 Flash}} \\",
        "",
        r"\cline{3-34}",
        "",
        "&",
    ]

    model_design_headers = []
    for model_idx, _ in enumerate(TABLE_MODELS):
        for design_idx, design in enumerate(DESIGNS):
            left = "!{\\vrule width 0.35pt}" if model_idx == 0 and design_idx == 0 else ""
            right = (
                "!{\\vrule width 0.35pt}"
                if design_idx == len(DESIGNS) - 1 and model_idx < len(TABLE_MODELS) - 1
                else "!{\\vrule width 0.25pt}"
            )
            if model_idx == len(TABLE_MODELS) - 1 and design_idx == len(DESIGNS) - 1:
                right = ""
            model_design_headers.append(
                rf"& \multicolumn{{4}}{{{left}c{right}}}{{\textbf{{{design}}}}}"
            )
    lines.append("\n".join(model_design_headers) + r" \\")
    lines.extend(
        [
            "",
            r"\cline{3-34}",
            "",
            "&",
        ]
    )

    metric_headers = " & ".join(rf"\textbf{{{metric_label}}}" for _, metric_label in METRICS)
    lines.append("\n".join(f"& {metric_headers}" for _ in TABLE_MODELS for _ in DESIGNS) + r" \\")
    lines.append(r"\midrule")

    present = set(mt["target_smell"].dropna().astype(str))
    for smell_type, type_smells in SMELL_TYPES.items():
        smells = [smell for smell in type_smells if smell in present]
        if not smells:
            continue  # a type with no smell in the data gets no rows and no rule
        for i, smell in enumerate(smells):
            values = _smell_cells(smell, pivot, rank_lookup, rounded)
            lines.append("")
            if smell in shaded_smells:
                # The white Type cell keeps the gray fill off the group label.
                lines.extend([r"\rowcolor{gray!20}", r"\cellcolor{white}"])
            if i == len(smells) - 1:
                # The label sits on the group's last row and spans upward.
                lines.append(rf"\multirow{{-{len(smells)}}}{{*}}{{{_TYPE_LABELS[smell_type]}}}")
            lines.append(rf"& \texttt{{{smell}}} & {' & '.join(values)} \\")
        lines.extend(["", r"\midrule"])

    averages = mt.groupby(["model_alias", "design"], observed=False)[
        [metric for metric, _ in METRICS]
    ].mean()
    avg_blocks = []
    for _, alias in TABLE_MODELS:
        for design in DESIGNS:
            block = []
            for metric, _ in METRICS:
                key = (alias, design)
                value = averages.loc[key, metric] if key in averages.index else np.nan
                block.append(rf"\textbf{{{_pct(value, rounded=rounded)}}}")
            avg_blocks.append(f"& {' & '.join(block)}")
    lines.append(r"\multicolumn{2}{l!{\vrule width 0.35pt}}{\textbf{Avg.}}")
    lines.append("\n".join(avg_blocks) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"
