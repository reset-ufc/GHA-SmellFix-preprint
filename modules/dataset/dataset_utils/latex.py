"""LaTeX table generators for paper artifacts under data/tables/.

Each generator writes one .tex file whose LaTeX template matches the current
paper style verbatim — only the numeric values are substituted. Paper authors
can \\input these files directly from tables/.
"""

from __future__ import annotations

from pathlib import Path


def _fmt_int(n: int | None) -> str:
    """Format integer with thousands separator; '---' if None."""
    return f"{n:,}" if n is not None else "---"


def write_ground_truth_per_smell(coverage_table: list[dict], output_path: Path) -> None:
    """Emit ground_truth_per_smell.tex matching current paper layout.

    coverage_table items: {id, family, smell, latex_name, detected, selected}
    """
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Per-smell coverage of the ground_truth (behavioral smells).}",
        r"\label{tab:ground_truth_per_smell}",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{p{0.55cm} l r r}",
        r"\toprule",
        r"\textbf{ID} & \textbf{Smell} & \textbf{Detected} & \textbf{Selected} \\",
        r"\midrule",
        "",
    ]

    current_family: str | None = None
    for row in coverage_table:
        if row["family"] != current_family:
            if current_family is not None:
                lines.append(r"\midrule")
            current_family = row["family"]
            lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{current_family}}}}} \\")
            lines.append(r"\midrule")

        lines.append(
            f"{row['id']:>2} & {row['latex_name']}"
            f" & {_fmt_int(row['detected'])} & {row['selected']} \\\\"
        )

    lines.extend(
        [
            "",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Smell display names (pascal → human-readable, matching Khatami et al. SCAM 2024, Table III)
# ---------------------------------------------------------------------------

_SMELL_DISPLAY: dict[str, str] = {
    "MissingPermExternalActions": (
        "Define permissions for workflows with external actions"
    ),
    "UnpinnedActionVersion": (
        "Use commit hash instead of tags for action versions"
    ),
    "MissingTokenPermissions": "Set permissions for GitHub Token",
    "MissingForkGuardPR": "Prevent running issue/PR actions on forks",
    "MissingJobTimeout": "Avoid jobs without timeouts",
    "MissingConcurrencyPR": (
        "Stop running workflows when there is a newer commit in PR"
    ),
    "MissingConcurrencyBranch": (
        "Stop running workflows when there is a newer commit in branch"
    ),
    "MissingPathFilters": (
        "Avoid running CI actions when no source code has changed"
    ),
    "MissingForkGuardSchedule": "Avoid executing scheduled workflows on forks",
    "UnsafeArtifactUploadFork": "Avoid uploading artifacts on forks",
    "MissingUploadCondition": r"Use \texttt{if} for upload-artifact action",
    "UnsafeDeployOnFork": "Avoid deploying jobs on forks",
    "MissingConcurrencyGlobal": (
        "Avoid starting new workflow while previous one is running"
    ),
    "MutableRunnerVersion": r"Use fixed version for \texttt{runs-on} argument",
    "SuboptimalCacheUsage": "Use cache parameter instead of cache option",
    "MultiCommandStep": "Use single-command steps",
    "SingleOSExecution": "Run tests on multiple OS's",
    "UnpinnedDependencies": "Specify package versions",
    "SingleLangVersionCI": "Run CI on multiple language versions",
}


def write_irr_summary(report: dict, output_path: Path) -> None:
    """Emit irr_summary.tex: per-pair Kappa, concordance, and mean summary row."""
    pairs = report["pairs"]
    ov = report["overall"]
    mean_k = report.get("mean_kappa")
    std_k = report.get("std_kappa")
    mean_interp = report.get("mean_kappa_interpretation", "")

    def _k(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else r"\text{---}"

    def _c(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else r"\text{---}"

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-rater reliability per annotator pair.}",
        r"\label{tab:irr_summary}",
        r"\begin{tabular}{l r r l r}",
        r"\toprule",
        r"\textbf{Pair} & \textbf{N} & \textbf{$\kappa$}"
        r" & \textbf{Interpretation} & \textbf{Concordance (\%)} \\",
        r"\midrule",
    ]
    for p in pairs:
        pair_label = p["pair_id"].replace("-", "--")
        lines.append(
            f"{pair_label} & {p['n']} & {_k(p['kappa'])}"
            f" & {p['kappa_interpretation']} & {_c(p['concordance_pct'])} \\\\"
        )
    k_str = _k(mean_k)
    if std_k is not None:
        k_str = rf"{k_str} $\pm$ {std_k:.4f}"
    lines.extend(
        [
            r"\midrule",
            f"Mean & {ov['n']} & {k_str} & {mean_interp} & {_c(ov['concordance_pct'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ground_truth_accuracy(
    report: dict,
    output_path: Path,
    caption: str = "Annotation accuracy vs ground_truth ground truth, per smell.",
    label: str = "tab:ground_truth_accuracy",
) -> None:
    """Emit ground_truth_accuracy.tex: per-smell annotation accuracy vs ground_truth ground truth."""
    per_smell = report["per_smell"]
    ov = report["overall"]

    def _pct(v: float | None) -> str:
        return f"{v:.2f}" if v is not None else r"\text{---}"

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"\textbf{Smell} & \textbf{N} & \textbf{Accuracy (\%)}"
        r" & \textbf{Present (\%)} & \textbf{Absent (\%)}"
        r" & \textbf{Recall (\%)} & \textbf{F1 (\%)} \\",
        r"\midrule",
    ]
    sorted_per_smell = sorted(
        per_smell,
        key=lambda s: s["accuracy"] if s["accuracy"] is not None else -1,
        reverse=True,
    )
    for s in sorted_per_smell:
        display = _SMELL_DISPLAY.get(s["smell"], s["smell"])
        lines.append(
            f"{display} & {s['n']} & {_pct(s['accuracy'])}"
            f" & {_pct(s['present_accuracy'])} & {_pct(s['absent_accuracy'])}"
            f" & {_pct(s.get('recall'))} & {_pct(s.get('f1'))} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Overall}} & {ov['n']} & {_pct(ov['accuracy'])}"
            rf" & {_pct(ov['present_accuracy'])} & {_pct(ov['absent_accuracy'])}"
            rf" & {_pct(ov.get('recall'))} & {_pct(ov.get('f1'))} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_irr_confusion_overall(report: dict, output_path: Path) -> None:
    """Emit irr_confusion_overall.tex: overall annotator A-vs-B 2x2 confusion matrix."""
    cm = report["overall"]["confusion_matrix"]
    pp, pa, ap, aa = cm["pp"], cm["pa"], cm["ap"], cm["aa"]
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Overall annotator agreement confusion matrix (A vs.\ B).}",
        r"\label{tab:irr_confusion_overall}",
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r" & \textbf{B: Present} & \textbf{B: Absent} \\",
        r"\midrule",
        rf"\textbf{{A: Present}} & {pp:,} & {pa:,} \\",
        rf"\textbf{{A: Absent}} & {ap:,} & {aa:,} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_distribution(report: dict, output_path: Path) -> None:
    """Emit label_distribution.tex: per-smell final_decision present/absent counts."""
    per_smell = report["per_smell"]

    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Distribution of human annotation labels per smell.}",
        r"\label{tab:label_distribution}",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabular}{l r r r}",
        r"\toprule",
        r"\textbf{Smell} & \textbf{Total} & \textbf{Present} & \textbf{Absent} \\",
        r"\midrule",
    ]

    def _counts(s: dict) -> tuple[int, int]:
        cm = s["confusion_matrix"]
        return cm["tp"] + cm["fp"], cm["tn"] + cm["fn"]

    sorted_smells = sorted(
        per_smell,
        key=lambda s: abs(_counts(s)[0] - _counts(s)[1]),
        reverse=False,
    )

    total_present = total_absent = 0
    for s in sorted_smells:
        cm = s["confusion_matrix"]
        human_present = cm["tp"] + cm["fp"]
        human_absent = cm["tn"] + cm["fn"]
        total_present += human_present
        total_absent += human_absent
        display = _SMELL_DISPLAY.get(s["smell"], s["smell"])
        lines.append(f"{display} & {s['n']} & {human_present} & {human_absent} \\\\")
    lines.extend(
        [
            r"\midrule",
            rf"\textbf{{Total}} & {total_present + total_absent}"
            rf" & {total_present} & {total_absent} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_accuracy_table(
    report: dict,
    output_path: Path,
    caption: str = "Evaluation of automated detection of GHA smells.",
    label: str = "tab:evaluation_accuracy",
) -> None:
    """Emit evaluation_accuracy.tex: per-smell TP, TN, FP, FN, Accuracy,
    Precision, Recall, F1-score, Present%, Absent%.

    Columns use the **analyzer-as-prediction** orientation:
      TP = ground_truth present & human present  (code ``tp``)
      TN = ground_truth absent  & human absent   (code ``tn``)
      FP = ground_truth present & human absent   (code ``fn``)
      FN = ground_truth absent  & human present  (code ``fp``)

    Footer: **Total** for count columns, **Median** for percentage columns.
    """
    import statistics

    per_smell = report["per_smell"]

    def _pct(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else r"\text{---}"

    def _compute_row(s: dict) -> dict:
        cm = s["confusion_matrix"]
        # Remap to analyzer-as-prediction orientation
        tp = cm["tp"]
        tn = cm["tn"]
        fp = cm["fn"]  # ground_truth present, human absent → analyzer FP
        fn = cm["fp"]  # ground_truth absent, human present → analyzer FN
        n = tp + tn + fp + fn
        accuracy = round((tp + tn) / n * 100, 1) if n > 0 else None
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else None
        recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = round(
                2 * (precision / 100) * (recall / 100)
                / ((precision / 100) + (recall / 100))
                * 100, 1,
            )
        else:
            f1 = None
        present_pct = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else None
        absent_pct = round(tn / (tn + fn) * 100, 1) if (tn + fn) > 0 else None
        return {
            "smell": s["smell"],
            "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "present_pct": present_pct, "absent_pct": absent_pct,
        }

    rows = [_compute_row(s) for s in per_smell]
    rows.sort(key=lambda r: r["f1"] if r["f1"] is not None else -1, reverse=True)

    # --- Totals (counts) & Medians (percentages) ---
    tot_n = sum(r["n"] for r in rows)
    tot_tp = sum(r["tp"] for r in rows)
    tot_tn = sum(r["tn"] for r in rows)
    tot_fp = sum(r["fp"] for r in rows)
    tot_fn = sum(r["fn"] for r in rows)

    def _median(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(vals), 1) if vals else None

    med_acc = _median("accuracy")
    med_prec = _median("precision")
    med_rec = _median("recall")
    med_f1 = _median("f1")
    med_present = _median("present_pct")
    med_absent = _median("absent_pct")

    # --- LaTeX output ---
    lines: list[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{l r r r r r r r r r r r}",
        r"\toprule",
        (
            r"\textbf{Smell} & \textbf{N} & \textbf{TP} & \textbf{TN}"
            r" & \textbf{FP} & \textbf{FN} & \textbf{Acc.\,(\%)}"
            r" & \textbf{Prec.\,(\%)} & \textbf{Rec.\,(\%)}"
            r" & \textbf{F1\,(\%)} & \textbf{Pres.\,(\%)}"
            r" & \textbf{Abs.\,(\%)} \\"
        ),
        r"\midrule",
    ]

    for r in rows:
        display = _SMELL_DISPLAY.get(r["smell"], r["smell"])
        lines.append(
            f"{display} & {r['n']} & {r['tp']} & {r['tn']}"
            f" & {r['fp']} & {r['fn']} & {_pct(r['accuracy'])}"
            f" & {_pct(r['precision'])} & {_pct(r['recall'])}"
            f" & {_pct(r['f1'])} & {_pct(r['present_pct'])}"
            f" & {_pct(r['absent_pct'])} \\\\"
        )

    lines.extend([
        r"\midrule",
        (
            rf"\textbf{{Total}} & {tot_n} & {tot_tp} & {tot_tn}"
            rf" & {tot_fp} & {tot_fn}"
            rf" & \multicolumn{{6}}{{c}}{{}} \\"
        ),
        (
            rf"\textbf{{Median}} & & & & & & {_pct(med_acc)}"
            rf" & {_pct(med_prec)} & {_pct(med_rec)}"
            rf" & {_pct(med_f1)} & {_pct(med_present)}"
            rf" & {_pct(med_absent)} \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Smell type (Step 2 of the paper's method) -> its smells. The coverage table groups
# its rows by type, in paper-id order within each type.
_SMELL_TYPES: dict[str, frozenset[str]] = {
    "Localized": frozenset(
        {
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
        }
    ),
    "Context-dependent": frozenset(
        {
            "MissingForkGuardPR",
            "MissingConcurrencyPR",
            "MissingConcurrencyBranch",
            "MissingPathFilters",
            "MissingForkGuardSchedule",
            "UnsafeDeployOnFork",
            "MissingConcurrencyGlobal",
            "SingleOSExecution",
            "SingleLangVersionCI",
        }
    ),
}
# Rotated label of each type, as the paper writes it (``\groupcell`` is defined in
# the paper's preamble).
_TYPE_LABELS: dict[str, str] = {
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


def _smells_by_type(listed: list[str], seen: set[str]) -> list[tuple[str, list[str]]]:
    """Split ``listed`` by smell type, keeping its order; types with no smell are dropped.

    ``seen`` is every smell of the ground truth. Any of them without a type raises, so no
    count reaches the Total row without a row of its own.
    """
    untyped = seen.difference(*_SMELL_TYPES.values())
    if untyped:
        raise ValueError(f"smells without a type in _SMELL_TYPES: {sorted(untyped)}")
    groups = [(t, [p for p in listed if p in members]) for t, members in _SMELL_TYPES.items()]
    return [(t, smells) for t, smells in groups if smells]


def write_ground_truth_validated_coverage(
    ground_truth: list[dict],
    output_path: Path,
    caption: str = "Overview of the ground truth coverage.",
    label: str = "tab:ground_truth_validated_coverage",
) -> None:
    """Emit ground_truth_validated_coverage.tex: per-smell Pos/Neg instance counts and
    LOC range over the human-validated ground_truth, grouped by smell type.

    ``ground_truth``: list of entries in the Step 12/12b schema — each
    ``{file_line_count, present_smells: [{pascal, ...}], absent_smells: [{pascal}]}``.

    Per smell:
      Positive     = files where the smell is a confirmed-present instance
      Negative     = files where the smell is a confirmed-absent instance
      Total        = Positive + Negative
      LOC_min/max  = file_line_count range over files referencing the smell
                     (either polarity)

    Rows are grouped by smell type (Localized / Context-dependent) under a rotated
    label and ordered by paper id within each type. Only smells that appear in the
    ground_truth are listed. Footer row totals Positive, Negative and Total.
    """
    from dataset_utils.ground_truth_sampling import SMELL_ORDER

    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    loc_min: dict[str, int] = {}
    loc_max: dict[str, int] = {}

    def _track(pascal: str, loc: int) -> None:
        if loc <= 0:
            return
        loc_min[pascal] = loc if pascal not in loc_min else min(loc_min[pascal], loc)
        loc_max[pascal] = loc if pascal not in loc_max else max(loc_max[pascal], loc)

    for entry in ground_truth:
        loc = entry.get("file_line_count") or 0
        for s in entry.get("present_smells", []):
            pos[s["pascal"]] = pos.get(s["pascal"], 0) + 1
            _track(s["pascal"], loc)
        for s in entry.get("absent_smells", []):
            neg[s["pascal"]] = neg.get(s["pascal"], 0) + 1
            _track(s["pascal"], loc)

    # Behavioral smells present in the ground_truth, in paper order.
    listed = [p for p in SMELL_ORDER if p in pos or p in neg]
    name_w = max((len(p) for p in listed), default=0)

    def _loc(pascal: str, which: dict) -> str:
        return str(which[pascal]) if pascal in which else r"--"

    # Positive, Negative, Total, LOC_min, LOC_max; each column is left-aligned to its
    # widest value.
    values = {
        p: [
            str(pos.get(p, 0)),
            str(neg.get(p, 0)),
            str(pos.get(p, 0) + neg.get(p, 0)),
            _loc(p, loc_min),
            _loc(p, loc_max),
        ]
        for p in listed
    }
    widths = [max((len(values[p][i]) for p in listed), default=0) for i in range(5)]

    lines: list[str] = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\tiny",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"\textbf{Type} &",
        r"\textbf{Smell} &",
        r"\textbf{Positive} &",
        r"\textbf{Negative} &",
        r"\textbf{Total} &",
        r"\textbf{LOC$_{\min}$} &",
        r"\textbf{LOC$_{\max}$} \\",
        r"\midrule",
    ]

    for i, (smell_type, smells) in enumerate(_smells_by_type(listed, set(pos) | set(neg))):
        if i:
            lines.extend(["", r"\midrule"])
        lines.extend(["", rf"\multirow{{{len(smells)}}}{{*}}{{{_TYPE_LABELS[smell_type]}}}"])
        for pascal in smells:
            pad = " " * (name_w - len(pascal))
            cells = " & ".join(v.ljust(w) for v, w in zip(values[pascal], widths, strict=True))
            lines.append(f"& \\texttt{{{pascal}}}{pad} & {cells} \\\\")

    tot_pos = sum(pos.values())
    tot_neg = sum(neg.values())
    # Total row sums Positive/Negative/Total only; LOC min/max are per-smell, not
    # meaningful to aggregate.
    lines.extend(
        [
            "",
            r"\midrule",
            r"\multicolumn{2}{l}{\textbf{Total}} &",
            rf"\textbf{{{tot_pos:,}}} &",
            rf"\textbf{{{tot_neg:,}}} &",
            rf"\textbf{{{tot_pos + tot_neg:,}}} &",
            r"-- &",
            r"-- \\",
            "",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ground_truth_summary(report: dict, output_path: Path) -> None:
    """Emit ground_truth_summary.tex matching current paper layout.

    report: ground_truth_sample_report.json contents (top_n_per_smell,
            total_ground_truth_entries, unique_smells_targeted,
            smells_with_full_coverage / partial / none,
            presentational_smells).
    """
    presentational_count = len(report.get("presentational_smells", []))
    n_per_polarity = report["n_per_polarity"]
    # Full-coverage counts are computed against the delivered (oversampled) pool,
    # not the per-polarity target, so the label must use the delivered threshold.
    n_delivered = report.get("n_delivered_per_polarity", n_per_polarity)
    lines = [
        r"\begin{table}[H]",
        r"\small",
        r"\centering",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\begin{tabularx}{\linewidth}{X|p{2cm}}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Value} \\",
        r"\hline",
        rf"N per polarity (positives, negatives) & {n_per_polarity} \\",
        r"\hline",
        rf"Total unique workflow files & {report['total_ground_truth_entries']} \\",
        r"\hline",
        rf"Behavioral smells targeted & {report['unique_smells_targeted']} \\",
        r"\hline",
        rf"Presentational smells (recorded, not evaluated) & {presentational_count} \\",
        r"\hline",
        rf"Smells with full positive coverage ($\geq{n_delivered}$)"
        rf" & {report['smells_with_full_pos']} \\",
        r"\hline",
        rf"Smells with full negative coverage ($\geq{n_delivered}$)"
        rf" & {report['smells_with_full_neg']} \\",
        r"\hline",
        r"\end{tabularx}",
        r"\end{adjustbox}",
        r"\caption{Ground truth Sample Report Summary}",
        r"\label{tab:ground_truth_summary}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
