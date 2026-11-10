"""Data-driven LaTeX table for RQ2 refactoring effectiveness and safety.

Every cell is recomputed from the construct-patch index and the resolved
manual-validation verdicts (rep1, the four table models; Gemma excluded).

No smell is excluded: all 19 smells count on both sides (285 positive + 285
negative pairs). SRR and NMR are reported over the **full ground-truth pool** (a
flat 285 positive / 285 negative pairs): an ``uncertain`` verdict is simply not
counted in the numerator (not a confirmed removal / not a confirmed unnecessary
edit), so the denominator stays 285 -- no ragged ``285 - uncertain``. SVR and CCR
stay **conditional** rates whose denominators are the produced / valid-removing
patches (not 285). Denominators:

- **Pos. Patches** -- produced positive patches over all 285 ground-truth
  positive pairs; raw production/detection yield (uncertain still counts: a patch
  was produced).
- **SRR** -- ``resolved_verdict == "removed"`` over the full **285** positive
  pairs (end-to-end; non-produced and uncertain pairs count as not removed).
- **SVR** -- ``n_invalid_added == 0`` over produced positive patches.
- **CCR** -- ``resolved_construct_ok == "yes"`` over **valid smell-removing**
  patches (``svr_pass`` AND ``resolved_verdict == "removed"``), excluding patches
  whose construct verdict is uncertain.
- **NMR** -- ``resolved_verdict == "unnecessary"`` over the full **285** negative
  pairs (uncertain and ``necessary`` edits are not counted as unnecessary; the
  numerators sum to the total of the negative-edit-types table).

Avg. rows are the macro mean of the per-model rates (Pos. Patches is the mean
produced count, rounded). SVR/CCR remain conditional on a patch being produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.designs import DESIGNS

try:
    from utils import dataloader
except ImportError:  # notebooks run with utils/ on the path
    import utils.dataloader as dataloader


ROOT = dataloader.ROOT
TABLE_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]

CONSTRUCT_INDEX = ROOT / "data" / "artifacts" / "constructs" / "construct_per_patch_index.csv"
MANUAL_VALIDATION = ROOT / "data" / "validation" / "manual_validation_final.csv"
GROUND_TRUTH = ROOT / "data" / "ground_truth_n15.json"


def _pct_count(num: int, den: int) -> str:
    if den == 0:
        return "--"
    return rf"{num / den * 100:.2f}\%\,{{\scriptsize({num}/{den})}}"


def _pct(value: float) -> str:
    if pd.isna(value):
        return "--"
    return rf"{value * 100:.2f}\%"


def _gt_pair_counts() -> tuple[int, int]:
    """All present and all absent pairs in the n15 ground truth (285 / 285)."""
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    positive = sum(len(e.get("present_smells", [])) for e in gt)
    negative = sum(len(e.get("absent_smells", [])) for e in gt)
    return positive, negative


def compute_metrics() -> pd.DataFrame:
    """Per (design, model) effectiveness/safety metrics from the latest data."""
    aliases = [alias for _, alias in TABLE_MODELS]

    index = pd.read_csv(CONSTRUCT_INDEX)
    index = index[index["rep"].eq(1) & index["model"].isin(aliases)].copy()
    index["svr_pass"] = index["n_invalid_added"].fillna(0).eq(0)

    manual = pd.read_csv(MANUAL_VALIDATION)
    manual = manual[manual["rep"].eq(1) & manual["model"].isin(aliases)].copy()

    positive_pairs, negative_pairs = _gt_pair_counts()

    rows = []
    for design in DESIGNS:
        for display_model, model in TABLE_MODELS:
            produced = index[
                index["design"].eq(design)
                & index["model"].eq(model)
                & index["gt_positive"].eq(True)
            ]
            manual_pos = manual[
                manual["design"].eq(design)
                & manual["model"].eq(model)
                & manual["gt_positive"].eq(True)
            ]
            manual_neg = manual[
                manual["design"].eq(design)
                & manual["model"].eq(model)
                & manual["gt_positive"].eq(False)
            ]
            produced_patches = len(produced)
            if produced_patches != len(manual_pos):
                raise ValueError(
                    f"patch/validation row mismatch for {design}/{model}: "
                    f"{produced_patches} produced vs {len(manual_pos)} validated"
                )
            # CCR is over VALID SMELL-REMOVING patches (svr_pass AND removed), per
            # the paper text -- join schema validity to the resolved verdicts.
            # ``uncertain`` is null: excluded from every verdict-based denominator
            # (SRR / CCR / NMR). SVR (schema) and Pos. Patches (production) keep /285.
            joined_pos = produced[["stem", "svr_pass"]].merge(
                manual_pos[["stem", "resolved_verdict", "resolved_construct_ok"]],
                on="stem",
                how="left",
            )
            valid_removing = joined_pos[
                joined_pos["svr_pass"] & joined_pos["resolved_verdict"].eq("removed")
            ]
            ccr_scored = valid_removing[
                valid_removing["resolved_construct_ok"].ne("uncertain")
            ]
            # SRR/NMR denominators are the full GT pool (flat 285): uncertain folds
            # into the pool as a non-removal / non-unnecessary rather than dropping
            # out, so the denominator never reads as a ragged ``285 - uncertain``.
            rows.append(
                {
                    "design": design,
                    "display_model": display_model,
                    "produced": produced_patches,
                    "positive_pairs": positive_pairs,
                    "SRR_num": int(manual_pos["resolved_verdict"].eq("removed").sum()),
                    "SRR_den": positive_pairs,
                    "SVR_num": int(produced["svr_pass"].sum()),
                    "CCR_num": int(ccr_scored["resolved_construct_ok"].eq("yes").sum()),
                    "CCR_den": int(len(ccr_scored)),
                    "negative_pairs": negative_pairs,
                    "NMR_num": int(manual_neg["resolved_verdict"].eq("unnecessary").sum()),
                    "NMR_den": negative_pairs,
                }
            )

    metrics = pd.DataFrame(rows)
    metrics["SRR"] = metrics["SRR_num"] / metrics["SRR_den"]
    metrics["SVR"] = metrics["SVR_num"] / metrics["produced"]
    metrics["CCR"] = metrics["CCR_num"] / metrics["CCR_den"]
    metrics["NMR"] = metrics["NMR_num"] / metrics["NMR_den"]
    return metrics


def _build_latex(metrics: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Refactoring effectiveness and safety by design and model.}",
        r"\label{tab:rq2_refactoring_effectiveness_by_design_and_model}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Design}}",
        r"& \multirow{2}{*}{\textbf{Model}}",
        r"& \multicolumn{4}{c}{\textbf{Ground-truth-positive pairs}}",
        r"& \multicolumn{1}{c}{\textbf{Ground-truth-negative pairs}} \\",
        r"\cmidrule(lr){3-6}",
        r"\cmidrule(lr){7-7}",
        r"&",
        r"& \textbf{Pos. Patches}",
        r"& \textbf{SRR$\uparrow$}",
        r"& \textbf{SVR$\uparrow$}",
        r"& \textbf{CCR$\uparrow$}",
        r"& \textbf{NMR$\downarrow$} \\",
        r"\midrule",
    ]
    for design in DESIGNS:
        sub = metrics[metrics["design"].eq(design)]
        lines.append(rf"\multirow{{5}}{{*}}{{\textbf{{{design}}}}}")
        for row in sub.itertuples(index=False):
            lines.append(
                rf"& {row.display_model} & {row.produced}/{row.positive_pairs} & "
                rf"{_pct_count(row.SRR_num, row.SRR_den)} & "
                rf"{_pct_count(row.SVR_num, row.produced)} & "
                rf"{_pct_count(row.CCR_num, row.CCR_den)} & "
                rf"{_pct_count(row.NMR_num, row.NMR_den)} \\"
            )
        avg = sub[["SRR", "SVR", "CCR", "NMR"]].mean()
        avg_produced = int(round(sub["produced"].mean()))
        pairs = int(sub["positive_pairs"].iloc[0])
        lines.append(
            rf"& \textbf{{Avg.}} & \textbf{{{avg_produced}/{pairs}}} & "
            rf"\textbf{{{_pct(avg.SRR)}}} & \textbf{{{_pct(avg.SVR)}}} & "
            rf"\textbf{{{_pct(avg.CCR)}}} & \textbf{{{_pct(avg.NMR)}}} \\"
        )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def refactoring_effectiveness_by_design_and_model_table(
    output_path: Path | str = (
        Path("data/artifacts") / "rq2_refactoring_effectiveness_by_design_and_model.tex"
    ),
) -> str:
    """Compute, write, and return the RQ2 refactoring effectiveness table."""
    latex = _build_latex(compute_metrics())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex


refactoring_effectiveness_safety_table = (
    refactoring_effectiveness_by_design_and_model_table
)


if __name__ == "__main__":
    print(refactoring_effectiveness_by_design_and_model_table())
