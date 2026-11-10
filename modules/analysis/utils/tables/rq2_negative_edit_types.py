"""Data-driven LaTeX table for RQ2 unnecessary-edit types on negatives.

We inspect ground-truth-negative pairs for which the pipeline *modified* the
workflow despite the target smell being absent. Since abstention is the only
correct response, **any** modification is treated as unnecessary, and each
modified negative output is classified by its HIGHEST-RISK edit (precedence:
behaviour-relevant > localized > cosmetic).

Scope (matches every other RQ table): ``rep == 1``, the four table models
(Gemma excluded), ``gt_positive == False``.

Denominator -- "unnecessary edits" per design
----------------------------------------------
The denominator is ``resolved_verdict == "unnecessary"`` -> **93** negatives
(D1/D2 = 38/55). This matches the NMR numerator of the effectiveness table
cell-for-cell, so the two tables report the *same* unnecessary-edit population --
no denominator gap for a reviewer to question.

The validation queue lists inspected negatives across all designs. We keep
only the D1/D2 confirmed unnecessary edits: ``uncertain`` is
null (excluded), and ``necessary`` / ``preserved`` are not unnecessary edits.
Every unnecessary patch carries construct occurrences and none is cosmetic, so the
two edit-type cells sum to the column total; the generator fails otherwise.

Edit-type classification (from ``construct_per_patch.csv``)
----------------------------------------------------------
The behaviour CSVs (``behavior_*_rep1.csv``) carry only a binary
expected/unexpected-modification outcome, and ``manual_validation_final.csv``
carries only a binary ``resolved_potential_behavior_impact`` -- neither is the 3-way
label. We derive it from the per-patch construct occurrences, mapping each occurrence to a risk
tier and taking the maximum tier per output:

* **Behaviour-relevant** (tier 3): any construct that affects triggers, runners
  (``runs-on`` / ``container``), dependencies / concurrency (``job
  orchestration`` other than timeouts), commands, reused Actions / workflows,
  matrices, environment variables, contexts, or step structure -- i.e. every
  catalogued feature except ``naming``, and every uncatalogued (``other``)
  structural edit.
* **Localized but unnecessary** (tier 2): plausible smell-fix-shaped edits. In
  the negative data this is the ``jobs.<job_id>.timeout-minutes`` construct
  (added or changed). The other localized exemplars in the prose -- ``permissions``
  and step-condition (``step orchestration``) edits -- have zero occurrences on
  negatives.
* **Cosmetic** (tier 1): naming changes (``jobs.<job_id>.steps[*].name``);
  formatting / comment-only edits would carry no construct at all. No modified
  negative output bottoms out at cosmetic (every one also touches a higher tier
  or a behaviour construct). The paper's table therefore has no Cosmetic row, and
  its text says that no cosmetic edit was observed.

Each cell = ``count / unnecessary-edits`` for that column.

The output is only the paper's ``table`` environment, verbatim.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.designs import DESIGNS

try:
    from utils import dataloader
except ImportError:  # notebooks run with utils/ on the path
    import utils.dataloader as dataloader


ROOT = dataloader.ROOT
TABLE_MODELS = ["gpt54-nano", "gemini35-flash", "haiku45", "deepseek-v4-flash"]
MODEL_DISPLAY = {
    "gpt54-nano": "GPT-5.4-nano",
    "gemini35-flash": "Gemini 3.5 Flash",
    "haiku45": "Claude Haiku 4.5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
}
PATCH_KEYS = ["file_key", "smell", "design", "model"]

CONSTRUCT_INDEX = ROOT / "data" / "artifacts" / "constructs" / "construct_per_patch_index.csv"
CONSTRUCT_PER_PATCH = ROOT / "data" / "artifacts" / "constructs" / "construct_per_patch.csv"
MANUAL_VALIDATION = ROOT / "data" / "validation" / "manual_validation_final.csv"

# Edit-type rows, in display order, paired with the construct tier they read. The
# cosmetic tier (1) has no row: the paper omits it because it has no cases.
EDIT_TYPES = [
    ("Localized but unnecessary edits", 2),
    ("Behavior-relevant edits", 3),
]


def _construct_tier(feature: str, construct: str) -> int:
    """Risk tier of one construct occurrence (3 = highest)."""
    if feature == "naming":
        return 1
    if "timeout-minutes" in str(construct):
        return 2
    return 3


def _pct_count(num: int, den: int) -> str:
    if den == 0:
        return "--"
    return rf"{num / den * 100:.2f}\%\,{{\scriptsize({num}/{den})}}"


def _scope(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["rep"] == 1)
        & (df["model"].isin(TABLE_MODELS))
        & (df["design"].isin(DESIGNS))
        & (df["gt_positive"] == False)  # noqa: E712 - pandas boolean mask
    ].copy()


def _compute() -> tuple[dict[tuple[str, str, int], int], dict[tuple[str, str], int]]:
    """Return (counts[(model, design, tier)], denom[(model, design)]).

    Both keyed per (model, design); the tier counts of one (model, design) must
    sum to its denominator (every unnecessary negative carries a construct and
    lands in exactly one row via its highest-risk occurrence). A patch without a
    construct, or a cosmetic-only one, would have no row, so it raises.

    Denominator = ``resolved_verdict == "unnecessary"`` patches, so it
    equals the NMR numerator of the effectiveness table cell-for-cell.
    """
    index = _scope(pd.read_csv(CONSTRUCT_INDEX))
    verdicts = pd.read_csv(MANUAL_VALIDATION)[["stem", "resolved_verdict"]]
    index = index.merge(verdicts, on="stem", how="left")
    unnec = index[index["resolved_verdict"] == "unnecessary"][PATCH_KEYS].drop_duplicates()
    denom = {
        (m, design): int(((unnec["model"] == m) & (unnec["design"] == design)).sum())
        for m in TABLE_MODELS
        for design in DESIGNS
    }

    occ = _scope(pd.read_csv(CONSTRUCT_PER_PATCH))
    # Keep only occurrences belonging to an unnecessary patch.
    occ = occ.merge(unnec, on=PATCH_KEYS, how="inner")
    occ["tier"] = [
        _construct_tier(f, c) for f, c in zip(occ["feature"], occ["construct"])
    ]
    per_patch = occ.groupby(PATCH_KEYS)["tier"].max().reset_index()

    counts: dict[tuple[str, str, int], int] = {}
    for m in TABLE_MODELS:
        for design in DESIGNS:
            sub = per_patch[(per_patch["model"] == m) & (per_patch["design"] == design)]
            for _, tier in EDIT_TYPES:
                counts[(m, design, tier)] = int((sub["tier"] == tier).sum())
            in_rows = sum(counts[(m, design, tier)] for _, tier in EDIT_TYPES)
            if in_rows != denom[(m, design)]:
                raise ValueError(
                    f"{m} {design}: {denom[(m, design)]} unnecessary edits but {in_rows} "
                    "in the table rows (a cosmetic-only or construct-free patch has no row)"
                )
    return counts, denom


def _build_latex() -> str:
    counts, denom = _compute()
    den_model = {m: sum(denom[(m, design)] for design in DESIGNS) for m in TABLE_MODELS}
    den_design = {design: sum(denom[(m, design)] for m in TABLE_MODELS) for design in DESIGNS}
    overall_den = sum(den_model.values())

    def model_line(model: str, tier: int) -> str:
        cells = [
            _pct_count(counts[(model, design, tier)], denom[(model, design)])
            for design in DESIGNS
        ]
        overall_num = sum(counts[(model, design, tier)] for design in DESIGNS)
        cells.append(_pct_count(overall_num, den_model[model]))
        body = "\n& ".join(cells)
        return f"& {MODEL_DISPLAY[model]}\n& {body} \\\\"

    def group(label: str, tier: int) -> str:
        head = rf"\multirow{{4}}{{*}}{{{label}}}"
        lines = "\n".join(model_line(m, tier) for m in TABLE_MODELS)
        return f"{head}\n{lines}"

    rows = "\n\\midrule\n".join(group(label, tier) for label, tier in EDIT_TYPES)

    total_cells = [rf"\textbf{{{den_design[design]}}}" for design in DESIGNS]
    total_cells.append(rf"\textbf{{{overall_den}}}")
    totals = " & ".join(total_cells)

    header = r"""
\begin{table}[ht!]
\centering
\scriptsize
\caption{Types of unnecessary edits in modified ground-truth-negative pairs by design and model.}
\label{tab:rq2_negative_edit_types}
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.06}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{llrrr}
\toprule
\textbf{Edit type} & \textbf{Model}
& \textbf{D1}
& \textbf{D2}
& \textbf{Overall} \\
\midrule
"""
    footer = (
        "\n\\midrule\n\\textbf{Unnecessary edits} & \\textbf{All models}\n& "
        + totals
        + " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n"
    )
    return (header + rows + footer).strip() + "\n"


def negative_edit_types_table(
    output_path: Path | str = Path("data/artifacts") / "rq2_negative_edit_types.tex",
) -> str:
    """Write and return the RQ2 NEGATIVE EDIT TYPES LaTeX table.

    Cells are recomputed from the construct-patch index (denominator: modified
    negative outputs per design) and per-patch construct occurrences (highest
    risk tier per output). See the module docstring for num/den definitions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(negative_edit_types_table())
