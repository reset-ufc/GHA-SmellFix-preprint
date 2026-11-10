"""LaTeX table for rq2 detection misses and clean removal by model and design.

The per-model breakdown of the end-to-end vs. conditional table
(``rq2_end_to_end_conditional``), whose scored runs and counts are reused here, so
the scope, unit and outcome rule are the same: rep1, D1/D2, and every assessable
ground-truth-positive run counted on its own.

One row per model and an All models row, in bold, that pools the four:
  * Detection misses = Failed or No Patch runs whose detection verdict was absent,
                       under D1 and under D2.
  * End-to-end CR    = Clean Removal over every assessable positive run.
  * Conditional CR   = Clean Removal over the assessable positive runs whose
                       structured detection verdict was ``present``.
  * Delta            = D2 - D1 in percentage points, from unrounded rates.

The All models row equals the D1 and D2 rows of the end-to-end vs. conditional
table.

The output is only the paper's ``table`` environment.
"""

from pathlib import Path

import pandas as pd

from utils.designs import DESIGNS
from utils.tables.rq2_end_to_end_conditional import _counts, _scored_positive_runs

# Display name + alias, fixed row order (as in the RQ3 paired-tests table).
_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]

_HEADER = r"""
\begin{table}[ht!]
\centering
\caption{Detection misses and Clean Removal (CR) under D1 and D2, by model. $\Delta$ = D2~$-$~D1 in percentage points, computed from unrounded values.}
\label{tab:rq2_clean_removal_by_model}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{\textbf{Detection misses}} & \multicolumn{3}{c}{\textbf{End-to-end CR}} & \multicolumn{3}{c}{\textbf{Conditional CR}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}
\textbf{Model} & D1 & D2 & D1 & D2 & $\Delta$ & D1 & D2 & $\Delta$ \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}"""


def _rate(c: dict) -> float:
    """Clean Removal as a percentage of the frame's assessable outputs."""
    return 100.0 * c["cr"] / c["outputs"]


def _build_row(label: str, runs: pd.DataFrame, bold: bool = False) -> str:
    """One model's line: misses, then end-to-end and conditional CR with their deltas."""
    end_to_end, conditional = {}, {}
    for design in DESIGNS:
        sub = runs[runs["design"] == design]
        end_to_end[design] = _counts(sub)
        conditional[design] = _counts(sub[sub["smell_detected"].astype(bool)])
    cells = [label]
    cells += [f"{end_to_end[design]['misses']:,}" for design in DESIGNS]
    for block in (end_to_end, conditional):
        cells += [f"{_rate(block[design]):.1f}\\%" for design in DESIGNS]
        cells.append(f"{_rate(block['D2']) - _rate(block['D1']):+.1f}")
    if bold:
        cells = [rf"\textbf{{{cell}}}" for cell in cells]
    return " & ".join(cells) + r" \\"


def _build_latex() -> str:
    """Assemble the full LaTeX table from the scored runs."""
    pos = _scored_positive_runs()
    aliases = [alias for _, alias in _MODELS]
    if sorted(pos["model"].unique()) != sorted(aliases):
        raise ValueError(f"expected models {aliases}, got {sorted(pos['model'].unique())}")
    rows = [_build_row(name, pos[pos["model"] == alias]) for name, alias in _MODELS]
    total = _build_row("All models", pos, bold=True)
    body = "\n".join([_HEADER, *rows, r"\midrule", total, _FOOTER])
    return body.strip() + "\n"


def clean_removal_by_model_table(
    output_path: Path | str = Path("data/artifacts") / "rq2_clean_removal_by_model.tex",
) -> str:
    """Write and return the RQ2 CLEAN REMOVAL BY MODEL LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(clean_removal_by_model_table())
