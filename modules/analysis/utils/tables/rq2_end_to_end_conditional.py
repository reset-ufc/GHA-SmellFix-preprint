"""LaTeX table for rq2 end-to-end vs. conditional refactoring outcomes.

Scope, unit and outcome rule are those of the construct-aware outcomes table
(``rq2_refactoring_outcomes_by_smell``), whose loader and CR / UX / BR / FN mapping
are reused here: rep1, four models, D1/D2, and every ground-truth-positive run
counted on its own (19 smells x 15 pairs x 4 models x 2 designs = 2280 runs). A run
with an ``uncertain`` value in any adjudicated field is null and dropped, so the
end-to-end Overall row equals the column sums of that table.

Perspectives (row groups):
  * E2E (end-to-end)            = every assessable positive run.
  * Cond. (conditional on       = the assessable positive runs whose structured
    detection)                    detection verdict was ``present``
                                  (``smell_detected``).

Each perspective has one row per design (D1, D2) and an All row that pools both, so
every count in an All row is the sum of the D1 and D2 rows. As in the paper, the
All row of the conditional perspective is in bold.

Columns:
  * Out.        = assessable runs in the row's scope.
  * Det. miss   = Failed or No Patch runs whose detection verdict was absent.
                  These runs have no manual verdict. The conditional rows
                  exclude them by definition and show ``--``.
  * Patch fail  = Failed or No Patch runs whose smell was detected.
  * SRR         = CR + UX + BR, as a share of Out.
  * CR          = Clean Removal, as a share of Out.

Det. miss + Patch fail = the Failed or No Patch total of the outcomes table.

The output is only the paper's ``wraptable`` environment.
"""

from pathlib import Path

import pandas as pd

from utils.designs import DESIGNS
from utils.tables.rq2_refactoring_outcomes_by_smell import _load_joined, _positive_category

# Design pool: 19 smells x 15 positive pairs x 4 models x 2 designs.
_EXPECTED_POSITIVE_RUNS = 2280
# Rows of each perspective, in display order; All pools D1 and D2.
_ROWS = [*DESIGNS, "All"]

_HEADER = r"""
\begin{wraptable}{r}{7.5cm}
\centering
\caption{End-to-end and conditional refactoring outcomes on positive outputs.}
\label{tab:rq2_end_to_end_conditional}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.02}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Perspective} & \textbf{Design} & \textbf{Out.} &
\textbf{\begin{tabular}[c]{@{}r@{}}Det.\\miss\end{tabular}} &
\textbf{\begin{tabular}[c]{@{}r@{}}Patch\\fail\end{tabular}} &
\textbf{SRR $\uparrow$} & \textbf{CR $\uparrow$} \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}
\end{wraptable}"""


def _bold(cell: str, bold: bool) -> str:
    return rf"\textbf{{{cell}}}" if bold else cell


def _pct_cell(num: int, den: int, bold: bool = False) -> str:
    """Render a ``percentage {\\scriptsize(num)}`` cell, one decimal as in the text."""
    pct = _bold(f"{100.0 * num / den:.1f}\\%", bold)
    return f"{pct} {{\\scriptsize({_bold(f'{num:,}', bold)})}}"


def _counts(pos: pd.DataFrame) -> dict:
    """Outputs, detection misses, patch failures, SRR and CR of a scored frame."""
    failed = pos["cat"] == "FN"
    detected = pos["smell_detected"].astype(bool)
    return {
        "outputs": len(pos),
        "misses": int((failed & ~detected).sum()),
        "patch_failures": int((failed & detected).sum()),
        "srr": int(pos["cat"].isin(["CR", "UX", "BR"]).sum()),
        "cr": int((pos["cat"] == "CR").sum()),
    }


def _build_block(
    label: str, frames: dict[str, pd.DataFrame], show_misses: bool, bold_all: bool = False
) -> str:
    """Rows of one perspective: D1 and D2, then All below a partial rule."""
    rows = [rf"\multirow{{3}}{{*}}{{{label}}}"]
    for row in _ROWS:
        c = _counts(frames[row])
        bold = bold_all and row == "All"
        if row == "All":
            rows.append(r"\cmidrule(lr){2-7}")
        cells = [
            "",
            _bold(row, bold),
            _bold(f"{c['outputs']:,}", bold),
            _bold(f"{c['misses']:,}", bold) if show_misses else "--",
            _bold(f"{c['patch_failures']:,}", bold),
            _pct_cell(c["srr"], c["outputs"], bold),
            _pct_cell(c["cr"], c["outputs"], bold),
        ]
        rows.append(" & ".join(cells).lstrip() + r" \\")
    return "\n".join(rows)


def _scored_positive_runs() -> pd.DataFrame:
    """Assessable positive runs, each with its CR / UX / BR / FN code in ``cat``."""
    joined = _load_joined()
    pos = joined[joined["population"] == "positive"].copy()
    if len(pos) != _EXPECTED_POSITIVE_RUNS:
        raise ValueError(f"expected {_EXPECTED_POSITIVE_RUNS} positive runs, got {len(pos)}")
    pos["cat"] = pos.apply(_positive_category, axis=1)
    return pos[pos["cat"].notna()]  # drop null (uncertain) runs


def _build_latex() -> str:
    """Assemble the full LaTeX table from the joined data."""
    pos = _scored_positive_runs()
    by_design = {row: pos if row == "All" else pos[pos["design"] == row] for row in _ROWS}
    detected = {row: f[f["smell_detected"].astype(bool)] for row, f in by_design.items()}
    end_to_end = _build_block("E2E", by_design, show_misses=True)
    conditional = _build_block("Cond.", detected, show_misses=False, bold_all=True)
    body = "\n".join([_HEADER, end_to_end, r"\midrule", conditional, _FOOTER])
    return body.strip() + "\n"


def end_to_end_conditional_table(
    output_path: Path | str = Path("data/artifacts") / "rq2_end_to_end_conditional.tex",
) -> str:
    """Write and return the RQ2 END-TO-END VS. CONDITIONAL LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(end_to_end_conditional_table())
