"""LaTeX table for RQ1 overall detection effectiveness."""
from pathlib import Path

import pandas as pd

from utils import detection

DISPLAY = {
    "haiku45": "Claude Haiku 4.5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "gemini35-flash": "Gemini 3.5 Flash",
    "gpt54-nano": "GPT-5.4-nano",
}
COLS = [("f1", "F1"), ("precision", "Pr"), ("recall", "Re"), ("accuracy", "Acc")]

HEADER = r"""\begin{wraptable}{r}{7.5cm}
\centering
\tiny
\caption{Overall detection effectiveness across designs. The best and second-best model values per configuration are shown in \textbf{bold} and \underline{underlined}, respectively.}
\label{tab:rq1_overall_detection}
%\setlength{\tabcolsep}{3.2pt}
%\renewcommand{\arraystretch}{1.06}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Design}
& \textbf{Model}
& \textbf{$n$}
& \textbf{F1$\uparrow$}
& \textbf{Pr$\uparrow$}
& \textbf{Re$\uparrow$}
& \textbf{Acc$\uparrow$} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{wraptable}"""


def fn_cell(value: float, ranked: list[float]) -> str:
    """Percent cell; bold if best in its column, underlined if second best."""
    s = "--" if value != value else f"{value * 100:.2f}\\%"
    if ranked and value == ranked[0]:
        return rf"\textbf{{{s}}}"
    if len(ranked) > 1 and value == ranked[1]:
        return rf"\underline{{{s}}}"
    return s


def overall_detection_table(
    df: pd.DataFrame,
    output_path: Path | str = Path("data/artifacts") / "rq1_overall_detection.tex",
) -> tuple[str, Path]:
    """LaTeX for the RQ1 overall-detection table: one block per design,
    models ranked by F1, plus a (macro-mean) Avg. row. Best metric in each block
    is bold, second best underlined. Returns the ``.tex`` string, ready to paste."""
    by = detection.metrics_by(df, "model_alias", "design")
    designs = detection.DESIGNS
    parts = [HEADER]
    for i, design in enumerate(designs):
        block = by[by["design"] == design].sort_values("f1", ascending=False)
        ranked = {col: sorted(block[col], reverse=True) for col, _ in COLS}
        rows = [rf"\multirow{{5}}{{*}}{{\textbf{{{design}}}}}"]
        for _, r in block.iterrows():
            cells = " & ".join(fn_cell(r[col], ranked[col]) for col, _ in COLS)
            rows.append(rf"& {DISPLAY[r['model_alias']]:<19}& {int(r['n']):<4} & {cells} \\")
        avg = " & ".join(rf"\textbf{{{block[col].mean() * 100:.2f}\%}}" for col, _ in COLS)
        rows.append(rf"& \textbf{{Avg.}}{'':<7}& --   & {avg} \\")
        parts.append("\n".join(rows))
        if i < len(designs) - 1:
            parts.append(r"\midrule")
    latex = "\n".join(parts) + "\n" + FOOTER
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex, output_path
