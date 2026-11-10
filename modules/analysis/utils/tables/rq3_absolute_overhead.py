"""LaTeX table for rq3 absolute operational overhead by model and design.

Scope and unit are those of RQ1 and of the paired tests: rep1, four models, D1/D2,
and one run per (workflow, smell) pair of the n15 ground truth
(``data/ground_truth_n15.json``), so every (model, design) has 570 runs.
DeepSeek V4 Flash's runs on the rest of the full ground truth are left out, and so
are the 11 older runs of the other three models outside n15.

Per run, with the billable tokens and cost of ``utils.cost_latency.add_cost``
(token counts are summed over all LLM calls):
  * Latency       = ``run_duration_seconds``, the wall clock of the pipeline run.
  * Input tokens  = ``token_usage.total.input_tokens``, cache-read tokens included.
  * Output tokens = ``token_usage.total.output_tokens``, plus ``reasoning_tokens``
                    for Gemini 3.5 Flash, whose thinking tokens Agno records apart
                    from its output tokens.
  * Est. cost     = (input tokens x P_in + output tokens x P_out) / 10^6 USD, with
                    the prices of ``utils.cost_latency.PRICING``: the paper's
                    formula, with no cache discount.

Each cell is the median over the 570 runs; the cost cell is the median per-run cost
times 1,000. The medians are rounded half up (latency to 0.1 s, tokens to 1, cost to
0.01), and Delta = D2 - D1 of the rounded medians, so it matches the two cells
beside it.

The output is only the paper's ``table*`` environment.
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from utils import cost_latency, detection
from utils.designs import DESIGNS

# Display name + alias, fixed row order (as in the RQ3 paired-tests table).
_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]
# Balanced n15 ground truth: 19 smells x (15 positive + 15 negative) pairs.
_EXPECTED_RUNS = 570
# Column blocks, in display order: per-run value and the step its medians round to.
_BLOCKS = [
    ("latency_s", "0.1"),
    ("billable_input_tokens", "1"),
    ("billable_output_tokens", "1"),
    ("cost_1k", "0.01"),
]

_HEADER = r"""
\begin{table*}[ht!]
\centering
\scriptsize
\caption{Median operational overhead per \textit{(workflow, smell)} run by design and model. Cost reports median per-run estimated cost multiplied by 1,000. $\Delta$ = D2--D1.}
\label{tab:rq3_absolute_overhead}
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.05}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{l|rrr|rrr|rrr|rrr}
\toprule
\multirow{2}{*}{\textbf{Model}}
& \multicolumn{3}{c|}{\textbf{Latency (s)}}
& \multicolumn{3}{c|}{\textbf{Input tokens}}
& \multicolumn{3}{c|}{\textbf{Output tokens}}
& \multicolumn{3}{c}{\textbf{Median est. cost $\times$1K (USD)}} \\
\cmidrule(lr){2-4}
\cmidrule(lr){5-7}
\cmidrule(lr){8-10}
\cmidrule(lr){11-13}
& \textbf{D1} & \textbf{D2} & \textbf{$\Delta$}
& \textbf{D1} & \textbf{D2} & \textbf{$\Delta$}
& \textbf{D1} & \textbf{D2} & \textbf{$\Delta$}
& \textbf{D1} & \textbf{D2} & \textbf{$\Delta$} \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}"""


def _runs() -> pd.DataFrame:
    """The rep1 n15 runs of the four models, with their tokens and cost per run."""
    runs = detection.add_n15_label(cost_latency.load_cost_latency())
    runs = runs[(runs["repetition"] == 1) & runs["gt_n15"].notna()].copy()
    if runs.duplicated(["model_alias", "design", "file_key", "target_smell"]).any():
        raise ValueError("more than one rep1 run for a (model, design, pair)")
    counts = runs.groupby(["model_alias", "design"], observed=True).size()
    expected = {(alias, design) for _, alias in _MODELS for design in DESIGNS}
    if set(counts.index) != expected or (counts != _EXPECTED_RUNS).any():
        raise ValueError(f"expected {_EXPECTED_RUNS} runs per model and design, "
                         f"got {counts.to_dict()}")
    runs = cost_latency.add_cost(runs)
    runs["cost_1k"] = 1000 * runs["cost_usd"]
    return runs


def _rounded(value: float, step: str) -> Decimal:
    """``value`` rounded half up to a multiple of ``step``, from its shortest repr."""
    return Decimal(str(float(value))).quantize(Decimal(step), rounding=ROUND_HALF_UP)


def _model_cells(runs: pd.DataFrame) -> list[tuple[str, str, str]]:
    """The D1, D2 and Delta cells of each column block, for one model's runs."""
    cells = []
    for column, step in _BLOCKS:
        d1, d2 = (
            _rounded(runs.loc[runs["design"] == design, column].median(), step)
            for design in DESIGNS
        )
        cells.append((f"{d1:,}", f"{d2:,}", f"{d2 - d1:+,}"))
    return cells


def _render(cells: dict[str, list[tuple[str, str, str]]]) -> str:
    """The table environment, from each model's (D1, D2, Delta) cells per block."""
    models = []
    for label, alias in _MODELS:
        lines = [label, *("& " + " & ".join(block) for block in cells[alias])]
        models.append("\n".join(lines) + r" \\")
    return "\n".join([_HEADER, "\n\n".join(models), _FOOTER]).strip() + "\n"


def _build_latex() -> str:
    """Assemble the full LaTeX table from the runs."""
    runs = _runs()
    return _render({alias: _model_cells(runs[runs["model_alias"] == alias])
                    for _, alias in _MODELS})


def absolute_overhead_table(
    output_path: Path | str = Path("data/artifacts") / "rq3_absolute_overhead.tex",
) -> str:
    """Write and return the RQ3 ABSOLUTE OVERHEAD LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(absolute_overhead_table())
