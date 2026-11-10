"""LaTeX table for rq3 paired tests across designs (``tab:rq3_paired_tests_across_configurations``).

D1 vs. D2 for each model on four outcomes, with the (workflow, smell) pair as the unit:
each pair has one D1 and one D2 output of the same model on the same input, and the two
are compared directly. Scope and runs are those of the tables each outcome comes from:
rep1, four models, and the 570 pairs of the n15 ground truth
(``data/ground_truth_n15.json``).

Each pair contributes one value per design:
  * Detection correctness    = 1 when the pair's detection verdict matches the n15
                               ground truth, else 0: the scoring of the RQ1 tables
                               (``utils.detection``). All 570 pairs.
  * End-to-end clean removal = 1 when the run is Clean Removal, else 0, with the runs and
                               the CR / UX / BR / FN rule of the end-to-end table
                               (``rq2_end_to_end_conditional``): a detection miss counts
                               as a failure, and an ``uncertain`` run is dropped. Only the
                               ground-truth-positive pairs whose D1 and D2 runs are both
                               assessable are compared.
  * Latency                  = the run's ``run_duration_seconds``.
  * Est. cost                = the run's estimated cost, with the runs and the cost
                               formula of the absolute overhead table
                               (``rq3_absolute_overhead``).

Tests are two-sided and come from ``utils.stats``: McNemar's exact test for the two
binary outcomes (a binomial test on the pairs where D1 and D2 disagree), and the paired
Wilcoxon signed-rank test for latency and cost, with zero differences dropped. The Holm
family is one outcome's four model tests, and a test is significant when its
Holm-adjusted p is below 0.05.

The effect (Eff.) is D2 - D1 over the pairs the test uses (:func:`paired_effects`): the
difference of the two rates for a binary outcome, and the median of the paired
differences for latency (s) and estimated cost (USD x 1,000).

The output is only the paper's ``table`` environment.
"""

from pathlib import Path

import pandas as pd

from utils import detection, stats
from utils.designs import DESIGNS
from utils.tables.rq2_end_to_end_conditional import _scored_positive_runs
from utils.tables.rq3_absolute_overhead import _runs

# Display name + alias, fixed row order.
_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]
# n15 ground truth: 19 smells x (15 positive + 15 negative) pairs.
_PAIRS = 570
_ALPHA = 0.05
_UNIT = ["file_key", "smell"]
_KEYS = ["model_alias", "design", *_UNIT]

_HEADER = r"""
\begin{table}[ht!]
\centering
\tiny
\caption{Paired D1--D2 tests for effectiveness and overhead. Binary effects are D2--D1 rate differences; overhead effects are median paired D2--D1 differences. $p$-values are Holm-corrected within each metric.}

\label{tab:rq3_paired_tests_across_configurations}
%\setlength{\tabcolsep}{2.1pt}
%\renewcommand{\arraystretch}{1.06}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{ll|rrc|rrc|rrc|rrc}
\toprule
& & \multicolumn{3}{c|}{\textbf{Detection correctness}}
& \multicolumn{3}{c|}{\textbf{End-to-end clean removal}}
& \multicolumn{3}{c|}{\textbf{Latency}}
& \multicolumn{3}{c}{\textbf{Est. cost}} \\
\cmidrule(lr){3-5}
\cmidrule(lr){6-8}
\cmidrule(lr){9-11}
\cmidrule(lr){12-14}
\textbf{Comparison} & \textbf{Model}
& $p_{\mathrm{Holm}}$ & Eff. & Sig.
& $p_{\mathrm{Holm}}$ & Eff. & Sig.
& $p_{\mathrm{Holm}}$ & Eff. & Sig.
& $p_{\mathrm{Holm}}$ & Eff. & Sig. \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}"""


def _detection_values() -> pd.DataFrame:
    """Each (model, design, pair)'s detection correctness, 1 or 0."""
    runs = detection.add_n15_label(detection.load_results())
    runs = runs[
        (runs["repetition"] == 1)
        & runs["design"].isin(DESIGNS)
        & runs["model_alias"].isin([alias for _, alias in _MODELS])
    ].dropna(subset=["gt_n15", "smell_detected"])
    counts = runs.groupby(["model_alias", "design"]).size()
    if len(counts) != len(_MODELS) * len(DESIGNS) or (counts != _PAIRS).any():
        raise ValueError(f"expected {_PAIRS} scored pairs per model and design")
    correct = runs["smell_detected"].astype(bool) == runs["gt_n15"].astype(bool)
    runs = runs.assign(smell=runs["target_smell"], value=correct.astype(float))
    return runs[[*_KEYS, "value"]]


def _clean_removal_values() -> pd.DataFrame:
    """Each assessable positive (model, design, pair)'s end-to-end Clean
    Removal, 1 or 0."""
    pos = _scored_positive_runs()
    pos = pos.rename(columns={"model": "model_alias"})
    return pos.assign(value=(pos["cat"] == "CR").astype(float))[[*_KEYS, "value"]]


def _overhead_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each (model, design, pair)'s latency (s) and estimated cost (USD x 1,000)."""
    runs = _runs().rename(columns={"target_smell": "smell"})
    runs[["model_alias", "design"]] = runs[["model_alias", "design"]].astype(str)
    latency = runs.assign(value=runs["latency_s"])[[*_KEYS, "value"]]
    cost = runs.assign(value=runs["cost_1k"])[[*_KEYS, "value"]]
    return latency, cost


def _outcome_tests(values: pd.DataFrame, binary: bool) -> pd.DataFrame:
    """One outcome's D1-D2 test per model (indexed by alias), over the model's pairs.

    ``stats.by_model`` corrects within one model's design pairs, and D1-D2 is the
    only pair, so ``p_holm`` is recomputed here across the outcome's four models."""
    if values.duplicated(_KEYS).any():
        raise ValueError("more than one value for a (model, design, pair)")
    tests = stats.by_model(values, "value", _UNIT, binary=binary).set_index("model")
    tests = tests.loc[[alias for _, alias in _MODELS]]
    if (tests["pair"] != "D1-D2").any():
        raise ValueError("expected one D1-D2 test per model")
    tests["p_holm"] = stats.holm(tests["p_raw"].tolist())
    return tests


def paired_effects(values: pd.DataFrame, binary: bool) -> pd.DataFrame:
    """Each model's number of pairs and D2 - D1 effect, indexed by alias: the difference
    of the rates for a binary outcome, the median paired difference otherwise. A binary
    outcome also counts its discordant pairs, the ones with a 1 only under D1
    (``d1_only``) or only under D2 (``d2_only``): the only pairs McNemar's test uses. A
    pair counts only when both designs have a value, as in the tests."""
    wide = (values.set_index(["model_alias", *_UNIT, "design"])["value"]
            .unstack("design").dropna(subset=list(DESIGNS)))
    diff = wide["D2"] - wide["D1"]
    by_model = diff.groupby(level="model_alias")
    effects = pd.DataFrame({"pairs": by_model.size(),
                            "effect": by_model.mean() if binary else by_model.median()})
    if binary:
        effects["d1_only"] = (diff < 0).groupby(level="model_alias").sum()
        effects["d2_only"] = (diff > 0).groupby(level="model_alias").sum()
    return effects


def _fmt_p(p: float) -> str:
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_effect(effect: float, decimals: int) -> str:
    """A signed effect; a negative value that rounds to zero prints as ``0.00``."""
    if effect != effect:
        return "--"
    text = f"{effect:+.{decimals}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _render(cells: dict[str, list[list[str]]]) -> str:
    """The table environment, from each model's formatted cells (3 per outcome)."""
    models = []
    for label, alias in _MODELS:
        lines = [f"& {label}", *("& " + " & ".join(outcome) for outcome in cells[alias])]
        models.append("\n".join(lines) + r" \\")
    rows = rf"\multirow{{{len(_MODELS)}}}{{*}}{{D1 vs. D2}}" + "\n" + "\n\n".join(models)
    return "\n".join([_HEADER, rows, _FOOTER]).strip() + "\n"


def _build_latex() -> str:
    """Assemble the full LaTeX table from the four outcomes' tests and effects."""
    latency, cost = _overhead_values()
    # Each outcome: its values, whether it is binary, and the decimals of its effect.
    outcomes = [
        (_detection_values(), True, 2),
        (_clean_removal_values(), True, 2),
        (latency, False, 1),
        (cost, False, 2),
    ]
    cells = {alias: [] for _, alias in _MODELS}
    for values, binary, decimals in outcomes:
        tests = _outcome_tests(values, binary)
        effects = paired_effects(values, binary)
        for _, alias in _MODELS:
            p_holm = tests.at[alias, "p_holm"]
            sig = "Yes" if p_holm < _ALPHA else "No"
            effect = _fmt_effect(effects.at[alias, "effect"], decimals)
            cells[alias].append([_fmt_p(p_holm), effect, sig])
    return _render(cells)


def paired_tests_across_designs_table(
    output_path: Path | str = Path("data/artifacts") / "rq3_paired_tests_across_designs.tex",
) -> str:
    """Write and return the RQ3 PAIRED TESTS ACROSS DESIGNS LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(paired_tests_across_designs_table())
