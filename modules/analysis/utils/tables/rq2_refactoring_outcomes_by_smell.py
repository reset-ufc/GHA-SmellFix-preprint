"""LaTeX table for rq2 construct-aware refactoring outcomes by target smell.

Methodology (rep1 only, four models GPT-5.4-nano / Gemini 3.5 Flash /
Claude Haiku 4.5 / DeepSeek V4 Flash; Gemma excluded -- it is absent from the
validation artifacts):

The unit is the **single run**: each ground-truth pair ``(workflow, smell)`` from
the **n=15 ground truth** (15 positive + 15 negative pairs per smell) is evaluated
under 4 models x 2 designs = 8 runs (rep1). We do NOT collapse the 8 runs into a
pair-level majority; every run is counted on its own. Each smell starts from
15 x 4 x 2 = **120 positive runs and 120 negative runs**. ``uncertain`` verdicts
are treated as **null and excluded** (see below), so the actual per-smell pool is
at most 120; every cell is a share of that **valid pool**, and the Total columns
show the per-smell valid count. Counting at the run level (vs a pair majority vote)
keeps the model/design variance that a majority would erase -- e.g. minority
BR or Failed runs inside an otherwise-clean pair stay visible.
End-to-end view: detection false negatives and missing patches count as
``Failed or No Patch``, not as drops from the denominator (only ``uncertain`` is
dropped).

We join the full behaviour index (``artifacts/validation/behavior_index_rep1.csv``,
4560 GT-scoped rows = 2280 positive + 2280 negative (D1/D2 scope); supplies the population,
schema validity ``patch_valid`` and the modified/not-modified outcome) with the
resolved manual verdicts (``data/validation/manual_validation_final.csv``, 2933
adjudicated rows; supplies ``resolved_verdict`` /
``resolved_potential_behavior_impact`` (impact below) / ``resolved_construct_ok``).
Instances absent from the manual file (detection false negatives and unparseable
trees) keep a null verdict and fall into ``Failed or No Patch``.

Ground-truth-positive partition (per run, counted out of the valid pool):
  * Clean Removal              = verdict ``removed`` AND impact ``no`` AND
                                 construct_ok ``yes`` AND schema-valid
                                 (``patch_valid``). Removes the smell, preserves
                                 validity, no behaviour-relevant unexpected change.
  * Removal w/ Potential Behavior-Impacting Change (BR)
                               = verdict ``removed`` AND impact ``yes`` (takes
                                 precedence over the construct/validity checks).
  * Removal w/ Unexpected Chg. = verdict ``removed``, impact ``no``, but not Clean
                                 for a CONFIRMED reason: construct_ok ``no`` or a
                                 schema-invalid patch.
  * Failed or No Patch         = verdict ``not removed``, detection FN (NaN), or
                                 no validated patch. NOT a valid removal, but a
                                 confirmed failure (distinct from null/uncertain).
  * EXCLUDED (null)            = an ``uncertain`` value in ANY adjudicated field
                                 (verdict, impact, or construct_ok). Dropped from the
                                 pool entirely -- not scored in any bucket.

Ground-truth-negative partition (per run, out of the valid pool):
  * Unnecessary Edit   = resolved_verdict ``unnecessary`` (the pipeline edited a
                         workflow where the target smell is absent).
  * Correct Abstention = the valid pool minus Unnecessary Edit (unmodified /
                         preserved / necessary). An ``uncertain`` verdict is excluded
                         (null); negatives are scored on the verdict only.

The output is only the paper's ``table*`` environment, verbatim: abbreviated column
heads (SRR, CR, UX, BR, FN, CA, UE) and no notes block, since the paper text defines
the abbreviations. The rotated group labels are written out as the paper writes them.
Rows follow the smell types and order of ``utils.detection.SMELL_TYPES``.

Gray rows (``\\cellcolor`` on the data cells) are the five worst smells: lowest
Clean Removal, ties broken toward more BR then more Failed/No-Patch
runs.

The last row (``Overall``, in bold) pools the 19 smells: each percentage is the column's
total count over the total assessable runs, with that total count in parentheses; its
Assessable/Expected cells are totals. This is the rate that the RQ2 text and the
end-to-end table use.
"""

from pathlib import Path

import pandas as pd

try:
    from utils import dataloader
except ImportError:  # pragma: no cover - notebook execution fallback
    import utils.dataloader as dataloader  # type: ignore
from utils.designs import DESIGNS
from utils.detection import SMELL_TYPES

_REPO_ROOT = dataloader.ROOT
_BEHAVIOR_INDEX = _REPO_ROOT / "data" / "artifacts" / "validation" / "behavior_index_rep1.csv"
_MANUAL_FINAL = _REPO_ROOT / "data" / "validation" / "manual_validation_final.csv"

# The two row groups, in the paper's order (the smell-level detection table's).
_LOCALIZED = SMELL_TYPES["Localized"]
_CONTEXT = SMELL_TYPES["Context-dependent"]
# First cell of each group: the rotated label, verbatim from the paper.
_LOCALIZED_LABEL = "\n".join(
    [
        r"\multirow{10}{*}{",
        r"\rotatebox[origin=c]{90}{",
        r"        \shortstack{",
        r"            \textsc{ \textbf{Localized}}}",
        r"        }",
        r"    }",
        "    ",
    ]
)
_CONTEXT_LABEL = "\n".join(
    [
        r"\multirow{9}{*}{\rotatebox[origin=c]{90}{",
        r"        \shortstack{",
        r"            \textsc{ \textbf{Context-}}\\",
        r"            \textsc{ \textbf{dependent}}",
        r"        }",
        r"    }}",
    ]
)
# Design pool per smell and polarity: 15 pairs x 4 models x 2 designs.
_EXPECTED_RUNS = 120

_HEADER = r"""
\begin{table*}[ht!]
\centering
\caption{Construct-aware refactoring outcomes by target smell. Values aggregate models and designs; percentages include counts in parentheses. Gray rows highlight the lowest-CR smells.}

\label{tab:rq2_refactoring_outcomes_by_smell}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrl|rrl}
\toprule
\multirow{2}{*}{\textbf{Type}} &
  \multirow{2}{*}{\textbf{Smell}} &
  \multicolumn{6}{c|}{\textbf{Ground-truth-positive outputs}} &
  \multicolumn{3}{c}{\textbf{Ground-truth-negative outputs}} \\
\cmidrule(lr){3-8}
\cmidrule(lr){9-11}
 &
 &
  \textbf{\begin{tabular}[c]{r}SRR $\uparrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}CR $\uparrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}UX $\downarrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}BR $\downarrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}FN $\downarrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{l}Assessable/Expected\\ Pos. Outputs\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}CA $\uparrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{r}UE $\downarrow$\end{tabular}} &
  \textbf{\begin{tabular}[c]{l}Assessable/Expected\\ Negative Outputs\end{tabular}} \\
\midrule"""

_FOOTER = r"""\bottomrule
\end{tabular}%
}
\end{table*}"""


def _pct_cell(num: int, den: int) -> str:
    """Render a ``percentage{\\scriptsize(num)}`` cell.

    The denominator (``den``, the per-smell 120-run pool) is shared by every cell,
    so it is shown once in the Total columns and omitted from the count here.
    """
    pct = 100.0 * num / den
    return f"{pct:.0f}\\%~{{\\scriptsize({num})}}"


def _positive_category(row: pd.Series) -> str | None:
    """Map one positive run to CR / UX / BR / FN, or ``None`` to EXCLUDE it.

    Any run with an ``uncertain`` value in ANY adjudicated field (verdict, impact, or
    construct_ok) is NULL: the outcome cannot be confirmed, so the run is dropped
    from every count and denominator (scored as neither success nor failure). This
    is distinct from a NaN verdict (detection miss / no patch) and ``not removed``,
    which stay as Failed or No Patch. A confirmed ``yes`` impact stays BR.
    """
    if (
        row["resolved_verdict"] == "uncertain"
        or row["resolved_potential_behavior_impact"] == "uncertain"
        or row["resolved_construct_ok"] == "uncertain"
    ):
        return None
    verdict = row["resolved_verdict"]
    if verdict == "removed":
        if row["resolved_potential_behavior_impact"] == "yes":
            return "BR"
        if (
            row["resolved_potential_behavior_impact"] == "no"
            and row["resolved_construct_ok"] == "yes"
            and bool(row["patch_valid"])
        ):
            return "CR"
        return "UX"
    return "FN"


def _load_joined() -> pd.DataFrame:
    """Join the rep1 behaviour index with resolved manual verdicts."""
    idx = pd.read_csv(_BEHAVIOR_INDEX)
    manual = pd.read_csv(_MANUAL_FINAL)
    verdict_cols = [
        "stem",
        "resolved_verdict",
        "resolved_potential_behavior_impact",
        "resolved_construct_ok",
    ]
    joined = idx.merge(manual[verdict_cols], on="stem", how="left")
    return joined[joined["design"].isin(DESIGNS)].copy()


def _pos_counts(pos: pd.DataFrame) -> dict:
    """Run-level CR/UX/BR/FN counts for a positive sub-frame (``cat`` already set)."""
    return {code: int((pos["cat"] == code).sum()) for code in ("CR", "UX", "BR", "FN")}


def _worst_smells(joined: pd.DataFrame, n: int = 5) -> set:
    """The ``n`` worst smells: lowest run-level Clean Removal, ties to more
    BR then more Failed/No-Patch (the more dangerous / less fixed)."""
    scores = []
    for smell in _LOCALIZED + _CONTEXT:
        pos = joined[(joined["smell"] == smell) & (joined["population"] == "positive")].copy()
        pos["cat"] = pos.apply(_positive_category, axis=1)
        c = _pos_counts(pos)
        scores.append((c["CR"], -c["BR"], -c["FN"], smell))
    scores.sort()
    return {smell for *_, smell in scores[:n]}


def _smell_counts(smell: str, joined: pd.DataFrame) -> dict:
    """Run-level outcome counts and assessable/expected pools for one smell.

    Runs are counted individually (no pair-level majority collapse). ``uncertain``
    runs are excluded as null, so each smell's positive/negative pool is at most
    120 (15 pairs x 4 models x 2 designs) and the Assessable/Expected columns show
    the actual valid count per smell.
    """
    pos = joined[(joined["smell"] == smell) & (joined["population"] == "positive")].copy()
    neg = joined[(joined["smell"] == smell) & (joined["population"] == "negative")].copy()
    # Expected = the full per-smell/polarity design pool (15 pairs x 4 models x 2
    # designs = 120); assessable = what survives after dropping uncertain runs.
    expected_pos = len(pos)
    expected_neg = len(neg)
    if expected_pos != _EXPECTED_RUNS or expected_neg != _EXPECTED_RUNS:
        raise ValueError(
            f"{smell}: expected {_EXPECTED_RUNS} runs per polarity, "
            f"got {expected_pos} positive and {expected_neg} negative"
        )
    pos["cat"] = pos.apply(_positive_category, axis=1)
    # Drop null (uncertain) runs before counting / denominators. Positive runs are
    # scored on verdict+impact+construct, so any uncertain field drops them (see
    # _positive_category); negative runs are scored only on the verdict (unnecessary
    # vs abstention), so only an uncertain verdict drops them.
    neg = neg[neg["resolved_verdict"] != "uncertain"]

    c = _pos_counts(pos)
    n_neg = len(neg)
    ue = int((neg["resolved_verdict"] == "unnecessary").sum())
    return {
        "srr": c["CR"] + c["UX"] + c["BR"],  # removed at any quality: CR + UX + BR
        "cr": c["CR"],
        "ux": c["UX"],
        "br": c["BR"],
        "fn": c["FN"],
        "n_pos": sum(c.values()),
        "expected_pos": expected_pos,
        "ca": n_neg - ue,
        "ue": ue,
        "n_neg": n_neg,
        "expected_neg": expected_neg,
    }


def _build_row(
    group_cell: str,
    smell: str,
    joined: pd.DataFrame,
    highlight: bool = False,
) -> str:
    """Build the LaTeX body line for one smell (without the trailing ``\\\\``)."""
    c = _smell_counts(smell, joined)
    # Highlight worst smells by shading the DATA columns only (\cellcolor on cols
    # 2-11), never the Type column. A full-row \rowcolor would paint its opaque
    # fill over the group's \multirow label where the label centres onto a shaded
    # row (e.g. Context-dependent centres on SingleOSExecution), hiding it.
    cc = r"\cellcolor{gray!10}" if highlight else ""
    cells = [
        f"\\texttt{{{smell}}}",
        f"~{_pct_cell(c['srr'], c['n_pos'])}",
        f"~{_pct_cell(c['cr'], c['n_pos'])}",
        f"~{_pct_cell(c['ux'], c['n_pos'])}",
        f"~{_pct_cell(c['br'], c['n_pos'])}",
        f"~{_pct_cell(c['fn'], c['n_pos'])}",
        f"~{c['n_pos']}/{c['expected_pos']}",
        f"~{_pct_cell(c['ca'], c['n_neg'])}",
        f"~{_pct_cell(c['ue'], c['n_neg'])}",
        f"~{c['n_neg']}/{c['expected_neg']}",
    ]
    return f"{group_cell} &\n" + " &\n".join(f"  {cc}{cell}" for cell in cells)


def _overall_cell(counts: list[dict], key: str, den: str) -> str:
    """Render the pooled percentage over all smells, with the total count in parentheses."""
    total = sum(c[key] for c in counts)
    pooled = 100.0 * total / sum(c[den] for c in counts)
    return f"{pooled:.0f}\\%~{{\\scriptsize({total:,})}}"


def _build_overall_row(joined: pd.DataFrame) -> str:
    """Build the Overall line (without the trailing ``\\\\``), every cell in bold.

    Each percentage cell is the column's total count over the total assessable runs
    of the 19 smells. The parentheses and the Assessable/Expected cells show totals.
    """
    counts = [_smell_counts(smell, joined) for smell in _LOCALIZED + _CONTEXT]

    def total(key: str) -> str:
        return f"{sum(c[key] for c in counts):,}"

    cells = [
        f"~{_overall_cell(counts, 'srr', 'n_pos')}",
        f"~{_overall_cell(counts, 'cr', 'n_pos')}",
        f"~{_overall_cell(counts, 'ux', 'n_pos')}",
        f"~{_overall_cell(counts, 'br', 'n_pos')}",
        f"~{_overall_cell(counts, 'fn', 'n_pos')}",
        f"~{total('n_pos')}/{total('expected_pos')}",
        f"~{_overall_cell(counts, 'ca', 'n_neg')}",
        f"~{_overall_cell(counts, 'ue', 'n_neg')}",
        f"~{total('n_neg')}/{total('expected_neg')}",
    ]
    label = r"\multicolumn{2}{l}{\textbf{Overall}}"
    return f"{label} &\n" + " &\n".join(f"  \\textbf{{{cell}}}" for cell in cells)


def _build_group(first_cell: str, smells: list[str], joined: pd.DataFrame, worst: set) -> str:
    """Rows of one smell group, blank-line separated; ``first_cell`` holds the label."""
    rows = []
    for i, smell in enumerate(smells):
        group_cell = first_cell if i == 0 else " "
        rows.append(_build_row(group_cell, smell, joined, highlight=smell in worst) + r" \\")
    return "\n\n".join(rows)


def _build_latex() -> str:
    """Assemble the full LaTeX table from the joined data."""
    joined = _load_joined()
    worst = _worst_smells(joined, n=5)
    localized = _build_group(_LOCALIZED_LABEL, _LOCALIZED, joined, worst)
    context = _build_group(_CONTEXT_LABEL, _CONTEXT, joined, worst)
    overall = _build_overall_row(joined) + r" \\"
    body = "\n\n".join([_HEADER, localized, r"\midrule", context, r"\midrule", overall, _FOOTER])
    return body.strip() + "\n"


def refactoring_outcomes_by_smell_table(
    output_path: Path | str = Path("data/artifacts") / "rq2_refactoring_outcomes_by_smell.tex",
) -> str:
    """Write and return the RQ2 REFACTORING OUTCOMES BY SMELL LaTeX table."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = _build_latex()
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(refactoring_outcomes_by_smell_table())
