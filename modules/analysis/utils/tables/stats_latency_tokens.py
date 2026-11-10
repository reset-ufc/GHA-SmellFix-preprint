"""LaTeX tables for the design effect on per-instance latency and tokens
(tab:stats_latency, tab:stats_tokens).

Each is a per-design-pair significance table (Cliff's delta + paired Wilcoxon,
Holm-corrected) built from the rep1 cost/latency frame — the D1-D2 comparison.

Self-loading (Variant B): pulls the frame from :mod:`utils.cost_latency` and
scores it with :mod:`utils.stats`, then writes the ``.tex``.
"""
from pathlib import Path

import pandas as pd

from utils import cost_latency as cl
from utils import stats
from utils.dataloader import ROOT
from utils.designs import DESIGNS

UNIT = ["file_key", "target_smell"]

DISPLAY = {
    "gpt54-nano": "GPT-5.4-nano", "gemini35-flash": "Gemini 3.5 Flash",
    "haiku45": "Claude Haiku 4.5", "deepseek-v4-flash": "DeepSeek V4 Flash",
    "gemma4": "Gemma 4",
}
_MODEL_ORDER = list(DISPLAY)


def _fmt_p(p: float) -> str:
    if p != p:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def _render(stats_df: pd.DataFrame, *, caption: str, label: str) -> str:
    """Per-design-pair stats table: one multirow block per model, a row per pair
    carrying Cliff's delta (+magnitude), the Holm p-value, and a significance mark.
    ``stats_df`` is the frame from :func:`utils.stats.by_model`."""
    present = [m for m in _MODEL_ORDER if m in set(stats_df["model"])]
    by = {(r.model, r.pair): r for r in stats_df.itertuples()}
    pairs = list(dict.fromkeys(stats_df["pair"]))  # preserve pair order
    lines = [
        r"\begin{table}[!ht]", r"\centering", r"\scriptsize",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{llrrlrc}", r"\toprule",
        r"\textbf{Model} & \textbf{Pair} & \textbf{$n$} & "
        r"\textbf{Cliff's $\delta$} & \textbf{Magnitude} & "
        r"\textbf{$p_{\text{Holm}}$} & \textbf{Sig.} \\", r"\midrule",
    ]
    for mi, m in enumerate(present):
        lines.append(rf"\multirow{{{len(pairs)}}}{{*}}{{{DISPLAY[m]}}}")
        for p in pairs:
            r = by.get((m, p))
            if r is None:
                lines.append(rf"& {p} & -- & -- & -- & -- & -- \\")
                continue
            sig = r"$\bullet$" if bool(r.significant) else "--"
            d = "--" if r.cliffs_delta != r.cliffs_delta else f"{r.cliffs_delta:+.2f}"
            lines.append(
                rf"& {p} & {int(r.n_a)} & {d} & {r.magnitude} "
                rf"& {_fmt_p(r.p_holm)} & {sig} \\")
        lines.append(r"\bottomrule" if mi == len(present) - 1 else r"\midrule")
    lines += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def _rep1_frame() -> pd.DataFrame:
    """rep1 cost/latency rows with valid latency (the matched-instance universe)."""
    clf = cl.add_cost(cl.load_cost_latency())
    return clf[clf["repetition"] == 1].dropna(subset=["latency_s"]).copy()


def stats_latency_table(
    output_path: Path | str = ROOT / "data" / "artifacts" / "stats_latency.tex",
) -> str:
    """Design effect on per-instance latency (D1-D2), written
    to ``output_path``. Returns the LaTeX string."""
    st = stats.by_model(_rep1_frame(), value_col="latency_s", unit_cols=UNIT,
                        designs=list(DESIGNS))
    latex = _render(
        st,
        caption=(r"Design effect on per-instance latency "
                 r"(Cliff's $\delta$, paired Wilcoxon, Holm-corrected)."),
        label="tab:stats_latency",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex


def stats_tokens_table(
    output_path: Path | str = ROOT / "data" / "artifacts" / "stats_tokens.tex",
) -> str:
    """Design effect on per-instance total tokens (D1-D2),
    written to ``output_path``. Returns the LaTeX string."""
    st = stats.by_model(_rep1_frame(), value_col="total_tokens", unit_cols=UNIT,
                        designs=list(DESIGNS))
    latex = _render(
        st,
        caption=(r"Design effect on per-instance token usage "
                 r"(Cliff's $\delta$, paired Wilcoxon, Holm-corrected)."),
        label="tab:stats_tokens",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(stats_latency_table())
    print(stats_tokens_table())
