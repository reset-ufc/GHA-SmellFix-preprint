"""Cost-vs-latency engine over ``data/agents_runs_outputs``.

Each result JSON carries its own wall-clock latency (``run_duration_seconds``)
and token accounting (``token_usage.total``). One row per result = one refactoring
instance ``(file_key, smell, design, model, rep)``; we pool all repetitions, so
``n`` per cell = the number of instances the model actually produced.

Token counts are the ground truth in the data and need no external input, so the
**token view** (input/output/cache per instance) is the robust, primary metric.
USD cost is a thin layer on top: :func:`add_cost` applies the paper's formula with
the :data:`PRICING` table — the cloud APIs use published list rates (verified
2026-06-23); DeepSeek uses its provider weighted-avg effective rate (2026-06-25). See
:data:`PRICING` for the caveats.

This module keeps its own light loader (``load_cost_latency``) rather than going
through :func:`utils.dataloader.load_results`: it only needs latency + the flat
``token_usage.total`` counters, and re-globbing keeps the per-instance ``n`` exactly
as the cost baselines were produced (no dedup). Quality (F1) for the trade-off is
sourced from :mod:`utils.detection`, scored on the same rep1 rows.

Run ``python -m utils.cost_latency`` for a self-check.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.dataloader import ROOT, DATA_ROOT
from utils.designs import DESIGNS

DATA = DATA_ROOT
OUTPUTS = ROOT / "data" / "artifacts" / "cost_latency"  # generated tables/plots (regenerable)

# Fixed layout (order + display name), identical to detection.py. Every model
# currently has data; an alias=None entry (no model) would be omitted from aggregates.
TABLE_MODELS = [
    ("GPT-5.4-nano", "gpt54-nano"),
    ("Gemini 3.5 Flash", "gemini35-flash"),
    ("Claude Haiku 4.5", "haiku45"),
    ("DeepSeek V4 Flash", "deepseek-v4-flash"),
]
# Gemma 4 is excluded from the study (see utils/detection); its raw results stay on
# disk but the loader drops them. Pass exclude=() to opt back in.
EXCLUDE_MODELS = ("gemma4",)

# USD per 1,000,000 tokens, as in the paper's evaluated-LLMs table (tab:evaluated_llms),
# which is generated from this dict. The paper's cost formula prices every billable
# input token at ``in`` and every billable output token at ``out`` (see add_cost), so
# no cache discount is modelled.
# Cloud APIs use published LIST rates (verified 2026-06-23, unchanged on 2026-09-28):
#   haiku45        $1 / $5 per MTok
#   gpt54-nano     $0.20 / $1.25 (standard, short context)
#   gemini35-flash $1.50 / $9.00 (out incl. thinking)
# DeepSeek is priced from OpenRouter's WEIGHTED-AVG EFFECTIVE rate ($0.064/$0.251,
# 2026-06-25) over the providers that served it: that average already nets the
# prompt-cache discount.
PRICING = {
    "haiku45":           {"in": 1.00,  "out": 5.00},   # list rate
    "gpt54-nano":        {"in": 0.20,  "out": 1.25},   # list rate
    "gemini35-flash":    {"in": 1.50,  "out": 9.00},   # list rate
    "deepseek-v4-flash": {"in": 0.064, "out": 0.251},  # weighted-avg effective
}

# Models whose token accounting keeps the thinking tokens out of output_tokens: Agno
# maps Gemini's candidates_token_count, which excludes thoughts_token_count, to
# output_tokens. The other providers count reasoning inside output_tokens, so there
# reasoning_tokens is a subset and is never added on top.
REASONING_OUTSIDE_OUTPUT = frozenset({"gemini35-flash"})

TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens",
                "cache_read_tokens", "cache_write_tokens", "calls")
ROW_FIELDS = ("model_alias", "design", "repetition", "file_key",
              "target_smell", "run_duration_seconds", "agent_call_count")


def load_cost_latency(data: Path = DATA, exclude: tuple = EXCLUDE_MODELS) -> pd.DataFrame:
    """Every result JSON under data/agents_runs_outputs as one row: latency + flattened token
    totals. Unparseable / token-less files are skipped and recorded in
    ``df.attrs['skipped']``. Models in ``exclude`` are dropped (default: Gemma 4)."""
    rows, bad = [], []
    for f in sorted(Path(data).glob("*/design_*/results/*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bad.append(str(f.relative_to(data)))
            continue
        total = (d.get("token_usage") or {}).get("total") or {}
        row = {k: d.get(k) for k in ROW_FIELDS}
        row |= {k: total.get(k, 0) or 0 for k in TOKEN_FIELDS}
        row["file"] = f.name
        rows.append(row)
    df = pd.DataFrame(rows)
    if exclude:
        df = df[~df["model_alias"].isin(exclude)].reset_index(drop=True)
    df["latency_s"] = pd.to_numeric(df["run_duration_seconds"], errors="coerce")
    df["design"] = pd.Categorical(df["design"], DESIGNS, ordered=True)
    df.attrs["skipped"] = bad
    return df


def add_cost(df: pd.DataFrame, pricing: dict = PRICING) -> pd.DataFrame:
    """Add each instance's billable tokens and estimated cost, the paper's formula.

    * ``billable_input_tokens``  = ``input_tokens``. OpenAI, OpenRouter and Gemini
      count the cache-read tokens inside it; Anthropic reports them apart, and no
      Claude run has any.
    * ``billable_output_tokens`` = ``output_tokens``, plus ``reasoning_tokens`` for
      the models in :data:`REASONING_OUTSIDE_OUTPUT`.
    * ``total_tokens`` = billable input + billable output.
    * ``cost_usd`` = (billable input x in + billable output x out) / 10^6, priced from
      ``pricing[model_alias]``; NaN when the model has no pricing entry."""
    df = df.copy()
    thinking = df["model_alias"].isin(REASONING_OUTSIDE_OUTPUT)
    df["billable_input_tokens"] = df["input_tokens"]
    df["billable_output_tokens"] = (df["output_tokens"]
                                    + df["reasoning_tokens"].where(thinking, 0))
    df["total_tokens"] = df["billable_input_tokens"] + df["billable_output_tokens"]
    p_in = df["model_alias"].map({alias: p["in"] for alias, p in pricing.items()})
    p_out = df["model_alias"].map({alias: p["out"] for alias, p in pricing.items()})
    df["cost_usd"] = (df["billable_input_tokens"] * p_in
                      + df["billable_output_tokens"] * p_out) / 1e6
    return df


# Named aggregation (pandas 3.x-stable; avoids groupby.apply on the grouping cols).
# "n" via size == len(group) (counts every row, NaN included), matching the old _agg.
_AGG = dict(
    n=("latency_s", "size"),
    calls_mean=("agent_call_count", "mean"),
    latency_med=("latency_s", "median"),
    latency_mean=("latency_s", "mean"),
    in_mean=("input_tokens", "mean"),
    out_mean=("output_tokens", "mean"),
    tokens_mean=("total_tokens", "mean"),
    cost_mean=("cost_usd", "mean"),
    cost_total=("cost_usd", "sum"),
)


def per_model(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: median/mean latency, mean tokens, mean & total cost."""
    out = df.groupby("model_alias", observed=True).agg(**_AGG).reset_index()
    return out.sort_values("cost_mean").reset_index(drop=True)


def per_model_design(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (design, model): the same aggregates, split by design."""
    out = (df.groupby(["design", "model_alias"], observed=True)
             .agg(**_AGG).reset_index())
    return out.sort_values(["design", "model_alias"]).reset_index(drop=True)


def overhead_by_design(df: pd.DataFrame) -> pd.DataFrame:
    """Overhead ladder, pooled over models: one row per design with mean
    agent calls, mean total tokens and mean/median latency per instance, plus
    each quantity as a ratio to the D1 baseline (the *overhead multiplier*).

    This is what characterizes the cost of task decomposition: ``calls_x`` /
    ``tokens_x`` / ``latency_x`` say how many times the baseline a design
    spends. (Decomposition need not inflate either: smaller focused calls can
    undercut one monolithic prompt.)"""
    g = (df.groupby("design", observed=True)
           .agg(n=("latency_s", "size"),
                calls_mean=("agent_call_count", "mean"),
                tokens_mean=("total_tokens", "mean"),
                out_mean=("output_tokens", "mean"),
                latency_med=("latency_s", "median"),
                latency_mean=("latency_s", "mean")))
    g = g.reindex([d for d in DESIGNS if d in g.index])
    base = g.iloc[0]
    g["calls_x"] = g["calls_mean"] / base["calls_mean"]
    g["tokens_x"] = g["tokens_mean"] / base["tokens_mean"]
    g["latency_x"] = g["latency_mean"] / base["latency_mean"]
    return g.reset_index()


def tradeoff_frame(data: Path = DATA) -> pd.DataFrame:
    """One row per (design, model) joining detection quality with cost/latency.

    F1/Pr/Re/Acc are scored on the rep1 rows against the n15 ground truth (the same
    ``detection.metrics`` engine as RQ1), and cost/latency are aggregated on the SAME
    rep1 rows for a fair trade-off. Columns: design, model_alias, n, F1, Pr, Re,
    Acc, calls_mean, latency_med, latency_mean, cost_mean, tokens_mean."""
    from utils import detection  # local: keeps cost_latency importable without sklearn

    cl1 = add_cost(load_cost_latency(data))
    cl1 = cl1[cl1["repetition"] == 1]
    cost = per_model_design(cl1)[
        ["design", "model_alias", "calls_mean", "latency_med",
         "latency_mean", "cost_mean", "tokens_mean"]]

    det = detection.add_n15_label(detection.load_results(data))
    det = det[det["repetition"] == 1].dropna(subset=["gt_n15", "smell_detected"])
    det = det[~det["model_alias"].isin(EXCLUDE_MODELS)]
    recs = []
    for (design, model), g in det.groupby(["design", "model_alias"], observed=True):
        m = detection.metrics(g["gt_n15"], g["smell_detected"])
        recs.append({"design": str(design), "model_alias": model, "n": m["n"],
                     "F1": m["f1"], "Pr": m["precision"], "Re": m["recall"],
                     "Acc": m["accuracy"]})
    mt = pd.DataFrame.from_records(
        recs, columns=["design", "model_alias", "n", "F1", "Pr", "Re", "Acc"])
    mt["design"] = pd.Categorical(mt["design"], DESIGNS, ordered=True)

    merged = mt.merge(cost, on=["design", "model_alias"], how="left")
    return merged.sort_values(["design", "model_alias"]).reset_index(drop=True)


# marker per model for the trade-off facets (cycled over present models).
_MARKERS = ["o", "s", "^", "D", "v", "P"]

# Combined-scatter style: each design gets a fixed (marker, colour) so the
# two combined figures (cost_vs_latency, f1_vs_cost) read consistently. The design is
# doubly encoded (shape + colour); model identity is the per-point text label.
DESIGN_STYLE = {
    "D1": ("o", "#2ca02c"),  # circle, green
    "D2": ("s", "#e8b800"),  # square, amber (yellow)
}

# label offset (in points) + alignment for each of 8 directions around a marker.
_PLACE = {
    "right": dict(xytext=(6, 0), ha="left", va="center"),
    "left": dict(xytext=(-6, 0), ha="right", va="center"),
    "up": dict(xytext=(0, 7), ha="center", va="bottom"),
    "down": dict(xytext=(0, -7), ha="center", va="top"),
    "upright": dict(xytext=(5, 5), ha="left", va="bottom"),
    "upleft": dict(xytext=(-5, 5), ha="right", va="bottom"),
    "downright": dict(xytext=(5, -5), ha="left", va="top"),
    "downleft": dict(xytext=(-5, -5), ha="right", va="top"),
}


def _style_map(tf: pd.DataFrame) -> dict:
    """alias -> (full display name, marker, color), in fixed TABLE_MODELS order
    over only the models actually present in ``tf``."""
    import matplotlib.pyplot as plt
    disp = {a: d for d, a in TABLE_MODELS if a}
    present = [a for _, a in TABLE_MODELS if a in set(tf["model_alias"])]
    return {a: (disp[a], _MARKERS[i % len(_MARKERS)], plt.cm.tab10.colors[i])
            for i, a in enumerate(present)}


def _legend_handles(style: dict) -> list:
    """Line2D proxies (marker+colour+full name per model) for a model legend."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker=m, color=c, linestyle="", label=d)
            for d, m, c in style.values()]


def _draw_panel(ax, sub: pd.DataFrame, x: str, y: str, style: dict,
                labels: dict = None, placeholders: list = None) -> None:
    """Scatter one design's models onto ``ax`` — marker+colour per model.

    Identity is shown one of two ways. By default (``labels`` is None) the markers
    carry no text and the caller adds a legend. If ``labels`` (``model_alias ->
    direction`` in :data:`_PLACE`) is given, each marker is instead annotated with
    its full model name in that direction — and the caller drops the legend.
    ``placeholders`` is a list of ``(display_name, ax_x, ax_y, direction)`` drawn
    as hollow markers in *axes-fraction* coordinates (so they sit in a fixed
    corner and never rescale the data axes) for expected models with no data yet.
    Padded x-margins keep the markers off the frame."""
    import matplotlib.pyplot as plt
    for r in sub.itertuples():
        if r.model_alias not in style:
            continue
        disp, m, c = style[r.model_alias]
        ax.scatter(getattr(r, x), getattr(r, y), marker=m, color=c, s=95,
                   edgecolor="black", linewidth=0.4, zorder=3)
        if labels:
            ax.annotate(disp, (getattr(r, x), getattr(r, y)), fontsize=7,
                        textcoords="offset points",
                        **_PLACE[labels.get(r.model_alias, "right")])
    if placeholders:
        idx = {d: i for i, (d, _) in enumerate(TABLE_MODELS)}
        for name, ax_x, ax_y, d in placeholders:
            c = plt.cm.tab10.colors[idx[name]]
            ax.scatter(ax_x, ax_y, marker="X", s=85, facecolors="none",
                       edgecolors=c, linewidth=1.3, zorder=3, transform=ax.transAxes)
            ax.annotate(name, (ax_x, ax_y), xycoords="axes fraction", fontsize=7,
                        color="gray", textcoords="offset points", **_PLACE[d])
    ax.margins(x=0.20, y=0.15)
    ax.grid(True, alpha=0.3)


def plot_combined_scatter(frame: pd.DataFrame, x: str, y: str, *,
                          xlabel: str, ylabel: str, yscale: float = 1.0,
                          label_dirs: dict = None, placeholders: list = None,
                          figsize: tuple = (4.5, 3.5)):
    """Single-panel scatter of every ``(design, model)`` row — each point
    labelled ``'<model> <design>'``. The design drives BOTH the marker and the
    colour via :data:`DESIGN_STYLE` (so D1/D2 are visually distinct twice over);
    model identity is carried by the label text. This is the all-points-on-one-axis
    style shared by ``cost_vs_latency`` and the combined ``f1_vs_cost`` (cf. the
    faceted :func:`plot_tradeoff`, which instead colours by model).

    ``x``/``y`` are columns of ``frame``; ``y`` is multiplied by ``yscale`` (e.g.
    1000 to show USD per 1k instances). ``label_dirs`` maps ``(model_alias,
    design) -> direction`` in :data:`_PLACE` (default ``'right'``);
    ``placeholders`` is a list of ``(display_name, x, y, direction)`` drawn as
    hollow gray X markers in DATA coords for expected-but-absent models (no design,
    hence neutral). Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    disp = {a: d for d, a in TABLE_MODELS if a}
    label_dirs = label_dirs or {}

    fig, ax = plt.subplots(figsize=figsize)
    for r in frame.itertuples():
        if r.model_alias not in disp:
            continue
        design = str(r.design)
        marker, color = DESIGN_STYLE[design]
        xv, yv = getattr(r, x), getattr(r, y) * yscale
        ax.scatter(xv, yv, s=95, marker=marker, color=color, edgecolor="black",
                   linewidth=0.5, zorder=3)
        ax.annotate(f"{disp[r.model_alias]} {design}", (xv, yv), fontsize=6,
                    textcoords="offset points",
                    **_PLACE[label_dirs.get((r.model_alias, design), "right")])
    if placeholders:
        for name, xv, yv, d in placeholders:
            ax.scatter(xv, yv, marker="X", s=110, facecolors="none",
                       edgecolors="gray", linewidth=1.5, zorder=3)
            ax.annotate(name, (xv, yv), fontsize=6, color="gray",
                        textcoords="offset points", **_PLACE[d])
    ax.margins(x=0.16, y=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    present = [d for d in DESIGNS if d in set(frame["design"].astype(str))]
    ax.legend(handles=[Line2D([], [], marker=DESIGN_STYLE[d][0], color=DESIGN_STYLE[d][1],
                              linestyle="", markeredgecolor="black", label=d)
                       for d in present],
              title="design", fontsize=6, loc="best")
    fig.tight_layout()
    return fig


def plot_tradeoff(tf: pd.DataFrame, x: str = "cost_mean", *, y: str = "F1",
                  xlabel: str = "mean cost per instance (USD)",
                  label_dirs: dict = None, placeholders: list = None):
    """Faceted F1-vs-``x`` scatter, one subplot per design. ``x`` is any
    numeric column of ``tf`` (``cost_mean`` / ``latency_med`` / ``calls_mean``).
    No figure title (the LaTeX caption supplies it); only the D1/D2 panel
    labels remain.

    Identity defaults to a shared bottom legend (full names). Pass ``label_dirs``
    (``{design: {model_alias: direction}}``) to instead print each model's name
    next to its marker (no legend) — and ``placeholders`` (a list of
    ``(name, ax_x, ax_y, direction)``) to add expected-but-no-data models as
    hollow markers in a fixed corner of every panel. Returns the figure."""
    import matplotlib.pyplot as plt

    style = _style_map(tf)
    designs = [d for d in DESIGNS if d in set(tf["design"].astype(str))]
    fig, axes = plt.subplots(1, len(designs), figsize=(3.8 * len(designs), 3.6),
                             sharey=True, squeeze=False)
    for ax, design in zip(axes[0], designs):
        _draw_panel(ax, tf[tf["design"].astype(str) == design], x, y, style,
                    labels=(label_dirs or {}).get(design), placeholders=placeholders)
        ax.set_title(design)
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel(y)
    if label_dirs:
        fig.tight_layout()
    else:
        fig.legend(handles=_legend_handles(style), loc="lower center",
                   ncol=len(style), fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save_tradeoff_per_design(tf: pd.DataFrame, out_dir: Path, stem: str,
                             x: str = "cost_mean", *, y: str = "F1",
                             xlabel: str = "mean cost per instance (USD)",
                             label_dirs: dict = None, placeholders: list = None) -> list:
    """Save the per-design single-panel figures into their own directory
    ``out_dir/stem/`` as ``d1.png`` / ``d2.png`` — one panel each
    (same as a facet of :func:`plot_tradeoff`: full names, no title) — for a row
    of LaTeX ``subfigure``\\ s that share one y-axis.

    To make that sharing correct: all panels use the SAME y-range, and only
    ``d1`` carries the y-axis (``F1``) ticks+label — ``d2`` blanks that
    margin. The axes geometry is fixed and identical across the panels (no tight
    crop), so the plot boxes line up when placed side by side. Figures are closed
    (not shown), so a notebook still renders only the combined plot. Returns the
    saved paths."""
    import matplotlib.pyplot as plt

    style = _style_map(tf)
    designs = [d for d in DESIGNS if d in set(tf["design"].astype(str))]
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    ylo, yhi = float(tf[y].min()), float(tf[y].max())
    pad = (yhi - ylo) * 0.18 or 0.01  # headroom for the topmost label
    paths = []
    for i, design in enumerate(designs):
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        _draw_panel(ax, tf[tf["design"].astype(str) == design], x, y, style,
                    labels=(label_dirs or {}).get(design), placeholders=placeholders)
        ax.set_ylim(ylo - pad, yhi + pad)  # shared range, overrides _draw_panel y-margin
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(y)
            # one model legend for the whole subfigure row, on d1 (lower-right is
            # empty for these data); identity via legend, unless per-point names.
            if not label_dirs:
                ax.legend(handles=_legend_handles(style), loc="lower right",
                          fontsize=6.5, framealpha=0.9)
        else:
            ax.tick_params(labelleft=False)  # share d1's y-axis: blank the numbers + label
        # Fixed, identical geometry on every panel so the boxes align in LaTeX.
        fig.subplots_adjust(left=0.185, right=0.965, bottom=0.165, top=0.97)
        p = d / f"{design.lower()}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_overhead(df: pd.DataFrame):
    """Two panels — mean tokens and mean latency per instance — as the overhead
    ladder across designs. x = model (full names); the D1/D2 bars use a
    light→dark sequence so the per-model trend of decomposition is legible. No
    figure title (the LaTeX caption supplies it). Returns the Matplotlib figure."""
    import matplotlib.pyplot as plt
    import numpy as np

    disp = {a: d for d, a in TABLE_MODELS if a}
    models = [a for _, a in TABLE_MODELS if a in set(df["model_alias"])]
    designs = [d for d in DESIGNS if d in set(df["design"].astype(str))]
    shade = ["#9ecae1", "#4292c6", "#08519c", "#08306b"]  # light -> dark
    g = (df.groupby(["design", "model_alias"], observed=True)
           .agg(tokens=("total_tokens", "mean"),
                latency=("latency_s", "mean")).reset_index())
    g["design"] = g["design"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    w = 0.8 / len(designs)
    panels = [("tokens", "mean tokens per instance"),
              ("latency", "mean latency per instance (s)")]
    for ax, (col, ylab) in zip(axes, panels):
        for j, design in enumerate(designs):
            vals = [g.loc[(g.model_alias == a) & (g.design == design), col].mean()
                    for a in models]
            x = np.arange(len(models)) + (j - (len(designs) - 1) / 2) * w
            ax.bar(x, vals, width=w, color=shade[j % len(shade)], label=design,
                   edgecolor="black", linewidth=0.3, zorder=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([disp[a] for a in models], rotation=12, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(title="design", fontsize=8, ncol=len(designs))
    fig.tight_layout()
    return fig


def _fmt_usd(x: float) -> str:
    return "--" if x != x else f"\\${x * 1000:.3f}"  # x!=x catches NaN; cost per 1k instances


def build_latex_table(pm: pd.DataFrame) -> str:
    """Render tab:cost_latency with the FIXED model layout (present models
    filled, absent ones dashed). Latency in seconds, cost in USD per 1k
    instances (per-instance figures are sub-cent)."""
    by = {r.model_alias: r for r in pm.itertuples()}
    lines = [
        r"\begin{table}[!ht]", r"\centering", r"\small",
        r"\caption{Cost vs.\ latency per refactoring instance, by model "
        r"(cost in USD per 1{,}000 instances).}",
        r"\label{tab:cost_latency}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{$n$} & \textbf{Lat.\ med (s)} "
        r"& \textbf{Lat.\ mean (s)} & \textbf{Tokens} & \textbf{Cost/1k} \\",
        r"\midrule",
    ]
    for disp, alias in TABLE_MODELS:
        r = by.get(alias) if alias else None
        if r is None:
            lines.append(rf"{disp:<19}& -- & -- & -- & -- & -- \\")
        else:
            lines.append(
                rf"{disp:<19}& {int(r.n)} & {r.latency_med:.1f} "
                rf"& {r.latency_mean:.1f} & {r.tokens_mean:,.0f} "
                rf"& {_fmt_usd(r.cost_mean)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _selfcheck() -> None:
    # Use real aliases so build_latex_table fills rows: 2 priced haiku45
    # instances, 1 unpriced gpt54-nano instance (-> NaN cost).
    df = pd.DataFrame({
        "model_alias": ["haiku45", "haiku45", "gpt54-nano"],
        "design": pd.Categorical(["D1", "D1", "D2"], DESIGNS, ordered=True),
        "repetition": [1, 2, 1], "file_key": ["a", "a", "b"],
        "target_smell": ["X", "X", "Y"],
        "run_duration_seconds": [10.0, 20.0, 5.0], "agent_call_count": [1, 1, 1],
        "input_tokens": [1_000_000, 1_000_000, 500_000],
        "output_tokens": [200_000, 200_000, 100_000],
        "reasoning_tokens": [0, 0, 50_000],
        "cache_read_tokens": [0, 0, 0], "cache_write_tokens": [0, 0, 0],
        "calls": [1, 1, 1],
    })
    df["latency_s"] = df["run_duration_seconds"]
    priced = add_cost(df, pricing={"haiku45": {"in": 1.0, "out": 5.0}})
    # haiku45: 1.0*1 + 5.0*0.2 = $2.00 per instance; gpt54-nano has no entry -> NaN.
    assert abs(priced.loc[0, "cost_usd"] - 2.0) < 1e-9, priced["cost_usd"].tolist()
    assert priced.loc[2, "cost_usd"] != priced.loc[2, "cost_usd"], "unpriced should be NaN"

    # Billable tokens: cache reads stay inside input_tokens and GPT's reasoning
    # inside output_tokens; only Gemini's thinking is added on top of its output.
    rules = add_cost(pd.DataFrame({
        "model_alias": ["gpt54-nano", "gemini35-flash"],
        "input_tokens": [1_000, 1_000], "output_tokens": [100, 100],
        "reasoning_tokens": [60, 60], "cache_read_tokens": [800, 800],
        "cache_write_tokens": [0, 0],
    }))
    assert rules["billable_input_tokens"].tolist() == [1_000, 1_000], rules
    assert rules["billable_output_tokens"].tolist() == [100, 160], rules
    assert rules["total_tokens"].tolist() == [1_100, 1_160], rules
    assert abs(rules.loc[1, "cost_usd"] - (1_000 * 1.50 + 160 * 9.00) / 1e6) < 1e-12, rules

    pm = per_model(priced)
    h = pm[pm.model_alias == "haiku45"].iloc[0]
    assert h.n == 2 and h.latency_med == 15.0 and h.latency_mean == 15.0, h
    assert abs(h.cost_mean - 2.0) < 1e-9 and abs(h.cost_total - 4.0) < 1e-9, h

    pmc = per_model_design(priced)
    assert len(pmc) == 2 and set(pmc.model_alias) == {"haiku45", "gpt54-nano"}, pmc

    ob = overhead_by_design(priced)
    assert ob.loc[0, "design"] == "D1" and abs(ob.loc[0, "tokens_x"] - 1.0) < 1e-9, ob
    assert abs(ob.loc[0, "calls_x"] - 1.0) < 1e-9, ob  # D1 is its own baseline

    import matplotlib
    matplotlib.use("Agg")
    pmc_f = pmc.assign(F1=0.85)  # synthetic y for the combined-scatter smoke test
    fig = plot_combined_scatter(pmc_f, x="latency_med", y="F1", xlabel="lat", ylabel="F1",
                                placeholders=[("(placeholder)", 10, 0.8, "right")])
    assert len(fig.axes) == 1 and fig.axes[0].get_ylabel() == "F1", fig.axes

    tex = build_latex_table(pm)
    assert r"\label{tab:cost_latency}" in tex
    assert "DeepSeek V4 Flash" in tex and "Gemma 4" not in tex  # gemma dropped from layout
    assert "\\$2000.000" in tex  # $2/instance -> $2000 per 1k instances
    print("cost_latency self-check: OK")


if __name__ == "__main__":
    _selfcheck()
