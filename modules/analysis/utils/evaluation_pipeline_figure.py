# ruff: noqa: RUF001  # Unicode math glyphs in the labels are deliberate typography.
"""Method diagram of the evaluation step (RQ1 detection, RQ2 refactoring) with real cases.

Run ``python -m utils.evaluation_pipeline_figure`` to write
``data/artifacts/figures/evaluation_pipeline.{pdf,svg,png}``. The pipeline counts are the
paper's study-design constants; the cases are abridged from ``data/artifacts/constructs/
yaml_diffs/<stem>.diff`` with verdicts and notes from ``data/validation/
manual_validation_final.csv``, labelled by the RQ2 table classifiers.
"""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.textpath import TextToPath

from utils.dataloader import ROOT

OUT = ROOT / "data" / "artifacts" / "figures"

W = 6.75  # inches; figure* width of a two-column paper
H_PIPE, H_CASES = 3.45, 1.64

INK, SUB, LINE = "#1F2328", "#59636E", "#6E7781"
EDGE, LANE, MUTED = "#C4CAD1", "#F4F6F8", "#E9ECEF"
# The only accent: the RQ1 verdict that restricts RQ2's conditional denominator.
ACCENT = "#0072B2"
# Okabe-Ito severity scale shared by both outcome lists and the cases; every use is labelled.
GOOD, MILD, SEVERE, NONE, COSMETIC = "#009E73", "#E69F00", "#D55E00", "#4D4D4D", "#A0A7AF"
ADD_BG, DEL_BG, CODE_BG = "#E6FFEC", "#FFEBE9", "#F6F8FA"

TITLE_PT, DETAIL_PT = 7.0, 6.2

# Abridged: nesting shown with one-space indents and long values cut with "…"; every kept
# token is verbatim.
CASES = [
    {
        "stem": "0xpolygonmiden@miden-vm@changelog@D2@rep1@missingconcurrencypr@gemini35-flash",
        "outcome": "Clean Removal",
        "color": GOOD,
        "pair": "positive",
        "repo": "0xpolygonmiden/miden-vm",
        "meta": "MissingConcurrencyPR · Gemini 3.5 Flash",
        "diff": [
            (" ", " contents: read"),
            ("+", "concurrency:"),
            ("+", " group: ${{ github.workflow }}…"),
            ("+", " cancel-in-progress: true"),
        ],
        "constructs": "concurrency.{group, cancel-in-progress}",
        "note": "Everything correct",
    },
    {
        "stem": "balena-io@open-balena@flowzone@D1@rep1@missingconcurrencypr@deepseek-v4-flash",
        "outcome": "Removal w/ Unexpected Changes",
        "color": MILD,
        "pair": "positive",
        "repo": "balena-io/open-balena",
        "meta": "MissingConcurrencyPR · DeepSeek V4 Flash",
        "diff": [
            ("-", ' id-token: "write"  # AWS…'),
            ("+", ' id-token: "write"'),
            ("+", "concurrency: …"),
            ("-", "# prevent duplicate workflow…"),
        ],
        "constructs": "concurrency.{group, cancel-in-progress}",
        "note": "Removes some comments unnecessarily",
    },
    {
        "stem": "elastic@beats@xpack-auditbeat-macos-unit-tests@D1@rep1@unpinnedactionversion"
        "@gpt54-nano",
        "outcome": "Potential Behavior-Impacting Change",
        "color": SEVERE,
        "pair": "positive",
        "repo": "elastic/beats",
        "meta": "UnpinnedActionVersion · GPT-5.4-nano",
        "diff": [
            (" ", "runs-on: ${{ matrix.os }}"),
            (" ", "steps:"),
            ("-", " uses: actions/checkout@v6"),
            ("+", " uses: actions/checkout@9a0f5b…"),
        ],
        "constructs": "jobs.<job_id>.steps[*].uses",
        "note": "hash may be invalid and may not present the correct version",
    },
    {
        "stem": "pytorch@rl@test-windows-optdepts@D1@rep1@missingjobtimeout@deepseek-v4-flash",
        "outcome": "Localized Unnecessary Edit",
        "color": MILD,
        "pair": "negative",
        "repo": "pytorch/rl",
        "meta": "MissingJobTimeout · DeepSeek V4 Flash",
        "diff": [
            (" ", "unittests-cpu:"),
            ("+", " timeout-minutes: 60"),
            (" ", " strategy: …"),
            (" ", " uses: …/windows_job.yml@main"),
        ],
        "constructs": "jobs.<job_id>.timeout-minutes",
        "note": '…without "steps" prevents the workflow from being executed',
    },
]


def _lines_in(text, pt):
    return (text.count("\n") + 1) * pt * 1.22 / 72


def shrink_to_width(text, width):
    """Scale a text's font down until it spans at most ``width`` inches.

    The span uses unhinted font metrics: they scale with the font size, and the PDF and SVG
    outputs use them too.
    """
    span = (
        TextToPath().get_text_width_height_descent(
            text.get_text(), text.get_fontproperties(), ismath=False
        )[0]
        / 72
    )
    if span > width:
        text.set_fontsize(text.get_fontsize() * width / span)


def rounded(ax, x, y, w, h, *, fc, ec=EDGE, lw=0.6, r=0.045, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z
        )
    )


def node(ax, x, y, w, h, title, detail="", *, fill="white", code=""):
    """Rounded box with a bold title over a grey detail block, centered as one unit."""
    rounded(ax, x, y, w, h, fc=fill)
    gap = 0.035
    parts = [(title, TITLE_PT, {"weight": "bold", "color": INK})]
    if detail:
        parts.append((detail, DETAIL_PT, {"color": SUB}))
    if code:
        parts.append((code, 5.6, {"color": SUB, "family": "monospace"}))
    total = sum(_lines_in(t, pt) for t, pt, _ in parts) + gap * (len(parts) - 1)
    top = y + h / 2 + total / 2
    for text, pt, style in parts:
        ax.text(
            x + w / 2,
            top,
            text,
            ha="center",
            va="top",
            size=pt,
            zorder=3,
            linespacing=1.22,
            **style,
        )
        top -= _lines_in(text, pt) + gap


def chip(ax, x, y, w, text, color, h=0.16):
    rounded(ax, x, y, w, h, fc=color + "1F", lw=0.4, r=0.03)
    ax.add_patch(Rectangle((x, y), 0.045, h, fc=color, ec="none", zorder=3))
    ax.text(x + 0.1, y + h / 2, text, ha="left", va="center", size=6.3, color=INK, zorder=4)


def header(ax, x, y, w, label, count, h=0.18):
    rounded(ax, x, y, w, h, fc="white", r=0.03)
    ax.text(
        x + 0.08,
        y + h / 2,
        label,
        ha="left",
        va="center",
        size=TITLE_PT,
        weight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        x + w - 0.08,
        y + h / 2,
        count,
        ha="right",
        va="center",
        size=DETAIL_PT,
        color=SUB,
        zorder=3,
    )


def arrow(ax, start, end, *, color=LINE):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>,head_length=3.2,head_width=1.7",
            lw=0.7,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def path(ax, xs, ys, *, color=LINE, ls="-"):
    ax.plot(xs, ys, color=color, lw=0.7, ls=ls, zorder=4, solid_capstyle="butt")


def lane(ax, x, y, w, h, badge, title, note=""):
    rounded(ax, x, y, w, h, fc=LANE, ec="none", r=0.07, z=1)
    by = y + h - 0.12
    b = ax.text(
        x + 0.1,
        by,
        f" {badge} ",
        ha="left",
        va="center",
        size=6.4,
        weight="bold",
        color="white",
        zorder=3,
        bbox={"boxstyle": "round,pad=0.25,rounding_size=0.5", "fc": INK, "ec": "none"},
    )
    t = ax.annotate(
        title,
        xycoords=b,
        xy=(1, 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        size=7.6,
        weight="bold",
        color=INK,
        zorder=3,
    )
    if note:
        ax.annotate(
            note,
            xycoords=t,
            xy=(1, 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            size=6.0,
            color=SUB,
        )


def column_head(ax, x, y, text):
    ax.text(x, y, text.upper(), ha="left", va="center", size=5.6, weight="bold", color=SUB)


def draw_pipeline(ax):
    # Columns shared by both lanes, so boxes align vertically across RQ1 and RQ2.
    in_x, in_w = 0.04, 1.08
    c1_x, c1_w = 1.42, 1.62
    c2_x, c2_w = 3.3, 1.95
    c3_x, c3_w = 5.45, 1.14
    route_x = 6.66

    lane(ax, 1.3, 2.58, 5.41, 0.84, "RQ1", "Detection analysis")
    r1_y, r1_h = 2.66, 0.52
    r1_mid = r1_y + r1_h / 2
    node(
        ax,
        c1_x,
        r1_y,
        c1_w,
        r1_h,
        "Verdict vs. ground truth",
        "present / absent per (workflow, smell)\npositive + negative pairs",
    )
    node(ax, 3.3, r1_y, 1.3, r1_h, "Effectiveness", "Precision · Recall\nF1 · Accuracy")
    node(
        ax,
        4.86,
        r1_y,
        1.73,
        r1_h,
        "Stability ($k = 3$ runs)",
        "GPT-5.4-nano, Claude Haiku 4.5\naccuracy spread = max − min\nper (model, design)",
    )
    arrow(ax, (c1_x + c1_w, r1_mid), (3.3, r1_mid))
    arrow(ax, (4.6, r1_mid), (4.86, r1_mid))

    node(
        ax,
        in_x,
        r1_y,
        in_w,
        r1_h,
        "Ground truth",
        "(workflow, smell)\npresent / absent",
        fill=MUTED,
    )
    node(
        ax,
        in_x,
        1.0,
        in_w,
        0.98,
        "Pipeline\noutputs",
        "4 models × 2 designs\n× 570 pairs\n= 4,560 outputs",
        fill=MUTED,
    )
    arrow(ax, (in_x + in_w, r1_mid), (c1_x, r1_mid))
    fork_x, out_y = 1.22, 1.75
    path(ax, [in_x + in_w, fork_x, fork_x], [out_y, out_y, 2.8])
    arrow(ax, (fork_x, 2.8), (c1_x, 2.8))
    arrow(ax, (fork_x, out_y), (c1_x, out_y))
    ax.plot(fork_x, out_y, "o", ms=2.2, color=LINE, zorder=5)

    lane(ax, 1.3, 0.04, 5.41, 2.36, "RQ2", "Refactoring assessment")
    head_y = 2.09
    column_head(ax, c1_x, head_y, "Automated checks")
    column_head(ax, c2_x, head_y, "Blind manual annotation")
    column_head(ax, c3_x, head_y, "Reported rates")

    mid_x = c1_x + c1_w / 2
    node(ax, c1_x, 1.52, c1_w, 0.46, "Workflow validity", "YAML parse + GHA structure")
    node(
        ax,
        c1_x,
        0.9,
        c1_w,
        0.46,
        "Original vs. refactored diff",
        "per (workflow, smell, model, design)",
    )
    node(
        ax,
        c1_x,
        0.12,
        c1_w,
        0.62,
        "Construct extraction",
        "YAML paths → 117 of 197 constructs\n(Bardsiri et al.) vs. expected scope",
        code="steps[0].uses → steps[*].uses",
    )
    arrow(ax, (mid_x, 1.52), (mid_x, 1.36))
    arrow(ax, (mid_x, 0.9), (mid_x, 0.74))

    # A bracket collects the automated evidence the annotators work from.
    br_x = 3.17
    path(ax, [br_x, br_x], [0.43, 1.89])
    for y in (1.75, 1.13, 0.43):
        path(ax, [c1_x + c1_w, br_x], [y, y])

    header(ax, c2_x, 1.80, c2_w, "Positive pairs", "2,280 outputs")
    for i, (label, color) in enumerate(
        [
            ("Clean Removal", GOOD),
            ("Removal w/ Unexpected Changes", MILD),
            ("Potential Behavior-Impacting Change", SEVERE),
            ("Failed or No Patch (incl. invalid)", NONE),
        ]
    ):
        chip(ax, c2_x + 0.1, 1.6 - i * 0.19, c2_w - 0.1, label, color)
    header(ax, c2_x, 0.75, c2_w, "Negative pairs", "2,280 · if modified")
    for i, (label, color) in enumerate(
        [
            ("Cosmetic Edit", COSMETIC),
            ("Localized Unnecessary Edit", MILD),
            ("Behavior-Relevant Unnecessary Edit", SEVERE),
        ]
    ):
        chip(ax, c2_x + 0.1, 0.55 - i * 0.19, c2_w - 0.1, label, color)
    arrow(ax, (br_x, 1.89), (c2_x, 1.89))
    arrow(ax, (br_x, 0.84), (c2_x, 0.84))

    # Positive outcomes feed both rates.
    e2e_y, cond_y = 1.46, 0.88
    gb_x = c2_x + c2_w + 0.09
    path(ax, [c2_x + c2_w + 0.03, gb_x, gb_x, c2_x + c2_w + 0.03], [1.76, 1.76, 1.03, 1.03])
    arrow(ax, (gb_x, e2e_y + 0.24), (c3_x, e2e_y + 0.24))
    arrow(ax, (gb_x, cond_y + 0.24), (c3_x, cond_y + 0.24))
    node(ax, c3_x, e2e_y, c3_w, 0.48, "End-to-end", "all ground-truth\npositives")
    node(ax, c3_x, cond_y, c3_w, 0.48, "Conditional", "positives detected\n$\\it{present}$")
    node(
        ax,
        c3_x,
        0.12,
        c3_w,
        0.62,
        "Reliability",
        "$\\kappa = 0.758$ pre-consensus\n3rd-author adjudication\nboth uncertain →\n"
        "not assessable",
        fill=MUTED,
    )

    # Accent link: RQ1's verdict defines RQ2's conditional denominator.
    gap_y, cond_mid = 2.49, cond_y + 0.24
    path(
        ax,
        [2.75, 2.75, route_x, route_x],
        [r1_y, gap_y, gap_y, cond_mid],
        color=ACCENT,
        ls=(0, (3, 1.6)),
    )
    arrow(ax, (route_x, cond_mid), (c3_x + c3_w, cond_mid), color=ACCENT)
    ax.text(
        4.7,
        gap_y,
        "verdict = present restricts the conditional denominator",
        ha="center",
        va="center",
        size=5.8,
        style="italic",
        color=ACCENT,
        zorder=5,
        bbox={"fc": "white", "ec": "none", "pad": 0.6},
    )


def case_card(ax, x, top, w, case, h):
    """One real output: outcome strip, provenance, abridged diff, constructs, annotator note."""
    code_pt, row = 5.0, 0.08
    rounded(ax, x, top - h, w, h, fc="white")
    color = case["color"]
    y = top - 0.2
    ax.add_patch(Rectangle((x + 0.001, y), w - 0.002, 0.2, fc=color + "26", ec="none", zorder=2.5))
    ax.add_patch(Rectangle((x + 0.001, y), 0.045, 0.2, fc=color, ec="none", zorder=3))
    outcome = ax.text(
        x + 0.1,
        y + 0.1,
        case["outcome"],
        ha="left",
        va="center",
        size=6.3,
        weight="bold",
        color=INK,
        zorder=4,
    )
    shrink_to_width(outcome, w - 0.155)
    ax.text(
        x + 0.08,
        y - 0.08,
        case["repo"],
        ha="left",
        va="center",
        size=6.0,
        weight="bold",
        color=INK,
        zorder=4,
    )
    smell, model = case["meta"].split(" · ")
    gt = "present" if case["pair"] == "positive" else "absent"
    ax.text(x + 0.08, y - 0.175, smell, ha="left", va="center", size=5.6, color=SUB, zorder=4)
    ax.text(
        x + 0.08,
        y - 0.265,
        f"{model} · smell {gt}",
        ha="left",
        va="center",
        size=5.6,
        color=SUB,
        zorder=4,
    )

    cx, cw = x + 0.05, w - 0.1
    diff_top = y - 0.35
    diff_h = row * len(case["diff"]) + 0.04
    ax.add_patch(Rectangle((cx, diff_top - diff_h), cw, diff_h, fc=CODE_BG, ec="none", zorder=2.5))
    for i, (sign, text) in enumerate(case["diff"]):
        ly = diff_top - 0.02 - (i + 0.5) * row
        if sign != " ":
            ax.add_patch(
                Rectangle(
                    (cx, ly - row / 2),
                    cw,
                    row,
                    fc=ADD_BG if sign == "+" else DEL_BG,
                    ec="none",
                    zorder=2.6,
                )
            )
        ax.text(
            cx + 0.03,
            ly,
            f"{sign} {text}",
            ha="left",
            va="center",
            size=code_pt,
            family="monospace",
            color=INK if sign != " " else SUB,
            zorder=4,
        )

    ax.text(
        x + 0.08,
        diff_top - diff_h - 0.08,
        f"Δ {case['constructs']}",
        ha="left",
        va="center",
        size=5.4,
        color=INK,
        zorder=4,
    )
    ax.text(
        x + 0.08,
        diff_top - diff_h - 0.16,
        textwrap.fill(f"“{case['note']}”", 40),
        ha="left",
        va="top",
        size=5.4,
        style="italic",
        color=SUB,
        zorder=4,
        linespacing=1.22,
    )


def draw_cases(ax):
    lane(
        ax,
        0.04,
        0.02,
        6.67,
        H_CASES - 0.06,
        "Cases",
        "Real outputs from the study",
        note="abridged diffs · annotator notes quoted verbatim",
    )
    gap, x0 = 0.08, 0.12
    w = (6.67 - 2 * 0.08 - 3 * gap) / 4
    for i, case in enumerate(CASES):
        case_card(ax, x0 + i * (w + gap), H_CASES - 0.25, w, case, h=1.27)


def draw():
    fig = plt.figure(figsize=(W, H_PIPE + H_CASES))
    frac = H_CASES / (H_PIPE + H_CASES)
    pipe = fig.add_axes((0, frac, 1, 1 - frac))
    cases = fig.add_axes((0, 0, 1, frac))
    for ax, h in ((pipe, H_PIPE), (cases, H_CASES)):
        ax.set(xlim=(0, W), ylim=(0, h))
        ax.axis("off")
    draw_pipeline(pipe)
    draw_cases(cases)
    return fig


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
            "font.monospace": ["Fira Mono", "DejaVu Sans Mono"],
            "mathtext.fontset": "stixsans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = draw()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"evaluation_pipeline.{ext}", dpi=300, facecolor="white")
    print(OUT / "evaluation_pipeline.pdf")


if __name__ == "__main__":
    main()
