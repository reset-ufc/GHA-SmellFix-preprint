"""LaTeX table for cost vs. latency per refactoring instance (tab:cost_latency).

Self-loading (Variant B): builds the per-model cost/latency aggregate straight
from ``data/agents_runs_outputs`` via :mod:`utils.cost_latency` and writes the ``.tex``.
"""
from pathlib import Path

from utils import cost_latency as cl


def cost_latency_table(
    output_path: Path | str = cl.OUTPUTS / "cost_latency.tex",
) -> str:
    """Render the RQ cost/latency table (fixed model layout, USD per 1k instances)
    and write it to ``output_path``. Returns the LaTeX string."""
    df = cl.add_cost(cl.load_cost_latency())
    pm = cl.per_model(df)
    latex = cl.build_latex_table(pm)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex


if __name__ == "__main__":
    print(cost_latency_table())
