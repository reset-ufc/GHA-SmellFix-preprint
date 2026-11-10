"""LaTeX table for RQ1 smell-level detection effectiveness."""
from pathlib import Path

import pandas as pd

from utils import detection


def smell_level_detection_effectiveness_table(
    df: pd.DataFrame | Path | str | None = None,
    output_path: Path | str = Path("data/artifacts")
    / "rq1_smell_level_detection_effectiveness.tex",
) -> str:
    """Write and return the dynamic RQ1 smell-level detection table."""
    if df is not None and not isinstance(df, pd.DataFrame):
        output_path = df
        df = None
    if df is None:
        df = detection.load_results()

    latex = detection.build_smell_level_latex_table(
        df,
        label="tab:rq1_smell_level_detection_effectiveness",
        rounded=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex
