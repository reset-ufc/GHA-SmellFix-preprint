"""Load the raw agent-run result JSONs into one pandas DataFrame.

Each file under ``data/agents_runs_outputs/<output-dir>/design_d*/results/*.json`` is one
detection/refactor instance. We keep **every** field from the JSON (nested ones
like ``token_usage`` / ``model_ids`` stay as dict-valued columns) plus two
provenance columns: ``file`` (filename) and ``source_dir`` (the per-model
output dir). The ``model_alias`` field already tags each row by model, so a
single combined frame covers every model — filter on ``model_alias`` for one.
The ``design`` field names each run's design (D1/D2); see :mod:`utils.designs`.

Run ``python dataloader.py`` for a self-check.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Workspace root = nearest ancestor holding both pyproject.toml and modules/
# (each member also has a pyproject.toml, so "nearest pyproject" is not enough).
# The agent runs live under data/agents_runs_outputs/ unless STUDY_DATA_ROOT
# points elsewhere.
ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "modules").is_dir()
)
DATA_ROOT = Path(os.environ.get("STUDY_DATA_ROOT") or ROOT / "data" / "agents_runs_outputs")

# One row per (model, design, rep, workflow, smell). A cell can appear in two
# output dirs once gap-fill/top-up runs are merged next to the originals; dedup
# so counts aren't double-fed (no-op when dirs don't overlap).
_DEDUP_KEYS = ["model_alias", "design", "repetition", "file_key", "target_smell"]

# Gemma 4 is not comparable on the shared n15 ground truth, so it is dropped
# study-wide at the loader (per the README: "excluded from the study loaders").
EXCLUDE_MODELS = ("gemma4",)


def load_results(data_root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Every result JSON under ``data_root`` as one row, all fields kept.

    Unparseable files are skipped and listed in ``df.attrs['unparseable']``.
    Rows are deduped on the natural key and sorted by filename for determinism; models in
    :data:`EXCLUDE_MODELS` (Gemma 4) are dropped.
    """
    data_root = Path(data_root)
    rows, bad = [], []
    for f in sorted(data_root.glob("*/design_d*/results/*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bad.append(str(f.relative_to(data_root)))
            continue
        rows.append(d | {"file": f.name, "source_dir": f.relative_to(data_root).parts[0]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=_DEDUP_KEYS, keep="first").reset_index(drop=True)
        df = df[~df["model_alias"].isin(EXCLUDE_MODELS)].reset_index(drop=True)
    df.attrs["unparseable"] = bad
    return df
