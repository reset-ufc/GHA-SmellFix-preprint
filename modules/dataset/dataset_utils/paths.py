from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("study_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").is_dir():
            return candidate
    msg = (
        "Could not locate replication-package root "
        "(no pyproject.toml + modules/ found upwards from src)."
    )
    raise RuntimeError(msg)


def data_dir() -> Path:
    env = os.environ.get("study_DATA_DIR")
    if env:
        return Path(env).resolve()
    return repo_root() / "data"
