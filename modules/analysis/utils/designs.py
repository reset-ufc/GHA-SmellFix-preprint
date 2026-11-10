"""The two orchestration designs the paper compares.

The paper calls them D1, the joint design (one agent detects the target smell and, when it is
present, returns the patch in the same pass), and D2, the decomposed design (a detection agent
decides first, and a refactoring agent runs only on detected smells).

Stage 2 stores each run under its design: the ``design_d1/`` or ``design_d2/`` folder, the
``design`` field of its result JSON and trace, and the ``@D1@`` / ``@D2@`` part of its file
names. The manual-validation and review sheets use the same names.
"""
from __future__ import annotations

DESIGNS = ("D1", "D2")


def run_dir(design: str) -> str:
    """The folder that holds a design's runs: ``"D1"`` -> ``"design_d1"``."""
    if design not in DESIGNS:
        raise ValueError(f"unknown design: {design!r}")
    return f"design_{design.lower()}"
