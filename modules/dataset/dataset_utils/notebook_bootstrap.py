"""Idempotent notebook bootstrap helper.

Call ``setup()`` at the top of any notebook. Safe in:

* IPython kernels (Jupyter, JupyterLab, Antigravity, VS Code)
* Plain ``python`` interpreter (when notebook is converted to a script
  via ``jupyter nbconvert --to script``)
* Linux, macOS, Windows

What it does
------------
1. ``chdir`` to the repository root so ``Path("data/X")`` resolves
   regardless of where the kernel was launched from.
2. Reset ``sys.argv`` to ``["notebook"]`` so the script's ``argparse``
   uses its own defaults instead of Jupyter's CLI flags.

Both operations are intrinsically idempotent — calling ``setup()`` more
than once is a no-op.
"""

from __future__ import annotations

import os
import sys

from dataset_utils.paths import repo_root


def setup() -> None:
    os.chdir(repo_root())
    sys.argv = ["notebook"]
