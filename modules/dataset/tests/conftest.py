"""Make ``dataset_utils`` importable when running pytest from the dataset module
without an installed wheel (mirrors the notebooks' __file__ sys.path hack)."""

import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))
