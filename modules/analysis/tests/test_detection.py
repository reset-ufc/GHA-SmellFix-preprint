"""Tests for utils.detection — the accuracy spread over repeated runs."""

from __future__ import annotations

import pandas as pd
import pytest
from utils.detection import accuracy_spread

TRUTH = [True, True, False, False]


def _runs(model: str, repetition: int, predictions: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_alias": model,
            "design": "D1",
            "repetition": repetition,
            "gt_n15": pd.array(TRUTH, dtype="boolean"),
            "smell_detected": predictions,
        }
    )


def test_spread_is_max_minus_min_accuracy_over_repetitions():
    df = pd.concat(
        [
            _runs("m", 1, [True, True, False, False]),
            _runs("m", 2, [True, False, False, False]),
            _runs("m", 3, [False, False, True, False]),
        ]
    )
    row = accuracy_spread(df).iloc[0]
    assert (row["rep1"], row["rep2"], row["rep3"]) == (1.0, 0.75, 0.25)
    assert row["spread"] == pytest.approx(0.75)


def test_pairs_run_once_are_left_out():
    df = pd.concat(
        [
            _runs("once", 1, TRUTH),
            _runs("twice", 1, TRUTH),
            _runs("twice", 2, [not t for t in TRUTH]),
        ]
    )
    spread = accuracy_spread(df)
    assert spread["model_alias"].tolist() == ["twice"]
    assert spread["spread"].tolist() == [1.0]
