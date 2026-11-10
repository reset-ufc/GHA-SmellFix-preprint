"""Slug naming for synthetic vs real instances.

A synthetic positive's file_key (from the smell-synthesizer merge) ends with the
``##synth`` marker. The pipeline must (a) strip it to a clean stem and (b) re-tag
the slug with a trailing ``@synth`` so synthetic results/patches are distinct from
the real instance and filterable downstream.
"""

from __future__ import annotations

from gha_agents.agents import ValidSmells
from gha_agents.main import _build_slug, _is_synthetic, _stem_from_file_key

_SMELL = next(iter(ValidSmells))


def _slug(file_key: str) -> str:
    return _build_slug(
        file_key=file_key,
        agents=1,
        repetition=0,
        target_smell=_SMELL,
        model_alias=None,
    )


def test_is_synthetic_detects_marker():
    assert _is_synthetic("o@r@ci.yml##synth")
    assert not _is_synthetic("o@r@ci.yml")


def test_stem_strips_marker_and_extension():
    assert _stem_from_file_key("o@r@ci.yml##synth") == "o@r@ci"
    assert _stem_from_file_key("o@r@ci.yaml##synth") == "o@r@ci"
    # real keys unchanged
    assert _stem_from_file_key("o@r@ci.yml") == "o@r@ci"


def test_synthetic_slug_gets_trailing_synth_tag():
    real = _slug("o@r@ci.yml")
    synth = _slug("o@r@ci.yml##synth")
    smell = _SMELL.name.lower()
    assert real == f"o@r@ci@D1@rep0@{smell}"
    # same stem + a clean trailing @synth (no '.yml##synth' cruft in the middle)
    assert synth == f"o@r@ci@D1@rep0@{smell}@synth"
    assert synth.endswith("@synth")
    assert "##synth" not in synth


def test_real_slug_unchanged_by_synth_support():
    # Regression: non-synthetic naming must be byte-identical to the legacy format.
    assert _slug("owner@repo@deploy.yml") == f"owner@repo@deploy@D1@rep0@{_SMELL.name.lower()}"


# --- downstream slug parsers must read the model past the trailing @synth tag ---


def test_reconcile_slug_model_ignores_synth_tag():
    from gha_agents.reconcile import _slug_model

    # matrix slug: stem(owner@repo@file) @D1 @rep1 @smell @model [@synth]
    assert _slug_model("o@r@f@D1@rep1@smell@claudeX") == "claudeX"
    assert _slug_model("o@r@f@D1@rep1@smell@claudeX@synth") == "claudeX"  # not "synth"


def test_coverage_alias_of_slug_ignores_synth_tag():
    from pathlib import Path

    from gha_agents.coverage_report import _alias_of_slug

    assert _alias_of_slug(Path("o@r@f@D1@rep1@smell@claudeX.json")) == "claudeX"
    assert _alias_of_slug(Path("o@r@f@D1@rep1@smell@claudeX@synth.json")) == "claudeX"
