"""Tests for the new ground_truth schema (present_smells / absent_smells).

Covers the two behaviours the schema migration introduced:

* ``run_pipeline`` derives its evaluation pairs from ``present_smells`` (positive)
  and ``absent_smells`` (negative) — never the full catalog — and threads each
  pair's polarity through as ``ground_truth_positive`` (Form 1: inherited from the
  origin list, not re-derived).
* ``scripts/sample_smoke_ground_truth.py`` produces a deterministic, balanced
  K-per-smell-per-polarity slice whose entries carry only the sampled targets.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from gha_agents import main
from gha_agents.agents import ValidSmells
from gha_agents.env import persist_traces

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_sampler():
    spec = importlib.util.spec_from_file_location(
        "sample_smoke_ground_truth", _SCRIPTS / "sample_smoke_ground_truth.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _entry(file_key: str, present: list[str], absent: list[str]) -> dict:
    return {
        "file_key": file_key,
        "yaml_content": "name: CI\non: [push]\n",
        "repository": "o/r",
        "file_path": ".github/workflows/ci.yml",
        "default_branch": "main",
        "present_smells": [{"pascal": p, "count": 1} for p in present],
        "absent_smells": [{"pascal": a} for a in absent],
    }


# ─── Pair enumeration + polarity inheritance ──────────────────────────────────


def test_run_pipeline_enumerates_present_positive_and_absent_negative(monkeypatch, tmp_path):
    """present_smells → (positive) pairs, absent_smells → (negative) pairs; a file
    with neither contributes nothing; ground_truth_positive is the inherited polarity."""
    captured: list[tuple[str, str, bool]] = []

    def fake_process_entry(*, entry, target_pascal, ground_truth_positive, **_kwargs):
        captured.append((entry["file_key"], target_pascal, ground_truth_positive))
        return {"file": entry["file_key"], "smells": "0", "solved": "—", "status": "OK"}

    monkeypatch.setattr(main, "_process_entry", fake_process_entry)

    entries = [
        _entry("a", present=["P1", "P2"], absent=["N1"]),
        _entry("b", present=[], absent=["N2"]),
        _entry("c", present=[], absent=[]),  # neither list → no pairs
    ]
    main.run_pipeline(
        entries=entries,
        agents=1,
        repetition=1,
        max_iterations=3,
        output_dir=tmp_path / "out",
        error_log=tmp_path / "error.log",
    )

    assert ("a", "P1", True) in captured
    assert ("a", "P2", True) in captured
    assert ("a", "N1", False) in captured
    assert ("b", "N2", False) in captured
    assert all(file_key != "c" for file_key, _, _ in captured)
    assert len(captured) == 4


def test_process_entry_persists_inherited_ground_truth_positive(monkeypatch, tmp_path):
    """The success path of _process_entry writes the INHERITED ground_truth polarity
    verbatim to the result JSON under the ``ground_truth`` key (Form 1 — not
    re-derived) for BOTH polarities, and emits no smell_paper_id. Guards the
    realized persisted field that drives every downstream P/R/F1 metric, which
    the enumeration test (monkeypatching _process_entry away) cannot reach."""
    smell = next(iter(ValidSmells))
    entry = _entry("owner@repo@ci.yml", present=[smell.name], absent=[])
    fake_state = SimpleNamespace(trace_calls=[], token_usage=None)
    canned = {
        "smell_detected": True,
        "agent_call_count": 1,
        "patched_workflow": entry["yaml_content"],
        "patches_per_iteration": [],
        "detection_loop_converged": True,
        "refactor_loop_converged": True,
        "refactor_validation_agrees": None,
    }
    monkeypatch.setattr(main, "_run_for_smell", lambda **_kw: (dict(canned), fake_state))
    monkeypatch.setattr(main, "_build_result_metadata", lambda **_kw: {})
    monkeypatch.setattr(main, "_prompt_hashes_for", lambda *_a, **_k: {})

    # ground_truth_positive=False is the adversarial case: the smell IS in present_smells,
    # so a membership re-derivation (the forbidden Form 2) would wrongly yield True.
    for positive in (True, False):
        out = tmp_path / f"out-{positive}"
        with persist_traces(False):
            main._process_entry(
                entry=entry,
                target_pascal=smell.name,
                ground_truth_positive=positive,
                agents=1,
                repetition=1,
                max_iterations=3,
                error_log=tmp_path / "error.log",
                output_dir=out,
                model_alias=None,
                resume=False,
            )
        slug = main._build_slug(
            file_key=entry["file_key"],
            agents=1,
            repetition=1,
            target_smell=smell,
            model_alias=None,
        )
        saved = json.loads((out / "results" / f"{slug}.json").read_text(encoding="utf-8"))
        assert saved["ground_truth"] is positive
        assert "smell_paper_id" not in saved


def test_run_pipeline_skips_files_with_no_targets(monkeypatch, tmp_path):
    called = False

    def fake_process_entry(**_kwargs):
        nonlocal called
        called = True
        return {"file": "x", "smells": "0", "solved": "—", "status": "OK"}

    monkeypatch.setattr(main, "_process_entry", fake_process_entry)
    main.run_pipeline(
        entries=[_entry("only", present=[], absent=[])],
        agents=1,
        repetition=1,
        max_iterations=3,
        output_dir=tmp_path / "out",
        error_log=tmp_path / "error.log",
    )
    assert called is False


# ─── Smoke sampler ────────────────────────────────────────────────────────────


def test_sampler_balances_k_per_smell_per_polarity():
    sampler = _load_sampler()
    entries = [_entry(f"pos{i}", ["S1"], []) for i in range(5)]
    entries += [_entry(f"neg{i}", [], ["S1"]) for i in range(5)]
    smoke = sampler.sample_smoke_ground_truth(entries, per_smell=3)
    pos = sum(len(e["present_smells"]) for e in smoke)
    neg = sum(len(e["absent_smells"]) for e in smoke)
    assert pos == 3
    assert neg == 3


def test_sampler_is_deterministic_regardless_of_input_order():
    sampler = _load_sampler()
    entries = [_entry(f"e{i}", ["S1"], ["S2"]) for i in range(10)]
    forward = sampler.sample_smoke_ground_truth(entries, per_smell=3)
    reverse = sampler.sample_smoke_ground_truth(list(reversed(entries)), per_smell=3)
    assert forward == reverse


def test_sampler_filters_unsampled_targets_off_a_kept_entry():
    """An entry kept for one smell drops a present smell that was capped out by
    earlier candidates — only the sampled targets survive on the entry."""
    sampler = _load_sampler()
    entries = [
        _entry("e0", ["S2"], []),
        _entry("e1", ["S2"], []),
        _entry("e2", ["S1", "S2"], []),  # S2 capped (e0,e1 win); kept only for S1
    ]
    smoke = sampler.sample_smoke_ground_truth(entries, per_smell=2)
    e2 = next(e for e in smoke if e["file_key"] == "e2")
    assert {s["pascal"] for s in e2["present_smells"]} == {"S1"}
