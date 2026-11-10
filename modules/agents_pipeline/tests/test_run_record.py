"""Unit tests for ``RunRecord`` + ``CombinedResultAssembler`` (#43).

Pins the typed record's ``dump()`` shape independently of the end-to-end
characterization snapshots (#31): those prove byte-equivalence once the runners
are rewired; these pin the named constructors and the D1/D2 dump shape directly.
"""

from __future__ import annotations

import pytest
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from gha_agents.record import CombinedResultAssembler, RunRecord
from gha_agents.state import RoleTokenUsage, TokenUsage, WorkflowState
from pydantic import ValidationError

_SMELL = next(iter(ValidSmells))
_R = "Walked each job and step against the smell definition before deciding."
_YAML = "name: CI\non: [push]\n"
# A minimal workflow that parses AND conforms to the vendored GitHub Actions
# schema, so ``patch_valid`` is True / ``patch_validation_error`` is None and the
# exact-dict assertions below stay stable (no brittle jsonschema message text).
_VALID_PATCH = (
    "on: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n"
)


def _detection(
    *, smell_detected: bool, count: int, ranges: list[EvidenceRange]
) -> DetectionContract:
    return DetectionContract(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=smell_detected,
        detection_justification="found" if smell_detected else "absent",
        count=count,
        evidence_ranges=ranges,
    )


def _state_with_calls(total_calls: int, **fields) -> WorkflowState:
    state = WorkflowState(**fields)
    state.token_usage = RoleTokenUsage(total=TokenUsage(calls=total_calls))
    return state


# ─── CombinedResultAssembler ─────────────────────────────────────────────────


def test_from_d1_is_identity_passthrough():
    combined = CombinedResult(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="d",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
        patched_workflow="patched",
        refactor_justification="r",
        detection_confidence=None,
    )
    assert CombinedResultAssembler.from_d1(combined) is combined


def test_from_d2_absent_returns_original_verbatim_with_zeroed_signal():
    det = _detection(smell_detected=False, count=0, ranges=[])
    combined = CombinedResultAssembler.from_d2_absent(_SMELL, det, _YAML)
    # Equivalent to the legacy hand-built construction in teams.D2Workflow.run.
    assert combined == CombinedResult(
        reasoning=det.reasoning,
        target_smell=_SMELL,
        smell_detected=False,
        detection_justification=det.detection_justification,
        count=0,
        evidence_ranges=[],
        patched_workflow=_YAML,
    )
    assert combined.patched_workflow == _YAML
    assert combined.refactor_justification == ""


def test_from_d2_present_threads_detection_and_refactor():
    det = _detection(smell_detected=True, count=2, ranges=[EvidenceRange(start_line=3, end_line=4)])
    rep = RefactorResult(
        reasoning="ignored-by-synthesis",
        patched_workflow="PATCHED",
        refactor_justification="scoped the token",
    )
    combined = CombinedResultAssembler.from_d2_present(_SMELL, det, rep)
    assert combined == CombinedResult(
        reasoning=det.reasoning,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification=det.detection_justification,
        count=det.count,
        evidence_ranges=list(det.evidence_ranges),
        patched_workflow="PATCHED",
        refactor_justification="scoped the token",
    )
    # Detection reasoning (not the refactor agent's) is carried into the row.
    assert combined.reasoning == det.reasoning


# ─── RunRecord ───────────────────────────────────────────────────────────────


def _combined_present() -> CombinedResult:
    return CombinedResult(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="Missing timeout-minutes on the build job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        patched_workflow=_VALID_PATCH,
        refactor_justification="Added timeout-minutes.",
    )


def test_from_standard_dump_omits_all_c3_only_keys():
    combined = _combined_present()
    state = _state_with_calls(1)

    dump = RunRecord.from_standard(combined, state).dump()

    # A standard (D1/D2) run drives no feedback loops, so none of the loop /
    # per-iteration keys are padded into the result JSON.
    for key in (
        "patches_per_iteration",
        "detection_loop_converged",
        "detection_loop_iterations",
        "detection_loop_iterations_log",
        "refactor_loop_converged",
        "refactor_loop_iterations",
        "refactor_loop_iterations_log",
        "refactor_validation_agrees",
    ):
        assert key not in dump
    assert dump == {
        "patched_workflow": _VALID_PATCH,
        "patch_valid": True,
        "patch_validation_error": None,
        "smell_detected": True,
        "detection_justification": "Missing timeout-minutes on the build job.",
        "refactor_justification": "Added timeout-minutes.",
        "detection_confidence": None,
        "detection_confidence_justification": "",
        "refactor_confidence": None,
        "refactor_confidence_justification": "",
        "count": 1,
        "evidence_ranges": [{"start_line": 4, "end_line": 6}],
        "token_usage": state.token_usage.model_dump(),
        "agent_call_count": 1,
    }


def test_records_invalid_patch_as_observability_without_gating():
    # Observability (#1 "close the validation loop"): a patch that parses but
    # violates the GitHub Actions schema is recorded as patch_valid=False with the
    # error report — the record is still produced and the patch is left untouched
    # (no rejection, no retry). The gate only watches; it never gates.
    combined = CombinedResult(
        reasoning=_R,
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="d",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
        patched_workflow="name: CI\n",  # valid YAML but no `on:`/`jobs:` → schema-invalid
        refactor_justification="r",
    )

    dump = RunRecord.from_standard(combined, _state_with_calls(1)).dump()

    assert dump["patch_valid"] is False
    assert dump["patch_validation_error"] is not None
    assert "schema error" in dump["patch_validation_error"].lower()
    assert dump["patched_workflow"] == "name: CI\n"  # the patch itself is unchanged


def test_count_is_carried_but_record_does_not_score_on_it():
    # count is diagnostic-only — present in the dump, never a presence signal.
    combined = _combined_present()
    dump = RunRecord.from_standard(combined, _state_with_calls(1)).dump()
    assert dump["count"] == 1
    assert dump["smell_detected"] is True  # presence is its own field, not derived from count


def test_run_record_rejects_unknown_field():
    with pytest.raises(ValidationError):
        RunRecord(unknown_field=1)  # type: ignore[call-arg]
