"""Unit tests for gha_agents.state — Pydantic models for teams state."""

from types import SimpleNamespace

import pytest
from gha_agents.agents import (
    DetectionContract,
    EvidenceRange,
    RefactorValidationResult,
    ValidSmells,
)
from gha_agents.state import (
    CallMetrics,
    DetectionIterationLogEntry,
    DetectionPhaseEntry,
    DetectionValidationPhaseEntry,
    RefactorIterationLogEntry,
    RefactorPhaseEntry,
    RefactorValidationPhaseEntry,
    RoleTokenUsage,
    TokenUsage,
    WorkflowState,
)
from pydantic import ValidationError

_SMELL = next(iter(ValidSmells))
_ROLES = ("detection", "detection_validation", "refactor", "refactor_validation", "combined")


# ─── TokenUsage ──────────────────────────────────────────────────────────────


def test_token_usage_default_zeroed():
    tu = TokenUsage()
    assert tu.input_tokens == 0
    assert tu.output_tokens == 0
    assert tu.cache_read_tokens == 0
    assert tu.cache_write_tokens == 0
    assert tu.calls == 0


def test_token_usage_dump_keys():
    tu = TokenUsage(
        input_tokens=10,
        output_tokens=20,
        reasoning_tokens=5,
        cache_read_tokens=30,
        cache_write_tokens=40,
        calls=2,
    )
    assert tu.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "cache_read_tokens": 30,
        "cache_write_tokens": 40,
        "calls": 2,
    }


def test_token_usage_roundtrip_json():
    tu = TokenUsage(input_tokens=5, output_tokens=7, calls=1)
    payload = tu.model_dump_json()
    parsed = TokenUsage.model_validate_json(payload)
    assert parsed == tu


# ─── CallMetrics ─────────────────────────────────────────────────────────────


def test_call_metrics_default_zeroed():
    cm = CallMetrics()
    assert cm.input_tokens == 0
    assert cm.output_tokens == 0
    assert cm.reasoning_tokens == 0
    assert cm.cache_read_tokens == 0
    assert cm.cache_write_tokens == 0
    assert cm.total_tokens == 0
    assert cm.time_to_first_token is None
    assert cm.duration_ms == 0


def test_call_metrics_from_agno_response_extracts_provider_fields():
    metrics = SimpleNamespace(
        input_tokens=1200,
        output_tokens=340,
        reasoning_tokens=180,
        cache_read_tokens=900,
        cache_write_tokens=0,
        total_tokens=1540,
        time_to_first_token=0.83,
    )
    cm = CallMetrics.from_agno_response(SimpleNamespace(metrics=metrics), duration_ms=2100)
    assert cm.input_tokens == 1200
    assert cm.output_tokens == 340
    assert cm.reasoning_tokens == 180
    assert cm.cache_read_tokens == 900
    assert cm.total_tokens == 1540
    assert cm.time_to_first_token == 0.83
    assert cm.duration_ms == 2100


def test_call_metrics_from_agno_response_tolerates_missing_metrics():
    cm = CallMetrics.from_agno_response(SimpleNamespace(metrics=None), duration_ms=50)
    assert cm.input_tokens == 0
    assert cm.time_to_first_token is None
    assert cm.duration_ms == 50


# ─── RoleTokenUsage ──────────────────────────────────────────────────────────


def test_role_token_usage_default_has_all_roles_zeroed():
    rtu = RoleTokenUsage()
    dumped = rtu.model_dump()
    assert dumped["total"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "calls": 0,
    }
    # No role has run yet — by_role serializes empty rather than five idle buckets,
    # so the result JSON can't be misread as four agents that never made a call.
    assert dumped["by_role"] == {}


def test_role_token_usage_mutation_via_attribute_access():
    rtu = RoleTokenUsage()
    rtu.by_role["detection"].input_tokens += 100
    rtu.by_role["detection"].calls += 1
    rtu.total.input_tokens += 100
    rtu.total.calls += 1
    dumped = rtu.model_dump()
    assert dumped["by_role"]["detection"]["input_tokens"] == 100
    assert dumped["by_role"]["detection"]["calls"] == 1
    assert dumped["total"]["input_tokens"] == 100
    assert dumped["total"]["calls"] == 1


def test_role_token_usage_roundtrip_json():
    rtu = RoleTokenUsage()
    rtu.by_role["refactor"].output_tokens = 42
    rtu.by_role["refactor"].calls = 1
    rtu.total.output_tokens = 42
    rtu.total.calls = 1
    payload = rtu.model_dump_json()
    parsed = RoleTokenUsage.model_validate_json(payload)
    assert parsed.by_role["refactor"].output_tokens == 42
    assert parsed.total.output_tokens == 42
    # Only the role that ran round-trips; the four idle roles are pruned on dump.
    assert set(parsed.by_role.keys()) == {"refactor"}


# ─── IterationLogEntry variants ──────────────────────────────────────────────


def test_detection_iteration_log_entry_nested_schema_dump():
    entry = DetectionIterationLogEntry(
        iteration=1,
        detection=DetectionPhaseEntry(
            smell_detected=True,
            detection_justification="found at L4-L7",
            metrics=CallMetrics(input_tokens=100, output_tokens=20, duration_ms=1234),
        ),
        validation=DetectionValidationPhaseEntry(
            agrees=False,
            validated_smell_detected=True,
            disagreement_feedback="line range looks off",
            metrics=CallMetrics(input_tokens=120, output_tokens=15, duration_ms=560),
        ),
    )
    dumped = entry.model_dump()
    assert dumped["iteration"] == 1
    assert dumped["detection"]["detection_justification"].startswith("found")
    assert dumped["detection"]["metrics"]["input_tokens"] == 100
    assert dumped["detection"]["metrics"]["duration_ms"] == 1234
    assert dumped["validation"]["agrees"] is False
    assert dumped["validation"]["disagreement_feedback"] == "line range looks off"
    assert dumped["validation"]["metrics"]["output_tokens"] == 15


def test_detection_validation_phase_disagreement_feedback_optional():
    entry = DetectionValidationPhaseEntry(
        agrees=True, validated_smell_detected=True
    )
    dumped = entry.model_dump()
    assert dumped["disagreement_feedback"] is None
    assert dumped["metrics"]["input_tokens"] == 0


def test_detection_phase_entry_carries_evidence_range_and_raw_response():
    entry = DetectionPhaseEntry(
        smell_detected=True,
        detection_justification="found at L4-L7",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=7)],
        raw_response='{"smell_detected": true, "count": 1}',
    )
    dumped = entry.model_dump()
    assert dumped["count"] == 1
    assert dumped["evidence_ranges"] == [{"start_line": 4, "end_line": 7}]
    assert dumped["raw_response"].startswith("{")


def test_detection_phase_entry_evidence_and_raw_response_default_empty():
    entry = DetectionPhaseEntry(smell_detected=False, detection_justification="absent")
    assert entry.count == 0
    assert entry.evidence_ranges == []
    assert entry.raw_response == ""
    assert entry.reasoning_content is None


def test_phase_entries_carry_provider_reasoning_content():
    # Provider's native reasoning/thinking stream is stored per call, beside the
    # structured raw_response, on both the primary and refactor phase entries.
    det = DetectionPhaseEntry(
        smell_detected=True, detection_justification="found", reasoning_content="A1 thinking"
    )
    assert det.reasoning_content == "A1 thinking"
    rep = RefactorPhaseEntry(
        patched_chars=1, patched_sha256="c" * 64, reasoning_content="A3 thinking"
    )
    assert rep.reasoning_content == "A3 thinking"


def test_refactor_iteration_log_entry_nested_schema_dump():
    entry = RefactorIterationLogEntry(
        iteration=2,
        refactor=RefactorPhaseEntry(
            patched_chars=512,
            patched_sha256="a" * 64,
            metrics=CallMetrics(input_tokens=300, output_tokens=80, duration_ms=2345),
        ),
        validation=RefactorValidationPhaseEntry(
            agrees=True,
            remaining_count=0,
            disagreement_feedback=None,
            metrics=CallMetrics(input_tokens=200, output_tokens=40, duration_ms=678),
        ),
    )
    dumped = entry.model_dump()
    assert dumped["iteration"] == 2
    assert dumped["refactor"]["patched_chars"] == 512
    assert dumped["refactor"]["patched_sha256"] == "a" * 64
    assert dumped["refactor"]["metrics"]["duration_ms"] == 2345
    assert dumped["validation"]["agrees"] is True
    assert dumped["validation"]["disagreement_feedback"] is None


# ─── WorkflowState ───────────────────────────────────────────────────────────


def test_workflow_state_default_shape():
    state = WorkflowState()
    dumped = state.model_dump()
    # Parity with the fresh WorkflowState each run starts with.
    expected_keys = {
        "detection_loop_converged",
        "detection_loop_iterations",
        "detection_loop_iterations_log",
        "refactor_loop_converged",
        "refactor_loop_iterations",
        "refactor_loop_iterations_log",
        "last_refactor_validation",
        "last_patched",
        "final_detection",
        "token_usage",
        "combined_result",
    }
    assert expected_keys.issubset(dumped.keys())
    assert dumped["detection_loop_converged"] is False
    assert dumped["detection_loop_iterations"] == 0
    assert dumped["detection_loop_iterations_log"] == []
    assert dumped["refactor_loop_converged"] is None
    assert dumped["refactor_loop_iterations"] == 0
    assert dumped["refactor_loop_iterations_log"] == []
    assert dumped["last_refactor_validation"] is None
    assert dumped["last_patched"] is None
    assert dumped["final_detection"] is None
    assert dumped["combined_result"] is None
    assert dumped["token_usage"]["total"]["calls"] == 0
    # Fresh state — no role has run, so by_role serializes empty.
    assert dumped["token_usage"]["by_role"] == {}


def test_workflow_state_iterations_log_accepts_typed_entries():
    state = WorkflowState(
        detection_loop_iterations_log=[
            DetectionIterationLogEntry(
                iteration=1,
                detection=DetectionPhaseEntry(
                    smell_detected=True, detection_justification="present"
                ),
                validation=DetectionValidationPhaseEntry(
                    agrees=True, validated_smell_detected=True
                ),
            )
        ],
        refactor_loop_iterations_log=[
            RefactorIterationLogEntry(
                iteration=1,
                refactor=RefactorPhaseEntry(
                    patched_chars=42, patched_sha256="b" * 64
                ),
                validation=RefactorValidationPhaseEntry(
                    agrees=True, remaining_count=0, disagreement_feedback=None
                ),
            ),
        ],
    )
    dumped = state.model_dump()
    assert dumped["detection_loop_iterations_log"][0]["detection"]["smell_detected"] is True
    assert dumped["detection_loop_iterations_log"][0]["validation"]["agrees"] is True
    assert dumped["refactor_loop_iterations_log"][0]["refactor"]["patched_chars"] == 42
    assert dumped["refactor_loop_iterations_log"][0]["validation"]["agrees"] is True


def test_workflow_state_holds_typed_detection_and_refactor_results():
    detection = DetectionContract(
        reasoning="Checked each job against the smell definition line by line before deciding.",
        target_smell=_SMELL,
        smell_detected=True,
        detection_justification="present at L1-L3",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=3)],
    )
    last = RefactorValidationResult(
        reasoning="Re-read the patched workflow and confirmed no smell instances remain.",
        agrees=True,
        remaining_count=0,
        disagreement_feedback=None,
    )

    state = WorkflowState(
        final_detection=detection,
        last_refactor_validation=last,
        last_patched="fixed: yaml",
    )
    assert state.final_detection is detection
    assert state.last_refactor_validation is last
    assert state.last_patched == "fixed: yaml"


def test_workflow_state_roundtrip_json_preserves_shape():
    state = WorkflowState(
        detection_loop_converged=True,
        detection_loop_iterations=2,
        detection_loop_iterations_log=[
            DetectionIterationLogEntry(
                iteration=1,
                detection=DetectionPhaseEntry(
                    smell_detected=True, detection_justification="iter1"
                ),
                validation=DetectionValidationPhaseEntry(
                    agrees=False,
                    validated_smell_detected=True,
                    disagreement_feedback="off",
                ),
            ),
            DetectionIterationLogEntry(
                iteration=2,
                detection=DetectionPhaseEntry(
                    smell_detected=True, detection_justification="iter2"
                ),
                validation=DetectionValidationPhaseEntry(
                    agrees=True, validated_smell_detected=True
                ),
            ),
        ],
    )
    payload = state.model_dump_json()
    parsed = WorkflowState.model_validate_json(payload)
    assert parsed.detection_loop_converged is True
    assert parsed.detection_loop_iterations == 2
    assert len(parsed.detection_loop_iterations_log) == 2
    assert parsed.detection_loop_iterations_log[1].validation.agrees is True
    assert parsed.detection_loop_iterations_log[0].validation.disagreement_feedback == "off"


def test_workflow_state_rejects_unknown_field():
    with pytest.raises(ValidationError):
        WorkflowState(unknown_field=123)
