"""Tests for the result JSON schema produced by _process_file helpers."""

import gha_agents.env as env_mod
import pytest
from gha_agents.agents import (
    CombinedResult,
    DetectionContract,
    DetectionValidationResult,
    EvidenceRange,
    RefactorResult,
    RefactorValidationResult,
    ValidSmells,
)
from gha_agents.main import _build_result_metadata
from gha_agents.state import RoleTokenUsage
from pydantic import ValidationError

_FIRST_SMELL = next(iter(ValidSmells))


@pytest.fixture()
def fake_env(monkeypatch):
    monkeypatch.setattr(env_mod, "_ensure_env_loaded", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    for role in env_mod._AGENT_ROLES:
        monkeypatch.delenv(f"AGENT_{role.upper()}_MODEL", raising=False)
    token = env_mod._model_override_var.set(
        dict.fromkeys(env_mod._AGENT_ROLES, "default/model")
    )
    yield
    env_mod._model_override_var.reset(token)


def _usage_with(*roles: str) -> RoleTokenUsage:
    """A RoleTokenUsage where each named role recorded exactly one call."""
    usage = RoleTokenUsage()
    for role in roles:
        usage.by_role[role].calls = 1
        usage.total.calls += 1
    return usage


def test_build_result_metadata_c1(fake_env):
    meta = _build_result_metadata(
        agents=1, repetition=1, token_usage=_usage_with("combined")
    )
    assert meta["design"] == "D1"
    assert meta["repetition"] == 1
    assert isinstance(meta["model_ids"], dict)
    # Only the combined agent ran — model_ids must not list the four idle roles.
    assert "combined" in meta["model_ids"]
    assert "detection" not in meta["model_ids"]
    # Cost fields were dropped — USD is derived downstream from tokens + model_ids.
    assert "run_cost_usd" not in meta
    assert "price_table_version" not in meta


def test_build_result_metadata_c2(fake_env):
    meta = _build_result_metadata(agents=2, repetition=2, token_usage=RoleTokenUsage())
    assert meta["design"] == "D2"
    assert meta["repetition"] == 2


def test_build_result_metadata_model_ids_only_active_roles(fake_env):
    usage = _usage_with("detection", "refactor")
    meta = _build_result_metadata(agents=2, repetition=1, token_usage=usage)
    for role in ("detection", "refactor"):
        assert role in meta["model_ids"]
    # The combined role never ran in D2, so it must not appear.
    assert "combined" not in meta["model_ids"]


# ── DetectionContract (binary, per target_smell) ────────────────────────────────


def test_detection_result_smell_detected_true_carries_count_and_evidence_ranges():
    result = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Missing top-level permissions block with external uses step.",
        count=2,
        evidence_ranges=[
            EvidenceRange(start_line=7, end_line=10),
            EvidenceRange(start_line=22, end_line=22),
        ],
    )
    assert result.target_smell == _FIRST_SMELL
    assert result.smell_detected is True
    assert result.detection_justification.startswith("Missing")
    assert result.count == 2
    assert [(r.start_line, r.end_line) for r in result.evidence_ranges] == [(7, 10), (22, 22)]


def test_detection_result_smell_detected_false_has_zero_count_and_empty_ranges():
    result = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="No external uses step present; catalog rule does not apply.",
        count=0,
    )
    assert result.smell_detected is False
    assert result.count == 0
    assert result.evidence_ranges == []


def test_detection_result_absence_smell_count_with_empty_ranges():
    """Absence smells (missing concurrency / paths / comments) carry count >= 1
    with evidence_ranges = [] because the smell is the lack of a block and has
    no natural line anchor."""
    result = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Workflow has no concurrency block; nothing to point to.",
        count=1,
        evidence_ranges=[],
    )
    assert result.count == 1
    assert result.evidence_ranges == []


@pytest.mark.parametrize(
    "missing_field",
    ["reasoning", "target_smell", "smell_detected", "detection_justification", "count"],
)
def test_detection_result_requires_core_fields(missing_field):
    payload = {
        "reasoning": "Checked the workflow against the smell definition step by step.",
        "target_smell": _FIRST_SMELL,
        "smell_detected": False,
        "detection_justification": "n/a",
        "count": 0,
    }
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        DetectionContract(**payload)


def test_detection_result_invariant_smell_detected_implies_count_positive():
    with pytest.raises(ValidationError, match="smell_detected"):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=True,
            detection_justification="claims detected but count=0",
            count=0,
        )


def test_detection_result_invariant_count_zero_implies_not_detected():
    with pytest.raises(ValidationError, match="smell_detected"):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="claims not detected but count=3",
            count=3,
        )


def test_detection_result_allows_more_evidence_ranges_than_count():
    """evidence_ranges is supporting evidence, not a one-per-instance list, so it
    may exceed count — a single instance is often evidenced by several ranges
    (e.g. a ``schedule:`` trigger plus the unguarded job it enables). The old
    ``len(evidence_ranges) <= count`` bound was dropped; it was the single
    dominant cause of structured-output rejections across models."""
    detection = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="1 instance evidenced by 3 ranges",
        count=2,
        evidence_ranges=[
            EvidenceRange(start_line=1, end_line=2),
            EvidenceRange(start_line=3, end_line=4),
            EvidenceRange(start_line=5, end_line=6),
        ],
    )
    assert detection.count == 2
    assert len(detection.evidence_ranges) == 3


def test_detection_result_absent_smell_allows_evidence_ranges():
    """An absent verdict (count=0) MAY still localize the enclosing block the
    agent inspected to reach it (negative evidence). count stays the sole
    instance authority, so a range alongside count=0 is no longer rejected --
    forbidding it was a dominant cause of structured-output parse failures."""
    result = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="claims absent but points to the block inspected",
        count=0,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
    )
    assert result.count == 0
    assert result.smell_detected is False
    assert [(r.start_line, r.end_line) for r in result.evidence_ranges] == [(1, 2)]


def test_detection_result_rejects_legacy_evidence_start_end_fields():
    """The scalar evidence_start_line / evidence_end_line fields were replaced
    by the multi-instance evidence_ranges list (paper §3 Step 7); Pydantic must
    reject the old shape."""
    with pytest.raises(ValidationError):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
            evidence_start_line=1,
            evidence_end_line=2,
        )


def test_evidence_range_rejects_end_before_start():
    with pytest.raises(ValidationError):
        EvidenceRange(start_line=10, end_line=5)


def test_evidence_range_accepts_single_line_instance():
    r = EvidenceRange(start_line=7, end_line=7)
    assert r.start_line == 7
    assert r.end_line == 7


def test_detection_result_requires_reasoning():
    # `reasoning` is a mandatory leading CoT field; omitting it must fail validation.
    with pytest.raises(ValidationError):
        DetectionContract(
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
        )


def test_detection_result_rejects_legacy_present_field():
    # Renamed to ``smell_detected`` for clarity in aggregated result JSON.
    with pytest.raises(ValidationError):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            present=False,
            count=0,
            detection_justification="n/a",
        )


def test_detection_result_rejects_legacy_smells_list():
    with pytest.raises(ValidationError):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
            smells=[],
        )


def test_config_smell_class_removed():
    with pytest.raises(ImportError):
        from gha_agents.agents import ConfigSmell  # noqa: F401


# ── DetectionContract (single detection contract; no workflow echo, #41) ──────


def test_detection_contract_rejects_original_workflow():
    # #41 collapsed the per-scenario detection results into one DetectionContract
    # that never echoes the YAML (it is threaded through typed run state instead).
    # extra='forbid' guards that the original_workflow echo can never come back.
    with pytest.raises(ValidationError):
        DetectionContract(
            reasoning="Checked the workflow against the smell definition step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
            original_workflow="on: [push]\n",
        )


# ── CombinedResult (D1 and unified final output) ──────────────────────────────


def test_combined_result_carries_target_smell_detected_justification_and_patch():
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Detected missing timeout-minutes on the build job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        patched_workflow="on: [push]\n",
    )
    assert result.target_smell == _FIRST_SMELL
    assert result.smell_detected is True
    assert result.detection_justification.startswith("Detected")
    assert result.patched_workflow == "on: [push]\n"


def test_combined_result_accepts_evidence_ranges():
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Detected missing permissions block on the Lagom job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=11, end_line=11)],
        patched_workflow="on: [push]\n",
    )
    assert result.count == 1
    assert [(r.start_line, r.end_line) for r in result.evidence_ranges] == [(11, 11)]


def test_combined_result_evidence_ranges_default_to_empty_when_absent():
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="No smell present.",
        count=0,
        patched_workflow="on: [push]\n",
    )
    assert result.evidence_ranges == []
    assert result.count == 0


def test_combined_result_refactor_justification_defaults_to_empty():
    # Negative path (no refactor performed) → refactor_justification falls back to "".
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="No smell present.",
        count=0,
        patched_workflow="on: [push]\n",
    )
    assert result.refactor_justification == ""


def test_combined_result_carries_refactor_justification():
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Detected missing timeout-minutes on the build job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=3, end_line=8)],
        patched_workflow="on: [push]\n",
        refactor_justification="Added timeout-minutes: 30 to the build job.",
    )
    assert result.refactor_justification == "Added timeout-minutes: 30 to the build job."


def test_combined_result_confidence_optional_defaults_to_none():
    """D2 synthesis path: detection/refactor confidence not elicited at agent level."""
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="No smell present.",
        count=0,
        patched_workflow="on: [push]\n",
    )
    assert result.detection_confidence is None
    assert result.detection_confidence_justification == ""
    assert result.refactor_confidence is None
    assert result.refactor_confidence_justification == ""


def test_combined_result_accepts_c1_detection_and_refactor_confidence():
    """D1 combined agent populates detection/refactor confidence per the prompt's rules."""
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="Detected missing timeout-minutes on the build job.",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
        detection_confidence="high",
        detection_confidence_justification="Workflow clearly lacks timeout-minutes on every job.",
        patched_workflow="on: [push]\n",
        refactor_justification="Added timeout-minutes: 30 to the build job.",
        refactor_confidence="high",
        refactor_confidence_justification=(
            "Localized edit adding timeout-minutes; behavior preserved."
        ),
    )
    assert result.detection_confidence == "high"
    assert result.refactor_confidence == "high"


def test_combined_result_rejects_legacy_severity_field():
    """severity (and its justification) was dropped from the schema; Pydantic must reject it."""
    with pytest.raises(ValidationError):
        CombinedResult(
            reasoning="Worked through detection and the patch step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=True,
            detection_justification="Detected missing timeout-minutes on the build job.",
            count=1,
            evidence_ranges=[EvidenceRange(start_line=4, end_line=6)],
            severity="minor",
            patched_workflow="on: [push]\n",
        )


def test_combined_result_rejects_legacy_smells_list():
    with pytest.raises(ValidationError):
        CombinedResult(
            reasoning="Worked through detection and the patch step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
            patched_workflow="",
            smells=[],
        )


def test_combined_result_rejects_legacy_present_field():
    with pytest.raises(ValidationError):
        CombinedResult(
            reasoning="Worked through detection and the patch step by step.",
            target_smell=_FIRST_SMELL,
            present=False,
            detection_justification="n/a",
            count=0,
            patched_workflow="",
        )


def test_combined_result_rejects_legacy_evidence_start_end_fields():
    with pytest.raises(ValidationError):
        CombinedResult(
            reasoning="Worked through detection and the patch step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=False,
            detection_justification="n/a",
            count=0,
            patched_workflow="",
            evidence_start_line=1,
            evidence_end_line=2,
        )


def test_combined_result_invariant_count_matches_smell_detected():
    with pytest.raises(ValidationError, match="smell_detected"):
        CombinedResult(
            reasoning="Worked through detection and the patch step by step.",
            target_smell=_FIRST_SMELL,
            smell_detected=True,
            detection_justification="claims detected but count=0",
            count=0,
            patched_workflow="",
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "reasoning",
        "target_smell",
        "smell_detected",
        "detection_justification",
        "count",
        "patched_workflow",
    ],
)
def test_combined_result_requires_core_fields(missing_field):
    payload = {
        "reasoning": "Worked through detection and the patch step by step.",
        "target_smell": _FIRST_SMELL,
        "smell_detected": True,
        "detection_justification": "n/a",
        "count": 1,
        "evidence_ranges": [EvidenceRange(start_line=1, end_line=2)],
        "patched_workflow": "x: 1\n",
    }
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        CombinedResult(**payload)


# ── RefactorValidationResult (agrees + remaining_count + disagreement_feedback) ──


def test_refactor_validation_result_accepts_agreement_with_zero_remaining():
    result = RefactorValidationResult(
        reasoning="Re-derived the verdict independently before comparing.",
        agrees=True,
        remaining_count=0,
        disagreement_feedback=None,
    )
    assert result.agrees is True
    assert result.remaining_count == 0
    assert result.disagreement_feedback is None


def test_refactor_validation_result_accepts_disagreement_with_remaining_instances():
    result = RefactorValidationResult(
        reasoning="Re-derived the verdict independently before comparing.",
        agrees=False,
        remaining_count=1,
        disagreement_feedback="Job 'deploy' still lacks timeout-minutes at line 42.",
    )
    assert result.agrees is False
    assert result.remaining_count == 1
    assert "line 42" in result.disagreement_feedback


def test_refactor_validation_result_rejects_agreement_with_nonzero_remaining():
    # Acceptance is one-directional: agrees=true ⇒ remaining_count=0, enforced at
    # parse time so a contradictory acceptance never reaches the TSRR metric.
    with pytest.raises(ValidationError):
        RefactorValidationResult(
            reasoning="Re-derived the verdict independently before comparing.",
            agrees=True,
            remaining_count=2,
            disagreement_feedback=None,
        )


def test_refactor_validation_result_accepts_disagreement_with_zero_remaining():
    # A validator may reject a patch for a non-count reason — broken YAML, an
    # invalid Actions schema, a behaviour-changing edit — while no target-smell
    # instance remains. agrees=false ∧ remaining_count=0 (with feedback) is a valid
    # verdict; the old invariant forced remaining_count>=1 on every rejection,
    # contradicting the prompt's own rules and rejecting these rows (Bug-G).
    result = RefactorValidationResult(
        reasoning="Re-derived the verdict independently before comparing.",
        agrees=False,
        remaining_count=0,
        disagreement_feedback="Patch removed the timeout but broke the YAML at line 12.",
    )
    assert result.agrees is False
    assert result.remaining_count == 0
    assert "line 12" in result.disagreement_feedback


def test_refactor_validation_result_rejects_agreement_with_feedback_populated():
    # When the validator accepts the patch, disagreement_feedback MUST be null —
    # populating it would leak phantom rejection text to the next iteration.
    with pytest.raises(ValidationError):
        RefactorValidationResult(
            reasoning="Re-derived the verdict independently before comparing.",
            agrees=True,
            remaining_count=0,
            disagreement_feedback="should be null on acceptance",
        )


def test_refactor_validation_result_rejects_disagreement_without_feedback():
    # When the validator rejects the patch, disagreement_feedback MUST be
    # non-empty — the Refactor Agent needs concrete guidance for the retry.
    with pytest.raises(ValidationError):
        RefactorValidationResult(
            reasoning="Re-derived the verdict independently before comparing.",
            agrees=False,
            remaining_count=1,
            disagreement_feedback=None,
        )


@pytest.mark.parametrize(
    "missing_field",
    ["reasoning", "agrees", "remaining_count"],
)
def test_refactor_validation_result_requires_core_fields(missing_field):
    payload = {
        "reasoning": "Re-derived the verdict independently before comparing.",
        "agrees": True,
        "remaining_count": 0,
        "disagreement_feedback": None,
    }
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        RefactorValidationResult(**payload)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "smell_resolved",
        "justification",
        "patched_workflow",
        "introduced_regressions",
        "remaining_smell_count",
    ],
)
def test_refactor_validation_result_rejects_legacy_fields(legacy_field):
    # These fields were removed from the Melo-style schema; Pydantic must reject them
    # (regressions are measured externally per paper §3 Step 7).
    payload = {
        "reasoning": "Re-derived the verdict independently before comparing.",
        "agrees": True,
        "remaining_count": 0,
        "disagreement_feedback": None,
        legacy_field: "x",
    }
    with pytest.raises(ValidationError):
        RefactorValidationResult(**payload)


# ── DetectionValidationResult (Agent 2 validation output) ─────────────────────


def _inner_detection(smell_detected: bool = True) -> DetectionContract:
    return DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=smell_detected,
        detection_justification="Example justification.",
        count=1 if smell_detected else 0,
    )


def test_detection_validation_agrees_without_feedback():
    result = DetectionValidationResult(
        reasoning="Re-derived the verdict independently before comparing.",
        agrees=True,
        validated_result=_inner_detection(),
    )
    assert result.agrees is True
    assert result.disagreement_feedback is None
    assert result.validated_result.smell_detected is True


def test_detection_validation_disagrees_with_feedback():
    corrected = DetectionContract(
        reasoning="Checked the workflow against the smell definition step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=False,
        detection_justification="Pattern does not match catalog definition after re-reading.",
        count=0,
    )
    result = DetectionValidationResult(
        reasoning="Re-derived the verdict independently before comparing.",
        agrees=False,
        validated_result=corrected,
        disagreement_feedback="Agent 1 misclassified: the workflow uses a pinned SHA.",
    )
    assert result.agrees is False
    assert result.validated_result.smell_detected is False
    assert result.disagreement_feedback.startswith("Agent 1")


@pytest.mark.parametrize(
    "missing_field",
    ["reasoning", "agrees", "validated_result"],
)
def test_detection_validation_requires_core_fields(missing_field):
    payload = {
        "reasoning": "Re-derived the verdict independently before comparing.",
        "agrees": True,
        "validated_result": _inner_detection(),
    }
    payload.pop(missing_field)
    with pytest.raises(ValidationError):
        DetectionValidationResult(**payload)


def test_detection_validation_rejects_extras():
    with pytest.raises(ValidationError):
        DetectionValidationResult(
            reasoning="Re-derived the verdict independently before comparing.",
            agrees=True,
            validated_result=_inner_detection(),
            corrected_result=None,  # not a defined field
        )


# ── RefactorResult (patched YAML + refactor justification) ───────────────────────


def test_refactor_result_carries_patch_and_justification():
    result = RefactorResult(
        reasoning="Located the affected keys and applied the minimal edit.",
        patched_workflow="on: [push]\n",
        refactor_justification="Pinned actions/checkout@v4 to its commit SHA.",
    )
    assert result.patched_workflow == "on: [push]\n"
    assert result.refactor_justification.startswith("Pinned")


def test_refactor_result_requires_justification():
    # refactor_justification is mandatory — the refactor agent must explain the edit it made.
    with pytest.raises(ValidationError):
        RefactorResult(
            reasoning="Located the affected keys and applied the minimal edit.",
            patched_workflow="on: [push]\n",
        )


# ── patched_workflow wrapper-stripping (defensive sanitization) ───────────────

_RAW_YAML = "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"


@pytest.mark.parametrize(
    "wrapped",
    [
        f"<workflow>\n{_RAW_YAML}</workflow>",
        f"<workflow>\n{_RAW_YAML}\n</workflow>",
        f"<WORKFLOW>\n{_RAW_YAML}</WORKFLOW>",
        f"```yaml\n{_RAW_YAML}```",
        f"```yml\n{_RAW_YAML}```",
        f"```\n{_RAW_YAML}```",
        f"```yaml\n<workflow>\n{_RAW_YAML}</workflow>\n```",
        f"<workflow>\n```yaml\n{_RAW_YAML}```\n</workflow>",
    ],
)
def test_refactor_result_strips_workflow_and_fence_wrappers(wrapped):
    result = RefactorResult(
        reasoning="Located the affected keys and applied the minimal edit.",
        patched_workflow=wrapped,
        refactor_justification="stub",
    )
    assert result.patched_workflow == _RAW_YAML.rstrip("\n")


def test_refactor_result_leaves_clean_yaml_untouched():
    result = RefactorResult(
        reasoning="Located the affected keys and applied the minimal edit.",
        patched_workflow=_RAW_YAML,
        refactor_justification="stub",
    )
    assert result.patched_workflow == _RAW_YAML


def test_combined_result_strips_workflow_wrapper_from_patched_workflow():
    result = CombinedResult(
        reasoning="Worked through detection and the patch step by step.",
        target_smell=_FIRST_SMELL,
        smell_detected=True,
        detection_justification="x",
        count=1,
        evidence_ranges=[EvidenceRange(start_line=1, end_line=2)],
        patched_workflow=f"<workflow>\n{_RAW_YAML}</workflow>",
    )
    assert result.patched_workflow == _RAW_YAML.rstrip("\n")
