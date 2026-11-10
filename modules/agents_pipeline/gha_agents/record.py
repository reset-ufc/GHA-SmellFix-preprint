"""Typed per-run record + the CombinedResult synthesis named constructors (#43).

Two duplications collapse here:

* :class:`CombinedResultAssembler` centralises the hand-built
  :class:`~gha_agents.agents.CombinedResult` syntheses that lived inline in
  ``teams.py`` (D2 present/absent). D1 needs no synthesis — its combined agent
  already returns a ``CombinedResult`` — so ``from_d1`` is the identity that
  documents the seam. The absent branch returns the original workflow verbatim
  with a zeroed localization signal (``count=0``, no ranges).

* :class:`RunRecord` is the closed (``extra='forbid'``) typed record whose
  :meth:`~RunRecord.dump` *is* the per-run result JSON, replacing the ~15-key
  dict that ``main._run_standard`` built by hand. ``count`` is carried but stays
  diagnostic-only — presence is the ``smell_detected`` field, never derived from
  the count.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .agents import (
    CombinedResult,
    Confidence,
    DetectionContract,
    EvidenceRange,
    RefactorResult,
    ValidSmells,
)
from .state import (
    RoleTokenUsage,
    WorkflowState,
)
from .tools import validate_github_actions_syntax


class CombinedResultAssembler:
    """Named constructors for :class:`CombinedResult` across D1/D2.

    Static methods only — a namespace, not a stateful object. Each constructor
    mirrors one former inline construction site so the synthesis lives in one
    tested place.
    """

    @staticmethod
    def from_d1(combined: CombinedResult) -> CombinedResult:
        """D1: the combined agent already produced the unified result — pass it through."""
        return combined

    @staticmethod
    def _absent(
        target_smell: ValidSmells, detection: DetectionContract, original_workflow: str
    ) -> CombinedResult:
        """Smell-absent synthesis: original workflow verbatim, zeroed signal."""
        return CombinedResult(
            reasoning=detection.reasoning,
            target_smell=target_smell,
            smell_detected=False,
            detection_justification=detection.detection_justification,
            detection_confidence=detection.detection_confidence,
            detection_confidence_justification=detection.detection_confidence_justification,
            count=0,
            evidence_ranges=[],
            patched_workflow=original_workflow,
        )

    @staticmethod
    def _present(
        target_smell: ValidSmells,
        detection: DetectionContract,
        *,
        patched_workflow: str,
        refactor_justification: str,
        refactor_confidence: Confidence | None,
        refactor_confidence_justification: str,
    ) -> CombinedResult:
        """Smell-present synthesis: detection signal + the resolved patch.

        Both confidence pairs are threaded through: detection's from the converged
        detection, refactor's from the (last) refactor result, so the synthesized
        CombinedResult carries the same calibration the model emitted.
        """
        return CombinedResult(
            reasoning=detection.reasoning,
            target_smell=target_smell,
            smell_detected=True,
            detection_justification=detection.detection_justification,
            detection_confidence=detection.detection_confidence,
            detection_confidence_justification=detection.detection_confidence_justification,
            count=detection.count,
            evidence_ranges=list(detection.evidence_ranges),
            patched_workflow=patched_workflow,
            refactor_justification=refactor_justification,
            refactor_confidence=refactor_confidence,
            refactor_confidence_justification=refactor_confidence_justification,
        )

    @staticmethod
    def from_d2_absent(
        target_smell: ValidSmells, detection: DetectionContract, original_workflow: str
    ) -> CombinedResult:
        """D2 presence-gate took the absent branch — skip refactor, echo the input."""
        return CombinedResultAssembler._absent(target_smell, detection, original_workflow)

    @staticmethod
    def from_d2_present(
        target_smell: ValidSmells, detection: DetectionContract, refactor: RefactorResult
    ) -> CombinedResult:
        """D2 presence-gate took the present branch — pair detection with the single patch."""
        return CombinedResultAssembler._present(
            target_smell,
            detection,
            patched_workflow=refactor.patched_workflow,
            refactor_justification=refactor.refactor_justification,
            refactor_confidence=refactor.refactor_confidence,
            refactor_confidence_justification=refactor.refactor_confidence_justification,
        )

class RunRecord(BaseModel):
    """Closed typed record whose :meth:`dump` is the per-run result JSON.

    Built from a :class:`CombinedResult` plus the populated
    :class:`~gha_agents.state.WorkflowState`. D1/D2 drive no feedback loops, so
    the record carries only the single detection/refactor outcome.
    """

    model_config = ConfigDict(extra="forbid")

    patched_workflow: str
    # Observability only (#1 "close the validation loop"): does the final patch
    # still parse + conform to the vendored GitHub Actions schema? Recorded for
    # every run; never gates control flow or alters ``patched_workflow``.
    patch_valid: bool
    patch_validation_error: str | None
    smell_detected: bool
    detection_justification: str
    refactor_justification: str
    # Self-reported calibration the prompts elicit (low/medium/high + a 1-4 sentence
    # justification each). detection_* is None on no-detection paths where it is not
    # separately elicited; refactor_* is None when smell_detected=false (no refactor).
    detection_confidence: Confidence | None
    detection_confidence_justification: str
    refactor_confidence: Confidence | None
    refactor_confidence_justification: str
    count: int
    evidence_ranges: list[EvidenceRange]
    token_usage: RoleTokenUsage
    agent_call_count: int

    @staticmethod
    def _patch_validity(patched_workflow: str) -> tuple[bool, str | None]:
        """Validate the final patch against the vendored GitHub Actions schema.

        Observability only: returns ``(True, None)`` when the patch parses and
        conforms, else ``(False, <report>)`` carrying the human-readable error.
        Never alters control flow or the patch itself.
        """
        report = validate_github_actions_syntax(patched_workflow)
        return (True, None) if report == "OK" else (False, report)

    @classmethod
    def from_standard(cls, combined: CombinedResult, state: WorkflowState) -> RunRecord:
        """D1/D2: the single detection/refactor outcome, no feedback loops."""
        token_usage = state.token_usage
        patch_valid, patch_validation_error = cls._patch_validity(combined.patched_workflow)
        return cls(
            patched_workflow=combined.patched_workflow,
            patch_valid=patch_valid,
            patch_validation_error=patch_validation_error,
            smell_detected=combined.smell_detected,
            detection_justification=combined.detection_justification,
            refactor_justification=combined.refactor_justification,
            detection_confidence=combined.detection_confidence,
            detection_confidence_justification=combined.detection_confidence_justification,
            refactor_confidence=combined.refactor_confidence,
            refactor_confidence_justification=combined.refactor_confidence_justification,
            count=combined.count,
            evidence_ranges=list(combined.evidence_ranges),
            token_usage=token_usage,
            agent_call_count=token_usage.total.calls,
        )

    def dump(self) -> dict[str, Any]:
        """Serialize to the result dict for the run."""
        return self.model_dump()
