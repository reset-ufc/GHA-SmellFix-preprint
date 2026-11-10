"""output_contract.py — the single declarative source of each role's output shape.

The output contract used to be stated twice that merely happened to agree: the
``*_OUTPUT_FORMAT`` prose constants in :mod:`prompts` and the hand-written
Pydantic models in :mod:`agents`. This module makes the declaration single: each
:class:`OutputContract` owns the field order + the prose class-name, *renders*
the ``<output_format>`` block from them (so the prose can never drift from the
field list), and a boot-time assertion proves the declared order equals the
bound schema's ``model_fields`` — failing fast if a model is ever reordered,
extended, or trimmed out from under the prompt.

Kept deliberately framework-free and schema-free (it imports neither agno nor
:mod:`agents`) so it sits *below* both ``prompts`` and ``agents`` in the import
graph: ``prompts`` sources its rendered blocks from here, and ``agents`` calls
:func:`assert_output_contracts_consistent` at import with the schema classes
injected — no import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The chain-of-thought field leads every structured output (paper §3): the model
# reasons before committing to a verdict, so ``reasoning`` is always first.
_LEADING_FIELD = "reasoning"


@dataclass(frozen=True, slots=True)
class OutputContract:
    """One role's output shape: the prose class-name plus the ordered field list.

    ``prose_name`` is the token rendered after ``Emit `` — it is a *declared*
    string, NOT ``schema.__name__``: the detection block renders
    ``DetectionResult`` (the model-facing label kept stable across the rename to
    the ``DetectionContract`` class), and the rendered bytes are stable (they
    feed ``PROMPT_HASHES``).
    """

    prose_name: str
    fields: tuple[str, ...]

    def render(self) -> str:
        """Render the exact ``<output_format>`` block (single line, no padding)."""
        field_list = ", ".join(self.fields)
        return f"<output_format>Emit {self.prose_name} exactly: {field_list}. No extra fields.</output_format>"

    def assert_reasoning_leads(self) -> None:
        if not self.fields or self.fields[0] != _LEADING_FIELD:
            got = self.fields[0] if self.fields else None
            raise ValueError(
                f"OutputContract '{self.prose_name}' must lead with {_LEADING_FIELD!r}, got {got!r}"
            )


# Keyed to mirror the distinct ``*_OUTPUT_FORMAT`` blocks prompts.py wires into
# _render_system_prompt. ``detection`` is the single detection block used by the
# D2 single-shot detection agent.
OUTPUT_CONTRACTS: dict[str, OutputContract] = {
    "combined": OutputContract(
        prose_name="CombinedResult",
        fields=(
            "reasoning",
            "target_smell",
            "smell_detected",
            "detection_justification",
            "count",
            "evidence_ranges",
            "detection_confidence",
            "detection_confidence_justification",
            "patched_workflow",
            "refactor_justification",
            "refactor_confidence",
            "refactor_confidence_justification",
        ),
    ),
    "detection": OutputContract(
        prose_name="DetectionResult",
        fields=(
            "reasoning",
            "target_smell",
            "smell_detected",
            "detection_justification",
            "count",
            "evidence_ranges",
            "detection_confidence",
            "detection_confidence_justification",
        ),
    ),
    "detection_validation": OutputContract(
        prose_name="DetectionValidationResult",
        fields=(
            "reasoning",
            "agrees",
            "validated_result",
            "disagreement_feedback",
            "validation_confidence",
            "validation_confidence_justification",
        ),
    ),
    "refactor": OutputContract(
        prose_name="RefactorResult",
        fields=(
            "reasoning",
            "patched_workflow",
            "refactor_justification",
            "refactor_confidence",
            "refactor_confidence_justification",
        ),
    ),
    "refactor_validation": OutputContract(
        prose_name="RefactorValidationResult",
        fields=(
            "reasoning",
            "agrees",
            "remaining_count",
            "disagreement_feedback",
            "validation_confidence",
            "validation_confidence_justification",
        ),
    ),
}


def assert_output_contracts_consistent(schema_by_key: Mapping[str, Any]) -> None:
    """Round-trip every contract against its bound schema; raise on any drift.

    Pure and dependency-injected (the schema classes are passed in), so it runs
    at :mod:`agents` import for fail-fast boot validation and is unit-testable in
    isolation with a deliberately-corrupted mapping. ``schema_by_key`` must cover
    exactly the contract keys.
    """
    missing = set(OUTPUT_CONTRACTS) - set(schema_by_key)
    if missing:
        raise ValueError(f"No schema bound for output contracts: {sorted(missing)}")
    for key, contract in OUTPUT_CONTRACTS.items():
        contract.assert_reasoning_leads()
        schema_fields = tuple(schema_by_key[key].model_fields)
        if contract.fields != schema_fields:
            raise ValueError(
                f"OutputContract '{key}' field list {contract.fields} does not match "
                f"bound schema {schema_by_key[key].__name__} model_fields {schema_fields}"
            )
