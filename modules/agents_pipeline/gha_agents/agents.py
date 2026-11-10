"""
agents.py — Pydantic models and Agno agent definitions for Agents.

The evaluation unit is one (workflow, target_smell) pair (paper Step 4). Each
agent is specialized for ONE target smell at construction time: its system
prompt is rendered via the ``build_*_prompt`` helpers with the smell definition
interpolated (knowledge augmentation, Melo et al. §3); user turns carry only
data (workflow YAML, evidence, peer feedback) wrapped in XML.

Models: ``DetectionContract`` (binary decision), ``DetectionValidationResult``
(validation verdict on it), ``RefactorResult`` (refactored YAML + justification),
``RefactorValidationResult`` (validation verdict — agrees + remaining_count, which
TSRR uses — + disagreement_feedback), ``CombinedResult`` (D1 single-pass and
unified final output).

Agent factory functions (lazy model instantiation, smell-specialized):
``make_detection_agent`` → ``DetectionContract``, ``make_refactor_agent`` →
``RefactorResult``, ``make_combined_agent`` → ``CombinedResult`` (detect + refactor).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from agno.agent import Agent
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .env import is_json_mode_active
from .model_factory import ModelFactory
from .output_contract import assert_output_contracts_consistent
from .prompts import (
    SMELL_CATALOG,
    Scenario,
    build_combined_prompt,
    build_detection_prompt,
    build_refactor_prompt,
    prompt_hash,
)
from .tools import (
    validate_github_actions_syntax,
    validate_yaml_syntax,
)
from .utils import is_verbose

__all__ = [
    "CombinedResult",
    "DetectionContract",
    "DetectionValidationResult",
    "EvidenceRange",
    "RefactorResult",
    "RefactorValidationResult",
    "ValidSmells",
    "make_combined_agent",
    "make_detection_agent",
    "make_refactor_agent",
]

# Dynamically construct a native string Enum: member name = PascalCase pascal
# (matches ground_truth present_smells[].pascal / absent_smells[].pascal), value =
# human-readable Khatami name.
ValidSmells = Enum(
    "ValidSmells",
    {s["pascal"]: s["name"] for s in SMELL_CATALOG},
    type=str,
)


# Defensive: some models echo back the input demarcation tags (``<workflow>``)
# or wrap the YAML in markdown code fences. Strip those wrappers up to two
# nesting layers so downstream tooling (gha-ci-detector, schema validator)
# always sees raw YAML. Clean inputs (no wrapper match) are returned untouched
# so legitimate trailing whitespace is preserved.
_FENCE_RE = re.compile(
    r"\A\s*```(?:[A-Za-z0-9_+-]+)?[ \t]*\n(.*?)\n*```\s*\Z",
    re.DOTALL,
)
_WORKFLOW_TAG_RE = re.compile(
    r"\A\s*<workflow>\s*(.*?)\n*</workflow>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _strip_yaml_wrappers(value: str) -> str:
    if not isinstance(value, str):
        return value
    s = value
    for _ in range(2):
        m = _FENCE_RE.match(s)
        if m:
            s = m.group(1)
            continue
        m = _WORKFLOW_TAG_RE.match(s)
        if m:
            s = m.group(1)
            continue
        break
    return s


# ─── Pydantic Models ──────────────────────────────────────────────────────────

# Calibration scale shared by every agent's role-prefixed ``*_confidence`` field.
Confidence = Literal["low", "medium", "high"]


class EvidenceRange(BaseModel):
    """One 1-based, inclusive line range that exhibits a target smell instance.

    Smells can occur multiple times in a single workflow (ground_truth validation
    found 1–8 instances per (workflow, smell) pair). Each instance the agent
    localizes is recorded as one ``EvidenceRange``; the parent ``DetectionContract``
    / ``CombinedResult`` carries the full list plus a ``count`` field (see those
    models for the schema invariants).
    """

    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(
        ge=1,
        description="First line (1-based, inclusive) of the smell instance.",
    )
    end_line: int = Field(
        ge=1,
        description="Last line (1-based, inclusive) of the smell instance.",
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> EvidenceRange:
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )
        return self


def _validate_multi_instance(*, smell_detected: bool, count: int) -> None:
    """Shared invariant check for the two multi-instance result models.

    Enforces ONLY the count/detected consistency: ``count >= 0`` and
    ``smell_detected == (count > 0)``. ``evidence_ranges`` is deliberately NOT
    constrained against ``count``. It is supporting line-range evidence, NOT a
    strict one-per-instance list: it may hold MORE entries than ``count`` (one
    instance evidenced by several ranges, e.g. a ``schedule:`` trigger plus the
    unguarded job it enables); it may be EMPTY even when ``count > 0`` for
    absence smells (missing ``concurrency`` block, no comments anywhere) with no
    natural line anchor; and it may be NON-EMPTY even when ``count == 0`` — an
    absent verdict still localizes the enclosing block the agent inspected to
    reach that verdict (negative evidence). Both bounds — the old
    ``len(evidence_ranges) <= count`` and the ``count == 0 => no evidence`` rule
    — were dropped: each rejected reasonable model outputs and was a dominant
    cause of structured-output rejections across models. ``count`` is the sole
    instance authority; downstream metrics read it, never ``len(evidence_ranges)``.
    """
    if count < 0:
        raise ValueError(f"count ({count}) must be >= 0")
    if smell_detected != (count > 0):
        raise ValueError(
            f"smell_detected={smell_detected} contradicts count={count}; "
            "smell_detected must equal (count > 0)."
        )


class DetectionContract(BaseModel):
    """Binary detection decision for a single target smell, multi-instance.

    ``count`` carries the total number of instances of the target smell the
    agent claims to have found (LLM-emitted). ``evidence_ranges`` carries the
    line ranges for the instances the agent could localize — possibly an
    empty list even when ``count > 0`` for absence smells (e.g. missing
    ``concurrency`` block, no comments anywhere) where there is no specific
    line pattern to anchor on; see :func:`_validate_multi_instance` for the
    full invariant.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step chain-of-thought reasoning worked through before committing "
            "to the detection verdict below, grounded in concrete evidence from the "
            "workflow. Natural language; no YAML blocks, code snippets, or markdown fences."
        )
    )
    target_smell: ValidSmells = Field(
        description=(
            "The target configuration smell this decision refers to "
            "(must match the smell specified in the system prompt)."
        )
    )
    smell_detected: bool = Field(
        description="True iff the target smell is detected in the workflow."
    )
    detection_justification: str = Field(
        description=(
            "A 1-4 sentence explanation in natural language, grounded in concrete evidence "
            "from the workflow. Reference specific keys, values, or line numbers inline; "
            "include no YAML blocks, code snippets, or markdown fences."
        )
    )
    count: int = Field(
        ge=0,
        description=(
            "Total number of instances of the target smell detected in the workflow. "
            "0 iff smell_detected=false. Absence smells (missing concurrency / paths "
            "filter / matrix / comments) may report count >= 1 with an empty "
            "evidence_ranges when there is no natural line anchor."
        ),
    )
    evidence_ranges: list[EvidenceRange] = Field(
        default_factory=list,
        description=(
            "1-based inclusive line ranges grounding the verdict: the localized "
            "smell instances when count > 0, or the enclosing block the agent "
            "inspected when count == 0 (negative evidence). May be empty when there "
            "is no natural line anchor. Not a one-per-instance list and not bounded "
            "by count -- count is the sole instance authority."
        ),
    )
    detection_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the detection decision: high = workflow clearly satisfies "
            "or clearly does not satisfy the smell definition; medium = decision depends "
            "on contextual interpretation (e.g. whether a job is a deployment / privileged "
            "job); low = workflow is incomplete, heavily templated, or ambiguous."
        ),
    )
    detection_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the confidence level was assigned, in "
            "natural language grounded in concrete evidence from the workflow. Reference "
            "specific keys, values, or line numbers inline; include no YAML blocks, code "
            "snippets, or markdown fences."
        ),
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> DetectionContract:
        _validate_multi_instance(
            smell_detected=self.smell_detected,
            count=self.count,
        )
        return self


class RefactorResult(BaseModel):
    """Output of the refactor agent."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step chain-of-thought reasoning worked through before producing the "
            "patched workflow below, grounded in concrete evidence from the original "
            "workflow. Natural language; no YAML blocks, code snippets, or markdown fences."
        )
    )
    patched_workflow: str = Field(
        description="The complete patched YAML workflow (full file, not a diff)."
    )
    refactor_justification: str = Field(
        description=(
            "A 1-4 sentence explanation in natural language, grounded in concrete "
            "evidence from the refactored workflow, of the edit applied to "
            "patched_workflow — which keys or lines changed and why the change "
            "removes the smell without altering observable behavior. Reference "
            "specific keys, values, or line numbers inline; include no YAML blocks, "
            "code snippets, or markdown fences."
        )
    )
    refactor_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the refactoring: high = edit is localized, clearly removes "
            "all detected instances, and is unlikely to change behavior; medium = edit "
            "likely removes the smell but depends on contextual assumptions (permissions, "
            "cache semantics, action compatibility, event/branch semantics); low = the "
            "workflow is incomplete/ambiguous or the fix may alter behavior."
        ),
    )
    refactor_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the refactor confidence level was assigned, "
            "focusing on completeness, safety, and behavior preservation. Reference specific "
            "keys, values, or line numbers inline; include no YAML blocks, code snippets, or "
            "markdown fences."
        ),
    )

    @field_validator("patched_workflow", mode="before")
    @classmethod
    def _strip_wrappers(cls, v: object) -> object:
        return _strip_yaml_wrappers(v) if isinstance(v, str) else v


class RefactorValidationResult(BaseModel):
    """Agent 4 verdict on the Refactor Agent's proposed patch.

    Vocabulary mirrors :class:`DetectionValidationResult`: ``agrees`` is true
    iff the validator accepts the patch (target smell removed and YAML valid),
    in which case ``disagreement_feedback`` is ``None``; otherwise the
    validator populates ``disagreement_feedback`` with concrete guidance the
    Refactor Agent will consume on the next iteration. ``remaining_count``
    carries the validator's quantitative verdict — the number of target-smell
    instances it still observes in the refactored workflow — and provides the
    per-iteration signal needed by the Target-Smell Reduction Rate (TSRR)
    metric. Regression counting against unrelated smells is measured
    externally by re-running the static analyzer on the patched workflow
    (paper §3 Step 7), so it is intentionally absent from this schema.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step chain-of-thought reasoning worked through before forming the "
            "accept/reject verdict below, grounded in concrete evidence from the refactored "
            "workflow. Natural language; no YAML blocks, code snippets, or markdown fences."
        )
    )
    agrees: bool = Field(
        description=(
            "True iff the validator accepts the Refactor Agent's patch — the target "
            "smell is no longer present in the refactored workflow and the YAML is "
            "syntactically valid."
        )
    )
    remaining_count: int = Field(
        ge=0,
        description=(
            "Total number of target-smell instances still present in the refactored "
            "workflow. Required invariant: remaining_count=0 when agrees=true; "
            "remaining_count>=1 when agrees=false (the validator's count of instances "
            "the Refactor Agent still needs to remove on the next iteration)."
        ),
    )
    disagreement_feedback: str | None = Field(
        default=None,
        description=(
            "A 1-4 sentence explanation in natural language, grounded in concrete evidence "
            "from the refactored workflow, consumed by the Refactor Agent in the next "
            "iteration; populated only when agrees=false. Reference specific keys, values, "
            "or line numbers inline; include no YAML blocks, code snippets, or markdown "
            "fences."
        ),
    )
    validation_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the validator's accept/reject verdict: high = acceptance or "
            "rejection is clear from the patched workflow; medium = the verdict depends on "
            "contextual assumptions (permissions, action compatibility, deployment intent, "
            "cache/event semantics); low = the workflow is incomplete/ambiguous or the patch "
            "cannot be reliably assessed."
        ),
    )
    validation_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the validation confidence level was assigned. "
            "Reference specific keys, values, or line numbers inline; include no YAML blocks, "
            "code snippets, or markdown fences."
        ),
    )

    @model_validator(mode="after")
    def _check_agreement_invariant(self) -> RefactorValidationResult:
        """Enforce the verdict's internal consistency at parse time.

        Acceptance is fully constrained: ``agrees=true`` requires
        ``remaining_count=0`` and a null ``disagreement_feedback``. Rejection
        requires concrete feedback but does NOT constrain ``remaining_count``:
        the ``REFACTOR_VALIDATION_RULES`` prompt lets the validator reject a
        patch for a non-count reason — invalid YAML, an invalid Actions schema,
        a behaviour-changing or unrelated edit — while no target-smell instance
        remains, i.e. ``agrees=false`` with ``remaining_count=0``. The prior
        invariant forced ``remaining_count>=1`` on every rejection, contradicting
        that prompt rule and raising on those valid verdicts (Bug-G).
        """
        if self.agrees:
            if self.remaining_count != 0:
                raise ValueError(
                    f"agrees=true contradicts remaining_count={self.remaining_count}; "
                    "remaining_count must be 0 when the validator accepts the patch."
                )
            if self.disagreement_feedback is not None:
                raise ValueError(
                    "agrees=true contradicts disagreement_feedback being set; "
                    "disagreement_feedback must be null when the validator accepts the patch."
                )
        elif not (self.disagreement_feedback or "").strip():
            raise ValueError(
                "agrees=false requires a non-empty disagreement_feedback; "
                "the Refactor Agent needs concrete guidance for the next iteration."
            )
        return self


class DetectionValidationResult(BaseModel):
    """Agent 2 verdict on the Detection Agent's decision."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step chain-of-thought reasoning worked through before forming the "
            "agree/disagree verdict below, grounded in concrete evidence from the "
            "workflow. Natural language; no YAML blocks, code snippets, or markdown fences."
        )
    )
    agrees: bool = Field(
        description="True iff the validator endorses the Detection Agent's decision."
    )
    validated_result: DetectionContract = Field(
        description=(
            "Final detection to carry forward: mirrors the Detection Agent when "
            "agrees=true, or the validator's own corrected decision otherwise."
        )
    )
    disagreement_feedback: str | None = Field(
        default=None,
        description=(
            "A 1-4 sentence explanation in natural language, grounded in concrete evidence "
            "from the workflow, consumed by the Detection Agent in the next iteration; "
            "populated only when agrees=false. Reference specific keys, values, or line "
            "numbers inline; include no YAML blocks, code snippets, or markdown fences."
        ),
    )
    validation_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the validator's agree/disagree verdict: high = agreement or "
            "disagreement is clear from the workflow; medium = the verdict depends on "
            "contextual interpretation (e.g. whether a job is a deployment / privileged "
            "job); low = the workflow is incomplete/ambiguous or the peer result cannot be "
            "reliably assessed."
        ),
    )
    validation_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the validation confidence level was assigned. "
            "Reference specific keys, values, or line numbers inline; include no YAML blocks, "
            "code snippets, or markdown fences."
        ),
    )


class CombinedResult(BaseModel):
    """Unified output: binary decision for the target smell plus patched YAML.

    ``detection_confidence`` / ``refactor_confidence`` (and their justifications)
    are populated by the D1 combined agent, whose prompt elicits them as part of
    the structured output. The D2 synthesis path in ``teams.py`` constructs
    ``CombinedResult`` from intermediate per-agent outputs that do not carry
    calibration metadata, so those fields default to ``None``/empty string in
    that pathway. Downstream consumers must treat ``None`` as "not assessed at
    the agent level", not as missing data.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description=(
            "Step-by-step chain-of-thought reasoning worked through before committing to "
            "the detection verdict and, when present, the refactoring below, grounded in "
            "concrete evidence from the workflow. Natural language; no YAML blocks, code "
            "snippets, or markdown fences."
        )
    )
    target_smell: ValidSmells = Field(
        description="The target configuration smell this run evaluated."
    )
    smell_detected: bool = Field(
        description="True iff the target smell was predicted present in the workflow."
    )
    detection_justification: str = Field(
        description=(
            "A 1-4 sentence explanation of the detection decision, written in natural "
            "language and grounded in concrete evidence from the workflow. Reference "
            "specific keys, values, job names, step names, trigger names, or line "
            "numbers inline; include no YAML blocks, code snippets, or markdown fences."
        )
    )
    count: int = Field(
        ge=0,
        description=(
            "Total number of instances of the target smell detected in the workflow. "
            "0 iff smell_detected=false. Absence smells (missing concurrency / paths "
            "filter / matrix / comments) may report count >= 1 with an empty "
            "evidence_ranges when there is no natural line anchor."
        ),
    )
    evidence_ranges: list[EvidenceRange] = Field(
        default_factory=list,
        description=(
            "1-based inclusive line ranges grounding the verdict: the localized "
            "smell instances when count > 0, or the enclosing block the agent "
            "inspected when count == 0 (negative evidence). Used by Diff "
            "Out-of-Scope Rate (paper §3 Step 7). May be empty when there is no "
            "natural line anchor. Not bounded by count -- count is the sole "
            "instance authority."
        ),
    )
    detection_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the detection decision (D1 combined agent only): "
            "high = workflow clearly satisfies or clearly does not satisfy the smell "
            "definition; medium = decision depends on contextual interpretation "
            "(e.g. whether a job is a deployment / privileged job); low = workflow "
            "is incomplete, heavily templated, or ambiguous. None in D2 synthesis "
            "paths where confidence is not separately elicited."
        ),
    )
    detection_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the detection confidence level was "
            "assigned, written in natural language and grounded in concrete evidence "
            "from the workflow. Reference specific keys, values, or line numbers inline; "
            "include no YAML blocks, code snippets, or markdown fences. Empty string "
            "when detection_confidence is None (D2 synthesis)."
        ),
    )
    patched_workflow: str = Field(
        description=(
            "Patched YAML; equals the original workflow verbatim when smell_detected=false."
        )
    )
    refactor_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation in natural language, grounded in concrete "
            "evidence from the edit applied, of the refactoring written into "
            "patched_workflow — which keys or lines changed and why the change "
            "removes the smell without altering observable behavior. Empty string "
            "when smell_detected=false (no refactoring was performed). Reference "
            "specific keys, values, or line numbers inline; include no YAML blocks, "
            "code snippets, or markdown fences."
        ),
    )
    refactor_confidence: Confidence | None = Field(
        default=None,
        description=(
            "Calibration of the refactoring (D1 combined agent only): high = edit is "
            "localized, clearly removes all detected instances, and is unlikely to change "
            "behavior; medium = edit likely removes the smell but depends on contextual "
            "assumptions; low = the workflow is incomplete/ambiguous or the fix may alter "
            "behavior. None when smell_detected=false or in D2 synthesis paths where "
            "refactor confidence is not separately elicited."
        ),
    )
    refactor_confidence_justification: str = Field(
        default="",
        description=(
            "A 1-4 sentence explanation of why the refactor confidence level was assigned, "
            "focusing on completeness, safety, and behavior preservation. Empty string when "
            "refactor_confidence is None (smell_detected=false or D2 synthesis). Reference "
            "specific keys, values, or line numbers inline; include no YAML blocks, code "
            "snippets, or markdown fences."
        ),
    )

    @field_validator("patched_workflow", mode="before")
    @classmethod
    def _strip_wrappers(cls, v: object) -> object:
        return _strip_yaml_wrappers(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check_invariants(self) -> CombinedResult:
        _validate_multi_instance(
            smell_detected=self.smell_detected,
            count=self.count,
        )
        return self


# ─── Output-contract boot check ───────────────────────────────────────────────
# Fail fast at import if any output schema drifts from the OutputContract that
# renders its <output_format> block (field order or reasoning-leads invariant).
# The schema classes are injected so output_contract.py stays schema-free and
# below this module in the import graph.
assert_output_contracts_consistent(
    {
        "combined": CombinedResult,
        "detection": DetectionContract,
        "detection_validation": DetectionValidationResult,
        "refactor": RefactorResult,
        "refactor_validation": RefactorValidationResult,
    }
)


# ─── Agent Factories ──────────────────────────────────────────────────────────


_MODEL_FACTORY = ModelFactory()


def _agent_name(base: str, rendered_prompt: str, target_smell: ValidSmells) -> str:
    """Compose a debug-friendly agent name: '<base> [SmellPascal] [hash8]'."""
    return f"{base} [{target_smell.name}] [{prompt_hash(rendered_prompt)[:8]}]"


# Every factory below passes resolve_in_context=False. Agno's default renders the
# instructions and the user turn through string.Template.safe_substitute, which
# rewrites a workflow's `$$` (e.g. `TMPFILE=/tmp/tmp.$$`) to `$` before the model
# sees it. The pipeline passes no template variables, so rendering can only alter
# the YAML; with it off the model receives the user turn byte for byte.


def make_detection_agent(target_smell: ValidSmells, scenario: Scenario) -> Agent:
    """Create the detection agent for ``scenario`` (``"D2"`` single-shot).

    The agent emits the single :class:`DetectionContract`. D2 does not echo the
    workflow back — the YAML is threaded through typed run state, not re-emitted
    by the model.
    """
    rendered = build_detection_prompt(target_smell.value, scenario)
    output_schema = DetectionContract
    # No tools: detection emits only the structured DetectionContract. The former
    # enumerate_workflow_lines helper (line-number prefixing) was a deterministic
    # convenience the model didn't need — the full YAML is already in the user turn —
    # and under use_json_mode some open-weight models emitted the *tool call* as JSON
    # text instead of the contract, which Agno left as a raw str and the #58 guard
    # then rejected. Dropping it removes that failure surface (not referenced in any
    # prompt, so PROMPT_HASHES is unaffected).
    # use_json_mode is run-scoped (is_json_mode_active(), all roles below),
    # defaulting True: Agno requests a loose JSON object + the schema as prompt
    # text. That levels open-weight models served over OpenAI-compatible endpoints
    # (some open-weight models fragment under native json_schema). A per-model
    # `sampling.json_mode: false` flips one model to Agno's native strict
    # json_schema response_format — required for reasoning models (e.g. nemotron)
    # that otherwise stream chain-of-thought into content and never emit the JSON
    # (the #58 guard then rejects the str). Read at construction, mirroring how
    # model_factory.build reads is_thinking_active(). Not in PROMPT_HASHES.
    return Agent(
        name=_agent_name(f"Detection Agent ({scenario})", rendered, target_smell),
        model=_MODEL_FACTORY.build("detection"),
        instructions=rendered,
        output_schema=output_schema,
        use_json_mode=is_json_mode_active(),
        debug_mode=is_verbose(),
        resolve_in_context=False,
    )


def make_refactor_agent(target_smell: ValidSmells, scenario: Scenario) -> Agent:
    """Create the refactor agent for ``scenario`` (``"D2"`` single-shot)."""
    rendered = build_refactor_prompt(target_smell.value, scenario)
    return Agent(
        name=_agent_name(f"Refactor Agent ({scenario})", rendered, target_smell),
        model=_MODEL_FACTORY.build("refactor"),
        instructions=rendered,
        tools=[validate_yaml_syntax, validate_github_actions_syntax],
        output_schema=RefactorResult,
        use_json_mode=is_json_mode_active(),
        debug_mode=is_verbose(),
        resolve_in_context=False,
    )


def make_combined_agent(target_smell: ValidSmells) -> Agent:
    rendered = build_combined_prompt(target_smell.value)
    return Agent(
        name=_agent_name("Detection & Refactor Agent", rendered, target_smell),
        model=_MODEL_FACTORY.build("combined"),
        instructions=rendered,
        tools=[validate_yaml_syntax, validate_github_actions_syntax],
        output_schema=CombinedResult,
        use_json_mode=is_json_mode_active(),
        debug_mode=is_verbose(),
        resolve_in_context=False,
    )
