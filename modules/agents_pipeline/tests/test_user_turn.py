"""Unit tests for UserTurnBuilder + co-located workflow input validation (#46).

Pins the typed-slot → ``<inputs>`` XML for every role (byte-equivalent to the
retired ``build_*_user_prompt`` helpers), the single-source tag set the denylist
derives from, and the relocated GHA workflow validation.
"""

from __future__ import annotations

import re

import pytest
from gha_agents import prompts
from gha_agents.errors import WorkflowValidationError
from gha_agents.user_turn import (
    USER_TURN_BUILDERS,
    Slot,
    UserTurnBuilder,
    all_user_turn_tags,
    validate_workflow_input,
)
from gha_agents.utils import xml_wrap


def _joined(*blocks: str) -> str:
    return "\n\n".join(b for b in blocks if b.strip())


# ─── per-role slot → XML ──────────────────────────────────────────────────────


def test_workflow_builder_wraps_single_slot():
    out = USER_TURN_BUILDERS["workflow"].build(workflow="name: CI\n")
    assert out == xml_wrap("workflow", "name: CI\n")


def test_detection_validation_builder():
    out = USER_TURN_BUILDERS["detection_validation"].build(
        workflow="WF", peer_detection_result="{...}"
    )
    assert out == _joined(xml_wrap("workflow", "WF"), xml_wrap("peer_detection_result", "{...}"))


def test_c2_refactor_builder():
    out = USER_TURN_BUILDERS["d2_refactor"].build(original_workflow="WF", detection_result="DR")
    assert out == _joined(xml_wrap("original_workflow", "WF"), xml_wrap("detection_result", "DR"))


def test_refactor_validation_builder():
    out = USER_TURN_BUILDERS["refactor_validation"].build(
        original_workflow="WF", detection_result="DR", proposed_refactor="PR"
    )
    assert out == _joined(
        xml_wrap("original_workflow", "WF"),
        xml_wrap("detection_result", "DR"),
        xml_wrap("proposed_refactor", "PR"),
    )


# ─── slot validation ──────────────────────────────────────────────────────────


def test_required_slot_missing_raises():
    with pytest.raises(ValueError, match="required"):
        USER_TURN_BUILDERS["detection_validation"].build(workflow="WF")


def test_unknown_slot_rejected():
    with pytest.raises(ValueError, match="unknown"):
        USER_TURN_BUILDERS["workflow"].build(workflow="WF", bogus="x")


def test_custom_builder_tags_order_preserved():
    builder = UserTurnBuilder(Slot("a"), Slot("b", required=False, default="·"))
    assert builder.tags == ("a", "b")
    assert builder.build(a="1") == _joined(xml_wrap("a", "1"), xml_wrap("b", "·"))


# ─── single-source tag set ────────────────────────────────────────────────────


def test_all_user_turn_tags_is_union_of_every_role():
    assert all_user_turn_tags() == frozenset(
        {
            "workflow",
            "repository_context",
            "peer_detection_result",
            "original_workflow",
            "detection_result",
            "proposed_refactor",
        }
    )


# ─── repository_context: omit-if-absent slot + formatter ──────────────────────


def test_repository_context_omitted_when_absent():
    """Every workflow-bearing role drops the block entirely (no "None") when no
    repository_context is supplied — an unenriched ground_truth degrades gracefully."""
    assert USER_TURN_BUILDERS["workflow"].build(workflow="WF") == xml_wrap("workflow", "WF")
    assert "repository_context" not in USER_TURN_BUILDERS["d2_refactor"].build(
        original_workflow="WF", detection_result="DR"
    )


def test_repository_context_rendered_right_after_workflow():
    """When supplied, the block sits immediately after the workflow it describes."""
    out = USER_TURN_BUILDERS["workflow"].build(workflow="WF", repository_context="- x: y")
    assert out == _joined(xml_wrap("workflow", "WF"), xml_wrap("repository_context", "- x: y"))
    refac = USER_TURN_BUILDERS["d2_refactor"].build(
        original_workflow="WF", repository_context="- x: y", detection_result="DR"
    )
    assert refac == _joined(
        xml_wrap("original_workflow", "WF"),
        xml_wrap("repository_context", "- x: y"),
        xml_wrap("detection_result", "DR"),
    )


def test_format_repository_context_lists_present_fields_in_order():
    from gha_agents.user_turn import format_repository_context

    assert format_repository_context(
        current_repository="o/r",
        repository_owner="o",
        default_branch="main",
        workflow_file_path=".github/workflows/ci.yml",
    ) == (
        "- current_repository: o/r\n"
        "- repository_owner: o\n"
        "- default_branch: main\n"
        "- workflow_file_path: .github/workflows/ci.yml"
    )
    # Missing fields are dropped, present ones keep their fixed order.
    assert (
        format_repository_context(current_repository="o/r", default_branch="dev")
        == "- current_repository: o/r\n- default_branch: dev"
    )
    # Nothing present → None, so the slot is omitted rather than half-empty.
    assert format_repository_context() is None


# ─── relocated workflow validation ────────────────────────────────────────────


def test_validate_accepts_a_plausible_workflow():
    validate_workflow_input("f.yml", "name: CI\non: [push]\njobs:\n  b:\n    runs-on: x\n")


def test_validate_rejects_oversize():
    with pytest.raises(WorkflowValidationError):
        validate_workflow_input("f.yml", "on: [push]\n" + "#" + "a" * (2 * 1024 * 1024))


def test_validate_rejects_malformed_yaml():
    with pytest.raises(WorkflowValidationError):
        validate_workflow_input("f.yml", "key: [unclosed\n")


def test_validate_rejects_non_mapping():
    with pytest.raises(WorkflowValidationError):
        validate_workflow_input("f.yml", "- just\n- a\n- list\n")


def test_validate_rejects_missing_github_actions_keys():
    with pytest.raises(WorkflowValidationError):
        validate_workflow_input("f.yml", "random: data\nfoo: bar\n")


# ─── the system-prompt <inputs> doc agrees with the builder (single source) ───

_INPUTS_BLOCK_BY_ROLE = {
    "workflow": (prompts.D1_INPUTS_BLOCK, prompts.D2_DETECTION_INPUTS_BLOCK),
    "detection_validation": (prompts.DETECTION_VALIDATION_INPUTS_BLOCK,),
    "d2_refactor": (prompts.D2_REFACTOR_INPUTS_BLOCK,),
    "refactor_validation": (prompts.REFACTOR_VALIDATION_INPUTS_BLOCK,),
}


def _documented_tags(block: str) -> set[str]:
    # tags documented as `<tag>`; the wrapping <inputs></inputs> is not a slot.
    return set(re.findall(r"<([a-z_]+)>", block)) - {"inputs"}


@pytest.mark.parametrize("role, blocks", list(_INPUTS_BLOCK_BY_ROLE.items()))
def test_inputs_block_documents_exactly_the_builder_slots(role: str, blocks: tuple[str, ...]):
    expected = set(USER_TURN_BUILDERS[role].tags)
    for block in blocks:
        assert _documented_tags(block) == expected


def test_denylist_covers_every_user_turn_tag():
    # The system-template guard denies every runtime tag, derived from the builder.
    assert all_user_turn_tags() <= prompts.DYNAMIC_USER_PLACEHOLDERS
