"""Tests for gha_agents.prompts (smell catalog + system prompt templates)."""

from __future__ import annotations

import hashlib
import re

import pytest
from gha_agents.prompts import (
    D1_COMBINED_TEMPLATE,
    D2_DETECTION_TEMPLATE,
    D2_REFACTOR_TEMPLATE,
    PROMPT_HASHES,
    SMELL_CATALOG,
    build_combined_prompt,
    build_detection_prompt,
    build_refactor_prompt,
    get_all_prompt_hashes,
    prompt_hash,
)

_SAMPLE_SMELL = SMELL_CATALOG[0]["name"]

_ALL_TEMPLATES = [
    D1_COMBINED_TEMPLATE,
    D2_DETECTION_TEMPLATE,
    D2_REFACTOR_TEMPLATE,
]

_ALL_BUILDERS = [
    build_combined_prompt,
    lambda s: build_detection_prompt(s, "D2"),
    lambda s: build_refactor_prompt(s, "D2"),
]


def test_all_templates_are_non_empty_strings_with_placeholders():
    for template in _ALL_TEMPLATES:
        assert isinstance(template, str)
        assert template.strip()
        # Modular templates interpolate the smell via a composed block, not raw
        # {smell_name}/{smell_description} slots (those live inside TARGET_SMELL_BLOCK).
        assert "{target_smell_block}" in template
        assert "{reasoning_block}" in template


def test_all_builders_interpolate_smell_name_and_description():
    entry = SMELL_CATALOG[0]
    for builder in _ALL_BUILDERS:
        rendered = builder(entry["name"])
        assert isinstance(rendered, str)
        assert entry["name"] in rendered
        assert entry["description"] in rendered
        # Placeholders must be fully resolved.
        assert "{smell_name}" not in rendered
        assert "{smell_description}" not in rendered


def test_all_builders_include_melo_cot_trigger():
    cot = "Let's think step by step before emitting the structured output."
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        assert cot in rendered


def test_combined_persona_is_fair_generalist():
    """D1's combined agent is a generalist engineer — no security-only framing."""
    prompt = build_combined_prompt(_SAMPLE_SMELL)
    first_line = prompt.split("\n", 1)[0].lower()
    assert "github actions" in first_line
    assert "security" not in first_line


@pytest.mark.parametrize("scenario", ["D2"])
def test_detection_persona_is_engineer(scenario):
    """Detection (in D2) is a GitHub Actions engineer — auditing is its
    task, not its job title — and carries no security-only framing."""
    prompt = build_detection_prompt(_SAMPLE_SMELL, scenario)
    first_line = prompt.split("\n", 1)[0].lower()
    assert "engineer" in first_line
    assert "auditor" not in first_line
    assert "security" not in first_line


def test_c2_detection_template_states_single_shot_explicitly():
    """D2 detection runs once and has no peer reviewer — the prompt should say so."""
    prompt = build_detection_prompt(_SAMPLE_SMELL, "D2").lower()
    assert "single shot" in prompt or "single-shot" in prompt
    assert "no peer reviewer" in prompt or "no peer review" in prompt
    # D2 must NOT contain feedback-loop guidance.
    assert "peer reviewer's feedback" not in prompt
    assert "next iteration" not in prompt


def test_c2_refactor_template_states_single_shot_explicitly():
    """D2 refactor runs once and has no peer reviewer — the prompt should say so."""
    prompt = build_refactor_prompt(_SAMPLE_SMELL, "D2").lower()
    # The D2 refactor task states its single-execution nature as the absence of
    # the validator/feedback machinery (vs D2 detection's literal "single shot").
    assert "no validation agent" in prompt and "no feedback loop" in prompt
    assert "no peer reviewer" in prompt or "no peer review" in prompt
    # D2 must NOT contain feedback-loop guidance.
    assert "previous refactor attempt" not in prompt
    assert "next iteration" not in prompt


def test_justification_rule_appears_in_templates_with_justification():
    """Every rendered prompt that emits a justification carries the 1-4 sentence /
    natural-language / grounded-in-concrete-X / no-YAML rule. The rule text lives in
    interpolated rule fragments, so this asserts on the rendered prompt rather than the
    raw template. Detection-side justifications are grounded in "concrete evidence";
    refactor-side justifications in the "concrete edit" made — both satisfy the rule."""
    for builder in _ALL_BUILDERS:
        text = builder(_SAMPLE_SMELL).lower()
        assert "1-4 sentence" in text
        assert "natural language" in text
        assert "no yaml" in text
        assert "concrete evidence" in text or "concrete edit" in text


@pytest.mark.parametrize("scenario", ["D2"])
def test_refactor_template_requests_a_justification(scenario):
    """D2 refactor agents must emit a justification alongside the patched workflow."""
    rendered = build_refactor_prompt(_SAMPLE_SMELL, scenario)
    assert "refactor_justification" in rendered
    assert "patched_workflow, refactor_justification" in rendered


def test_combined_template_requests_refactor_justification():
    """D1's combined agent must emit a separate refactor_justification field."""
    rendered = build_combined_prompt(_SAMPLE_SMELL)
    assert "refactor_justification" in rendered
    assert "patched_workflow, refactor_justification" in rendered


@pytest.mark.skip(reason="Gated: persona threat-to-validity framing is a user methodology call.")
def test_prompts_docstring_notes_persona_threat_to_validity():
    """Persona differentiation is part of the decomposition treatment — note it as a TtV."""
    import gha_agents.prompts as prompts_mod

    doc = (prompts_mod.__doc__ or "").lower()
    assert "persona" in doc
    assert "threat-to-validity" in doc or "threat to validity" in doc


def test_field_descriptions_state_justification_rules():
    """agents.py Field descriptions for justification/disagreement_feedback carry the new rule."""
    from gha_agents.agents import (
        CombinedResult,
        DetectionContract,
        DetectionValidationResult,
        RefactorResult,
        RefactorValidationResult,
    )

    phrases = ("1-4 sentence", "natural language", "concrete evidence", "no yaml")
    fields = [
        ("DetectionContract.detection_justification", DetectionContract),
        ("CombinedResult.detection_justification", CombinedResult),
        ("CombinedResult.refactor_justification", CombinedResult),
        ("RefactorResult.refactor_justification", RefactorResult),
        ("RefactorValidationResult.disagreement_feedback", RefactorValidationResult),
        ("DetectionValidationResult.disagreement_feedback", DetectionValidationResult),
    ]
    fields = [
        (label, model.model_fields[label.split(".", 1)[1]].description)
        for label, model in fields
    ]
    for label, desc in fields:
        text = (desc or "").lower()
        for phrase in phrases:
            assert phrase in text, f"missing '{phrase}' in {label}"


def test_builders_reject_unknown_smell():
    for builder in _ALL_BUILDERS:
        with pytest.raises(ValueError, match="Unknown target smell"):
            builder("not-a-real-smell")


def test_prompt_hash_is_sha256_hex():
    rendered = build_combined_prompt(_SAMPLE_SMELL)
    expected = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    assert prompt_hash(rendered) == expected
    assert len(prompt_hash(rendered)) == 64


def test_different_smells_produce_different_hashes():
    a = prompt_hash(build_detection_prompt(SMELL_CATALOG[0]["name"], "D2"))
    b = prompt_hash(build_detection_prompt(SMELL_CATALOG[1]["name"], "D2"))
    assert a != b


def test_build_detection_prompt_rejects_invalid_scenario():
    with pytest.raises(ValueError, match="only defined for D2"):
        build_detection_prompt(_SAMPLE_SMELL, "D1")  # type: ignore[arg-type]


def test_build_refactor_prompt_rejects_invalid_scenario():
    with pytest.raises(ValueError, match="only defined for D2"):
        build_refactor_prompt(_SAMPLE_SMELL, "D1")  # type: ignore[arg-type]


def test_get_all_prompt_hashes_c1_has_one_role_with_all_smells():
    hashes = get_all_prompt_hashes("D1")
    assert set(hashes.keys()) == {"combined"}
    assert set(hashes["combined"].keys()) == {s["pascal"] for s in SMELL_CATALOG}
    assert all(len(h) == 64 for h in hashes["combined"].values())


def test_get_all_prompt_hashes_c2_has_two_roles():
    hashes = get_all_prompt_hashes("D2")
    assert set(hashes.keys()) == {"detection", "refactor"}
    for role_hashes in hashes.values():
        assert set(role_hashes.keys()) == {s["pascal"] for s in SMELL_CATALOG}


def test_get_all_prompt_hashes_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_all_prompt_hashes("C99")  # type: ignore[arg-type]


def test_prompt_hashes_table_matches_helper():
    for scenario, roles in PROMPT_HASHES.items():
        assert roles == get_all_prompt_hashes(scenario)


# ─── XML-tagged prompts (paper §3) ────────────────────────────────────────────


_XML_TAGS = (
    "<role>", "<task>", "<target_smell", "<decision_rules>", "<reasoning>", "<output_format>",
)


def test_all_templates_use_xml_tags():
    """Paper claims 'XML-tagged prompts'. Every system prompt must structure
    its sections with XML tags rather than plain prose headers."""
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        for tag in _XML_TAGS:
            assert tag in rendered, f"missing {tag!r} in template from {builder.__name__}"


def test_target_smell_tag_carries_name_attribute():
    """The <target_smell> opening tag exposes the smell name as an attribute
    so a structured reader (or the model itself) can extract it without prose
    parsing."""
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        assert f'<target_smell name="{_SAMPLE_SMELL}">' in rendered


def test_cot_trigger_lives_inside_reasoning_tag():
    """The Chain-of-Thought trigger sits inside <reasoning>...</reasoning>, not
    as free-floating prose, so the structure mirrors the paper's claim that the
    prompt design 'combines role-based prompting with Chain-of-Thought reasoning'."""
    cot = "Let's think step by step before emitting the structured output."
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        reasoning_block = rendered.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0]
        assert cot in reasoning_block


def test_emit_field_list_lives_inside_output_format_tag():
    """The 'Emit X exactly: ...' field list sits inside <output_format>, so the
    schema contract is structurally separable from the rest of the prompt."""
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        of_block = rendered.split("<output_format>", 1)[1].split("</output_format>", 1)[0]
        assert "Emit " in of_block
        assert "No extra fields" in of_block


def test_decision_rules_tag_wraps_the_rules_block():
    """All bulleted decision rules sit inside <decision_rules>...</decision_rules>;
    no rule bullets float outside the tag."""
    for builder in _ALL_BUILDERS:
        rendered = builder(_SAMPLE_SMELL)
        dr_block = rendered.split("<decision_rules>", 1)[1].split("</decision_rules>", 1)[0]
        # At least one bulleted rule lives inside the block.
        assert dr_block.lstrip().startswith("-") or "\n-" in dr_block


# ─── User-turn data contract (system <inputs> ↔ UserTurnBuilder) ───────────────
# Each role's system prompt declares, in its <inputs> block, exactly the XML tags
# that role's UserTurnBuilder emits into the user turn — and the dump in
# _example_user_turns() renders exactly those tags. This guarantees the user
# prompt carries precisely the data the system prompt says it expects, with no
# silent drift between the two halves of the message pair.

from gha_agents.prompts import _example_user_turns  # noqa: E402
from gha_agents.user_turn import (  # noqa: E402
    D2_REFACTOR_BUILDER,
    WORKFLOW_BUILDER,
)

# (scenario, role, rendered-system-prompt, the UserTurnBuilder for that role).
# Mirrors the call sites in teams.py: D1Workflow/D2Workflow use WORKFLOW_BUILDER;
# D2 refactor uses D2_REFACTOR_BUILDER.
_ROLE_CONTRACT = [
    ("D1", "combined", build_combined_prompt(_SAMPLE_SMELL), WORKFLOW_BUILDER),
    ("D2", "detection", build_detection_prompt(_SAMPLE_SMELL, "D2"), WORKFLOW_BUILDER),
    ("D2", "refactor", build_refactor_prompt(_SAMPLE_SMELL, "D2"), D2_REFACTOR_BUILDER),
]


def _inputs_block_tags(system_prompt: str) -> list[str]:
    """The `<tag>` names documented inside the system prompt's <inputs> block."""
    block = system_prompt.split("<inputs>", 1)[1].split("</inputs>", 1)[0]
    return re.findall(r"`<(\w+)>`", block)


@pytest.mark.parametrize(
    "scenario,role,system_prompt,builder",
    _ROLE_CONTRACT,
    ids=[f"{s}-{r}" for s, r, _, _ in _ROLE_CONTRACT],
)
def test_inputs_block_declares_exactly_the_builder_tags(scenario, role, system_prompt, builder):
    """The <inputs> block documents exactly the tags the role's builder emits."""
    assert _inputs_block_tags(system_prompt) == list(builder.tags)


_DOC_ENTRY = {
    "file_key": "o@r@ci.yml",
    "yaml_content": "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n",  # noqa: E501
    "repository": "o/r",
    "file_path": ".github/workflows/ci.yml",
    "default_branch": "main",
    "present_smells": [{"pascal": SMELL_CATALOG[0]["pascal"], "count": 1}],
    "absent_smells": [],
}


def test_dumped_user_turn_carries_exactly_the_declared_tags():
    """The dump's example user turn renders exactly the tags each role declares —
    no missing data the system prompt expects, no undocumented tag smuggled in."""
    user_turns = _example_user_turns(_DOC_ENTRY, _SAMPLE_SMELL)
    for scenario, role, system_prompt, builder in _ROLE_CONTRACT:
        turn = user_turns[(scenario, role)]
        top_level_tags = re.findall(r"^<(\w+)>$", turn, flags=re.MULTILINE)
        assert top_level_tags == list(builder.tags), f"{scenario}:{role}"
        assert _inputs_block_tags(system_prompt) == top_level_tags, f"{scenario}:{role}"


def test_dumped_user_turn_uses_real_entry_workflow_and_repo_context():
    """The grounded dump injects the entry's own workflow + repository_context —
    not a hardcoded mock — into the user turn."""
    turn = _example_user_turns(_DOC_ENTRY, _SAMPLE_SMELL)[("D1", "combined")]
    assert _DOC_ENTRY["yaml_content"] in turn
    assert "- current_repository: o/r" in turn
    assert "- default_branch: main" in turn


def test_dump_repository_context_is_byte_identical_to_the_runtime_block():
    """The <repository_context> the doc shows is exactly what the pipeline sends at
    runtime: the dump and the per-entry pipeline both wrap
    repository_context_from_entry(entry) in the same builder slot. Asserting the
    rendered block — not just the values — means the doc can never drift from the
    bytes the model actually receives for that entry."""
    from gha_agents.user_turn import repository_context_from_entry
    from gha_agents.utils import xml_wrap

    runtime_block = xml_wrap("repository_context", repository_context_from_entry(_DOC_ENTRY))
    for (scenario, role), turn in _example_user_turns(_DOC_ENTRY, _SAMPLE_SMELL).items():
        assert runtime_block in turn, f"{scenario}:{role}"
